import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, MultipleLocator
import yfinance as yf
from fredapi import Fred

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Macro Regime Monitor", layout="wide")
st.title("🛡️ Institutional Risk & Liquidity Monitor")

# REQ: Font safety
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

if "FRED_API_KEY" in st.secrets:
    api_key = st.secrets["FRED_API_KEY"]
else:
    api_key = st.text_input("Enter FRED API Key", type="password")

if not api_key:
    st.info("Please enter your FRED API key to begin.")
    st.stop()

fred = Fred(api_key=api_key)

# --- 2. DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_master_data():
    start_date = "1950-01-01"
    try:
        df_sp = yf.download("^GSPC", start=start_date, interval="1d", progress=False)
        if not df_sp.empty:
            if isinstance(df_sp.columns, pd.MultiIndex): df_sp.columns = df_sp.columns.get_level_values(0)
            df_sp = df_sp[['Close']].rename(columns={'Close': 'SP500'})
            # REMOVED .date conversion to keep it as a DatetimeIndex
            df_sp.index = pd.to_datetime(df_sp.index) 
    except: df_sp = pd.DataFrame()

    # 1. Initialize df_macro first to avoid UnboundLocalError
    df_macro = pd.DataFrame()

    # 2. Fetch FINRA Monthly Excel Data
    try:
        finra_url = "https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx"
        # Added engine='openpyxl' to solve the dependency error
        df_finra = pd.read_excel(finra_url, usecols=[0, 1], skiprows=0, engine='openpyxl')
        df_finra.columns = ['Date', 'Margin_Debt']
        df_finra['Date'] = pd.to_datetime(df_finra['Date'])
        df_finra.set_index('Date', inplace=True)
        df_macro = df_finra.dropna().sort_index()
    except Exception as e:
        st.warning(f"FINRA Excel Load Failed: {e}. Ensure 'openpyxl' is installed.")

    # 3. Fetch FRED Series
    series_ids = {
        'VIXCLS': 'VIX', 'BAMLH0A0HYM2': 'HY_Spread', 'DTWEXBGS': 'USD_Index',
        'WALCL': 'Fed_Assets', 'M2SL': 'M2', 'CPIAUCSL': 'CPI',
        'WTREGEN': 'TGA', 'RRPONTSYD': 'RRP', 'TB3MS': '3M_Bill',
        'SOFR': 'SOFR', 'TGCRRATE': 'TGCR', 'DFII10': 'Real_10Y_Yield',
        'T10Y2Y': 'Yield_Curve_2s10s', 'USREC': 'Recessions'
    }
    
    for s_id, name in series_ids.items():
        try:
            s = fred.get_series(s_id, observation_start=start_date)
            if s is not None:
                # Keep as DatetimeIndex (removed .date)
                df_macro[name] = pd.to_datetime(s)
        except: pass

    # 4. ALIGNMENT FIX: Force macro data to match S&P 500 dates exactly
    master_index = df_sp.index
    df_combined = pd.concat([df_sp, df_macro.reindex(master_index)], axis=1).sort_index()
    df_combined = df_combined.ffill().dropna(subset=['SP500'])
    
    return df_combined

df = get_master_data()

# --- 3. CALCULATIONS ---
if not df.empty:
    df['Net_Liq'] = df.get('Fed_Assets', 0) - (df.get('TGA', 0).fillna(0) + df.get('RRP', 0).fillna(0))
    df['Net_Liq_YoY'] = df['Net_Liq'].pct_change(365) * 100
    df['Net_Liq_SMA'] = df['Net_Liq'].rolling(21).mean()
    df['CPI_YoY'] = df.get('CPI', pd.Series(dtype=float)).pct_change(365) * 100
    df['M2_Real_Growth'] = (df.get('M2', pd.Series(dtype=float)).pct_change(365) * 100) - df['CPI_YoY']
    # REQ: SMA 200
    df['SP500_SMA200'] = df['SP500'].rolling(200).mean()
    
    # Calculation: YoY Growth of the FINRA Margin Debt
    df['Margin_Velocity'] = df['Margin_Debt'].pct_change(365) * 100
    
    if 'SOFR' in df.columns and 'TGCR' in df.columns:
        df['Funding_Stress'] = (df['SOFR'] - df['TGCR']).interpolate().ffill() * 100
    else: df['Funding_Stress'] = 0 
        
    df['HY_Z'] = (df['HY_Spread'] - df['HY_Spread'].rolling(1095).mean()) / df['HY_Spread'].rolling(1095).std()
    df['VIX_SMA'] = df['VIX'].rolling(20).mean()

    # REQ: 7-Point Scoring
    s1 = (df.get('Net_Liq_YoY', 0) > 0).astype(int) * 20
    s2 = (df.get('Net_Liq', 0) > df.get('Net_Liq_SMA', 0)).astype(int) * 20
    s3 = (df.get('M2_Real_Growth', 0) > 0).astype(int) * 15
    s4 = (df.get('Real_10Y_Yield', 5) < 1.0).astype(int) * 15
    s5 = (df.get('HY_Z', 1) < 0.0).astype(int) * 10
    s6 = (df.get('Funding_Stress', 50) < 15).astype(int) * 10
    s7 = (df.get('VIX', 50) < df.get('VIX_SMA', 0)).astype(int) * 10
    
    df['Total_Score'] = s1 + s2 + s3 + s4 + s5 + s6 + s7
    df['Allocation_Pct'] = df['Total_Score'].apply(lambda s: 100 if s >= 80 else (75 if s >= 60 else (40 if s >= 40 else 0)))

# --- 4. PERIOD SLIDER ---
# Convert index to Timestamps before finding min/max to match pd.date_range
start_date = pd.Timestamp(df.index.min())
end_date = pd.Timestamp(df.index.max())

timeline = sorted(list(set(pd.date_range(start_date, end_date, freq='MS').tolist() + [end_date])))
start_s, end_s = st.select_slider("Select Period", options=timeline, value=(timeline[-121], timeline[-1]), format_func=lambda x: x.strftime('%Y-%m'))
p_df = df.loc[start_s:end_s]

# --- 5. PLOTTING ---
fig, axes = plt.subplots(11, 1, figsize=(14, 75))

def format_ax(ax, title, use_log=False):
    ax.set_title(title, loc='left', fontweight='bold', fontsize=14)
    ax.set_xlim(p_df.index.min(), p_df.index.max()) # <--- LOCK THE SCALE
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.grid(True, which='minor', axis='x', color='gray', linestyle=':', alpha=0.2)
    ax.grid(True, which='major', axis='x', color='gray', linestyle='-', alpha=0.4)
    ax.grid(True, which='major', axis='y', alpha=0.4)
    
    if use_log:
        ax.set_yscale('log')
        # REQ: 1,000 Point Intervals
        ax.yaxis.set_major_locator(MultipleLocator(1000))
    
    if 'Recessions' in p_df.columns:
        ax.fill_between(p_df.index, 0, 1, where=p_df['Recessions']>0, color='gray', alpha=0.2, transform=ax.get_xaxis_transform(), zorder=-1)
    
    ax.tick_params(labelbottom=True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))

def get_s(col): return p_df[col] if col in p_df.columns else pd.Series(np.zeros(len(p_df)), index=p_df.index)

# 1. SP500 (REQ: Log Scale + SMA 200 + 1000pt Grid)
axes[0].plot(p_df.index, p_df['SP500'], color='black', lw=2)
if 'SP500_SMA200' in p_df.columns:
    axes[0].plot(p_df.index, p_df['SP500_SMA200'], color='red', ls='--', lw=1.5, label='200D SMA')
    axes[0].legend(loc='upper left')
format_ax(axes[0], "1. S&P 500 (Log) vs 200D SMA", use_log=True)

# 2-10 
axes[1].plot(p_df.index, get_s('Allocation_Pct'), color='blue', lw=1.5); format_ax(axes[1], "2. System Allocation %")
axes[2].plot(p_df.index, get_s('Net_Liq'), color='darkgreen'); format_ax(axes[2], "3. Net Liquidity Path")
axes[3].plot(p_df.index, get_s('M2_Real_Growth'), color='purple'); format_ax(axes[3], "4. Real M2 Growth")
axes[4].plot(p_df.index, get_s('HY_Spread'), color='orange'); axes[4].invert_yaxis(); format_ax(axes[4], "5. HY Spread (Inverted)")
axes[5].plot(p_df.index, get_s('Real_10Y_Yield'), color='darkblue'); format_ax(axes[5], "6. Real 10Y Yield")
axes[6].plot(p_df.index, get_s('Yield_Curve_2s10s'), color='darkgreen'); format_ax(axes[6], "7. Yield Curve")
axes[7].plot(p_df.index, get_s('USD_Index'), color='navy'); format_ax(axes[7], "8. USD Index")
axes[8].plot(p_df.index, get_s('VIX'), color='red', alpha=0.6); format_ax(axes[8], "9. VIX")
axes[9].plot(p_df.index, get_s('Funding_Stress'), color='blue'); format_ax(axes[9], "10. Funding Stress")

# 11. Daily Leverage Proxy (Equity / M2 Ratio Z-Score)
axes[10].axhline(0, color='black', lw=1)
axes[10].axhline(2, color='red', ls='--', alpha=0.5) # Danger Zone
axes[10].plot(p_df.index, get_s('Margin_Velocity'), color='firebrick', lw=1.5)
axes[10].fill_between(p_df.index, get_s('Margin_Velocity'), 0, color='firebrick', alpha=0.2)
format_ax(axes[10], "11. FINRA debt velocity")

plt.tight_layout(pad=4.0)
st.pyplot(fig)
st.download_button("📥 DOWNLOAD CSV", p_df.to_csv().encode('utf-8'), "macro_monitor.csv", "text/csv")