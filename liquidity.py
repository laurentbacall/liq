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
    # We will collect all series in a dictionary to join them all at once at the end
    data_series = {}

    # 1. S&P 500 (Master Daily Index)
    try:
        df_sp = yf.download("^GSPC", start=start_date, interval="1d", progress=False)
        if not df_sp.empty:
            if isinstance(df_sp.columns, pd.MultiIndex): 
                df_sp.columns = df_sp.columns.get_level_values(0)
            data_series['SP500'] = df_sp['Close']
    except Exception as e:
        st.error(f"Yahoo Finance Error: {e}")

    # 2. FRED Macro Series
    # We fetch these independently so they don't clip each other
    fred_ids = {
        'VIXCLS': 'VIX',
        'BAMLH0A0HYM2': 'HY_Spread',
        'DTWEXBGS': 'USD_Index',
        'WALCL': 'Fed_Assets',
        'M2SL': 'M2',
        'CPIAUCSL': 'CPI',
        'T10Y2Y': 'Yield_Curve_2s10s',
        'REAINTRATREARAT10Y': 'Real_10Y_Yield'
    }

    for s_id, name in fred_ids.items():
        try:
            s = fred.get_series(s_id, observation_start=start_date)
            s.index = pd.to_datetime(s.index).tz_localize(None) # Match Yahoo timezone
            data_series[name] = s
        except:
            st.warning(f"Failed to fetch {name} from FRED")

    # 3. FINRA Margin Debt (Monthly Excel)
    try:
        finra_url = "https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx"
        df_finra = pd.read_excel(finra_url, engine='openpyxl')
        # Standardize the FINRA cleanup (Skip headers, pick first 2 cols)
        df_finra = df_finra.iloc[1:].copy() 
        df_finra.columns = ['Date', 'Margin_Debt']
        df_finra['Date'] = pd.to_datetime(df_finra['Date'])
        df_finra.set_index('Date', inplace=True)
        # Ensure it's a Series and timezone-naive
        s_margin = df_finra['Margin_Debt'].dropna().tz_localize(None)
        data_series['Margin_Debt'] = s_margin
    except:
        st.warning("Failed to fetch FINRA Margin Debt")

    # 4. THE FIX: CONCATENATE ALL (Union of all indices)
    # This prevents CPI from being clipped by FINRA or VIX by SP500
    df = pd.concat(data_series, axis=1)
    df = df.sort_index()
    
    # 5. Fill gaps (Forward fill monthly data to daily)
    df = df.ffill()

    # 6. Calculations (Use the unified dataframe)
    if 'CPI' in df.columns:
        # YoY CPI for growth calc
        df['CPI_YoY'] = df['CPI'].pct_change(365)
    
    if 'M2' in df.columns and 'CPI' in df.columns:
        # Real M2 Growth calculation
        m2_growth = df['M2'].pct_change(365)
        df['M2_Real_Growth'] = m2_growth - df['CPI_YoY'].fillna(0)

    # System Allocation Score (Simplified logic)
    # We define liquidity as (Fed Assets / CPI) + (Margin Debt / CPI)
    if all(col in df.columns for col in ['Fed_Assets', 'Margin_Debt', 'CPI']):
        liq_index = (df['Fed_Assets'] + df['Margin_Debt']) / df['CPI']
        df['Allocation_Pct'] = (liq_index / liq_index.rolling(window=252*2).max()) * 100

    # Finally, filter to dates where we have S&P 500 price
    return df.dropna(subset=['SP500'])


df = get_master_data()

# --- 3. CALCULATIONS ---
if not df.empty:
    df['Net_Liq'] = df.get('Fed_Assets', 0) - (df.get('TGA', 0).fillna(0) + df.get('RRP', 0).fillna(0))
    df['Net_Liq_YoY'] = df['Net_Liq'].pct_change(periods=252) * 100
    df['Net_Liq_SMA'] = df['Net_Liq'].rolling(21).mean()
    df['CPI_YoY'] = df.get('CPI', pd.Series(dtype=float)).pct_change(365) * 100
    df['M2_Real_Growth'] = (df.get('M2', pd.Series(dtype=float)).pct_change(365) * 100) - df['CPI_YoY']
    # REQ: SMA 200
    df['SP500_SMA200'] = df['SP500'].rolling(window=200).mean()
    
    # Calculation: YoY Growth of the FINRA Margin Debt
    # This works now because the index is strictly Datetime
    df['Margin_Velocity'] = df['Margin_Debt'].pct_change(periods=252) * 100
    
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
df.index = pd.to_datetime(df.index)
all_dates = df.index

# Create a clean list of Month-Start dates
timeline = pd.date_range(start=all_dates.min(), end=all_dates.max(), freq='MS').tolist()
if all_dates.max() not in timeline:
    timeline.append(all_dates.max())

start_s, end_s = st.select_slider(
    "Select Period", 
    options=timeline, 
    value=(timeline[-121], timeline[-1]), 
    format_func=lambda x: x.strftime('%Y-%m')
)

# Use truncate for a clean, error-free slice
p_df = df.truncate(before=start_s, after=end_s)

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
    # FORCE ALIGNMENT: Every chart must have the same start/end points
    ax.set_xlim(start_s, end_s)
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