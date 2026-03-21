import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
import yfinance as yf
from fredapi import Fred

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Macro Regime Monitor", layout="wide")
st.title("🛡️ Institutional Risk & Liquidity Monitor")

# Global Matplotlib Fixes for Streamlit Cloud
plt.rcParams['axes.formatter.use_mathtext'] = False
plt.rcParams['figure.max_open_warning'] = 0

if "FRED_API_KEY" in st.secrets:
    api_key = st.secrets["FRED_API_KEY"]
else:
    api_key = st.text_input("Enter FRED API Key", type="password")

if not api_key:
    st.info("Please enter your FRED API key to begin.")
    st.stop()

fred = Fred(api_key=api_key)

# --- 2. DATA FETCHING (Anti-Crash & Live Sync) ---
@st.cache_data(ttl=3600)
def get_master_data():
    # Requirement: Anti-Crash (Yahoo -> FRED failover)
    df_sp = pd.DataFrame()
    try:
        df_sp = yf.download("^GSPC", start="1950-01-01", interval="1d", progress=False)
        if not df_sp.empty:
            if isinstance(df_sp.columns, pd.MultiIndex):
                df_sp.columns = df_sp.columns.get_level_values(0)
            df_sp = df_sp[['Close']].rename(columns={'Close': 'SP500'})
    except:
        pass

    if df_sp.empty:
        try:
            # Fallback to FRED S&P 500 (Starts 2014)
            s_fred_sp = fred.get_series('SP500')
            df_sp = s_fred_sp.to_frame('SP500')
        except: pass

    series_ids = {
        'WALCL': 'Fed_Assets', 'WTREGEN': 'TGA', 'RRPONTSYD': 'RRP',
        'TB3MS': '3M_Bill', 'CPIAUCSL': 'CPI', 'M2SL': 'M2', 
        'BAMLH0A0HYM2': 'HY_Spread', 'SOFR': 'SOFR', 'TGCRRATE': 'TGCR', 
        'VIXCLS': 'VIX', 'DFII10': 'Real_10Y_Yield',
        'T10Y2Y': 'Yield_Curve_2s10s', 'DTWEXBGS': 'USD_Index', 
        'USREC': 'Recessions'
    }
    
    df_macro = pd.DataFrame()
    for s_id, name in series_ids.items():
        try:
            s = fred.get_series(s_id)
            if s is not None:
                df_macro[name] = s
        except: pass

    # Requirement: Real-time sync (Join on SP500 daily timeline)
    df_master = df_sp.join(df_macro, how='outer')
    df_master.index = pd.to_datetime(df_master.index).tz_localize(None)
    df_master = df_master.sort_index().ffill()
    df_master = df_master.dropna(subset=['SP500'])
    return df_master

df = get_master_data()

# --- 3. CALCULATIONS ---
if not df.empty:
    df['Net_Liq'] = df['Fed_Assets'] - (df.get('TGA', 0).fillna(0) + df.get('RRP', 0).fillna(0))
    df['Net_Liq_YoY'] = df['Net_Liq'].pct_change(365) * 100
    df['Net_Liq_SMA'] = df['Net_Liq'].rolling(21).mean()
    df['CPI_YoY'] = df['CPI'].pct_change(365) * 100
    df['M2_Real_Growth'] = (df['M2'].pct_change(365) * 100) - df['CPI_YoY']
    df['SP500_SMA200'] = df['SP500'].rolling(200).mean()
    
    if 'HY_Spread' in df.columns:
        df['HY_Z'] = (df['HY_Spread'] - df['HY_Spread'].rolling(1095).mean()) / df['HY_Spread'].rolling(1095).std()
    if 'VIX' in df.columns:
        df['VIX_SMA'] = df['VIX'].rolling(20).mean()
    if 'SOFR' in df.columns and 'TGCR' in df.columns:
        df['Funding_Stress'] = (df['SOFR'] - df['TGCR']).interpolate().ffill() * 100

    # Score Engine
    s1 = (df['Net_Liq_YoY'] > 0).astype(int) * 20
    s2 = (df['Net_Liq'] > df['Net_Liq_SMA']).astype(int) * 20
    s3 = (df['M2_Real_Growth'] > 0).astype(int) * 15
    s4 = (df.get('Real_10Y_Yield', 5) < 1.0).astype(int) * 15
    s5 = (df.get('HY_Z', 1) < 0.0).astype(int) * 10
    s6 = (df.get('Funding_Stress', 50) < 15).astype(int) * 10
    s7 = (df.get('VIX', 50) < df.get('VIX_SMA', 0)).astype(int) * 10
    
    df['Total_Score'] = s1 + s2 + s3 + s4 + s5 + s6 + s7
    df['Allocation_Pct'] = df['Total_Score'].apply(lambda s: 100 if s >= 80 else (75 if s >= 60 else (40 if s >= 40 else 0)))

# --- 4. PERIOD SLIDER ---
monthly_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')
start, end = st.select_slider("Select Period", options=monthly_range, value=(monthly_range[-120], monthly_range[-1]), format_func=lambda x: x.strftime('%Y'))
p_df = df.loc[start:end]

# --- 5. PLOTTING ---
fig, axes = plt.subplots(11, 1, figsize=(14, 68))

def format_ax(ax, title):
    # Requirement: Year labels on every chart
    ax.set_title(title, loc='left', fontweight='bold', fontsize=13)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Requirement: Quarterly lighter vertical grid + Yearly vertical
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.grid(True, which='minor', axis='x', color='gray', linestyle=':', alpha=0.2)
    ax.grid(True, which='major', axis='x', color='gray', linestyle='-', alpha=0.4)
    
    # Requirement: Horizontal grids
    ax.grid(True, which='major', axis='y', alpha=0.4)
    
    # Requirement: Gray Recession Shading
    if 'Recessions' in p_df.columns:
        ax.fill_between(p_df.index, 0, 1, where=p_df['Recessions']>0, 
                        color='gray', alpha=0.15, transform=ax.get_xaxis_transform(), zorder=-1)
    
    ax.tick_params(labelbottom=True, labelsize=10)
    
    # Fix for ValueError (Scientific Notation)
    y_formatter = ScalarFormatter(useOffset=False)
    y_formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(y_formatter)

def get_s(col):
    return p_df[col] if col in p_df.columns else pd.Series(np.zeros(len(p_df)), index=p_df.index)

# 1. SP500
axes[0].plot(p_df.index, p_df['SP500'], color='black', lw=2)
if 'SP500_SMA200' in p_df.columns:
    axes[0].plot(p_df.index, p_df['SP500_SMA200'], color='red', linestyle='--', lw=1.2, label='200D SMA')
    axes[0].legend(loc='upper left')
axes[0].set_yscale('log')
format_ax(axes[0], "1. S&P 500 (Log) vs 200-Day SMA")

# 2. Allocation
axes[1].fill_between(p_df.index, get_s('Allocation_Pct'), color='blue', alpha=0.1)
axes[1].plot(p_df.index, get_s('Allocation_Pct'), color='blue', lw=1.5)
axes[1].set_ylim(-5, 105)
format_ax(axes[1], "2. System Allocation %")

# 3. Liquidity
nl, ns = get_s('Fed_Assets'), get_s('Net_Liq_SMA') # Visualizing the raw liquidity path
axes[2].plot(p_df.index, nl, color='black', alpha=0.3)
axes[2].plot(p_df.index, ns, color='red', ls='--', lw=1)
format_ax(axes[2], "3. Net Liquidity Path")

# 4. M2
axes[3].axhline(0, color='red', linestyle=':')
axes[3].plot(p_df.index, get_s('M2_Real_Growth'), color='purple')
format_ax(axes[3], "4. Real M2 Growth (Must be > 0)")

# 5. HY Spread
if 'HY_Z' in p_df.columns:
    axes[4].axhline(0, color='red', linestyle=':')
    axes[4].plot(p_df.index, p_df['HY_Z'], color='orange')
    axes[4].invert_yaxis()
format_ax(axes[4], "5. HY Spread Z-Score (Inverted)")

# 6. Real 10Y
axes[5].axhline(1.0, color='red', linestyle=':')
axes[5].plot(p_df.index, get_s('Real_10Y_Yield'), color='darkblue')
format_ax(axes[5], "6. Real 10Y Yield (Target < 1.0%)")

# 7. Curve
axes[6].axhline(0, color='black')
axes[6].plot(p_df.index, get_s('Yield_Curve_2s10s'), color='darkgreen')
format_ax(axes[6], "7. Yield Curve (10Y-2Y)")

# 8. USD
axes[7].plot(p_df.index, get_s('USD_Index'), color='navy')
format_ax(axes[7], "8. USD Index")

# 9. VIX
axes[8].plot(p_df.index, get_s('VIX'), color='red', lw=1, alpha=0.7)
axes[8].plot(p_df.index, get_s('VIX_SMA'), color='black', ls='--', lw=1)
format_ax(axes[8], "9. VIX vs 20D SMA")

# 10. Funding
axes[9].axhline(15, color='red', linestyle=':')
axes[9].plot(p_df.index, get_s('Funding_Stress'), color='blue')
format_ax(axes[9], "10. Funding Stress (SOFR-TGCR)")

# 11. Leverage (Hidden if missing)
axes[10].plot(p_df.index, get_s('USREC'), color='brown', alpha=0) # Placeholder
format_ax(axes[10], "11. Economic Cycle (Recession Overlay Active)")

plt.tight_layout(pad=4.0)
st.pyplot(fig)

# --- 6. DOWNLOAD CSV BUTTON ---
# Requirement: Download CSV
st.markdown("---")
csv_data = p_df.to_csv().encode('utf-8')
st.download_button(label="📥 Download Dataset (CSV)", data=csv_data, file_name='macro_monitor.csv', mime='text/csv')