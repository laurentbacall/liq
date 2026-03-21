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
    series_dict = {}

    # 1. Yahoo Finance S&P 500
    try:
        sp = yf.download("^GSPC", start=start_date, interval="1d", progress=False)
        if not sp.empty:
            if isinstance(sp.columns, pd.MultiIndex): sp.columns = sp.columns.get_level_values(0)
            series_dict['SP500'] = sp['Close']
    except: pass

    # 2. FRED Series (Fetch everything individually)
    fred_ids = {
        'VIXCLS': 'VIX', 'BAMLH0A0HYM2': 'HY_Spread', 'CPIAUCSL': 'CPI',
        'WALCL': 'Fed_Assets', 'M2SL': 'M2', 'WTREGEN': 'TGA', 
        'RRPONTSYD': 'RRP', 'DTWEXBGS': 'USD_Index', 'T10Y2Y': 'Yield_Curve_2s10s',
        'DFII10': 'Real_10Y_Yield','SOFR': 'SOFR','TGCRRATE': 'TGCR'
    }
    for fid, name in fred_ids.items():
        try:
            s = fred.get_series(fid, observation_start=start_date)
            s.index = pd.to_datetime(s.index).tz_localize(None)
            series_dict[name] = s
        except: pass
    # --- ADD THIS INSIDE get_master_data() ---
    try:
        # 3. Robust FINRA Scraper
        finra_url = "https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx"
        # Read the raw file first to find the anchor
        raw_f = pd.read_excel(finra_url)
        
        # FIX: Find the first row that actually contains a date (e.g., '2024-01' or 'Feb-20')
        # This prevents the "list index out of range" error
        mask = raw_f.iloc[:, 0].astype(str).str.contains('20', na=False)
        
        if mask.any():
            start_row = raw_f[mask].index[0]
            # Reload with the correct header offset
            df_f = pd.read_excel(finra_url, skiprows=start_row)
            
            # FINRA files often have 7+ columns; we only want the first two (Date, Debt)
            df_f = df_f.iloc[:, :2] 
            df_f.columns = ['Date', 'Margin_Debt']
            
            # Convert to datetime and clean
            df_f['Date'] = pd.to_datetime(df_f['Date'], errors='coerce')
            df_f = df_f.dropna(subset=['Date', 'Margin_Debt'])
            
            # Ensure Margin_Debt is numeric (removes commas/strings)
            df_f['Margin_Debt'] = pd.to_numeric(df_f['Margin_Debt'], errors='coerce')
            
            series_dict['Margin_Debt'] = df_f.set_index('Date')['Margin_Debt']
        else:
            st.sidebar.warning("Could not find data rows in FINRA file.")
    except Exception as e:
        st.sidebar.error(f"FINRA Scraper Error: {e}")

    # 3. Join everything on the UNION of all dates
    # This prevents CPI from being cut off if FINRA or S&P 500 is missing data
    df = pd.concat(series_dict, axis=1).sort_index()

    # 4. Apply forward fill to expand monthly data to daily
    # Daily series like VIX and HY will NOT be affected because they already have daily data
    df = df.ffill()

    # 5. Trim to only show dates where the S&P 500 actually exists
    if 'SP500' in df.columns:
        df = df.dropna(subset=['SP500'])
    
    return df


df = get_master_data()

# --- 3. CALCULATIONS ---
if not df.empty:
    # SAFETY: Ensure all required columns exist as Series (prevents AttributeError)
    required_cols = ['Fed_Assets', 'TGA', 'RRP', 'CPI', 'Margin_Debt', 'SP500', 'VIX', 'HY_Spread']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0  # Create as float series of zeros

    # Now the math is safe
    df['Net_Liq'] = df['Fed_Assets'] - (df['TGA'].fillna(0) + df['RRP'].fillna(0))
    df['Net_Liq_YoY'] = df['Net_Liq'].pct_change(periods=252) * 100
    
    # CPI YoY calculation (for real growth)
    df['CPI_YoY'] = df['CPI'].pct_change(periods=365) * 100
    
    # Real M2 Growth
    if 'M2' in df.columns:
        m2_growth = df['M2'].pct_change(periods=365) * 100
        df['M2_Real_Growth'] = m2_growth - df['CPI_YoY'].fillna(0)
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
    # 2. FINRA Debt Velocity
    if 'Margin_Debt' in df.columns:
        # Since it's monthly data joined to daily, we use 252 (trading days) or 12 (months) 
        # based on how it's sampled. 252 is safer for daily charts.
        df['Margin_Velocity'] = df['Margin_Debt'].pct_change(periods=252) * 100
    else:
        df['Margin_Velocity'] = 0

# --- 3. CALCULATIONS & SCORING ---
if not df.empty:
    # 1. Ensure all columns exist as Series to prevent the 'bool' AttributeError
    required_cols = {
        'Net_Liq_YoY': 0, 'Net_Liq': 0, 'Net_Liq_SMA': 0, 
        'M2_Real_Growth': 0, 'Real_10Y_Yield': 5, 'HY_Z': 1, 
        'Funding_Stress': 50, 'VIX': 20, 'VIX_SMA': 20
    }
    for col, default in required_cols.items():
        if col not in df.columns:
            df[col] = default # This creates a Series filled with the default value

    # 2. Now calculate signals safely (using actual Series)
    s1 = (df['Net_Liq_YoY'] > 0).astype(int) * 20
    s2 = (df['Net_Liq'] > df['Net_Liq_SMA']).astype(int) * 20
    s3 = (df['M2_Real_Growth'] > 0).astype(int) * 15
    s4 = (df['Real_10Y_Yield'] < 1.0).astype(int) * 15
    s5 = (df['HY_Z'] < 0.0).astype(int) * 10
    s6 = (df['Funding_Stress'] < 15).astype(int) * 10
    s7 = (df['VIX'] < df['VIX_SMA']).astype(int) * 10

    df['Allocation_Pct'] = s1 + s2 + s3 + s4 + s5 + s6 + s7

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