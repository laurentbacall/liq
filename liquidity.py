import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from fredapi import Fred
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Macro Regime Monitor", layout="wide")
st.title("🛡️ Institutional Risk & Liquidity Monitor")

CACHE_FILE = "macro_data_persistence.parquet"

if "FRED_API_KEY" in st.secrets:
    api_key = st.secrets["FRED_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter FRED API Key", type="password")

if not api_key:
    st.info("Please enter your FRED API key to begin.")
    st.stop()

fred = Fred(api_key=api_key)

# --- 2. DATA UTILITIES ---

def fetch_finra_margin():
    """Fetches and cleans FINRA Margin Debt data."""
    url = "https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx"
    try:
        df_finra = pd.read_excel(url, skiprows=4)
        df_finra = df_finra.dropna(subset=['Year-Month'])
        df_finra['Date'] = pd.to_datetime(df_finra['Year-Month'], format='%Y-%m') + pd.offsets.MonthEnd(0)
        df_finra = df_finra.set_index('Date')
        margin_col = [c for c in df_finra.columns if 'Debit Balances' in str(c)][0]
        s = df_finra[margin_col].astype(float)
        s.name = "Margin_Debt"
        return s.to_frame() # Return as DF to ensure index stays consistent
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_master_data():
    # 1. Load Persistence
    if os.path.exists(CACHE_FILE):
        df_main = pd.read_parquet(CACHE_FILE)
        # Ensure index is datetime
        df_main.index = pd.to_datetime(df_main.index)
        last_sync = df_main.index.max()
    else:
        df_main = pd.DataFrame()
        last_sync = pd.to_datetime("1950-01-01")

    # Only fetch if the last data point is older than yesterday
    if not df_main.empty and (pd.Timestamp.now() - last_sync).days < 1:
        return df_main

    updates = []

    # 2. Fetch FRED
    series_ids = {
        'WALCL': 'Fed_Assets', 'WTREGEN': 'TGA', 'RRPONTSYD': 'RRP',
        'TB3MS': '3M_Bill', 'CPIAUCSL': 'CPI', 'M2SL': 'M2', 
        'BAMLH0A0HYM2': 'HY_Spread', 'BAMLC0A0CM': 'IG_Spread',
        'SOFR': 'SOFR', 'TGCR': 'TGCR', 'VIXCLS': 'VIX_FRED',
        'USREC': 'Recessions'
    }
    
    for s_id, name in series_ids.items():
        try:
            s = fred.get_series(s_id, observation_start=last_sync)
            if not s.empty:
                s.name = name
                updates.append(s.to_frame())
        except: continue
    
    # 3. Fetch Yahoo (SP500 & VIX)
    try:
        yf_df = yf.download(["^GSPC", "^VIX"], start=last_sync, interval="1d", progress=False)
        if not yf_df.empty:
            # Handle MultiIndex and extract 'Close'
            if 'Close' in yf_df.columns:
                close_data = yf_df['Close']
                close_data.columns = ['SP500', 'VIX']
                close_data.index = close_data.index.tz_localize(None)
                updates.append(close_data)
    except: pass

    # 4. Fetch FINRA
    finra_data = fetch_finra_margin()
    if not finra_data.empty:
        updates.append(finra_data)

    # 5. Merge & Save
    if updates:
        new_data = pd.concat(updates, axis=1)
        # Force index to Datetime to prevent the TypeError
        new_data.index = pd.to_datetime(new_data.index)
        new_data = new_data.resample('D').ffill()
        
        if not df_main.empty:
            df_main = new_data.combine_first(df_main)
        else:
            df_main = new_data

        df_main.sort_index().to_parquet(CACHE_FILE)
    
    return df_main

# Load Data
df = get_master_data()

# --- 3. CALCULATIONS ---
# Ensure we have data before calculating
if not df.empty:
    df = df.ffill()

    # Liquidity Decomposition
    if 'Fed_Assets' in df.columns:
        df['Net_Liq'] = df['Fed_Assets'] - (df.get('TGA', 0).fillna(0) + df.get('RRP', 0).fillna(0))
        df['Net_Liq_Z'] = (df['Net_Liq'] - df['Net_Liq'].rolling(1095).mean()) / df['Net_Liq'].rolling(1095).std()

    if 'CPI' in df.columns:
        df['CPI_YoY'] = df['CPI'].pct_change(365) * 100
        if 'M2' in df.columns:
            df['M2_Real_Growth'] = (df['M2'].pct_change(365) * 100) - df['CPI_YoY']
            df['M2_Real_Z'] = (df['M2_Real_Growth'] - df['M2_Real_Growth'].rolling(1095).mean()) / df['M2_Real_Growth'].rolling(1095).std()
        df['Real_3M_Rate'] = df.get('3M_Bill', 0) - df['CPI_YoY']

    # Quality Spread
    if 'HY_Spread' in df.columns and 'IG_Spread' in df.columns:
        df['Quality_Spread'] = df['HY_Spread'] - df['IG_Spread']

    # Leverage
    if 'Margin_Debt' in df.columns and 'SP500' in df.columns:
        df['Leverage_Ratio'] = df['Margin_Debt'] / df['SP500']
        df['Leverage_Z'] = (df['Leverage_Ratio'] - df['Leverage_Ratio'].rolling(2500).mean()) / df['Leverage_Ratio'].rolling(2500).std()

    # Funding
    if 'SOFR' in df.columns and 'TGCR' in df.columns:
        df['Funding_Stress'] = df['SOFR'] - df['TGCR']

# --- 4. DASHBOARD ---
if df.empty:
    st.error("No data could be retrieved. Check your API key and connection.")
    st.stop()

# Date Selection
monthly_range = pd.date_range(start='1950-01-01', end=df.index.max(), freq='MS')
start_date, end_date = st.sidebar.select_slider(
    "Select Period", options=monthly_range, 
    value=(monthly_range[-240], monthly_range[-1]), format_func=lambda x: x.strftime('%Y')
)

p_df = df.loc[start_date:end_date].copy()

fig, axes = plt.subplots(5, 1, figsize=(14, 22), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1, 1]})

def apply_institutional_style(ax, title):
    ax.set_title(title, loc='left', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.15)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', labelbottom=True)
    if 'Recessions' in p_df.columns:
        ax.fill_between(p_df.index, ax.get_ylim()[0], ax.get_ylim()[1], where=p_df['Recessions']>0, color='gray', alpha=0.1)

# Panel Plotting
# 1. Market & Leverage
if 'SP500' in p_df.columns:
    axes[0].plot(p_df.index, p_df['SP500'], color='black', lw=2)
    axes[0].set_yscale('log')
    if 'Leverage_Z' in p_df.columns:
        ax0_2 = axes[0].twinx()
        ax0_2.plot(p_df.index, p_df['Leverage_Z'], color='orange', alpha=0.4)
apply_institutional_style(axes[0], "1. Market & Leverage (Z)")

# 2. Liquidity
if 'Net_Liq_Z' in p_df.columns:
    axes[1].plot(p_df.index, p_df['Net_Liq_Z'], color='purple', label="Fed Net Liq")
if 'M2_Real_Z' in p_df.columns:
    axes[1].plot(p_df.index, p_df['M2_Real_Z'], color='blue', alpha=0.5, label="M2 Real")
axes[1].axhline(0, color='black', lw=1)
apply_institutional_style(axes[1], "2. Monetary Flow (Z)")

# 3. Real Rates
if 'Real_3M_Rate' in p_df.columns:
    axes[2].plot(p_df.index, p_df['Real_3M_Rate'], color='red')
    axes[2].axhline(2, color='darkred', ls='--', alpha=0.5)
    axes[2].axhline(0, color='green', ls='--', alpha=0.5)
apply_institutional_style(axes[2], "3. Real 3M Bill Rate (%)")

# 4. Quality Spread
if 'Quality_Spread' in p_df.columns:
    axes[3].plot(p_df.index, p_df['Quality_Spread'], color='brown')
apply_institutional_style(axes[3], "4. Quality Spread (HY-IG)")

# 5. Funding & Vol
if 'Funding_Stress' in p_df.columns:
    axes[4].plot(p_df.index, p_df['Funding_Stress'], color='cyan')
if 'VIX' in p_df.columns:
    ax4_2 = axes[4].twinx()
    ax4_2.plot(p_df.index, p_df['VIX'], color='red', alpha=0.2)
apply_institutional_style(axes[4], "5. Funding Stress & VIX")

plt.tight_layout()
st.pyplot(fig)