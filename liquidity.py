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

# Persistence File
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

@st.cache_data(ttl=86400)
def fetch_finra_margin():
    """Fetches and cleans FINRA Margin Debt data from their Excel source."""
    url = "https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx"
    try:
        # Note: FINRA formatting often requires skipping header rows
        df_finra = pd.read_excel(url, skiprows=4)
        df_finra = df_finra.dropna(subset=['Year-Month'])
        # Convert 'Year-Month' to datetime
        df_finra['Date'] = pd.to_datetime(df_finra['Year-Month'], format='%Y-%m') + pd.offsets.MonthEnd(0)
        df_finra = df_finra.set_index('Date')
        # We want the 'Debit Balances in Customers' Securities Margin Accounts'
        margin_col = [c for c in df_finra.columns if 'Debit Balances' in str(c)][0]
        s = df_finra[margin_col].astype(float)
        s.name = "Margin_Debt"
        return s
    except Exception as e:
        st.warning(f"Could not update FINRA data: {e}. Using latest cached value.")
        return pd.Series(dtype=float)

@st.cache_data(ttl=3600)
def get_master_data():
    # 1. Load Persistence
    if os.path.exists(CACHE_FILE):
        df = pd.read_parquet(CACHE_FILE)
        last_sync = df.index.max()
    else:
        df = pd.DataFrame()
        last_sync = pd.to_datetime("1950-01-01")

    # 2. Only fetch if the last data point is older than yesterday
    if not df.empty and (pd.Timestamp.now() - last_sync).days < 1:
        return df

    # 3. Fetch FRED Updates
    series_ids = {
        'WALCL': 'Fed_Assets', 'WTREGEN': 'TGA', 'RRPONTSYD': 'RRP',
        'TB3MS': '3M_Bill', 'CPIAUCSL': 'CPI', 'M2SL': 'M2', 
        'BAMLH0A0HYM2': 'HY_Spread', 'BAMLC0A0CM': 'IG_Spread',
        'SOFR': 'SOFR', 'TGCR': 'TGCR', 'VIXCLS': 'VIX_FRED',
        'USREC': 'Recessions'
    }
    
    updates = []
    for s_id, name in series_ids.items():
        try:
            s = fred.get_series(s_id, observation_start=last_sync)
            s.name = name
            updates.append(s)
        except: continue
    
    # 4. Fetch Yahoo Updates
    try:
        yf_tickers = ["^GSPC", "^VIX"]
        yf_df = yf.download(yf_tickers, start=last_sync, interval="1d", progress=False)['Close']
        yf_df.index = yf_df.index.tz_localize(None)
        yf_df.rename(columns={'^GSPC': 'SP500', '^VIX': 'VIX'}, inplace=True)
        updates.append(yf_df)
    except: pass

    # 5. Fetch FINRA Update
    updates.append(fetch_finra_margin())

    # 6. Merge & Save
    new_data = pd.concat(updates, axis=1).resample('D').ffill()
    if not df.empty:
        # Use combine_first to update existing rows and append new ones
        df = new_data.combine_first(df)
    else:
        df = new_data

    df.sort_index().to_parquet(CACHE_FILE)
    return df

# Load Data
df = get_master_data()

# --- 3. CALCULATIONS ---
df = df.ffill()

# A. Liquidity Decomposition
# Pillar 1: Fed Net Liquidity (Z-Score)
df['Net_Liq'] = df['Fed_Assets'] - (df['TGA'].fillna(0) + df['RRP'].fillna(0))
df['Net_Liq_Z'] = (df['Net_Liq'] - df['Net_Liq'].rolling(1095).mean()) / df['Net_Liq'].rolling(1095).std()

# Pillar 2: M2 Real Growth (Z-Score)
df['CPI_YoY'] = df['CPI'].pct_change(365) * 100
df['M2_Real_Growth'] = (df['M2'].pct_change(365) * 100) - df['CPI_YoY']
df['M2_Real_Z'] = (df['M2_Real_Growth'] - df['M2_Real_Growth'].rolling(1095).mean()) / df['M2_Real_Growth'].rolling(1095).std()

# Pillar 3: Real 3M Rate (Absolute)
df['Real_3M_Rate'] = df['3M_Bill'] - df['CPI_YoY']

# B. Credit Quality: Quality Spread (HY - IG)
df['Quality_Spread'] = df['HY_Spread'] - df['IG_Spread']

# C. Trading Leverage: Margin Debt to S&P Proxy
# FINRA data is monthly, we ratio it against SP500 price
df['Leverage_Ratio'] = df['Margin_Debt'] / df['SP500']
df['Leverage_Z'] = (df['Leverage_Ratio'] - df['Leverage_Ratio'].rolling(2500).mean()) / df['Leverage_Ratio'].rolling(2500).std()

# D. Funding Stress
df['Funding_Stress'] = df['SOFR'] - df['TGCR']

# --- 4. DASHBOARD & PLOTTING ---

# Date Selection
monthly_range = pd.date_range(start='1950-01-01', end=df.index.max(), freq='MS')
start_date, end_date = st.sidebar.select_slider(
    "Select Analysis Period", 
    options=monthly_range, 
    value=(monthly_range[-240], monthly_range[-1]), 
    format_func=lambda x: x.strftime('%Y')
)

p_df = df.loc[start_date:end_date].copy()

fig, axes = plt.subplots(5, 1, figsize=(14, 22), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1, 1]})

def apply_institutional_style(ax, title, label_y=""):
    ax.set_title(title, loc='left', fontweight='bold', fontsize=13, color='#2E2E2E')
    ax.set_ylabel(label_y, fontsize=10)
    ax.grid(True, which='major', axis='both', color='#E0E0E0', linestyle='-', alpha=0.5)
    # Ensure Year Labels on every panel
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', labelbottom=True, labelsize=10)
    if 'Recessions' in p_df.columns:
        ax.fill_between(p_df.index, ax.get_ylim()[0], ax.get_ylim()[1], where=p_df['Recessions']>0, color='gray', alpha=0.1)

# PANEL 1: Market Performance & Leverage
axes[0].plot(p_df.index, p_df['SP500'], color='black', lw=2, label="S&P 500")
axes[0].set_yscale('log')
ax0_2 = axes[0].twinx()
ax0_2.plot(p_df.index, p_df['Leverage_Z'], color='orange', alpha=0.4, label="Leverage Z")
ax0_2.fill_between(p_df.index, 1.5, p_df['Leverage_Z'], where=(p_df['Leverage_Z']>=1.5), color='red', alpha=0.2)
apply_institutional_style(axes[0], "1. Market Performance (Log) & Trading Leverage (Z-Score)")

# PANEL 2: Liquidity Flow Decomposition (Z-Scores)
axes[1].plot(p_df.index, p_df['Net_Liq_Z'], color='purple', lw=1.5, label="Fed Net Liquidity (Z)")
axes[1].plot(p_df.index, p_df['M2_Real_Z'], color='blue', lw=1.2, alpha=0.6, label="M2 Real Growth (Z)")
axes[1].axhline(0, color='black', lw=1)
apply_institutional_style(axes[1], "2. Monetary Flow Components (Z-Scores)")
axes[1].legend(loc='upper left', frameon=False)

# PANEL 3: Real 3M Bill Rate (Absolute %)
axes[2].plot(p_df.index, p_df['Real_3M_Rate'], color='#D32F2F', lw=1.5)
axes[2].axhline(2, color='darkred', linestyle='--', alpha=0.6, label="Restrictive (>2%)")
axes[2].axhline(0, color='green', linestyle='--', alpha=0.6, label="Accommodative (<0%)")
axes[2].fill_between(p_df.index, 2, p_df['Real_3M_Rate'], where=(p_df['Real_3M_Rate']>=2), color='red', alpha=0.1)
apply_institutional_style(axes[2], "3. Real 3M Bill Rate (Inflation-Adjusted %)")
axes[2].legend(loc='upper left', frameon=False)

# PANEL 4: Credit Quality Spread (HY - IG)
axes[3].plot(p_df.index, p_df['Quality_Spread'], color='#5D4037', lw=1.5)
apply_institutional_style(axes[3], "4. Quality Spread (HY - IG) [Default Risk Premium]")

# PANEL 5: Funding Stress & Equity Volatility
axes[4].plot(p_df.index, p_df['Funding_Stress'], color='#0097A7', label="SOFR-TGCR Spread")
ax4_2 = axes[4].twinx()
ax4_2.plot(p_df.index, p_df['VIX'], color='red', alpha=0.2, label="VIX Index")
apply_institutional_style(axes[4], "5. Plumbing (SOFR-TGCR) & Fear (VIX)")
axes[4].legend(loc='upper left', frameon=False)

plt.tight_layout()
st.pyplot(fig)