import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from fredapi import Fred
import os
import requests
import io

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Macro Regime Monitor", layout="wide")
st.title("🛡️ Institutional Risk & Liquidity Monitor")

# Incremental versioning for the cache
CACHE_FILE = "macro_persistence_v5.parquet"

if "FRED_API_KEY" in st.secrets:
    api_key = st.secrets["FRED_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter FRED API Key", type="password")

if not api_key:
    st.info("Please enter your FRED API key to begin.")
    st.stop()

fred = Fred(api_key=api_key)

# --- 2. ROBUST DATA FETCHING ---

def fetch_finra_margin():
    """Fetches FINRA Margin Debt with error handling."""
    url = "https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        df_finra = pd.read_excel(io.BytesIO(response.content), skiprows=4)
        df_finra = df_finra.dropna(subset=['Year-Month'])
        df_finra['Date'] = pd.to_datetime(df_finra['Year-Month'], format='%Y-%m') + pd.offsets.MonthEnd(0)
        df_finra = df_finra.set_index('Date')
        margin_col = [c for c in df_finra.columns if 'Debit Balances' in str(c)][0]
        s = df_finra[margin_col].astype(float)
        s.name = "Margin_Debt"
        return s.to_frame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_master_data():
    if os.path.exists(CACHE_FILE):
        try:
            df_main = pd.read_parquet(CACHE_FILE)
            df_main.index = pd.to_datetime(df_main.index)
            last_sync = df_main.index.max()
        except:
            df_main = pd.DataFrame()
            last_sync = pd.to_datetime("1950-01-01")
    else:
        df_main = pd.DataFrame()
        last_sync = pd.to_datetime("1950-01-01")

    if not df_main.empty and (pd.Timestamp.now() - last_sync).days < 1:
        return df_main

    updates = []
    series_ids = {
        'WALCL': 'Fed_Assets', 'WTREGEN': 'TGA', 'RRPONTSYD': 'RRP',
        'TB3MS': '3M_Bill', 'CPIAUCSL': 'CPI', 'M2SL': 'M2', 
        'BAMLH0A0HYM2': 'HY_Spread', 'BAMLC0A0CM': 'IG_Spread',
        'SOFR': 'SOFR', 'TGCR': 'TGCR', 'VIXCLS': 'VIX_FRED',
        'BOGZ1FL663067003Q': 'Margin_Proxy',
        'USREC': 'Recessions'
    }
    
    for s_id, name in series_ids.items():
        try:
            s = fred.get_series(s_id, observation_start=last_sync)
            if not s.empty:
                s.name = name
                updates.append(s.to_frame())
        except: continue
    
    try:
        yf_df = yf.download(["^GSPC", "^VIX"], start=last_sync, interval="1d", progress=False)
        if not yf_df.empty:
            close_data = yf_df['Close'].copy()
            close_data.columns = ['SP500', 'VIX']
            close_data.index = close_data.index.tz_localize(None)
            updates.append(close_data)
    except: pass

    finra_data = fetch_finra_margin()
    if not finra_data.empty: updates.append(finra_data)

    if updates:
        new_data = pd.concat(updates, axis=1)
        new_data.index = pd.to_datetime(new_data.index)
        new_data = new_data.resample('D').ffill()
        df_main = new_data.combine_first(df_main) if not df_main.empty else new_data
        df_main.sort_index().to_parquet(CACHE_FILE)
    
    return df_main

df = get_master_data()

# --- 3. REFINED CALCULATIONS ---
if not df.empty:
    df = df.ffill()

    # 1. Consolidated Monetary Flow
    if 'Fed_Assets' in df.columns and 'M2' in df.columns:
        df['Net_Liq'] = df['Fed_Assets'] - (df.get('TGA', 0).fillna(0) + df.get('RRP', 0).fillna(0))
        net_liq_z = (df['Net_Liq'] - df['Net_Liq'].rolling(1095).mean()) / df['Net_Liq'].rolling(1095).std()
        df['CPI_YoY'] = df['CPI'].pct_change(365) * 100
        m2_growth = (df['M2'].pct_change(365) * 100) - df['CPI_YoY']
        m2_z = (m2_growth - m2_growth.rolling(1095).mean()) / m2_growth.rolling(1095).std()
        df['Monetary_Impulse_Z'] = (net_liq_z.fillna(0) + m2_z.fillna(0)) / 2

    # 2. Quality Spread Z-Score
    if 'HY_Spread' in df.columns and 'IG_Spread' in df.columns:
        df['Quality_Spread_Abs'] = df['HY_Spread'] - df['IG_Spread']
        df['Quality_Spread_Z'] = (df['Quality_Spread_Abs'] - df['Quality_Spread_Abs'].rolling(1095).mean()) / df['Quality_Spread_Abs'].rolling(1095).std()

    # 3. Leverage (Safety Logic)
    lev_data = df.get('Margin_Debt', df.get('Margin_Proxy', pd.Series(index=df.index, dtype=float)))
    if 'SP500' in df.columns and not lev_data.dropna().empty:
        df['Leverage_Ratio'] = lev_data / df['SP500']
        df['Leverage_Z'] = (df['Leverage_Ratio'] - df['Leverage_Ratio'].rolling(2500).mean()) / df['Leverage_Ratio'].rolling(2500).std()

    # 4. Funding (Fix for Visibility)
    if 'SOFR' in df.columns and 'TGCR' in df.columns:
        df['Funding_Stress'] = (df['SOFR'] - df['TGCR']).interpolate().ffill() * 100

# --- 4. DATA EXPORT FEATURE ---
st.sidebar.markdown("---")
st.sidebar.subheader("Export Data")
csv = df.to_csv().encode('utf-8')
st.sidebar.download_button("Download All Series (CSV)", data=csv, file_name="macro_audit_data.csv", mime="text/csv")

# --- 5. DASHBOARD PLOTTING ---
monthly_range = pd.date_range(start='1950-01-01', end=df.index.max(), freq='MS')
start, end = st.sidebar.select_slider("Period", options=monthly_range, value=(monthly_range[-240], monthly_range[-1]), format_func=lambda x: x.strftime('%Y'))
p_df = df.loc[start:end]

fig, axes = plt.subplots(5, 1, figsize=(14, 24), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1, 1]})

def apply_style(ax, title):
    ax.set_title(title, loc='left', fontweight='bold', fontsize=13)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.grid(True, which='major', axis='x', color='gray', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', axis='x', color='gray', linestyle=':', alpha=0.1)
    ax.grid(True, which='major', axis='y', color='gray', linestyle=':', alpha=0.2)
    ax.tick_params(labelbottom=True)
    if 'Recessions' in p_df.columns:
        ax.fill_between(p_df.index, ax.get_ylim()[0], ax.get_ylim()[1], where=p_df['Recessions']>0, color='gray', alpha=0.1)

# Panel 1: S&P & Leverage
axes[0].plot(p_df.index, p_df.get('SP500', 0), color='black', lw=2)
axes[0].set_yscale('log')
if 'Leverage_Z' in p_df.columns:
    ax0_2 = axes[0].twinx()
    ax0_2.plot(p_df.index, p_df['Leverage_Z'], color='orange', alpha=0.5)
apply_style(axes[0], "1. S&P 500 & Trading Leverage Z-Score")

# Panel 2: Monetary Flow
if 'Monetary_Impulse_Z' in p_df.columns:
    axes[1].plot(p_df.index, p_df['Monetary_Impulse_Z'], color='purple', lw=1.5)
    axes[1].axhline(0, color='black', lw=1)
    axes[1].fill_between(p_df.index, 0, p_df['Monetary_Impulse_Z'], where=p_df['Monetary_Impulse_Z']>0, color='green', alpha=0.2)
    axes[1].fill_between(p_df.index, 0, p_df['Monetary_Impulse_Z'], where=p_df['Monetary_Impulse_Z']<0, color='red', alpha=0.2)
apply_style(axes[1], "2. Monetary Impulse (Unified Z-Score)")

# Panel 3: Real Rates
axes[2].plot(p_df.index, p_df.get('Real_3M_Rate', 0), color='red')
axes[2].axhline(2, color='darkred', ls='--', alpha=0.5)
axes[2].axhline(0, color='green', ls='--', alpha=0.5)
apply_style(axes[2], "3. Real 3M Bill Rate (%)")

# Panel 4: Quality Spread Z
if 'Quality_Spread_Z' in p_df.columns:
    axes[3].plot(p_df.index, p_df['Quality_Spread_Z'], color='brown')
    axes[3].axhline(0, color='black', lw=1)
apply_style(axes[3], "4. Quality Spread (HY-IG) Z-Score")

# Panel 5: Funding & VIX
if 'Funding_Stress' in p_df.columns:
    axes[4].plot(p_df.index, p_df['Funding_Stress'], color='cyan', label="SOFR-TGCR (bps)")
if 'VIX' in p_df.columns:
    ax4_2 = axes[4].twinx()
    ax4_2.plot(p_df.index, p_df['VIX'], color='red', alpha=0.2, label="VIX")
apply_style(axes[4], "5. Funding Stress (bps) & VIX")

plt.tight_layout()
st.pyplot(fig)