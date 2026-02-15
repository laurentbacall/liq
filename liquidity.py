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

CACHE_FILE = "macro_persistence_v7.parquet"

if "FRED_API_KEY" in st.secrets:
    api_key = st.secrets["FRED_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter FRED API Key", type="password")

if not api_key:
    st.info("Please enter your FRED API key to begin.")
    st.stop()

fred = Fred(api_key=api_key)

# --- 2. DATA FETCHING & ALERT SYSTEM ---

@st.cache_data(ttl=3600)
def get_master_data():
    series_ids = {
        'WALCL': 'Fed_Assets', 'WTREGEN': 'TGA', 'RRPONTSYD': 'RRP',
        'TB3MS': '3M_Bill', 'CPIAUCSL': 'CPI', 'M2SL': 'M2', 
        'BAMLH0A0HYM2': 'HY_Spread', 'SOFR': 'SOFR', 'TGCRRATE': 'TGCR', 
        'VIXCLS': 'VIX_FRED', 'BOGZ1FL663067003Q': 'Margin_Proxy',
        'USREC': 'Recessions'
    }
    
    # Explicit Series Existence Check
    missing_series = []
    updates = []
    
    for s_id, name in series_ids.items():
        try:
            s = fred.get_series(s_id)
            if s is None or s.empty:
                missing_series.append(f"{name} ({s_id})")
            else:
                s.name = name
                updates.append(s.to_frame())
        except Exception:
            missing_series.append(f"{name} ({s_id})")

    if missing_series:
        st.warning(f"⚠️ Critical Data Missing from FRED: {', '.join(missing_series)}")

    # Fetch Yahoo Finance Data
    try:
        yf_df = yf.download(["^GSPC", "^VIX"], start="1950-01-01", interval="1d", progress=False)
        if not yf_df.empty:
            close_data = yf_df['Close'].copy()
            close_data.columns = ['SP500', 'VIX']
            close_data.index = close_data.index.tz_localize(None)
            updates.append(close_data)
    except: pass

    if updates:
        df_main = pd.concat(updates, axis=1)
        df_main.index = pd.to_datetime(df_main.index)
        df_main = df_main.resample('D').ffill().ffill()
        return df_main
    return pd.DataFrame()

df = get_master_data()

# --- 3. CALCULATIONS ---
if not df.empty:
    # Monetary Impulse
    df['Net_Liq'] = df['Fed_Assets'] - (df.get('TGA', 0).fillna(0) + df.get('RRP', 0).fillna(0))
    net_liq_z = (df['Net_Liq'] - df['Net_Liq'].rolling(1095).mean()) / df['Net_Liq'].rolling(1095).std()
    df['CPI_YoY'] = df['CPI'].pct_change(365) * 100
    m2_growth = (df['M2'].pct_change(365) * 100) - df['CPI_YoY']
    m2_z = (m2_growth - m2_growth.rolling(1095).mean()) / m2_growth.rolling(1095).std()
    df['Monetary_Impulse_Z'] = (net_liq_z.fillna(0) + m2_z.fillna(0)) / 2

    # Restored HY Spread Z only (Original style)
    if 'HY_Spread' in df.columns:
        df['HY_Z'] = (df['HY_Spread'] - df['HY_Spread'].rolling(1095).mean()) / df['HY_Spread'].rolling(1095).std()

    # Real Rates
    df['Real_3M_Rate'] = df.get('3M_Bill', 0) - df.get('CPI_YoY', 0)
    
    # Funding Stress (bps)
    if 'SOFR' in df.columns and 'TGCR' in df.columns:
        df['Funding_Stress'] = (df['SOFR'] - df['TGCR']).interpolate().ffill() * 100

# --- 4. EXPORT (Sorted: Newest First) ---
st.sidebar.markdown("---")
# Sorting descending for the audit CSV
audit_df = df.sort_index(ascending=False)
csv = audit_df.to_csv().encode('utf-8')
st.sidebar.download_button("📥 Download Audit CSV (Newest First)", data=csv, file_name="macro_audit_latest.csv", mime="text/csv")

# --- 5. DASHBOARD PLOTTING ---
monthly_range = pd.date_range(start='1950-01-01', end=df.index.max(), freq='MS')
start, end = st.sidebar.select_slider("Period", options=monthly_range, value=(monthly_range[-240], monthly_range[-1]), format_func=lambda x: x.strftime('%Y'))
p_df = df.loc[start:end]

fig, axes = plt.subplots(5, 1, figsize=(14, 24), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1, 1]})

def apply_style(ax, title):
    ax.set_title(title, loc='left', fontweight='bold', fontsize=13)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    # Restoration of Quarterly Vertical Lines
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.grid(True, which='major', axis='x', color='gray', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', axis='x', color='gray', linestyle=':', alpha=0.15)
    ax.grid(True, which='major', axis='y', color='gray', linestyle=':', alpha=0.2)
    ax.tick_params(labelbottom=True)

# Panel 1: S&P 500
axes[0].plot(p_df.index, p_df.get('SP500', pd.Series(0, index=p_df.index)), color='black', lw=2)
axes[0].set_yscale('log')
apply_style(axes[0], "1. S&P 500 (Log Scale)")

# Panel 2: Monetary Impulse
if 'Monetary_Impulse_Z' in p_df.columns:
    axes[1].plot(p_df.index, p_df['Monetary_Impulse_Z'], color='purple', lw=1.5)
    axes[1].axhline(0, color='black', lw=1)
    axes[1].fill_between(p_df.index, 0, p_df['Monetary_Impulse_Z'], where=p_df['Monetary_Impulse_Z']>0, color='green', alpha=0.2)
    axes[1].fill_between(p_df.index, 0, p_df['Monetary_Impulse_Z'], where=p_df['Monetary_Impulse_Z']<0, color='red', alpha=0.2)
apply_style(axes[1], "2. Monetary Impulse Z-Score")

# Panel 3: HY Spread Z (Restored with Amber/Cyan)
if 'HY_Z' in p_df.columns:
    axes[2].plot(p_df.index, p_df['HY_Z'], color='black', lw=0.8, alpha=0.5)
    axes[2].axhline(0, color='black', lw=1)
    axes[2].fill_between(p_df.index, 0, p_df['HY_Z'], where=p_df['HY_Z']>0, color='orange', alpha=0.4, label="Risk-Off (Amber)")
    axes[2].fill_between(p_df.index, 0, p_df['HY_Z'], where=p_df['HY_Z']<0, color='cyan', alpha=0.3, label="Risk-On (Cyan)")
apply_style(axes[2], "3. High Yield Spread Z-Score (Swapped)")

# Panel 4: Real Rates (Swapped)
axes[3].plot(p_df.index, p_df.get('Real_3M_Rate', pd.Series(0, index=p_df.index)), color='red')
axes[3].axhline(2, color='darkred', ls='--', alpha=0.5)
axes[3].axhline(0, color='green', ls='--', alpha=0.5)
apply_style(axes[3], "4. Real 3M Bill Rate (%) (Swapped)")

# Panel 5: Funding Stress & VIX
if 'Funding_Stress' in p_df.columns:
    axes[4].plot(p_df.index, p_df['Funding_Stress'], color='blue', lw=1.5, label="SOFR-TGCR")
if 'VIX' in p_df.columns:
    ax4_2 = axes[4].twinx()
    ax4_2.plot(p_df.index, p_df['VIX'], color='red', alpha=0.15)
apply_style(axes[4], "5. Funding Stress (bps) & VIX")

plt.subplots_adjust(left=0.08, right=0.92, top=0.97, bottom=0.05, hspace=0.3)
st.pyplot(fig)