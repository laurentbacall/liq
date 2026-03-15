import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from fredapi import Fred
from datetime import datetime, timedelta

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Macro Regime Monitor", layout="wide")
st.title("🛡️ Institutional Risk & Liquidity Monitor")

if "FRED_API_KEY" in st.secrets:
    api_key = st.secrets["FRED_API_KEY"]
else:
    api_key = st.text_input("Enter FRED API Key", type="password")

if not api_key:
    st.info("Please enter your FRED API key to begin.")
    st.stop()

fred = Fred(api_key=api_key)

# --- 2. DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_master_data():
    series_ids = {
        'WALCL': 'Fed_Assets', 'WTREGEN': 'TGA', 'RRPONTSYD': 'RRP',
        'TB3MS': '3M_Bill', 'CPIAUCSL': 'CPI', 'M2SL': 'M2', 
        'BAMLH0A0HYM2': 'HY_Spread', 'SOFR': 'SOFR', 'TGCRRATE': 'TGCR', 
        'VIXCLS': 'VIX', 'SP500': 'SP500_FRED',
        'BOGZ1FL663067003Q': 'Margin_Proxy', 'USREC': 'Recessions',
        'DFII10': 'Real_10Y_Yield', 'T10Y2Y': 'Yield_Curve_2s10s', 'DTWEXBGS': 'USD_Index'
    }
    
    updates = []
    # 1. Fetch FRED Data
    for s_id, name in series_ids.items():
        try:
            s = fred.get_series(s_id)
            if s is not None:
                updates.append(s.to_frame(name))
        except: pass

    # 2. Yahoo Finance Strategy
    # A. Historical SP500 Patch (Weekly) to avoid rate limits
    try:
        yf_hist = yf.download("^GSPC", start="1950-01-01", end="2014-01-01", interval="1wk", progress=False)
        if not yf_hist.empty:
            hist_close = yf_hist['Close'].to_frame('SP500_Hist')
            hist_close.index = hist_close.index.tz_localize(None)
            updates.append(hist_close)
    except: pass

    # B. Recent Patch (Daily) for the last 30 days
    try:
        yf_recent = yf.download(["^GSPC", "^VIX"], period="1mo", interval="1d", progress=False)
        if not yf_recent.empty:
            recent_close = yf_recent['Close'].copy()
            recent_close.columns = ['SP500_Recent', 'VIX_Recent']
            recent_close.index = recent_close.index.tz_localize(None)
            updates.append(recent_close)
    except: pass

    if updates:
        df_main = pd.concat(updates, axis=1)
        df_main.index = pd.to_datetime(df_main.index)
        df_main = df_main.sort_index()

        # Merge SP500 sources: Recent > FRED > Historical
        df_main['SP500'] = df_main['SP500_Recent'].combine_first(df_main['SP500_FRED']).combine_first(df_main['SP500_Hist'])
        df_main['VIX'] = df_main['VIX_Recent'].combine_first(df_main['VIX'])
        
        return df_main.ffill().ffill()
    return pd.DataFrame()

df = get_master_data()

# --- 3. CALCULATIONS ---
if not df.empty:
    df['Net_Liq'] = df['Fed_Assets'] - (df.get('TGA', 0).fillna(0) + df.get('RRP', 0).fillna(0))
    df['Net_Liq_YoY'] = df['Net_Liq'].pct_change(365) * 100
    df['CPI_YoY'] = df['CPI'].pct_change(365) * 100
    df['M2_Real_Level'] = df['M2'] / (df['CPI'] / 100)
    df['M2_Real_Growth'] = (df['M2'].pct_change(365) * 100) - df['CPI_YoY']
    
    if 'HY_Spread' in df.columns:
        df['HY_Z'] = (df['HY_Spread'] - df['HY_Spread'].rolling(1095).mean()) / df['HY_Spread'].rolling(1095).std()

    df['Real_3M_Rate'] = df.get('3M_Bill', 0) - df.get('CPI_YoY', 0)
    
    if 'Margin_Proxy' in df.columns and 'SP500' in df.columns:
        df['Lev_Ratio'] = df['Margin_Proxy'] / df['SP500']
        # FIX: Added min_periods=500 to allow the curve to start earlier than 10 years
        df['Leverage_Z'] = (df['Lev_Ratio'] - df['Lev_Ratio'].rolling(2500, min_periods=500).mean()) / \
                           df['Lev_Ratio'].rolling(2500, min_periods=500).std()

    if 'SOFR' in df.columns and 'TGCRRATE' in df.columns:
        df['Funding_Stress'] = (df['SOFR'] - df['TGCRRATE']).interpolate().ffill() * 100

# --- 4. UI: PERIOD SLIDER ---
monthly_range = pd.date_range(start='1950-01-01', end=df.index.max(), freq='MS')
start, end = st.select_slider("Select Monitoring Period", options=monthly_range, value=(monthly_range[-240], monthly_range[-1]), format_func=lambda x: x.strftime('%Y'))
p_df = df.loc[start:end]

# --- 5. PLOTTING ---
fig, axes = plt.subplots(9, 1, figsize=(14, 48), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1, 1, 1, 1, 1, 1]})

def apply_style(ax, title, invert_y=False):
    ax.set_title(title, loc='left', fontweight='bold', fontsize=13)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(True, which='major', axis='both', color='gray', linestyle='-', alpha=0.3)
    ax.tick_params(labelbottom=True, labelsize=10)
    if invert_y: ax.invert_yaxis()
    if 'Recessions' in p_df.columns:
        ax.fill_between(p_df.index, ax.get_ylim()[0], ax.get_ylim()[1], where=p_df['Recessions']>0, color='gray', alpha=0.15)

# 1. S&P 500 (Enhanced with Minor Gridlines for Log Scale)
axes[0].plot(p_df.index, p_df.get('SP500', 0), color='black', lw=2)
axes[0].set_yscale('log')
axes[0].grid(True, which='minor', axis='y', color='gray', linestyle=':', alpha=0.1)
apply_style(axes[0], "1. S&P 500 (Log Scale)")

# 2. Net Liquidity
ax2_l = axes[1]; ax2_g = axes[1].twinx()
ax2_l.fill_between(p_df.index, p_df['Net_Liq'], color='blue', alpha=0.08)
ax2_g.plot(p_df.index, p_df['Net_Liq_YoY'], color='blue', lw=1.5)
ax2_g.axhline(0, color='black', lw=1, alpha=0.5)
apply_style(axes[1], "2. Net Liquidity: Level ($) & Actual YoY % Growth")

# 3. M2 Real
ax3_l = axes[2]; ax3_g = axes[2].twinx()
ax3_l.fill_between(p_df.index, p_df['M2_Real_Level'], color='purple', alpha=0.08)
ax3_g.plot(p_df.index, p_df['M2_Real_Growth'], color='purple', lw=1.5)
ax3_g.axhline(0, color='black', lw=1, alpha=0.5)
apply_style(axes[2], "3. Real M2: Level & Actual YoY % Growth")

# 4. HY Spread
ax4_l = axes[3]; ax4_z = axes[3].twinx()
if 'HY_Spread' in p_df.columns:
    ax4_l.fill_between(p_df.index, p_df['HY_Spread'], color='orange', alpha=0.1)
    ax4_z.plot(p_df.index, p_df['HY_Z'], color='black', lw=1.2, alpha=0.7)
    ax4_z.invert_yaxis()
apply_style(axes[3], "4. High Yield Spread: Absolute Level & Z-Score (Inverted)", invert_y=True)

# 5. Real Rates
axes[4].plot(p_df.index, p_df.get('Real_10Y_Yield', 0), color='darkblue', label='Real 10Y (TIPS)', lw=1.8)
axes[4].plot(p_df.index, p_df.get('Real_3M_Rate', 0), color='red', label='Real 3M Bill', alpha=0.4, lw=1)
axes[4].axhline(0, color='black', lw=1); axes[4].legend(loc='upper left', fontsize='small')
apply_style(axes[4], "5. Real Rates (%) - Valuation Gravity")

# 6-9: Other indicators
if 'Yield_Curve_2s10s' in p_df.columns: axes[5].plot(p_df.index, p_df['Yield_Curve_2s10s'], color='darkgreen', lw=1.5)
apply_style(axes[5], "6. Yield Curve (10Y-2Y Spread)")
if 'USD_Index' in p_df.columns: axes[6].plot(p_df.index, p_df['USD_Index'], color='navy', lw=1.5)
apply_style(axes[6], "7. U.S. Dollar Index")
if 'Funding_Stress' in p_df.columns: axes[7].plot(p_df.index, p_df['Funding_Stress'], color='blue', lw=1.2)
apply_style(axes[7], "8. Funding Stress (bps)")
if 'Leverage_Z' in p_df.columns: axes[8].plot(p_df.index, p_df['Leverage_Z'], color='orange', lw=1.5)
apply_style(axes[8], "9. Systemic Leverage Z-Score")

plt.subplots_adjust(left=0.08, right=0.92, top=0.98, bottom=0.02, hspace=0.45)
st.pyplot(fig)