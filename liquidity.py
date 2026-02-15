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
        'VIXCLS': 'VIX_FRED', 'BOGZ1FL663067003Q': 'Margin_Proxy',
        'USREC': 'Recessions',
        'DFII10': 'Real_10Y_Yield',
        'T10Y2Y': 'Yield_Curve_2s10s',
        'DTWEXBGS': 'USD_Index'
    }
    
    updates = []
    for s_id, name in series_ids.items():
        try:
            s = fred.get_series(s_id)
            if s is not None:
                s.name = name
                updates.append(s.to_frame())
        except: pass

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
        df_main = df_main.sort_index().ffill().ffill()
        return df_main
    return pd.DataFrame()

df = get_master_data()

# --- 3. CALCULATIONS ---
if not df.empty:
    # 1. Net Liquidity Level + Growth Z
    df['Net_Liq'] = df['Fed_Assets'] - (df.get('TGA', 0).fillna(0) + df.get('RRP', 0).fillna(0))
    df['Net_Liq_YoY'] = df['Net_Liq'].pct_change(365) * 100
    df['Net_Liq_Growth_Z'] = (df['Net_Liq_YoY'] - df['Net_Liq_YoY'].rolling(1095).mean()) / df['Net_Liq_YoY'].rolling(1095).std()
    
    # 2. M2 Real Level + Growth Z
    df['CPI_YoY'] = df['CPI'].pct_change(365) * 100
    df['M2_Real_Level'] = df['M2'] / (df['CPI'] / 100)
    df['M2_Real_Growth'] = (df['M2'].pct_change(365) * 100) - df['CPI_YoY']
    df['M2_Growth_Z'] = (df['M2_Real_Growth'] - df['M2_Real_Growth'].rolling(1095).mean()) / df['M2_Real_Growth'].rolling(1095).std()

    # 3. HY Spread Z
    if 'HY_Spread' in df.columns:
        df['HY_Z'] = (df['HY_Spread'] - df['HY_Spread'].rolling(1095).mean()) / df['HY_Spread'].rolling(1095).std()

    # 4. Real Rates
    df['Real_3M_Rate'] = df.get('3M_Bill', 0) - df.get('CPI_YoY', 0)
    
    # 5. Leverage Z
    if 'Margin_Proxy' in df.columns and 'SP500' in df.columns:
        df['Lev_Ratio'] = df['Margin_Proxy'] / df['SP500']
        df['Leverage_Z'] = (df['Lev_Ratio'] - df['Lev_Ratio'].rolling(2500).mean()) / df['Lev_Ratio'].rolling(2500).std()

    # 6. Funding Stress
    if 'SOFR' in df.columns and 'TGCR' in df.columns:
        df['Funding_Stress'] = (df['SOFR'] - df['TGCR']).interpolate().ffill() * 100

# --- 4. UI: PERIOD SLIDER ---
monthly_range = pd.date_range(start='1950-01-01', end=df.index.max(), freq='MS')
start, end = st.select_slider("Select Monitoring Period", options=monthly_range, value=(monthly_range[-240], monthly_range[-1]), format_func=lambda x: x.strftime('%Y'))
p_df = df.loc[start:end]

# --- 5. DASHBOARD PLOTTING ---
fig, axes = plt.subplots(9, 1, figsize=(14, 48), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1, 1, 1, 1, 1, 1]})

def apply_style(ax, title, invert_y=False):
    ax.set_title(title, loc='left', fontweight='bold', fontsize=13)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.grid(True, which='major', axis='x', color='gray', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', axis='x', color='gray', linestyle=':', alpha=0.15)
    ax.grid(True, which='major', axis='y', color='gray', linestyle=':', alpha=0.2)
    
    # FIX: Restore labels on shared x-axis
    ax.tick_params(labelbottom=True, labelsize=10)
    
    if invert_y: ax.invert_yaxis()
    if 'Recessions' in p_df.columns:
        # Shading restored for every panel
        ax.fill_between(p_df.index, ax.get_ylim()[0], ax.get_ylim()[1], where=p_df['Recessions']>0, color='gray', alpha=0.15)

# 1. S&P 500
axes[0].plot(p_df.index, p_df.get('SP500', 0), color='black', lw=2)
axes[0].set_yscale('log')
apply_style(axes[0], "1. S&P 500 (Log Scale)")

# 2. Net Liquidity: Level (Shaded) + Growth Z (Line)
ax2_level = axes[1]
ax2_z = axes[1].twinx()
ax2_level.fill_between(p_df.index, p_df['Net_Liq'], color='blue', alpha=0.08)
ax2_z.plot(p_df.index, p_df['Net_Liq_Growth_Z'], color='blue', lw=1.5)
ax2_z.axhline(0, color='black', lw=1, alpha=0.5)
ax2_z.fill_between(p_df.index, 0, p_df['Net_Liq_Growth_Z'], where=p_df['Net_Liq_Growth_Z']>0, color='green', alpha=0.2)
ax2_z.fill_between(p_df.index, 0, p_df['Net_Liq_Growth_Z'], where=p_df['Net_Liq_Growth_Z']<0, color='red', alpha=0.2)
apply_style(axes[1], "2. Net Liquidity: Absolute Level ($) & Growth Z-Score")

# 3. M2 Real: Level (Shaded) + Growth Z (Line)
ax3_level = axes[2]
ax3_z = axes[2].twinx()
ax3_level.fill_between(p_df.index, p_df['M2_Real_Level'], color='purple', alpha=0.08)
ax3_z.plot(p_df.index, p_df['M2_Growth_Z'], color='purple', lw=1.5)
ax3_z.axhline(0, color='black', lw=1, alpha=0.5)
ax3_z.fill_between(p_df.index, 0, p_df['M2_Growth_Z'], where=p_df['M2_Growth_Z']>0, color='teal', alpha=0.2)
ax3_z.fill_between(p_df.index, 0, p_df['M2_Growth_Z'], where=p_df['M2_Growth_Z']<0, color='orange', alpha=0.2)
apply_style(axes[2], "3. Real M2: Absolute Level & Growth Z-Score")

# 4. HY Spread Z (Inverted)
if 'HY_Z' in p_df.columns:
    axes[3].plot(p_df.index, p_df['HY_Z'], color='black', lw=0.8, alpha=0.4)
    axes[3].axhline(0, color='black', lw=1)
    axes[3].fill_between(p_df.index, 0, p_df['HY_Z'], where=p_df['HY_Z']>0, color='orange', alpha=0.4)
    axes[3].fill_between(p_df.index, 0, p_df['HY_Z'], where=p_df['HY_Z']<0, color='cyan', alpha=0.3)
apply_style(axes[3], "4. High Yield Spread Z-Score (Inverted: Negative Stress at Top)", invert_y=True)

# 5. Real Rates (10Y & 3M)
axes[4].plot(p_df.index, p_df.get('Real_10Y_Yield', 0), color='darkblue', label='Real 10Y (TIPS)', lw=1.8)
axes[4].plot(p_df.index, p_df.get('Real_3M_Rate', 0), color='red', label='Real 3M Bill', alpha=0.4, lw=1)
axes[4].axhline(0, color='black', lw=1)
axes[4].legend(loc='upper left', fontsize='small')
apply_style(axes[4], "5. Real Rates (%) - Valuation Gravity")

# 6. Yield Curve
if 'Yield_Curve_2s10s' in p_df.columns:
    axes[5].plot(p_df.index, p_df['Yield_Curve_2s10s'], color='darkgreen', lw=1.5)
    axes[5].axhline(0, color='black', lw=1)
    axes[5].fill_between(p_df.index, 0, p_df['Yield_Curve_2s10s'], where=p_df['Yield_Curve_2s10s']<0, color='red', alpha=0.2)
apply_style(axes[5], "6. Yield Curve (10Y-2Y Spread)")

# 7. USD Index
if 'USD_Index' in p_df.columns:
    axes[6].plot(p_df.index, p_df['USD_Index'], color='navy', lw=1.5)
apply_style(axes[6], "7. U.S. Dollar Index (Trade Weighted)")

# 8. Funding Stress & VIX
if 'Funding_Stress' in p_df.columns:
    axes[7].plot(p_df.index, p_df['Funding_Stress'], color='blue', lw=1.2)
if 'VIX' in p_df.columns:
    ax7_2 = axes[7].twinx()
    ax7_2.plot(p_df.index, p_df['VIX'], color='red', alpha=0.1)
apply_style(axes[7], "8. Funding Stress (bps) & VIX")

# 9. Leverage
if 'Leverage_Z' in p_df.columns:
    axes[8].plot(p_df.index, p_df['Leverage_Z'], color='orange', lw=1.5)
    axes[8].axhline(0, color='black', lw=1)
apply_style(axes[8], "9. Systemic Leverage Z-Score")

plt.subplots_adjust(left=0.08, right=0.92, top=0.98, bottom=0.02, hspace=0.45)
st.pyplot(fig)

# --- 6. EXPORT ---
st.markdown("---")
csv = df.sort_index(ascending=False).to_csv().encode('utf-8')
st.download_button("📥 Download Combined Audit CSV", data=csv, file_name="macro_monitor_final.csv", mime="text/csv")