import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from fredapi import Fred

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
        'SP500': 'SP500_FRED',
        'WALCL': 'Fed_Assets', 'WTREGEN': 'TGA', 'RRPONTSYD': 'RRP',
        'TB3MS': '3M_Bill', 'CPIAUCSL': 'CPI', 'M2SL': 'M2', 
        'BAMLH0A0HYM2': 'HY_Spread', 'SOFR': 'SOFR', 'TGCRRATE': 'TGCR', 
        'VIXCLS': 'VIX', 'BOGZ1FL663067003Q': 'Margin_Proxy',
        'USREC': 'Recessions', 'DFII10': 'Real_10Y_Yield',
        'T10Y2Y': 'Yield_Curve_2s10s', 'DTWEXBGS': 'USD_Index'
    }
    
    updates = []
    for s_id, name in series_ids.items():
        try:
            s = fred.get_series(s_id)
            if s is not None:
                updates.append(s.to_frame(name))
        except: pass

    df_main = pd.concat(updates, axis=1) if updates else pd.DataFrame()

    # Checklist Item 10: Anti-Crash Fail-safe for Yahoo
    try:
        yf_df = yf.download("^GSPC", start="1950-01-01", interval="1d", progress=False)
        if not yf_df.empty:
            if isinstance(yf_df.columns, pd.MultiIndex):
                yf_df.columns = yf_df.columns.get_level_values(0)
            df_main['SP500_YF'] = yf_df['Close']
    except:
        st.warning("Yahoo Finance Rate Limited. Falling back to FRED (2014+).")

    if not df_main.empty:
        df_main.index = pd.to_datetime(df_main.index).tz_localize(None)
        # Checklist Item 1: S&P 500 Foundation
        if 'SP500_YF' in df_main.columns:
            df_main['SP500'] = df_main['SP500_YF'].combine_first(df_main.get('SP500_FRED', pd.Series(dtype='float64')))
        else:
            df_main['SP500'] = df_main.get('SP500_FRED', pd.Series(dtype='float64'))
        df_main = df_main.sort_index().ffill()
        
    return df_main

df = get_master_data()

# --- 3. CALCULATIONS ---
if not df.empty:
    # Item 2: Liquidity
    df['Net_Liq'] = df['Fed_Assets'] - (df.get('TGA', 0).fillna(0) + df.get('RRP', 0).fillna(0))
    df['Net_Liq_YoY'] = df['Net_Liq'].pct_change(365) * 100
    
    # Item 3 & 5: M2 and Real Rates
    df['CPI_YoY'] = df['CPI'].pct_change(365) * 100
    df['M2_Real_Level'] = df['M2'] / (df['CPI'] / 100)
    df['M2_Real_Growth'] = (df['M2'].pct_change(365) * 100) - df['CPI_YoY']
    
    # RESTORED: Real 3M Rate calculation
    df['Real_3M_Rate'] = df.get('3M_Bill', 0) - df.get('CPI_YoY', 0)
    
    # Item 4: HY Spread Z-Score
    if 'HY_Spread' in df.columns:
        df['HY_Z'] = (df['HY_Spread'] - df['HY_Spread'].rolling(1095).mean()) / df['HY_Spread'].rolling(1095).std()

    # Item 9: Leverage (min_periods=500 for earlier start)
    if 'Margin_Proxy' in df.columns and 'SP500' in df.columns:
        df['Lev_Ratio'] = df['Margin_Proxy'] / df['SP500']
        df['Leverage_Z'] = (df['Lev_Ratio'] - df['Lev_Ratio'].rolling(2500, min_periods=500).mean()) / \
                           df['Lev_Ratio'].rolling(2500, min_periods=500).std()

    # Item 8: Funding Stress
    if 'SOFR' in df.columns and 'TGCR' in df.columns:
        df['Funding_Stress'] = (df['SOFR'] - df['TGCR']).interpolate().ffill() * 100

# --- 4. PERIOD SLIDER ---
monthly_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')
start, end = st.select_slider("Select Monitoring Period", options=monthly_range, value=(monthly_range[-120], monthly_range[-1]), format_func=lambda x: x.strftime('%Y'))
p_df = df.loc[start:end]

# --- 5. PLOTTING ---
fig, axes = plt.subplots(9, 1, figsize=(14, 48), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1, 1, 1, 1, 1, 1]})

def apply_style(ax, title):
    # Item 10: Year labels on every chart
    ax.set_title(title, loc='left', fontweight='bold', fontsize=13)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(True, which='major', axis='both', color='gray', linestyle='-', alpha=0.3)
    ax.tick_params(labelbottom=True, labelsize=10)
    # Item 10: Gray Recession Shading
    if 'Recessions' in p_df.columns:
        ax.fill_between(p_df.index, ax.get_ylim()[0], ax.get_ylim()[1], where=p_df['Recessions']>0, color='gray', alpha=0.15)

# 1. S&P 500 (Item 1: Log Scale + Minor Gridlines)
axes[0].plot(p_df.index, p_df['SP500'], color='black', lw=2)
axes[0].set_yscale('log')
axes[0].grid(True, which='minor', axis='y', color='gray', linestyle=':', alpha=0.1)
apply_style(axes[0], "1. S&P 500 (Log Scale)")

# 2. Net Liquidity (Item 2: Shading + YoY Line + Green/Red Fills)
ax2_y = axes[1].twinx()
axes[1].fill_between(p_df.index, p_df['Net_Liq'], color='blue', alpha=0.08)
ax2_y.plot(p_df.index, p_df['Net_Liq_YoY'], color='blue', lw=1.5)
ax2_y.axhline(0, color='black', lw=1, alpha=0.5)
ax2_y.fill_between(p_df.index, 0, p_df['Net_Liq_YoY'], where=p_df['Net_Liq_YoY']>0, color='green', alpha=0.2)
ax2_y.fill_between(p_df.index, 0, p_df['Net_Liq_YoY'], where=p_df['Net_Liq_YoY']<0, color='red', alpha=0.2)
apply_style(axes[1], "2. Net Liquidity: Level & YoY % Growth")

# 3. M2 Real (Item 3: Shading + YoY Line + Teal/Orange Fills)
ax3_y = axes[2].twinx()
axes[2].fill_between(p_df.index, p_df['M2_Real_Level'], color='purple', alpha=0.08)
ax3_y.plot(p_df.index, p_df['M2_Real_Growth'], color='purple', lw=1.5)
ax3_y.axhline(0, color='black', lw=1, alpha=0.5)
ax3_y.fill_between(p_df.index, 0, p_df['M2_Real_Growth'], where=p_df['M2_Real_Growth']>0, color='teal', alpha=0.2)
ax3_y.fill_between(p_df.index, 0, p_df['M2_Real_Growth'], where=p_df['M2_Real_Growth']<0, color='orange', alpha=0.2)
apply_style(axes[2], "3. Real M2: Level & YoY % Growth")

# 4. HY Spread (Item 4: Absolute AND Z-Score Inverted)
ax4_z = axes[3].twinx()
if 'HY_Spread' in p_df.columns:
    axes[3].fill_between(p_df.index, p_df['HY_Spread'], color='orange', alpha=0.1)
    ax4_z.plot(p_df.index, p_df['HY_Z'], color='black', lw=1.2)
    axes[3].invert_yaxis()
    ax4_z.invert_yaxis()
apply_style(axes[3], "4. High Yield Spread & Z-Score (Inverted)")

# 5. Real Rates (Item 5: BOTH Real 10Y and Real 3M)
axes[4].plot(p_df.index, p_df.get('Real_10Y_Yield', 0), color='darkblue', label='Real 10Y (TIPS)', lw=1.8)
axes[4].plot(p_df.index, p_df.get('Real_3M_Rate', 0), color='red', label='Real 3M Bill', alpha=0.4, lw=1)
axes[4].axhline(0, color='black', lw=1)
axes[4].legend(loc='upper left', fontsize='small')
apply_style(axes[4], "5. Real Rates (%)")

# 6. Yield Curve (Item 6: Red/Green Fills)
if 'Yield_Curve_2s10s' in p_df.columns:
    axes[5].plot(p_df.index, p_df['Yield_Curve_2s10s'], color='darkgreen', lw=1.5)
    axes[5].axhline(0, color='black', lw=1)
    axes[5].fill_between(p_df.index, 0, p_df['Yield_Curve_2s10s'], where=p_df['Yield_Curve_2s10s']<0, color='red', alpha=0.2)
    axes[5].fill_between(p_df.index, 0, p_df['Yield_Curve_2s10s'], where=p_df['Yield_Curve_2s10s']>0, color='green', alpha=0.1)
apply_style(axes[5], "6. Yield Curve (10Y-2Y)")

# 7. USD Index (Item 7)
if 'USD_Index' in p_df.columns:
    axes[6].plot(p_df.index, p_df['USD_Index'], color='navy', lw=1.5)
apply_style(axes[6], "7. U.S. Dollar Index")

# 8. Funding Stress & VIX (Item 8: Blue Line + Red Twin VIX)
if 'Funding_Stress' in p_df.columns:
    axes[7].plot(p_df.index, p_df['Funding_Stress'], color='blue', lw=1.2)
if 'VIX' in p_df.columns:
    ax8_v = axes[7].twinx()
    ax8_v.plot(p_df.index, p_df['VIX'], color='red', alpha=0.2)
apply_style(axes[7], "8. Funding Stress & VIX")

# 9. Leverage (Item 9: Starts early)
if 'Leverage_Z' in p_df.columns:
    axes[8].plot(p_df.index, p_df['Leverage_Z'], color='orange', lw=1.5)
    axes[8].axhline(0, color='black', lw=1)
apply_style(axes[8], "9. Systemic Leverage Z-Score")

plt.subplots_adjust(left=0.08, right=0.92, top=0.98, bottom=0.02, hspace=0.45)
st.pyplot(fig)