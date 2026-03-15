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

    try:
        yf_df = yf.download("^GSPC", start="1950-01-01", interval="1d", progress=False)
        if not yf_df.empty:
            if isinstance(yf_df.columns, pd.MultiIndex):
                yf_df.columns = yf_df.columns.get_level_values(0)
            df_main['SP500_YF'] = yf_df['Close']
    except: pass

    if not df_main.empty:
        df_main.index = pd.to_datetime(df_main.index).tz_localize(None)
        if 'SP500_YF' in df_main.columns:
            df_main['SP500'] = df_main['SP500_YF'].combine_first(df_main.get('SP500_FRED', pd.Series(dtype='float64')))
        df_main = df_main.sort_index().ffill()
    return df_main

df = get_master_data()

# --- 3. CALCULATIONS & SCORING ---
if not df.empty:
    # A. Liquidity Pillar
    df['Net_Liq'] = df['Fed_Assets'] - (df.get('TGA', 0).fillna(0) + df.get('RRP', 0).fillna(0))
    df['Net_Liq_YoY'] = df['Net_Liq'].pct_change(365) * 100
    df['Net_Liq_SMA'] = df['Net_Liq'].rolling(21).mean()
    
    # B. Valuation Pillar
    df['CPI_YoY'] = df['CPI'].pct_change(365) * 100
    df['M2_Real_Growth'] = (df['M2'].pct_change(365) * 100) - df['CPI_YoY']
    df['M2_Real_Level'] = df['M2'] / (df['CPI'] / 100)
    
    # C. Credit & Vol Pillars
    if 'HY_Spread' in df.columns:
        df['HY_Z'] = (df['HY_Spread'] - df['HY_Spread'].rolling(1095).mean()) / df['HY_Spread'].rolling(1095).std()
    if 'VIX' in df.columns:
        df['VIX_SMA'] = df['VIX'].rolling(20).mean()
    if 'SOFR' in df.columns and 'TGCR' in df.columns:
        df['Funding_Stress'] = (df['SOFR'] - df['TGCR']).interpolate().ffill() * 100

    # D. ALLOCATION SCORING ENGINE (Vectorized for Graphing)
    liq_score = ( (df['Net_Liq_YoY'] > 0).astype(int) * 15 ) + ( (df['Net_Liq'] > df['Net_Liq_SMA']).astype(int) * 25 )
    credit_score = ( (df.get('HY_Z', 1) < 0.5).astype(int) * 15 ) + ( (df.get('Funding_Stress', 50) < 15).astype(int) * 15 )
    val_score = ( (df.get('Real_10Y_Yield', 5) < 1.5).astype(int) * 10 ) + ( (df['M2_Real_Growth'] > -2).astype(int) * 10 )
    vol_score = ( (df.get('VIX', 50) < df.get('VIX_SMA', 0)).astype(int) * 10 )
    
    df['Total_Score'] = liq_score + credit_score + val_score + vol_score
    
    # Map Score to Allocation Steps
    def map_alloc(s):
        if s >= 85: return 100
        if s >= 60: return 75
        if s >= 40: return 40
        return 0
    df['Allocation_Pct'] = df['Total_Score'].apply(map_alloc)

# --- 4. PERIOD SLIDER ---
monthly_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')
start, end = st.select_slider("Select Monitoring Period", options=monthly_range, value=(monthly_range[-120], monthly_range[-1]), format_func=lambda x: x.strftime('%Y'))
p_df = df.loc[start:end]

# --- 5. PLOTTING ---
fig, axes = plt.subplots(11, 1, figsize=(14, 58), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})

def apply_style(ax, title):
    ax.set_title(title, loc='left', fontweight='bold', fontsize=13)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(True, which='major', axis='both', color='gray', linestyle='-', alpha=0.3)
    ax.tick_params(labelbottom=True, labelsize=10)
    if 'Recessions' in p_df.columns:
        ax.fill_between(p_df.index, ax.get_ylim()[0], ax.get_ylim()[1], where=p_df['Recessions']>0, color='gray', alpha=0.15)

# 1. S&P 500
axes[0].plot(p_df.index, p_df['SP500'], color='black', lw=2)
axes[0].set_yscale('log')
apply_style(axes[0], "1. S&P 500 (Log Scale)")

# 2. NEW: Allocation %
axes[1].fill_between(p_df.index, p_df['Allocation_Pct'], color='blue', alpha=0.2)
axes[1].plot(p_df.index, p_df['Allocation_Pct'], color='blue', lw=1.5)
axes[1].set_ylim(-5, 105)
apply_style(axes[1], "2. Systematic Allocation (% S&P 500 Exposure)")

# 3. Net Liquidity (RIBBON CROSSOVER)
axes[2].plot(p_df.index, p_df['Net_Liq'], color='black', lw=0.8, alpha=0.5)
axes[2].plot(p_df.index, p_df['Net_Liq_SMA'], color='black', linestyle='--', lw=1.2)
axes[2].fill_between(p_df.index, p_df['Net_Liq'], p_df['Net_Liq_SMA'], where=p_df['Net_Liq']>=p_df['Net_Liq_SMA'], color='green', alpha=0.3)
axes[2].fill_between(p_df.index, p_df['Net_Liq'], p_df['Net_Liq_SMA'], where=p_df['Net_Liq']<p_df['Net_Liq_SMA'], color='red', alpha=0.3)
apply_style(axes[2], "3. Liquidity Ribbon (Price vs 21D SMA)")

# 4. M2 Real
ax4_y = axes[3].twinx()
axes[3].fill_between(p_df.index, p_df['M2_Real_Level'], color='purple', alpha=0.08)
ax4_y.plot(p_df.index, p_df['M2_Real_Growth'], color='purple', lw=1.5)
ax4_y.axhline(-2, color='red', linestyle=':', lw=1.5)
apply_style(axes[3], "4. Real M2 Growth (Red Dotted = -2% Threshold)")

# 5. HY Spread (Inverted)
ax5_z = axes[4].twinx()
if 'HY_Spread' in p_df.columns:
    axes[4].fill_between(p_df.index, p_df['HY_Spread'], color='orange', alpha=0.1)
    ax5_z.plot(p_df.index, p_df['HY_Z'], color='black', lw=1.2)
    ax5_z.axhline(0.5, color='red', linestyle=':', lw=1.5)
    axes[4].invert_yaxis()
    ax5_z.invert_yaxis()
apply_style(axes[4], "5. HY Spread Z-Score (Red Dotted = 0.5 Threshold)")

# 6. Real Rates
axes[5].plot(p_df.index, p_df.get('Real_10Y_Yield', 0), color='darkblue', lw=1.8, label='Real 10Y')
axes[5].axhline(1.5, color='red', linestyle=':', lw=1.5)
axes[5].axhline(0, color='black', lw=1)
apply_style(axes[5], "6. Real 10Y Yield (Red Dotted = 1.5% Threshold)")

# 7. Yield Curve
axes[6].plot(p_df.index, p_df.get('T10Y2Y', 0), color='darkgreen')
axes[6].axhline(0, color='black')
apply_style(axes[6], "7. Yield Curve")

# 8. USD
axes[7].plot(p_df.index, p_df.get('USD_Index', 0), color='navy')
apply_style(axes[7], "8. USD Index")

# 9. VIX Standalone
axes[8].plot(p_df.index, p_df.get('VIX', 0), color='red', lw=1)
axes[8].plot(p_df.index, p_df.get('VIX_SMA', 0), color='black', linestyle='--', lw=1)
apply_style(axes[8], "9. VIX vs 20D SMA")

# 10. Funding
axes[9].plot(p_df.index, p_df.get('Funding_Stress', 0), color='blue')
axes[9].axhline(15, color='red', linestyle=':', lw=1.5)
apply_style(axes[9], "10. Funding Stress (15bps Threshold)")

# 11. Leverage
axes[10].plot(p_df.index, p_df.get('Leverage_Z', 0), color='orange')
apply_style(axes[10], "11. Systemic Leverage Z-Score")

plt.subplots_adjust(left=0.08, right=0.92, top=0.98, bottom=0.04, hspace=0.6)
st.pyplot(fig)

# --- 6. DOWNLOAD FEATURE ---
csv = p_df.to_csv().encode('utf-8')
st.download_button(label="📥 Download Dashboard Data (CSV)", data=csv, file_name='macro_monitor_data.csv', mime='text/csv')