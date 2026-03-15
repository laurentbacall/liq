import streamlit as st
import pandas as pd
import numpy as np
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

# --- 3. CALCULATIONS & ALLOCATION MATH ---
if not df.empty:
    # Liquidity
    df['Net_Liq'] = df['Fed_Assets'] - (df.get('TGA', 0).fillna(0) + df.get('RRP', 0).fillna(0))
    df['Net_Liq_YoY'] = df['Net_Liq'].pct_change(365) * 100
    df['Net_Liq_SMA'] = df['Net_Liq'].rolling(21).mean()
    
    # Valuation
    df['CPI_YoY'] = df['CPI'].pct_change(365) * 100
    df['M2_Real_Growth'] = (df['M2'].pct_change(365) * 100) - df['CPI_YoY']
    
    # Credit/Vol
    if 'HY_Spread' in df.columns:
        df['HY_Z'] = (df['HY_Spread'] - df['HY_Spread'].rolling(1095).mean()) / df['HY_Spread'].rolling(1095).std()
    if 'VIX' in df.columns:
        df['VIX_SMA'] = df['VIX'].rolling(20).mean()
    if 'SOFR' in df.columns and 'TGCR' in df.columns:
        df['Funding_Stress'] = (df['SOFR'] - df['TGCR']).interpolate().ffill() * 100

    # SCORING ENGINE (Clean 100 Point Scale)
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
fig, axes = plt.subplots(11, 1, figsize=(14, 55), sharex=True)

def plot_line(ax, data, title, color='black', alpha=1.0, lw=1.5, ls='-'):
    if not data.empty:
        ax.plot(data.index, data, color=color, alpha=alpha, lw=lw, linestyle=ls)
    ax.set_title(title, loc='left', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)

# 1. SP500
plot_line(axes[0], p_df['SP500'], "1. S&P 500 (Log)")
axes[0].set_yscale('log')

# 2. Allocation
axes[1].fill_between(p_df.index, p_df['Allocation_Pct'], color='blue', alpha=0.1)
plot_line(axes[1], p_df['Allocation_Pct'], "2. Target Allocation %", color='blue')
axes[1].set_ylim(-5, 105)

# 3. Liquidity Ribbon
axes[2].plot(p_df.index, p_df['Net_Liq_SMA'], color='black', linestyle='--', alpha=0.5)
axes[2].fill_between(p_df.index, p_df['Net_Liq'], p_df['Net_Liq_SMA'], where=p_df['Net_Liq']>=p_df['Net_Liq_SMA'], color='green', alpha=0.3)
axes[2].fill_between(p_df.index, p_df['Net_Liq'], p_df['Net_Liq_SMA'], where=p_df['Net_Liq']<p_df['Net_Liq_SMA'], color='red', alpha=0.3)
plot_line(axes[2], p_df['Net_Liq'], "3. Net Liquidity Flow", alpha=0.3)

# 4. M2 Real
axes[3].axhline(0, color='red', linestyle=':')
plot_line(axes[3], p_df['M2_Real_Growth'], "4. Real M2 Growth (Target > 0%)", color='purple')

# 5. HY Z-Score
if 'HY_Z' in p_df.columns:
    axes[4].axhline(0, color='red', linestyle=':')
    plot_line(axes[4], p_df['HY_Z'], "5. HY Spread Z-Score (Inverted, Target < 0)", color='orange')
    axes[4].invert_yaxis()

# 6. Real 10Y
axes[5].axhline(1.0, color='red', linestyle=':')
plot_line(axes[5], p_df.get('Real_10Y_Yield', pd.Series(dtype='float64')), "6. Real 10Y Yield (Target < 1.0%)", color='darkblue')

# 7. Curve
plot_line(axes[6], p_df.get('Yield_Curve_2s10s', pd.Series(dtype='float64')), "7. Yield Curve", color='darkgreen')
axes[6].axhline(0, color='black', lw=1)

# 8. USD
plot_line(axes[7], p_df.get('USD_Index', pd.Series(dtype='float64')), "8. USD Index", color='navy')

# 9. VIX
plot_line(axes[8], p_df.get('VIX', pd.Series(dtype='float64')), "9. VIX vs 20D SMA", color='red', lw=1)
plot_line(axes[8], p_df.get('VIX_SMA', pd.Series(dtype='float64')), "", color='black', ls='--', alpha=0.6)

# 10. Funding
axes[9].axhline(15, color='red', linestyle=':')
plot_line(axes[9], p_df.get('Funding_Stress', pd.Series(dtype='float64')), "10. Funding Stress (Target < 15bps)", color='blue')

# 11. Leverage
plot_line(axes[10], p_df.get('Leverage_Z', pd.Series(dtype='float64')), "11. Leverage Z-Score", color='brown')

plt.tight_layout()
st.pyplot(fig)

# --- 6. DOWNLOAD ---
st.download_button(label="📥 Download CSV", data=p_df.to_csv().encode('utf-8'), file_name='macro_data.csv')