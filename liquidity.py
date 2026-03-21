import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import yfinance as yf
from fredapi import Fred
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Macro Regime Monitor", layout="wide")
st.title("🛡️ Institutional Risk & Liquidity Monitor")

# Absolute shutdown of LaTeX rendering to prevent the ValueError
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'custom'

if "FRED_API_KEY" in st.secrets:
    api_key = st.secrets["FRED_API_KEY"]
else:
    api_key = st.text_input("Enter FRED API Key", type="password")

if not api_key:
    st.info("Please enter your FRED API key to begin.")
    st.stop()

fred = Fred(api_key=api_key)

# --- 2. DATA FETCHING (Anti-Freeze & Multi-Source Sync) ---
@st.cache_data(ttl=3600)
def get_master_data():
    start_date = "1950-01-01"
    
    # A. Fetch S&P 500 (Master Timeline)
    df_sp = pd.DataFrame()
    try:
        df_sp = yf.download("^GSPC", start=start_date, interval="1d", progress=False)
        if not df_sp.empty:
            if isinstance(df_sp.columns, pd.MultiIndex):
                df_sp.columns = df_sp.columns.get_level_values(0)
            df_sp = df_sp[['Close']].rename(columns={'Close': 'SP500'})
            # Normalize index to date-only to ensure perfect alignment with FRED
            df_sp.index = pd.to_datetime(df_sp.index).date
    except: pass

    # Anti-Crash: Backup S&P 500 from FRED if Yahoo fails
    if df_sp.empty:
        try:
            s_fred_sp = fred.get_series('SP500', observation_start=start_date)
            df_sp = s_fred_sp.to_frame('SP500')
            df_sp.index = pd.to_datetime(df_sp.index).date
        except: pass

    # B. Fetch Macro & Daily Indicators from FRED
    series_ids = {
        'VIXCLS': 'VIX',                # Daily
        'BAMLH0A0HYM2': 'HY_Spread',    # Daily
        'DTWEXBGS': 'USD_Index',        # Daily
        'WALCL': 'Fed_Assets',          # Weekly
        'M2SL': 'M2',                   # Monthly
        'CPIAUCSL': 'CPI',              # Monthly
        'WTREGEN': 'TGA', 'RRPONTSYD': 'RRP', 'TB3MS': '3M_Bill',
        'SOFR': 'SOFR', 'TGCRRATE': 'TGCR', 'DFII10': 'Real_10Y_Yield',
        'T10Y2Y': 'Yield_Curve_2s10s', 'USREC': 'Recessions'
    }
    
    df_macro = pd.DataFrame()
    for s_id, name in series_ids.items():
        try:
            s = fred.get_series(s_id, observation_start=start_date)
            if s is not None:
                # Force every FRED series to date-only index
                s.index = pd.to_datetime(s.index).date
                df_macro[name] = s
        except: pass

    # C. Merge & Sync (The Fix for the "Flat Line" Freeze)
    # Using concat on date-only indices prevents timestamp mismatch gaps
    df_combined = pd.concat([df_sp, df_macro], axis=1).sort_index()
    
    # Forward fill carries weekly/monthly data to daily rows without freezing daily series
    df_combined = df_combined.ffill()
    
    # Remove weekends/non-trading days
    df_combined = df_combined.dropna(subset=['SP500'])
    
    # Reconvert to Datetime for matplotlib compatibility
    df_combined.index = pd.to_datetime(df_combined.index)
    return df_combined

df = get_master_data()

# --- 3. CALCULATIONS ---
if not df.empty:
    df['Net_Liq'] = df['Fed_Assets'] - (df.get('TGA', 0).fillna(0) + df.get('RRP', 0).fillna(0))
    df['Net_Liq_YoY'] = df['Net_Liq'].pct_change(365) * 100
    df['Net_Liq_SMA'] = df['Net_Liq'].rolling(21).mean()
    df['CPI_YoY'] = df['CPI'].pct_change(365) * 100
    df['M2_Real_Growth'] = (df['M2'].pct_change(365) * 100) - df['CPI_YoY']
    df['SP500_SMA200'] = df['SP500'].rolling(200).mean()
    
    if 'HY_Spread' in df.columns:
        df['HY_Z'] = (df['HY_Spread'] - df['HY_Spread'].rolling(1095).mean()) / df['HY_Spread'].rolling(1095).std()
    if 'VIX' in df.columns:
        df['VIX_SMA'] = df['VIX'].rolling(20).mean()
    if 'SOFR' in df.columns and 'TGCR' in df.columns:
        df['Funding_Stress'] = (df['SOFR'] - df['TGCR']).interpolate().ffill() * 100

    # Allocation Logic
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
timeline_options = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS').tolist()
if df.index.max() not in timeline_options:
    timeline_options.append(df.index.max())
timeline_options = sorted(list(set(timeline_options)))

start_sel, end_sel = st.select_slider("Select Period", options=timeline_options, value=(timeline_options[-121], timeline_options[-1]), format_func=lambda x: x.strftime('%Y-%m'))
p_df = df.loc[start_sel:end_sel]

# --- 5. PLOTTING (The Global Requirement Wrapper) ---
fig, axes = plt.subplots(11, 1, figsize=(14, 75))

def plain_formatter(x, pos):
    return f'{x:,.0f}'

def format_ax(ax, title, use_log=False):
    ax.set_title(title, loc='left', fontweight='bold', fontsize=14)
    # 1. Year labels on bottom of every chart
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # 2. Quarterly lighter vertical grid + Yearly solid grid
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.grid(True, which='minor', axis='x', color='gray', linestyle=':', alpha=0.2)
    ax.grid(True, which='major', axis='x', color='gray', linestyle='-', alpha=0.4)
    # 3. Horizontal grids on all charts
    ax.grid(True, which='major', axis='y', alpha=0.4)
    # 4. Gray Recession Shading on every chart
    if 'Recessions' in p_df.columns:
        ax.fill_between(p_df.index, 0, 1, where=p_df['Recessions']>0, color='gray', alpha=0.2, transform=ax.get_xaxis_transform(), zorder=-1)
    
    ax.tick_params(labelbottom=True)
    if use_log: ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(plain_formatter))

def get_s(col):
    return p_df[col] if col in p_df.columns else pd.Series(np.zeros(len(p_df)), index=p_df.index)

# [Charts 1-11]
axes[0].plot(p_df.index, p_df['SP500'], color='black', lw=2)
if 'SP500_SMA200' in p_df.columns:
    axes[0].plot(p_df.index, p_df['SP500_SMA200'], color='red', ls='--', lw=1.2)
format_ax(axes[0], "1. S&P 500 (Log) vs 200D SMA", use_log=True)

axes[1].fill_between(p_df.index, get_s('Allocation_Pct'), color='blue', alpha=0.1)
axes[1].plot(p_df.index, get_s('Allocation_Pct'), color='blue', lw=1.5)
format_ax(axes[1], "2. System Allocation %")

axes[2].plot(p_df.index, get_s('Net_Liq'), color='darkgreen', lw=1.5)
axes[2].plot(p_df.index, get_s('Net_Liq_SMA'), color='black', ls='--', alpha=0.5)
format_ax(axes[2], "3. Net Liquidity Path")

axes[3].axhline(0, color='red', ls=':')
axes[3].plot(p_df.index, get_s('M2_Real_Growth'), color='purple')
format_ax(axes[3], "4. Real M2 Growth")

if 'HY_Z' in p_df.columns:
    axes[4].plot(p_df.index, p_df['HY_Z'], color='orange')
    axes[4].invert_yaxis()
    axes[4].axhline(0, color='red', ls=':')
format_ax(axes[4], "5. HY Spread Z-Score (Inverted)")

axes[5].axhline(1.0, color='red', ls=':')
axes[5].plot(p_df.index, get_s('Real_10Y_Yield'), color='darkblue')
format_ax(axes[5], "6. Real 10Y Yield")

axes[6].axhline(0, color='black')
axes[6].plot(p_df.index, get_s('Yield_Curve_2s10s'), color='darkgreen')
format_ax(axes[6], "7. Yield Curve (10Y-2Y)")

axes[7].plot(p_df.index, get_s('USD_Index'), color='navy')
format_ax(axes[7], "8. USD Index")

axes[8].plot(p_df.index, get_s('VIX'), color='red', alpha=0.6)
axes[8].plot(p_df.index, get_s('VIX_SMA'), color='black', ls='--')
format_ax(axes[8], "9. VIX vs 20D SMA")

axes[9].axhline(15, color='red', ls=':')
axes[9].plot(p_df.index, get_s('Funding_Stress'), color='blue')
format_ax(axes[9], "10. Funding Stress (SOFR-TGCR)")

axes[10].plot(p_df.index, get_s('Recessions'), color='gray', alpha=0.5)
format_ax(axes[10], "11. Recession Indicator (Overlay Active on all)")

plt.tight_layout(pad=4.0)
st.pyplot(fig)

# --- 6. DOWNLOAD CSV BUTTON ---
st.markdown("---")
csv_data = p_df.to_csv().encode('utf-8')
st.download_button(label="📥 DOWNLOAD CSV", data=csv_data, file_name='macro_monitor.csv', mime='text/csv')