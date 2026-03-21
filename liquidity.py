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

# FIX: Use 'stix' or 'cm' which are self-contained in Matplotlib 
# This avoids searching for system fonts like 'cursive' or 'Apple Chancery'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['text.usetex'] = False

if "FRED_API_KEY" in st.secrets:
    api_key = st.secrets["FRED_API_KEY"]
else:
    api_key = st.text_input("Enter FRED API Key", type="password")

if not api_key:
    st.info("Please enter your FRED API key to begin.")
    st.stop()

fred = Fred(api_key=api_key)

# --- 2. DATA FETCHING (Anti-Freeze Engine) ---
@st.cache_data(ttl=3600)
def get_master_data():
    start_date = "1950-01-01"
    
    # S&P 500 Daily
    try:
        df_sp = yf.download("^GSPC", start=start_date, interval="1d", progress=False)
        if not df_sp.empty:
            if isinstance(df_sp.columns, pd.MultiIndex):
                df_sp.columns = df_sp.columns.get_level_values(0)
            df_sp = df_sp[['Close']].rename(columns={'Close': 'SP500'})
            df_sp.index = pd.to_datetime(df_sp.index).date
    except:
        df_sp = pd.DataFrame()

    # Macro & Indicators
    series_ids = {
        'VIXCLS': 'VIX', 'BAMLH0A0HYM2': 'HY_Spread', 'DTWEXBGS': 'USD_Index',
        'WALCL': 'Fed_Assets', 'M2SL': 'M2', 'CPIAUCSL': 'CPI',
        'WTREGEN': 'TGA', 'RRPONTSYD': 'RRP', 'TB3MS': '3M_Bill',
        'SOFR': 'SOFR', 'TGCRRATE': 'TGCR', 'DFII10': 'Real_10Y_Yield',
        'T10Y2Y': 'Yield_Curve_2s10s', 'USREC': 'Recessions'
    }
    
    df_macro = pd.DataFrame()
    for s_id, name in series_ids.items():
        try:
            s = fred.get_series(s_id, observation_start=start_date)
            if s is not None:
                s.index = pd.to_datetime(s.index).date
                df_macro[name] = s
        except: pass

    # CONCAT ensures daily rows (VIX) match SP500 rows exactly
    df_combined = pd.concat([df_sp, df_macro], axis=1).sort_index()
    df_combined = df_combined.ffill()
    df_combined = df_combined.dropna(subset=['SP500'])
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

    s_liq = (df['Net_Liq_YoY'] > 0).astype(int) * 40
    s_m2 = (df['M2_Real_Growth'] > 0).astype(int) * 30
    s_vix = (df.get('VIX', 50) < df.get('VIX_SMA', 0)).astype(int) * 30
    df['Allocation_Pct'] = (s_liq + s_m2 + s_vix).clip(0, 100)

# --- 4. PERIOD SLIDER ---
timeline = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS').tolist()
if df.index.max() not in timeline: timeline.append(df.index.max())
start_s, end_s = st.select_slider("Select Period", options=sorted(list(set(timeline))), 
                                  value=(timeline[-121], timeline[-1]), format_func=lambda x: x.strftime('%Y-%m'))
p_df = df.loc[start_s:end_s]

# --- 5. PLOTTING ---
try:
    fig, axes = plt.subplots(11, 1, figsize=(14, 75))

    def format_ax(ax, title, use_log=False):
        ax.set_title(title, loc='left', fontweight='bold', fontsize=14)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.grid(True, which='minor', axis='x', color='gray', linestyle=':', alpha=0.2)
        ax.grid(True, which='major', axis='x', color='gray', linestyle='-', alpha=0.4)
        ax.grid(True, which='major', axis='y', alpha=0.4)
        if 'Recessions' in p_df.columns:
            ax.fill_between(p_df.index, 0, 1, where=p_df['Recessions']>0, color='gray', alpha=0.2, transform=ax.get_xaxis_transform(), zorder=-1)
        ax.tick_params(labelbottom=True)
        if use_log: ax.set_yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))

    def get_s(col): return p_df[col] if col in p_df.columns else pd.Series(np.zeros(len(p_df)), index=p_df.index)

    # 1-11 Plots
    axes[0].plot(p_df.index, p_df['SP500'], color='black', lw=2)
    format_ax(axes[0], "1. S&P 500 (Log)", use_log=True)

    axes[1].plot(p_df.index, get_s('Allocation_Pct'), color='blue', lw=1.5)
    format_ax(axes[1], "2. System Allocation %")

    axes[2].plot(p_df.index, get_s('Net_Liq'), color='darkgreen')
    format_ax(axes[2], "3. Net Liquidity Path")

    axes[3].plot(p_df.index, get_s('M2_Real_Growth'), color='purple')
    format_ax(axes[3], "4. Real M2 Growth")

    if 'HY_Z' in p_df.columns:
        axes[4].plot(p_df.index, p_df['HY_Z'], color='orange')
        axes[4].invert_yaxis()
    format_ax(axes[4], "5. HY Spread Z-Score")

    axes[5].plot(p_df.index, get_s('Real_10Y_Yield'), color='darkblue')
    format_ax(axes[5], "6. Real 10Y Yield")

    axes[6].plot(p_df.index, get_s('Yield_Curve_2s10s'), color='darkgreen')
    format_ax(axes[6], "7. Yield Curve")

    axes[7].plot(p_df.index, get_s('USD_Index'), color='navy')
    format_ax(axes[7], "8. USD Index")

    axes[8].plot(p_df.index, get_s('VIX'), color='red', alpha=0.6)
    format_ax(axes[8], "9. VIX")

    axes[9].plot(p_df.index, get_s('Funding_Stress'), color='blue')
    format_ax(axes[9], "10. Funding Stress")

    axes[10].plot(p_df.index, get_s('Recessions'), color='gray', alpha=0.5)
    format_ax(axes[10], "11. Recession Indicator")

    plt.tight_layout(pad=4.0)
    st.pyplot(fig)
except Exception as e:
    st.error(f"Visualization Error: {e}")

# --- 6. DOWNLOAD ---
st.download_button("📥 DOWNLOAD CSV", p_df.to_csv().encode('utf-8'), "macro_monitor.csv", "text/csv")