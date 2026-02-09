import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from fredapi import Fred

# --- 1. CONFIGURATION & SECRETS ---
st.set_page_config(page_title="Macro Regime Monitor", layout="wide")
st.title("🛡️ Institutional Risk & Liquidity Monitor")

if "FRED_API_KEY" in st.secrets:
    api_key = st.secrets["FRED_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter FRED API Key", type="password")

if not api_key:
    st.info("Please enter your FRED API key to begin.")
    st.stop()

fred = Fred(api_key=api_key)

# --- 2. DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_data():
    # Removed WILL5000PR as it was discontinued by FRED in June 2024
    series_ids = {
        'WALCL': 'Fed Assets', 'WTREGEN': 'TGA', 'RRPONTSYD': 'Reverse Repo',
        'TB3MS': '3M Bill', 'CPIAUCSL': 'CPI', 'M2SL': 'M2 Supply', 
        'BAMLH0A0HYM2': 'HY_Spread', 'SP500': 'SP500_FRED',
        'USREC': 'Recessions'
    }
    df_list = []
    for s_id, name in series_ids.items():
        try:
            s = fred.get_series(s_id)
            if s is not None and not s.empty:
                s.name = name
                df_list.append(s)
        except Exception: continue
    
    # Replacement for Wilshire: Yahoo Finance S&P 500 (History back to 1970s)
    sp_history = pd.Series(dtype=float)
    try:
        yf_df = yf.download("^GSPC", start="1970-01-01", interval="1mo", progress=False)
        if not yf_df.empty and 'Close' in yf_df.columns:
            sp_history = yf_df['Close']
            if isinstance(sp_history, pd.DataFrame): sp_history = sp_history.iloc[:, 0]
    except Exception: pass

    df = pd.concat(df_list, axis=1).resample('D').ffill()
    
    # --- S&P 500 STITCHING LOGIC (No Wilshire dependency) ---
    # Layer 1: FRED Native (Usually starts 2011)
    sp_final = df['SP500_FRED'] if 'SP500_FRED' in df.columns else pd.Series(index=df.index, dtype=float)
    
    # Layer 2: Yahoo Finance (Fills the gap from 1970-2011)
    if not sp_history.empty:
        yf_daily = sp_history.resample('D').interpolate(method='linear')
        sp_final = sp_final.combine_first(yf_daily)
    
    df['SP500'] = sp_final
    return df

df = get_data()

# --- 3. CALCULATIONS ---
df = df.ffill() 

# Net Liquidity Calculation
if 'Fed Assets' in df.columns:
    # Safely handle missing TGA/RRP by filling with 0
    tga = df['TGA'] if 'TGA' in df.columns else 0
    rrp = df['Reverse Repo'] if 'Reverse Repo' in df.columns else 0
    df['Net_Liquidity'] = df['Fed Assets'] - (tga.fillna(0) + rrp.fillna(0))
    df['Liquidity_Flow'] = df['Net_Liquidity'].pct_change(365) * 100

# Inflation & Growth
if 'CPI' in df.columns:
    df['CPI_YoY'] = df['CPI'].pct_change(365) * 100
    if 'M2 Supply' in df.columns:
        df['M2_Real_Growth'] = (df['M2 Supply'].pct_change(365) * 100) - df['CPI_YoY']
    if '3M Bill' in df.columns:
        df['Real_Rate'] = df['3M Bill'] - df['CPI_YoY']
        df['Rate_Momentum'] = df['Real_Rate'] - df['Real_Rate'].shift(365)

# Z-Score Logic
lookback = st.sidebar.slider("Z-Score Lookback (Years)", 1, 10, 3)
window = 365 * lookback
freq_map = {'Liquidity_Flow': 7, 'M2_Real_Growth': 30, 'Rate_Momentum': 30}

liq_pillars = [c for c in ['Liquidity_Flow', 'M2_Real_Growth', 'Rate_Momentum'] if c in df.columns]
for col in liq_pillars:
    min_obs = 30 * freq_map.get(col, 1)
    roll = df[col].rolling(window=window, min_periods=min(min_obs, window))
    mult = -1 if col == 'Rate_Momentum' else 1
    df[f'{col}_Z'] = ((df[col] - roll.mean()) / roll.std()) * mult

# Credit Health Z-Score
if 'HY_Spread' in df.columns:
    roll_hy = df['HY_Spread'].rolling(window=window, min_periods=30)
    df['Credit_Z'] = (((df['HY_Spread'] - roll_hy.mean()) / roll_hy.std()) * -1).ffill()

# Final Aggregate
z_cols = [f'{c}_Z' for c in liq_pillars if f'{c}_Z' in df.columns]
df['Aggregate_Liquidity'] = df[z_cols].mean(axis=1) if z_cols else 0

# --- 4. PLOTTING ---
monthly_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')
start_month, end_month = st.sidebar.select_slider(
    "Select Period", options=monthly_range, 
    value=(monthly_range[-120], monthly_range[-1]), format_func=lambda x: x.strftime('%Y')
)
plot_df = df.loc[start_month:end_month].copy()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

def apply_institutional_grid(ax, data):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    if 'Recessions' in data.columns:
        ax.fill_between(data.index, ax.get_ylim()[0], ax.get_ylim()[1], where=data['Recessions'] > 0, color='gray', alpha=0.15)
    ax.grid(True, which='major', axis='x', color='#4F4F4F', linestyle='-', alpha=0.6)
    ax.grid(True, which='major', axis='y', color='gray', linestyle=':', alpha=0.2)

# Panel 1: Market
ax1.plot(plot_df.index, plot_df['SP500'], color='#1f77b4', lw=2.5)
ax1.set_yscale('log')
ax1.set_title("S&P 500 Performance", loc='left', fontweight='bold')
apply_institutional_grid(ax1, plot_df)

# Panel 2: Liquidity
ax2.plot(plot_df.index, plot_df['Aggregate_Liquidity'], color='purple', lw=1.5)
ax2.axhline(0, color='black', lw=1)
ax2.fill_between(plot_df.index, 0, plot_df['Aggregate_Liquidity'], where=(plot_df['Aggregate_Liquidity']>=0), color='green', alpha=0.3)
ax2.fill_between(plot_df.index, 0, plot_df['Aggregate_Liquidity'], where=(plot_df['Aggregate_Liquidity']<0), color='red', alpha=0.3)
ax2.set_ylabel("Liquidity Z-Score")
apply_institutional_grid(ax2, plot_df)

# Panel 3: Credit
if 'Credit_Z' in plot_df.columns:
    ax3.plot(plot_df.index, plot_df['Credit_Z'], color='orange', lw=1.5)
    ax3.axhline(0, color='black', lw=1)
    ax3.fill_between(plot_df.index, 0, plot_df['Credit_Z'], where=(plot_df['Credit_Z']>=0), color='cyan', alpha=0.2)
    ax3.fill_between(plot_df.index, 0, plot_df['Credit_Z'], where=(plot_df['Credit_Z']<0), color='red', alpha=0.2)
    ax3.set_ylabel("Credit Health Z")
apply_institutional_grid(ax3, plot_df)

st.pyplot(fig)