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
    api_key = st.sidebar.text_input("Enter FRED API Key", type="password")

if not api_key:
    st.info("Please enter your FRED API key to begin.")
    st.stop()

fred = Fred(api_key=api_key)

# --- 2. DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_data():
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
    
    df = pd.concat(df_list, axis=1).resample('D').ffill()
    
    # Yahoo S&P 500 History (Back to 1950)
    try:
        yf_df = yf.download("^GSPC", start="1950-01-01", interval="1mo", progress=False)
        if not yf_df.empty:
            if isinstance(yf_df.columns, pd.MultiIndex):
                sp_history = yf_df['Close'].iloc[:, 0]
            else:
                sp_history = yf_df['Close']
            
            sp_history.index = sp_history.index.tz_localize(None)
            yf_daily = sp_history.resample('D').interpolate(method='linear')
            
            if 'SP500_FRED' in df.columns:
                df['SP500'] = df['SP500_FRED'].combine_first(yf_daily)
            else:
                df['SP500'] = yf_daily
    except Exception:
        df['SP500'] = df.get('SP500_FRED', pd.Series(dtype=float))
        
    # Truncate all data to 1950 onwards
    df = df[df.index >= '1950-01-01']
    return df

df = get_data()

# --- 3. CALCULATIONS ---
df = df.ffill() 

if 'Fed Assets' in df.columns:
    tga = df['TGA'] if 'TGA' in df.columns else 0
    rrp = df['Reverse Repo'] if 'Reverse Repo' in df.columns else 0
    df['Net_Liquidity'] = df['Fed Assets'] - (tga.fillna(0) + rrp.fillna(0))
    df['Liquidity_Flow'] = df['Net_Liquidity'].pct_change(365) * 100

if 'CPI' in df.columns:
    df['CPI_YoY'] = df['CPI'].pct_change(365) * 100
    if 'M2 Supply' in df.columns:
        df['M2_Real_Growth'] = (df['M2 Supply'].pct_change(365) * 100) - df['CPI_YoY']
    if '3M Bill' in df.columns:
        df['Real_Rate'] = df['3M Bill'] - df['CPI_YoY']
        df['Rate_Momentum'] = df['Real_Rate'] - df['Real_Rate'].shift(365)

lookback = st.sidebar.slider("Z-Score Lookback (Years)", 1, 10, 3)
window = 365 * lookback
freq_map = {'Liquidity_Flow': 7, 'M2_Real_Growth': 30, 'Rate_Momentum': 30}

liq_pillars = [c for c in ['Liquidity_Flow', 'M2_Real_Growth', 'Rate_Momentum'] if c in df.columns]
for col in liq_pillars:
    min_obs = 30 * freq_map.get(col, 1)
    roll = df[col].rolling(window=window, min_periods=min(min_obs, window))
    mult = -1 if col == 'Rate_Momentum' else 1
    df[f'{col}_Z'] = ((df[col] - roll.mean()) / roll.std()) * mult

if 'HY_Spread' in df.columns:
    roll_hy = df['HY_Spread'].rolling(window=window, min_periods=30)
    df['Credit_Z'] = (((df['HY_Spread'] - roll_hy.mean()) / roll_hy.std()) * -1).ffill()

z_cols = [f'{c}_Z' for c in liq_pillars if f'{c}_Z' in df.columns]
df['Aggregate_Liquidity'] = df[z_cols].mean(axis=1) if z_cols else 0

# --- 4. PLOTTING ---
# Constrain the slider to start from 1950
monthly_range = pd.date_range(start='1950-01-01', end=df.index.max(), freq='MS')
start_month, end_month = st.sidebar.select_slider(
    "Select Analysis Period", options=monthly_range, 
    value=(monthly_range[-240], monthly_range[-1]), # Default to last 20 years
    format_func=lambda x: x.strftime('%Y')
)
plot_df = df.loc[start_month:end_month].copy()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

def apply_institutional_grid(ax, data):
    # This restores the labels and year ticks
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', labelbottom=True, rotation=0, labelsize=10)
    
    if 'Recessions' in data.columns:
        ax.fill_between(data.index, ax.get_ylim()[0], ax.get_ylim()[1], where=data['Recessions'] > 0, color='gray', alpha=0.15)
    ax.grid(True, which='major', axis='both', color='#4F4F4F', linestyle='-', alpha=0.3)

# Panel 1: S&P 500
ax1.plot(plot_df.index, plot_df['SP500'], color='#1f77b4', lw=2)
ax1.set_yscale('log')
ax1.set_title("S&P 500 Performance", loc='left', fontweight='bold')
apply_institutional_grid(ax1, plot_df)

# Panel 2: Liquidity
ax2.plot(plot_df.index, plot_df['Aggregate_Liquidity'], color='purple', lw=1.5)
ax2.axhline(0, color='black', lw=1)
ax2.fill_between(plot_df.index, 0, plot_df['Aggregate_Liquidity'], where=(plot_df['Aggregate_Liquidity']>=0), color='green', alpha=0.3)
ax2.fill_between(plot_df.index, 0, plot_df['Aggregate_Liquidity'], where=(plot_df['Aggregate_Liquidity']<0), color='red', alpha=0.3)
ax2.set_ylabel("Liquidity Z")
apply_institutional_grid(ax2, plot_df)

# Panel 3: Credit
if 'Credit_Z' in plot_df.columns:
    ax3.plot(plot_df.index, plot_df['Credit_Z'], color='orange', lw=1.5)
    ax3.axhline(0, color='black', lw=1)
    ax3.fill_between(plot_df.index, 0, plot_df['Credit_Z'], where=(plot_df['Credit_Z']>=0), color='cyan', alpha=0.2)
    ax3.fill_between(plot_df.index, 0, plot_df['Credit_Z'], where=(plot_df['Credit_Z']<0), color='red', alpha=0.2)
    ax3.set_ylabel("Credit Z")
apply_institutional_grid(ax3, plot_df)

plt.tight_layout()
st.pyplot(fig)