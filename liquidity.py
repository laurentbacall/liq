import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from fredapi import Fred
import datetime

# --- 1. CONFIGURATION & SECRETS ---
st.set_page_config(page_title="Macro Regime Monitor", layout="wide")
st.title("🛡️ Institutional Risk & Liquidity Monitor")

# Handle API Key from Streamlit Secrets or Sidebar fallback
if "FRED_API_KEY" in st.secrets:
    api_key = st.secrets["FRED_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter FRED API Key", type="password")

if not api_key:
    st.info("Please enter your FRED API key in the sidebar or add it to Streamlit Secrets to begin.")
    st.stop()

fred = Fred(api_key=api_key)

# --- 2. DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_data():
    series_ids = {
        'WALCL': 'Fed Assets', 'WTREGEN': 'TGA', 'RRPONTSYD': 'Reverse Repo',
        'TB3MS': '3M Bill', 'CPIAUCSL': 'CPI', 'M2SL': 'M2 Supply', 
        'BAMLH0A0HYM2': 'HY_Spread',
        'SP500': 'SP500_FRED',
        'WILL5000PR': 'WILL_Proxy' 
    }
    df_list = []
    for s_id, name in series_ids.items():
        try:
            s = fred.get_series(s_id)
            s.name = name
            df_list.append(s)
        except Exception: continue
    
    # Yahoo Monthly Fetch
    sp_history = pd.Series(dtype=float)
    try:
        yf_df = yf.download("^GSPC", start="1970-01-01", interval="1mo", progress=False)
        if not yf_df.empty:
            if 'Close' in yf_df.columns:
                # Ensure we handle the yfinance DataFrame vs Series structure
                sp_history = yf_df['Close']
                if isinstance(sp_history, pd.DataFrame):
                    sp_history = sp_history.iloc[:, 0]
    except Exception:
        pass

    df = pd.concat(df_list, axis=1).resample('D').ffill()

    # CRITICAL FIX: Only resample if we have a DatetimeIndex
    if not sp_history.empty and isinstance(sp_history.index, pd.DatetimeIndex):
        yf_daily = sp_history.resample('D').interpolate(method='linear')
        df['SP500'] = df['SP500_FRED'].combine_first(yf_daily)
    else:
        # If Yahoo fails, rely on FRED and then Proxy
        df['SP500'] = df['SP500_FRED']
    
    # Ultimate fallback to Wilshire 5000
    if 'SP500' not in df.columns or df['SP500'].dropna().empty:
        df['SP500'] = df['WILL_Proxy']
        
    return df

df = get_data()

# --- 3. CALCULATIONS ---
df_calc = df.ffill()
df['Net_Liquidity'] = df_calc['Fed Assets'] - (df_calc['TGA'].fillna(0) + df_calc['Reverse Repo'].fillna(0))
df['Liquidity_Flow'] = df['Net_Liquidity'].pct_change(365) * 100
df['CPI_YoY'] = df_calc['CPI'].pct_change(365) * 100
df['M2_Real_Growth'] = (df_calc['M2 Supply'].pct_change(365) * 100) - df['CPI_YoY']
df['Real_Rate'] = df_calc['3M Bill'] - df['CPI_YoY']
df['Rate_Momentum'] = df['Real_Rate'] - df['Real_Rate'].shift(365)

lookback = st.sidebar.slider("Z-Score Lookback (Years)", 1, 10, 3)
window = 365 * lookback

# Z-Scores
liq_pillars = {'Liquidity_Flow': 1, 'M2_Real_Growth': 1, 'Rate_Momentum': -1}
for col, mult in liq_pillars.items():
    roll = df[col].rolling(window=window, min_periods=30)
    df[f'{col}_Z'] = ((df[col] - roll.mean()) / roll.std()) * mult

df['Aggregate_Liquidity'] = df[[f'{c}_Z' for c in liq_pillars.keys()]].mean(axis=1)
roll_hy = df['HY_Spread'].rolling(window=window, min_periods=30)
df['Credit_Z'] = ((df['HY_Spread'] - roll_hy.mean()) / roll_hy.std()) * -1

# --- 4. MONTHLY SELECTOR ---
monthly_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')
start_month, end_month = st.sidebar.select_slider(
    "Select Analysis Period",
    options=monthly_range,
    value=(monthly_range[-120], monthly_range[-1]),
    format_func=lambda x: x.strftime('%Y')
)
plot_df = df.loc[start_month:end_month].copy()

# --- 5. THE CHARTS (Multi-Axis Labels & Dense Grid) ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, 
                                    gridspec_kw={'height_ratios': [2, 1, 1]})

def apply_institutional_grid(ax):
    # Setup Locators
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    
    # FORCE LABELS ON ALL AXES
    ax.tick_params(labelbottom=True, which='major', labelsize=10)
    
    # Grid Styling
    ax.grid(True, which='major', axis='x', color='#4F4F4F', linestyle='-', alpha=0.6, linewidth=1.2)
    ax.grid(True, which='minor', axis='x', color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.grid(True, which='major', axis='y', color='gray', linestyle=':', alpha=0.2)

# Panel 1: S&P 500
ax1.plot(plot_df.index, plot_df['SP500'], color='#1f77b4', lw=2.5)
ax1.set_yscale('log')
if not plot_df['SP500'].dropna().empty:
    ymin, ymax = plot_df['SP500'].min() * 0.95, plot_df['SP500'].max() * 1.05
    ax1.set_ylim(ymin, ymax)
ax1.set_title("Market Performance (Log Scale)", loc='left', fontweight='bold', fontsize=12)
apply_institutional_grid(ax1)

# Panel 2: Liquidity Flow
ax2.plot(plot_df.index, plot_df['Aggregate_Liquidity'], color='purple', lw=1.5)
ax2.axhline(0, color='black', lw=1.2)
ax2.fill_between(plot_df.index, 0, plot_df['Aggregate_Liquidity'], where=(plot_df['Aggregate_Liquidity']>=0), color='green', alpha=0.3)
ax2.fill_between(plot_df.index, 0, plot_df['Aggregate_Liquidity'], where=(plot_df['Aggregate_Liquidity']<0), color='red', alpha=0.3)
ax2.set_ylabel("Liquidity Flow Index")
apply_institutional_grid(ax2)

# Panel 3: Credit Health (HY Spreads)
ax3.plot(plot_df.index, plot_df['Credit_Z'], color='orange', lw=1.5)
ax3.axhline(0, color='black', lw=1.2)
ax3.fill_between(plot_df.index, 0, plot_df['Credit_Z'], where=(plot_df['Credit_Z']>=0), color='cyan', alpha=0.2)
ax3.fill_between(plot_df.index, 0, plot_df['Credit_Z'], where=(plot_df['Credit_Z']<0), color='red', alpha=0.2)
ax3.set_ylabel("Credit Health (Z)")
apply_institutional_grid(ax3)

# Add space between charts to accommodate labels
plt.subplots_adjust(hspace=0.4)
st.pyplot(fig)