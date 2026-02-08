import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from fredapi import Fred
import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Macro Regime Monitor", layout="wide")
st.title("🛡️ Institutional Risk & Liquidity Monitor")

if "FRED_API_KEY" in st.secrets:
    api_key = st.secrets["FRED_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter FRED API Key", type="password")

if not api_key:
    st.info("Please enter your FRED API key in the sidebar to begin.")
    st.stop()

fred = Fred(api_key=api_key)

# --- 2. DATA FETCHING (Robust Hybrid) ---
@st.cache_data(ttl=3600)
def get_data():
    # A. Fetch FRED Macro + S&P 500 Daily
    series_ids = {
        'WALCL': 'Fed Assets', 'WTREGEN': 'TGA', 'RRPONTSYD': 'Reverse Repo',
        'TB3MS': '3M Bill', 'CPIAUCSL': 'CPI', 'M2SL': 'M2 Supply', 
        'BAMLH0A0HYM2': 'HY_Spread',
        'SP500': 'SP500_FRED',    # Daily from ~2011
        'WILL5000PR': 'WILL_Proxy' 
    }
    df_list = []
    for s_id, name in series_ids.items():
        try:
            s = fred.get_series(s_id)
            s.name = name
            df_list.append(s)
        except Exception: continue
    
    # B. Fetch Yahoo Monthly for long history
    sp_history = pd.Series(dtype=float)
    try:
        yf_df = yf.download("^GSPC", start="1970-01-01", interval="1mo", progress=False)
        if not yf_df.empty:
            # Handle possible MultiIndex or simple Series
            if 'Close' in yf_df.columns:
                sp_history = yf_df['Close']
                if isinstance(sp_history, pd.DataFrame):
                    sp_history = sp_history.iloc[:, 0]
    except Exception:
        pass

    df = pd.concat(df_list, axis=1).resample('D').ffill()

    # C. MERGE: Hybrid S&P 500
    # 1. Interpolate monthly Yahoo to daily to fill pre-2011 gaps
    yf_daily = sp_history.resample('D').interpolate(method='linear')
    
    # 2. Prefer FRED Daily (High resolution), fallback to Yahoo (Interpolated)
    # This automatically "stitches" the two series together
    df['SP500'] = df['SP500_FRED'].combine_first(yf_daily)
    
    # 3. Last resort fallback
    if 'SP500' not in df.columns or df['SP500'].dropna().empty:
        df['SP500'] = df['WILL_Proxy']
        
    return df

df = get_data()

# --- 3. CALCULATIONS (Modern Pandas Syntax) ---
# Ensure no NAs before pct_change
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
    format_func=lambda x: x.strftime('%b %Y')
)

plot_df = df.loc[start_month:end_month].copy()

# --- 5. THE CHARTS (With Universal Locator) ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, 
                                    gridspec_kw={'height_ratios': [2, 1, 1]})

def apply_dense_grid(ax):
    # MonthLocator(interval=3) is the universal version of Quarterly
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    ax.grid(True, which='major', axis='x', color='gray', linestyle='--', alpha=0.4)
    ax.grid(True, which='major', axis='y', color='gray', linestyle=':', alpha=0.2)
    
    # Clean up labels so we only see the Year once per 4 ticks
    for label in ax.xaxis.get_ticklabels()[::4]:
        label.set_visible(True)

# Panel 1: S&P 500
ax1.plot(plot_df.index, plot_df['SP500'], color='#1f77b4', lw=2.5)
ax1.set_yscale('log')
if not plot_df['SP500'].dropna().empty:
    ymin, ymax = plot_df['SP500'].min() * 0.95, plot_df['SP500'].max() * 1.05
    ax1.set_ylim(ymin, ymax)
ax1.set_title("Market Performance (Hybrid Log Scale)", loc='left', fontweight='bold')
apply_dense_grid(ax1)

# Panel 2: Liquidity Flow
ax2.plot(plot_df.index, plot_df['Aggregate_Liquidity'], color='purple', lw=1.5)
ax2.axhline(0, color='black', lw=1.2)
ax2.fill_between(plot_df.index, 0, plot_df['Aggregate_Liquidity'], where=(plot_df['Aggregate_Liquidity']>=0), color='green', alpha=0.3)
ax2.fill_between(plot_df.index, 0, plot_df['Aggregate_Liquidity'], where=(plot_df['Aggregate_Liquidity']<0), color='red', alpha=0.3)
ax2.set_ylabel("Liquidity Flow Index")
apply_dense_grid(ax2)

# Panel 3: Credit Health
ax3.plot(plot_df.index, plot_df['Credit_Z'], color='orange', lw=1.5)
ax3.axhline(0, color='black', lw=1.2)
ax3.fill_between(plot_df.index, 0, plot_df['Credit_Z'], where=(plot_df['Credit_Z']>=0), color='cyan', alpha=0.2)
ax3.fill_between(plot_df.index, 0, plot_df['Credit_Z'], where=(plot_df['Credit_Z']<0), color='red', alpha=0.2)
ax3.set_ylabel("Credit Health (Z)")
apply_dense_grid(ax3)

plt.tight_layout()
st.pyplot(fig)