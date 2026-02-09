import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from fredapi import Fred

# --- 1. CONFIGURATION & SECRETS ---
st.set_page_config(page_title="Macro Regime Monitor", layout="wide")
st.title("🛡️ Institutional Risk & Liquidity Monitor")

# API Key handling: Prioritizes Streamlit Secrets for cloud deployment
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
        'WILL5000PR': 'WILL_Proxy', 'USREC': 'Recessions'
    }
    df_list = []
    for s_id, name in series_ids.items():
        try:
            s = fred.get_series(s_id)
            s.name = name
            df_list.append(s)
        except Exception: continue
    
    # Yahoo Monthly Fallback (Critical for history before 2011)
    sp_history = pd.Series(dtype=float)
    try:
        yf_df = yf.download("^GSPC", start="1970-01-01", interval="1mo", progress=False)
        if not yf_df.empty and 'Close' in yf_df.columns:
            sp_history = yf_df['Close']
            if isinstance(sp_history, pd.DataFrame): sp_history = sp_history.iloc[:, 0]
    except Exception: pass

    # Resample all FRED series to Daily frequency
    df = pd.concat(df_list, axis=1).resample('D').ffill()
    
    # --- S&P 500 STITCHING LOGIC ---
    # Layer 1: Native FRED S&P 500 (Starts ~2011)
    sp_final = df['SP500_FRED']
    
    # Layer 2: Yahoo Monthly (Interpolated to fill the 1970-2011 gap)
    if not sp_history.empty and isinstance(sp_history.index, pd.DatetimeIndex):
        yf_daily = sp_history.resample('D').interpolate(method='linear')
        sp_final = sp_final.combine_first(yf_daily)
    
    # Layer 3: Wilshire 5000 Proxy (Fills any remaining holes in Yahoo/FRED)
    df['SP500'] = sp_final.combine_first(df['WILL_Proxy'])
        
    return df

df = get_data()

# --- 3. CALCULATIONS ---
# Pre-fill raw data so rolling windows don't hit NaN "holes"
df = df.ffill() 

# Absolute Calculations
df['Net_Liquidity'] = df['Fed Assets'] - (df['TGA'].fillna(0) + df['Reverse Repo'].fillna(0))
df['Liquidity_Flow'] = df['Net_Liquidity'].pct_change(365) * 100
df['CPI_YoY'] = df['CPI'].pct_change(365) * 100
df['M2_Real_Growth'] = (df['M2 Supply'].pct_change(365) * 100) - df['CPI_YoY']
df['Real_Rate'] = df['3M Bill'] - df['CPI_YoY']
df['Rate_Momentum'] = df['Real_Rate'] - df['Real_Rate'].shift(365)

# Z-Score Parameters
lookback = st.sidebar.slider("Z-Score Lookback (Years)", 1, 10, 3)
window = 365 * lookback

# Frequency Mapping for "30 Unique Data Points" Logic
# Monthly series need ~900 daily rows to represent 30 data points
freq_map = {'Liquidity_Flow': 7, 'M2_Real_Growth': 30, 'Rate_Momentum': 30, 'Credit_Z': 1}

# Calculate Pillars
liq_pillars = ['Liquidity_Flow', 'M2_Real_Growth', 'Rate_Momentum']
for col in liq_pillars:
    min_obs = 30 * freq_map.get(col, 1)
    roll = df[col].rolling(window=window, min_periods=min(min_obs, window))
    mult = -1 if col == 'Rate_Momentum' else 1
    df[f'{col}_Z'] = ((df[col] - roll.mean()) / roll.std()) * mult

# Credit Z-Score (Specific fill for HY Spread weekend holes)
min_credit_obs = 30 * freq_map['Credit_Z']
roll_hy = df['HY_Spread'].rolling(window=window, min_periods=min_credit_obs)
df['Credit_Z'] = (((df['HY_Spread'] - roll_hy.mean()) / roll_hy.std()) * -1).ffill()

df['Aggregate_Liquidity'] = df[[f'{c}_Z' for c in liq_pillars]].mean(axis=1)

# --- 4. SELECTOR & PLOTTING ---
monthly_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')
start_month, end_month = st.sidebar.select_slider(
    "Select Analysis Period",
    options=monthly_range,
    value=(monthly_range[-120], monthly_range[-1]),
    format_func=lambda x: x.strftime('%Y')
)
plot_df = df.loc[start_month:end_month].copy()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, 
                                    gridspec_kw={'height_ratios': [2, 1, 1]})

def apply_institutional_grid(ax, data):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.tick_params(labelbottom=True, which='major', labelsize=10)
    
    # Recession Shading (Gray Bands)
    if 'Recessions' in data.columns:
        ax.fill_between(data.index, ax.get_ylim()[0], ax.get_ylim()[1], 
                        where=data['Recessions'] > 0, color='gray', alpha=0.15)
    
    # High-Density Grid Lines
    ax.grid(True, which='major', axis='x', color='#4F4F4F', linestyle='-', alpha=0.6, linewidth=1.2)
    ax.grid(True, which='minor', axis='x', color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.grid(True, which='major', axis='y', color='gray', linestyle=':', alpha=0.2)

# Panel 1: S&P 500
ax1.plot(plot_df.index, plot_df['SP500'], color='#1f77b4', lw=2.5)
ax1.set_yscale('log')
ax1.set_title("S&P 500 Performance (Log Scale)", loc='left', fontweight='bold', fontsize=12)
apply_institutional_grid(ax1, plot_df)

# Panel 2: Liquidity Flow
ax2.plot(plot_df.index, plot_df['Aggregate_Liquidity'], color='purple', lw=1.5)
ax2.axhline(0, color='black', lw=1.2)
ax2.fill_between(plot_df.index, 0, plot_df['Aggregate_Liquidity'], 
                 where=(plot_df['Aggregate_Liquidity']>=0), color='green', alpha=0.3)
ax2.fill_between(plot_df.index, 0, plot_df['Aggregate_Liquidity'], 
                 where=(plot_df['Aggregate_Liquidity']<0), color='red', alpha=0.3)
ax2.set_ylabel("Liquidity Index (Z)")
apply_institutional_grid(ax2, plot_df)

# Panel 3: Credit Health
ax3.plot(plot_df.index, plot_df['Credit_Z'], color='orange', lw=1.5)
ax3.axhline(0, color='black', lw=1.2)
ax3.fill_between(plot_df.index, 0, plot_df['Credit_Z'], 
                 where=(plot_df['Credit_Z']>=0), color='cyan', alpha=0.2)
ax3.fill_between(plot_df.index, 0, plot_df['Credit_Z'], 
                 where=(plot_df['Credit_Z']<0), color='red', alpha=0.2)
ax3.set_ylabel("Credit Health (Z)")
apply_institutional_grid(ax3, plot_df)

plt.subplots_adjust(hspace=0.4)
st.pyplot(fig)