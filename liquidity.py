import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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

# --- 2. DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_data():
    series_ids = {
        'WALCL': 'Fed Assets', 'WTREGEN': 'TGA', 'RRPONTSYD': 'Reverse Repo',
        'TB3MS': '3M Bill', 'CPIAUCSL': 'CPI', 'M2SL': 'M2 Supply', 
        'BAMLH0A0HYM2': 'HY_Spread',
        'SP500': 'SP500_FRED' # Backup S&P 500
    }
    df_list = []
    for s_id, name in series_ids.items():
        try:
            s = fred.get_series(s_id)
            s.name = name
            df_list.append(s)
        except Exception:
            continue
    
    # Try Yahoo Finance with Fallback
    try:
        # Fetching with auto_adjust to reduce data overhead
        yf_data = yf.download("^GSPC", start="1970-01-01", progress=False, auto_adjust=True)
        if not yf_data.empty:
            sp500 = yf_data['Close']
            sp500.name = "SP500"
            df_list.append(sp500)
        else:
            raise ValueError("Yahoo data empty")
    except Exception as e:
        st.warning("Yahoo Finance Rate Limit hit. Using FRED data (starts 2016).")
        # Fallback logic: Use FRED's SP500 column
        df_temp = pd.concat(df_list, axis=1)
        if 'SP500_FRED' in df_temp.columns:
            df_temp['SP500'] = df_temp['SP500_FRED']
            return df_temp.resample('D').ffill()

    df = pd.concat(df_list, axis=1).resample('D').ffill()
    # Handle multi-index columns if YFinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    return df

df = get_data()

# --- 3. CALCULATIONS (Fixed FutureWarnings) ---
df['Net_Liquidity'] = df['Fed Assets'] - (df['TGA'].fillna(0) + df['Reverse Repo'].fillna(0))

# Clean NAs before pct_change to avoid FutureWarnings
df['Liquidity_Flow'] = df['Net_Liquidity'].ffill().pct_change(365) * 100
df['CPI_YoY'] = df['CPI'].ffill().pct_change(365) * 100
df['M2_Real_Growth'] = (df['M2 Supply'].ffill().pct_change(365) * 100) - df['CPI_YoY']
df['Real_Rate'] = df['3M Bill'] - df['CPI_YoY']
df['Rate_Momentum'] = df['Real_Rate'] - df['Real_Rate'].shift(365)

lookback = st.sidebar.slider("Z-Score Lookback (Years)", 1, 10, 3)
window = 365 * lookback

# Z-Scores
liq_pillars = {'Liquidity_Flow': 1, 'M2_Real_Growth': 1, 'Rate_Momentum': -1}
for col, mult in liq_pillars.items():
    roll = df[col].rolling(window=window, min_periods=30)
    df[f'{col}_Z'] = ((df[col] - roll.mean()) / roll.std()) * mult
    df[f'{col}_Z'] = df[f'{col}_Z'].ffill()

df['Aggregate_Liquidity'] = df[[f'{c}_Z' for c in liq_pillars.keys()]].mean(axis=1)

roll_hy = df['HY_Spread'].rolling(window=window, min_periods=30)
df['Credit_Z'] = ((df['HY_Spread'] - roll_hy.mean()) / roll_hy.std()) * -1
df['Credit_Z'] = df['Credit_Z'].ffill()

# --- 4. MONTHLY SLIDER ---
st.sidebar.subheader("Timeline Settings")
monthly_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')

start_month, end_month = st.sidebar.select_slider(
    "Select Analysis Period",
    options=monthly_range,
    value=(monthly_range[-60], monthly_range[-1]),
    format_func=lambda x: x.strftime('%b %Y')
)

plot_df = df.loc[start_month:end_month].copy()

# --- 5. CHARTS ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, 
                                    gridspec_kw={'height_ratios': [2, 1, 1]})

def apply_sync_grid(ax):
    ax.grid(True, which='major', axis='x', color='gray', linestyle='--', alpha=0.4)
    ax.grid(True, which='major', axis='y', color='gray', linestyle=':', alpha=0.2)

# Panel 1: S&P 500
if 'SP500' in plot_df.columns:
    ax1.plot(plot_df.index, plot_df['SP500'], color='#1f77b4', lw=2.5)
    ax1.set_yscale('log')
    if not plot_df['SP500'].dropna().empty:
        ymin, ymax = plot_df['SP500'].min() * 0.95, plot_df['SP500'].max() * 1.05
        ax1.set_ylim(ymin, ymax)
    ax1.set_title("S&P 500 Index (Log Scale)", loc='left', fontweight='bold')
    apply_sync_grid(ax1)

# Panel 2: Liquidity
ax2.plot(plot_df.index, plot_df['Aggregate_Liquidity'], color='purple', lw=1.5)
ax2.axhline(0, color='black', lw=1.2)
ax2.fill_between(plot_df.index, 0, plot_df['Aggregate_Liquidity'], where=(plot_df['Aggregate_Liquidity']>=0), color='green', alpha=0.3)
ax2.fill_between(plot_df.index, 0, plot_df['Aggregate_Liquidity'], where=(plot_df['Aggregate_Liquidity']<0), color='red', alpha=0.3)
ax2.set_ylabel("Liquidity Flow Index")
apply_sync_grid(ax2)

# Panel 3: Credit
ax3.plot(plot_df.index, plot_df['Credit_Z'], color='orange', lw=1.5)
ax3.axhline(0, color='black', lw=1.2)
ax3.fill_between(plot_df.index, 0, plot_df['Credit_Z'], where=(plot_df['Credit_Z']>=0), color='cyan', alpha=0.2)
ax3.fill_between(plot_df.index, 0, plot_df['Credit_Z'], where=(plot_df['Credit_Z']<0), color='red', alpha=0.2)
ax3.set_ylabel("Credit Health (Z)")
apply_sync_grid(ax3)

plt.tight_layout()
st.pyplot(fig)