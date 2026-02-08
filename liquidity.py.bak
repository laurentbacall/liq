import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
import datetime

# --- 1. CONFIGURATION & UI ---
st.set_page_config(page_title="Macro Liquidity Dashboard", layout="wide")
st.title("🌊 Institutional Liquidity Monitor")

# --- 2. AUTHENTICATION ---
if "FRED_API_KEY" in st.secrets:
    api_key = st.secrets["FRED_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter FRED API Key", type="password")

if not api_key:
    st.info("Please enter your FRED API key to begin.")
    st.stop()

fred = Fred(api_key=api_key)

# Sidebar Controls
st.sidebar.header("Dashboard Settings")
lookback_years = st.sidebar.slider("Z-Score Lookback (Years)", 1, 10, 3)

# --- 3. DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_data():
    series_ids = {
        'WALCL': 'Fed Assets', 'WTREGEN': 'TGA', 'RRPONTSYD': 'Reverse Repo',
        'TB3MS': '3M Bill', 'CPIAUCSL': 'CPI', 'M2SL': 'M2 Supply', 'SP500': 'SP500'
    }
    df_list = []
    for s_id, name in series_ids.items():
        try:
            s = fred.get_series(s_id)
            s.name = name
            df_list.append(s)
        except Exception as e:
            st.error(f"Error fetching {s_id}: {e}")
    
    df = pd.concat(df_list, axis=1).resample('D').ffill()
    return df

df = get_data()

# --- 4. CALCULATIONS ---
df['Net Liquidity'] = df['Fed Assets'] - (df['TGA'].fillna(0) + df['Reverse Repo'].fillna(0))
df['CPI_YoY'] = df['CPI'].pct_change(365) * 100
df['Real 3M Rate'] = df['3M Bill'] - df['CPI_YoY']
df['M2_YoY'] = df['M2 Supply'].pct_change(365) * 100
df['M2 Real Growth'] = df['M2_YoY'] - df['CPI_YoY']

# Z-Scores with ffill to prevent trailing NaNs
window = 365 * lookback_years
cols_to_z = {'Net Liquidity': 1, 'Real 3M Rate': -1, 'M2 Real Growth': 1}
for col, mult in cols_to_z.items():
    roll = df[col].rolling(window=window, min_periods=30)
    df[f'{col}_Z'] = ((df[col] - roll.mean()) / roll.std()) * mult
    df[f'{col}_Z'] = df[f'{col}_Z'].ffill()

z_cols = [f'{c}_Z' for c in cols_to_z.keys()]
df['Aggregate_Index'] = df[z_cols].mean(axis=1)

# --- 5. DYNAMIC TIME SPLICING ---
st.sidebar.subheader("Timescale Customization")
min_date = df.index.min().to_pydatetime()
max_date = df.index.max().to_pydatetime()

# Date range slider for total control
start_date, end_date = st.sidebar.slider(
    "Select Analysis Period",
    min_value=min_date,
    max_value=max_date,
    value=(max_date - datetime.timedelta(days=365*3), max_date)
)

# Create the filtered dataframe for plotting
plot_df = df.loc[start_date:end_date]

# --- 6. TOP METRICS (Last Valid Value Search) ---
def get_last_valid(series):
    return series.dropna().iloc[-1] if not series.dropna().empty else 0.0

m1, m2, m3, m4 = st.columns(4)
idx_val = get_last_valid(df['Aggregate_Index'])
status = "EXPANDING" if idx_val > 1 else "CONTRACTING" if idx_val < -1 else "NEUTRAL"

m1.metric("Liquidity Index", f"{idx_val:.2f} Z", delta=status)
m2.metric("Real 3M Rate", f"{get_last_valid(df['Real 3M Rate']):.2f}%")
m3.metric("M2 Real Growth", f"{get_last_valid(df['M2 Real Growth']):.2f}%")
m4.metric("Net Fed Liquidity", f"${get_last_valid(df['Net Liquidity'])/1e6:.2f}T")

st.divider()

# --- 7. ADAPTIVE LOG CHARTS ---
col_chart, col_stats = st.columns([3, 1])

with col_chart:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # S&P 500 (Log Scale + Adaptive Limits)
    ax1.plot(plot_df.index, plot_df['SP500'], color='#1f77b4', lw=2)
    ax1.set_yscale('log')
    
    # Adapt Y-axis limits to the actual range in the selected period (plus a 5% buffer)
    if not plot_df['SP500'].dropna().empty:
        y_min = plot_df['SP500'].min() * 0.95
        y_max = plot_df['SP500'].max() * 1.05
        ax1.set_ylim(y_min, y_max)
    
    ax1.set_title(f"S&P 500 Performance (Log Scale: {start_date.year} - {end_date.year})")
    ax1.fill_between(plot_df.index, y_min, y_max, where=(plot_df['Aggregate_Index'] > 1), color='green', alpha=0.1)
    ax1.fill_between(plot_df.index, y_min, y_max, where=(plot_df['Aggregate_Index'] < -1), color='red', alpha=0.1)

    # Liquidity Index Panel
    ax2.fill_between(plot_df.index, 0, plot_df['Aggregate_Index'], where=(plot_df['Aggregate_Index'] > 0), color='green', alpha=0.4)
    ax2.fill_between(plot_df.index, 0, plot_df['Aggregate_Index'], where=(plot_df['Aggregate_Index'] < 0), color='red', alpha=0.4)
    ax2.axhline(1, ls=':', color='green')
    ax2.axhline(-1, ls=':', color='red')
    ax2.set_ylabel("Z-Score")
    
    st.pyplot(fig)

with col_stats:
    st.subheader("Correlation Analysis")
    # Show correlation between SP500 and the Index for the selected period
    if not plot_df[['SP500', 'Aggregate_Index']].dropna().empty:
        corr = plot_df['SP500'].corr(plot_df['Aggregate_Index'])
        st.write(f"**Index Correlation:**")
        st.code(f"{corr:.2f}")
        st.caption("Correlation between S&P 500 and Liquidity in the current window.")