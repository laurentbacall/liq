import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
import io

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
timescale = st.sidebar.radio("View Timescale", ["1Y", "3Y", "5Y", "Max"], index=2)

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

window = 365 * lookback_years
cols_to_z = {'Net Liquidity': 1, 'Real 3M Rate': -1, 'M2 Real Growth': 1}

for col, multiplier in cols_to_z.items():
    roll_mean = df[col].rolling(window=window, min_periods=30).mean()
    roll_std = df[col].rolling(window=window, min_periods=30).std()
    df[f'{col}_Z'] = ((df[col] - roll_mean) / roll_std) * multiplier
    df[f'{col}_Z'] = df[f'{col}_Z'].ffill() # Carry forward last known regime

z_cols = [f'{c}_Z' for c in cols_to_z.keys()]
df['Aggregate_Index'] = df[z_cols].mean(axis=1)

# Filter by selected timescale
end_date = df.index.max()
if timescale == "1Y": start_date = end_date - pd.Timedelta(days=365)
elif timescale == "3Y": start_date = end_date - pd.Timedelta(days=365*3)
elif timescale == "5Y": start_date = end_date - pd.Timedelta(days=365*5)
else: start_date = df.index.min()
plot_df = df.loc[start_date:end_date]

# --- 5. TOP METRICS (Robust NaN handling) ---
def get_last(series):
    return series.dropna().iloc[-1] if not series.dropna().empty else 0.0

m1, m2, m3, m4 = st.columns(4)
idx_val = get_last(df['Aggregate_Index'])
status = "EXPANDING" if idx_val > 1 else "CONTRACTING" if idx_val < -1 else "NEUTRAL"

m1.metric("Liquidity Index", f"{idx_val:.2f} Z", delta=status)
m2.metric("Real 3M Rate", f"{get_last(df['Real 3M Rate']):.2f}%")
m3.metric("M2 Real Growth", f"{get_last(df['M2 Real Growth']):.2f}%")
m4.metric("Net Fed Liquidity", f"${get_last(df['Net Liquidity'])/1e6:.2f}T")

st.divider()

# --- 6. CHARTS ---
col_chart, col_stats = st.columns([3, 1])

with col_chart:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Panel 1: S&P 500 (LOG SCALE)
    ax1.plot(plot_df.index, plot_df['SP500'], color='#1f77b4', lw=1.5)
    ax1.set_yscale('log') # This enables the Logarithmic scale
    ax1.set_title(f"S&P 500 Performance ({timescale} - Log Scale)")
    ax1.fill_between(plot_df.index, plot_df['SP500'].min(), plot_df['SP500'].max(), 
                     where=(plot_df['Aggregate_Index'] > 1), color='green', alpha=0.1)
    ax1.fill_between(plot_df.index, plot_df['SP500'].min(), plot_df['SP500'].max(), 
                     where=(plot_df['Aggregate_Index'] < -1), color='red', alpha=0.1)

    # Panel 2: Composite Index
    ax2.plot(plot_df.index, plot_df['Aggregate_Index'], color='gray', alpha=0.5)
    ax2.fill_between(plot_df.index, 0, plot_df['Aggregate_Index'], where=(plot_df['Aggregate_Index'] > 0), color='green', alpha=0.4)
    ax2.fill_between(plot_df.index, 0, plot_df['Aggregate_Index'], where=(plot_df['Aggregate_Index'] < 0), color='red', alpha=0.4)
    ax2.axhline(1, ls=':', color='green')
    ax2.axhline(-1, ls=':', color='red')
    
    st.pyplot(fig)

with col_stats:
    st.subheader("Component Health")
    for col in cols_to_z.keys():
        val = get_last(df[f'{col}_Z'])
        st.write(f"**{col}**")
        st.progress((max(min(val, 3), -3) + 3) / 6)
        st.caption(f"Current: {val:.2f} Z")