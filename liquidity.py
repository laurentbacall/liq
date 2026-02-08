import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Macro Flow Monitor", layout="wide")
st.title("🌊 Liquidity Flow & Correlation Monitor")

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
        'TB3MS': '3M Bill', 'CPIAUCSL': 'CPI', 'M2SL': 'M2 Supply', 'SP500': 'SP500'
    }
    df_list = []
    for s_id, name in series_ids.items():
        s = fred.get_series(s_id)
        s.name = name
        df_list.append(s)
    df = pd.concat(df_list, axis=1).resample('D').ffill()
    return df

df = get_data()

# --- 3. FLOW CALCULATIONS ---
# Net Liquidity Flow (YoY % Change)
df['Net Liquidity'] = df['Fed Assets'] - (df['TGA'].fillna(0) + df['Reverse Repo'].fillna(0))
df['Liquidity_Flow'] = df['Net Liquidity'].pct_change(365) * 100

# Real Rate Momentum (YoY Point Change)
df['CPI_YoY'] = df['CPI'].pct_change(365) * 100
df['Real_Rate'] = df['3M Bill'] - df['CPI_YoY']
df['Rate_Momentum'] = df['Real_Rate'] - df['Real_Rate'].shift(365)

# M2 Real Growth (YoY Real)
df['M2_Real_Growth'] = (df['M2 Supply'].pct_change(365) * 100) - df['CPI_YoY']

# --- 4. Z-SCORES ---
lookback = st.sidebar.slider("Z-Score Lookback (Years)", 1, 10, 3)
window = 365 * lookback
flows = {'Liquidity_Flow': 1, 'Rate_Momentum': -1, 'M2_Real_Growth': 1}

for col, mult in flows.items():
    roll = df[col].rolling(window=window, min_periods=30)
    df[f'{col}_Z'] = ((df[col] - roll.mean()) / roll.std()) * mult
    df[f'{col}_Z'] = df[f'{col}_Z'].ffill()

df['Aggregate_Index'] = df[[f'{c}_Z' for c in flows.keys()]].mean(axis=1)

# --- 5. DYNAMIC FILTERS ---
min_d, max_d = df.index.min().to_pydatetime(), df.index.max().to_pydatetime()
start_d, end_d = st.sidebar.slider("Analysis Period", min_d, max_d, (max_d - datetime.timedelta(days=1095), max_d))
plot_df = df.loc[start_d:end_d].copy()

# --- 6. SIDEBAR CORRELATIONS ---
st.sidebar.subheader("Individual Flow Correlations")
for col in flows.keys():
    # Correlation of the Z-score version vs SP500 price
    valid = plot_df[['SP500', f'{col}_Z']].dropna()
    if not valid.empty:
        c_val = valid['SP500'].corr(valid[f'{col}_Z'])
        color = "green" if c_val > 0.4 else "red" if c_val < -0.2 else "gray"
        st.sidebar.markdown(f"**{col.replace('_', ' ')}**")
        st.sidebar.markdown(f":{color}[Correlation: {c_val:.2f}]")

# --- 7. MATPLOTLIB ADAPTIVE CHARTS ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                               gridspec_kw={'height_ratios': [2, 1]})

# Top Chart: S&P 500 (Adaptive Log Scale)
ax1.plot(plot_df.index, plot_df['SP500'], color='#1f77b4', lw=2, label="S&P 500")
ax1.set_yscale('log')

# Dynamically set Y-limits based on the visible data range
if not plot_df['SP500'].dropna().empty:
    ymin, ymax = plot_df['SP500'].min() * 0.95, plot_df['SP500'].max() * 1.05
    ax1.set_ylim(ymin, ymax)
    # Background Shading based on Aggregate Index
    ax1.fill_between(plot_df.index, ymin, ymax, where=(plot_df['Aggregate_Index'] > 1), color='green', alpha=0.1)
    ax1.fill_between(plot_df.index, ymin, ymax, where=(plot_df['Aggregate_Index'] < -1), color='red', alpha=0.1)

ax1.set_title(f"Market Performance vs Liquidity Flows ({start_d.year}-{end_d.year})")
ax1.grid(True, which="both", ls="-", alpha=0.2)
ax1.legend()

# Bottom Chart: Aggregate Flow Index
ax2.plot(plot_df.index, plot_df['Aggregate_Index'], color='black', lw=1.5)
ax2.fill_between(plot_df.index, 0, plot_df['Aggregate_Index'], 
                 where=(plot_df['Aggregate_Index'] >= 0), color='green', alpha=0.3)
ax2.fill_between(plot_df.index, 0, plot_df['Aggregate_Index'], 
                 where=(plot_df['Aggregate_Index'] < 0), color='red', alpha=0.3)
ax2.axhline(0, color='black', lw=1)
ax2.set_ylabel("Composite Z-Score")

plt.tight_layout()
st.pyplot(fig)

# --- 8. TOP METRICS ---
def get_last(series):
    return series.dropna().iloc[-1] if not series.dropna().empty else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Liquidity Flow", f"{get_last(df['Liquidity_Flow']):.1f}% YoY")
c2.metric("Rate Momentum", f"{get_last(df['Rate_Momentum']):.2f}% YoY")
c3.metric("M2 Real Growth", f"{get_last(df['M2_Real_Growth']):.1f}% YoY")