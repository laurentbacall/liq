import streamlit as st
import pandas as pd
from fredapi import Fred
from streamlit_echarts import st_echarts
import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Macro Liquidity Flow", layout="wide")
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

# --- 3. FLOW CALCULATIONS (YoY Changes) ---
# Net Liquidity Flow (YoY % Change)
df['Net Liquidity'] = df['Fed Assets'] - (df['TGA'].fillna(0) + df['Reverse Repo'].fillna(0))
df['Liquidity_Flow'] = df['Net Liquidity'].pct_change(365) * 100

# Real Rate Momentum (YoY Point Change)
df['CPI_YoY'] = df['CPI'].pct_change(365) * 100
df['Real_Rate'] = df['3M Bill'] - df['CPI_YoY']
df['Rate_Momentum'] = df['Real_Rate'] - df['Real_Rate'].shift(365)

# M2 Real Growth (Already YoY)
df['M2_Real_Growth'] = (df['M2 Supply'].pct_change(365) * 100) - df['CPI_YoY']

# --- 4. Z-SCORES OF FLOWS ---
lookback = st.sidebar.slider("Z-Score Lookback (Years)", 1, 10, 3)
window = 365 * lookback

# Define Polarity: 
# Liquidity Flow Up = Bullish (1)
# Rate Momentum Up = Bearish (-1)
# M2 Growth Up = Bullish (1)
flows = {'Liquidity_Flow': 1, 'Rate_Momentum': -1, 'M2_Real_Growth': 1}

for col, mult in flows.items():
    roll = df[col].rolling(window=window, min_periods=30)
    df[f'{col}_Z'] = ((df[col] - roll.mean()) / roll.std()) * mult
    df[f'{col}_Z'] = df[f'{col}_Z'].ffill()

df['Aggregate_Index'] = df[[f'{c}_Z' for c in flows.keys()]].mean(axis=1)

# --- 5. DYNAMIC FILTERS ---
min_d, max_d = df.index.min().to_pydatetime(), df.index.max().to_pydatetime()
start_d, end_d = st.sidebar.slider("Analysis Period", min_d, max_d, (max_d - datetime.timedelta(days=1095), max_d))
plot_df = df.loc[start_d:end_d].dropna(subset=['SP500'])

# --- 6. INDIVIDUAL CORRELATIONS ---
st.sidebar.subheader("Individual Correlations")
for col in flows.keys():
    c_val = plot_df['SP500'].corr(plot_df[f'{col}_Z'])
    st.sidebar.write(f"**{col.replace('_', ' ')}:**")
    color = "green" if c_val > 0.4 else "red" if c_val < -0.2 else "gray"
    st.sidebar.markdown(f":{color}[{c_val:.2f} Correlation]")

# --- 7. INTERACTIVE ECHARTS ---
dates = plot_df.index.strftime('%Y-%m-%d').tolist()
sp500_data = plot_df['SP500'].round(2).tolist()
index_data = plot_df['Aggregate_Index'].round(2).tolist()

options = {
    "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
    "legend": {"data": ["S&P 500", "Aggregate Flow Index"]},
    "grid": [{"left": "5%", "right": "5%", "height": "50%"}, 
             {"left": "5%", "right": "5%", "top": "65%", "height": "25%"}],
    "xAxis": [{"type": "category", "data": dates, "gridIndex": 0},
              {"type": "category", "data": dates, "gridIndex": 1}],
    "yAxis": [
        {"type": "log", "name": "S&P 500", "gridIndex": 0, "min": "dataMin", "max": "dataMax"},
        {"type": "value", "name": "Z-Score", "gridIndex": 1}
    ],
    "series": [
        {"name": "S&P 500", "type": "line", "data": sp500_data, "xAxisIndex": 0, "yAxisIndex": 0, "symbol": "none"},
        {"name": "Aggregate Flow Index", "type": "line", "data": index_data, "xAxisIndex": 1, "yAxisIndex": 1, 
         "symbol": "none", "areaStyle": {"opacity": 0.2}}
    ]
}

st_echarts(options=options, height="600px")

# --- 8. COMPONENT BREAKDOWN ---
st.subheader("Component Flow Analysis")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Liquidity Flow (YoY)", f"{df['Liquidity_Flow'].dropna().iloc[-1]:.1f}%")
    st.caption("Rate of change in Fed Net Liquidity")
with c2:
    st.metric("Rate Momentum (YoY)", f"{df['Rate_Momentum'].dropna().iloc[-1]:.2f}%")
    st.caption("YoY Basis point change in Real 3M Rate")
with c3:
    st.metric("M2 Real Growth", f"{df['M2_Real_Growth'].dropna().iloc[-1]:.1f}%")
    st.caption("YoY Real M2 expansion")