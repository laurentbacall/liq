import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
import datetime

# --- 1. CONFIGURATION & UI ---
st.set_page_config(page_title="Macro Liquidity Dashboard", layout="wide")
st.title("🌊 Institutional Liquidity Monitor")
st.sidebar.header("Settings")

# Securely handle the API Key
api_key = st.sidebar.text_input("Enter FRED API Key", type="password")
lookback_years = st.sidebar.slider("Z-Score Lookback (Years)", 1, 5, 3)

if not api_key:
    st.warning("Please enter your FRED API key in the sidebar to fetch data.")
    st.stop()

fred = Fred(api_key=api_key)

# --- 2. DATA FETCHING (Cached for performance) ---
@st.cache_data(ttl=3600) # Updates every hour
def get_data():
    series_ids = {
        'WALCL': 'Fed Assets',
        'WTREGEN': 'TGA',
        'RRPONTSYD': 'Reverse Repo',
        'TB3MS': '3M Bill',
        'CPIAUCSL': 'CPI',
        'M2SL': 'M2 Supply',
        'SP500': 'S&P 500'
    }
    
    df_list = []
    for s_id, name in series_ids.items():
        s = fred.get_series(s_id)
        s.name = name
        df_list.append(s)
    
    df = pd.concat(df_list, axis=1).resample('D').ffill()
    return df

with st.spinner('Fetching Macro Data from FRED...'):
    df = get_data()

# --- 3. CALCULATIONS ---
# Net Liquidity
df['Net Liquidity'] = df['Fed Assets'] - (df['TGA'].fillna(0) + df['Reverse Repo'].fillna(0))

# Real 3M Rate
df['CPI_YoY'] = df['CPI'].pct_change(365) * 100
df['Real 3M Rate'] = df['3M Bill'] - df['CPI_YoY']

# M2 Real Growth
df['M2_YoY'] = df['M2 Supply'].pct_change(365) * 100
df['M2 Real Growth'] = df['M2_YoY'] - df['CPI_YoY']

# Z-Scores
window = 365 * lookback_years
cols_to_z = {'Net Liquidity': 1, 'Real 3M Rate': -1, 'M2 Real Growth': 1} # -1 for Real Rate (Lower is better)

for col, multiplier in cols_to_z.items():
    roll = df[col].rolling(window=window)
    df[f'{col}_Z'] = ((df[col] - roll.mean()) / roll.std()) * multiplier

df['Aggregate_Index'] = df[[f'{c}_Z' for c in cols_to_z.keys()]].mean(axis=1)

# --- 4. DASHBOARD LAYOUT ---
last_val = df['Aggregate_Index'].iloc[-1]
status = "BULLISH" if last_val > 1 else "BEARISH" if last_val < -1 else "NEUTRAL"
color = "green" if last_val > 1 else "red" if last_val < -1 else "white"

st.metric("Current Liquidity Regime", f"{last_val:.2f} Z", delta=status, delta_color="normal")

col1, col2 = st.columns([2, 1])

with col1:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # S&P 500 Plot
    ax1.plot(df.index, df['S&P 500'], color='gold', lw=2)
    ax1.set_title("S&P 500 vs Liquidity Regimes")
    ax1.fill_between(df.index, df['S&P 500'].min(), df['S&P 500'].max(), where=(df['Aggregate_Index'] > 1), color='green', alpha=0.15)
    ax1.fill_between(df.index, df['S&P 500'].min(), df['S&P 500'].max(), where=(df['Aggregate_Index'] < -1), color='red', alpha=0.15)
    
    # Index Plot
    ax2.plot(df.index, df['Aggregate_Index'], color='cyan')
    ax2.axhline(1, ls='--', color='green', alpha=0.5)
    ax2.axhline(-1, ls='--', color='red', alpha=0.5)
    ax2.set_ylabel("Z-Score")
    
    st.pyplot(fig)

with col2:
    st.subheader("Component Health")
    st.write("Individual contribution to liquidity:")
    for col in cols_to_z.keys():
        z_val = df[f'{col}_Z'].iloc[-1]
        st.write(f"**{col}:** {z_val:.2f}")