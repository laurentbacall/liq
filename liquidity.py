import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
    st.info("Please enter your FRED API key to begin.")
    st.stop()

fred = Fred(api_key=api_key)

# --- 2. DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_data():
    series_ids = {
        'WALCL': 'Fed Assets', 'WTREGEN': 'TGA', 'RRPONTSYD': 'Reverse Repo',
        'TB3MS': '3M Bill', 'CPIAUCSL': 'CPI', 'M2SL': 'M2 Supply', 
        'SP500': 'SP500', 'BAMLH0A0HYM2': 'HY_Spread' 
    }
    df_list = []
    for s_id, name in series_ids.items():
        s = fred.get_series(s_id)
        s.name = name
        df_list.append(s)
    df = pd.concat(df_list, axis=1).resample('D').ffill()
    return df

df = get_data()

# --- 3. CALCULATIONS ---
# A. Liquidity Engine (Flows)
df['Net_Liquidity'] = df['Fed Assets'] - (df['TGA'].fillna(0) + df['Reverse Repo'].fillna(0))
df['Liquidity_Flow'] = df['Net_Liquidity'].pct_change(365) * 100
df['CPI_YoY'] = df['CPI'].pct_change(365) * 100
df['M2_Real_Growth'] = (df['M2 Supply'].pct_change(365) * 100) - df['CPI_YoY']
df['Real_Rate'] = df['3M Bill'] - df['CPI_YoY']
df['Rate_Momentum'] = df['Real_Rate'] - df['Real_Rate'].shift(365)

# B. Z-Scores
lookback = st.sidebar.slider("Z-Score Lookback (Years)", 1, 10, 3)
window = 365 * lookback

# 1. THE AGGREGATE LIQUIDITY (The 3 Pillars of Flow)
liq_pillars = {'Liquidity_Flow': 1, 'M2_Real_Growth': 1, 'Rate_Momentum': -1}
for col, mult in liq_pillars.items():
    roll = df[col].rolling(window=window, min_periods=30)
    df[f'{col}_Z'] = ((df[col] - roll.mean()) / roll.std()) * mult
    df[f'{col}_Z'] = df[f'{col}_Z'].ffill()

df['Aggregate_Liquidity'] = df[[f'{c}_Z' for c in liq_pillars.keys()]].mean(axis=1)

# 2. THE CREDIT STRESS (Standalone Pillar)
# Inverted so UP = Healthy (Tight Spreads), DOWN = Stress (Wide Spreads)
roll_hy = df['HY_Spread'].rolling(window=window, min_periods=30)
df['Credit_Z'] = ((df['HY_Spread'] - roll_hy.mean()) / roll_hy.std()) * -1
df['Credit_Z'] = df['Credit_Z'].ffill()

# --- 4. FILTERS ---
min_d, max_d = df.index.min().to_pydatetime(), df.index.max().to_pydatetime()
start_d, end_d = st.sidebar.slider("Period", min_d, max_d, (max_d - datetime.timedelta(days=365*5), max_d))
plot_df = df.loc[start_d:end_d].copy()

# --- 5. THE TRIPLE PANEL CHART ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, 
                                    gridspec_kw={'height_ratios': [2, 1, 1]})

# Panel 1: S&P 500
ax1.plot(plot_df.index, plot_df['SP500'], color='#1f77b4', lw=2)
ax1.set_yscale('log')
if not plot_df['SP500'].dropna().empty:
    ymin, ymax = plot_df['SP500'].min() * 0.95, plot_df['SP500'].max() * 1.05
    ax1.set_ylim(ymin, ymax)

# Panel 2: Aggregate Liquidity Flow
ax2.plot(plot_df.index, plot_df['Aggregate_Liquidity'], color='purple', lw=1.5, label="Liquidity Flow Index")
ax2.axhline(0, color='black', lw=1, alpha=0.5)
ax2.fill_between(plot_df.index, 0, plot_df['Aggregate_Liquidity'], where=(plot_df['Aggregate_Liquidity']>=0), color='green', alpha=0.2)
ax2.fill_between(plot_df.index, 0, plot_df['Aggregate_Liquidity'], where=(plot_df['Aggregate_Liquidity']<0), color='red', alpha=0.2)
ax2.set_ylabel("Liquidity Flow")
ax2.legend(loc='upper left')

# Panel 3: Credit Stress (HY Spreads)
ax3.plot(plot_df.index, plot_df['Credit_Z'], color='orange', lw=1.5, label="Credit Health (Z)")
ax3.axhline(0, color='black', lw=1, alpha=0.5)
ax3.fill_between(plot_df.index, 0, plot_df['Credit_Z'], where=(plot_df['Credit_Z']>=0), color='cyan', alpha=0.2)
ax3.fill_between(plot_df.index, 0, plot_df['Credit_Z'], where=(plot_df['Credit_Z']<0), color='red', alpha=0.2)
ax3.set_ylabel("Credit Health")
ax3.legend(loc='upper left')

plt.tight_layout()
st.pyplot(fig)

# --- 6. INTERPRETATION HELPER ---
with st.expander("📝 How to interpret this data (2016 & 2022 Context)"):
    st.write("""
    **1. The "Capitulation" Signal:** Notice how in late 2018 and late 2022, the market bottomed only *after* the Liquidity Index went below -1. 
    A Z-score of -1 to -2 often represents "Maximum Pessimism" where the Fed is forced to pivot.
    
    **2. The Credit Divergence:** Look at **Credit Health** (Bottom Panel). If the S&P 500 is rising but Credit Health is falling (moving towards red), 
    the market is in a dangerous divergence. 
    
    **3. 2016-2017 Mystery:** During this time, Liquidity Flow was low, but Credit Health was extremely high (Cyan). 
    This tells you that even though the Fed wasn't pumping, the private sector (banks/corporate lending) was healthy enough to carry the market.
    """)