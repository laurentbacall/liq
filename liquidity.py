import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
import io

# --- 1. CONFIGURATION & UI ---
st.set_page_config(page_title="Macro Liquidity Dashboard", layout="wide")
st.title("🌊 Institutional Liquidity Monitor")

# --- 2. AUTHENTICATION (Secrets vs User Input) ---
if "FRED_API_KEY" in st.secrets:
    api_key = st.secrets["FRED_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter FRED API Key", type="password", help="Get a free key at research.stlouisfed.org")

if not api_key:
    st.info("👋 Welcome! Please enter your FRED API key in the sidebar to begin.")
    st.stop()

fred = Fred(api_key=api_key)
lookback_years = st.sidebar.slider("Z-Score Lookback (Years)", 1, 10, 3)

# --- 3. DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_data():
    series_ids = {
        'WALCL': 'Fed Assets',
        'WTREGEN': 'TGA',
        'RRPONTSYD': 'Reverse Repo',
        'TB3MS': '3M Bill',
        'CPIAUCSL': 'CPI',
        'M2SL': 'M2 Supply',
        'SP500': 'SP500'
    }
    
    df_list = []
    for s_id, name in series_ids.items():
        try:
            s = fred.get_series(s_id)
            s.name = name
            df_list.append(s)
        except Exception as e:
            st.error(f"Error fetching {s_id}: {e}")
    
    # Align frequencies: Resample to Daily and Forward Fill
    df = pd.concat(df_list, axis=1).resample('D').ffill()
    return df

with st.spinner('Syncing with Federal Reserve Database...'):
    df = get_data()

# --- 4. HARDENED CALCULATIONS ---
# Fill drains with 0 and calculate Net Liquidity
df['Net Liquidity'] = df['Fed Assets'] - (df['TGA'].fillna(0) + df['Reverse Repo'].fillna(0))

# Real 3M Rate calculation
df['CPI_YoY'] = df['CPI'].pct_change(365) * 100
df['Real 3M Rate'] = df['3M Bill'] - df['CPI_YoY']

# M2 Real Growth calculation
df['M2_YoY'] = df['M2 Supply'].pct_change(365) * 100
df['M2 Real Growth'] = df['M2_YoY'] - df['CPI_YoY']

# Z-Scores with min_periods=30 for resilience
window = 365 * lookback_years
cols_to_z = {'Net Liquidity': 1, 'Real 3M Rate': -1, 'M2 Real Growth': 1}

for col, multiplier in cols_to_z.items():
    roll_mean = df[col].rolling(window=window, min_periods=30).mean()
    roll_std = df[col].rolling(window=window, min_periods=30).std()
    df[f'{col}_Z'] = ((df[col] - roll_mean) / roll_std) * multiplier

# Composite Index
z_cols = [f'{c}_Z' for c in cols_to_z.keys()]
df['Aggregate_Index'] = df[z_cols].mean(axis=1)

# --- 5. DASHBOARD DISPLAY ---
last_idx = df['Aggregate_Index'].iloc[-1]
status = "EXPANDING" if last_idx > 1 else "CONTRACTING" if last_idx < -1 else "NEUTRAL"
status_color = "green" if last_idx > 1 else "red" if last_idx < -1 else "gray"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Liquidity Index", f"{last_idx:.2f} Z", delta=status)
m2.metric("Real 3M Rate", f"{df['Real 3M Rate'].iloc[-1]:.2f}%")
m3.metric("M2 Real Growth", f"{df['M2 Real Growth'].iloc[-1]:.2f}%")
m4.metric("Net Fed Liquidity", f"${df['Net Liquidity'].iloc[-1]/1e6:.2f}T")

st.divider()

col_chart, col_stats = st.columns([3, 1])

with col_chart:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    plt.subplots_adjust(hspace=0.1)

    # Panel 1: S&P 500
    ax1.plot(df.index, df['SP500'], color='#1f77b4', lw=1.5, label='S&P 500')
    ax1.fill_between(df.index, df['SP500'].min(), df['SP500'].max(), where=(df['Aggregate_Index'] > 1), color='green', alpha=0.1)
    ax1.fill_between(df.index, df['SP500'].min(), df['SP500'].max(), where=(df['Aggregate_Index'] < -1), color='red', alpha=0.1)
    ax1.set_title("Market Response to Liquidity Conditions", fontsize=12)
    ax1.legend(loc='upper left')

    # Panel 2: Aggregate Index
    ax2.plot(df.index, df['Aggregate_Index'], color='gray', alpha=0.5)
    ax2.fill_between(df.index, 0, df['Aggregate_Index'], where=(df['Aggregate_Index'] > 0), color='green', alpha=0.4)
    ax2.fill_between(df.index, 0, df['Aggregate_Index'], where=(df['Aggregate_Index'] < 0), color='red', alpha=0.4)
    ax2.axhline(1, ls=':', color='green', alpha=0.6)
    ax2.axhline(-1, ls=':', color='red', alpha=0.6)
    ax2.set_ylabel("Composite Z-Score")
    
    st.pyplot(fig)

with col_stats:
    st.subheader("Component Health")
    for col in cols_to_z.keys():
        val = df[f'{col}_Z'].iloc[-1]
        st.write(f"**{col}**")
        
        # 1. Check if the value is NaN before processing
        if pd.isna(val):
            st.warning(f"Data for {col} is currently unavailable.")
            st.progress(0.0)
            st.caption("Current Z-Score: N/A")
        else:
            # 2. Clamp and Normalize (0 to 1 range)
            # This ensures even a Z-score of +5.0 doesn't exceed 1.0
            clamped_val = max(min(val, 3), -3)
            normalized_val = (clamped_val + 3) / 6
            
            # 3. Final safety check for Streamlit's 0.0-1.0 requirement
            normalized_val = float(max(min(normalized_val, 1.0), 0.0))
            
            st.progress(normalized_val)
            st.caption(f"Current Z-Score: {val:.2f}")

    st.divider()
    
    # --- REPORT DOWNLOAD ---
    st.subheader("Export Data")
    # We export the last 5 years of processed signals
    report_df = df[['SP500', 'Aggregate_Index'] + z_cols].tail(1825)
    csv = report_df.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Liquidity Report (CSV)",
        data=csv,
        file_name='liquidity_report.csv',
        mime='text/csv',
    )

# --- 6. DATA HEALTH CHECK ---
with st.expander("🔍 Data Diagnostics (Check for Lags)"):
    health_start = df.apply(lambda x: x.first_valid_index())
    health_end = df.apply(lambda x: x.last_valid_index())
    diagnostics = pd.DataFrame({'Data Starts': health_start, 'Data Ends': health_end})
    st.table(diagnostics)