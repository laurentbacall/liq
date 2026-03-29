import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, MultipleLocator
import yfinance as yf
import os
from fredapi import Fred

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Macro Regime Monitor", layout="wide")
st.title("🛡️ Institutional Risk & Liquidity Monitor")

# REQ: Font safety
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

if "FRED_API_KEY" in st.secrets:
    api_key = st.secrets["FRED_API_KEY"]
else:
    api_key = st.text_input("Enter FRED API Key", type="password")

if not api_key:
    st.info("Please enter your FRED API key to begin.")
    st.stop()

fred = Fred(api_key=api_key)

# --- 2. DATA ENGINE ---
@st.cache_data(ttl=7200)
def get_master_data():
    start_date = "1995-01-01"
    series_dict = {}
    csv_file = "historical_data.csv" 
    tickers = ["^GSPC", "^VIX", "^W5000"]
    mapping = {"^GSPC": "SP500", "^VIX": "VIX", "^W5000": "W5000"}

    # Initialize df_combined as empty so we have a fallback
    df_combined = pd.DataFrame()

    try:
        if os.path.exists(csv_file):
            # 1. Load the local CSV with Multi-headers
            df_local = pd.read_csv(csv_file, header=[0, 1], index_col=0, parse_dates=True)
            
            # 2. IMPORTANT: Keep only the 'Close' prices (removes High/Low/Open/Volume)
            # This turns the 15 columns back into the 3 we need (^GSPC, ^VIX, ^W5000)
            if 'Close' in df_local.columns.levels[0]:
                df_local = df_local['Close']
        
            # 3. Clean the index (removes the "Date" text row and NaNs)
            df_local.index = pd.to_datetime(df_local.index, errors='coerce')
            df_local = df_local[df_local.index.notnull()].sort_index()
            
            # 4. Initialize our master dataframe with the local data
            df_combined = df_local 
            
            # 5. Identify the last date to fetch the 'missing' days from Yahoo
            last_date = df_combined.index.max() # Fixed: used df_combined instead of empty df
            fetch_start = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            
            try:
                new_data = yf.download(tickers, start=fetch_start, progress=False)
                if not new_data.empty:
                    # Flatten YF MultiIndex to get 'Close'
                    if isinstance(new_data.columns, pd.MultiIndex):
                        new_data = new_data['Close']
                    
                    # Merge local + new
                    df_combined = pd.concat([df_combined, new_data]).sort_index()
                    df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
            except Exception:
                st.sidebar.warning("YF Rate Limit: Using historical CSV data only.")
        else:
            # Full download fallback
            df_combined = yf.download(tickers, start=start_date, progress=False)
            if isinstance(df_combined.columns, pd.MultiIndex):
                df_combined = df_combined['Close']

        # Map to series_dict
        for yf_sym, clean_name in mapping.items():
            if yf_sym in df_combined.columns:
                series_dict[clean_name] = df_combined[yf_sym]

    except Exception as e:
        st.sidebar.error(f"Critical Data Error: {e}")

    # 2. FRED Series (Fetch everything individually)
    fred_ids = {
        'BAMLH0A0HYM2': 'HY_Spread', 'CPIAUCSL': 'CPI',
        'WALCL': 'Fed_Assets', 'M2SL': 'M2', 'WTREGEN': 'TGA', 
        'RRPONTSYD': 'RRP', 'DTWEXBGS': 'USD_Index', 'T10Y2Y': 'Yield_Curve_2s10s',
        'DFII10': 'Real_10Y_Yield','SOFR': 'SOFR','TGCRRATE': 'TGCR'
    }
    for fid, name in fred_ids.items():
        try:
            s = fred.get_series(fid, observation_start=start_date)
            s.index = pd.to_datetime(s.index).tz_localize(None)
            series_dict[name] = s
        except: pass
    # --- ADD THIS INSIDE get_master_data() ---
    try:
        # 3. Robust FINRA Scraper
        finra_url = "https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx"
        # Read the raw file first to find the anchor
        raw_f = pd.read_excel(finra_url)
        
        # FIX: Find the first row that actually contains a date (e.g., '2024-01' or 'Feb-20')
        # This prevents the "list index out of range" error
        mask = raw_f.iloc[:, 0].astype(str).str.contains('20', na=False)
        
        if mask.any():
            start_row = raw_f[mask].index[0]
            # Reload with the correct header offset
            df_f = pd.read_excel(finra_url, skiprows=start_row)
            
            # FINRA files often have 7+ columns; we only want the first two (Date, Debt)
            df_f = df_f.iloc[:, :2] 
            df_f.columns = ['Date', 'Margin_Debt']
            
            # Convert to datetime and clean
            df_f['Date'] = pd.to_datetime(df_f['Date'], errors='coerce')
            df_f = df_f.dropna(subset=['Date', 'Margin_Debt'])
            
            # Ensure Margin_Debt is numeric (removes commas/strings)
            df_f['Margin_Debt'] = pd.to_numeric(df_f['Margin_Debt'], errors='coerce')
            
            series_dict['Margin_Debt'] = df_f.set_index('Date')['Margin_Debt']
        else:
            st.sidebar.warning("Could not find data rows in FINRA file.")
    except Exception as e:
        st.sidebar.error(f"FINRA Scraper Error: {e}")

    # Final Safety Check: If every single data source failed, stop here
    if not series_dict:
        return pd.DataFrame() 

    # 1. Join everything on the UNION of all dates
    df = pd.concat(series_dict, axis=1).sort_index()

    # 2. Forward fill monthly/weekly data (M2, CPI, FINRA) into daily slots
    df = df.ffill()

    # 3. Trim to only show dates where the S&P 500 actually exists
    # This removes weekends/holidays where FRED has data but markets are closed
    if 'SP500' in df.columns:
        df = df.dropna(subset=['SP500'])
    
    return df

def calculate_bear_markets(series):
    """Identifies Peak-to-Trough periods without overlapping layers."""
    df_bear = pd.DataFrame({'Price': series}).dropna()
    episodes = []
    search_start_idx = 0
    
    while search_start_idx < len(df_bear):
        # 1. Look for a crash from a FRESH peak starting at this point
        slice_df = df_bear.iloc[search_start_idx:].copy()
        slice_df['Rolling_Peak'] = slice_df['Price'].cummax()
        slice_df['Drawdown'] = (slice_df['Price'] - slice_df['Rolling_Peak']) / slice_df['Rolling_Peak']
        
        # Check if a 20% drop exists in the remaining data
        trigger_mask = slice_df['Drawdown'] <= -0.20
        if not trigger_mask.any():
            break
            
        # 2. Identify Peak and Trough
        trigger_date = slice_df.index[trigger_mask][0]
        peak_val = slice_df.loc[trigger_date, 'Rolling_Peak']
        peak_date = slice_df[:trigger_date][slice_df['Price'] == peak_val].index[-1]
        
        trough_price = slice_df.loc[trigger_date, 'Price']
        trough_date = trigger_date
        recovery_date = None
        
        # 3. Track the Absolute Trough until a 20% recovery is confirmed
        for date in slice_df.index[slice_df.index >= trigger_date]:
            current_price = slice_df.loc[date, 'Price']
            if current_price < trough_price:
                trough_price = current_price
                trough_date = date
            
            # Exit condition: 20% rally from the lowest point
            if (current_price - trough_price) / trough_price >= 0.20:
                recovery_date = date
                break
        
        # 4. Save the Peak-to-Trough period
        episodes.append((peak_date, trough_date))
        
        # 5. CRITICAL: Restart searching only AFTER the recovery was confirmed
        if recovery_date:
            search_start_idx = df_bear.index.get_loc(recovery_date) + 1
        else:
            # We are currently in a bear market with no recovery yet
            break
            
    return episodes

df = get_master_data()

# --- 3. CALCULATIONS ---
if not df.empty:
    # SAFETY: Ensure all required columns exist as Series (prevents AttributeError)
    required_cols = ['Fed_Assets', 'TGA', 'RRP', 'CPI', 'Margin_Debt', 'SP500', 'VIX', 'HY_Spread']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0  # Create as float series of zeros

    # Now the math is safe
    df['Net_Liq'] = df['Fed_Assets'] - (df['TGA'].fillna(0) + df['RRP'].fillna(0))
    df['Net_Liq_YoY'] = df['Net_Liq'].pct_change(periods=252) * 100
    
    # CPI YoY calculation (for real growth)
    if 'CPI' in df.columns:
        df['CPI_YoY'] = df['CPI'].pct_change(periods=252) * 100
    
    # Real M2 Growth
    if 'M2' in df.columns:
        m2_growth = df['M2'].pct_change(periods=252) * 100
        df['M2_Real_Growth'] = m2_growth - df['CPI_YoY'].fillna(0)
    # REQ: SMA 200
    df['SP500_SMA200'] = df['SP500'].rolling(window=200).mean()
    
    # Calculation: Margin Debt to Market Cap Ratio
    if 'Margin_Debt' in df.columns and 'W5000' in df.columns:
        # We calculate the raw ratio
        df['Margin_Market_Ratio'] = df['Margin_Debt'] / df['W5000']
        
        # Calculate a 3-year rolling Z-Score (756 trading days)
        # This shows how "stretched" leverage is relative to recent history
        rolling_mean = df['Margin_Market_Ratio'].rolling(window=756).mean()
        rolling_std = df['Margin_Market_Ratio'].rolling(window=756).std()
        df['Margin_Ratio_Z'] = (df['Margin_Market_Ratio'] - rolling_mean) / rolling_std
    else:
        df['Margin_Market_Ratio'] = 0
        df['Margin_Ratio_Z'] = 0
    
    if 'SOFR' in df.columns and 'TGCR' in df.columns:
        df['Funding_Stress'] = (df['SOFR'] - df['TGCR']).interpolate().ffill() * 100
    else: df['Funding_Stress'] = 0 
        
    df['HY_Z'] = (df['HY_Spread'] - df['HY_Spread'].rolling(756).mean()) / df['HY_Spread'].rolling(756).std()
    df['VIX_SMA'] = df['VIX'].rolling(20).mean()
    # 2. FINRA Debt Velocity
    if 'Margin_Debt' in df.columns:
        # Since it's monthly data joined to daily, we use 252 (trading days) or 12 (months) 
        # based on how it's sampled. 252 is safer for daily charts.
        df['Margin_Velocity'] = df['Margin_Debt'].pct_change(periods=252) * 100
    else:
        df['Margin_Velocity'] = 0
    # Calculate Bear Market Episodes
    if 'SP500' in df.columns:
        bear_episodes = calculate_bear_markets(df['SP500'])
    else:
        bear_episodes = []
# --- 3. CALCULATIONS & SCORING ---
if not df.empty:
    # 1. Ensure all columns exist as Series to prevent the 'bool' AttributeError
    required_cols = {
        'Net_Liq_YoY': 0, 'Net_Liq': 0, 'Net_Liq_SMA': 0, 
        'M2_Real_Growth': 0, 'Real_10Y_Yield': 5, 'HY_Z': 1, 
        'Funding_Stress': 50, 'VIX': 20, 'VIX_SMA': 20
    }
    for col, default in required_cols.items():
        if col not in df.columns:
            df[col] = default # This creates a Series filled with the default value

    # 2. Now calculate signals safely (using actual Series)
    s1 = (df['Net_Liq_YoY'] > 0).astype(int) * 20
    s2 = (df['Net_Liq'] > df['Net_Liq_SMA']).astype(int) * 20
    s3 = (df['M2_Real_Growth'] > 0).astype(int) * 15
    s4 = (df['Real_10Y_Yield'] < 1.0).astype(int) * 15
    s5 = (df['HY_Z'] < 0.0).astype(int) * 10
    s6 = (df['Funding_Stress'] < 15).astype(int) * 10
    s7 = (df['VIX'] < df['VIX_SMA']).astype(int) * 10

    df['Allocation_Pct'] = s1 + s2 + s3 + s4 + s5 + s6 + s7

# --- 4. PERIOD SLIDER ---
df.index = pd.to_datetime(df.index)
all_dates = df.index

# Create a clean list of Month-Start dates
timeline = pd.date_range(start=all_dates.min(), end=all_dates.max(), freq='MS').tolist()
if all_dates.max() not in timeline:
    timeline.append(all_dates.max())

start_s, end_s = st.select_slider(
    "Select Period", 
    options=timeline, 
    value=(timeline[-121], timeline[-1]), 
    format_func=lambda x: x.strftime('%Y-%m')
)

# Use truncate for a clean, error-free slice
p_df = df.truncate(before=start_s, after=end_s)

# --- 5. PLOTTING ---
#fig, axes = plt.subplots(11, 1, figsize=(14, 75))
fig, axes = plt.subplots(nrows=12, ncols=1, figsize=(12, 48), sharex=True)
plt.subplots_adjust(hspace=0.35)

def format_ax(ax, title, use_log=False):
    ax.set_title(title, loc='left', fontweight='bold', fontsize=14)
    ax.set_xlim(p_df.index.min(), p_df.index.max()) # <--- LOCK THE SCALE
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.grid(True, which='minor', axis='x', color='gray', linestyle=':', alpha=0.2)
    ax.grid(True, which='major', axis='x', color='gray', linestyle='-', alpha=0.4)
    ax.grid(True, which='major', axis='y', alpha=0.4)
    # FORCE ALIGNMENT: Every chart must have the same start/end points
    ax.set_xlim(start_s, end_s)
    if use_log:
        ax.set_yscale('log')
        # REQ: 1,000 Point Intervals
        ax.yaxis.set_major_locator(MultipleLocator(1000))
    
    if 'Recessions' in p_df.columns:
        ax.fill_between(p_df.index, 0, 1, where=p_df['Recessions']>0, color='gray', alpha=0.2, transform=ax.get_xaxis_transform(), zorder=-1)
    
    ax.tick_params(labelbottom=True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))

def get_s(col): return p_df[col] if col in p_df.columns else pd.Series(np.zeros(len(p_df)), index=p_df.index)

# 1. SP500 (REQ: Log Scale + SMA 200 + 1000pt Grid)
axes[0].plot(p_df.index, p_df['SP500'], color='black', lw=2)
if 'SP500_SMA200' in p_df.columns:
    axes[0].plot(p_df.index, p_df['SP500_SMA200'], color='red', ls='--', lw=1.5, label='200D SMA')
    axes[0].legend(loc='upper left')
format_ax(axes[0], "1. S&P 500 (Log) vs 200D SMA", use_log=True)

# 2 Allocation 
axes[1].plot(p_df.index, get_s('Allocation_Pct'), color='blue', lw=1.5); format_ax(axes[1], "2. System Allocation %")


# axes[4].plot(p_df.index, get_s('HY_Spread'), color='orange'); axes[4].invert_yaxis(); format_ax(axes[4], "5. HY Spread (Inverted)")

# 5. HY Spread (Inverted) + HY Z-Score
ax2 = axes[2]
ax2_twin = ax2.twinx()  # Create secondary axis

# Plot HY Spread (Primary - Left Axis)
ax2.plot(p_df.index, get_s('HY_Spread'), color='orange', lw=1.5, label='HY Spread')
ax2.invert_yaxis() 

# Plot HY Z-Score (Secondary - Right Axis)
# We use a semi-transparent fill or a dashed line to keep it readable
ax2_twin.plot(p_df.index, get_s('HY_Z'), color='gray', lw=1, alpha=0.5, label='HY Z-Score')
ax2_twin.axhline(0, color='black', lw=0.5, alpha=0.3) # Zero line for Z-score

# Formatting
format_ax(ax2, "5. HY Spread (Inverted) & HY Z-Score")

# Adjust right-side labels for the twin axis
ax2_twin.set_ylabel('Z-Score', fontsize=10, alpha=0.7)
ax2_twin.tick_params(axis='y', labelsize=9)

# Combine legends from both axes
lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=9)

# 11. Leverage Proxy: Margin Debt / W5000 Ratio & Z-Score
ax3 = axes[3]
ax3_twin = ax3.twinx()

# Plot Raw Ratio (Left Axis - Purple)
ax3.plot(p_df.index, get_s('Margin_Market_Ratio'), color='purple', lw=1.5, label='Margin/W5000 Ratio')
ax3.set_ylabel('Raw Ratio', color='purple', fontsize=10)

# Plot Z-Score (Right Axis - Firebrick)
ax3_twin.plot(p_df.index, get_s('Margin_Ratio_Z'), color='firebrick', lw=1, alpha=0.7, label='Ratio Z-Score')
ax3_twin.axhline(0, color='black', lw=1, alpha=0.5)
ax3_twin.axhline(2, color='red', ls='--', alpha=0.5) # Danger Zone (+2 Sigma)
ax3_twin.axhline(-2, color='blue', ls='--', alpha=0.5) # De-leveraging (-2 Sigma)

# Shading for high-leverage "Danger Zones"
ax3_twin.fill_between(p_df.index, get_s('Margin_Ratio_Z'), 2, 
                       where=(get_s('Margin_Ratio_Z') >= 2), 
                       color='red', alpha=0.2, interpolate=True)

format_ax(ax3, "6. Leverage Proxy (Margin Debt / Wilshire 5000)")

# Combine Legends
lines, labels = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines + lines2, labels + labels2, loc='upper left')


# 6-10
axes[4].plot(p_df.index, get_s('Net_Liq'), color='darkgreen'); format_ax(axes[4], "5. Net Liquidity Path")
axes[5].plot(p_df.index, get_s('CPI_YoY'), color='purple'); format_ax(axes[5], "6. CPI YoY")
axes[5].plot(p_df.index, get_s('M2_Real_Growth'), color='purple'); format_ax(axes[6], "6. Real M2 Growth")
axes[7].plot(p_df.index, get_s('Real_10Y_Yield'), color='darkblue'); format_ax(axes[7], "7. Real 10Y Yield")
axes[8].p6ot(p_df.index, get_s('Yield_Curve_2s10s'), color='darkgreen'); format_ax(axes[8], "8. Yield Curve")
axes[9].plot(p_df.index, get_s('USD_Index'), color='navy'); format_ax(axes[9], "9. USD Index")
axes[10].plot(p_df.index, get_s('USD_Index'), color='navy'); format_ax(axes[10], "9. USD Index")
axes[11].plot(p_df.index, get_s('VIX'), color='red', alpha=0.6); format_ax(axes[11], "10. VIX")
axes[12].plot(p_df.index, get_s('Funding_Stress'), color='blue'); format_ax(axes[12], "11. Funding Stress")


plt.tight_layout(pad=4.0)

# --- 4. UNIVERSAL BEAR MARKET SHADING (No Legend) ---
for ax in axes:
    for start, end in bear_episodes:
        # We removed the 'label' argument so it stays out of the legend
        ax.axvspan(start, end, color='gray', alpha=0.15)

# Ensure no bear market legend code remains
# (Delete any axes[0].legend(...) specifically for bear markets)

st.pyplot(fig)
st.download_button("📥 DOWNLOAD CSV", p_df.to_csv().encode('utf-8'), "macro_monitor.csv", "text/csv")