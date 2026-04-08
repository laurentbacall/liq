import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, MultipleLocator
import yfinance as yf
import os
from fredapi import Fred
from matplotlib.ticker import FormatStrFormatter

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
        'DFII10': 'Real_10Y_Yield','SOFR': 'SOFR','TGCRRATE': 'TGCR', 'DEXUSEU': 'EURUSD',
        'EXGEUS': 'USDDEM','DGS3MO': 'Fed_3M', 'DGS2': 'Fed_2Y', 'DGS10': 'Fed_10Y'
    }
    for fid, name in fred_ids.items():
        try:
            s = fred.get_series(fid, observation_start=start_date)
            s.index = pd.to_datetime(s.index).tz_localize(None)
            series_dict[name] = s
        except Exception as e:
            print(f"FRED fetch failed for {fid}: {e}")
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
    # --- 4. SYNTHETIC CURRENCY LOGIC ---
    # Convert DEM/USD to EUR/USD for pre-1999 data (Fixed rate: 1.95583)
    if 'USDDEM' in series_dict:
        series_dict['EURUSD_pre1999'] = 1.95583 / series_dict['USDDEM']
        
        if 'EURUSD' in series_dict:
            # Stitch the two series together
            eur_full = pd.concat([
                series_dict['EURUSD_pre1999'][:"1998-12-31"],
                series_dict['EURUSD']
            ]).sort_index()
            
            # Create the final USD/EUR (inverse) series
            series_dict['USDEUR_FULL'] = 1 / eur_full

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
    required_cols = ['Fed_Assets', 'TGA', 'RRP', 'CPI', 'Margin_Debt', 'SP500', 'VIX', 'HY_Spread', 'EURUSD', 'USDDEM', 'VIX']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0  # Create as float series of zeros

    # Now the math is safe
    df['VIX_SMA14'] = df['VIX'].rolling(window=14).mean()
    df['Net_Liq'] = df['Fed_Assets'] - (df['TGA'].fillna(0) + df['RRP'].fillna(0))
    df['Net_Liq_YoY'] = df['Net_Liq'].pct_change(periods=252) * 100
    
    # CPI YoY calculation (for real growth)
    if 'CPI' in df.columns:
        df['CPI_YoY'] = df['CPI'].pct_change(periods=252) * 100
    
    # Real M2 Growth
    if 'M2' in df.columns:
        m2_growth = df['M2'].pct_change(periods=252) * 100
        df['M2_Real_Growth'] = m2_growth - df['CPI_YoY'].fillna(0)
    # REQ: SMA 200 and SMA 50
    df['SP500_SMA200'] = df['SP500'].rolling(window=200).mean()
    df['SP500_SMA50'] = df['SP500'].rolling(window=50).mean()
    # --- UPDATED LEVERAGE LOGIC: Monthly Smoothing ---
    if 'Margin_Debt' in df.columns and 'W5000' in df.columns:
        # 1. Identify the days where Margin Debt actually changes (monthly arrivals)
        # This creates a series that is NaN except on the day the new monthly value drops
        monthly_margin = df['Margin_Debt'].drop_duplicates()

        # 2. Calculate the 30-day moving average for W5000
        w5000_30d_avg = df['W5000'].rolling(window=30).mean()

        # 3. Create the Ratio only on those monthly 'change' dates
        # We align the monthly debt value with the 30-day average of the market
        df['Margin_Market_Ratio'] = monthly_margin / w5000_30d_avg
        
        # 4. Forward fill the ratio so it persists until the next month's data arrives
        df['Margin_Market_Ratio'] = df['Margin_Market_Ratio'].ffill()

        # 5. Re-calculate the 3-year rolling Z-Score based on this cleaner ratio
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
    # 2Y-3M Spread calculation
    if 'Fed_2Y' in df.columns and 'Fed_3M' in df.columns:
        df['Spread_2Y3M'] = df['Fed_2Y'] - df['Fed_3M']
    else:
        df['Spread_2Y3M'] = 0.0

    # --- SMA Spread (Golden/Death Cross Oscillator) ---
    if 'SP500_SMA50' in df.columns and 'SP500_SMA200' in df.columns:
        df['SMA_Spread'] = df['SP500_SMA50'] - df['SP500_SMA200']
    else:
        df['SMA_Spread'] = 0.0
    # --- BULLISH DIVERGENCE LOGIC ---
    #df['SPY_Low'] = df['SP500'].rolling(window=250).min()
    #df['HY_Peak'] = df['HY_Spread'].rolling(window=250).max()
    #spy_making_new_low = df['SP500'] <= df['SPY_Low']
    #hy_not_confirming = df['HY_Spread'] < df['HY_Peak']
    #df['Bull_Divergence'] = (spy_making_new_low & hy_not_confirming).astype(int)
    #df['Divergence_Signal'] = df['Bull_Divergence'].rolling(window=10).sum() >= 3
    # --- NEW VIX RE-ENTRY LOGIC ---
    # 1. Identify if VIX has peaked above 40 in the last 60 days
    vix_high_peak = df['VIX'].rolling(window=60).max() > 40
    
    # 2. Check if VIX is currently closing below its 14-day SMA
    vix_below_sma = df['VIX'] < df['VIX_SMA14']
    

    
    # Leverage Calculations for Exit
    monthly_z = df['Margin_Ratio_Z'].resample('ME').last()
    three_months_above_2 = (monthly_z > 2) & (monthly_z.shift(1) > 2) & (monthly_z.shift(2) > 2)
    prev_max = monthly_z.shift(1).combine(monthly_z.shift(2), max)
    is_below_recent_peak = monthly_z < prev_max
    
    lev_trigger_monthly = three_months_above_2 & is_below_recent_peak
    df['Leverage_Exit_Signal'] = lev_trigger_monthly.reindex(df.index, method='ffill').fillna(False)
    
    # Exit and Re-entry conditions
    death_cross = df['SP500_SMA50'] < df['SP500_SMA200']
    exit_trigger = df['Leverage_Exit_Signal'] & (~death_cross)
    #reentry_trigger = df['Divergence_Signal'] & (death_cross)
    reentry_trigger = vix_high_peak & vix_below_sma
    # --- DYNAMIC ALLOCATION LOGIC ---
    # --- 1. CONFIGURATION PARAMETERS ---
    # --- CONFIGURATION ---
    n_months = 6  
    m_months = 6  
    
    # Calculate the step size per month (not per day)
    # We want to move 80% total over N months
    exit_step_monthly = 80 / n_months   
    entry_step_monthly = 80 / m_months

    allocations = []
    current_alloc = 90.0
    target_alloc = 90.0 

    # Get the list of last trading days for each month in the dataset
    last_days = df.resample('ME').index

    for i in range(len(df)):
        current_date = df.index[i]
        
        # 1. Update the Target based on signals (Signals can happen any day)
        if exit_trigger.iloc[i]:
            target_alloc = 10.0
        elif reentry_trigger.iloc[i]:
            target_alloc = 90.0
            
        # 2. ONLY update the actual allocation if it is the last trading day of the month
        if current_date in last_days:
            if current_alloc > target_alloc:
                # Move down by the monthly step
                current_alloc = max(target_alloc, current_alloc - exit_step_monthly)
            elif current_alloc < target_alloc:
                # Move up by the monthly step
                current_alloc = min(target_alloc, current_alloc + entry_step_monthly)
        
        # On all other days, current_alloc remains unchanged from the previous iteration
        allocations.append(current_alloc)

    df['Allocation_Pct'] = allocations

    # --- 3. PERFORMANCE & FINAL VALUE CALCULATION ---
    if 'SP500' in df.columns:
        df['Market_Returns'] = df['SP500'].pct_change().fillna(0)
        df['Strategy_Returns'] = df['Market_Returns'] * (df['Allocation_Pct'] / 100)
        
        # Growth of $1
        df['Strategy_Cum'] = (1 + df['Strategy_Returns']).cumprod()
        df['SPY_Cum'] = (1 + df['Market_Returns']).cumprod()

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
    value=(timeline[-361], timeline[-1]), 
    format_func=lambda x: x.strftime('%Y-%m')
)

# Use truncate for a clean, error-free slice
p_df = df.truncate(before=start_s, after=end_s)

# --- 5. PLOTTING ---
#fig, axes = plt.subplots(11, 1, figsize=(14, 75))
fig, axes = plt.subplots(nrows=15, ncols=1, figsize=(12, 48), sharex=True)
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
axes[0].plot(p_df.index, p_df['SP500'], color='black', lw=0.5)
if 'SP500_SMA200' in p_df.columns:
    axes[0].plot(p_df.index, p_df['SP500_SMA200'], color='red', ls='--', lw=1.5, label='200D SMA')
if 'SP500_SMA50' in p_df.columns:
    axes[0].plot(p_df.index, p_df['SP500_SMA50'], color='blue', ls=':', lw=1.2, label='50D SMA')
axes[0].legend(loc='upper left')
format_ax(axes[0], "1. S&P 500 (Log) vs 200D SMA", use_log=True)

# 2 Allocation 
#axes[1].plot(p_df.index, get_s('Allocation_Pct'), color='blue', lw=1.5); format_ax(axes[1], "2. System Allocation %")
# 2. Allocation & Performance Comparison
ax1 = axes[1]
ax1_twin = ax1.twinx()

# Plot Allocation % (Background Fill)
ax1.fill_between(p_df.index, get_s('Allocation_Pct'), 0, color='blue', alpha=0.05, label='Allocation %')
ax1.set_ylim(0, 110)
ax1.set_ylabel('Allocation %', fontsize=10)

# RE-BASING: Force both lines to start at 1.0 for the selected period
strat_start = p_df['Strategy_Cum'].iloc[0]
spy_start = p_df['SPY_Cum'].iloc[0]

strategy_final = p_df['Strategy_Cum'].iloc[-1] / strat_start
spy_final = p_df['SPY_Cum'].iloc[-1] / spy_start

# Strategy Line with Final Value in Label
ax1_twin.plot(p_df.index, p_df['Strategy_Cum'] / strat_start, 
              color='navy', lw=1.5, label=f'Tactical Strategy: ${strategy_final:.2f}')

# S&P Line with Final Value in Label
ax1_twin.plot(p_df.index, p_df['SPY_Cum'] / spy_start, 
              color='gray', lw=1, alpha=0.7, label=f'S&P 500: ${spy_final:.2f}')

# Final Touch: Add text annotations at the end of the lines
ax1_twin.text(p_df.index[-1], strategy_final, f' ${strategy_final:.2f}', color='navy', fontweight='bold')
ax1_twin.text(p_df.index[-1], spy_final, f' ${spy_final:.2f}', color='gray')

ax1_twin.set_yscale('log')
ax1_twin.set_ylabel('Growth of $1', fontsize=10)
format_ax(ax1, "2. Strategy vs. S&P 500 Benchmark (Re-based to $1)")

# Combined Legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(loc='upper left', fontsize=9)
ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=9)

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
# 5. Short-Term Rates vs Inflation
axes[5].plot(p_df.index, get_s('CPI_YoY'), color='black', lw=2, label='CPI YoY %', zorder=5)
axes[5].plot(p_df.index, get_s('Fed_3M'), color='teal', lw=1, label='3M Rate', alpha=0.8)
axes[5].axhline(0, color='gray', lw=0.5, alpha=0.5)
format_ax(axes[5], "6a. Short-Term: 3M Rate vs. CPI YoY")
axes[5].legend(loc='upper left', fontsize=9)
axes[5].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# 6. Medium & Long-Term Yields
axes[6].plot(p_df.index, get_s('Fed_2Y'), color='royalblue', lw=1.2, label='2Y Rate')
axes[6].plot(p_df.index, get_s('Fed_10Y'), color='darkblue', lw=1.2, label='10Y Rate')
axes[6].axhline(0, color='gray', lw=0.5, alpha=0.5)
format_ax(axes[6], "6b. Term Structure: 2Y and 10Y Rates")
axes[6].legend(loc='upper left', fontsize=9)
axes[6].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axes[7].plot(p_df.index, get_s('M2_Real_Growth'), color='purple'); format_ax(axes[7], "6. Real M2 Growth")
axes[8].plot(p_df.index, get_s('Real_10Y_Yield'), color='darkblue'); format_ax(axes[8], "7. Real 10Y Yield")
axes[8].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# 8. Yield Curves (10Y-2Y and 2Y-3M)
axes[9].plot(p_df.index, get_s('Yield_Curve_2s10s'), color='darkgreen', lw=1.5, label='10Y-2Y (Eco Cycle)')
axes[9].plot(p_df.index, get_s('Spread_2Y3M'), color='limegreen', lw=1.2, label='2Y-3M (Fed Pivot)')

# Add a horizontal line at 0 to show Inversion
axes[9].axhline(0, color='black', lw=1, alpha=0.5)

# Formatting
format_ax(axes[9], "8. Yield Curves: 10Y-2Y & 2Y-3M")
axes[9].legend(loc='upper left', fontsize=9)
axes[10].plot(p_df.index, get_s('USDEUR_FULL'), color='navy'); format_ax(axes[10], "9. USD/EUR")
axes[10].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
axes[11].plot(p_df.index, get_s('USD_Index'), color='navy'); format_ax(axes[11], "9. USD Index")
#axes[12].plot(p_df.index, get_s('VIX'), color='red', alpha=0.6); format_ax(axes[12], "10. VIX")
# 10. VIX & Re-entry Signal
axes[12].plot(p_df.index, get_s('VIX'), color='red', alpha=0.3, label='VIX')
axes[12].plot(p_df.index, get_s('VIX_SMA14'), color='darkred', lw=1.5, label='14D SMA')
axes[12].axhline(40, color='black', ls=':', alpha=0.5, label='Panic Line (40)')

format_ax(axes[12], "10. VIX & Re-entry Signal (VIX < 14D SMA after 40 Peak)")
axes[12].legend(loc='upper left', fontsize=8)
axes[13].plot(p_df.index, get_s('Funding_Stress'), color='blue'); format_ax(axes[13], "11. Funding Stress")
# 15. SMA Spread (50D - 200D)
axes[14].plot(p_df.index, get_s('SMA_Spread'), color='black', lw=1, label='50D - 200D')

# Fill Area: Green for Golden Cross momentum, Red for Death Cross momentum
axes[14].fill_between(p_df.index, get_s('SMA_Spread'), 0, 
                       where=(get_s('SMA_Spread') >= 0), color='green', alpha=0.3)
axes[14].fill_between(p_df.index, get_s('SMA_Spread'), 0, 
                       where=(get_s('SMA_Spread') < 0), color='red', alpha=0.3)

axes[14].axhline(0, color='black', lw=1) # Zero line
format_ax(axes[14], "15. SMA Momentum (50D SMA - 200D SMA)")
axes[14].legend(loc='upper left', fontsize=9)

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