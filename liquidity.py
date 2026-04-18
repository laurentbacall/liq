# Todo: get the EY version from https://en.macromicro.me/series/1636/us-sp500-earnings-yield
# Todo: see to make it run locally (avec crosshair)
# Todo: regarder pour une version moins noisy du market breadth
# Todo: challenge for curve fitting and sense with Zozo and genAIs
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, MultipleLocator
import yfinance as yf
import os
import streamlit.components.v1 as components
from fredapi import Fred
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import LogFormatterExponent
from matplotlib.ticker import ScalarFormatter
import plotly.tools as tls
import plotly.graph_objects as go
import requests

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Macro Regime Monitor", layout="wide")
st.title("🛡️ SP500 allocation model")

# REQ: Font safety 
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['font.family'] = 'sans-serif'

if "FRED_API_KEY" in st.secrets:
    api_key = st.secrets["FRED_API_KEY"]
else:
    api_key = st.text_input("Enter FRED API Key", type="password")

if not api_key:
    st.info("Please enter your FRED API key to begin.")
    st.stop()

fred = Fred(api_key=api_key)

def get_macromicro_ey():
    """
    Scrapes the S&P 500 Earnings Yield (Series 1636) directly from MacroMicro.
    """
    url = "https://www.macromicro.me/charts/data/1636" # Direct data endpoint for EY
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://en.macromicro.me/series/1636/us-sp500-earnings-yield",
        "Authorization": "Bearer YOUR_TOKEN_IF_YOU_HAVE_ONE" # Optional, usually works without for limited requests
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        
        # MacroMicro JSON structure: data['charts']['c:1636']['series'][0]['data']
        # This structure can vary slightly; we target the specific series data
        series_data = data['data']['c:1636']['series'][0]['data']
        
        df_ey = pd.DataFrame(series_data, columns=['date', 'EY'])
        df_ey['date'] = pd.to_datetime(df_ey['date'])
        df_ey.set_index('date', inplace=True)
        return df_ey
    except Exception as e:
        st.error(f"MacroMicro Scrape Failed: {e}")
        return pd.DataFrame()

# --- 2. DATA ENGINE ---
@st.cache_data(ttl=7200)
def get_master_data():
    start_date = "1995-01-01"
    series_dict = {}
    csv_file = "historical_data.csv" 
    tickers = ["^GSPC", "^VIX", "^W5000", "RSP", "BRK-B"]
    mapping = {"^GSPC": "SP500", "^VIX": "VIX", "^W5000": "W5000", "RSP": "RSP", "BRK-B": "BRK"}

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
        'EXGEUS': 'USDDEM','DGS3MO': 'Fed_3M', 'DGS2': 'Fed_2Y', 'DGS10': 'Fed_10Y', 'GDP': 'GDP'
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
    df = pd.concat(series_dict, axis=1, sort=True).sort_index()

    # 2. Forward fill monthly/weekly data (M2, CPI, FINRA) into daily slots
    df = df.ffill()

    # 3. Trim to only show dates where the S&P 500 actually exists
    # This removes weekends/holidays where FRED has data but markets are closed
    if 'SP500' in df.columns:
        df = df.dropna(subset=['SP500'])
    # --- VALUATION DATA LOADING (Shiller CSV) ---
    # 5. Handle Valuation Data (Shiller Spreadsheet)
    try:
        # Fetching from the live URL provided
        shiller_url = "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/1b9b0a8a-aa83-40bc-a151-c19ef387b564/ie_data.xls"
        # We read the 'Data' sheet, skipping the top descriptive header rows
        shiller_raw = pd.read_excel(shiller_url, sheet_name='Data', skiprows=7)
        
        # Clean Date: Shiller uses YYYY.1 for Oct, YYYY.01 for Jan
        def parse_shiller_date(d):
            try:
                s = str(d)
                if '.' not in s: return pd.NaT
                year, month_dec = s.split('.')
                # .1 -> 10, .01 -> 1
                month_str = month_dec.ljust(2, '0') if len(month_dec) == 1 else month_dec
                return pd.Timestamp(year=int(year), month=int(month_str), day=1)
            except: return pd.NaT

        shiller_raw['Date_dt'] = shiller_raw['Date'].apply(parse_shiller_date)
        shiller_raw = shiller_raw.dropna(subset=['Date_dt']).set_index('Date_dt')
        
        # Convert columns to numeric (avoiding string errors)
        shiller_raw['P'] = pd.to_numeric(shiller_raw['P'], errors='coerce')
        shiller_raw['E'] = pd.to_numeric(shiller_raw['E'], errors='coerce')
        shiller_raw['CAPE'] = pd.to_numeric(shiller_raw['CAPE'], errors='coerce')
        
        # Calculate Earnings Yield
        shiller_raw['EY'] = (shiller_raw['E'] / shiller_raw['P']) * 100
        
        # Join to master df (df is now defined, so no UnboundLocalError)
        df = df.join(shiller_raw[['CAPE', 'EY']], how='left')
        
        # Forward fill monthly data to daily bars
        df[['CAPE', 'EY']] = df[['CAPE', 'EY']].ffill()
        
    except Exception as e:
        st.warning(f"Note: Could not fetch live Shiller data. Check URL or firewall. Error: {e}")

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
        peak_mask = slice_df.loc[:trigger_date, 'Price'] == peak_val
        peak_date = slice_df.loc[:trigger_date][peak_mask].index[-1]
        
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
    required_cols = ['Fed_Assets', 'TGA', 'RRP', 'CPI', 'Margin_Debt', 'SP500', 'VIX', 'HY_Spread', 'EURUSD', 'USDDEM', 'VIX', 'RSP', 'W5000']
    for col in required_cols:
        if col not in df.columns:
            #df[col] = 0.0  # Create as float series of zeros
            df[col] = np.nan

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

    # Calculate the 50-day SMA for the High Yield Spread
    df['HY_Spread_SMA50'] = df['HY_Spread'].rolling(window=50).mean()
    # --- VALUATION CALCULATIONS ---
    # 1. Buffett Indicators
    # Since GDP is quarterly, ffill() spreads the value across the quarter
    df['GDP_Filled'] = df['GDP'].ffill()
    df['Buffett_v1'] = (df['W5000'] / df['GDP_Filled']) 
    df['Buffett_v2'] = (df['W5000'] / (df['GDP_Filled'] + df['Net_Liq'] / 1000))

    # 2. Earnings Yield vs 10Y
    df['EY_10Y_Spread'] = df['EY'] - df['Fed_10Y']
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
    
    # --- 3. CALCULATIONS UPDATE (Hybrid Breadth) ---

    # 1. Create a Synthetic Ratio: Use RSP if available, otherwise use Wilshire 5000
    # We normalize W5000 to the first available RSP value to keep the scale consistent
    # --- HYBRID MARKET BREADTH (W5000 -> RSP) ---
    # 1. Calculate the raw ratios
    df['RSP_Ratio'] = df['RSP'] / df['SP500']
    df['W5000_Ratio'] = df['W5000'] / df['SP500']

    # 2. Find the transition point (first day RSP has data)
    first_rsp_date = df['RSP_Ratio'].first_valid_index()

    if first_rsp_date is not None:
        # 3. Calculate adjustment factor to make W5000 match RSP at launch
        adj_factor = df.loc[first_rsp_date, 'RSP_Ratio'] / df.loc[first_rsp_date, 'W5000_Ratio']
        # 4. Create the Hybrid Ratio: RSP where available, otherwise rebased W5000
        df['Breadth_Ratio'] = df['RSP_Ratio'].fillna(df['W5000_Ratio'] * adj_factor)
    else:
        # Fallback if RSP is totally missing
        df['Breadth_Ratio'] = df['W5000_Ratio']

    # 5. Calculate Momentum on the Hybrid Ratio
    # This calculation is now stable through 2002 and 2003
    df['Ratio_SMA20'] = df['Breadth_Ratio'].rolling(window=20).mean()
    df['Breadth_Spread'] = ((df['Breadth_Ratio'] / df['Ratio_SMA20']) - 1) * 100
    
    breadth_confirmed = df['Breadth_Spread'] > 0

    # --- Improved Breadth Calculation ---
    # --- Rapid Breadth Calculation (5D vs 20D) ---
    df['RSP_SP500_Ratio'] = df['RSP'] / df['SP500']

    # Fast and Slow SMAs
    df['Ratio_5D'] = df['RSP_SP500_Ratio'].rolling(window=5).mean()
    df['Ratio_20D'] = df['RSP_SP500_Ratio'].rolling(window=20).mean()

    # The Spread: Positive = Short-term outperformance (Healthy)
    df['Breadth_Rapid'] = (df['Ratio_5D'] - df['Ratio_20D']) / df['Ratio_20D'] * 100

    # Leverage Calculations for Exit
    monthly_z = df['Margin_Ratio_Z'].resample('ME').last()
    three_months_above_2 = (monthly_z > 2) & (monthly_z.shift(1) > 2) & (monthly_z.shift(2) > 2)
    prev_max = monthly_z.shift(1).combine(monthly_z.shift(2), max)
    is_below_recent_peak = monthly_z < prev_max
    
    # THE FIX: Shift the trigger by 1 month to account for data reporting lag
    # This ensures a signal generated at the end of Month T is only 'seen' at the start of Month T+1
    lev_trigger_monthly = (three_months_above_2 & is_below_recent_peak).shift(1)

    # Re-align with the daily dataframe
    df['Leverage_Exit_Signal'] = lev_trigger_monthly.reindex(df.index, method='ffill').fillna(False)
    
    # Exit and Re-entry conditions
    death_cross = df['SP500_SMA50'] < df['SP500_SMA200']
    exit_trigger = df['Leverage_Exit_Signal'] & (~death_cross)
    # --- SEQUENTIAL RE-ENTRY LOGIC ---
    vix_high_peak = df['VIX'].rolling(window=126).max() > 40
    vix_below_sma = df['VIX'] < df['VIX_SMA14']

    # This will now work in 2002-2003 using the W5000 proxy
    reentry_trigger = vix_high_peak & vix_below_sma & (df['Breadth_Spread'] > 0)
    # --- DYNAMIC ALLOCATION LOGIC ---
    # --- 1. CONFIGURATION PARAMETERS ---
    # --- CONFIGURATION ---
    n_months = 6  
    m_months = 6  
    
    exit_step_monthly = 90 / n_months   
    entry_step_monthly = 90 / m_months

    allocations = []
    current_alloc = 100.0
    target_alloc = 100.0 

    # FIX: Get the actual index labels for the last day of each month
    last_days = df.resample('ME').last().index

    for i in range(len(df)):
        current_date = df.index[i]
        
        # 1. Update Target (can change any day)
        if exit_trigger.iloc[i]:
            target_alloc = 10.0
        elif reentry_trigger.iloc[i]:
            target_alloc = 100.0
            
        # 2. Update actual allocation ONLY on the last trading day of the month
        if current_date in last_days:
            if current_alloc > target_alloc:
                current_alloc = max(target_alloc, current_alloc - exit_step_monthly)
            elif current_alloc < target_alloc:
                current_alloc = min(target_alloc, current_alloc + entry_step_monthly)
        
        allocations.append(current_alloc)

    df['Allocation_Pct'] = allocations

    # --- 3. PERFORMANCE & FINAL VALUE CALCULATION ---
    if 'SP500' in df.columns:
        df['Market_Returns'] = df['SP500'].pct_change().fillna(0)
        df['Strategy_Returns'] = df['Market_Returns'] * (df['Allocation_Pct'] / 100)
        
        # Growth of $1
        df['Strategy_Cum'] = (1 + df['Strategy_Returns']).cumprod()
        df['SPY_Cum'] = (1 + df['Market_Returns']).cumprod()

    # --- Tactical 50/50 SP500 & BRK-B Simulation ---
    if 'SP500' in df.columns and 'BRK' in df.columns and 'Allocation_Pct' in df.columns:
        # 1. Calculate Daily Returns
        df['Ret_SP500'] = df['SP500'].pct_change().fillna(0)
        df['Ret_BRK'] = df['BRK'].pct_change().fillna(0)
    
        # 2. Simulation variables
        active_val = 1.0        # Value of the 50/50 basket
        strat_val = 1.0         # Value of the tactical strategy
        w_sp, w_brk = 0.5, 0.5  # Initial Weights
    
        strat_vals = []
    
        for i in range(len(df)):
            # a. Update the 50/50 basket value based on daily returns
            # We assume the 'basket' exists regardless of the tactical signal
            # to track its own internal growth/rebalancing
            ret_sp = df['Ret_SP500'].iloc[i]
            ret_brk = df['Ret_BRK'].iloc[i]
        
            # Basket daily return
            day_ret_basket = (w_sp * ret_sp) + (w_brk * ret_brk)
            active_val *= (1 + day_ret_basket)
        
            # b. Update weights based on price drift
            # (Needed to keep the ratio correct between rebalance dates)
            sp_val = w_sp * (1 + ret_sp)
            brk_val = w_brk * (1 + ret_brk)
            total_val = sp_val + brk_val
            w_sp, w_brk = sp_val / total_val, brk_val / total_val
        
            # c. Semi-Annual Rebalancing (End of March and October)
            curr_date = df.index[i]
            if i < len(df) - 1:
                next_date = df.index[i+1]
                if (curr_date.month in [3, 10]) and (next_date.month != curr_date.month):
                    w_sp, w_brk = 0.5, 0.5 # Reset to 50/50
        
            # d. Tactical Logic (Continuous Allocation with 10% Floor)
            # Your df['Allocation_Pct'] already contains the 10-100 values
            allocation = df['Allocation_Pct'].iloc[i] / 100.0
        
            # Strategy Return = (Allocated % * Basket Return) + (Cash % * 0)
            # Since you are never 'off', this scaling is continuous
            day_ret_strat = day_ret_basket * allocation
        
            strat_val *= (1 + day_ret_strat)
            strat_vals.append(strat_val)
        
        df['Tactical_5050_Cum'] = strat_vals

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
plot_order = [
    "SP500", "Allocation", "Allocation_5050", "Leverage", "VIX", "Breadth", "Breadth2", "CPI_3M", 
    "Net_Liq", "M2_Growth", "HY_Spread", "Rates_2Y_10Y", 
    "Yield_Curves", "USD_EUR", "USD_Index", 
    "Funding_Stress", "SMA_Momentum",
    "Val_Buffett", "Val_CAPE", "Val_EY", "Val_EY_Macro"
]

fig, axes = plt.subplots(nrows=len(plot_order), ncols=1, figsize=(12, 4 * len(plot_order)), sharex=True)
plt.subplots_adjust(hspace=0.6, top=0.95, bottom=0.05)

# 1. DEFINE HELPER FUNCTIONS FIRST
def get_s(col): 
    return p_df[col] if col in p_df.columns else pd.Series(np.zeros(len(p_df)), index=p_df.index)

def format_ax(ax, title, use_log=False):
    ax.set_title(title, loc='left', fontweight='bold', fontsize=14)
    ax.grid(True, which='major', axis='y', alpha=0.4)
    ax.set_xlim(start_s, end_s)
    if use_log:
        ax.set_yscale('log')
    ax.tick_params(labelbottom=True)
    #ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.2f}' if x < 10 else f'{x:,.0f}'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# 2. CREATE THE MAP
ax_map = {name: axes[i] for i, name in enumerate(plot_order)}

# 3. PLOTTING LOGIC
# S&P 500
ax = ax_map["SP500"]
ax.plot(p_df.index, p_df['SP500'], color='black', lw=0.5)
if 'SP500_SMA200' in p_df.columns:
    ax.plot(p_df.index, p_df['SP500_SMA200'], color='red', ls='dashed', lw=1.5, label='200D SMA')
if 'SP500_SMA50' in p_df.columns:
    ax.plot(p_df.index, p_df['SP500_SMA50'], color='blue', ls='dotted', lw=1.2, label='50D SMA')
ax.legend(loc='center left')
format_ax(ax, "S&P 500, 50-d & 200-d SMA", use_log=True)

# Allocation & Performance
ax = ax_map["Allocation"]
ax_twin = ax.twinx()
strat_series, spy_series = get_s('Strategy_Cum'), get_s('SPY_Cum')
strat_start, spy_start = strat_series.iloc[0], spy_series.iloc[0]
strategy_final = strat_series.iloc[-1] / strat_start
spy_final = spy_series.iloc[-1] / spy_start

ax.fill_between(p_df.index, get_s('Allocation_Pct'), 0, color='blue', alpha=0.05, label='Allocation %')
ax_twin.plot(p_df.index, strat_series / strat_start, color='navy', lw=1, label=f'Tactical: ${strategy_final:.2f}')
ax_twin.plot(p_df.index, spy_series / spy_start, color='gray', lw=1, alpha=0.7, label=f'S&P 500: ${spy_final:.2f}')
ax_twin.set_yscale('log')
ax_twin.yaxis.set_major_formatter(ScalarFormatter())
format_ax(ax, "Tactical Strategy vs. S&P 500 Performance")
ax.legend(loc='upper left', fontsize=9); ax_twin.legend(loc='lower left', fontsize=9)


# --- Allocation & Performance (50/50 Tactical with Floor) ---
if "Allocation_5050" in ax_map:
    ax = ax_map["Allocation_5050"]
    ax_twin = ax.twinx()
    
    # 1. Prepare Series
    strat_series = get_s('Tactical_5050_Cum')
    spy_series = get_s('SPY_Cum')
    
    # 2. Normalize to the start of the visible period (Base $1.00)
    strat_start = strat_series.iloc[0] if strat_series.iloc[0] != 0 else 1
    spy_start = spy_series.iloc[0] if spy_series.iloc[0] != 0 else 1
    
    strategy_final = strat_series.iloc[-1] / strat_start
    spy_final = spy_series.iloc[-1] / spy_start

    # 3. Plot Allocation Shading (The Floor will be visible at 10%)
    ax.fill_between(p_df.index, get_s('Allocation_Pct'), 0, color='blue', alpha=0.05, label='Allocation %')
    ax.set_ylim(0, 105) # Keeps the 10% floor visually consistent
    
    # 4. Plot Performance Lines
    ax_twin.plot(p_df.index, strat_series / strat_start, color='navy', lw=1.2, 
                 label=f'Tactical (50/50): ${strategy_final:.2f}')
    ax_twin.plot(p_df.index, spy_series / spy_start, color='gray', lw=1, alpha=0.7, 
                 label=f'S&P 500: ${spy_final:.2f}')
    
    # 5. Log Scale and Formatting
    ax_twin.set_yscale('log')
    ax_twin.yaxis.set_major_formatter(ScalarFormatter())
    
    format_ax(ax, "Tactical (50% SPY / 50% BRK) vs. S&P 500 Performance")
    
    # 6. Legends inside the graph
    ax.legend(loc='upper left', fontsize=9, frameon=True)
    ax_twin.legend(loc='lower left', fontsize=9, frameon=True)

# --- 14. Leverage Proxy ---
if "Leverage" in ax_map:
    ax = ax_map["Leverage"]
    ax_twin = ax.twinx()

    # 1. Plot the lines (Note the commas after ln1 and ln2)
    # This unpacks the list returned by .plot() so ln1 is a Line2D object
    ln1, = ax.plot(p_df.index, get_s('Margin_Market_Ratio'), color='purple', lw=1.5, label='Margin/W5000 Ratio')
    ln2, = ax_twin.plot(p_df.index, get_s('Margin_Ratio_Z'), color='firebrick', lw=1, alpha=0.7, label='Z-Score')

    # 2. Add the threshold line
    ax_twin.axhline(2, color='red', ls='dashed', alpha=0.5)

    # 3. Shade the excess (Z-Score > 2)
    ax_twin.fill_between(
        p_df.index, 
        get_s('Margin_Ratio_Z'), 
        2, 
        where=(get_s('Margin_Ratio_Z') > 2),
        color='red', 
        alpha=0.3
    )

    # 4. Create a custom legend proxy for the shaded area
    from matplotlib.patches import Patch
    red_patch = Patch(color='red', alpha=0.3, label='Excess Leverage (>2σ)')

    # 5. Consolidate all handles and labels
    # By using the variables defined above, we avoid the NameError
    handles = [ln1, ln2, red_patch]
    labels = [h.get_label() for h in handles]

    # 6. Place legend in the middle right
    ax.legend(handles, labels, loc='center left', fontsize=9, frameon=False)

    format_ax(ax, "Leverage Proxy (Margin Debt / Wilshire 5000)")

# VIX
ax = ax_map["VIX"]
ax.plot(p_df.index, get_s('VIX'), color='red', alpha=0.3, label='VIX')
ax.plot(p_df.index, get_s('VIX_SMA14'), color='darkred', lw=1.5, label='14D SMA')
ax.axhline(40, color='black', ls='dotted', label='Panic Line (40)')
format_ax(ax, "VIX & Re-entry Signal")

# --- 3M Rate vs. CPI YoY ---
if "CPI_3M" in ax_map:
    ax = ax_map["CPI_3M"]
    # Plot the lines and provide labels directly in the plot call
    ax.plot(p_df.index, get_s('CPI_YoY'), color='black', lw=2, label='CPI YoY')
    ax.plot(p_df.index, get_s('Fed_3M'), color='teal', lw=1, label='3M Rate')
    
    # Simple legend call: Matplotlib automatically picks up 'CPI YoY' and '3M Rate'
    # bbox_to_anchor places it in the middle-right outside the plot
    ax.legend(loc='center left', fontsize=9, frameon=False)
    
    format_ax(ax, "3M Rate vs. CPI YoY")

# Net Liquidity
ax = ax_map["Net_Liq"]
ax.plot(p_df.index, get_s('Net_Liq'), color='darkgreen')
format_ax(ax, "FED Net Liquidity")

# Real M2
ax = ax_map["M2_Growth"]
ax.plot(p_df.index, get_s('M2_Real_Growth'), color='purple')
format_ax(ax, "Real M2 YoY Growth")

# --- HY Spread (Inverted) & Z-Score ---
if "HY_Spread" in ax_map:
    ax = ax_map["HY_Spread"]
    ax_twin = ax.twinx()
    
    # 1. Plot Daily Spread (Raw) - Captured as ln1
    # Assuming 'HY_Spread' is the column name in your dataframe
    ln1, = ax.plot(p_df.index, get_s('HY_Spread'), color='blue', lw=0.5, alpha=0.4, label='HY Spread Daily')
    
    # 2. Plot SMA 50 - Captured as ln2
    ln2, = ax.plot(p_df.index, get_s('HY_Spread_SMA50'), color='navy', lw=1.5, label='HY Spread SMA50')
    
    # Invert the primary axis
    ax.invert_yaxis()
    
    # 3. Plot Z-Score on the twin axis - Captured as ln3
    ln3, = ax_twin.plot(p_df.index, get_s('HY_Z'), color='gray', alpha=0.5, label='Z-Score')
    
    # 4. Consolidate handles and labels for the legend
    # This ensures all three lines appear in the same legend box
    hy_handles = [ln1, ln2, ln3]
    hy_labels = [l.get_label() for l in hy_handles]
    
    # 5. Place legend inside or outside based on preference
    # Here it is placed inside the graph (upper left)
    ax.legend(hy_handles, hy_labels, loc='center left', fontsize=8, frameon=True)
    
    format_ax(ax, "HY Spread (Inverted) & Z-Score")

# --- 2Y and 10Y Rates ---
if "Rates_2Y_10Y" in ax_map:
    ax = ax_map["Rates_2Y_10Y"]
    ax.plot(p_df.index, get_s('Fed_2Y'), color='blue', label='2Y Yield')
    ax.plot(p_df.index, get_s('Fed_10Y'), color='red', label='10Y Yield')
    
    # Matplotlib automatically finds labels for a single axis
    ax.legend(loc='center left', fontsize=9, frameon=False)
    format_ax(ax, "Treasury Yields (2Y vs 10Y)")

# --- Yield Curves ---
if "Yield_Curves" in ax_map:
    ax = ax_map["Yield_Curves"]
    ax.plot(p_df.index, get_s('Yield_Curve_2s10s'), color='darkgreen', label='2s10s')
    ax.plot(p_df.index, get_s('Spread_2Y3M'), color='limegreen', label='2Y-3M')
    ax.axhline(0, color='black', lw=1, alpha=0.5)
    
    # Simple legend: automatically finds '2s10s' and '2Y-3M'
    ax.legend(loc='lower left', fontsize=9, frameon=False)
    format_ax(ax, "Yield Curves 10Y-2Y and 2Y-3M")

# --- USD/EUR ---
if "USD_EUR" in ax_map:
    ax = ax_map["USD_EUR"]
    ax.plot(p_df.index, get_s('USDEUR_FULL'), color='blue', label='USD/EUR')
    
    # Simple legend
    ax.legend(loc='center left', fontsize=9, frameon=False)
    format_ax(ax, "USD / EUR Exchange Rate")

# USD Index
ax = ax_map["USD_Index"]
ax.plot(p_df.index, get_s('USD_Index'), color='navy')
format_ax(ax, "USD Index (DXY)")

# Breadth
# --- 13. Market Breadth Plot (Simplified) ---
ax = ax_map["Breadth"]

# Use the pre-calculated spread from the sliced dataframe
breadth_data = p_df['Breadth_Spread']

ax.plot(p_df.index, breadth_data, color='black', lw=1, label='Breadth Momentum (%)')

# Fill Areas
ax.fill_between(p_df.index, breadth_data, 0, 
                where=(breadth_data >= 0), color='green', alpha=0.3)
ax.fill_between(p_df.index, breadth_data, 0, 
                where=(breadth_data < 0), color='red', alpha=0.3)

ax.axhline(0, color='black', lw=1, alpha=0.5)
format_ax(ax, "Market Breadth Momentum (RSP/SP500 vs 20D SMA)")
ax.legend(loc='upper left', fontsize=9)

# --- 13. Market Breadth Plot (Rapid 5/20) ---
if "Breadth2" in ax_map:
    ax = ax_map["Breadth2"]
    
    # Plot the Rapid Spread
    ax.plot(p_df.index, p_df['Breadth_Rapid'], color='black', lw=1, label='5D vs 20D Momentum')
    
    # Fill based on direction
    ax.fill_between(p_df.index, p_df['Breadth_Rapid'], 0, 
                    where=(p_df['Breadth_Rapid'] >= 0), color='green', alpha=0.4)
    ax.fill_between(p_df.index, p_df['Breadth_Rapid'], 0, 
                    where=(p_df['Breadth_Rapid'] < 0), color='red', alpha=0.4)
    
    # Add the zero line
    ax.axhline(0, color='black', lw=1.5, alpha=0.7)
    
    # Add horizontal 'Extreme' lines to spot potential exhaustion
    # These values might need adjustment based on historical lookback
    ax.axhline(0.5, color='green', ls='--', alpha=0.3)
    ax.axhline(-0.5, color='red', ls='--', alpha=0.3)

    format_ax(ax, "Rapid Market Breadth (RSP/SP500: 5D vs 20D SMA)")
    ax.legend(loc='upper left', fontsize=9, frameon=True)

# Funding Stress
ax = ax_map["Funding_Stress"]
ax.plot(p_df.index, get_s('Funding_Stress'), color='blue')
format_ax(ax, "Funding Stress (SOFR-TGCR)")

# SMA Momentum
ax = ax_map["SMA_Momentum"]
ax.plot(p_df.index, get_s('SMA_Spread'), color='black')
ax.fill_between(p_df.index, get_s('SMA_Spread'), 0, where=(get_s('SMA_Spread') >= 0), color='green', alpha=0.3)
ax.fill_between(p_df.index, get_s('SMA_Spread'), 0, where=(get_s('SMA_Spread') < 0), color='red', alpha=0.3)
format_ax(ax, "SMA Momentum (50D - 200D)")

# --- 16. Buffett Indicator Plot ---
if "Val_Buffett" in ax_map:
    ax = ax_map["Val_Buffett"]
    ax.plot(p_df.index, p_df['Buffett_v1'], label='W5000 / GDP (Standard)', color='blue')
    ax.plot(p_df.index, p_df['Buffett_v2'], label='W5000 / (GDP + Liq) (Fed Adjusted)', color='purple', linestyle='--')
    ax.axhline(p_df['Buffett_v1'].mean(), color='blue', alpha=0.3, label='Avg v1')
    format_ax(ax, "Buffett Indicator (Market Cap vs Output & Liquidity)")
    ax.legend()

# --- 17. Shiller CAPE Plot ---
if "Val_CAPE" in ax_map:
    ax = ax_map["Val_CAPE"]
    ax.plot(p_df.index, p_df['CAPE'], color='darkred', lw=2)
    ax.axhline(20, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(30, color='red', linestyle=':', alpha=0.5)
    format_ax(ax, "Shiller CAPE Ratio")
    ax.fill_between(p_df.index, p_df['CAPE'], 30, where=(p_df['CAPE'] > 30), color='red', alpha=0.1)

# --- 18. Earnings Yield vs 10Y ---
if "Val_EY" in ax_map:
    ax = ax_map["Val_EY"]
    ax.plot(p_df.index, p_df['EY'], label='S&P 500 Earnings Yield', color='green')
    ax.plot(p_df.index, p_df['Fed_10Y'], label='10Y Treasury Yield', color='orange')
    # Plot the Spread on a twin axis or as a filled area
    ax.fill_between(p_df.index, p_df['EY'], p_df['Fed_10Y'], 
                    where=(p_df['EY'] > p_df['Fed_10Y']), color='green', alpha=0.1, label='Equity Risk Premium')
    format_ax(ax, "Earnings Yield vs 10Y Yield (Valuation Gap)")
    ax.legend()
# --- 18. S&P 500 EY vs 10Y Yield (MacroMicro Style) ---
if "Val_EY_Macro" in ax_map:
    ax = ax_map["Val_EY_Macro"]
    
    # 1. Fetch/Prepare Data
    ey_df = get_macromicro_ey()
    
    # Check if we actually got data before proceeding
    if not ey_df.empty:
        # Align with your main dataframe index
        p_df['Macro_EY'] = ey_df['EY'].reindex(p_df.index, method='ffill')
    
        # 2. Plot the Yields (ONLY if data exists)
        ax.plot(p_df.index, p_df['Macro_EY'], color='royalblue', lw=2, label='S&P 500 Earnings Yield (MacroMicro)')
        ax.plot(p_df.index, p_df['Fed_10Y'], color='darkorange', lw=2, label='US 10Y Treasury Yield')
        
        # 3. Shade the "Value Gap" (Equity Risk Premium)
        ax.fill_between(p_df.index, p_df['Macro_EY'], p_df['Fed_10Y'], 
                        where=(p_df['Macro_EY'] > p_df['Fed_10Y']), 
                        color='green', alpha=0.15, label='ERP Surplus')
        
        ax.fill_between(p_df.index, p_df['Macro_EY'], p_df['Fed_10Y'], 
                        where=(p_df['Macro_EY'] <= p_df['Fed_10Y']), 
                        color='red', alpha=0.15, label='ERP Deficit')

        format_ax(ax, "Earnings Yield vs. 10Y (MacroMicro Valuation)")
        ax.legend(loc='upper left', fontsize=9, frameon=True)
    else:
        # If scraper fails, show a message on the axis so the app doesn't crash
        ax.text(0.5, 0.5, "MacroMicro Data Unavailable (Blocked or Offline)", 
                transform=ax.transAxes, ha='center', va='center', color='red')
        format_ax(ax, "Earnings Yield vs. 10Y (MacroMicro Valuation - FAILED)")
    else:
        st.warning("MacroMicro EY data is currently unavailable.")
#plt.tight_layout(pad=4.0)
fig.subplots_adjust(hspace=0.6, wspace=0.3, top=0.95, bottom=0.05)

# --- UNIVERSAL FORMATTING, GRID & SHADING ---

# 1. Calculate zoom level
visible_days = (p_df.index[-1] - p_df.index[0]).days

# 2. Define Locators
if visible_days <= 92:
    # Quarter or less: 
    # Major = Mondays
    major_loc = mdates.WeekdayLocator(byweekday=mdates.MO)
    # Minor = Monday, Tuesday, Wednesday, Thursday, Friday (Skipping Sat/Sun)
    minor_loc = mdates.WeekdayLocator(byweekday=(mdates.MO, mdates.TU, mdates.WE, mdates.TH, mdates.FR))
    date_fmt = mdates.DateFormatter('%b %d')
elif visible_days <= 730:
    # 2 Years or less: Major = Months, Minor = Mondays
    major_loc = mdates.MonthLocator()
    minor_loc = mdates.WeekdayLocator(byweekday=mdates.MO)
    date_fmt = mdates.DateFormatter('%b %Y')
else:
    # Default: Major = Years, Minor = Quarters
    major_loc = mdates.YearLocator()
    minor_loc = mdates.MonthLocator(bymonth=(1, 4, 7, 10))
    date_fmt = mdates.DateFormatter('%Y')

# 3. Apply to all axes
for ax in axes:
    ax.xaxis.set_major_locator(major_loc)
    ax.xaxis.set_minor_locator(minor_loc)
    ax.xaxis.set_major_formatter(date_fmt)
    
    # Major Grid: Solid
    ax.grid(visible=True, which='major', axis='x', color='gray', linestyle='-', alpha=0.3)
    # Minor Grid: Dotted (This will now only show 5 lines per week on the 1-quarter view)
    ax.grid(visible=True, which='minor', axis='x', color='gray', linestyle=':', alpha=0.15)

    # Shading for Bear Markets
    for start, end in bear_episodes:
        ax.axvspan(start, end, color='gray', alpha=0.15)

st.pyplot(fig, clear_figure=True)
st.download_button("📥 DOWNLOAD CSV", p_df.to_csv().encode('utf-8'), "macro_monitor.csv", "text/csv")

