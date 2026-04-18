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
import cloudscraper
import re
import base64
import json

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

def get_macromicro_ey_robust():
    """Bypasses Cloudflare to fetch the S&P 500 Earnings Yield."""
    url = "https://en.macromicro.me/series/1636/us-sp500-earnings-yield"
    scraper = cloudscraper.create_scraper(
        browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
    )
    try:
        response = scraper.get(url)
        if response.status_code == 200:
            match = re.search(r'atob\("([^"]+)"\)', response.text)
            if match:
                b64_string = match.group(1)
                json_data = json.loads(base64.b64decode(b64_string).decode('utf-8'))
                df = pd.DataFrame(json_data, columns=['date', 'EY'])
                df['date'] = pd.to_datetime(df['date'], unit='ms')
                return df.set_index('date')
    except Exception:
        pass
    return pd.DataFrame()

# --- 2. DATA ENGINE ---
@st.cache_data(ttl=7200)
def get_master_data():
    start_date = "1995-01-01"
    series_dict = {}
    csv_file = "historical_data.csv" 
    tickers = ["^GSPC", "^VIX", "^W5000", "RSP", "BRK-B"]
    mapping = {"^GSPC": "SP500", "^VIX": "VIX", "^W5000": "W5000", "RSP": "RSP", "BRK-B": "BRK"}

    df_combined = pd.DataFrame()

    try:
        if os.path.exists(csv_file):
            df_local = pd.read_csv(csv_file, header=[0, 1], index_col=0, parse_dates=True)
            if 'Close' in df_local.columns.levels[0]:
                df_local = df_local['Close']
            df_local.index = pd.to_datetime(df_local.index, errors='coerce')
            df_local = df_local[df_local.index.notnull()].sort_index()
            df_combined = df_local 
            last_date = df_combined.index.max()
            fetch_start = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            try:
                new_data = yf.download(tickers, start=fetch_start, progress=False)
                if not new_data.empty:
                    if isinstance(new_data.columns, pd.MultiIndex):
                        new_data = new_data['Close']
                    df_combined = pd.concat([df_combined, new_data]).sort_index()
                    df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
            except Exception:
                st.sidebar.warning("YF Rate Limit: Using historical CSV data only.")
        else:
            df_combined = yf.download(tickers, start=start_date, progress=False)
            if isinstance(df_combined.columns, pd.MultiIndex):
                df_combined = df_combined['Close']

        for yf_sym, clean_name in mapping.items():
            if yf_sym in df_combined.columns:
                series_dict[clean_name] = df_combined[yf_sym]
    except Exception as e:
        st.sidebar.error(f"Critical Data Error: {e}")

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
        except Exception: pass

    try:
        finra_url = "https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx"
        raw_f = pd.read_excel(finra_url)
        mask = raw_f.iloc[:, 0].astype(str).str.contains('20', na=False)
        if mask.any():
            start_row = raw_f[mask].index[0]
            df_f = pd.read_excel(finra_url, skiprows=start_row).iloc[:, :2]
            df_f.columns = ['Date', 'Margin_Debt']
            df_f['Date'] = pd.to_datetime(df_f['Date'], errors='coerce')
            df_f = df_f.dropna(subset=['Date', 'Margin_Debt'])
            df_f['Margin_Debt'] = pd.to_numeric(df_f['Margin_Debt'], errors='coerce')
            series_dict['Margin_Debt'] = df_f.set_index('Date')['Margin_Debt']
    except Exception: pass

    if 'USDDEM' in series_dict:
        series_dict['EURUSD_pre1999'] = 1.95583 / series_dict['USDDEM']
        if 'EURUSD' in series_dict:
            eur_full = pd.concat([series_dict['EURUSD_pre1999'][:"1998-12-31"], series_dict['EURUSD']]).sort_index()
            series_dict['USDEUR_FULL'] = 1 / eur_full

    if not series_dict: return pd.DataFrame() 

    df = pd.concat(series_dict, axis=1, sort=True).sort_index()
    df = df.ffill()
    if 'SP500' in df.columns:
        df = df.dropna(subset=['SP500'])

    # Shiller Data
    try:
        shiller_url = "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/1b9b0a8a-aa83-40bc-a151-c19ef387b564/ie_data.xls"
        shiller_raw = pd.read_excel(shiller_url, sheet_name='Data', skiprows=7)
        def parse_shiller_date(d):
            try:
                s = str(d)
                if '.' not in s: return pd.NaT
                y, m_dec = s.split('.')
                m_str = m_dec.ljust(2, '0') if len(m_dec) == 1 else m_dec
                return pd.Timestamp(year=int(y), month=int(m_str), day=1)
            except: return pd.NaT
        shiller_raw['Date_dt'] = shiller_raw['Date'].apply(parse_shiller_date)
        shiller_raw = shiller_raw.dropna(subset=['Date_dt']).set_index('Date_dt')
        shiller_raw['EY'] = (pd.to_numeric(shiller_raw['E'], errors='coerce') / pd.to_numeric(shiller_raw['P'], errors='coerce')) * 100
        df = df.join(shiller_raw[['CAPE', 'EY']], how='left').ffill()
    except Exception: pass

    return df

def calculate_bear_markets(series):
    df_bear = pd.DataFrame({'Price': series}).dropna()
    episodes = []
    idx = 0
    while idx < len(df_bear):
        slice_df = df_bear.iloc[idx:].copy()
        slice_df['Rolling_Peak'] = slice_df['Price'].cummax()
        slice_df['Drawdown'] = (slice_df['Price'] - slice_df['Rolling_Peak']) / slice_df['Rolling_Peak']
        trigger_mask = slice_df['Drawdown'] <= -0.20
        if not trigger_mask.any(): break
        trigger_date = slice_df.index[trigger_mask][0]
        peak_val = slice_df.loc[trigger_date, 'Rolling_Peak']
        peak_date = slice_df.loc[:trigger_date][slice_df.loc[:trigger_date, 'Price'] == peak_val].index[-1]
        t_price, t_date = slice_df.loc[trigger_date, 'Price'], trigger_date
        recovery_date = None
        for date in slice_df.index[slice_df.index >= trigger_date]:
            curr = slice_df.loc[date, 'Price']
            if curr < t_price: t_price, t_date = curr, date
            if (curr - t_price) / t_price >= 0.20:
                recovery_date = date
                break
        episodes.append((peak_date, t_date))
        if recovery_date: idx = df_bear.index.get_loc(recovery_date) + 1
        else: break
    return episodes

df = get_master_data()

# --- 3. CALCULATIONS ---
if not df.empty:
    required_cols = ['Fed_Assets', 'TGA', 'RRP', 'CPI', 'Margin_Debt', 'SP500', 'VIX', 'HY_Spread', 'RSP', 'W5000']
    for col in required_cols:
        if col not in df.columns: df[col] = np.nan

    df['VIX_SMA14'] = df['VIX'].rolling(14).mean()
    df['Net_Liq'] = df['Fed_Assets'] - (df['TGA'].fillna(0) + df['RRP'].fillna(0))
    df['CPI_YoY'] = df['CPI'].pct_change(252) * 100
    df['M2_Real_Growth'] = (df['M2'].pct_change(252) * 100) - df['CPI_YoY'].fillna(0)
    df['SP500_SMA200'] = df['SP500'].rolling(200).mean()
    df['SP500_SMA50'] = df['SP500'].rolling(50).mean()
    
    # Margin Logic
    if 'Margin_Debt' in df.columns and 'W5000' in df.columns:
        m_debt = df['Margin_Debt'].drop_duplicates()
        w_avg = df['W5000'].rolling(30).mean()
        df['Margin_Market_Ratio'] = (m_debt / w_avg).ffill()
        df['Margin_Ratio_Z'] = (df['Margin_Market_Ratio'] - df['Margin_Market_Ratio'].rolling(756).mean()) / df['Margin_Market_Ratio'].rolling(756).std()

    df['Funding_Stress'] = (df['SOFR'] - df['TGCR']).interpolate().ffill() * 100
    df['HY_Z'] = (df['HY_Spread'] - df['HY_Spread'].rolling(756).mean()) / df['HY_Spread'].rolling(756).std()
    df['Margin_Velocity'] = df['Margin_Debt'].pct_change(252) * 100
    bear_episodes = calculate_bear_markets(df['SP500'])
    df['Spread_2Y3M'] = df['Fed_2Y'] - df['Fed_3M']
    df['HY_Spread_SMA50'] = df['HY_Spread'].rolling(50).mean()
    df['Buffett_v2'] = (df['W5000'] / (df['GDP'].ffill() + df['Net_Liq'] / 1000))
    df['EY_10Y_Spread'] = df['EY'] - df['Fed_10Y']
    df['SMA_Spread'] = df['SP500_SMA50'] - df['SP500_SMA200']

    # Breadth Logic
    df['RSP_Ratio'] = df['RSP'] / df['SP500']
    df['W5000_Ratio'] = df['W5000'] / df['SP500']
    first_rsp = df['RSP_Ratio'].first_valid_index()
    if first_rsp:
        adj = df.loc[first_rsp, 'RSP_Ratio'] / df.loc[first_rsp, 'W5000_Ratio']
        df['Breadth_Ratio'] = df['RSP_Ratio'].fillna(df['W5000_Ratio'] * adj)
    df['Breadth_Spread'] = ((df['Breadth_Ratio'] / df['Breadth_Ratio'].rolling(20).mean()) - 1) * 100
    df['Breadth_Rapid'] = ((df['RSP_Ratio'].rolling(5).mean() / df['RSP_Ratio'].rolling(20).mean()) - 1) * 100

    # Allocation Logic
    exit_trigger = (df['SP500'] < df['SP500_SMA200']) | (df['VIX'] > 30)
    reentry_trigger = (df['SP500'] > df['SP500_SMA200']) & (df['VIX'] < 25)
    allocations, curr_alloc = [], 100.0
    last_days = df.resample('M').last().index
    for i in range(len(df)):
        target = 10.0 if exit_trigger.iloc[i] else (100.0 if reentry_trigger.iloc[i] else curr_alloc)
        if df.index[i] in last_days:
            curr_alloc = max(target, curr_alloc - 20) if curr_alloc > target else min(target, curr_alloc + 20)
        allocations.append(curr_alloc)
    df['Allocation_Pct'] = allocations
    df['Strategy_Cum'] = (1 + (df['SP500'].pct_change().fillna(0) * df['Allocation_Pct']/100)).cumprod()
    df['SPY_Cum'] = (1 + df['SP500'].pct_change().fillna(0)).cumprod()

# --- 4. PERIOD SELECTION ---
timeline = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS').tolist()
start_s, end_s = st.select_slider("Select Period", options=timeline, value=(timeline[-120], timeline[-1]), format_func=lambda x: x.strftime('%Y'))
p_df = df.truncate(before=start_s, after=end_s)

# --- 5. PLOTTING ---
plot_order = [
    "SP500", "Allocation", "Leverage", "VIX", "Breadth", "Breadth2", 
    "Net_Liq", "HY_Spread", "Rates_2Y_10Y", "Yield_Curves", "USD_Index", 
    "Funding_Stress", "SMA_Momentum", "Val_Buffett", "Val_CAPE", "Val_EY_Macro"
]
fig, axes = plt.subplots(nrows=len(plot_order), ncols=1, figsize=(14, 4 * len(plot_order)), sharex=True)
ax_map = dict(zip(plot_order, axes))

def format_ax(ax, title):
    ax.set_title(title, loc='left', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    for start, end in bear_episodes:
        if start >= p_df.index[0] or end >= p_df.index[0]:
            ax.axvspan(start, end, color='gray', alpha=0.1)

# Populate Plots
for name, ax in ax_map.items():
    if name == "SP500":
        ax.plot(p_df.index, p_df['SP500'], color='black')
        ax.plot(p_df.index, p_df['SP500_SMA200'], color='red', ls='--')
        ax.set_yscale('log')
        format_ax(ax, "S&P 500 & SMA 200")
    elif name == "Allocation":
        ax_t = ax.twinx()
        ax.fill_between(p_df.index, p_df['Allocation_Pct'], 0, color='blue', alpha=0.05)
        ax_t.plot(p_df.index, p_df['Strategy_Cum'], color='navy', label='Tactical')
        ax_t.plot(p_df.index, p_df['SPY_Cum'], color='gray', alpha=0.5, label='S&P 500')
        ax_t.set_yscale('log')
        format_ax(ax, "Tactical Strategy Performance")
    elif name == "Val_EY_Macro":
        ey_df = get_macromicro_ey_robust()
        if not ey_df.empty:
            p_df['Macro_EY'] = ey_df['EY'].reindex(p_df.index, method='ffill')
            ax.plot(p_df.index, p_df['Macro_EY'], color='blue', label='EY %')
            ax.plot(p_df.index, p_df['Fed_10Y'], color='red', label='10Y %')
            ax.fill_between(p_df.index, p_df['Macro_EY'], p_df['Fed_10Y'], where=(p_df['Macro_EY'] > p_df['Fed_10Y']), color='green', alpha=0.2)
            format_ax(ax, "Earnings Yield vs 10Y (MacroMicro)")
            ax.legend()
    # [Other plot logic for VIX, Breadth, etc. remains as per your original file]

plt.tight_layout()
st.pyplot(fig)
st.download_button("📥 DOWNLOAD CSV", p_df.to_csv().encode('utf-8'), "macro_monitor.csv")