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

# --- NEW: ROBUST MACROMICRO SCRAPER ---
def get_macromicro_ey_robust():
    url = "https://en.macromicro.me/series/1636/us-sp500-earnings-yield"
    scraper = cloudscraper.create_scraper(
        browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
    )
    try:
        response = scraper.get(url, timeout=10)
        if response.status_code == 200:
            match = re.search(r'atob\("([^"]+)"\)', response.text)
            if match:
                b64_string = match.group(1)
                json_data = json.loads(base64.b64decode(b64_string).decode('utf-8'))
                df_ey = pd.DataFrame(json_data, columns=['date', 'EY'])
                df_ey['date'] = pd.to_datetime(df_ey['date'], unit='ms')
                return df_ey.set_index('date')
    except: pass
    return pd.DataFrame()

# --- 2. DATA ENGINE ---
@st.cache_data(ttl=7200)
def get_master_data():
    start_date = "1995-01-01"
    series_dict = {}
    
    # Financial Assets
    tickers = ["^GSPC", "^VIX", "^W5000", "RSP", "BRK-B"]
    mapping = {"^GSPC": "SP500", "^VIX": "VIX", "^W5000": "W5000", "RSP": "RSP", "BRK-B": "BRK"}
    
    try:
        df_yf = yf.download(tickers, start=start_date, progress=False)
        if isinstance(df_yf.columns, pd.MultiIndex):
            df_yf = df_yf['Close']
        for yf_sym, clean_name in mapping.items():
            if yf_sym in df_yf.columns:
                series_dict[clean_name] = df_yf[yf_sym]
    except Exception as e:
        st.sidebar.error(f"YFinance Error: {e}")

    # FRED Series
    fred_ids = {
        'BAMLH0A0HYM2': 'HY_Spread', 
        'CPIAUCSL': 'CPI',
        'WALCL': 'Fed_Assets', 
        'M2SL': 'M2',
        'WTREGEN': 'TGA', 
        'RRPONTSYD': 'RRP', 
        'DTWEXBGS': 'USD_Index',
        'T10Y2Y': 'Yield_Curve_2s10s',
        'DFII10': 'Real_10Y_Yield',
        'SOFR': 'SOFR',
        'TGCRRATE': 'TGCR',
        'DEXUSEU': 'EURUSD',
        'EXGEUS': 'USDDEM',
        'DGS3MO': 'Fed_3M',
        'DGS2': 'Fed_2Y',
        'DGS10': 'Fed_10Y',
        'GDP': 'GDP'
    }
    
    for fid, name in fred_ids.items():
        try:
            s = fred.get_series(fid, observation_start=start_date)
            s.index = pd.to_datetime(s.index).tz_localize(None)
            series_dict[name] = s
        except: pass

    # Margin Debt
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
            series_dict['Margin_Debt'] = df_f.set_index('Date')['Margin_Debt']
    except: pass

    if not series_dict: return pd.DataFrame()
    
    df = pd.concat(series_dict, axis=1).sort_index().ffill()
    
    # Shiller CAPE integration
    try:
        shiller_url = "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/1b9b0a8a-aa83-40bc-a151-c19ef387b564/ie_data.xls"
        shiller_raw = pd.read_excel(shiller_url, sheet_name='Data', skiprows=7)
        def parse_shiller_date(d):
            try:
                s = str(d)
                y, m_dec = s.split('.')
                m = int(m_dec.ljust(2, '0'))
                return pd.Timestamp(year=int(y), month=m, day=1)
            except: return pd.NaT
        shiller_raw['Date_dt'] = shiller_raw['Date'].apply(parse_shiller_date)
        shiller_raw = shiller_raw.dropna(subset=['Date_dt']).set_index('Date_dt')
        shiller_raw['EY_Shiller'] = (shiller_raw['E'] / shiller_raw['P']) * 100
        df = df.join(shiller_raw[['CAPE', 'EY_Shiller']], how='left').ffill()
    except: pass

    return df

# --- 3. CALCULATIONS ---
df = get_master_data()

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

if not df.empty:
    df['VIX_SMA14'] = df['VIX'].rolling(14).mean()
    df['Net_Liq'] = df['Fed_Assets'] - (df['TGA'].fillna(0) + df['RRP'].fillna(0))
    df['CPI_YoY'] = df['CPI'].pct_change(252) * 100
    df['SP500_SMA200'] = df['SP500'].rolling(200).mean()
    df['SP500_SMA50'] = df['SP500'].rolling(50).mean()
    df['Funding_Stress'] = (df['SOFR'] - df['TGCR']).interpolate().ffill() * 100
    df['HY_Z'] = (df['HY_Spread'] - df['HY_Spread'].rolling(756).mean()) / df['HY_Spread'].rolling(756).std()
    df['Buffett_v2'] = (df['W5000'] / (df['GDP'].ffill() + df['Net_Liq'] / 1000))
    df['SMA_Spread'] = df['SP500_SMA50'] - df['SP500_SMA200']
    
    # Margin & Breadth
    df['Margin_Market_Ratio'] = (df['Margin_Debt'] / df['W5000'].rolling(30).mean()).ffill()
    df['Margin_Ratio_Z'] = (df['Margin_Market_Ratio'] - df['Margin_Market_Ratio'].rolling(756).mean()) / df['Margin_Market_Ratio'].rolling(756).std()
    df['RSP_Ratio'] = df['RSP'] / df['SP500']
    df['Breadth_Rapid'] = ((df['RSP_Ratio'].rolling(5).mean() / df['RSP_Ratio'].rolling(20).mean()) - 1) * 100
    
    # Bear shading
    bear_episodes = calculate_bear_markets(df['SP500'])

    # --- ALLOCATION STRATEGY ---
    exit_trigger = (df['SP500'] < df['SP500_SMA200']) | (df['VIX'] > 30)
    reentry_trigger = (df['SP500'] > df['SP500_SMA200']) & (df['VIX'] < 25)
    
    allocations, curr_alloc = [], 100.0
    # FIX: Changed 'M' to 'ME' for Pandas 3.0
    last_days = df.resample('ME').last().index 
    
    for i in range(len(df)):
        target = 10.0 if exit_trigger.iloc[i] else (100.0 if reentry_trigger.iloc[i] else curr_alloc)
        if df.index[i] in last_days:
            if curr_alloc > target: curr_alloc = max(target, curr_alloc - 20)
            elif curr_alloc < target: curr_alloc = min(target, curr_alloc + 20)
        allocations.append(curr_alloc)
    
    df['Allocation_Pct'] = allocations
    df['Strategy_Cum'] = (1 + (df['SP500'].pct_change().fillna(0) * df['Allocation_Pct']/100)).cumprod()
    df['SPY_Cum'] = (1 + df['SP500'].pct_change().fillna(0)).cumprod()

    # --- 4. PERIOD SELECTION ---
    timeline = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS').tolist()
    start_s, end_s = st.select_slider("Select Period", options=timeline, value=(timeline[-120], timeline[-1]), format_func=lambda x: x.strftime('%Y'))
    p_df = df.truncate(before=start_s, after=end_s)

    # --- 5. THE 21-PANEL PLOTTING MACHINE ---
    fig, axes = plt.subplots(nrows=21, ncols=1, figsize=(14, 84))
    
    def apply_style(ax, title, log=False):
        ax.set_title(title, loc='left', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        if log: ax.set_yscale('log')
        for start, end in bear_episodes:
            if end >= p_df.index[0]: ax.axvspan(start, end, color='gray', alpha=0.1)

    # 1. Price
    axes[0].plot(p_df.index, p_df['SP500'], color='black')
    axes[0].plot(p_df.index, p_df['SP500_SMA200'], color='red', ls='--')
    apply_style(axes[0], "S&P 500 & SMA 200", log=True)

    # 2. Strategy
    ax2_t = axes[1].twinx()
    axes[1].fill_between(p_df.index, p_df['Allocation_Pct'], 0, color='blue', alpha=0.05)
    ax2_t.plot(p_df.index, p_df['Strategy_Cum'], color='navy', label='Tactical')
    ax2_t.plot(p_df.index, p_df['SPY_Cum'], color='gray', alpha=0.5, label='Buy & Hold')
    apply_style(axes[1], "Strategy Performance & Exposure")

    # 3. Allocation Line
    axes[2].plot(p_df.index, p_df['Allocation_Pct'], color='blue')
    apply_style(axes[2], "Target Equity Allocation %")

    # 4. Leverage Proxy
    axes[3].plot(p_df.index, p_df['Margin_Market_Ratio'], color='purple')
    ax3_t = axes[3].twinx()
    ax3_t.plot(p_df.index, p_df['Margin_Ratio_Z'], color='red', alpha=0.3)
    apply_style(axes[3], "Margin Debt / Market Cap")

    # 5. VIX
    axes[4].plot(p_df.index, p_df['VIX'], color='orange')
    axes[4].axhline(30, color='red', ls='--')
    apply_style(axes[4], "VIX Volatility Index")

    # 6. Breadth
    axes[5].plot(p_df.index, p_df['Breadth_Rapid'], color='green')
    axes[5].axhline(0, color='black', lw=0.5)
    apply_style(axes[5], "Breadth Momentum (RSP/SPY)")

    # 7. Net Liquidity
    axes[6].plot(p_df.index, p_df['Net_Liq'], color='teal')
    apply_style(axes[6], "Fed Net Liquidity (Assets - TGA - RRP)")

    # 8. HY Spread
    axes[7].plot(p_df.index, p_df['HY_Spread'], color='brown')
    apply_style(axes[7], "High Yield Corporate Spread")

    # 9. Yield Curve
    axes[8].plot(p_df.index, p_df['Yield_Curve_2s10s'], color='darkblue')
    axes[8].axhline(0, color='red', ls='-')
    apply_style(axes[8], "Yield Curve (10Y - 2Y)")

    # 10. Real Yields
    axes[9].plot(p_df.index, p_df['Real_10Y_Yield'], color='magenta')
    apply_style(axes[9], "US 10Y Real Yield")

    # 11. USD Index
    axes[10].plot(p_df.index, p_df['USD_Index'], color='darkgreen')
    apply_style(axes[10], "US Dollar Index (DXY)")

    # 12. Funding Stress
    axes[11].plot(p_df.index, p_df['Funding_Stress'], color='red')
    apply_style(axes[11], "Repo Market Stress (SOFR - TGCR) bps")

    # 13. SMA Momentum
    axes[12].fill_between(p_df.index, p_df['SMA_Spread'], 0, where=(p_df['SMA_Spread']>0), color='green', alpha=0.3)
    axes[12].fill_between(p_df.index, p_df['SMA_Spread'], 0, where=(p_df['SMA_Spread']<0), color='red', alpha=0.3)
    apply_style(axes[12], "SMA Momentum (50d - 200d)")

    # 14. Buffett Indicator
    axes[13].plot(p_df.index, p_df['Buffett_v2'], color='darkgoldenrod')
    apply_style(axes[13], "Buffett Indicator v2 (MCap / GDP+Liq)")

    # 15. CAPE Ratio
    if 'CAPE' in p_df.columns:
        axes[14].plot(p_df.index, p_df['CAPE'], color='darkred')
        apply_style(axes[14], "Shiller CAPE Ratio")

    # 16. M2 Growth
    axes[15].plot(p_df.index, p_df['M2'].pct_change(252)*100, color='blue')
    apply_style(axes[15], "M2 Money Supply YoY %")

    # 17. CPI YoY
    axes[16].plot(p_df.index, p_df['CPI_YoY'], color='orange')
    apply_style(axes[16], "Inflation CPI YoY %")

    # 18. VALUATION PANEL (MacroMicro Integration)
    ax18 = axes[17]
    ey_df = get_macromicro_ey_robust()
    if not ey_df.empty:
        # Reindexing to match the plot period
        p_df['Macro_EY'] = ey_df['EY'].reindex(p_df.index, method='ffill')
        ax18.plot(p_df.index, p_df['Macro_EY'], color='royalblue', label='Earnings Yield %')
        ax18.plot(p_df.index, p_df['Fed_10Y'], color='darkorange', label='10Y Treasury %')
        ax18.fill_between(p_df.index, p_df['Macro_EY'], p_df['Fed_10Y'], 
                          where=(p_df['Macro_EY'] > p_df['Fed_10Y']), color='green', alpha=0.15)
        ax18.legend(loc='upper left')
    apply_style(ax18, "Earnings Yield vs 10Y Treasury (Valuation)")

    # 19. EURUSD
    axes[18].plot(p_df.index, p_df['EURUSD'], color='navy')
    apply_style(axes[18], "EUR/USD Exchange Rate")

    # 20. Fed Assets
    axes[19].plot(p_df.index, p_df['Fed_Assets'], color='black')
    apply_style(axes[19], "Federal Reserve Total Assets")

    # 21. VIX SMA
    axes[20].plot(p_df.index, p_df['VIX_SMA14'], color='orange')
    apply_style(axes[20], "VIX 14-Day SMA")

    # Formatting Dates
    visible_days = (p_df.index[-1] - p_df.index[0]).days
    if visible_days <= 730:
        major_loc = mdates.MonthLocator()
        date_fmt = mdates.DateFormatter('%b %Y')
    else:
        major_loc = mdates.YearLocator()
        date_fmt = mdates.DateFormatter('%Y')

    for ax in axes:
        ax.xaxis.set_major_locator(major_loc)
        ax.xaxis.set_major_formatter(date_fmt)

    plt.tight_layout()
    st.pyplot(fig)