"""
Data fetching and cleaning for the Bitcoin transaction fees analysis.

Pulls daily time series from blockchain.com, cross-checks with Yahoo Finance,
and produces a clean master DataFrame ready for econometric analysis.
"""

import os
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ============================================================================
# CONSTANTS
# ============================================================================

HALVING_DATES = [
    '2012-11-28',
    '2016-07-09',
    '2020-05-11',
    '2024-04-19',
]

BLOCKCHAIN_API_BASE = 'https://api.blockchain.info/charts'

# blockchain.com chart names → friendly column names
BLOCKCHAIN_CHARTS = {
    'transaction-fees':     'fees_btc',         # total daily fees in BTC (Y variable)
    'transaction-fees-usd': 'fees_usd',         # for sanity-checking
    'market-price':         'price_usd',        # BTC/USD (main X)
    'mempool-size':         'mempool_bytes',    # congestion proxy
    'mempool-count':        'mempool_count',
    'n-transactions':       'n_transactions',
    'avg-block-size':       'avg_block_size',
    'hash-rate':            'hash_rate',
    'difficulty':           'difficulty',
}

CACHE_DIR = 'data/raw'


# ============================================================================
# FETCHING
# ============================================================================

def fetch_blockchain_chart(chart_name: str, timespan: str = 'all',
                            use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch a single chart from blockchain.com's charts API.

    URL pattern: {BLOCKCHAIN_API_BASE}/{chart_name}?timespan={timespan}&format=csv
    Returns a DataFrame with DatetimeIndex and one column named `chart_name`.
    """
    cache_path = os.path.join(CACHE_DIR, f'{chart_name}.csv')

    # 1. Load from cache if the FILE exists
    if use_cache and os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col='date', parse_dates=['date'])

    # 2. Fetch from the API
    url = f'{BLOCKCHAIN_API_BASE}/{chart_name}'
    params = {'timespan': timespan, 'format': 'csv', 'sampled': 'false'}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    # 3. Parse — the CSV has NO header, just timestamp,value rows
    df = pd.read_csv(
    StringIO(response.text),
    header=None,
    names=['date', chart_name],
    parse_dates=['date'],
    )
    df = df.set_index('date')
    df = df.resample('D').mean().ffill()

    # 4. Save to cache — directory in makedirs, FILE in to_csv
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_csv(cache_path)

    # 5. Return
    return df

def fetch_all_blockchain_charts(charts: Optional[dict] = None,
                                 use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch every chart in BLOCKCHAIN_CHARTS and outer-merge them on date.

    Returns one DataFrame with the friendly column names from BLOCKCHAIN_CHARTS.
    """
    if charts is None:
        charts = BLOCKCHAIN_CHARTS

    merged = None
    for chart_name, friendly_name in charts.items():
        df = fetch_blockchain_chart(chart_name, use_cache=use_cache)
        df = df.rename(columns={chart_name: friendly_name})

        if merged is None:
            merged = df
        else:
            merged = merged.join(df, how='outer')   # join on the DatetimeIndex

    merged = merged.sort_index()
    return merged


def fetch_yfinance_btc(start: Optional[str] = None,
                       end: Optional[str] = None) -> pd.DataFrame:
    """
    Pull BTC-USD daily closing prices from Yahoo Finance for the cross-source check.

    Returns DataFrame with DatetimeIndex and column 'price_usd_yf'.
    """
    raw = yf.download('BTC-USD', start=start, end=end,
                      auto_adjust=True, progress=False)

    # Recent yfinance versions return MultiIndex columns like ('Close', 'BTC-USD').
    # Flatten to just the price level if that's the case.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Keep only the closing price, rename it
    df = raw[['Close']].rename(columns={'Close': 'price_usd_yf'})

    # Drop timezone if present (tz_localize returns a NEW index — must reassign)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df.index.name = 'date'
    return df


def compare_price_sources(df_blockchain: pd.DataFrame,
                           df_yfinance: pd.DataFrame) -> dict:
    """
    Compare blockchain.com price vs Yahoo Finance price.

    Returns dict with 'correlation', 'mean_abs_diff', 'rmse', and 'merged'
    (the merged DataFrame, useful for plotting in the notebook).
    """
    # Pull just the price column from the blockchain frame
    bc = df_blockchain[['price_usd']].rename(columns={'price_usd': 'price_bc'})
    yf_ = df_yfinance.rename(columns={'price_usd_yf': 'price_yf'})

    # Inner-merge on the date index → only dates present in BOTH sources
    merged = bc.join(yf_, how='inner').dropna()

    diff = merged['price_bc'] - merged['price_yf']

    # Pearson correlation between the two series
    correlation = merged['price_bc'].corr(merged['price_yf'])

    # Absolute error metrics (in USD)
    mean_abs_diff = diff.abs().mean()
    rmse = np.sqrt((diff ** 2).mean())

    # Relative error (%) — more meaningful given the huge price range
    mean_abs_pct_diff = (diff.abs() / merged['price_yf']).mean() * 100

    return {
        'correlation': correlation,
        'mean_abs_diff': mean_abs_diff,
        'rmse': rmse,
        'mean_abs_pct_diff': mean_abs_pct_diff,
        'n_obs': len(merged),
        'merged': merged,
    }


# ============================================================================
# CLEANING & TRANSFORMS
# ============================================================================

def clean(df: pd.DataFrame, drop_before: Optional[str] = '2011-01-01', ffill_limit: int = 2) -> pd.DataFrame:
    """
    Basic cleaning of the master DataFrame.

    - Drop very early sparse data (default: pre-2011).
    - Forward-fill small gaps; drop rows that are still all-NaN.
    - Drop rows where the dependent variable (fees_btc) is zero or missing.
    """
    df = df.copy()                          # never mutate the caller's frame

    # 1. Drop early sparse data
    if drop_before is not None:
        df = df[df.index >= drop_before]

    # 2. Forward-fill SMALL gaps in explanatory variables only.
    #    `limit=ffill_limit` stops us from carrying a value across a long blackout.
    #    We deliberately exclude fees_btc — see docstring.
    explanatory = [c for c in df.columns if c != 'fees_btc']
    df[explanatory] = df[explanatory].ffill(limit=ffill_limit)

    # 3. Drop rows that are entirely NaN (e.g. dates before any series starts)
    df = df.dropna(how='all')

    # 4. Drop rows where the dependent variable is missing or zero.
    #    Zero daily fees only occur in the earliest, near-empty days — not real data.
    df = df[df['fees_btc'].notna() & (df['fees_btc'] > 0)]

    return df


def add_log_transforms(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Add log-transformed columns (prefix 'log_').

    Replaces zeros with NaN before taking the log to avoid -inf.
    """
    df = df.copy()
    for col in cols:
        df[f'log_{col}'] = np.log(df[col].replace(0, np.nan))
    return df


def add_differences(df: pd.DataFrame, cols: list, periods: int = 1) -> pd.DataFrame:
    """
    Add first-differenced columns (prefix 'd_').

    Useful once you've established the levels are non-stationary.
    """
    df = df.copy()
    for col in cols:
        df[f'd_{col}'] = df[col].diff(periods=periods)
    return df


def split_by_halving(df: pd.DataFrame,
                      halving_dates: Optional[list] = None) -> dict:
    """
    Split DataFrame into 5 sub-periods bounded by the halving dates.

    Returns dict mapping era label (e.g. 'era_1_pre_2012') to sub-DataFrame.
    """
    if halving_dates is None:
        halving_dates = HALVING_DATES

    # Build boundary list: [start_of_data, halving_1, ..., halving_n, end_of_data]
    cuts = [df.index.min()] + [pd.Timestamp(d) for d in halving_dates] + [df.index.max()]

    eras = {}
    for i in range(len(cuts) - 1):
        lo, hi = cuts[i], cuts[i + 1]

        # Half-open intervals [lo, hi) so a halving date belongs to the era it opens,
        # and no row is double-counted at the boundaries.
        # The final era is closed on the right to include the last day.
        if i == len(cuts) - 2:
            mask = (df.index >= lo) & (df.index <= hi)
        else:
            mask = (df.index >= lo) & (df.index < hi)

        # Label with the year of the upper halving boundary for readability
        if i == 0:
            label = f'era_{i+1}_pre_{halving_dates[0][:4]}'
        elif i == len(cuts) - 2:
            label = f'era_{i+1}_post_{halving_dates[-1][:4]}'
        else:
            label = f'era_{i+1}_{halving_dates[i-1][:4]}_{halving_dates[i][:4]}'

        eras[label] = df.loc[mask].copy()

    return eras


# ============================================================================
# ORCHESTRATOR
# ============================================================================

def build_master_dataset(use_cache: bool = True,
                          include_yfinance: bool = True) -> pd.DataFrame:
    """
    Top-level convenience function: fetch → clean → add transforms.

    Call this from the notebook to get an analysis-ready DataFrame in one line.
    """
    # TODO: fetch_all_blockchain_charts → clean → add_log_transforms
    # → add_differences → optionally merge yfinance → return.
    # 1. Fetch and clean the blockchain.com data
    df = fetch_all_blockchain_charts(use_cache=use_cache)
    df = clean(df)

    # 2. Define which level variables get logged + differenced.
    #    These should be strictly positive series (prices, counts, sizes).
    level_cols = [
        'fees_btc',
        'price_usd',
        'mempool_bytes',
        'mempool_count',
        'n_transactions',
        'avg_block_size',
        'hash_rate',
        'difficulty',
    ]
    # Guard against any column not present (e.g. if you fetched a subset)
    level_cols = [c for c in level_cols if c in df.columns]

    # 3. Add log transforms, then log-differences (≈ log-returns)
    df = add_log_transforms(df, level_cols)
    log_cols = [f'log_{c}' for c in level_cols]
    df = add_differences(df, log_cols)

    # 4. Optionally merge the Yahoo Finance price for the cross-source check
    if include_yfinance:
        start = df.index.min().strftime('%Y-%m-%d')
        yf_df = fetch_yfinance_btc(start=start)
        df = df.join(yf_df, how='left')   # left join: keep blockchain.com as the spine

    return df