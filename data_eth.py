"""
Ethereum data module for the fees-vs-price extension.

Mirrors data.py but targets Ether. blockchain.com is Bitcoin-only, so we use the
CoinMetrics community API, which exposes daily total fees in native units (ETH),
USD price, and transaction count with no API key.

The econometric wrappers in econometrics.py are asset-agnostic, so once this
module produces an analysis-ready DataFrame with d_log_* columns, the same
ec.* functions used for Bitcoin apply unchanged.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import requests

# Reuse the generic transforms from the Bitcoin module (they take a column list
# and are not asset-specific).
import data


# ============================================================================
# CONSTANTS
# ============================================================================

COINMETRICS_BASE = 'https://community-api.coinmetrics.io/v4/timeseries/asset-metrics'

# Metrics to request. FeeTotNtv = total fees in native units (ETH) -> dependent.
# PriceUSD = price. TxCnt = transaction count (demand proxy).
CM_METRICS = {
    'PriceUSD':  'price_usd',     # ETH price in USD (main regressor)
    'FeeTotNtv': 'fees_eth',      # total daily fees in ETH (dependent variable)
    'TxCnt':     'n_transactions',# daily transaction count (demand / congestion proxy)
}

# Structural-break candidates for Ethereum (the analogues of BTC halvings).
# EIP-1559 (London) overhauled the fee market; the Merge switched to PoS.
EIP1559_DATE = '2021-08-05'
MERGE_DATE   = '2022-09-15'
ETH_EVENTS   = [EIP1559_DATE, MERGE_DATE]

CACHE_DIR = 'data/raw'


# ============================================================================
# FETCHING
# ============================================================================

def fetch_coinmetrics(asset: str = 'eth',
                      metrics: Optional[dict] = None,
                      use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch daily metrics for `asset` from the CoinMetrics community API.

    Returns a DataFrame with a DatetimeIndex and the friendly column names
    from CM_METRICS. Handles pagination via the API's next_page_url.
    """
    if metrics is None:
        metrics = CM_METRICS

    cache_path = os.path.join(CACHE_DIR, f'coinmetrics_{asset}.csv')
    if use_cache and os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col='date', parse_dates=['date'])

    params = {
        'assets':     asset,
        'metrics':    ','.join(metrics.keys()),
        'frequency':  '1d',
        'page_size':  '10000',
    }

    rows = []
    url = COINMETRICS_BASE
    while url:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        rows.extend(payload.get('data', []))
        # The API returns a full next_page_url when more pages exist.
        url = payload.get('next_page_url')
        params = None  # next_page_url already encodes the params

    if not rows:
        raise RuntimeError('CoinMetrics returned no data; check asset/metrics.')

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['time']).dt.tz_localize(None).dt.normalize()
    df = df.set_index('date').sort_index()

    # Keep and rename only the requested metric columns; coerce to numeric.
    keep = {k: v for k, v in metrics.items() if k in df.columns}
    df = df[list(keep.keys())].rename(columns=keep)
    df = df.apply(pd.to_numeric, errors='coerce')

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_csv(cache_path)

    return df


def fetch_yfinance_eth(start: Optional[str] = None,
                       end: Optional[str] = None) -> pd.DataFrame:
    """ETH-USD daily close from Yahoo Finance, for the cross-source price check."""
    import yfinance as yf
    raw = yf.download('ETH-USD', start=start, end=end,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    out = raw[['Close']].rename(columns={'Close': 'price_usd_yf'})
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index.name = 'date'
    return out


def compare_price_sources(df_cm: pd.DataFrame,
                          df_yf: pd.DataFrame) -> dict:
    """Compare CoinMetrics ETH price vs Yahoo Finance (same logic as data.py)."""
    cm = df_cm[['price_usd']].rename(columns={'price_usd': 'price_cm'})
    yf_ = df_yf.rename(columns={'price_usd_yf': 'price_yf'})
    merged = cm.join(yf_, how='inner').dropna()
    diff = merged['price_cm'] - merged['price_yf']
    return {
        'correlation':       merged['price_cm'].corr(merged['price_yf']),
        'mean_abs_diff':     diff.abs().mean(),
        'rmse':              np.sqrt((diff ** 2).mean()),
        'mean_abs_pct_diff': (diff.abs() / merged['price_yf']).mean() * 100,
        'n_obs':             len(merged),
        'merged':            merged,
    }


# ============================================================================
# CLEANING & TRANSFORMS
# ============================================================================

def clean(df: pd.DataFrame, drop_before: Optional[str] = '2016-01-01',
          ffill_limit: int = 2) -> pd.DataFrame:
    """
    Clean the Ethereum master frame. Mirrors data.clean but keys on fees_eth.
    Ethereum launched mid-2015; we drop the sparse first months by default.
    """
    df = df.copy()
    if drop_before is not None:
        df = df[df.index >= drop_before]

    explanatory = [c for c in df.columns if c != 'fees_eth']
    df[explanatory] = df[explanatory].ffill(limit=ffill_limit)

    df = df.dropna(how='all')
    df = df[df['fees_eth'].notna() & (df['fees_eth'] > 0)]
    return df


def split_by_events(df: pd.DataFrame,
                    events: Optional[list] = None) -> dict:
    """
    Split into eras bounded by the Ethereum protocol events (EIP-1559, Merge),
    analogous to data.split_by_halving. Returns dict of era label -> sub-frame.
    """
    if events is None:
        events = ETH_EVENTS

    cuts = [df.index.min()] + [pd.Timestamp(d) for d in events] + [df.index.max()]
    labels = ['pre_eip1559', 'eip1559_to_merge', 'post_merge']

    eras = {}
    for i in range(len(cuts) - 1):
        lo, hi = cuts[i], cuts[i + 1]
        if i == len(cuts) - 2:
            mask = (df.index >= lo) & (df.index <= hi)
        else:
            mask = (df.index >= lo) & (df.index < hi)
        eras[f'era_{i+1}_{labels[i]}'] = df.loc[mask].copy()
    return eras


# ============================================================================
# ORCHESTRATOR
# ============================================================================

def build_master_dataset(use_cache: bool = True,
                         include_yfinance: bool = True) -> pd.DataFrame:
    """
    Fetch -> clean -> add log and log-difference transforms for Ether.
    Produces the same d_log_* column convention as data.build_master_dataset.
    """
    df = fetch_coinmetrics('eth', use_cache=use_cache)
    df = clean(df)

    level_cols = ['fees_eth', 'price_usd', 'n_transactions']
    level_cols = [c for c in level_cols if c in df.columns]

    # Reuse the generic transforms from the Bitcoin module.
    df = data.add_log_transforms(df, level_cols)
    log_cols = [f'log_{c}' for c in level_cols]
    df = data.add_differences(df, log_cols)

    if include_yfinance:
        start = df.index.min().strftime('%Y-%m-%d')
        df = df.join(fetch_yfinance_eth(start=start), how='left')

    return df