"""Price pipeline: adjusted closes from Yahoo, cached to a parquet mirror."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(os.environ.get("STATARB_CACHE", "data"))


def _cache_path(start: str, end: str) -> Path:
    return CACHE_DIR / f"prices_{start}_{end}.parquet"


def load_prices(tickers: list[str], start: str, end: str,
                use_cache: bool = True) -> pd.DataFrame:
    """Wide DataFrame of adjusted closes. Caches by date window so reruns are
    reproducible and offline. Drops any ticker missing >5% of observations."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(start, end)

    if use_cache and path.exists():
        cached = pd.read_parquet(path)
        if set(tickers).issubset(cached.columns):
            return cached[tickers]

    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close[tickers] if set(tickers).issubset(close.columns) else close

    close = close.sort_index().ffill(limit=1)
    close = close[close.columns[close.isna().mean() < 0.05]].dropna(how="any")
    close.to_parquet(path)
    return close


def align_pair(prices: pd.DataFrame, y: str, x: str) -> pd.DataFrame:
    """Two-column, gap-free log-price frame for one pair. I trade the log spread
    so the hedge ratio is a unitless elasticity and the scale stays stable."""
    import numpy as np

    return prices[[y, x]].dropna().apply(np.log)
