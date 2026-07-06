"""Central defaults for the screening and backtest scripts.

I keep configuration in one small module rather than a YAML file so the whole
project has no config-parser dependency and every default is type-checked where
it is used. Scripts import DEFAULTS and let argparse override individual fields.
"""

from __future__ import annotations

# A compact, liquid universe of names that tend to move in related pairs.
# I keep this deliberately small so the O(n^2) screen stays cheap; swap in your
# own tickers freely. Cointegration is decided on data, not on this list.
UNIVERSE = [
    # US majors / ETFs that commonly cointegrate within a theme
    "KO", "PEP",        # beverages
    "XOM", "CVX",       # integrated oil
    "MA", "V",          # payments
    "HD", "LOW",        # home improvement
    "GS", "MS",         # bulge-bracket banks
    "EWA", "EWC",       # Australia / Canada country ETFs (classic Chan pair)
    "GLD", "GDX",       # gold vs gold miners
    "SPY", "IVV",       # near-identical S&P trackers (sanity pair)
]

DEFAULTS = dict(
    # Data
    start="2015-01-01",
    end="2024-12-31",
    # Formation / trading split: the fraction of the sample used only for
    # screening and never traded on.
    formation_frac=0.5,
    # Screening thresholds
    min_corr=0.80,
    max_pvalue=0.05,
    run_johansen=True,
    # Kalman noise
    delta=1e-4,
    r=1e-3,
    # Signal
    zscore_window=60,
    entry=2.0,
    exit=0.5,
    stop=4.0,
    # Backtest
    cost_bps=1.0,
    rebalance_hedge=True,
    execution_lag=1,
    periods_per_year=252,
)
