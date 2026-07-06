"""Synthetic-data checks for the parts that carry correctness risk.

These use only numpy/pandas so they run without market data or statsmodels.
I verify three things:

  1. the Kalman filter recovers a known, drifting hedge ratio;
  2. the backtest has no look-ahead (past returns are invariant to a future
     shuffle of prices);
  3. the backtest PnL matches a hand-computed value on a tiny fixture.

Run with `pytest -q` or execute directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.kalman import KalmanHedge
from src.signals import SignalConfig, generate_positions, rolling_zscore
from src.backtest import BacktestConfig, run_backtest


def _synthetic_pair(n=1500, seed=0):
    """Cointegrated log-price pair with a hedge ratio that drifts 1.0 -> 1.6.

    x is a random walk; y = beta_t * x + stationary noise, so (y - beta_t x) is
    stationary by construction and beta_t is the thing the filter should track.
    """
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 0.01, n)) + 4.0            # log price of x
    beta_true = np.linspace(1.0, 1.6, n)
    noise = np.zeros(n)
    for t in range(1, n):                                   # AR(1) stationary spread
        noise[t] = 0.95 * noise[t - 1] + rng.normal(0, 0.01)
    y = beta_true * x + noise
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return (pd.Series(y, idx, name="Y"),
            pd.Series(x, idx, name="X"),
            pd.Series(beta_true, idx, name="beta_true"))


def test_kalman_recovers_drifting_beta():
    y, x, beta_true = _synthetic_pair()
    res = KalmanHedge(delta=1e-3, r=1e-3).filter(y, x)
    # Compare after a burn-in; correlation of the filtered path with the truth
    # and terminal-level accuracy are both meaningful here.
    warm = 250
    corr = np.corrcoef(res.beta.iloc[warm:], beta_true.iloc[warm:])[0, 1]
    end_err = abs(res.beta.iloc[-1] - beta_true.iloc[-1])
    print(f"[kalman] beta corr={corr:.3f}  terminal_abs_err={end_err:.3f}")
    assert corr > 0.9, corr
    assert end_err < 0.1, end_err


def test_backtest_has_no_lookahead():
    """Positions come from the z-score, which is causal; the realised return for
    day t must not depend on prices after t. I check that perturbing the final
    price leaves every return except the last one unchanged."""
    y, x, _ = _synthetic_pair(seed=1)
    logp = pd.concat([y, x], axis=1)
    res = KalmanHedge(delta=1e-3, r=1e-3).filter(y, x)
    pos = generate_positions(rolling_zscore(res.spread, window=60), SignalConfig())

    bt1 = run_backtest(logp, "Y", "X", pos, res.beta, BacktestConfig())

    logp2 = logp.copy()
    logp2.iloc[-1, 0] += 0.5                                # shock the last y
    bt2 = run_backtest(logp2, "Y", "X", pos, res.beta, BacktestConfig())

    diff = (bt1.returns.iloc[:-2] - bt2.returns.iloc[:-2]).abs().max()
    print(f"[lookahead] max |return diff| before the shock = {diff:.2e}")
    assert diff < 1e-12, diff


def test_backtest_pnl_matches_hand_calc():
    """Three-day fixture with a fixed unit position and no costs. I hold the
    spread long on day 0 (weights effective day 1) and check day-1 return equals
    w_y*r_y + w_x*r_x by hand."""
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    y = pd.Series(np.log([100.0, 102.0, 102.0]), idx, name="Y")
    x = pd.Series(np.log([50.0, 50.0, 50.0]), idx, name="X")
    logp = pd.concat([y, x], axis=1)

    pos = pd.Series([1.0, 1.0, 0.0], idx)                   # long spread days 0-1
    beta = pd.Series([2.0, 2.0, 2.0], idx)                  # fixed hedge ratio

    cfg = BacktestConfig(cost_bps=0.0, rebalance_hedge=True, execution_lag=1)
    bt = run_backtest(logp, "Y", "X", pos, beta, cfg)

    # Day-0 weights: gross = 1+2 = 3; w_y=1/3, w_x=-2/3. Effective on day 1.
    ry1 = np.log(102.0) - np.log(100.0)                     # y moved, x flat
    rx1 = 0.0
    expected = (1 / 3) * ry1 + (-2 / 3) * rx1
    got = bt.returns.iloc[1]
    print(f"[pnl] expected={expected:.6f}  got={got:.6f}")
    assert abs(got - expected) < 1e-12, (got, expected)


if __name__ == "__main__":
    test_kalman_recovers_drifting_beta()
    test_backtest_has_no_lookahead()
    test_backtest_pnl_matches_hand_calc()
    print("\nAll synthetic checks passed.")
