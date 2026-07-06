"""Backtest one pair out-of-sample with the Kalman hedge and rolling-z signal.

Usage:
    python scripts/run_backtest.py --y KO --x PEP
    python scripts/run_backtest.py --y EWA --x EWC --cost-bps 2 --no-rehedge

The filter runs over the whole sample so the hedge ratio is warm by the time the
trading window starts, but performance is only reported on the held-out trading
half -- the same split screen_pairs.py used. I print the scorecard and, if
matplotlib is available, save spread/z-score/equity plots next to the repo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DEFAULTS, UNIVERSE
from src.data import load_prices, align_pair
from src.kalman import KalmanHedge
from src.signals import SignalConfig, generate_positions, rolling_zscore
from src.backtest import BacktestConfig, run_backtest
from src.metrics import summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--y", required=True, help="dependent-leg ticker")
    p.add_argument("--x", required=True, help="hedge-leg ticker")
    p.add_argument("--start", default=DEFAULTS["start"])
    p.add_argument("--end", default=DEFAULTS["end"])
    p.add_argument("--formation-frac", type=float, default=DEFAULTS["formation_frac"])
    p.add_argument("--delta", type=float, default=DEFAULTS["delta"])
    p.add_argument("--r", type=float, default=DEFAULTS["r"])
    p.add_argument("--zscore-window", type=int, default=DEFAULTS["zscore_window"])
    p.add_argument("--entry", type=float, default=DEFAULTS["entry"])
    p.add_argument("--exit", type=float, default=DEFAULTS["exit"])
    p.add_argument("--stop", type=float, default=DEFAULTS["stop"])
    p.add_argument("--cost-bps", type=float, default=DEFAULTS["cost_bps"])
    p.add_argument("--no-rehedge", action="store_true",
                   help="freeze the hedge ratio at entry instead of daily re-hedging")
    p.add_argument("--plots", action="store_true", help="save diagnostic plots")
    args = p.parse_args()

    tickers = sorted({args.y, args.x} | set(UNIVERSE))
    prices = load_prices(tickers, args.start, args.end)
    logp = align_pair(prices, args.y, args.x)

    # Filter on the full sample so the hedge ratio is warm at the trading split.
    res = KalmanHedge(delta=args.delta, r=args.r).filter(logp[args.y], logp[args.x])
    z = rolling_zscore(res.spread, window=args.zscore_window)
    pos = generate_positions(z, SignalConfig(args.entry, args.exit, args.stop))

    bt = run_backtest(
        logp, args.y, args.x, pos, res.beta,
        BacktestConfig(cost_bps=args.cost_bps,
                       rebalance_hedge=not args.no_rehedge),
    )

    # Report only on the held-out trading window.
    split = int(len(logp) * args.formation_frac)
    trade_idx = logp.index[split:]
    ret_oos = bt.returns.loc[trade_idx]
    eq_oos = (1.0 + ret_oos).cumprod()
    stats = summary(ret_oos, eq_oos, bt.positions.loc[trade_idx],
                    bt.turnover.loc[trade_idx], DEFAULTS["periods_per_year"])

    print(f"\nPair {args.y}/{args.x}  out-of-sample "
          f"({trade_idx[0].date()} -> {trade_idx[-1].date()}, {len(trade_idx)} days)")
    print("-" * 52)
    for k, v in stats.items():
        print(f"  {k:14s} {v:12.4f}" if isinstance(v, float) else f"  {k:14s} {v:12d}")

    if args.plots:
        _save_plots(args.y, args.x, res, z, eq_oos, trade_idx)


def _save_plots(y, x, res, z, eq_oos, trade_idx) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; skipping plots")
        return

    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
    res.beta.plot(ax=ax[0], title=f"Kalman hedge ratio {y} ~ {x}")
    ax[0].set_ylabel("beta_t")
    z.loc[trade_idx].plot(ax=ax[1], title="Rolling z-score (out-of-sample)")
    for lvl in (-2, 0, 2):
        ax[1].axhline(lvl, color="grey", lw=0.6, ls="--")
    eq_oos.plot(ax=ax[2], title="Out-of-sample equity (unit gross, net of costs)")
    ax[2].set_ylabel("equity")
    fig.tight_layout()
    out = f"backtest_{y}_{x}.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved plots -> {out}")


if __name__ == "__main__":
    main()
