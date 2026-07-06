"""Screen the configured universe for cointegrated pairs on the formation window.

Usage:
    python scripts/screen_pairs.py                 # use config defaults
    python scripts/screen_pairs.py --no-johansen   # skip the Johansen cross-check
    python scripts/screen_pairs.py --out pairs.csv

I split the sample into a formation and a trading half up front and screen only
on the formation half, so the pairs I carry forward have never seen the data
they will be traded on. The output is a ranked table written to CSV that
run_backtest.py can consume.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DEFAULTS, UNIVERSE
from src.data import load_prices
from src.cointegration import screen_pairs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default=DEFAULTS["start"])
    p.add_argument("--end", default=DEFAULTS["end"])
    p.add_argument("--formation-frac", type=float, default=DEFAULTS["formation_frac"])
    p.add_argument("--min-corr", type=float, default=DEFAULTS["min_corr"])
    p.add_argument("--no-johansen", action="store_true")
    p.add_argument("--out", default="pairs.csv")
    args = p.parse_args()

    prices = load_prices(UNIVERSE, args.start, args.end)
    split = int(len(prices) * args.formation_frac)
    formation = prices.iloc[:split]
    print(f"Loaded {prices.shape[1]} tickers, {len(prices)} rows; "
          f"formation window = {len(formation)} rows "
          f"({formation.index[0].date()} -> {formation.index[-1].date()})")

    table = screen_pairs(
        formation,
        min_corr=args.min_corr,
        max_pvalue=DEFAULTS["max_pvalue"],
        run_johansen=not args.no_johansen,
    )
    if table.empty:
        print("No pairs cleared the correlation pre-filter.")
        return

    table.to_csv(args.out, index=False)
    passing = table[table["passes"]]
    print(f"\nScreened {len(table)} pairs; {len(passing)} pass all criteria.\n")
    cols = ["y", "x", "corr", "eg_pvalue", "eg_hedge_ratio", "half_life",
            "johansen_trace_ok", "passes"]
    print(table[cols].to_string(index=False,
                                float_format=lambda v: f"{v:.4f}"))
    print(f"\nWrote ranked table -> {args.out}")


if __name__ == "__main__":
    main()
