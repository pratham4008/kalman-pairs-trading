# Statistical Arbitrage — Kalman-Filter Pairs Trading

A statistical-arbitrage pairs-trading engine that estimates a **dynamic** hedge
ratio with a Kalman filter instead of a static OLS beta, screens for
cointegration with formal tests, and backtests out-of-sample with realistic
transaction costs.

The point of the project is rigour, not a headline Sharpe. Two things separate
it from the usual pairs-trading notebook:

1. **The hedge ratio adapts.** A cointegrating relationship drifts. I model the
   spread as a local linear regression whose coefficients follow a random walk
   and estimate them with a Kalman filter, so the hedge ratio tracks the
   relationship as it moves. A rolling-OLS hedge ratio is included as the
   baseline it has to beat.
2. **The evaluation is honest.** Pairs are screened only on a formation window
   and traded only on a held-out window. Signals are causal by construction, the
   position is lagged before it meets returns, and the daily re-hedging of the
   dynamic beta is charged for rather than assumed free. The correctness of the
   filter and the PnL accounting is verified on synthetic data.

## Method

The relationship between two log prices is modelled as

```
y_t = beta_t * x_t + alpha_t + eps_t        eps_t ~ N(0, R)
[beta_t, alpha_t] = [beta_{t-1}, alpha_{t-1}] + w_t     w_t ~ N(0, Q)
```

The state `[beta_t, alpha_t]` is filtered with a two-state Kalman filter (Chan
2013 formulation): random-walk transition, observation row `H_t = [x_t, 1]`, and
`Q = delta/(1-delta) * I` so a single `delta` controls how fast the hedge ratio
may move. Every quantity at time `t` uses only information up to `t`.

The tradable spread is the filter's one-step forecast error `e_t`. Rather than
standardise it by the filter's own predicted variance (which reflects parameter
uncertainty and rarely trips a fixed band), I take a **trailing rolling z-score**
of the spread level, since the trade is a bet on the level reverting. Entry,
exit and stop bands with hysteresis convert the z-score into `{-1, 0, +1}`
spread positions.

Screening uses a correlation pre-filter, then the **Engle-Granger** test
(residual ADF on the static-OLS spread), the **Johansen** trace test as a
multivariate cross-check, and the OU **half-life** to check reversion is fast
enough to trade.

## Repo layout

```
statarb-kalman-pairs/
├── config.py                 # universe and default parameters
├── src/
│   ├── data.py               # yfinance loader + parquet cache, log-space pairs
│   ├── cointegration.py      # Engle-Granger, Johansen, half-life, screening
│   ├── kalman.py             # dynamic hedge-ratio Kalman filter + OLS baseline
│   ├── signals.py            # rolling z-score + hysteresis state machine
│   ├── backtest.py           # causal, cost-aware, dollar-neutral backtest
│   └── metrics.py            # Sharpe, Sortino, drawdown, hit rate, turnover
├── scripts/
│   ├── screen_pairs.py       # screen the universe -> ranked CSV of pairs
│   └── run_backtest.py       # backtest one pair out-of-sample (+ optional plots)
├── notebooks/
│   └── walkthrough.ipynb     # end-to-end narrative walkthrough
├── results/                  # screen table and backtest output behind the Results section
└── tests/
    ├── test_synthetic.py     # Kalman recovery, no-look-ahead, PnL hand-check
    └── test_cointegration.py # EG separates cointegrated vs spurious pairs
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Screen the universe for cointegrated pairs (formation window only)
python scripts/screen_pairs.py --out results/pairs.csv

# 2. Backtest a chosen pair out-of-sample, with diagnostic plots
python scripts/run_backtest.py --y KO --x PEP --plots

# 3. Or work through the full narrative
jupyter notebook notebooks/walkthrough.ipynb
```

**Google Colab.** `notebooks/walkthrough.ipynb` runs as-is in Colab: the first
cell prompts you to upload `statarb-kalman-pairs.zip`, the second unzips it and
puts the package on the path, and the rest of the notebook runs unchanged. No
local setup required.

Prices are cached to `data/*.parquet` on first download so reruns are
reproducible and offline. The price cache is gitignored; the outputs behind the
Results section below are committed in `results/`.

## Key parameters (`config.py`)

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `formation_frac` | fraction of sample used only for screening | 0.5 |
| `delta` | Kalman process-noise knob (higher = faster-moving beta) | 1e-4 |
| `r` | Kalman observation noise | 1e-3 |
| `zscore_window` | trailing window for the spread z-score | 60 |
| `entry` / `exit` / `stop` | z-score bands (hysteresis) | 2.0 / 0.5 / 4.0 |
| `cost_bps` | cost per leg-trade, in basis points | 1.0 |
| `rebalance_hedge` | re-hedge daily (True) vs freeze beta at entry (False) | True |

## Verification

I do not report any performance number I have not produced from a real run;
every figure under Results comes from the commands shown there. What *is*
verified, on synthetic data with a known ground truth, is the machinery that
would otherwise silently corrupt those numbers:

- **Kalman recovery** — on a synthetic pair with a hedge ratio drifting 1.0→1.6,
  the filtered beta tracks the truth (correlation ≈ 0.998, terminal absolute
  error ≈ 0.03 after burn-in).
- **No look-ahead** — perturbing the final price leaves every prior daily return
  bit-for-bit unchanged (max difference 0.0).
- **PnL accounting** — the backtest return on a hand-computed fixture matches the
  analytical value to 1e-12.

```bash
python tests/test_synthetic.py      # runs without market data or statsmodels
pytest -q tests/                    # full suite (cointegration needs statsmodels)
```

## Results

Data: Yahoo Finance adjusted daily closes, 2016-01-04 to 2024-12-31. Formation
(screening) window 2016-01-04 to 2020-06-30; out-of-sample trading window
2020-07-01 to 2024-12-30 (1,132 trading days). All parameters are the
`config.py` defaults; nothing was tuned.

```bash
python scripts/screen_pairs.py --start 2016-01-04 --out results/pairs.csv
# once per passing pair: EWC/EWA, SPY/IVV, MA/V, CVX/EWA
python scripts/run_backtest.py --y EWC --x EWA --start 2016-01-04 > results/bt_EWC_EWA.txt
```

**Screening (formation window)**

4 of 41 candidate pairs passed. All passing pairs are shown; the full ranked
table is in `results/pairs.csv`, and each passing pair's backtest output is in
`results/bt_<y>_<x>.txt`.

| y | x | corr | EG p-value | hedge ratio | half-life | Johansen | passes |
|---|---|------|-----------|-------------|-----------|----------|--------|
| EWC | EWA | 0.971 | 0.0006 | 0.889 | 11.5 | yes | yes |
| SPY | IVV | 1.000 | 0.0019 | 0.999 | 3.9 | yes | yes |
| MA | V | 0.998 | 0.0047 | 1.273 | 12.7 | yes | yes |
| CVX | EWA | 0.909 | 0.0443 | 1.117 | 23.5 | yes | yes |

**Pair selection: chosen after seeing the out-of-sample results.** All four
passing pairs were backtested on the trading window, and EWC/EWA was picked
after comparing those results. That makes the scorecard below a best-of-four
result, not a clean held-out estimate: the choice used the test window. The
other three pairs, at 1bp with daily re-hedging, for comparison:

| pair | Sharpe | ann. return | max DD |
|------|--------|-------------|--------|
| SPY / IVV | -0.03 | -0.00% | -0.14% |
| MA / V | 0.01 | -0.01% | -5.78% |
| CVX / EWA | -0.24 | -1.47% | -12.35% |

The live forward run on EWC/EWA is the first truly out-of-sample test.

**Out-of-sample scorecard (traded pair: y = EWC, x = EWA; 1bp, daily re-hedge)**

| metric | value |
|--------|-------|
| Annualised return | 1.98% |
| Annualised vol | 1.91% |
| Sharpe | 1.04 |
| Sortino | 0.55 |
| Max drawdown | -3.32% |
| Calmar | 0.60 |
| Hit rate | 41.9% |
| Number of trades | 74 |
| Average turnover | 0.0654 |

Returns are per dollar of gross exposure. Hit rate counts every non-zero
return day, including days when only a cost was paid (for example the day
after an exit), so with costs it understates how often the market P&L was
positive (54.5% at 0bp).

**Cost / re-hedging sensitivity**

A robustness check only, with signal and Kalman parameters fixed at the defaults.

| cost (bps) | re-hedge | Sharpe | ann. return | max DD | avg turnover |
|-----------|----------|--------|-------------|--------|--------------|
| 0.0 | daily | 1.12 | 2.15% | -3.18% | 0.0654 |
| 1.0 | daily | 1.04 | 1.98% | -3.32% | 0.0654 |
| 5.0 | daily | 0.70 | 1.31% | -3.86% | 0.0654 |
| 1.0 | frozen | 1.04 | 1.99% | -3.32% | 0.0654 |

Each extra basis point costs roughly 0.17% a year in return. Freezing the hedge
ratio at entry makes almost no difference: with `delta = 1e-4` the beta moves
slowly, so daily re-hedging adds little turnover beyond the entry and exit
trades.

## Caveats

- The universe in `config.py` is hand-picked from names that tend to pair. A
  production screen would run over a full, point-in-time universe including
  delisted names, so that pair selection is not itself a survivorship bet.
- Cointegration estimated in-sample frequently weakens out-of-sample; that is
  precisely why performance is reported on the held-out window.
- 1bp per leg-trade is optimistic outside the most liquid names — use the
  sensitivity sweep to find the break-even before trusting any edge.

## References

- E. Chan, *Algorithmic Trading: Winning Strategies and Their Rationale* (2013)
  — the Kalman-filter hedge-ratio formulation.
- Engle & Granger (1987) — cointegration and error correction.
- Johansen (1991) — likelihood-based cointegration-rank testing.
