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
│   ├── screen_pairs.py       # screen the universe -> ranked pairs.csv
│   └── run_backtest.py       # backtest one pair out-of-sample (+ optional plots)
├── notebooks/
│   └── walkthrough.ipynb     # end-to-end narrative walkthrough
└── tests/
    ├── test_synthetic.py     # Kalman recovery, no-look-ahead, PnL hand-check
    └── test_cointegration.py # EG separates cointegrated vs spurious pairs
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Screen the universe for cointegrated pairs (formation window only)
python scripts/screen_pairs.py --out pairs.csv

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
reproducible and offline. Data and generated outputs are gitignored.

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

I do not report any performance number I have not produced from a real run, so
the tables below are blank. What *is* verified, on synthetic data with a known
ground truth, is the machinery that would otherwise silently corrupt those
numbers:

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

Run on your own data window and fill in. Nothing here is hard-coded.

**Screening (formation window)**

| y | x | corr | EG p-value | hedge ratio | half-life | Johansen | passes |
|---|---|------|-----------|-------------|-----------|----------|--------|
|   |   |      |           |             |           |          |        |

**Out-of-sample scorecard (traded pair)**

| metric | value |
|--------|-------|
| Annualised return |  |
| Annualised vol |  |
| Sharpe |  |
| Sortino |  |
| Max drawdown |  |
| Calmar |  |
| Hit rate |  |
| Number of trades |  |
| Average turnover |  |

**Cost / re-hedging sensitivity**

| cost (bps) | re-hedge | Sharpe | ann. return | max DD | avg turnover |
|-----------|----------|--------|-------------|--------|--------------|
| 0.0 | daily |  |  |  |  |
| 1.0 | daily |  |  |  |  |
| 5.0 | daily |  |  |  |  |
| 1.0 | frozen |  |  |  |  |

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
