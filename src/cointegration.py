"""Pair screening: correlation pre-filter plus Engle-Granger / Johansen tests.

All tests run on the formation window only; the trading window is held out.
The tests come from statsmodels rather than being re-implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class PairTest:
    y: str
    x: str
    corr: float
    eg_pvalue: float
    eg_hedge_ratio: float
    half_life: float
    johansen_trace_ok: bool | None = None

    @property
    def passes(self) -> bool:
        ok = self.eg_pvalue < 0.05 and np.isfinite(self.half_life) and self.half_life > 0
        if self.johansen_trace_ok is not None:
            ok = ok and self.johansen_trace_ok
        return ok


def engle_granger(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """(p-value, static hedge ratio). statsmodels' coint uses the correct
    critical values for a pre-estimated cointegrating vector."""
    from statsmodels.tsa.stattools import coint
    import statsmodels.api as sm

    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    _, pvalue, _ = coint(df["y"], df["x"], trend="c")
    beta = sm.OLS(df["y"], sm.add_constant(df["x"])).fit().params.iloc[1]
    return float(pvalue), float(beta)


def half_life(spread: pd.Series) -> float:
    """OU half-life from the AR(1) discretisation; inf if the spread doesn't revert."""
    import statsmodels.api as sm

    s = spread.dropna()
    lag = s.shift(1).dropna()
    delta = (s - s.shift(1)).dropna()
    lag, delta = lag.align(delta, join="inner")
    if len(lag) < 3:
        return np.nan
    beta = sm.OLS(delta, sm.add_constant(lag)).fit().params.iloc[1]
    if beta >= 0:
        return np.inf
    return float(-np.log(2) / np.log(1 + beta))


def johansen_ok(y: pd.Series, x: pd.Series, det_order: int = 0,
                k_ar_diff: int = 1) -> bool:
    """True if the Johansen trace test finds a cointegrating vector at 95%."""
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    res = coint_johansen(df.to_numpy(), det_order, k_ar_diff)
    return bool(res.lr1[0] > res.cvt[0, 1])       # trace stat vs 95% crit, r=0


def screen_pairs(prices: pd.DataFrame,
                 candidates: Iterable[tuple[str, str]] | None = None,
                 min_corr: float = 0.8,
                 max_pvalue: float = 0.05,
                 run_johansen: bool = True) -> pd.DataFrame:
    """Screen a formation-window price panel; return pairs ranked by EG p-value."""
    cols = list(prices.columns)
    if candidates is None:
        candidates = combinations(cols, 2)

    log = prices.apply(np.log)
    rows: list[PairTest] = []
    for a, b in candidates:
        pair = log[[a, b]].dropna()
        if len(pair) < 60:
            continue
        corr = pair[a].corr(pair[b])
        if abs(corr) < min_corr:
            continue

        # Orient the higher-mean-log-price leg as dependent (arbitrary, consistent).
        y_name, x_name = (a, b) if pair[a].mean() >= pair[b].mean() else (b, a)
        pv, beta = engle_granger(pair[y_name], pair[x_name])
        spread = pair[y_name] - beta * pair[x_name]
        hl = half_life(spread)
        joh = johansen_ok(pair[y_name], pair[x_name]) if run_johansen else None
        rows.append(PairTest(y_name, x_name, float(corr), pv, beta, hl, joh))

    if not rows:
        return pd.DataFrame(columns=[
            "y", "x", "corr", "eg_pvalue", "eg_hedge_ratio",
            "half_life", "johansen_trace_ok", "passes",
        ])

    out = pd.DataFrame([vars(r) | {"passes": r.passes} for r in rows])
    return out.sort_values("eg_pvalue").reset_index(drop=True)
