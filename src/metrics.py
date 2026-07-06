"""Performance statistics. Simple per-period returns, annualised by sqrt(N);
zero risk-free unless supplied. Definitions kept explicit, not library-hidden."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ann(periods_per_year: int) -> float:
    return float(np.sqrt(periods_per_year))


def sharpe(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    excess = returns - rf / periods_per_year
    sd = excess.std(ddof=1)
    return np.nan if (sd == 0 or np.isnan(sd)) else float(excess.mean() / sd * _ann(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    excess = returns - rf / periods_per_year
    dd = np.sqrt((excess[excess < 0] ** 2).mean())
    return np.nan if (dd == 0 or np.isnan(dd)) else float(excess.mean() / dd * _ann(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def annualised_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    n = len(returns.dropna())
    if n == 0:
        return np.nan
    total = (1.0 + returns.fillna(0.0)).prod()
    return -1.0 if total <= 0 else float(total ** (periods_per_year / n) - 1.0)


def annualised_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(returns.std(ddof=1) * _ann(periods_per_year))


def hit_rate(returns: pd.Series) -> float:
    active = returns[returns != 0]
    return np.nan if len(active) == 0 else float((active > 0).mean())


def count_trades(positions: pd.Series) -> int:
    p = positions.fillna(0.0)
    return int((p.diff().fillna(p) != 0).sum())


def summary(returns: pd.Series, equity: pd.Series, positions: pd.Series,
            turnover: pd.Series | None = None, periods_per_year: int = 252) -> dict:
    stats = {
        "ann_return": annualised_return(returns, periods_per_year),
        "ann_vol": annualised_vol(returns, periods_per_year),
        "sharpe": sharpe(returns, periods_per_year),
        "sortino": sortino(returns, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "hit_rate": hit_rate(returns),
        "n_trades": count_trades(positions),
    }
    dd = abs(stats["max_drawdown"])
    stats["calmar"] = stats["ann_return"] / dd if dd > 0 else np.nan
    if turnover is not None:
        stats["avg_turnover"] = float(turnover.mean())
    return stats
