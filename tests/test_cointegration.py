"""Cointegration checks. These need statsmodels, so they skip if it is absent.

I generate one genuinely cointegrated pair and one independent pair and assert
the Engle-Granger p-value separates them: low for the cointegrated pair, high
for the spurious one. This guards against a screen that would wave through any
correlated-but-non-stationary pair.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

statsmodels = pytest.importorskip("statsmodels")

from src.cointegration import engle_granger, half_life


def _cointegrated(n=800, seed=0):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 1, n)) + 100
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.9 * spread[t - 1] + rng.normal(0, 1)  # stationary
    y = 1.5 * x + spread
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.Series(y, idx), pd.Series(x, idx)


def _independent(n=800, seed=1):
    rng = np.random.default_rng(seed)
    y = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100)
    x = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    return y.set_axis(idx), x.set_axis(idx)


def test_engle_granger_separates_cointegrated_from_spurious():
    yc, xc = _cointegrated()
    yi, xi = _independent()
    p_coint, beta = engle_granger(yc, xc)
    p_spur, _ = engle_granger(yi, xi)
    assert p_coint < 0.05, p_coint
    assert p_spur > 0.10, p_spur
    assert abs(beta - 1.5) < 0.2, beta


def test_half_life_positive_for_mean_reverting_spread():
    yc, xc = _cointegrated()
    _, beta = engle_granger(yc, xc)
    hl = half_life(yc - beta * xc)
    assert np.isfinite(hl) and hl > 0, hl
