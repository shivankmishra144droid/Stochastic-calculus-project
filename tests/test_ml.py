import numpy as np
import pandas as pd
import pytest

from src.convergence.experiments import loglog_slope, strike_position
from src.core.binomial import binomial_price
from src.core.bs import bs_price
from src.ml.har import design, har_forecast
from src.ml.mlp_bs import target


@pytest.fixture
def rv():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2023-01-02", periods=500)
    var = np.exp(rng.normal(np.log(0.15**2 / 252), 0.5, len(idx)))
    return pd.Series(var, index=idx)


def test_har_uses_no_future_data(rv):
    for h in (5, 22):
        for date in (rv.index[200], rv.index[350]):
            sigma, last_target = har_forecast(rv, date, h)
            assert np.isfinite(sigma) and sigma > 0
            assert last_target <= date                       # every training target was observed by `date`


def test_har_target_is_forward_mean(rv):
    D = design(rv, 5)
    t = 30
    assert D["y"].iloc[t] == pytest.approx(rv.iloc[t + 1:t + 6].mean())
    assert D["target_end"].iloc[t] == rv.index[t + 5]


def test_crr_slope_is_minus_one():
    Ns = np.arange(50, 401)
    err = np.array([binomial_price(100, 100, 1, 0.05, 0.2, n) for n in Ns]) - bs_price(100, 100, 1, 0.05, 0.2)
    s, lo, hi = loglog_slope(Ns, err)
    assert -1.1 <= s <= -0.9


def test_averaging_reduces_error_5x():
    ref = bs_price(100, 100, 1, 0.05, 0.2)
    raw = [abs(binomial_price(100, 100, 1, 0.05, 0.2, n) - ref) for n in range(100, 201)]
    avg = [abs(0.5 * (binomial_price(100, 100, 1, 0.05, 0.2, n) + binomial_price(100, 100, 1, 0.05, 0.2, n + 1)) - ref)
           for n in range(100, 201)]
    assert max(raw) >= 5 * max(avg)


def test_strike_position_atm_parity():
    assert strike_position(100) == pytest.approx(0.0)      # even N: strike on a node
    assert strike_position(101) == pytest.approx(0.5)      # odd N: halfway between nodes


def test_mlp_target_homogeneity():
    X = np.array([[1.1, 0.5, 0.05, 0.3]])
    assert target(X)[0] * 100 == pytest.approx(bs_price(110, 100, 0.5, 0.05, 0.3))
