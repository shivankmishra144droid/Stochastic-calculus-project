import numpy as np
import pandas as pd
import pytest

from src.core.vol import as_of, ewma_vol, hist_vol, log_returns, parkinson_vol


@pytest.fixture
def gbm():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=3000)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.2 / np.sqrt(252), len(idx)))), index=idx)
    return close


def test_hist_and_ewma_recover_sigma(gbm):
    assert hist_vol(gbm, 252).dropna().mean() == pytest.approx(0.2, abs=0.01)
    assert ewma_vol(gbm).dropna().mean() == pytest.approx(0.2, abs=0.015)


def test_parkinson_on_synthetic_ranges():
    rng = np.random.default_rng(3)
    n, sig = 2000, 0.25
    # simulate intraday paths to get honest high/low
    paths = np.cumsum(rng.normal(0, sig / np.sqrt(252 * 390), (n, 390)), axis=1)
    high, low = pd.Series(np.exp(paths.max(1).clip(0))), pd.Series(np.exp(paths.min(1).clip(max=0)))
    assert parkinson_vol(high, low, 500).dropna().mean() == pytest.approx(sig, rel=0.1)


def test_no_look_ahead(gbm):
    v = hist_vol(gbm, 30)
    d = pd.Timestamp("2021-06-05")                       # a Saturday: must take Friday's value
    assert as_of(v, [d])[0] == v.loc[:d].dropna().iloc[-1]
    assert log_returns(gbm).index.equals(gbm.index)


def test_ewma_no_look_ahead(gbm):
    """Changing a later price must not change any earlier EWMA value (including the seed window)."""
    v = ewma_vol(gbm)
    bumped = gbm.copy()
    bumped.iloc[20] *= 1.05                               # inside the first 30 returns
    w = ewma_vol(bumped)
    assert v.iloc[:20].equals(w.iloc[:20])
    assert v.iloc[:30].isna().all() and np.isfinite(v.iloc[30])
