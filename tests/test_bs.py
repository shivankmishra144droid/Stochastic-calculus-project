import numpy as np
import pytest

import config
from src.core.bs import (black76_price, bs_greeks, bs_price, bs_vega, implied_vol,
                         implied_vol_array, implied_vol_forward)

BASE = (config.S0, config.K0, config.T0, config.R0, config.SIGMA0)


def test_baseline_values():
    assert bs_price(*BASE, "C") == pytest.approx(config.BS_CALL_REF, abs=1e-4)
    assert bs_price(*BASE, "P") == pytest.approx(config.BS_PUT_REF, abs=1e-4)


def test_put_call_parity_random():
    rng = np.random.default_rng(config.SEED)
    n = 10_000
    S, K = rng.uniform(50, 150, n), rng.uniform(50, 150, n)
    T, r = rng.uniform(0.02, 3, n), rng.uniform(0, 0.12, n)
    sig, q = rng.uniform(0.05, 0.8, n), rng.uniform(0, 0.05, n)
    gap = bs_price(S, K, T, r, sig, "C", q) - bs_price(S, K, T, r, sig, "P", q) \
        - (S * np.exp(-q * T) - K * np.exp(-r * T))
    assert np.max(np.abs(gap)) < 1e-10


@pytest.mark.parametrize("o", ["C", "P"])
@pytest.mark.parametrize("S,K,T,q", [(100, 100, 1, 0), (90, 110, 0.3, 0.02), (130, 100, 2, 0.01)])
def test_greeks_vs_finite_differences(o, S, K, T, q):
    r, s = 0.05, 0.25
    g = bs_greeks(S, K, T, r, s, o, q)
    f = lambda **kw: bs_price(**{**dict(S=S, K=K, T=T, r=r, sigma=s, option_type=o, q=q), **kw})
    h = 1e-4
    fd = dict(
        delta=(f(S=S + h) - f(S=S - h)) / (2 * h),
        gamma=(f(S=S + 1e-2) - 2 * f() + f(S=S - 1e-2)) / 1e-4,
        vega=(f(sigma=s + h) - f(sigma=s - h)) / (2 * h),
        theta=-(f(T=T + h) - f(T=T - h)) / (2 * h),
        rho=(f(r=r + h) - f(r=r - h)) / (2 * h),
    )
    for k, v in fd.items():
        assert g[k] == pytest.approx(v, rel=1e-4, abs=1e-6), k


def test_black76_matches_bs_on_forward():
    S, K, T, r, s, q = 100, 95, 0.5, 0.06, 0.3, 0.01
    F = S * np.exp((r - q) * T)
    for o in "CP":
        assert black76_price(F, K, T, r, s, o) == pytest.approx(bs_price(S, K, T, r, s, o, q), abs=1e-12)


def test_implied_vol_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(100):
        S, K = 100.0, rng.uniform(70, 130)
        T, r, s = rng.uniform(0.02, 2), rng.uniform(0, 0.1), rng.uniform(0.06, 0.9)
        o = rng.choice(["C", "P"])
        price = bs_price(S, K, T, r, s, o)
        if price < 1e-6:          # no information left in the price
            continue
        iv = implied_vol(price, S, K, T, r, o)
        assert bs_price(S, K, T, r, iv, o) == pytest.approx(price, abs=1e-7)   # always reprices
        if bs_vega(S, K, T, r, s) > 1e-3:   # sigma is identifiable only where vega is not ~0
            assert iv == pytest.approx(s, abs=1e-6)


def test_implied_vol_forward_and_array():
    F, K, T, r, s = 24000.0, 23000.0, 0.1, 0.064, 0.15
    p = black76_price(F, K, T, r, s, "P")
    assert implied_vol_forward(p, F, K, T, r, "P") == pytest.approx(s, abs=1e-7)
    out = implied_vol_array([p, p], [F, F], [K, K], T, r, ["P", "P"], forward=True)
    assert np.allclose(out, s, atol=1e-7)


def test_implied_vol_nan_outside_bounds():
    assert np.isnan(implied_vol(0.0001, 100, 50, 0.5, 0.06, "C"))      # below intrinsic
    assert np.isnan(implied_vol(150.0, 100, 50, 0.5, 0.06, "C"))       # above S
    assert np.isnan(implied_vol(-1.0, 100, 100, 0.5, 0.06, "P"))
