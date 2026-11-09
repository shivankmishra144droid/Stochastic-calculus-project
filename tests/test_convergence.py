"""Convergence helpers: rates, parity of the CRR error, the steps-needed rule and Richardson off the money."""
import numpy as np
import pytest

from src.convergence.experiments import loglog_slope, strike_position
from src.core.binomial import binomial_price
from src.core.bs import bs_price


def test_crr_slope_is_minus_one():
    Ns = np.arange(50, 401)
    err = np.array([binomial_price(100, 100, 1, 0.05, 0.2, n) for n in Ns]) - bs_price(100, 100, 1, 0.05, 0.2)
    s, C = loglog_slope(Ns, err)
    assert -1.1 <= s <= -0.9
    assert 1.0 < C < 3.0                                   # |e_N| ~ 2/N at the baseline


def test_averaging_reduces_error_5x():
    ref = bs_price(100, 100, 1, 0.05, 0.2)
    raw = [abs(binomial_price(100, 100, 1, 0.05, 0.2, n) - ref) for n in range(100, 201)]
    avg = [abs(0.5 * (binomial_price(100, 100, 1, 0.05, 0.2, n) + binomial_price(100, 100, 1, 0.05, 0.2, n + 1)) - ref)
           for n in range(100, 201)]
    assert max(raw) >= 5 * max(avg)


def test_strike_position_atm_parity():
    assert strike_position(100) == pytest.approx(0.0)      # even N: strike on a node
    assert strike_position(101) == pytest.approx(0.5)      # odd N: halfway between nodes


def test_steps_needed_ignores_lucky_crossings():
    """An error sequence that dips below tol once (a zero crossing) must not count; only the point after the last
    failure does, and only once enough later grid points confirm it."""
    from src.convergence.experiments import steps_needed
    grid = np.arange(1, 40)
    errs = {n: (1e-5 if n == 3 else 1.0 / n) for n in range(1, 50)}   # lucky dip at N = 3
    f = lambda n: errs[n]
    out = steps_needed(f, 0.0, (0.1,), grid, confirm=5)
    assert out[0.1] == 11                                              # max(1/N, 1/(N+1)) < 0.1 from N = 11
    assert steps_needed(f, 0.0, (1e-6,), grid, confirm=5)[1e-6] is None


def test_richardson_fails_off_the_money():
    """Richardson cancels the CRR error at K = S (clean C/N) but not at K = 110 (strike position varies with N)."""
    ref_atm, ref_otm = bs_price(100, 100, 1, 0.05, 0.2), bs_price(100, 110, 1, 0.05, 0.2)
    rich = lambda K, N: 2 * binomial_price(100, K, 1, 0.05, 0.2, 2 * N) - binomial_price(100, K, 1, 0.05, 0.2, N)
    assert abs(rich(100, 200) - ref_atm) < 1e-5
    assert abs(rich(110, 200) - ref_otm) > 1e-3
