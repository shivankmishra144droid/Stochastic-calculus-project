import numpy as np
import pytest

import config
from src.core.binomial import (binomial_greeks, binomial_price, binomial_price_cash_div,
                               binomial_price_discrete_div, binomial_price_timevarying, binomial_tree_values,
                               bs_escrowed, bs_smoothed_price, cash_div_quadrature, dividend_yield_equivalent,
                               tree_params)
from src.core.bs import bs_greeks, bs_price

S, K, T, R, SIG = config.S0, config.K0, config.T0, config.R0, config.SIGMA0


def test_one_step_replication():
    """Delta shares + B bonds replicate the payoff in both states, and V0 = Delta S + B = risk-neutral value."""
    u, d, p, _ = tree_params(S, K, T, R, SIG, 1)
    Cu, Cd = max(S * u - K, 0), max(S * d - K, 0)
    delta = (Cu - Cd) / (S * u - S * d)
    B = np.exp(-R * T) * (Cu - delta * S * u)
    assert delta * S * d + B * np.exp(R * T) == pytest.approx(Cd)
    assert binomial_price(S, K, T, R, SIG, 1) == pytest.approx(delta * S + B, abs=1e-12)


def test_two_step_by_hand_r0():
    u = 1.2
    sigma = np.log(u) / np.sqrt(0.5)
    d = 1 / u
    p = (1 - d) / (u - d)
    hand = p**2 * (100 * u * u - 100)                     # only up-up pays off
    assert binomial_price(100, 100, 1, 0.0, sigma, 2) == pytest.approx(hand, abs=1e-12)


def test_reproduces_proposal_table():
    assert binomial_price(S, K, T, R, SIG, 100) == pytest.approx(10.4306, abs=1e-4)
    assert binomial_price(S, K, T, R, SIG, 101) == pytest.approx(10.4680, abs=1e-4)


@pytest.mark.parametrize("method,N,tol", [("crr", 2000, 2e-3), ("jr", 2000, 3e-3), ("lr", 101, 1e-4)])
def test_converges_to_bs(method, N, tol):
    for o in "CP":
        assert binomial_price(S, K, T, R, SIG, N, o, method=method) == pytest.approx(bs_price(S, K, T, R, SIG, o), abs=tol)


@pytest.mark.parametrize("method", ["crr", "jr", "lr"])
@pytest.mark.parametrize("N", [1, 2, 7, 50, 51])
def test_put_call_parity_in_tree(method, N):
    c = binomial_price(S, 110, T, R, SIG, N, "C", 0.02, method)
    p = binomial_price(S, 110, T, R, SIG, N, "P", 0.02, method)
    assert c - p == pytest.approx(S * np.exp(-0.02 * T) - 110 * np.exp(-R * T), abs=1e-10)


def test_probability_bounds_and_arbitrage():
    for m in ("crr", "jr", "lr"):
        assert 0 < tree_params(S, K, T, R, SIG, 100, method=m)[2] < 1
    with pytest.raises(ValueError):              # huge r, tiny sigma, one big step: e^{r dt} > u
        binomial_price(S, K, 1.0, 0.5, 0.05, 1)


def test_tree_greeks_close_to_bs():
    g = binomial_greeks(S, K, T, R, SIG, 500)
    b = bs_greeks(S, K, T, R, SIG)
    assert g["delta"] == pytest.approx(b["delta"], abs=2e-3)
    assert g["gamma"] == pytest.approx(b["gamma"], abs=5e-4)
    assert g["theta"] == pytest.approx(b["theta"], abs=2e-2)
    assert g["vega"] == pytest.approx(b["vega"], rel=1e-2)
    assert g["rho"] == pytest.approx(b["rho"], rel=1e-2)


def test_tree_values_shape_and_root():
    stock, opt = binomial_tree_values(S, K, T, R, SIG, 4)
    assert [len(x) for x in stock] == [1, 2, 3, 4, 5]
    assert opt[0][0] == pytest.approx(binomial_price(S, K, T, R, SIG, 4))


def test_timevarying_constant_equals_crr():
    for N in (10, 101):
        tv = binomial_price_timevarying(S, K, T, lambda t: R, lambda t: SIG, N)
        assert tv == pytest.approx(binomial_price(S, K, T, R, SIG, N), abs=1e-9)


def test_timevarying_matches_bs_with_average_variance():
    sig_fn = lambda t: 0.15 if t < 0.5 else 0.30
    avg = np.sqrt((0.15**2 * 0.5 + 0.30**2 * 0.5) / 1.0)
    tv = binomial_price_timevarying(S, K, T, lambda t: R, sig_fn, 1001)
    assert tv == pytest.approx(bs_price(S, K, T, R, avg), abs=5e-3)


def test_dividends_zero_reduce_to_plain():
    assert bs_escrowed(S, K, T, R, SIG, []) == pytest.approx(bs_price(S, K, T, R, SIG))
    plain = binomial_price(S, K, T, R, SIG, 200)
    for m in ("proportional", "escrow"):
        assert binomial_price_discrete_div(S, K, T, R, SIG, 200, [], method=m) == pytest.approx(plain)
    # dividends after expiry are ignored
    assert bs_escrowed(S, K, T, R, SIG, [(1.5, 3.0)]) == pytest.approx(bs_price(S, K, T, R, SIG))


def test_escrow_tree_converges_to_escrowed_bs():
    divs = [(0.5, 3.0)]
    tree = binomial_price_discrete_div(S, K, T, R, SIG, 2000, divs, method="escrow")
    assert tree == pytest.approx(bs_escrowed(S, K, T, R, SIG, divs), abs=2e-3)


def test_cash_dividend_tree_converges_and_differs_from_escrow():
    divs = [(0.5, 3.0)]
    for o in "CP":
        ref = cash_div_quadrature(S, K, T, R, SIG, 0.5, 3.0, o)
        assert binomial_price_cash_div(S, K, T, R, SIG, 2000, divs, o) == pytest.approx(ref, abs=2e-3)
    # escrowed BS applies sigma to a smaller base, so it underprices the call relative to the cash model
    assert bs_escrowed(S, K, T, R, SIG, divs, "C") < cash_div_quadrature(S, K, T, R, SIG, 0.5, 3.0, "C") - 0.05
    assert binomial_price_cash_div(S, K, T, R, SIG, 500, [], "C") == pytest.approx(bs_price(S, K, T, R, SIG))


def test_bs_smoothing_is_more_accurate():
    ref = bs_price(S, K, T, R, SIG)
    errs = [abs(bs_smoothed_price(S, K, T, R, SIG, n) - ref) for n in range(100, 111)]
    raw = [abs(binomial_price(S, K, T, R, SIG, n) - ref) for n in range(100, 111)]
    assert max(errs) < max(raw)


@pytest.mark.parametrize("T,t_ex", [(1.0, 0.5), (30 / 365, 25 / 365), (0.25, 0.01)])
def test_dividend_yield_matches_escrowed_forward(T, t_ex):
    """q from dividend_yield_equivalent gives the same forward (and so the same put-call parity) as escrow."""
    divs = [(t_ex, 3.0)]
    q = dividend_yield_equivalent(S, T, R, divs)
    assert S * np.exp(-q * T) == pytest.approx(S - 3.0 * np.exp(-R * t_ex), rel=1e-12)
    c, p = bs_price(S, K, T, R, SIG, "C", q), bs_price(S, K, T, R, SIG, "P", q)
    ce, pe = bs_escrowed(S, K, T, R, SIG, divs, "C"), bs_escrowed(S, K, T, R, SIG, divs, "P")
    assert c - p == pytest.approx(ce - pe, abs=1e-10)
    assert dividend_yield_equivalent(S, T, R, []) == 0.0


def test_carry_spread_and_implied_dividend_recover_truth():
    """Synthetic forwards with a known funding spread s and a known dividend: the estimators recover both."""
    import pandas as pd
    from src.model_risk.experiments import carry_spread, implied_dividend
    r, s, D, S0_ = 0.06, 0.015, 5.5, 1500.0
    rows = []
    for days in (15, 45, 75):
        T = days / 365
        rows.append(dict(T=T, days=days, r=r, spans_dividend=False, spread=np.log(np.exp((r + s) * T)) / T - r))
    c = pd.DataFrame(rows)
    assert carry_spread(c) == pytest.approx(s, abs=1e-12)
    T, t_ex = 45 / 365, 25 / 365
    F = (S0_ - D * np.exp(-(r + s) * t_ex)) * np.exp((r + s) * T)
    assert implied_dividend(S0_, F, T, t_ex, r + s) == pytest.approx(D, rel=1e-12)
    # with the wrong (T-bill) carry the same forward implies a much smaller dividend
    assert implied_dividend(S0_, F, T, t_ex, r) < D - 2.0
