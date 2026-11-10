"""Binomial trees for European options: CRR, Jarrow-Rudd, Leisen-Reimer (proposal §2.1, PROJECT_PLAN.md §3.2).

Node convention: at step n the value array has n+1 entries, index 0 = all up-moves.
Children of node i are i (up) and i+1 (down), so one backward step is
    V_n = e^{-r dt} [p V_{n+1}[:-1] + (1 - p) V_{n+1}[1:]]       (proposal §2.1)
which is vectorised over the nodes of a time step.
"""
import numpy as np
from scipy.stats import norm

from src.core.bs import bs_price


# ---------------------------------------------------------------------------
# Parametrisations
# ---------------------------------------------------------------------------
def _peizer_pratt(z, n):
    """Peizer-Pratt inversion h(z) used by Leisen-Reimer (method 2)."""
    return 0.5 + 0.5 * np.sign(z) * np.sqrt(
        1.0 - np.exp(-((z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))) ** 2) * (n + 1.0 / 6.0)))


def tree_params(S, K, T, r, sigma, N, q=0.0, method="crr"):
    """Return (u, d, p, N) for the chosen parametrisation (Δt = T/N, b = r - q).

    crr : u = e^{sigma sqrt(dt)}, d = 1/u, p = (e^{b dt} - d)/(u - d)
    jr  : u, d = e^{(b - sigma^2/2) dt ± sigma sqrt(dt)}, p = (e^{b dt} - d)/(u - d)  (≈ 1/2, exact form is arbitrage-free)
    lr  : odd N only (even N is raised to N+1); p = h(d2), p' = h(d1), u = e^{b dt} p'/p, d = (e^{b dt} - p u)/(1 - p)
    """
    method = method.lower()
    if method == "lr" and N % 2 == 0:
        N += 1
    dt = T / N
    growth = np.exp((r - q) * dt)
    if method == "crr":
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
    elif method == "jr":
        drift = (r - q - 0.5 * sigma**2) * dt
        u = np.exp(drift + sigma * np.sqrt(dt))
        d = np.exp(drift - sigma * np.sqrt(dt))
    elif method == "lr":
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        p = _peizer_pratt(d2, N)
        p_prime = _peizer_pratt(d1, N)
        u = growth * p_prime / p
        d = (growth - p * u) / (1.0 - p)
        return u, d, p, N
    else:
        raise ValueError(f"unknown method {method!r}")
    p = (growth - d) / (u - d)
    return u, d, p, N


def _payoff(ST, K, option_type):
    return np.maximum(ST - K, 0.0) if option_type == "C" else np.maximum(K - ST, 0.0)


def _induct(S, K, T, r, N, u, d, p, option_type, keep_levels=0):
    """Backward induction. Returns V0 and, if keep_levels > 0, the value arrays of steps 1..keep_levels."""
    dt = T / N
    j = np.arange(N, -1, -1)
    V = _payoff(S * u**j * d ** (N - j), K, option_type)
    disc = np.exp(-r * dt)
    kept = {}
    for n in range(N - 1, -1, -1):
        V = disc * (p * V[:-1] + (1.0 - p) * V[1:])
        if 1 <= n <= keep_levels:
            kept[n] = V.copy()
    return float(V[0]), kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def binomial_price(S, K, T, r, sigma, N, option_type="C", q=0.0, method="crr"):
    """European option price from an N-step binomial tree (proposal §2.1).

    Raises ValueError if the risk-neutral probability is outside (0, 1), i.e. the
    no-arbitrage condition d < e^{(r-q) dt} < u fails.
    """
    u, d, p, N = tree_params(S, K, T, r, sigma, N, q, method)
    if not 0.0 < p < 1.0:
        raise ValueError(f"p = {p:.4f} outside (0,1): d < e^((r-q)dt) < u violated (arbitrage)")
    return _induct(S, K, T, r, N, u, d, p, option_type)[0]


def binomial_greeks(S, K, T, r, sigma, N, option_type="C", q=0.0, method="crr"):
    """Tree Greeks (PROJECT_PLAN.md §3.2).

    Delta = (V_u - V_d) / (S u - S d)
    Gamma = [(V_uu - V_ud)/(S u^2 - S u d) - (V_ud - V_dd)/(S u d - S d^2)] / [(S u^2 - S d^2)/2]
    Theta = (V_ud - V_0) / (2 dt)   (per year; exact centring only for CRR where u d = 1)
    Vega, Rho: central differences (sigma ± 0.001, r ± 0.0001) with the same N.
    """
    u, d, p, Nn = tree_params(S, K, T, r, sigma, N, q, method)
    if not 0.0 < p < 1.0:
        raise ValueError(f"p = {p:.4f} outside (0,1): d < e^((r-q)dt) < u violated (arbitrage)")
    V0, lv = _induct(S, K, T, r, Nn, u, d, p, option_type, keep_levels=2)
    dt = T / Nn
    (Vu, Vd), (Vuu, Vud, Vdd) = lv[1], lv[2]
    delta = (Vu - Vd) / (S * u - S * d)
    gamma = ((Vuu - Vud) / (S * u * u - S * u * d) - (Vud - Vdd) / (S * u * d - S * d * d)) / (0.5 * (S * u * u - S * d * d))
    theta = (Vud - V0) / (2 * dt)
    hs, hr = 1e-3, 1e-4
    vega = (binomial_price(S, K, T, r, sigma + hs, N, option_type, q, method)
            - binomial_price(S, K, T, r, sigma - hs, N, option_type, q, method)) / (2 * hs)
    rho = (binomial_price(S, K, T, r + hr, sigma, N, option_type, q, method)
           - binomial_price(S, K, T, r - hr, sigma, N, option_type, q, method)) / (2 * hr)
    return dict(price=V0, delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


def binomial_tree_values(S, K, T, r, sigma, N, option_type="C", q=0.0, method="crr"):
    """Stock and option values at every node (small N only, for drawing the tree).

    Returns (stock, option): lists indexed by step n, each an array of length n+1 (index 0 = all ups).
    """
    u, d, p, N = tree_params(S, K, T, r, sigma, N, q, method)
    stock = [S * u ** np.arange(n, -1, -1) * d ** np.arange(0, n + 1) for n in range(N + 1)]
    option = [None] * (N + 1)
    option[N] = _payoff(stock[N], K, option_type)
    disc = np.exp(-r * T / N)
    for n in range(N - 1, -1, -1):
        option[n] = disc * (p * option[n + 1][:-1] + (1 - p) * option[n + 1][1:])
    return stock, option


def amin_time_grid(T, N, sigma_fn, fine=20000):
    """Step boundaries t_0 = 0 < ... < t_N = T with sigma(t)^2 dt_i equal on every step (Amin's method).

    Total variance V = int_0^T sigma(t)^2 dt is split into N equal pieces c = V/N, so the tree
    keeps a constant u = e^{sqrt(c)} and stays recombining.
    """
    t = np.linspace(0.0, T, fine + 1)
    s2 = np.asarray([sigma_fn(x) for x in 0.5 * (t[1:] + t[:-1])]) ** 2
    cumvar = np.concatenate([[0.0], np.cumsum(s2 * np.diff(t))])
    c = cumvar[-1] / N
    bounds = np.interp(np.arange(N + 1) * c, cumvar, t)
    bounds[-1] = T
    return bounds, c


def binomial_price_timevarying(S, K, T, r_fn, sigma_fn, N, option_type="C", q=0.0):
    """CRR-style tree with deterministic time-varying sigma(t) and r(t) (proposal §2.4, Amin 1991).

    Step lengths dt_i are chosen so that sigma_i^2 dt_i = c is constant; then u = e^{sqrt c}, d = 1/u,
    and only p_i = (e^{(r_i - q) dt_i} - d)/(u - d) and the discount e^{-r_i dt_i} change per step.
    `r_fn` and `sigma_fn` are callables of t (years); pass `lambda t: const` for constants.
    With constant sigma and r this reproduces the CRR tree exactly.
    """
    bounds, c = amin_time_grid(T, N, sigma_fn)
    dts = np.diff(bounds)
    mids = 0.5 * (bounds[1:] + bounds[:-1])
    rs = np.array([r_fn(m) for m in mids])
    u = np.exp(np.sqrt(c))
    d = 1.0 / u
    ps = (np.exp((rs - q) * dts) - d) / (u - d)
    if np.any((ps <= 0) | (ps >= 1)):
        raise ValueError("risk-neutral probability outside (0,1) on some step; increase N")
    j = np.arange(N, -1, -1)
    V = _payoff(S * u**j * d ** (N - j), K, option_type)
    for i in range(N - 1, -1, -1):
        V = np.exp(-rs[i] * dts[i]) * (ps[i] * V[:-1] + (1 - ps[i]) * V[1:])
    return float(V[0])


def binomial_price_discrete_div(S, K, T, r, sigma, N, divs, option_type="C", method="proportional"):
    """Tree with discrete dividends on their ex-dates (proposal §2.4, PROJECT_PLAN.md §3.2).

    divs : list of (t_i, D_i) cash dividends, t_i in years from today. Only 0 < t_i <= T count.
    method:
      "proportional" - at the ex-date step k_i = round(t_i / dt) every node price is multiplied by (1 - delta_i),
                       delta_i = D_i / (S e^{r t_i}) (cash amount over the expected price on the ex-date).
                       The tree stays recombining.
      "escrow"       - the stock is split into a risky part S* = S - sum D_i e^{-r t_i}, which follows the
                       CRR tree, plus the known PV of dividends still to be paid. The node price at step n is
                       S*_node + PV of remaining dividends at that step.
    For a European option only terminal prices matter. With delta_i = D_i / (S e^{r t_i}) the proportional
    factor is S (1 - delta_i) = S - D_i e^{-r t_i} = S*, so the two methods give IDENTICAL European prices
    (and both converge to escrowed Black-Scholes). They differ only for early exercise / path dependence,
    or when delta_i is estimated from a realised price instead of the expected one.
    """
    divs = [(t, D) for t, D in divs if 0 < t <= T]
    dt = T / N
    u, d, p, N = tree_params(S, K, T, r, sigma, N, 0.0, "crr")
    j = np.arange(N, -1, -1)
    if method == "proportional":
        factor = np.prod([1 - D / (S * np.exp(r * t)) for t, D in divs]) if divs else 1.0
        ST = S * factor * u**j * d ** (N - j)
    elif method == "escrow":
        S_star = S - sum(D * np.exp(-r * t) for t, D in divs)
        ST = S_star * u**j * d ** (N - j)        # no dividends left to pay at T
    else:
        raise ValueError(f"unknown method {method!r}")
    V = _payoff(ST, K, option_type)
    disc = np.exp(-r * dt)
    for _ in range(N):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
    return float(V[0])


def binomial_price_cash_div(S, K, T, r, sigma, N, divs, option_type="C"):
    """Cash dividend paid on its ex-date (proposal §2.4): the FULL price S follows GBM and drops by D at t_ex.

    Hybrid tree: a CRR tree runs from 0 to the first ex-date t_1 (N_1 = round(N t_1 / T) steps). At each node the
    price drops by D_1 (floored at 0), and the remaining life is valued with Black-Scholes (later dividends escrowed):
        V_0 = e^{-r t_1} sum_j C(N_1, j) p^j (1-p)^{N_1-j} BS(S_j - D_1, K, T - t_1).
    Unlike the escrowed model, sigma applies to the whole price until the ex-date, so the two differ
    (escrowed BS underprices calls); this is the case where the tree's flexibility matters.
    """
    from scipy.stats import binom

    divs = sorted((t, D) for t, D in divs if 0 < t <= T)
    if not divs:
        return float(bs_price(S, K, T, r, sigma, option_type))
    (t1, D1), rest = divs[0], [(t - divs[0][0], D) for t, D in divs[1:]]
    N1 = max(1, int(round(N * t1 / T)))
    u, d, p, _ = tree_params(S, K, t1, r, sigma, N1, 0.0, "crr")
    j = np.arange(N1 + 1)
    S_ex = np.maximum(S * u**j * d ** (N1 - j) - D1, 1e-12)
    after = np.array([bs_escrowed(s, K, T - t1, r, sigma, rest, option_type) for s in S_ex])
    return float(np.exp(-r * t1) * np.sum(binom.pmf(j, N1, p) * after))


def cash_div_quadrature(S, K, T, r, sigma, t1, D, option_type="C", n=200):
    """Reference value for one cash dividend: e^{-r t1} E[BS(S_{t1} - D, K, T - t1)] by Gauss-Hermite quadrature."""
    z, w = np.polynomial.hermite_e.hermegauss(n)
    S_t = S * np.exp((r - 0.5 * sigma**2) * t1 + sigma * np.sqrt(t1) * z)
    vals = bs_price(np.maximum(S_t - D, 1e-12), K, T - t1, r, sigma, option_type)
    return float(np.exp(-r * t1) * np.sum(w * vals) / np.sqrt(2 * np.pi))


def bs_escrowed(S, K, T, r, sigma, divs, option_type="C"):
    """Black-Scholes with the escrowed-dividend approximation S -> S - sum D_i e^{-r t_i} (proposal §2.4)."""
    S_star = S - sum(D * np.exp(-r * t) for t, D in divs if 0 < t <= T)
    return bs_price(S_star, K, T, r, sigma, option_type)


def dividend_yield_equivalent(S, T, r, divs):
    """Continuous yield q with the same forward as the cash dividends (proposal §2.4):
    S e^{-qT} = S - sum D_i e^{-r t_i}  =>  q = -ln(1 - PV(D) / S) / T.

    (q = D/S is only right for one dividend and T = 1 year; for a one-month option it removes about D T, not D.)
    """
    pv = sum(D * np.exp(-r * t) for t, D in divs if 0 < t <= T)
    return float(-np.log(1.0 - pv / S) / T)


def bs_smoothed_price(S, K, T, r, sigma, N, option_type="C", q=0.0):
    """CRR tree whose last step is replaced by the Black-Scholes value over one dt ("BS-smoothing")."""
    u, d, p, N = tree_params(S, K, T, r, sigma, N, q, "crr")
    dt = T / N
    j = np.arange(N - 1, -1, -1)
    S_pen = S * u**j * d ** (N - 1 - j)
    V = np.asarray(bs_price(S_pen, K, dt, r, sigma, option_type, q), dtype=float)
    disc = np.exp(-r * dt)
    for _ in range(N - 1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
    return float(V[0])


__all__ = ["tree_params", "binomial_price", "binomial_greeks", "binomial_tree_values",
           "binomial_price_timevarying", "binomial_price_discrete_div", "binomial_price_cash_div",
           "cash_div_quadrature", "bs_escrowed", "dividend_yield_equivalent",
           "bs_smoothed_price", "amin_time_grid"]
