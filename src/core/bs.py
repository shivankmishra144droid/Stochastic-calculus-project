"""Black-Scholes model: prices, Greeks, Black-76 and implied volatility.

Equation references are to the proposal (Section 2.2) and PROJECT_PLAN.md §3.1.
All pricing functions are vectorised over NumPy arrays; `option_type` is "C" or "P"
(scalar or array). Time in years, rates and volatilities as decimals, `q` is a
continuous dividend yield.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    """d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt T),  d2 = d1 - sigma sqrt T  (proposal §2.2)."""
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    vol = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol
    return d1, d1 - vol


def _is_call(option_type):
    return np.asarray(option_type) == "C"


def bs_price(S, K, T, r, sigma, option_type="C", q=0.0):
    """Black-Scholes price (proposal §2.2).

    C = S e^{-qT} Phi(d1) - K e^{-rT} Phi(d2),   P = K e^{-rT} Phi(-d2) - S e^{-qT} Phi(-d1).
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    S, K, T = (np.asarray(x, dtype=float) for x in (S, K, T))
    dS, dK = S * np.exp(-q * T), K * np.exp(-r * T)
    call = dS * norm.cdf(d1) - dK * norm.cdf(d2)
    put = dK * norm.cdf(-d2) - dS * norm.cdf(-d1)
    out = np.where(_is_call(option_type), call, put)
    return out if out.ndim else float(out)


def bs_vega(S, K, T, r, sigma, q=0.0):
    """Vega = S e^{-qT} phi(d1) sqrt(T)  (per unit of sigma, not per 1%)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.asarray(S) * np.exp(-q * np.asarray(T)) * norm.pdf(d1) * np.sqrt(T)


def bs_greeks(S, K, T, r, sigma, option_type="C", q=0.0):
    """Closed-form Greeks (PROJECT_PLAN.md §3.1). Theta is per YEAR (divide by 365 for per day).

    Call: Delta = e^{-qT} Phi(d1), Gamma = e^{-qT} phi(d1) / (S sigma sqrt T), Vega = S e^{-qT} phi(d1) sqrt T,
          Theta = -S e^{-qT} phi(d1) sigma / (2 sqrt T) + q S e^{-qT} Phi(d1) - r K e^{-rT} Phi(d2),
          Rho = K T e^{-rT} Phi(d2).
    Put: the same Gamma and Vega; Delta, Theta, Rho follow from put-call parity.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    eq, er = np.exp(-q * T), np.exp(-r * T)
    pdf1 = norm.pdf(d1)
    call = _is_call(option_type)

    gamma = eq * pdf1 / (S * sigma * np.sqrt(T))
    vega = S * eq * pdf1 * np.sqrt(T)
    decay = -S * eq * pdf1 * sigma / (2 * np.sqrt(T))
    delta = np.where(call, eq * norm.cdf(d1), eq * (norm.cdf(d1) - 1))
    theta = np.where(call,
                     decay + q * S * eq * norm.cdf(d1) - r * K * er * norm.cdf(d2),
                     decay - q * S * eq * norm.cdf(-d1) + r * K * er * norm.cdf(-d2))
    rho = np.where(call, K * T * er * norm.cdf(d2), -K * T * er * norm.cdf(-d2))
    g = dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
    return {k: (v if np.ndim(v) else float(v)) for k, v in g.items()}


def black76_price(F, K, T, r, sigma, option_type="C"):
    """Black-76 price from the forward F (PROJECT_PLAN.md §3.1).

    C = e^{-rT} [F Phi(d1) - K Phi(d2)],  d1 = [ln(F/K) + sigma^2 T / 2] / (sigma sqrt T).
    Identical to bs_price with S = F e^{-rT} and q = 0.
    """
    F, T = np.asarray(F, dtype=float), np.asarray(T, dtype=float)
    return bs_price(F * np.exp(-r * T), K, T, r, sigma, option_type, 0.0)


def implied_vol(price, S, K, T, r, option_type="C", q=0.0, tol=1e-8, max_iter=50):
    """Implied volatility: solve BS(sigma) = price (proposal §2.4). Scalar inputs.

    1. No-arbitrage bounds: call  max(S e^{-qT} - K e^{-rT}, 0) < C < S e^{-qT};
                            put   max(K e^{-rT} - S e^{-qT}, 0) < P < K e^{-rT}.  Outside -> NaN.
    2. Newton from the Manaster-Koehler start sigma0 = sqrt(2 |ln(S/K) + (r-q) T| / T), clipped to [0.05, 1];
       sigma <- sigma - (BS(sigma) - price) / Vega(sigma).
    3. If Vega is tiny or sigma leaves [1e-4, 5]: bisection-type fallback (scipy brentq on [1e-4, 5]).
    """
    if not np.all(np.isfinite([price, S, K, T])) or T <= 0:
        return np.nan
    dS, dK = S * np.exp(-q * T), K * np.exp(-r * T)
    lower = max(dS - dK, 0.0) if option_type == "C" else max(dK - dS, 0.0)
    upper = dS if option_type == "C" else dK
    if not (lower < price < upper):
        return np.nan

    sigma = float(np.clip(np.sqrt(2 * abs(np.log(S / K) + (r - q) * T) / T), 0.05, 1.0))
    for _ in range(max_iter):
        diff = bs_price(S, K, T, r, sigma, option_type, q) - price
        if abs(diff) < tol:
            return sigma
        vega = float(bs_vega(S, K, T, r, sigma, q))
        if vega < 1e-8:
            break
        sigma -= diff / vega
        if not (1e-4 < sigma < 5.0):
            break

    f = lambda s: bs_price(S, K, T, r, s, option_type, q) - price
    try:
        if f(1e-4) * f(5.0) > 0:
            return np.nan
        return brentq(f, 1e-4, 5.0, xtol=1e-10)
    except (ValueError, RuntimeError):
        return np.nan


def implied_vol_forward(price, F, K, T, r, option_type="C"):
    """Implied volatility under Black-76 (forward F instead of spot); used with the implied forward."""
    return implied_vol(price, F * np.exp(-r * T), K, T, r, option_type, 0.0)


def implied_vol_array(price, S, K, T, r, option_type, q=0.0, forward=False):
    """Vectorised wrapper: element-wise implied_vol (or implied_vol_forward when forward=True, S = F)."""
    b = np.broadcast_arrays(np.asarray(price, float), np.asarray(S, float), np.asarray(K, float),
                            np.asarray(T, float), np.asarray(r, float), np.asarray(option_type), np.asarray(q, float))
    out = np.empty(b[0].shape)
    for i, (p, s, k, t, rr, o, qq) in enumerate(zip(*(x.ravel() for x in b))):
        out.flat[i] = (implied_vol_forward(p, s, k, t, rr, o) if forward
                       else implied_vol(p, s, k, t, rr, o, qq))
    return out
