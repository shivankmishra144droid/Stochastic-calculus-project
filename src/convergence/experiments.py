"""Convergence of binomial trees to Black-Scholes (proposal §2.3, PROJECT_PLAN.md §5, Person 3).

Figures: conv_price_vs_N, conv_loglog_error, conv_error_heatmap, conv_corrections
Tables : conv_rates, conv_runtime, conv_strike_position
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import config
from src.core.binomial import binomial_price, bs_smoothed_price
from src.core.bs import bs_price
from src.plotting import GREEN, GREY, PALETTE, SLATE, save_table, savefig

S0, K0, R0, SIG0, T0 = config.S0, config.K0, config.R0, config.SIGMA0, config.T0
NS = np.arange(1, 1001)


# ---------------------------------------------------------------------------
# Pricers for every scheme (callable N -> price) at the baseline contract
# ---------------------------------------------------------------------------
def schemes(S=S0, K=K0, T=T0, r=R0, sig=SIG0, o="C"):
    tree = lambda m: (lambda N: binomial_price(S, K, T, r, sig, N, o, method=m))
    crr = tree("crr")
    return {
        "CRR": crr,
        "Jarrow–Rudd": tree("jr"),
        "Leisen–Reimer": tree("lr"),
        "CRR avg(N, N+1)": lambda N: 0.5 * (crr(N) + crr(N + 1)),
        "BS-smoothed CRR": lambda N: bs_smoothed_price(S, K, T, r, sig, N, o),
        # Richardson needs V(N) and V(2N) on the same parity branch, so N is rounded up to even
        "Richardson 2V(2N) − V(N)": lambda N: 2 * crr(2 * (N + N % 2)) - crr(N + N % 2),
    }


def strike_position(N, S=S0, K=K0, T=T0, sig=SIG0):
    """lambda = frac(j*), j* = [ln(K/S) + N sigma sqrt(dt)] / (2 sigma sqrt(dt)): where K sits between terminal nodes."""
    h = sig * np.sqrt(T / N)
    j = (np.log(K / S) + N * h) / (2 * h)
    return j - np.floor(j)


def loglog_slope(Ns, err, n_min=50):
    """Slope of log|e_N| on log N (N >= n_min) with a 95% confidence interval."""
    m = (Ns >= n_min) & (np.abs(err) > 0)
    res = stats.linregress(np.log(Ns[m]), np.log(np.abs(err[m])))
    half = stats.t.ppf(0.975, m.sum() - 2) * res.stderr
    return res.slope, res.slope - half, res.slope + half


# ---------------------------------------------------------------------------
def price_vs_N():
    ref = bs_price(S0, K0, T0, R0, SIG0, "C")
    crr = np.array([binomial_price(S0, K0, T0, R0, SIG0, n) for n in NS])
    ev, od = NS % 2 == 0, NS % 2 == 1
    fig, ax = plt.subplots(figsize=(config.FIG_SINGLE, 2.3))
    ax.plot(NS[ev], crr[ev], color=GREEN, lw=0.9, label="even $N$")
    ax.plot(NS[od], crr[od], color=GREY, lw=0.9, label="odd $N$")
    ax.axhline(ref, color="k", ls="--", lw=0.8, label="Black–Scholes")
    ax.set_ylim(ref - 0.15, ref + 0.15)
    ax.set_xlabel("Steps $N$")
    ax.set_ylabel("CRR call price")
    ax.legend()
    return savefig(fig, "conv_price_vs_N"), crr - ref


def loglog_error():
    ref = bs_price(S0, K0, T0, R0, SIG0, "C")
    sch = schemes()
    rows = []
    fig, ax = plt.subplots(figsize=(config.FIG_SINGLE, 2.6))
    for (name, f), col in zip(list(sch.items())[:3], [GREEN, SLATE, "#B45309"]):
        err = np.array([f(n) for n in NS]) - ref
        Ns_eff = NS if name != "Leisen–Reimer" else NS + (NS % 2 == 0)   # LR uses odd N
        ax.loglog(Ns_eff, np.abs(err), ".", ms=1.5, color=col, label=name)
        for sub, mask in (("all", np.ones_like(NS, bool)), ("even", NS % 2 == 0), ("odd", NS % 2 == 1)):
            if name == "Leisen–Reimer" and sub != "all":
                continue
            s, lo, hi = loglog_slope(Ns_eff[mask], err[mask])
            rows.append(dict(method=name, subsequence=sub, slope=s, ci_low=lo, ci_high=hi,
                             err_N100=abs(err[99]), err_N1000=abs(err[-1])))
    ax.loglog(NS, 2 / NS, "k:", lw=0.8, label=r"$2/N$")
    ax.loglog(NS, 1 / NS**2, "k--", lw=0.6, label=r"$1/N^2$")
    ax.set_xlabel("Steps $N$")
    ax.set_ylabel(r"$|V_N - V^{BS}|$")
    ax.set_ylim(1e-9, 1)
    ax.legend(fontsize=6.5, markerscale=4)
    return savefig(fig, "conv_loglog_error"), save_table(pd.DataFrame(rows), "conv_rates")


def error_heatmap():
    """N * |error| averaged over N in {200, 201} (removes the even/odd parity), call option."""
    mny = np.array([0.8, 0.9, 1.0, 1.1, 1.2])          # K / S
    sigs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    Ts = np.array([1 / 12, 0.25, 0.5, 1.0, 2.0])

    def scaled_err(K, T, s):
        ref = bs_price(S0, K, T, R0, s, "C")
        return np.mean([n * abs(binomial_price(S0, K, T, R0, s, n) - ref) for n in (200, 201)])

    A = np.array([[scaled_err(m * S0, T0, s) for m in mny] for s in sigs])
    B = np.array([[scaled_err(m * S0, T, SIG0) for m in mny] for T in Ts])
    fig, ax = plt.subplots(1, 2, figsize=(config.FIG_FULL, 2.4))
    for a, M, ylab, yt in ((ax[0], A, r"$\sigma$", sigs), (ax[1], B, "$T$ (years)", np.round(Ts, 3))):
        im = a.imshow(M, origin="lower", cmap="Greens", aspect="auto")
        a.set_xticks(range(len(mny)), [f"{m:.1f}" for m in mny])
        a.set_yticks(range(len(yt)), yt)
        a.set_xlabel("Moneyness $K/S$")
        a.set_ylabel(ylab)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                       color="white" if M[i, j] > 0.6 * M.max() else "k")
        fig.colorbar(im, ax=a, label=r"$N\,|e_N|$")
    fig.tight_layout()
    return savefig(fig, "conv_error_heatmap")


def strike_position_table():
    """Signed CRR error x N against the strike position lambda, for ATM and OTM calls."""
    rows = []
    for K in (100.0, 110.0):
        ref = bs_price(S0, K, T0, R0, SIG0, "C")
        for n in range(100, 161):
            rows.append(dict(K=K, N=n, lam=strike_position(n, K=K),
                             N_err=n * (binomial_price(S0, K, T0, R0, SIG0, n) - ref)))
    return save_table(pd.DataFrame(rows), "conv_strike_position")


def corrections():
    ref = bs_price(S0, K0, T0, R0, SIG0, "C")
    Ns = np.unique(np.round(np.geomspace(10, 1000, 60)).astype(int))
    fig, ax = plt.subplots(figsize=(config.FIG_SINGLE, 2.6))
    for (name, f), col in zip(schemes().items(), PALETTE + ["#7C3AED"]):
        if name == "Jarrow–Rudd":
            continue
        err = np.abs(np.array([f(n) for n in Ns]) - ref)
        ax.loglog(Ns, np.maximum(err, 1e-12), "-", lw=1, color=col, label=name)
    ax.set_xlabel("Steps $N$")
    ax.set_ylabel(r"$|V_N - V^{BS}|$")
    ax.legend(fontsize=6.5)
    return savefig(fig, "conv_corrections")


def runtime_table(tols=(1e-2, 1e-3, 1e-4)):
    """For each scheme: smallest N on a grid with |error| < tol at both N and N+1, and its median runtime."""
    ref = bs_price(S0, K0, T0, R0, SIG0, "C")
    grid = np.unique(np.round(np.geomspace(2, 25000, 90)).astype(int))
    rows = []
    for name, f in schemes().items():
        # walk the grid once, recording the first N that meets each tolerance; stop when all are met
        first = {}
        for n in grid:
            err = max(abs(f(n) - ref), abs(f(n + 1) - ref))
            for tol in tols:
                if tol not in first and err < tol:
                    first[tol] = int(n)
            if len(first) == len(tols):
                break
        for tol in tols:
            N_ok = first.get(tol)
            if N_ok is None:
                rows.append(dict(method=name, tol=tol, N=np.nan, runtime_ms=np.nan))
                continue
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                f(N_ok)
                times.append((time.perf_counter() - t0) * 1e3)
            rows.append(dict(method=name, tol=tol, N=N_ok, runtime_ms=float(np.median(times))))
    return save_table(pd.DataFrame(rows), "conv_runtime")


def make_figures(cfg=config) -> list[str]:
    p1, _ = price_vs_N()
    p2, t1 = loglog_error()
    return [p1, p2, t1, error_heatmap(), strike_position_table(), corrections(), runtime_table()]
