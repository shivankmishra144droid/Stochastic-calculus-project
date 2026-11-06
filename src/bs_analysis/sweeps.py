"""Black-Scholes sensitivity analysis (PROJECT_PLAN.md §5, Person 1).

Figures: bs_price_surface, bs_greeks_vs_S, bs_sensitivity
Tables : bs_parity_residual, bs_limiting_cases
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from src.core.bs import bs_greeks, bs_price
from src.plotting import GREEN, GREY, PALETTE, SLATE, save_table, savefig

S0, K0, R0, SIG0, T0 = config.S0, config.K0, config.R0, config.SIGMA0, config.T0


def price_surface():
    S = np.linspace(60, 140, 81)
    T = np.linspace(0.01, 2, 80)
    SS, TT = np.meshgrid(S, T)
    C = bs_price(SS, K0, TT, R0, SIG0, "C")
    fig, ax = plt.subplots(figsize=(config.FIG_SINGLE, 2.6))
    cs = ax.contourf(SS, TT, C, levels=20, cmap="Greens")
    ax.contour(SS, TT, C, levels=[5, 10, 20, 30], colors=SLATE, linewidths=0.6)
    fig.colorbar(cs, ax=ax, label="Call price")
    ax.set_xlabel("Spot $S$")
    ax.set_ylabel("Maturity $T$ (years)")
    return savefig(fig, "bs_price_surface")


def greeks_vs_S():
    S = np.linspace(60, 140, 401)
    mats = {"1 week": 7 / 365, "1 month": 30 / 365, "1 year": 1.0}
    fig, axes = plt.subplots(2, 2, figsize=(config.FIG_FULL, 4.2), sharex=True)
    for (lab, T), col in zip(mats.items(), [GREEN, SLATE, GREY]):
        g = bs_greeks(S, K0, T, R0, SIG0, "C")
        axes[0, 0].plot(S, g["delta"], color=col, label=lab)
        axes[0, 1].plot(S, g["gamma"], color=col)
        axes[1, 0].plot(S, g["vega"], color=col)
        axes[1, 1].plot(S, g["theta"] / 365, color=col)
    for ax, name in zip(axes.flat, [r"$\Delta$", r"$\Gamma$", "Vega", r"$\Theta$ (per day)"]):
        ax.set_ylabel(name)
        ax.axvline(K0, color="k", lw=0.5, ls=":")
    for ax in axes[1]:
        ax.set_xlabel("Spot $S$ (strike 100)")
    axes[0, 0].legend()
    fig.tight_layout()
    return savefig(fig, "bs_greeks_vs_S")


def sensitivity():
    fig, ax = plt.subplots(1, 3, figsize=(config.FIG_FULL, 2.2), sharey=False)
    sig = np.linspace(0.05, 0.6, 100)
    r = np.linspace(0, 0.12, 100)
    T = np.linspace(0.01, 3, 100)
    for o, col in (("C", GREEN), ("P", SLATE)):
        lab = "call" if o == "C" else "put"
        ax[0].plot(sig, bs_price(S0, K0, T0, R0, sig, o), color=col, label=lab)
        ax[1].plot(r, bs_price(S0, K0, T0, r, SIG0, o), color=col)
        ax[2].plot(T, bs_price(S0, K0, T, R0, SIG0, o), color=col)
    ax[0].set_xlabel(r"Volatility $\sigma$")
    ax[1].set_xlabel("Rate $r$")
    ax[2].set_xlabel("Maturity $T$")
    ax[0].set_ylabel("Price (ATM, $S=K=100$)")
    ax[0].legend()
    fig.tight_layout()
    return savefig(fig, "bs_sensitivity")


def parity_residual(n=10_000):
    rng = np.random.default_rng(config.SEED)
    S, K = rng.uniform(50, 150, n), rng.uniform(50, 150, n)
    T, r = rng.uniform(0.02, 3, n), rng.uniform(0, 0.12, n)
    sig, q = rng.uniform(0.05, 0.8, n), rng.uniform(0, 0.05, n)
    gap = (bs_price(S, K, T, r, sig, "C", q) - bs_price(S, K, T, r, sig, "P", q)
           - (S * np.exp(-q * T) - K * np.exp(-r * T)))
    df = pd.DataFrame({"n_samples": [n], "max_abs_residual": [np.abs(gap).max()],
                       "mean_abs_residual": [np.abs(gap).mean()]})
    return save_table(df, "bs_parity_residual")


def limiting_cases():
    rows = []
    F_disc = S0 - K0 * np.exp(-R0 * T0)
    rows.append(("sigma -> 0 (1e-6)", "call", bs_price(S0, K0, T0, R0, 1e-6, "C"), max(F_disc, 0),
                 "discounted intrinsic of the forward: max(S - K e^{-rT}, 0)"))
    rows.append(("T -> 0 (1e-8), S = 110", "call", bs_price(110, K0, 1e-8, R0, SIG0, "C"), 10.0, "payoff (S - K)^+"))
    rows.append(("T -> 0 (1e-8), S = 90", "put", bs_price(90, K0, 1e-8, R0, SIG0, "P"), 10.0, "payoff (K - S)^+"))
    rows.append(("S -> inf (S = 1000)", "call", bs_price(1000, K0, T0, R0, SIG0, "C"), 1000 - K0 * np.exp(-R0 * T0),
                 "S - K e^{-rT}"))
    rows.append(("S -> 0 (S = 1e-3)", "put", bs_price(1e-3, K0, T0, R0, SIG0, "P"), K0 * np.exp(-R0 * T0) - 1e-3,
                 "K e^{-rT} - S"))
    df = pd.DataFrame(rows, columns=["case", "option", "bs_price", "limit", "limit_formula"])
    df["abs_diff"] = (df["bs_price"] - df["limit"]).abs()
    return save_table(df, "bs_limiting_cases")


def make_figures(cfg=config) -> list[str]:
    return [price_surface(), greeks_vs_S(), sensitivity(), parity_residual(), limiting_cases()]
