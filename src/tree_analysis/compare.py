"""Binomial tree analysis (PROJECT_PLAN.md §5, Person 2).

Figures: tree_diagram_N4, tree_greeks_vs_bs
Tables : tree_replication_example, tree_methods_comparison, tree_arbitrage_example
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from src.core.binomial import binomial_greeks, binomial_price, binomial_tree_values, tree_params
from src.core.bs import bs_greeks, bs_price
from src.plotting import GREEN, GREY, SLATE, save_table, savefig

S0, K0, R0, SIG0, T0 = config.S0, config.K0, config.R0, config.SIGMA0, config.T0


def replication_example():
    """One-period replication: Delta shares + B in the bond reproduce the call payoff in both states."""
    u, d, p, _ = tree_params(S0, K0, T0, R0, SIG0, 1)
    Cu, Cd = max(S0 * u - K0, 0), max(S0 * d - K0, 0)
    delta = (Cu - Cd) / (S0 * (u - d))
    B = np.exp(-R0 * T0) * (u * Cd - d * Cu) / (u - d)
    rows = [
        ("u", u), ("d", d), ("risk-neutral q", p), ("C_u", Cu), ("C_d", Cd),
        ("Delta (shares)", delta), ("B (bond, today)", B),
        ("replicating cost Delta*S0 + B", delta * S0 + B),
        ("risk-neutral value e^{-rT}[q C_u + (1-q) C_d]", np.exp(-R0 * T0) * (p * Cu + (1 - p) * Cd)),
        ("portfolio payoff, up state", delta * S0 * u + B * np.exp(R0 * T0)),
        ("portfolio payoff, down state", delta * S0 * d + B * np.exp(R0 * T0)),
    ]
    return save_table(pd.DataFrame(rows, columns=["quantity", "value"]), "tree_replication_example")


def tree_diagram(N=4):
    stock, opt = binomial_tree_values(S0, K0, T0, R0, SIG0, N, "C")
    fig, ax = plt.subplots(figsize=(config.FIG_FULL, 3.4))
    for n in range(N + 1):
        for i in range(n + 1):
            x, y = n, n - 2 * i
            if n < N:
                ax.plot([x, x + 1], [y, y + 1], color=GREY, lw=0.7, zorder=0)
                ax.plot([x, x + 1], [y, y - 1], color=GREY, lw=0.7, zorder=0)
            ax.text(x, y, f"{stock[n][i]:.2f}\n{opt[n][i]:.2f}", ha="center", va="center", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GREEN if opt[n][i] > 0 else GREY, lw=0.8))
    ax.set_xticks(range(N + 1))
    ax.set_xticklabels([f"$t={n * T0 / N:.2f}$" for n in range(N + 1)])
    ax.set_yticks([])
    ax.spines[["left"]].set_visible(False)
    ax.set_xlim(-0.5, N + 0.5)
    ax.set_title("")
    return savefig(fig, "tree_diagram_N4")


def methods_table():
    rows = []
    for label, K in (("ITM", 90.0), ("ATM", 100.0), ("OTM", 110.0)):
        ref = bs_price(S0, K, T0, R0, SIG0, "C")
        for N in (25, 51, 101, 201):
            row = dict(moneyness=label, K=K, N=N, bs=ref)
            for m in ("crr", "jr", "lr"):
                v = binomial_price(S0, K, T0, R0, SIG0, N, "C", method=m)
                row[m] = v
                row[f"{m}_err"] = v - ref
            rows.append(row)
    return save_table(pd.DataFrame(rows), "tree_methods_comparison")


def greeks_vs_bs():
    Ns = np.arange(10, 301)
    b = bs_greeks(S0, K0, T0, R0, SIG0, "C")
    G = {m: pd.DataFrame([binomial_greeks(S0, K0, T0, R0, SIG0, n, "C", method=m) for n in Ns]) for m in ("crr", "lr")}
    fig, axes = plt.subplots(1, 3, figsize=(config.FIG_FULL, 2.2))
    for ax, k, lab in zip(axes, ["delta", "gamma", "theta"], [r"$\Delta$", r"$\Gamma$", r"$\Theta$ (per year)"]):
        ax.plot(Ns, G["crr"][k], color=GREEN, lw=0.8, label="CRR")
        ax.plot(Ns, G["lr"][k], color=SLATE, lw=0.8, label="Leisen–Reimer")
        ax.axhline(b[k], color="k", ls="--", lw=0.8, label="Black–Scholes")
        ax.set_xlabel("$N$")
        ax.set_ylabel(lab)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return savefig(fig, "tree_greeks_vs_bs")


def arbitrage_example():
    """Parameters with e^{r dt} > u: CRR q > 1, so the bond dominates the stock (arbitrage)."""
    rows = []
    for r, sig, N in ((0.5, 0.05, 1), (0.5, 0.05, 4), (0.5, 0.05, 400), (0.05, 0.2, 1)):
        u, d, p, _ = tree_params(S0, K0, 1.0, r, sig, N)
        rows.append(dict(r=r, sigma=sig, N=N, u=u, d=d, growth=np.exp(r / N), q=p, arbitrage_free=bool(0 < p < 1)))
    return save_table(pd.DataFrame(rows), "tree_arbitrage_example")


def make_figures(cfg=config) -> list[str]:
    return [replication_example(), tree_diagram(), methods_table(), greeks_vs_bs(), arbitrage_example()]
