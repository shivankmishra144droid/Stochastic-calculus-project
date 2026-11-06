"""ML benchmark 2: decision-tree regressor trained on binomial-tree prices (Person 2).

A regression tree predicts a piecewise-constant function, so it can fit prices well while its
finite-difference Delta is zero almost everywhere and spikes at split boundaries:
accurate in price, useless for hedging.

Figures: ml_dtree_steps, ml_dtree_delta
Tables : ml_dtree_metrics
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

import config
from src.core.binomial import binomial_price
from src.core.bs import bs_greeks, bs_price
from src.ml.mlp_bs import RANGES
from src.plotting import GREEN, GREY, PALETTE, SLATE, save_table, savefig

N_TREE = 500
DEPTHS = (4, 8, 12, 20)


def dataset(n=20_000, seed=config.SEED):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(*RANGES[k], n) for k in ("m", "T", "r", "sigma")])
    y = np.array([binomial_price(m, 1.0, T, r, s, N_TREE, "C") for m, T, r, s in X])
    return X, y


def make_figures(cfg=config) -> list[str]:
    X, y = dataset()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=config.SEED)
    models, rows = {}, []
    S = np.linspace(80, 120, 801)
    Xline = np.column_stack([S / 100, np.ones_like(S), np.full_like(S, 0.05), np.full_like(S, 0.2)])
    for d in DEPTHS:
        m = DecisionTreeRegressor(max_depth=d, random_state=config.SEED).fit(Xtr, ytr)
        models[d] = m
        e = m.predict(Xte) - yte
        delta = np.gradient(100 * m.predict(Xline), S)
        rows.append(dict(depth=d, leaves=m.get_n_leaves(), mae_per_100=100 * np.abs(e).mean(),
                         max_abs_per_100=100 * np.abs(e).max(), share_delta_zero=np.mean(np.abs(delta) < 1e-9)))
    paths = [save_table(pd.DataFrame(rows), "ml_dtree_metrics")]

    fig, ax = plt.subplots(figsize=(config.FIG_SINGLE, 2.3))
    ax.plot(S, bs_price(S, 100, 1, 0.05, 0.2, "C"), color="k", lw=1, label="Black–Scholes")
    for d, col in zip((8, 20), (SLATE, GREEN)):
        ax.step(S, 100 * models[d].predict(Xline), where="mid", color=col, lw=0.9, label=f"tree, depth {d}")
    ax.set_xlabel("Spot $S$ ($K=100$, $T=1$, $\\sigma=0.2$)")
    ax.set_ylabel("Call price")
    ax.legend(fontsize=7)
    paths.append(savefig(fig, "ml_dtree_steps"))

    fig, ax = plt.subplots(figsize=(config.FIG_SINGLE, 2.3))
    ax.plot(S, bs_greeks(S, 100, 1, 0.05, 0.2, "C")["delta"], color="k", lw=1, label=r"BS $\Delta$")
    ax.plot(S, np.gradient(100 * models[12].predict(Xline), S), color=GREEN, lw=0.7, label=r"tree $\Delta$ (depth 12)")
    ax.set_xlabel("Spot $S$")
    ax.set_ylabel(r"$\Delta$ (finite difference)")
    ax.legend(fontsize=7)
    paths.append(savefig(fig, "ml_dtree_delta"))
    return paths
