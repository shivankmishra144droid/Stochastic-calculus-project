"""ML benchmark 1: neural-network surrogate of the Black-Scholes formula (Person 1).

Homogeneity C(S, K) = K c(S/K) lets the network learn c = C/K from (m = S/K, T, r, sigma).
Compared against the closed form: accuracy in range, extrapolation outside it, and
no-arbitrage shape constraints the formula satisfies by construction.

Figures: ml_mlp_error_vs_moneyness, ml_mlp_extrapolation
Tables : ml_mlp_metrics, ml_mlp_noarb
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.core.bs import bs_price
from src.plotting import GREEN, GREY, SLATE, save_table, savefig

RANGES = dict(m=(0.7, 1.3), T=(0.02, 2.0), r=(0.0, 0.1), sigma=(0.05, 0.6))


def sample(n, rng, ranges=RANGES):
    X = np.column_stack([rng.uniform(*ranges[k], n) for k in ("m", "T", "r", "sigma")])
    return X, target(X)


def target(X):
    m, T, r, s = X.T
    return bs_price(m, 1.0, T, r, s, "C")


def fit(n=50_000, seed=config.SEED):
    rng = np.random.default_rng(seed)
    X, y = sample(n, rng)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu",
                                                         max_iter=500, early_stopping=True, random_state=seed))
    model.fit(Xtr, ytr)
    return model, Xte, yte


def noarb_violations(model, n=41):
    """On a grid, share of points where the network's c(m) decreases in m (or in sigma), or is concave in m.
    Convexity in m is equivalent to convexity of C in K, since C(K) = K c(S/K)."""
    m = np.linspace(0.75, 1.25, n)
    rows = []
    for T in (0.1, 0.5, 1.0, 1.5):
        for s in (0.1, 0.2, 0.4):
            X = np.column_stack([m, np.full(n, T), np.full(n, 0.05), np.full(n, s)])
            c = model.predict(X)
            Xs = np.column_stack([np.full(n, 1.0), np.full(n, T), np.full(n, 0.05), np.linspace(0.06, 0.58, n)])
            cs = model.predict(Xs)
            rows.append(dict(T=T, sigma=s,
                             decreasing_in_S=np.mean(np.diff(c) < -1e-6),
                             decreasing_in_sigma=np.mean(np.diff(cs) < -1e-6),
                             non_convex_in_K=np.mean(np.diff(c, 2) < -1e-6)))
    return pd.DataFrame(rows)


def make_figures(cfg=config) -> list[str]:
    model, Xte, yte = fit()
    pred = model.predict(Xte)
    err = pred - yte
    rng = np.random.default_rng(config.SEED + 1)
    ext = {"σ in [0.6, 0.9]": dict(RANGES, sigma=(0.6, 0.9)), "T in [2, 3]": dict(RANGES, T=(2.0, 3.0))}
    rows = [dict(set="test (in range)", n=len(yte), mae=np.abs(err).mean(), max_abs=np.abs(err).max(),
                 mae_per_100=100 * np.abs(err).mean())]
    for name, rg in ext.items():
        X, y = sample(5000, rng, rg)
        e = model.predict(X) - y
        rows.append(dict(set=f"extrapolation {name}", n=len(y), mae=np.abs(e).mean(), max_abs=np.abs(e).max(),
                         mae_per_100=100 * np.abs(e).mean()))
    paths = [save_table(pd.DataFrame(rows), "ml_mlp_metrics"), save_table(noarb_violations(model), "ml_mlp_noarb")]

    # error vs moneyness (test set)
    fig, ax = plt.subplots(figsize=(config.FIG_SINGLE, 2.2))
    bins = np.linspace(0.7, 1.3, 25)
    idx = np.digitize(Xte[:, 0], bins)
    mae = [100 * np.abs(err[idx == i]).mean() for i in range(1, len(bins))]
    ax.plot(0.5 * (bins[1:] + bins[:-1]), mae, "o-", ms=2.5, color=GREEN)
    ax.set_xlabel("Moneyness $S/K$")
    ax.set_ylabel("MAE (price per $K=100$)")
    paths.append(savefig(fig, "ml_mlp_error_vs_moneyness"))

    # extrapolation: price along sigma and T at the money
    fig, ax = plt.subplots(1, 2, figsize=(config.FIG_FULL, 2.2))
    s = np.linspace(0.05, 0.9, 200)
    X = np.column_stack([np.ones_like(s), np.ones_like(s), np.full_like(s, 0.05), s])
    ax[0].plot(s, 100 * target(X), color="k", lw=1, label="Black–Scholes")
    ax[0].plot(s, 100 * model.predict(X), color=GREEN, lw=1, ls="--", label="MLP")
    ax[0].axvspan(0.6, 0.9, color=GREY, alpha=0.25, label="outside training")
    ax[0].set_xlabel(r"$\sigma$ ($S=K=100$, $T=1$)")
    ax[0].set_ylabel("Call price")
    ax[0].legend(fontsize=7)
    T = np.linspace(0.02, 3, 200)
    X = np.column_stack([np.ones_like(T), T, np.full_like(T, 0.05), np.full_like(T, 0.2)])
    ax[1].plot(T, 100 * target(X), color="k", lw=1)
    ax[1].plot(T, 100 * model.predict(X), color=GREEN, lw=1, ls="--")
    ax[1].axvspan(2, 3, color=GREY, alpha=0.25)
    ax[1].set_xlabel(r"$T$ ($S=K=100$, $\sigma=0.2$)")
    fig.tight_layout()
    paths.append(savefig(fig, "ml_mlp_extrapolation"))
    return paths
