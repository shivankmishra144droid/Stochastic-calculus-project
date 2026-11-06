"""ML benchmark 3: linear regression of the CRR error on the features theory predicts (Person 3).

Features: 1/N, (-1)^N / N, lambda/N, lambda^2/N, where lambda in [0, 1) is the strike's position
between two terminal nodes (src.convergence.experiments.strike_position).
Fitted on N in [20, 300], evaluated on N in [301, 500]; the fitted error is then subtracted
from the tree price (V_corrected = V_N - e_hat) and compared with the analytic corrections.

Figures: ml_error_regression
Tables : ml_error_regression
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import config
from src.convergence.experiments import strike_position
from src.core.binomial import binomial_price, bs_smoothed_price
from src.core.bs import bs_price
from src.plotting import GREEN, GREY, SLATE, save_table, savefig

S0, R0 = config.S0, config.R0
CONTRACTS = [(K, s, T) for K in (90.0, 100.0, 110.0) for s in (0.2, 0.3) for T in (0.5, 1.0)]
N_TRAIN, N_TEST = np.arange(20, 301), np.arange(301, 501)


def features(N, K, s, T, which="full"):
    N = np.asarray(N, float)
    lam = strike_position(N.astype(int), K=K, T=T, sig=s)
    cols = [1 / N, (-1) ** N / N]
    if which == "full":
        cols += [lam / N, lam**2 / N]
    return np.column_stack(cols)


def errors(K, s, T, Ns):
    ref = bs_price(S0, K, T, R0, s, "C")
    return np.array([binomial_price(S0, K, T, R0, s, int(n), "C") for n in Ns]) - ref


def make_figures(cfg=config) -> list[str]:
    rows, curves = [], {}
    for K, s, T in CONTRACTS:
        e_tr, e_te = errors(K, s, T, N_TRAIN), errors(K, s, T, N_TEST)
        for which in ("parity only", "full"):
            w = "full" if which == "full" else "parity"
            reg = LinearRegression(fit_intercept=False).fit(features(N_TRAIN, K, s, T, w), e_tr)
            pred = reg.predict(features(N_TEST, K, s, T, w))
            ss_res = ((e_te - pred) ** 2).sum()
            r2 = 1 - ss_res / ((e_te - e_te.mean()) ** 2).sum()
            ref = bs_price(S0, K, T, R0, s, "C")
            avg = [abs(0.5 * (binomial_price(S0, K, T, R0, s, int(n)) + binomial_price(S0, K, T, R0, s, int(n) + 1)) - ref)
                   for n in N_TEST[::10]]
            smooth = [abs(bs_smoothed_price(S0, K, T, R0, s, int(n)) - ref) for n in N_TEST[::10]]
            rows.append(dict(K=K, sigma=s, T=T, features=which, r2_test=r2,
                             **{f"coef_{i}": c for i, c in enumerate(reg.coef_)},
                             max_err_raw=np.abs(e_te).max(), max_err_regression=np.abs(e_te - pred).max(),
                             max_err_avg_N_N1=max(avg), max_err_bs_smoothing=max(smooth)))
            if s == 0.2 and T == 1.0 and K in (100.0, 110.0):
                curves[(K, which)] = (e_te, pred)
    tab = pd.DataFrame(rows)
    paths = [save_table(tab, "ml_error_regression")]

    fig, ax = plt.subplots(1, 2, figsize=(config.FIG_FULL, 2.2), sharey=False)
    for a, K in zip(ax, (100.0, 110.0)):
        e_te, pred = curves[(K, "full")]
        a.plot(N_TEST, N_TEST * e_te, color=GREY, lw=0.8, label="actual $N e_N$")
        a.plot(N_TEST, N_TEST * pred, color=GREEN, lw=0.8, ls="--", label="regression")
        a.plot(N_TEST, N_TEST * curves[(K, "parity only")][1], color=SLATE, lw=0.8, ls=":", label="1/N, (−1)^N/N only")
        a.set_xlabel("$N$ (test range)")
        a.set_title(f"$K={K:.0f}$, $S=100$", fontsize=8)
    ax[0].set_ylabel(r"$N\,(V_N - V^{BS})$")
    ax[0].legend(fontsize=6.5)
    fig.tight_layout()
    paths.append(savefig(fig, "ml_error_regression"))
    return paths
