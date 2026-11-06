"""ML benchmark 4: random forest for implied volatility, with a quadratic smile baseline (Person 4).

Quadratic smile per (symbol, date, expiry):  sigma(m) = a + b (m - 1) + c (m - 1)^2,  m = K/F
  a = level, b = skew, c = curvature (reported per regime).
Random forest: inputs (ln K/F, T, sigma_30d) -> implied vol. No regime label is used as an input.
Splits are by TIME, never random (strikes on the same day are near-duplicates):
  fold A: train Jan 2025 (normal)           -> test Apr 2025 (volatile)
  fold B: train Jan + Apr 2025              -> test Jul 2025 (calm)
Every prediction is turned into a price with Black-76 on the implied forward and compared with settlement.

Figures: ml_rf_iv_fit, ml_rf_importance
Tables : ml_smile_params, ml_rf_metrics
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import config
from src.core.bs import black76_price
from src.market.analysis import REGIME_ORDER, load_priced
from src.plotting import GREEN, GREY, SLATE, save_table, savefig

FEATURES = ["log_kf", "T", "hv30"]
FOLDS = {"A: Jan -> Apr": (["normal"], "volatile"), "B: Jan+Apr -> Jul": (["normal", "volatile"], "calm")}


def smile_data():
    df = load_priced()
    d = df[df["otm"] & df["iv"].notna() & df["hv30"].notna()].copy()
    d["log_kf"] = np.log(d["moneyness"])
    d["x"] = d["moneyness"] - 1
    return d


def quad_params(d):
    """a, b, c of the quadratic smile for every (symbol, date, expiry) with at least 5 strikes."""
    rows = []
    for (sym, date, exp), g in d.groupby(["symbol", "date", "expiry"]):
        if len(g) < 5:
            continue
        c, b, a = np.polyfit(g["x"], g["iv"], 2)
        rows.append(dict(symbol=sym, date=date, expiry=exp, regime=g["regime"].iloc[0], days=g["days"].iloc[0],
                         n=len(g), level_a=a, skew_b=b, curvature_c=c))
    return pd.DataFrame(rows)


def _quad_design(g):
    return np.column_stack([g["x"], g["x"] ** 2, g["T"], g["x"] * g["T"], g["hv30"]])


def evaluate(d):
    """IV RMSE and pricing MAE (bp of underlying) for each fold and method."""
    rows, preds, models = [], {}, {}
    for fold, (train_regs, test_reg) in FOLDS.items():
        tr, te = d[d["regime"].isin(train_regs)], d[d["regime"] == test_reg].copy()
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=config.SEED, n_jobs=-1)
        rf.fit(tr[FEATURES], tr["iv"])
        models[fold] = rf
        te["pred_rf"] = rf.predict(te[FEATURES])
        te["pred_const"] = te["hv30"]
        # quadratic smile + term + vol level, fitted on the training period per symbol (a forecast, like the RF)
        te["pred_quad_train"] = np.nan
        for sym, g in tr.groupby("symbol"):
            lin = LinearRegression().fit(_quad_design(g), g["iv"])
            mask = te["symbol"] == sym
            te.loc[mask, "pred_quad_train"] = lin.predict(_quad_design(te[mask]))
        # same-day quadratic fit: in-sample upper bound (it sees the test prices)
        te["pred_quad_sameday"] = np.nan
        for _, g in te.groupby(["symbol", "date", "expiry"]):
            if len(g) >= 5:
                te.loc[g.index, "pred_quad_sameday"] = np.polyval(np.polyfit(g["x"], g["iv"], 2), g["x"])
        for method in ("const", "quad_train", "rf", "quad_sameday"):
            p = te[f"pred_{method}"].clip(lower=0.01)
            ok = p.notna()
            price = black76_price(te.loc[ok, "F"], te.loc[ok, "strike"], te.loc[ok, "T"], te.loc[ok, "r"],
                                  p[ok], te.loc[ok, "option_type"])
            for sym in [None] + list(config.UNDERLYINGS):
                m = ok if sym is None else ok & (te["symbol"] == sym)
                pe = (price - te.loc[ok, "settle"])[m[ok].values] / te.loc[m, "S"]
                rows.append(dict(fold=fold, method=method, symbol=sym or "ALL", n=int(m.sum()),
                                 iv_rmse=np.sqrt(((p[m] - te.loc[m, "iv"]) ** 2).mean()),
                                 price_mae_bp=1e4 * pe.abs().mean()))
        preds[fold] = te
    return pd.DataFrame(rows), preds, models


def make_figures(cfg=config) -> list[str]:
    d = smile_data()
    qp = quad_params(d)
    qsum = qp.groupby(["symbol", "regime"])[["level_a", "skew_b", "curvature_c"]].median().reindex(
        pd.MultiIndex.from_product([list(config.UNDERLYINGS), REGIME_ORDER])).reset_index()
    qsum.columns = ["symbol", "regime", "level_a", "skew_b", "curvature_c"]
    metrics, preds, models = evaluate(d)
    paths = [save_table(qsum, "ml_smile_params"), save_table(metrics, "ml_rf_metrics")]

    fig, axes = plt.subplots(1, 2, figsize=(config.FIG_FULL, 2.3), sharey=True)
    for ax, fold in zip(axes, FOLDS):
        te = preds[fold]
        g = te[te["symbol"] == "NIFTY"]
        g = g[g["date"] == g["date"].min()]
        g = g[g["expiry"] == g.loc[g["days"] >= 7, "expiry"].min()].sort_values("moneyness")
        ax.plot(g["moneyness"], g["iv"], "o", ms=2, color="k", label="market")
        ax.plot(g["moneyness"], g["pred_rf"], "-", color=GREEN, lw=1, label="random forest")
        ax.plot(g["moneyness"], g["pred_quad_train"], "--", color=SLATE, lw=1, label="quadratic (train fit)")
        ax.plot(g["moneyness"], g["pred_const"], ":", color=GREY, lw=1, label="30d hist. (constant)")
        ax.set_title(f"fold {fold}: NIFTY {g['date'].iloc[0]:%d %b %Y}, {g['days'].iloc[0]}d", fontsize=8)
        ax.set_xlabel("$K/F$")
    axes[0].set_ylabel("Implied volatility")
    axes[0].legend(fontsize=6.5)
    fig.tight_layout()
    paths.append(savefig(fig, "ml_rf_iv_fit"))

    fig, ax = plt.subplots(figsize=(config.FIG_SINGLE, 1.8))
    imp = pd.DataFrame({f: models[f].feature_importances_ for f in FOLDS}, index=["ln(K/F)", "T", "σ 30d"])
    imp.plot.barh(ax=ax, color=[GREEN, SLATE], width=0.7)
    ax.set_xlabel("Impurity-based importance")
    ax.legend(fontsize=6.5, title="fold", title_fontsize=6.5)
    paths.append(savefig(fig, "ml_rf_importance"))
    return paths
