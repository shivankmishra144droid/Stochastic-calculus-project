"""ML chooses the sigma: can a model combine our volatility estimates into a better sigma than any single one?

Inputs (one row per underlying and trading day, all known at that day's close, no option prices):
    hv30, hv60, hv252, EWMA, Parkinson 30-day (src.data.clean) and India VIX / 100.
    hv252 needs 252 returns and is missing for NIFTY in early January 2025; there it is filled with hv60.
Target: that day's ATM implied volatility of the nearest expiry with >= 7 days (what the market uses).

Models
    LR (all)       linear regression on all six inputs
    RF (all)       random forest on all six inputs
    LR (30d+VIX)   linear regression on 30-day historical and VIX only (backward- plus forward-looking)

Each model is tested on data it has not seen, in two ways:
    unseen day     leave one trading day out (train on the other 13 days, all regimes)
    unseen regime  leave one volatility regime out (train on the other two regimes)
The predicted sigma then prices every contract of that day with Black-76 on the implied forward, exactly like the
single estimators, and pricing errors are compared in bp of the underlying with bootstrap CIs over trading dates.

Figures: ml_sigma_blend
Tables : ml_sigma_blend_iv, ml_sigma_blend_pricing
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import config
from src.core.bs import black76_price
from src.data.load import load_vix
from src.market.analysis import load_priced
from src.plotting import GREEN, GREY, SLATE, save_table, savefig
from src.stats import cluster_mean_ci

SINGLE = {"hv30": "30-day hist.", "hv60": "60-day hist.", "hv252": "252-day hist.", "ewma": "EWMA",
          "park30": "Parkinson 30d", "vix": "India VIX"}
ALL_INPUTS = list(SINGLE)
MODELS = {
    "LR (all)": (lambda: LinearRegression(), ALL_INPUTS),
    "RF (all)": (lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=config.SEED),
                 ALL_INPUTS),
    "LR (30d+VIX)": (lambda: LinearRegression(), ["hv30", "vix"]),
}
DESIGNS = {"unseen day": "date", "unseen regime": "regime"}
MIN_DAYS_TARGET = 7


def daily_table(df: pd.DataFrame, vix: pd.Series | None = None) -> pd.DataFrame:
    """One row per (symbol, date): the six inputs and the ATM implied vol of the nearest expiry with >= 7 days."""
    near = df[df["days"] >= MIN_DAYS_TARGET]
    near = near[near["expiry"] == near.groupby(["symbol", "date"])["expiry"].transform("min")]
    cols = ["hv30", "hv60", "hv252", "ewma", "park30"]
    t = near.groupby(["symbol", "date"]).agg(regime=("regime", "first"), atm_iv=("atm_iv", "first"),
                                             **{c: (c, "first") for c in cols}).reset_index()
    t["hv252_filled"] = t["hv252"].isna()
    t["hv252"] = t["hv252"].fillna(t["hv60"])
    vix = load_vix() / 100 if vix is None else vix
    t["vix"] = vix.reindex(pd.DatetimeIndex(t["date"])).to_numpy()
    if t[ALL_INPUTS + ["atm_iv"]].isna().any().any():
        raise ValueError("missing input or target in the daily table")
    return t


def out_of_sample(t: pd.DataFrame, make_model, inputs, group: str) -> pd.Series:
    """Prediction for every row from a model fitted WITHOUT that row's group (trading day or regime)."""
    pred = pd.Series(np.nan, index=t.index)
    for g in t[group].unique():
        test = t[group] == g
        model = make_model().fit(t.loc[~test, inputs], t.loc[~test, "atm_iv"])
        pred[test] = model.predict(t.loc[test, inputs])
    return pred.clip(lower=0.01)


def predictions(t: pd.DataFrame) -> pd.DataFrame:
    """The six inputs plus the out-of-sample sigma of every model under both test designs
    (columns '<model> | <design>'), one row per (symbol, date)."""
    out = t[["symbol", "date"] + ALL_INPUTS].copy()
    for name, (make, inputs) in MODELS.items():
        for design, group in DESIGNS.items():
            out[f"{name} | {design}"] = out_of_sample(t, make, inputs, group)
    return out


def pricing_errors(df: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    """|Black-76(sigma) - settlement| in bp of the underlying for every contract and every sigma choice."""
    d = df.drop(columns=[c for c in ALL_INPUTS if c in df.columns]).merge(
        sig, on=["symbol", "date"], how="left", validate="many_to_one")       # inputs as in the daily table
    choices = {**{k: d[k] for k in ALL_INPUTS},
               **{c: d[c] for c in sig.columns if " | " in c},
               "ATM IV (same day, in-sample)": d["atm_iv"]}
    err = pd.DataFrame({"date": d["date"]})
    for name, s in choices.items():
        price = black76_price(d["F"], d["strike"], d["T"], d["r"], s.to_numpy(float), d["option_type"])
        err[name] = 1e4 * np.abs(price - d["settle"]) / d["S"]
    return err


def make_figures(cfg=config) -> list[str]:
    df = load_priced()
    t = daily_table(df)
    sig = predictions(t)

    # 1. how well each sigma matches the ATM implied vol (one value per underlying and day)
    rows = [dict(sigma=SINGLE.get(c, c), rmse_vs_atm_iv=float(np.sqrt(((sig[c] - t["atm_iv"]) ** 2).mean())))
            for c in ALL_INPUTS + [c for c in sig.columns if " | " in c]]
    iv_tab = pd.DataFrame(rows).assign(n_days=len(t))
    paths = [save_table(iv_tab, "ml_sigma_blend_iv")]

    # 2. pricing error of every contract, with CIs over trading dates; paired differences vs 30-day and vs EWMA
    err = pricing_errors(df, sig)
    names = [c for c in err.columns if c != "date"]
    best_single = min(ALL_INPUTS, key=lambda c: err[c].mean())
    for c in names:
        err[f"{c} - hv30"] = err[c] - err["hv30"]
        err[f"{c} - best"] = err[c] - err[best_single]
    ci = cluster_mean_ci(err, [c for c in err.columns if c != "date"], "date")
    tab = pd.DataFrame([dict(
        sigma=SINGLE.get(c, c), kind=("single estimate" if c in SINGLE else "reference" if c.startswith("ATM")
                                      else c.split(" | ")[1]),
        mae_bp=ci.loc[c, "mean"], mae_lo=ci.loc[c, "lo"], mae_hi=ci.loc[c, "hi"],
        minus_hv30_bp=ci.loc[f"{c} - hv30", "mean"], minus_hv30_lo=ci.loc[f"{c} - hv30", "lo"],
        minus_hv30_hi=ci.loc[f"{c} - hv30", "hi"],
        minus_best_single_bp=ci.loc[f"{c} - best", "mean"], minus_best_single_lo=ci.loc[f"{c} - best", "lo"],
        minus_best_single_hi=ci.loc[f"{c} - best", "hi"], best_single=SINGLE[best_single],
        n_contracts=int(ci.loc[c, "n"]), n_dates=int(ci.loc[c, "n_clusters"])) for c in names])
    paths.append(save_table(tab, "ml_sigma_blend_pricing"))

    # 3. one bar chart: pricing error of each sigma choice
    order = ([SINGLE[c] for c in ALL_INPUTS] + [f"{m} | unseen day" for m in MODELS]
             + [f"{m} | unseen regime" for m in MODELS] + ["ATM IV (same day, in-sample)"])
    p = tab.set_index("sigma").loc[order].reset_index()
    colour = {"single estimate": GREY, "unseen day": GREEN, "unseen regime": SLATE, "reference": "#B45309"}
    fig, ax = plt.subplots(figsize=(config.FIG_FULL, 3.3))
    y = np.arange(len(p))[::-1]
    ax.barh(y, p["mae_bp"], color=[colour[k] for k in p["kind"]],
            xerr=np.vstack([p["mae_bp"] - p["mae_lo"], p["mae_hi"] - p["mae_bp"]]),
            error_kw=dict(lw=0.6, capsize=1.5, ecolor="k"))
    ax.set_yticks(y, p["sigma"], fontsize=7)
    ax.axvline(p.loc[p["sigma"] == SINGLE[best_single], "mae_bp"].iloc[0], color="k", ls=":", lw=0.7)
    ax.set_xlabel("Mean absolute pricing error (bp of underlying), 95% CI over trading dates")
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in colour.values()]
    ax.legend(handles, ["single estimate", "ML, tested on unseen days", "ML, tested on an unseen regime",
                        "ATM implied vol (reference)"], fontsize=6.5, ncol=2, loc="upper center",
              bbox_to_anchor=(0.4, -0.2))
    fig.tight_layout()
    paths.append(savefig(fig, "ml_sigma_blend"))
    return paths
