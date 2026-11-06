"""ML benchmark 5: HAR regression volatility forecast (Corsi 2009), fed into BS and the tree (Person 5).

Daily variance proxy RV_t: Parkinson ln^2(H/L)/(4 ln 2) (plan default) or squared close-to-close return.
Direct h-step model, h in {5, 22} trading days:
    mean(RV_{t+1..t+h}) = b0 + b_d RV_t + b_w mean(RV_{t-4..t}) + b_m mean(RV_{t-21..t}) + e
Fitted on an expanding window that only uses targets fully observed by the option date (no look-ahead).
Forecast sigma = sqrt(252 * forecast) prices every option with Black-76 on the implied forward.

Figures: ml_har_forecast, ml_har_pricing
Tables : ml_har_forecast_eval, ml_har_pricing
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import config
from src.core.binomial import binomial_price
from src.core.bs import black76_price
from src.core.vol import parkinson_var_daily
from src.data.load import load_spot
from src.market.analysis import REGIME_ORDER, load_priced
from src.plotting import GREEN, GREY, PALETTE, SLATE, save_table, savefig

HORIZONS = (5, 22)


def daily_rv(spot: pd.DataFrame, proxy: str = "parkinson") -> pd.Series:
    if proxy == "parkinson":
        return parkinson_var_daily(spot["high"], spot["low"]).rename("rv")
    return (np.log(spot["close"] / spot["close"].shift(1)) ** 2).rename("rv")


def design(rv: pd.Series, h: int) -> pd.DataFrame:
    """Features at t and the h-day-ahead target; `target_end` = date the target is fully observed."""
    df = pd.DataFrame({"rv_d": rv, "rv_w": rv.rolling(5).mean(), "rv_m": rv.rolling(22).mean()})
    df["y"] = rv[::-1].rolling(h).mean()[::-1].shift(-1)          # mean of RV_{t+1..t+h}
    pos = np.arange(len(rv))
    end = np.minimum(pos + h, len(rv) - 1)
    df["target_end"] = rv.index[end]
    df.loc[pos + h > len(rv) - 1, "y"] = np.nan
    return df


def har_forecast(rv: pd.Series, date, h: int):
    """Annualised volatility forecast for the next h trading days, made at the close of `date`.

    Returns (sigma, last_target_date_used). Only rows whose target window ended on or before `date` train the model.
    """
    D = design(rv, h)
    date = pd.Timestamp(date)
    hist = D[D.index <= date]
    train = hist[(hist["target_end"] <= date)].dropna()
    x_now = hist[["rv_d", "rv_w", "rv_m"]].dropna().iloc[[-1]]
    if len(train) < 60:
        return np.nan, None
    reg = LinearRegression().fit(train[["rv_d", "rv_w", "rv_m"]], train["y"])
    f = float(reg.predict(x_now)[0])
    f = max(f, 0.25 * float(x_now["rv_m"].iloc[0]))                # guard against negative forecasts
    return np.sqrt(252 * f), train["target_end"].max()


def forecast_eval(sym="NIFTY", h=22, start="2024-09-01", proxy="parkinson"):
    """Rolling out-of-sample forecast vs realised (same proxy), compared with the naive monthly average."""
    spot = load_spot(sym)
    rv = daily_rv(spot, proxy).dropna()
    D = design(rv, h)
    rows = []
    for t in rv.index[(rv.index >= pd.Timestamp(start))]:
        if pd.isna(D.loc[t, "y"]):
            continue
        sig, _ = har_forecast(rv, t, h)
        rows.append(dict(date=t, har=sig, naive_rv_m=np.sqrt(252 * D.loc[t, "rv_m"]),
                         realised=np.sqrt(252 * D.loc[t, "y"]),
                         hv30=np.log(spot["close"]).diff().loc[:t].tail(30).std() * np.sqrt(252)))
    return pd.DataFrame(rows)


def option_sigmas(df: pd.DataFrame, proxy: str) -> np.ndarray:
    out = np.full(len(df), np.nan)
    cache = {}
    for sym in df["symbol"].unique():
        rv = daily_rv(load_spot(sym), proxy).dropna()
        for i in np.where(df["symbol"] == sym)[0]:
            row = df.iloc[i]
            h = 5 if row["days"] * 252 / 365 <= 10 else 22
            key = (sym, row["date"], h)
            if key not in cache:
                cache[key] = har_forecast(rv, row["date"], h)[0]
            out[i] = cache[key]
    return out


def make_figures(cfg=config) -> list[str]:
    paths = []
    ev = pd.concat([forecast_eval(proxy=p).assign(proxy=p) for p in ("parkinson", "squared_return")])
    summ = ev.groupby("proxy").apply(lambda g: pd.Series({
        "n": len(g),
        "rmse_har": np.sqrt(((g["har"] - g["realised"]) ** 2).mean()),
        "rmse_naive_rv_m": np.sqrt(((g["naive_rv_m"] - g["realised"]) ** 2).mean()),
        "rmse_hv30": np.sqrt(((g["hv30"] - g["realised"]) ** 2).mean()),
    }), include_groups=False).reset_index()
    paths.append(save_table(summ, "ml_har_forecast_eval"))

    fig, ax = plt.subplots(figsize=(config.FIG_FULL, 2.2))
    e = ev[ev["proxy"] == "squared_return"]
    ax.plot(e["date"], e["realised"], color=GREY, lw=0.8, label="realised (next 22 days)")
    ax.plot(e["date"], e["har"], color=GREEN, lw=1, label="HAR forecast")
    ax.plot(e["date"], e["hv30"], color=SLATE, lw=0.8, ls="--", label="30-day historical")
    for reg, ds in config.REGIMES.items():
        ax.axvspan(pd.Timestamp(ds[0]), pd.Timestamp(ds[-1]), color=GREEN, alpha=0.12)
    ax.set_ylabel("NIFTY volatility")
    ax.legend(fontsize=7, ncol=3)
    paths.append(savefig(fig, "ml_har_forecast"))

    df = load_priced().reset_index(drop=True)
    for proxy in ("parkinson", "squared_return"):
        df[f"har_{proxy}"] = option_sigmas(df, proxy)
        df[f"bs_har_{proxy}"] = black76_price(df["F"], df["strike"], df["T"], df["r"], df[f"har_{proxy}"], df["option_type"])
    # tree price with the HAR input, on a subsample (the tree and Black-76 agree to < 0.2 bp; see mkt_model_vs_input)
    sub = df.sample(min(500, len(df)), random_state=config.SEED)
    tree_gap = [abs(binomial_price(F * np.exp(-r * T), K, T, r, s, config.CRR_STEPS_MARKET, o) - b) / S * 1e4
                for F, K, T, r, s, o, b, S in zip(sub["F"], sub["strike"], sub["T"], sub["r"],
                                                  sub["har_squared_return"], sub["option_type"],
                                                  sub["bs_har_squared_return"], sub["S"]) if np.isfinite(s)]
    inputs = {"hv30": "bs_hv30", "ewma": "bs_ewma", "park30": "bs_park30",
              "HAR (Parkinson)": "bs_har_parkinson", "HAR (sq. returns)": "bs_har_squared_return"}
    rows = []
    for (sym, reg), g in df.groupby(["symbol", "regime"]):
        for name, col in inputs.items():
            rows.append(dict(symbol=sym, regime=reg, vol_input=name,
                             mae_bp=1e4 * ((g[col] - g["settle"]).abs() / g["S"]).mean()))
    tab = pd.DataFrame(rows)
    tab["tree_vs_bs_gap_bp_mean"] = float(np.mean(tree_gap))
    paths.append(save_table(tab, "ml_har_pricing"))

    fig, axes = plt.subplots(1, 3, figsize=(config.FIG_FULL, 2.3), sharey=True)
    for ax, sym in zip(axes, config.UNDERLYINGS):
        t = tab[tab["symbol"] == sym].pivot(index="regime", columns="vol_input", values="mae_bp").reindex(REGIME_ORDER)
        t = t[list(inputs)]
        x = np.arange(len(t))
        w = 0.16
        for i, (c, col) in enumerate(zip(t.columns, PALETTE)):
            ax.bar(x + (i - 2) * w, t[c], w, color=col, label=c)
        ax.set_xticks(x, t.index)
        ax.set_title(sym, fontsize=8)
    axes[0].set_ylabel("Pricing MAE (bp of underlying)")
    axes[-1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    paths.append(savefig(fig, "ml_har_pricing"))
    return paths
