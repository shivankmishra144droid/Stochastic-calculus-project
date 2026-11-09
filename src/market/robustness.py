"""Robustness of the headline result to two choices the limitations section flags.

1. Liquidity filters. The main dataset keeps contracts with volume >= 10, open interest >= 50 and premium >= Rs 1.
   We rebuild the dataset with looser and stricter filters and recompute the error decomposition
   (numerical |BS - CRR(200)|, input choice, market error with 30-day hist. and with ATM IV), with date-bootstrap CIs.
   Trade-set settlements, static-arbitrage cleaning, maturity and moneyness filters stay on in every variant:
   without them the "market price" is NSE's own model price or an arbitrage, not a liquidity choice.

2. Clock. Volatilities are annualised over 252 trading days, but T counts calendar days / 365. On a trading-day
   clock the variance to expiry is sigma^2 * n_td / 252, where n_td is the number of trading-day returns to expiry
   (weekends and NSE holidays excluded). We reprice with that variance (discounting stays on calendar time) and
   compare market errors by maturity. ATM IV is fitted to the same day's prices, so its prices do not depend on
   the clock; only historical inputs move.

Tables: rob_liquidity, rob_clock
"""
from contextlib import contextmanager

import numpy as np
import pandas as pd

import config
from src.core.bs import black76_price
from src.data.clean import VOL_COLS, add_implied_vols, attach_inputs, filter_options
from src.data.load import load_all, load_spot
from src.market.analysis import MAT_BINS, MAT_LABELS, atm_iv, load_priced, price_with
from src.plotting import save_table
from src.stats import cluster_mean_ci

LIQUIDITY_VARIANTS = {
    "loosest (vol >= 1, no OI, no min price)": dict(MIN_VOLUME=1, MIN_OI=0, MIN_PRICE=0.05),
    "loose (vol >= 1, no OI)": dict(MIN_VOLUME=1, MIN_OI=0),
    "baseline (vol >= 10, OI >= 50)": dict(),
    "strict (vol >= 100, OI >= 500)": dict(MIN_VOLUME=100, MIN_OI=500),
}
METRICS = ["numerical_bs_vs_crr", "input_choice", "market_error_hv30", "market_error_atm_iv"]


@contextmanager
def overrides(**kw):
    """Temporarily set config attributes."""
    old = {k: getattr(config, k) for k in kw}
    try:
        for k, v in kw.items():
            setattr(config, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(config, k, v)


def decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Per-contract error components in bp of the underlying (same definitions as market.analysis.model_vs_input)."""
    S = df["S"]
    bs = np.column_stack([price_with(df, v, crr=False)["bs"] for v in VOL_COLS])
    i, j = np.triu_indices(len(VOL_COLS), k=1)
    with np.errstate(invalid="ignore"):
        pair = np.nanmean(np.abs(bs[:, i] - bs[:, j]), axis=1)
    p30 = price_with(df, "hv30", crr=True)
    p_atm = price_with(df, "atm_iv", crr=False)
    return pd.DataFrame({
        "date": df["date"],
        "numerical_bs_vs_crr": (p30["bs"] - p30["crr"]).abs() / S * 1e4,
        "input_choice": pair / S * 1e4,
        "market_error_hv30": (p30["bs"] - df["settle"]).abs() / S * 1e4,
        "market_error_atm_iv": (p_atm["bs"] - df["settle"]).abs() / S * 1e4,
    })


def liquidity_table(attached: pd.DataFrame | None = None) -> pd.DataFrame:
    attached = attach_inputs(load_all()) if attached is None else attached
    rows = []
    for name, kw in LIQUIDITY_VARIANTS.items():
        with overrides(**kw):
            df, _ = filter_options(attached)
        df = add_implied_vols(df)
        df["atm_iv"] = atm_iv(df)
        t = decomposition(df)
        ci = cluster_mean_ci(t, METRICS, "date")
        row = dict(variant=name, n_contracts=len(df), n_dates=int(ci["n_clusters"].iloc[0]),
                   median_volume=float(df["volume"].median()))
        for m in METRICS:
            row[m], row[f"{m}_lo"], row[f"{m}_hi"] = ci.loc[m, ["mean", "lo", "hi"]]
        row["input_over_numerical"] = row["input_choice"] / row["numerical_bs_vs_crr"]
        row["smile_over_numerical"] = row["market_error_atm_iv"] / row["numerical_bs_vs_crr"]
        rows.append(row)
    return pd.DataFrame(rows)


def trading_holidays() -> pd.DatetimeIndex:
    """Weekdays missing from the NIFTY history (NSE holidays) within its date range."""
    idx = load_spot("NIFTY").index
    weekdays = pd.bdate_range(idx.min(), idx.max())
    return weekdays.difference(idx)


def trading_days_to_expiry(date: pd.Series, expiry: pd.Series, holidays=None) -> np.ndarray:
    """Number of trading-day returns from the close on `date` to the close on `expiry` (days in (date, expiry]).
    Holidays are known only up to the end of the price history; after that only weekends are excluded."""
    holidays = trading_holidays() if holidays is None else holidays
    d0 = (date + pd.Timedelta(days=1)).to_numpy("datetime64[D]")
    d1 = (expiry + pd.Timedelta(days=1)).to_numpy("datetime64[D]")
    return np.busday_count(d0, d1, holidays=np.asarray(holidays, dtype="datetime64[D]"))


def clock_table(df: pd.DataFrame | None = None, inputs=("hv30", "ewma")) -> pd.DataFrame:
    df = load_priced() if df is None else df
    n_td = trading_days_to_expiry(df["date"], df["expiry"])
    scale = np.sqrt((n_td / 252.0) / df["T"].to_numpy())          # sigma_eff on the calendar clock
    t = pd.DataFrame({"date": df["date"], "bucket": pd.cut(df["days"], MAT_BINS, labels=MAT_LABELS)})
    cols = []
    for v in inputs:
        sig = df[v].to_numpy(float)
        for clock, s in (("calendar", sig), ("trading", sig * scale)):
            price = black76_price(df["F"], df["strike"], df["T"], df["r"], s, df["option_type"])
            t[f"{v}|{clock}"] = 1e4 * np.abs(price - df["settle"]) / df["S"]
        t[f"{v}|diff"] = t[f"{v}|trading"] - t[f"{v}|calendar"]
        cols += [f"{v}|calendar", f"{v}|trading", f"{v}|diff"]
    rows = []
    for bucket in MAT_LABELS + ["ALL"]:
        g = t if bucket == "ALL" else t[t["bucket"] == bucket]
        sub = n_td if bucket == "ALL" else n_td[(t["bucket"] == bucket).to_numpy()]
        ci = cluster_mean_ci(g, cols, "date")
        for v in inputs:
            rows.append(dict(maturity=bucket, sigma=v, n_contracts=len(g),
                             var_ratio_trading_over_calendar=float(np.mean((sub / 252) / (g.index.map(
                                 df["T"]).to_numpy()))),
                             calendar_bp=ci.loc[f"{v}|calendar", "mean"], trading_bp=ci.loc[f"{v}|trading", "mean"],
                             diff_bp=ci.loc[f"{v}|diff", "mean"], diff_lo=ci.loc[f"{v}|diff", "lo"],
                             diff_hi=ci.loc[f"{v}|diff", "hi"]))
    return pd.DataFrame(rows)


def make_figures(cfg=config) -> list[str]:
    return [save_table(liquidity_table(), "rob_liquidity"), save_table(clock_table(), "rob_clock")]
