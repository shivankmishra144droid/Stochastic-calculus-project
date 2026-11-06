"""Clean the raw option table and attach every model input (PROJECT_PLAN.md §4.3).

Pipeline (order matters; each step's removals are recorded for the report):
 1. keep config.UNDERLYINGS                      (load_all)
 2. attach S, days, T, r, regime and volatility estimates (no look-ahead)
 3. filters: has S, settle > 0, volume, open interest, days to expiry, nearest expiries
 4. implied forward per (symbol, date, expiry) from put-call parity; moneyness = K / F
 5. moneyness filter; OTM flag (puts K < F, calls K >= F) for smiles
 6. implied volatility (Black-76 on F, and Black-Scholes on spot for the before/after comparison)
"""
import numpy as np
import pandas as pd

import config
from src.core.bs import implied_vol_array
from src.core.vol import as_of, ewma_vol, hist_vol, parkinson_vol
from src.data.forward import implied_forwards
from src.data.load import load_all, load_spot

VOL_COLS = ["hv30", "hv60", "hv252", "ewma", "park30"]


def vol_estimates(spot: pd.DataFrame) -> pd.DataFrame:
    """All historical volatility estimates for one underlying, indexed by date."""
    c = spot["close"]
    return pd.DataFrame({
        "hv30": hist_vol(c, 30), "hv60": hist_vol(c, 60), "hv252": hist_vol(c, 252),
        "ewma": ewma_vol(c), "park30": parkinson_vol(spot["high"], spot["low"], 30),
    })


def attach_inputs(df: pd.DataFrame, spots: dict | None = None) -> pd.DataFrame:
    """Add S (bhavcopy underlying, else spot close), days, T = days/365, r, regime and vol estimates."""
    df = df.copy()
    parts = []
    for sym, g in df.groupby("symbol"):
        spot = spots[sym] if spots is not None else load_spot(sym)
        vols = vol_estimates(spot)
        g = g.assign(spot_close=as_of(spot["close"], g["date"]))
        for col in VOL_COLS:
            g[col] = as_of(vols[col], g["date"]) if vols[col].notna().any() else np.nan
        # hv252 needs 252 returns; the series must actually reach back that far on this date
        first_valid = vols["hv252"].first_valid_index()
        if first_valid is None:
            g["hv252"] = np.nan
        else:
            g.loc[g["date"] < first_valid, "hv252"] = np.nan
        parts.append(g)
    df = pd.concat(parts)
    df["S"] = df["underlying"].where(df["underlying"] > 0).fillna(df["spot_close"])
    df["days"] = (df["expiry"] - df["date"]).dt.days
    df["T"] = df["days"] / 365.0
    dstr = df["date"].dt.strftime("%Y-%m-%d")
    missing = sorted(set(dstr) - set(config.RISK_FREE))
    if missing:
        raise KeyError(f"no risk-free rate in config.RISK_FREE for {missing}")
    df["r"] = dstr.map(config.RISK_FREE)
    df["regime"] = dstr.map(config.DATE_TO_REGIME).fillna("other")
    return df.reset_index(drop=True)


def filter_options(df: pd.DataFrame):
    """Liquidity / maturity / moneyness filters + implied forward + OTM flag.

    Returns (clean_df, summary) where summary lists rows removed by each step.
    """
    log = [("loaded (3 underlyings)", len(df), 0)]

    def step(name, mask):
        nonlocal df
        before = len(df)
        df = df[mask].copy()
        log.append((name, len(df), before - len(df)))

    step("has spot price", df["S"].notna())
    step("settlement > 0", df["settle"] > 0)
    step(f"days to expiry >= {config.MIN_DAYS}", df["days"] >= config.MIN_DAYS)
    rank = df.groupby(["symbol", "date"])["expiry"].rank(method="dense")
    step(f"nearest {config.MAX_EXPIRIES} expiries", rank <= config.MAX_EXPIRIES)
    step(f"volume >= {config.MIN_VOLUME}", df["volume"] >= config.MIN_VOLUME)
    step(f"open interest >= {config.MIN_OI}", df["open_interest"] >= config.MIN_OI)

    fwd = implied_forwards(df)
    df = df.merge(fwd, on=["symbol", "date", "expiry"], how="left")
    step("implied forward available", df["F"].notna())
    df["moneyness"] = df["strike"] / df["F"]
    df["moneyness_spot"] = df["strike"] / df["S"]
    step(f"moneyness K/F in {config.MONEYNESS}", df["moneyness"].between(*config.MONEYNESS))
    df["otm"] = ((df["option_type"] == "P") & (df["strike"] < df["F"])) | \
                ((df["option_type"] == "C") & (df["strike"] >= df["F"]))
    summary = pd.DataFrame(log, columns=["step", "rows_left", "removed"])
    return df.reset_index(drop=True), summary


def add_implied_vols(df: pd.DataFrame) -> pd.DataFrame:
    """iv: Black-76 implied vol on the implied forward; iv_spot: Black-Scholes on spot with q = 0."""
    df = df.copy()
    df["iv"] = implied_vol_array(df["settle"], df["F"], df["strike"], df["T"], df["r"], df["option_type"], forward=True)
    df["iv_spot"] = implied_vol_array(df["settle"], df["S"], df["strike"], df["T"], df["r"], df["option_type"])
    return df


def build_dataset(save: bool = True, verbose: bool = True):
    """Full pipeline on data/raw -> (options table, filter summary). Writes data/processed/options_clean.csv."""
    raw = load_all()
    df = attach_inputs(raw)
    df, summary = filter_options(df)
    df = add_implied_vols(df)
    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.PROCESSED_DIR / "options_clean.csv", index=False)
    if verbose:
        print(summary.to_string(index=False))
        print(f"implied vol solved for {df['iv'].notna().mean():.1%} of {len(df)} contracts")
    return df, summary


def load_dataset(rebuild: bool = False):
    """Cached options table (rebuilds when missing or when rebuild=True)."""
    path = config.PROCESSED_DIR / "options_clean.csv"
    if rebuild or not path.exists():
        return build_dataset(verbose=False)[0]
    return pd.read_csv(path, parse_dates=["date", "expiry"])
