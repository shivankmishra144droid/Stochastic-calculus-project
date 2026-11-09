"""Clean the raw option table and attach every model input (PROJECT_PLAN.md §4.3).

Pipeline (order matters; each step's removals are recorded for the report):
 1. keep config.UNDERLYINGS                      (load_all)
 2. attach S, days, T, r, regime and volatility estimates (no look-ahead)
 3. filters: has S, settle > 0, price set by trades (NSE theoretical settlements dropped), volume, open interest,
    minimum premium, static arbitrage across strikes (monotone, convex, bounded slope; lowest-volume offender dropped), open interest, days to expiry, nearest expiries
 4. implied forward per (symbol, date, expiry) from put-call parity; moneyness = K / F
 5. moneyness filter; no-arbitrage lower bound on F; OTM flag (puts K < F, calls K >= F) for smiles
 6. implied volatility (Black-76 on F, and Black-Scholes on spot for the before/after comparison)
"""
import numpy as np
import pandas as pd

import config
from src.core.bs import implied_vol_array
from src.core.vol import as_of, ewma_vol, hist_vol, parkinson_vol
from src.data.forward import implied_forwards
from src.data.load import load_all, load_spot, total_return_close

VOL_COLS = ["hv30", "hv60", "hv252", "ewma", "park30"]


def vol_estimates(spot: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """All historical volatility estimates for one underlying, indexed by date.

    Close-to-close estimators use the dividend-adjusted close (an ex-date drop is not volatility);
    Parkinson uses the day's own high/low and needs no adjustment."""
    c = total_return_close(spot["close"], symbol) if symbol else spot["close"]
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
        vols = vol_estimates(spot, sym)
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


def _worst_strike_violation(K, P, V, call, disc, tol_abs, tol_rel):
    """Indices of the contracts in the worst static-arbitrage violation of one chain, or None.

    Checks, with K sorted: monotone (calls non-increasing, puts non-decreasing in K), slope |dP/dK| <= e^{-rT},
    and convexity (butterfly a P[i-1] + b P[i+1] - P[i] >= 0 with interpolation weights). A violation counts
    only beyond max(tol_abs, tol_rel * price)."""
    worst, idx = 0.0, None
    for i in range(len(K) - 1):
        dK, dP = K[i + 1] - K[i], P[i + 1] - P[i]
        tol = max(tol_abs, tol_rel * min(P[i], P[i + 1]))
        excess = max(dP if call else -dP, abs(dP) - disc * dK) - tol      # wrong direction, or too steep
        if excess > 0 and excess / tol > worst:
            worst, idx = excess / tol, (i, i + 1)
    for i in range(1, len(K) - 1):
        a = (K[i + 1] - K[i]) / (K[i + 1] - K[i - 1])
        bf = a * P[i - 1] + (1 - a) * P[i + 1] - P[i]
        tol = max(tol_abs, tol_rel * P[i])
        if -bf > tol and -bf / tol > worst:
            worst, idx = -bf / tol, (i - 1, i, i + 1)
    return idx


def strike_arbitrage_mask(df: pd.DataFrame, tol_abs=None, tol_rel=None) -> pd.Series:
    """True for contracts kept after static-arbitrage cleaning, chain by chain (symbol, date, expiry, type).

    Repeatedly finds the worst violation and drops the contract with the lowest volume in it (the least
    reliable price), until none exceeds the tolerance. Settlement prices are closing-half-hour VWAPs, so
    neighbouring strikes can be struck at different index levels (worst on fast days like 7 Apr 2025)."""
    tol_abs = config.ARB_TOL_ABS if tol_abs is None else tol_abs
    tol_rel = config.ARB_TOL_REL if tol_rel is None else tol_rel
    keep = pd.Series(True, index=df.index)
    for (_, _, _, o), g in df.groupby(["symbol", "date", "expiry", "option_type"]):
        g = g.sort_values("strike")
        disc = float(np.exp(-g["r"].iloc[0] * g["T"].iloc[0]))
        idx = list(g.index)
        while len(idx) >= 2:
            sub = g.loc[idx]
            bad = _worst_strike_violation(sub["strike"].to_numpy(float), sub["settle"].to_numpy(float),
                                          sub["volume"].to_numpy(float), o == "C", disc, tol_abs, tol_rel)
            if bad is None:
                break
            cand = [idx[j] for j in bad]
            drop = min(cand, key=lambda j: (g.at[j, "volume"], j))
            keep[drop] = False
            idx.remove(drop)
    return keep


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
    if config.TRADED_SETTLE_ONLY:
        step("price set by trades (settle = close)", np.isclose(df["settle"], df["close"]))
    step(f"days to expiry >= {config.MIN_DAYS}", df["days"] >= config.MIN_DAYS)
    rank = df.groupby(["symbol", "date"])["expiry"].rank(method="dense")
    step(f"nearest {config.MAX_EXPIRIES} expiries", rank <= config.MAX_EXPIRIES)
    step(f"volume >= {config.MIN_VOLUME}", df["volume"] >= config.MIN_VOLUME)
    step(f"open interest >= {config.MIN_OI}", df["open_interest"] >= config.MIN_OI)
    step(f"price >= Rs {config.MIN_PRICE:g} (tick <= 5%)", df["settle"] >= config.MIN_PRICE)
    step("no static arbitrage across strikes", strike_arbitrage_mask(df))

    fwd = implied_forwards(df)
    df = df.merge(fwd, on=["symbol", "date", "expiry"], how="left")
    step("implied forward available", df["F"].notna())
    df["moneyness"] = df["strike"] / df["F"]
    df["moneyness_spot"] = df["strike"] / df["S"]
    step(f"moneyness K/F in {config.MONEYNESS}", df["moneyness"].between(*config.MONEYNESS))
    # no-arbitrage lower bound on the implied forward: C >= e^{-rT}(F-K)^+, P >= e^{-rT}(K-F)^+.
    # Deep-ITM settlements below it (stale trades) admit no implied volatility and no model can match them.
    intrinsic = np.exp(-df["r"] * df["T"]) * np.where(df["option_type"] == "C", (df["F"] - df["strike"]).clip(lower=0),
                                                      (df["strike"] - df["F"]).clip(lower=0))
    step("price >= discounted intrinsic (no-arbitrage)", df["settle"] > intrinsic)
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
