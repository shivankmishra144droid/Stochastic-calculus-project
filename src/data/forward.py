"""Implied forward from put-call parity (PROJECT_PLAN.md §4.4).

Using spot S ignores carry (rate minus dividends) and NSE settlement timing, which shows up as a
jump between the put and call smiles at K/S = 1. For European options, parity gives
    C_K - P_K = e^{-rT} (F - K)   =>   F_K = K + e^{rT} (C_K - P_K).
We take the median of F_K over the strikes nearest spot where both a call and a put trade.
"""
import numpy as np
import pandas as pd


def forward_from_pairs(strikes, calls, puts, r, T, spot, n_strikes=5) -> float:
    """Median F_K over the n_strikes strikes closest to `spot`; NaN if no call/put pair exists."""
    strikes, calls, puts = map(np.asarray, (strikes, calls, puts))
    if len(strikes) == 0:
        return np.nan
    idx = np.argsort(np.abs(strikes - spot))[:n_strikes]
    return float(np.median(strikes[idx] + np.exp(r * T) * (calls[idx] - puts[idx])))


def implied_forwards(df: pd.DataFrame, n_strikes: int = 5, price_col: str = "settle") -> pd.DataFrame:
    """One implied forward per (symbol, date, expiry).

    df needs: symbol, date, expiry, strike, option_type, S, r, T and `price_col`.
    Returns columns symbol, date, expiry, F, n_pairs.
    """
    out = []
    keys = ["symbol", "date", "expiry"]
    for key, g in df.groupby(keys):
        piv = g.pivot_table(index="strike", columns="option_type", values=price_col, aggfunc="first")
        piv = piv.dropna(subset=[c for c in ("C", "P") if c in piv.columns])
        if not {"C", "P"} <= set(piv.columns) or piv.empty:
            out.append((*key, np.nan, 0))
            continue
        r, T, S = g["r"].iloc[0], g["T"].iloc[0], g["S"].iloc[0]
        F = forward_from_pairs(piv.index.values, piv["C"].values, piv["P"].values, r, T, S, n_strikes)
        out.append((*key, F, len(piv)))
    return pd.DataFrame(out, columns=keys + ["F", "n_pairs"])
