"""Model vs market on NSE options (PROJECT_PLAN.md §5, Person 4).

Every model price uses the implied forward F (Black-76 for BS; a CRR tree started at F e^{-rT} with q = 0),
so carry and dividends are taken from the market and the comparison isolates volatility.

Figures: mkt_vol_estimates, mkt_smile_<SYMBOL>, mkt_forward_fix, mkt_error_heatmap, mkt_model_vs_input
Tables : data_filter_summary, mkt_data_summary, mkt_vol_estimates, mkt_errors, mkt_model_vs_input
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from src.core.binomial import binomial_price
from src.core.bs import black76_price
from src.data.clean import VOL_COLS, build_dataset
from src.plotting import GREEN, GREY, PALETTE, SLATE, save_table, savefig

REGIME_ORDER = ["calm", "normal", "volatile"]
VOL_LABELS = {"hv30": "30-day hist.", "hv60": "60-day hist.", "hv252": "252-day hist.",
              "ewma": "EWMA", "park30": "Parkinson 30d", "atm_iv": "ATM implied"}
MNY_BINS = [0.8, 0.9, 0.97, 1.03, 1.1, 1.2]
MNY_LABELS = ["0.80-0.90", "0.90-0.97", "0.97-1.03", "1.03-1.10", "1.10-1.20"]
MAT_BINS = [0, 7, 30, 1000]
MAT_LABELS = ["<=7d", "8-30d", ">30d"]


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------
def atm_iv(df: pd.DataFrame) -> pd.Series:
    """ATM implied vol per (symbol, date, expiry): IV of the OTM option with K/F closest to 1."""
    otm = df[df["otm"] & df["iv"].notna()].copy()
    otm["dist"] = (otm["moneyness"] - 1).abs()
    best = otm.sort_values("dist").groupby(["symbol", "date", "expiry"])["iv"].first()
    return df.set_index(["symbol", "date", "expiry"]).index.map(best).to_numpy(dtype=float)


def price_with(df: pd.DataFrame, vol_col: str, crr: bool = True) -> pd.DataFrame:
    """Black-76 and CRR(N=200) prices on the implied forward with sigma = df[vol_col]."""
    sig = df[vol_col].to_numpy(float)
    out = pd.DataFrame(index=df.index)
    out["bs"] = black76_price(df["F"], df["strike"], df["T"], df["r"], sig, df["option_type"])
    if crr:
        S_fwd = (df["F"] * np.exp(-df["r"] * df["T"])).to_numpy()
        out["crr"] = [binomial_price(s, k, t, r, v, config.CRR_STEPS_MARKET, o) if np.isfinite(v) and v > 0 else np.nan
                      for s, k, t, r, v, o in zip(S_fwd, df["strike"], df["T"], df["r"], sig, df["option_type"])]
    return out


def priced_dataset():
    """Clean dataset + ATM IV + BS/CRR prices for every volatility input. Saved for the ML modules."""
    df, summary = build_dataset(save=True, verbose=False)
    df["atm_iv"] = atm_iv(df)
    for v in VOL_COLS + ["atm_iv"]:
        p = price_with(df, v)
        df[f"bs_{v}"], df[f"crr_{v}"] = p["bs"], p["crr"]
    df["mny_bucket"] = pd.cut(df["moneyness"], MNY_BINS, labels=MNY_LABELS, include_lowest=True)
    df["mat_bucket"] = pd.cut(df["days"], MAT_BINS, labels=MAT_LABELS)
    df.to_csv(config.PROCESSED_DIR / "options_priced.csv", index=False)
    return df, summary


def load_priced() -> pd.DataFrame:
    """Priced options table from data/processed (built on first use)."""
    path = config.PROCESSED_DIR / "options_priced.csv"
    if not path.exists():
        return priced_dataset()[0]
    return pd.read_csv(path, parse_dates=["date", "expiry"])


# ---------------------------------------------------------------------------
# Tables and figures
# ---------------------------------------------------------------------------
def data_tables(df, summary):
    counts = df.groupby(["regime", "symbol"]).agg(
        contracts=("settle", "size"), dates=("date", "nunique"), expiries=("expiry", "nunique"),
        iv_solved=("iv", lambda s: s.notna().mean())).reset_index()
    return [save_table(summary, "data_filter_summary"), save_table(counts, "mkt_data_summary")]


def vol_estimates(df):
    """Per regime and symbol: mean of each historical estimate vs mean ATM IV of the nearest expiry (>= 7 days)."""
    near = df[df["days"] >= 7]
    near = near[near["expiry"] == near.groupby(["symbol", "date"])["expiry"].transform("min")]
    tab = near.groupby(["symbol", "regime"])[VOL_COLS + ["atm_iv"]].mean().reset_index()
    path_t = save_table(tab, "mkt_vol_estimates")
    fig, axes = plt.subplots(1, 3, figsize=(config.FIG_FULL, 2.3), sharey=True)
    cols = VOL_COLS + ["atm_iv"]
    w = 0.13
    for ax, sym in zip(axes, config.UNDERLYINGS):
        t = tab[tab["symbol"] == sym].set_index("regime").reindex(REGIME_ORDER)
        x = np.arange(len(REGIME_ORDER))
        for i, (c, col) in enumerate(zip(cols, PALETTE + ["#7C3AED"])):
            ax.bar(x + (i - 2.5) * w, t[c], w, color=col if c != "atm_iv" else "k", label=VOL_LABELS[c])
        ax.set_xticks(x, REGIME_ORDER)
        ax.set_title(sym, fontsize=9)
    axes[0].set_ylabel("Annualised volatility")
    axes[-1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    return [path_t, savefig(fig, "mkt_vol_estimates")]


def smiles(df):
    """OTM implied-vol smile (Black-76 on F) per regime, one panel per regime; first date of each regime."""
    paths = []
    for sym in config.UNDERLYINGS:
        fig, axes = plt.subplots(1, 3, figsize=(config.FIG_FULL, 2.3), sharey=True)
        for ax, reg in zip(axes, REGIME_ORDER):
            g = df[(df["symbol"] == sym) & (df["regime"] == reg)]
            if g.empty:
                continue
            d = g["date"].min()
            g = g[(g["date"] == d) & g["otm"] & g["iv"].notna()]
            for (exp, ge), col in zip(g.groupby("expiry"), [GREEN, SLATE, GREY]):
                ge = ge.sort_values("moneyness")
                ax.plot(ge["moneyness"], ge["iv"], ".-", ms=2.5, lw=0.8, color=col,
                        label=f"{(exp - d).days}d")
            ax.axhline(g["hv30"].iloc[0], color="k", ls="--", lw=0.8, label="30d hist.")
            ax.set_title(f"{reg} ({d:%d %b %Y})", fontsize=8)
            ax.set_xlabel("$K/F$")
            ax.legend(fontsize=6)
        axes[0].set_ylabel("Implied volatility")
        fig.tight_layout()
        paths.append(savefig(fig, f"mkt_smile_{sym}"))
    return paths


def forward_fix(df, sym="NIFTY", date="2025-01-02"):
    """Smile with spot-based BS IV (puts and calls do not meet at K/S = 1) vs Black-76 IV on the implied forward."""
    g = df[(df["symbol"] == sym) & (df["date"] == pd.Timestamp(date))]
    g = g[g["expiry"] == g["expiry"].min()]
    g = g[g["moneyness"].between(0.95, 1.05)]
    fig, axes = plt.subplots(1, 2, figsize=(config.FIG_FULL, 2.3), sharey=True)
    for ax, (x, y, title) in zip(axes, [("moneyness_spot", "iv_spot", "BS on spot $S$"),
                                       ("moneyness", "iv", "Black-76 on implied forward $F$")]):
        for o, col, lab in (("P", SLATE, "puts"), ("C", GREEN, "calls")):
            h = g[g["option_type"] == o].sort_values(x)
            ax.plot(h[x], h[y], ".-", ms=3, lw=0.8, color=col, label=lab)
        ax.axvline(1, color="k", lw=0.5, ls=":")
        ax.set_xlabel("$K/S$" if x == "moneyness_spot" else "$K/F$")
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("Implied volatility")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return savefig(fig, "mkt_forward_fix")


def error_tables(df):
    rows = []
    keys = ["symbol", "regime", "mny_bucket", "mat_bucket"]
    for v in VOL_COLS + ["atm_iv"]:
        for model in ("bs", "crr"):
            e = df[f"{model}_{v}"] - df["settle"]
            rel = e.abs() / df["settle"]
            t = pd.DataFrame({**{k: df[k] for k in keys}, "e": e, "rel": rel}).dropna(subset=["e"])
            agg = t.groupby(keys, observed=True).agg(
                n=("e", "size"), bias=("e", "mean"), mae=("e", lambda x: x.abs().mean()),
                rmse=("e", lambda x: np.sqrt((x**2).mean())), mean_rel_err=("rel", "mean")).reset_index()
            agg.insert(0, "vol_input", v)
            agg.insert(0, "model", model.upper())
            rows.append(agg)
    return save_table(pd.concat(rows, ignore_index=True), "mkt_errors")


def error_heatmap(df, sym="NIFTY"):
    """Mean relative pricing error |BS(hv30) - market| / market for OTM options, moneyness x maturity, per regime."""
    fig, axes = plt.subplots(1, 3, figsize=(config.FIG_FULL, 2.2), sharey=True)
    g = df[(df["symbol"] == sym) & df["otm"]].copy()
    g["rel"] = (g["bs_hv30"] - g["settle"]).abs() / g["settle"]
    vmax = 1.0
    for ax, reg in zip(axes, REGIME_ORDER):
        M = (g[g["regime"] == reg].pivot_table(index="mat_bucket", columns="mny_bucket", values="rel",
                                               aggfunc="mean", observed=False)
             .reindex(index=MAT_LABELS, columns=MNY_LABELS))
        im = ax.imshow(M.values, cmap="Greens", vmin=0, vmax=vmax, aspect="auto", origin="lower")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M.values[i, j]):
                    ax.text(j, i, f"{100 * M.values[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                            color="white" if M.values[i, j] > 0.6 * vmax else "k")
        ax.set_xticks(range(len(MNY_LABELS)), [l.replace("-", "–") for l in MNY_LABELS], rotation=45, fontsize=6.5)
        ax.set_yticks(range(len(MAT_LABELS)), MAT_LABELS)
        ax.set_title(reg, fontsize=8)
        ax.set_xlabel("$K/F$")
    fig.colorbar(im, ax=axes, label="mean |error| / price", shrink=0.9)
    return savefig(fig, "mkt_error_heatmap")


def model_vs_input(df):
    """Headline comparison, in basis points of the underlying, averaged over contracts:
       model choice  = |BS - CRR(200)| with the same sigma (hv30)
       input choice  = mean pairwise |BS(sigma_i) - BS(sigma_j)| over the five historical estimates
       market error  = |BS(hv30) - market|
       best constant sigma = |BS(ATM IV) - market| (what remains is the smile)
    """
    S = df["S"]
    bs = df[[f"bs_{v}" for v in VOL_COLS]].to_numpy()
    i, j = np.triu_indices(len(VOL_COLS), k=1)                  # distinct pairs only; NaN pairs (e.g. hv252) skipped
    pair = np.nanmean(np.abs(bs[:, i] - bs[:, j]), axis=1)
    t = pd.DataFrame({
        "symbol": df["symbol"], "regime": df["regime"],
        "model_choice": (df["bs_hv30"] - df["crr_hv30"]).abs() / S * 1e4,
        "input_choice": pair / S * 1e4,
        "market_error_hv30": (df["bs_hv30"] - df["settle"]).abs() / S * 1e4,
        "market_error_atm_iv": (df["bs_atm_iv"] - df["settle"]).abs() / S * 1e4,
    })
    tab = t.groupby(["symbol", "regime"]).mean().reindex(
        pd.MultiIndex.from_product([list(config.UNDERLYINGS), REGIME_ORDER])).reset_index()
    tab.columns = ["symbol", "regime"] + list(tab.columns[2:])
    path_t = save_table(tab, "mkt_model_vs_input")

    labels = {"model_choice": "model choice |BS − CRR|", "input_choice": "input choice (σ estimates)",
              "market_error_hv30": "|model − market|, σ = 30d hist.",
              "market_error_atm_iv": "|model − market|, σ = ATM IV"}
    fig, ax = plt.subplots(figsize=(config.FIG_FULL, 2.6))
    x = np.arange(len(tab))
    w = 0.2
    for i, (c, col) in enumerate(zip(labels, [GREY, GREEN, SLATE, "#B45309"])):
        ax.bar(x + (i - 1.5) * w, tab[c], w, color=col, label=labels[c])
    ax.set_yscale("log")
    short = {"NIFTY": "NIFTY", "RELIANCE": "RIL", "HINDUNILVR": "HUL"}
    ax.set_xticks(x, [f"{short[s]}\n{r}" for s, r in zip(tab["symbol"], tab["regime"])], fontsize=7)
    ax.set_ylabel("Mean absolute difference\n(bp of underlying)")
    ax.legend(fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    return [path_t, savefig(fig, "mkt_model_vs_input")]


def make_figures(cfg=config) -> list[str]:
    df, summary = priced_dataset()
    paths = data_tables(df, summary)
    paths += vol_estimates(df)
    paths += smiles(df)
    paths.append(forward_fix(df))
    paths.append(error_tables(df))
    paths.append(error_heatmap(df))
    paths += model_vs_input(df)
    return paths
