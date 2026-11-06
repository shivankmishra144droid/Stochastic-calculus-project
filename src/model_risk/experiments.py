"""Model flexibility and model risk (proposal §3 item 5, PROJECT_PLAN.md §5, Person 5).

Figures: mr_dividends, mr_timevarying, mr_sigma_vs_N
Tables : mr_dividends_synthetic, mr_dividends_market, mr_timevarying, mr_sigma_vs_N
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from src.core.binomial import (binomial_price, binomial_price_cash_div, binomial_price_discrete_div,
                               binomial_price_timevarying, bs_escrowed)
from src.core.bs import bs_price
from src.data.load import load_dividends
from src.plotting import GREEN, GREY, PALETTE, SLATE, save_table, savefig

S0, K0, R0, SIG0, T0 = config.S0, config.K0, config.R0, config.SIGMA0, config.T0
N_TREE = 500


def dividend_models(S, K, T, r, sig, divs, o, annual_div):
    """Prices under the dividend treatments compared in the report (divs = [(t, D)])."""
    return {
        "BS, no dividend": bs_price(S, K, T, r, sig, o),
        "A: BS escrowed": bs_escrowed(S, K, T, r, sig, divs, o),
        "B: BS yield q = D_annual/S": bs_price(S, K, T, r, sig, o, annual_div / S),
        "C: tree, proportional drop": binomial_price_discrete_div(S, K, T, r, sig, N_TREE, divs, o, "proportional"),
        "D: tree, escrowed": binomial_price_discrete_div(S, K, T, r, sig, N_TREE, divs, o, "escrow"),
        "E: tree, cash drop on ex-date": binomial_price_cash_div(S, K, T, r, sig, N_TREE, divs, o),
    }


# ---------------------------------------------------------------------------
# 1. Discrete dividends
# ---------------------------------------------------------------------------
def dividends_synthetic():
    """S = 100, D = 3 paid at t = 0.5, T = 1: prices of each treatment across strikes."""
    divs = [(0.5, 3.0)]
    rows = []
    for K in np.arange(80, 121, 2.5):
        for o in ("C", "P"):
            m = dividend_models(S0, K, T0, R0, SIG0, divs, o, annual_div=3.0)
            rows.append(dict(K=K, option_type=o, **m))
    return pd.DataFrame(rows)


def dividends_market(df):
    """Reliance options on 14-18 Jul 2025 expiring after the Rs 5.50 ex-date (14 Aug 2025).

    Each model prices off spot S with sigma = the market ATM implied vol of that expiry, so only the
    dividend treatment differs. Also reports the dividend implied by the put-call-parity forward:
        D_impl = (S e^{rT} - F) e^{-r (T - t_ex)}.
    """
    divtab = load_dividends()
    out = []
    for _, dv in divtab.iterrows():
        sel = df[(df["symbol"] == dv["symbol"]) & (df["date"] < dv["ex_date"]) & (df["expiry"] > dv["ex_date"])]
        for _, row in sel.iterrows():
            t_ex = (dv["ex_date"] - row["date"]).days / 365
            divs = [(t_ex, dv["amount"])]
            m = dividend_models(row["S"], row["strike"], row["T"], row["r"], row["atm_iv"], divs,
                                row["option_type"], annual_div=dv["amount"])
            d_impl = (row["S"] * np.exp(row["r"] * row["T"]) - row["F"]) * np.exp(-row["r"] * (row["T"] - t_ex))
            out.append(dict(symbol=dv["symbol"], date=row["date"], expiry=row["expiry"], strike=row["strike"],
                            option_type=row["option_type"], settle=row["settle"], S=row["S"], dividend=dv["amount"],
                            implied_dividend=d_impl, **m))
    return pd.DataFrame(out)


def dividends(df):
    syn = dividends_synthetic()
    mkt = dividends_market(df)
    models = [c for c in syn.columns if c[:2] in ("BS", "A:", "B:", "C:", "D:", "E:")]
    paths = [save_table(syn, "mr_dividends_synthetic")]
    if len(mkt):
        err = pd.DataFrame({m: (mkt[m] - mkt["settle"]) for m in models})
        summ = pd.DataFrame({"model": models, "n": len(mkt),
                             "bias": err.mean().values, "mae": err.abs().mean().values,
                             "rmse": np.sqrt((err**2).mean()).values})
        summ["implied_dividend_median"] = mkt.groupby("date")["implied_dividend"].median().median()
        summ["announced_dividend"] = mkt["dividend"].iloc[0]
        # noise floor: the same parity "carry gap" S e^{rT} - F on expiries with NO dividend before expiry
        sym = mkt["symbol"].iloc[0]
        g = df[df["symbol"] == sym].groupby(["date", "expiry"]).first().reset_index()
        divs = load_dividends()
        exd = divs.loc[divs["symbol"] == sym, "ex_date"]
        nodiv = g[[not ((exd > d) & (exd < e)).any() for d, e in zip(g["date"], g["expiry"])]]
        gap = nodiv["S"] * np.exp(nodiv["r"] * nodiv["T"]) - nodiv["F"]
        summ["nodiv_carry_gap_std"] = gap.std()
        summ["nodiv_carry_gap_min"] = gap.min()
        summ["nodiv_carry_gap_max"] = gap.max()
        paths.append(save_table(summ, "mr_dividends_market"))

    fig, axes = plt.subplots(1, 2, figsize=(config.FIG_FULL, 2.4))
    c = syn[syn["option_type"] == "C"]
    colours = dict(zip(models, [GREY, "k", SLATE, "#B45309", "#2563EB", GREEN]))
    for m in models:
        if m == "A: BS escrowed":
            continue
        axes[0].plot(c["K"], c[m] - c["A: BS escrowed"], color=colours[m], lw=1.4 if m.startswith("E") else 1,
                     ls=":" if m.startswith("D") else "-", label=m)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel("Strike $K$ ($S=100$, $D=3$ at $t=0.5$)")
    axes[0].set_ylabel("Call price − escrowed BS")
    axes[0].legend(fontsize=6)
    if len(mkt):
        calls = mkt[mkt["option_type"] == "C"].copy()
        calls["mny"] = calls["strike"] / calls["S"]
        for m in models:
            g = calls.assign(e=calls[m] - calls["settle"]).groupby(pd.cut(calls["mny"], 8), observed=True)
            axes[1].plot(g["mny"].mean(), g["e"].mean(), ".-", color=colours[m], lw=1, ms=3, label=m)
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].set_xlabel("RELIANCE calls, $K/S$ (Jul 2025, ex-date 14 Aug)")
        axes[1].set_ylabel("Model − market (Rs)")
    fig.tight_layout()
    paths.append(savefig(fig, "mr_dividends"))
    return paths


# ---------------------------------------------------------------------------
# 2. Time-varying volatility
# ---------------------------------------------------------------------------
def forward_vol_curve(Ts, ivs):
    """Piecewise-constant forward volatility from an ATM implied-vol term structure:
    sigma_i^2 = (iv_i^2 T_i - iv_{i-1}^2 T_{i-1}) / (T_i - T_{i-1}) (clipped at a small positive floor)."""
    Ts, ivs = np.asarray(Ts), np.asarray(ivs)
    w = ivs**2 * Ts
    fwd_var = np.diff(np.concatenate([[0.0], w])) / np.diff(np.concatenate([[0.0], Ts]))
    fwd = np.sqrt(np.maximum(fwd_var, 1e-6))
    return lambda t: fwd[min(np.searchsorted(Ts, t, side="right"), len(fwd) - 1)], fwd


def timevarying(df, sym="NIFTY", date="2025-04-07"):
    """Tree with sigma(t) (Amin) vs BS with the average variance sigma_bar^2 = (1/T) int sigma^2 dt."""
    g = df[(df["symbol"] == sym) & (df["date"] == pd.Timestamp(date))]
    ts = g.groupby("expiry").agg(T=("T", "first"), iv=("atm_iv", "first"), F=("F", "first"), r=("r", "first"))
    ts = ts.dropna().sort_values("T")
    sig_fn, fwd = forward_vol_curve(ts["T"], ts["iv"])
    T, r, F = ts["T"].iloc[-1], ts["r"].iloc[-1], ts["F"].iloc[-1]
    S = F * np.exp(-r * T)
    sig_bar = ts["iv"].iloc[-1]                     # total variance / T of the last expiry
    rows = []
    for mny in (0.95, 1.0, 1.05):
        K = round(mny * F, -1)
        ref = bs_price(S, K, T, r, sig_bar, "C")
        for N in (25, 50, 100, 200, 400, 800):
            tv = binomial_price_timevarying(S, K, T, lambda t: r, sig_fn, N, "C")
            const = binomial_price(S, K, T, r, sig_bar, N, "C")
            rows.append(dict(K=K, K_over_F=mny, N=N, bs_avg_var=ref, tree_timevarying=tv, tree_const=const,
                             diff_tv_bs=tv - ref, diff_const_bs=const - ref))
    tab = pd.DataFrame(rows)

    fig, ax = plt.subplots(1, 2, figsize=(config.FIG_FULL, 2.3))
    edges = np.concatenate([[0], ts["T"].values]) * 365
    ax[0].stairs(fwd, edges, color=GREEN, lw=1.2, label=r"forward vol $\sigma(t)$")
    ax[0].plot(ts["T"] * 365, ts["iv"], "o", color=SLATE, ms=3, label="ATM implied vol")
    ax[0].axhline(sig_bar, color="k", ls="--", lw=0.8, label=r"$\bar\sigma$ (average variance)")
    ax[0].set_xlabel(f"Days ({sym}, {pd.Timestamp(date):%d %b %Y})")
    ax[0].set_ylabel("Volatility")
    ax[0].legend(fontsize=6.5)
    atm = tab[tab["K_over_F"] == 1.0]
    ax[1].loglog(atm["N"], atm["diff_tv_bs"].abs(), "o-", color=GREEN, ms=3, label=r"time-varying tree − BS($\bar\sigma$)")
    ax[1].loglog(atm["N"], atm["diff_const_bs"].abs(), "s--", color=GREY, ms=3, label=r"constant-$\bar\sigma$ CRR − BS($\bar\sigma$)")
    ax[1].set_xlabel("Steps $N$")
    ax[1].set_ylabel("|difference| (index points)")
    ax[1].legend(fontsize=6.5)
    fig.tight_layout()
    return [save_table(tab, "mr_timevarying"), savefig(fig, "mr_timevarying")]


# ---------------------------------------------------------------------------
# 3. Volatility error vs discretisation error
# ---------------------------------------------------------------------------
def sigma_vs_N():
    ref = bs_price(S0, K0, T0, R0, SIG0, "C")
    rows = []
    for rel in (0.01, 0.02, 0.05):
        up = bs_price(S0, K0, T0, R0, SIG0 * (1 + rel), "C") - ref
        dn = bs_price(S0, K0, T0, R0, SIG0 * (1 - rel), "C") - ref
        rows.append(dict(source=f"σ ±{rel:.0%} (relative)", abs_price_error=0.5 * (abs(up) + abs(dn))))
    for N in (10, 50, 200):
        e = 0.5 * (abs(binomial_price(S0, K0, T0, R0, SIG0, N) - ref) + abs(binomial_price(S0, K0, T0, R0, SIG0, N + 1) - ref))
        rows.append(dict(source=f"CRR N = {N}", abs_price_error=e))
    tab = pd.DataFrame(rows)
    tab["pct_of_price"] = 100 * tab["abs_price_error"] / ref
    fig, ax = plt.subplots(figsize=(config.FIG_SINGLE, 2.3))
    cols = [GREEN] * 3 + [GREY] * 3
    ax.barh(tab["source"][::-1], tab["abs_price_error"][::-1], color=cols[::-1])
    ax.set_xscale("log")
    ax.set_xlabel("Absolute call price error (baseline, $C=10.45$)")
    fig.tight_layout()
    return [save_table(tab, "mr_sigma_vs_N"), savefig(fig, "mr_sigma_vs_N")]


def make_figures(cfg=config) -> list[str]:
    path = config.PROCESSED_DIR / "options_priced.csv"
    if not path.exists():
        from src.market.analysis import priced_dataset
        priced_dataset()
    df = pd.read_csv(path, parse_dates=["date", "expiry"])
    return dividends(df) + timevarying(df) + sigma_vs_N()
