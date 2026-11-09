"""Model flexibility and model risk (proposal §3 item 5, PROJECT_PLAN.md §5, Person 5).

Figures: mr_dividends, mr_timevarying, mr_sigma_vs_N
Tables : mr_dividends_synthetic, mr_dividends_market, mr_timevarying, mr_sigma_vs_N
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from src.core.binomial import (binomial_price, binomial_price_cash_div, binomial_price_discrete_div,
                               binomial_price_timevarying, bs_escrowed, dividend_yield_equivalent)
from src.core.bs import bs_price
from src.data.load import load_dividends
from src.plotting import GREEN, GREY, PALETTE, SLATE, save_table, savefig

S0, K0, R0, SIG0, T0 = config.S0, config.K0, config.R0, config.SIGMA0, config.T0
N_TREE = 500


def dividend_models(S, K, T, r, sig, divs, o):
    """Prices under the dividend treatments compared in the report (divs = [(t, D)]).

    B uses the continuous yield with the SAME forward as the cash dividends (dividend_yield_equivalent).
    A European BS price depends on S and q only through S e^{-qT}, so B is identical to A (escrow); likewise
    C = D (see binomial_price_discrete_div). Only E (cash drop with sigma on the full price) is a different model.
    """
    return {
        "BS, no dividend": bs_price(S, K, T, r, sig, o),
        "A: BS escrowed": bs_escrowed(S, K, T, r, sig, divs, o),
        "B: BS yield q, same forward": bs_price(S, K, T, r, sig, o, dividend_yield_equivalent(S, T, r, divs)),
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
            m = dividend_models(S0, K, T0, R0, SIG0, divs, o)
            rows.append(dict(K=K, option_type=o, **m))
    return pd.DataFrame(rows)


CARRY_MIN_DAYS = 30      # shorter expiries give a carry rate too noisy to use (error in F/S is divided by T)
N_BOOT = 2000


def expiry_carry(df, sym):
    """One row per (date, expiry) of `sym`: S, F, T, r, implied carry rate ln(F/S)/T, its spread over the T-bill
    rate, and whether an ex-dividend date falls between the trade date and expiry."""
    g = df[df["symbol"] == sym].groupby(["date", "expiry"]).agg(
        S=("S", "first"), F=("F", "first"), T=("T", "first"), r=("r", "first"), days=("days", "first"),
        regime=("regime", "first")).reset_index()
    divs = load_dividends()
    exd = divs.loc[divs["symbol"] == sym, "ex_date"]
    g["spans_dividend"] = [((exd > d) & (exd < e)).any() for d, e in zip(g["date"], g["expiry"])]
    g["r_carry"] = np.log(g["F"] / g["S"]) / g["T"]
    g["spread"] = g["r_carry"] - g["r"]
    return g


def carry_spread(carry, min_days=CARRY_MIN_DAYS):
    """Funding spread over the T-bill rate implied by parity on expiries WITHOUT a dividend:
    T^2-weighted mean of ln(F/S)/T - r over expiries with >= min_days (the error of ln(F/S)/T scales like 1/T)."""
    c = carry[~carry["spans_dividend"] & (carry["days"] >= min_days)]
    if c.empty:
        return np.nan
    return float(np.average(c["spread"], weights=c["T"] ** 2))


def implied_dividend(S, F, T, t_ex, r):
    """D_impl = (S e^{rT} - F) e^{-r (T - t_ex)}: the cash dividend that makes the carry-r forward equal F."""
    return (S * np.exp(r * T) - F) * np.exp(-r * (T - t_ex))


def dividends_market(df, spreads=None):
    """Reliance options on 14-18 Jul 2025 expiring after the Rs 5.50 ex-date (14 Aug 2025).

    Each model prices off spot S with sigma = the market ATM implied vol of that expiry, so only the
    dividend treatment differs. Two carry assumptions:
      * T-bill:     the stock grows at the T-bill rate r (as in the rest of the project);
      * calibrated: it grows at r + s, s = carry_spread() from the same stock's no-dividend expiries.
        This is implemented by pricing from S e^{sT} (discounting stays at r), which shifts the forward exactly.
    Also reports the parity-implied dividend under both carry rates.
    """
    divtab = load_dividends()
    spreads = spreads or {}
    out = []
    for _, dv in divtab.iterrows():
        sel = df[(df["symbol"] == dv["symbol"]) & (df["date"] < dv["ex_date"]) & (df["expiry"] > dv["ex_date"])]
        s = spreads.get(dv["symbol"], 0.0)
        for _, row in sel.iterrows():
            t_ex = (dv["ex_date"] - row["date"]).days / 365
            divs = [(t_ex, dv["amount"])]
            m = dividend_models(row["S"], row["strike"], row["T"], row["r"], row["atm_iv"], divs,
                                row["option_type"])
            mc = dividend_models(row["S"] * np.exp(s * row["T"]), row["strike"], row["T"], row["r"], row["atm_iv"],
                                 divs, row["option_type"])
            out.append(dict(symbol=dv["symbol"], date=row["date"], expiry=row["expiry"], strike=row["strike"],
                            option_type=row["option_type"], settle=row["settle"], S=row["S"], dividend=dv["amount"],
                            carry_spread=s,
                            implied_dividend=implied_dividend(row["S"], row["F"], row["T"], t_ex, row["r"]),
                            implied_dividend_calibrated=implied_dividend(row["S"], row["F"], row["T"], t_ex,
                                                                         row["r"] + s),
                            **m, **{f"{k} | calibrated carry": v for k, v in mc.items()}))
    return pd.DataFrame(out)


def implied_dividend_bootstrap(carry, ex_date, n_boot=N_BOOT, seed=config.SEED):
    """Date-clustered bootstrap of the calibrated implied dividend.

    Trade dates are resampled with replacement; for each resample the spread is re-estimated from the
    no-dividend expiries of the drawn dates, and the implied dividend is the median over the drawn dates'
    dividend-spanning expiries. Returns (point estimate, 2.5%, 97.5%) and the same for the T-bill carry.
    """
    rng = np.random.default_rng(seed)
    dates = carry["date"].unique()
    span = carry[carry["spans_dividend"]].copy()
    span["t_ex"] = (pd.Timestamp(ex_date) - span["date"]).dt.days / 365

    def estimate(c, sp):
        s = carry_spread(c)
        if not np.isfinite(s) or sp.empty:
            return np.nan, np.nan
        cal = implied_dividend(sp["S"], sp["F"], sp["T"], sp["t_ex"], sp["r"] + s)
        raw = implied_dividend(sp["S"], sp["F"], sp["T"], sp["t_ex"], sp["r"])
        return float(np.median(cal)), float(np.median(raw))

    point = estimate(carry, span)
    by_date = {d: (carry[carry["date"] == d], span[span["date"] == d]) for d in dates}
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(dates, len(dates), replace=True)
        c = pd.concat([by_date[d][0] for d in draw])
        sp = pd.concat([by_date[d][1] for d in draw])
        boots.append(estimate(c, sp))
    b = np.array(boots, dtype=float)
    ci = np.nanpercentile(b, [2.5, 97.5], axis=0)
    return dict(cal=point[0], cal_lo=ci[0, 0], cal_hi=ci[1, 0], raw=point[1], raw_lo=ci[0, 1], raw_hi=ci[1, 1],
                n_boot_valid=int(np.isfinite(b[:, 0]).sum()))


def dividends(df):
    syn = dividends_synthetic()
    models = [c for c in syn.columns if c[:2] in ("BS", "A:", "B:", "C:", "D:", "E:")]
    paths = [save_table(syn, "mr_dividends_synthetic")]
    carries = {sym: expiry_carry(df, sym) for sym in config.CORP_ACTION_FILES}
    spreads = {sym: carry_spread(c) for sym, c in carries.items()}
    paths.append(save_table(pd.concat([c.assign(symbol=s) for s, c in carries.items()]), "mr_carry_by_expiry"))
    mkt = dividends_market(df, spreads)
    if len(mkt):
        sym = mkt["symbol"].iloc[0]
        rows = []
        for carry_label, suffix in (("T-bill", ""), ("calibrated", " | calibrated carry")):
            for m in models:
                e = mkt[m + suffix] - mkt["settle"]
                rows.append(dict(carry=carry_label, model=m, n=len(mkt), bias=e.mean(), mae=e.abs().mean(),
                                 rmse=np.sqrt((e**2).mean())))
        summ = pd.DataFrame(rows)
        dv = load_dividends()
        dv = dv[(dv["symbol"] == sym) & (dv["ex_date"] > mkt["date"].max())
                & (dv["ex_date"] < mkt["expiry"].min())]["ex_date"]
        assert len(dv) == 1, "the bootstrap assumes one ex-date inside the spanning expiries"
        boot = implied_dividend_bootstrap(carries[sym], dv.iloc[0])
        nodiv = carries[sym][~carries[sym]["spans_dividend"]]
        long_ = nodiv[nodiv["days"] >= CARRY_MIN_DAYS]
        summ = summ.assign(
            announced_dividend=mkt["dividend"].iloc[0],
            carry_spread=spreads[sym], carry_spread_n_expiries=len(long_),
            carry_spread_sd_long=long_["spread"].std(), carry_spread_sd_short=nodiv.loc[
                nodiv["days"] < CARRY_MIN_DAYS, "spread"].std(),
            implied_dividend_tbill=boot["raw"], implied_dividend_tbill_lo=boot["raw_lo"],
            implied_dividend_tbill_hi=boot["raw_hi"],
            implied_dividend_calibrated=boot["cal"], implied_dividend_calibrated_lo=boot["cal_lo"],
            implied_dividend_calibrated_hi=boot["cal_hi"], n_boot_valid=boot["n_boot_valid"])
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
