"""Offline tests of the NSE data pipeline on synthetic files (no real data needed).

Fake bhavcopies (both NSE formats) are generated from Black-76 with a known skewed smile
and a known forward, so the pipeline must recover both.
"""
import numpy as np
import pandas as pd
import pytest

import config
from src.core.bs import black76_price
from src.data import clean, load
from src.data.forward import forward_from_pairs, implied_forwards


def smile(m):
    return 0.14 + 0.35 * (m - 1.0) ** 2 - 0.10 * (m - 1.0)


R, CARRY = 0.06, 0.02          # forward F = S e^{CARRY T} (rate minus dividend yield)


def fake_chain(date, S, symbol="NIFTY"):
    rows = []
    for days in (2, 9, 30, 65, 90):
        T = days / 365
        F = S * np.exp(CARRY * T)
        for K in np.arange(round(S * 0.75, -1), S * 1.25, S * 0.01):
            for o in ("CE", "PE"):
                price = black76_price(F, K, T, R, smile(K / F), o[0])
                rows.append(dict(expiry=date + pd.Timedelta(days=days), K=round(K, 2), o=o,
                                 price=round(price, 2), S=S, symbol=symbol))
    return pd.DataFrame(rows)


def write_new(path, date, df):
    pd.DataFrame({
        "TradDt": date.strftime("%Y-%m-%d"), "FinInstrmTp": "IDO", "TckrSymb": df["symbol"],
        "XpryDt": df["expiry"].dt.strftime("%Y-%m-%d"), "StrkPric": df["K"], "OptnTp": df["o"],
        "ClsPric": df["price"], "SttlmPric": df["price"], "TtlTradgVol": 100,
        "OpnIntrst": 500, "UndrlygPric": df["S"],
    }).to_csv(path, index=False)


def write_old(path, date, df):
    pd.DataFrame({
        "INSTRUMENT": "OPTIDX", "SYMBOL": df["symbol"], "EXPIRY_DT": df["expiry"].dt.strftime("%d-%b-%Y"),
        "STRIKE_PR": df["K"], "OPTION_TYP": df["o"], "CLOSE": df["price"], "SETTLE_PR": df["price"],
        "CONTRACTS": 100, "OPEN_INT": 500, "TIMESTAMP": date.strftime("%d-%b-%Y").upper(),
    }).to_csv(path, index=False)


@pytest.fixture
def fake_raw(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    dates = pd.bdate_range("2023-01-02", "2025-10-31")
    rng = np.random.default_rng(0)
    close = 20000 * np.exp(np.cumsum(rng.normal(0, 0.15 / np.sqrt(252), len(dates))))
    spot = pd.DataFrame({"Index Name": "NIFTY 50", "Date": dates.strftime("%d %b %Y"),
                         "Open": close, "High": close * 1.004, "Low": close * 0.996, "Close": close})
    spot.iloc[:400].to_csv(raw / "spot_part1.csv", index=False)       # split across two files,
    spot.iloc[390:].to_csv(raw / "spot_part2.csv", index=False)        # with overlapping dates
    s_close = pd.Series(close, index=dates)
    d_old, d_new = pd.Timestamp("2024-03-15"), pd.Timestamp("2025-10-01")
    write_old(raw / "fo15MAR2024bhav.csv", d_old, fake_chain(d_old, float(s_close[d_old])))
    write_new(raw / "BhavCopy_NSE_FO_0_0_0_20251001_F_0000.csv", d_new, fake_chain(d_new, float(s_close[d_new])))
    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "SPOT_FILES", {"NIFTY": "spot_part*.csv"})
    monkeypatch.setattr(config, "UNDERLYINGS", {"NIFTY": "index"})
    monkeypatch.setattr(config, "RISK_FREE", {"2024-03-15": R, "2025-10-01": R})
    monkeypatch.setattr(config, "DATE_TO_REGIME", {"2024-03-15": "a", "2025-10-01": "b"})
    return raw


def test_load_both_formats_and_spot(fake_raw):
    df = load.load_all()
    assert set(df["date"].dt.year) == {2024, 2025}
    assert set(df["option_type"]) == {"C", "P"}
    assert df.loc[df["date"].dt.year == 2024, "underlying"].isna().all()   # old format has no underlying
    spot = load.load_spot("NIFTY")
    assert spot.index.is_unique and spot.index.is_monotonic_increasing
    assert len(spot) == len(pd.bdate_range("2023-01-02", "2025-10-31"))


def test_forward_recovery_synthetic():
    S, T, r, F = 100.0, 0.25, 0.06, 101.3
    K = np.arange(80, 121, 2.5)
    C = black76_price(F, K, T, r, 0.2, "C")
    P = black76_price(F, K, T, r, 0.2, "P")
    assert forward_from_pairs(K, C, P, r, T, S) == pytest.approx(F, rel=1e-10)
    # with prices rounded to the NSE tick (0.05) the error stays well below 0.1%
    Fh = forward_from_pairs(K, np.round(C / 0.05) * 0.05, np.round(P / 0.05) * 0.05, r, T, S)
    assert abs(Fh / F - 1) < 1e-3


def test_full_pipeline_recovers_forward_and_smile(fake_raw):
    df = clean.attach_inputs(load.load_all())
    df, summary = clean.filter_options(df)
    assert (df["days"] >= config.MIN_DAYS).all()
    assert df.groupby(["symbol", "date"])["expiry"].nunique().max() <= config.MAX_EXPIRIES
    assert summary["removed"].sum() == summary["rows_left"].iloc[0] - len(df)
    true_F = df["S"] * np.exp(CARRY * df["T"])
    assert np.allclose(df["F"], true_F, rtol=1e-3)
    df = clean.add_implied_vols(df)
    good = df[(df["days"] > 20) & df["moneyness"].between(0.9, 1.1) & df["iv"].notna()]
    assert len(good) > 50
    assert np.allclose(good["iv"], smile(good["moneyness"]), atol=3e-3)
    # vol estimates are present and close to the simulated 15%
    assert df["hv252"].dropna().between(0.10, 0.20).all()


def test_missing_rate_is_an_error(fake_raw, monkeypatch):
    monkeypatch.setattr(config, "RISK_FREE", {})
    with pytest.raises(KeyError):
        clean.attach_inputs(load.load_all())


@pytest.mark.parametrize("text,amount", [
    ("Annual General Meeting/Dividend - Rs 24 Per Share", 24.0),
    ("Interim Dividend - Rs 19 Per Share/Special Dividend - Rs 10 Per Share", 29.0),
    ("Dividend - Rs 5.50 Per Share", 5.5),
    ("Bonus 1:1", 0.0),
])
def test_parse_dividend(text, amount):
    assert load.parse_dividend(text) == amount


def test_implied_forwards_needs_pairs():
    df = pd.DataFrame(dict(symbol="X", date=pd.Timestamp("2025-01-01"), expiry=pd.Timestamp("2025-02-01"),
                           strike=[100.0], option_type=["C"], settle=[5.0], S=100.0, r=0.05, T=0.1))
    assert np.isnan(implied_forwards(df)["F"].iloc[0])


def test_dividends_are_split_adjusted():
    """Reliance's Rs 10 (19 Aug 2024) was paid before the 1:1 bonus: Rs 5 per current share, like load_spot."""
    dv = load.load_dividends()
    r = dv[(dv["symbol"] == "RELIANCE") & (dv["ex_date"] == pd.Timestamp("2024-08-19"))].iloc[0]
    assert r["amount_declared"] == 10.0 and r["amount"] == 5.0
    later = dv[(dv["symbol"] == "RELIANCE") & (dv["ex_date"] == pd.Timestamp("2025-08-14"))].iloc[0]
    assert later["amount"] == later["amount_declared"] == 5.5


def test_total_return_close_removes_dividend_drop():
    idx = pd.bdate_range("2025-01-01", periods=5)
    close = pd.Series([100.0, 101.0, 98.0, 98.5, 99.0], index=idx)       # ex-date on day 3, D = 3
    divs = pd.DataFrame({"ex_date": [idx[2]], "symbol": ["X"], "amount": [3.0]})
    tr = load.total_return_close(close, "X", divs)
    r = np.log(tr).diff()
    assert r.iloc[2] == pytest.approx(np.log((98.0 + 3.0) / 101.0))       # the drop is no longer a loss
    assert (tr.iloc[2:] == close.iloc[2:]).all()                           # later prices untouched


def test_short_sessions_dropped_and_bonus_adjusted():
    s = load.load_spot("RELIANCE")
    assert not s.index.isin(pd.to_datetime(config.SHORT_SESSIONS)).any()
    assert pd.Timestamp("2024-01-20") in s.index                           # full Saturday session kept
    r = np.log(s["close"]).diff()
    assert abs(r.loc["2024-10-28"]) < 0.05                                 # no fake -69% at the 1:1 bonus


def test_theoretical_settlements_filtered(fake_raw, monkeypatch):
    df = clean.attach_inputs(load.load_all())
    df.loc[df.index[:7], "settle"] = df.loc[df.index[:7], "close"] + 1.0   # NSE theoretical price != close
    out, summary = clean.filter_options(df)
    assert np.isclose(out["settle"], out["close"]).all()
    assert "price set by trades (settle = close)" in set(summary["step"])


def test_load_vix():
    v = load.load_vix()
    assert v.index.is_unique and v.index.is_monotonic_increasing
    assert 8 < v.min() and v.max() < 40                                    # India VIX is quoted in percent
    nifty = load.load_spot("NIFTY").loc[v.index.min():v.index.max()]
    assert set(nifty.index) == set(v.index)                                # same trading calendar


def test_below_intrinsic_prices_filtered(fake_raw):
    df = clean.attach_inputs(load.load_all())
    df, _ = clean.filter_options(df)
    disc = np.exp(-df["r"] * df["T"])
    intr = disc * np.where(df["option_type"] == "C", (df["F"] - df["strike"]).clip(lower=0),
                           (df["strike"] - df["F"]).clip(lower=0))
    assert (df["settle"] > intr).all()


def test_strike_arbitrage_filter_removes_planted_outlier():
    """A chain of Black-76 calls with one mispriced, low-volume strike: only that strike is dropped, and the
    clean chain is left intact (tick-size noise is tolerated)."""
    K = np.arange(90.0, 111.0, 2.5)
    P = np.round(black76_price(100.0, K, 0.25, 0.05, 0.2, "C") / 0.05) * 0.05
    df = pd.DataFrame(dict(symbol="X", date=pd.Timestamp("2025-01-02"), expiry=pd.Timestamp("2025-04-02"),
                           option_type="C", strike=K, settle=P, volume=100.0, r=0.05, T=0.25))
    assert clean.strike_arbitrage_mask(df).all()
    bad = df.copy()
    bad.loc[4, "settle"] += 3.0                       # middle strike too expensive: convexity broken
    bad.loc[4, "volume"] = 12.0
    m = clean.strike_arbitrage_mask(bad)
    assert not m[4] and m.drop(4).all()
