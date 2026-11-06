"""Load raw NSE files into tidy tables (PROJECT_PLAN.md §4.2).

* F&O bhavcopy, old format (before 8 Jul 2024) or UDiFF (after) -> one common option table
* Underlying OHLC history (niftyindices.com / NSE "Quote-Equity" CSVs) -> DataFrame by date
* Corporate actions (NSE CF-CA CSVs) -> dividend table

Raw files are read exactly as downloaded; nothing in data/raw is edited.
"""
import glob
import re

import numpy as np
import pandas as pd

import config

COLUMNS = ["date", "symbol", "instrument", "expiry", "strike", "option_type",
           "close", "settle", "volume", "open_interest", "underlying"]


# ---------------------------------------------------------------------------
# Bhavcopy
# ---------------------------------------------------------------------------
def _from_new(df: pd.DataFrame) -> pd.DataFrame:
    """UDiFF format (from 8 July 2024). IDO = index option, STO = stock option."""
    df = df[df["FinInstrmTp"].isin(["IDO", "STO"])]
    return pd.DataFrame({
        "date": pd.to_datetime(df["TradDt"]),
        "symbol": df["TckrSymb"].str.strip(),
        "instrument": df["FinInstrmTp"].map({"IDO": "index", "STO": "stock"}),
        "expiry": pd.to_datetime(df["XpryDt"]),
        "strike": df["StrkPric"].astype(float),
        "option_type": df["OptnTp"].str.strip().map({"CE": "C", "PE": "P"}),
        "close": df["ClsPric"].astype(float),
        "settle": df["SttlmPric"].astype(float),
        "volume": df["TtlTradgVol"].astype(float),
        "open_interest": df["OpnIntrst"].astype(float),
        "underlying": df["UndrlygPric"].astype(float),
    })


def _from_old(df: pd.DataFrame) -> pd.DataFrame:
    """Old format (before 8 July 2024). OPTIDX = index option, OPTSTK = stock option. No underlying price."""
    df = df.rename(columns=str.strip)
    df = df[df["INSTRUMENT"].isin(["OPTIDX", "OPTSTK"])]
    return pd.DataFrame({
        "date": pd.to_datetime(df["TIMESTAMP"], format="%d-%b-%Y"),
        "symbol": df["SYMBOL"].str.strip(),
        "instrument": df["INSTRUMENT"].map({"OPTIDX": "index", "OPTSTK": "stock"}),
        "expiry": pd.to_datetime(df["EXPIRY_DT"], format="%d-%b-%Y"),
        "strike": df["STRIKE_PR"].astype(float),
        "option_type": df["OPTION_TYP"].str.strip().map({"CE": "C", "PE": "P"}),
        "close": df["CLOSE"].astype(float),
        "settle": df["SETTLE_PR"].astype(float),
        "volume": df["CONTRACTS"].astype(float),
        "open_interest": df["OPEN_INT"].astype(float),
        "underlying": np.nan,
    })


def load_bhavcopy(path) -> pd.DataFrame:
    """One bhavcopy file (csv or csv.zip, either format) -> common option table."""
    df = pd.read_csv(path)
    out = _from_new(df) if "FinInstrmTp" in df.columns else _from_old(df)
    return out[COLUMNS].reset_index(drop=True)


def load_all(raw_dir=None, symbols=None) -> pd.DataFrame:
    """All bhavcopies in data/raw, restricted to config.UNDERLYINGS."""
    raw_dir = str(raw_dir or config.RAW_DIR)
    files = sorted(set(glob.glob(f"{raw_dir}/BhavCopy_NSE_FO*.csv*") + glob.glob(f"{raw_dir}/fo*bhav.csv*")))
    if not files:
        raise FileNotFoundError(f"No bhavcopy files in {raw_dir}")
    df = pd.concat([load_bhavcopy(f) for f in files], ignore_index=True)
    return df[df["symbol"].isin(symbols or list(config.UNDERLYINGS))].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Underlying price history
# ---------------------------------------------------------------------------
def _num(s: pd.Series) -> pd.Series:
    """'1,32,02,307' -> 13202307.0 (NSE uses Indian digit grouping)."""
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def load_spot(symbol: str, raw_dir=None) -> pd.DataFrame:
    """Daily open/high/low/close of the underlying, indexed by date, split/bonus-adjusted.

    Concatenates every file matching config.SPOT_FILES[symbol] and drops duplicate dates.
    """
    raw_dir = str(raw_dir or config.RAW_DIR)
    files = sorted(glob.glob(f"{raw_dir}/{config.SPOT_FILES[symbol]}"))
    if not files:
        raise FileNotFoundError(f"no spot history for {symbol} in {raw_dir}")
    parts = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig")
        df.columns = [c.strip().lower() for c in df.columns]
        dcol = next(c for c in df.columns if "date" in c)
        out = pd.DataFrame({c: _num(df[c]) for c in ("open", "high", "low", "close")})
        out.index = pd.to_datetime(df[dcol].astype(str).str.strip(), format="mixed", dayfirst=True)
        parts.append(out)
    s = pd.concat(parts).dropna(subset=["close"])
    s = s[~s.index.duplicated()].sort_index()
    for ex_date, ratio in config.SPLITS.get(symbol, []):
        s.loc[s.index < pd.Timestamp(ex_date)] /= ratio
    s.index.name = "date"
    return s


# ---------------------------------------------------------------------------
# Dividends
# ---------------------------------------------------------------------------
_RS = re.compile(r"Rs\.?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def parse_dividend(purpose: str) -> float:
    """'Interim Dividend - Rs 19 Per Share/Special Dividend - Rs 10 Per Share' -> 29.0; non-dividend -> 0."""
    if "dividend" not in str(purpose).lower():
        return 0.0
    return float(sum(float(x) for x in _RS.findall(str(purpose))))


def load_dividends(raw_dir=None) -> pd.DataFrame:
    """Cash dividends per share (ex-date, symbol, amount) from the NSE corporate-action files,
    plus config.EXTRA_DIVIDENDS for announcements after the download window."""
    raw_dir = str(raw_dir or config.RAW_DIR)
    rows = []
    for sym, pattern in config.CORP_ACTION_FILES.items():
        for f in glob.glob(f"{raw_dir}/{pattern}"):
            ca = pd.read_csv(f, encoding="utf-8-sig")
            ca.columns = [c.strip().upper() for c in ca.columns]
            for _, row in ca.iterrows():
                amt = parse_dividend(row["PURPOSE"])
                if amt > 0:
                    rows.append((pd.to_datetime(row["EX-DATE"], format="%d-%b-%Y"), sym, amt, "nse_corporate_actions"))
    for sym, d, amt in config.EXTRA_DIVIDENDS:
        rows.append((pd.Timestamp(d), sym, float(amt), "manual (config.EXTRA_DIVIDENDS)"))
    out = pd.DataFrame(rows, columns=["ex_date", "symbol", "amount", "source"])
    return out.drop_duplicates(["ex_date", "symbol"]).sort_values(["symbol", "ex_date"]).reset_index(drop=True)
