"""Volatility estimators from price history (proposal §2.4, PROJECT_PLAN.md §3.3).

All functions return annualised volatilities as a pandas Series indexed like the input.
The value on date t uses data up to and including t; to avoid look-ahead when pricing
an option on date t, take the value at t (close-of-day data, like the settlement price)
via `as_of(series, dates)`.
"""
import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    """R_t = ln(S_t / S_{t-1})."""
    return np.log(close / close.shift(1))


def hist_vol(close: pd.Series, window: int = 30, trading_days: int = 252) -> pd.Series:
    """Rolling standard deviation of daily log-returns times sqrt(252)."""
    return log_returns(close).rolling(window).std() * np.sqrt(trading_days)


def ewma_vol(close: pd.Series, lam: float = 0.94, trading_days: int = 252) -> pd.Series:
    """RiskMetrics EWMA: sigma_t^2 = lam sigma_{t-1}^2 + (1 - lam) R_t^2.

    Seeded on the 30th return with the variance of the first 30 returns (NaN before), so every value uses only
    returns up to and including its own date (no look-ahead)."""
    R = log_returns(close).dropna()
    var = np.full(len(R), np.nan)
    if len(R) < 2:
        return pd.Series(np.nan, index=close.index)
    k = min(30, len(R)) - 1
    var[k] = R.iloc[:k + 1].var()
    for i in range(k + 1, len(R)):
        var[i] = lam * var[i - 1] + (1 - lam) * R.iloc[i] ** 2
    out = pd.Series(np.sqrt(var * trading_days), index=R.index)
    return out.reindex(close.index)


def parkinson_var_daily(high: pd.Series, low: pd.Series) -> pd.Series:
    """Daily Parkinson variance proxy RV_t = ln^2(H_t / L_t) / (4 ln 2)."""
    return np.log(high / low) ** 2 / (4 * np.log(2))


def parkinson_vol(high: pd.Series, low: pd.Series, window: int = 30, trading_days: int = 252) -> pd.Series:
    """Range-based volatility: sigma^2 = (1 / (4 ln 2)) * mean of ln^2(H/L) over the window, annualised."""
    return np.sqrt(parkinson_var_daily(high, low).rolling(window).mean() * trading_days)


def as_of(series: pd.Series, dates) -> np.ndarray:
    """Latest available value on or before each date (no look-ahead)."""
    s = series.dropna().sort_index()
    return s.reindex(pd.DatetimeIndex(dates), method="ffill").to_numpy()
