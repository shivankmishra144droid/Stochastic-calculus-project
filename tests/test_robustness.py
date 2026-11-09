"""Robustness helpers: config overrides are undone, and the trading-day count skips weekends and holidays."""
import pandas as pd

import config
from src.market import robustness as rb


def test_overrides_restore_config():
    before = config.MIN_VOLUME
    try:
        with rb.overrides(MIN_VOLUME=1):
            assert config.MIN_VOLUME == 1
            raise RuntimeError
    except RuntimeError:
        pass
    assert config.MIN_VOLUME == before


def test_trading_days_skip_weekends_and_holidays():
    d = pd.Series(pd.to_datetime(["2025-01-03", "2025-01-03", "2025-01-06"]))      # Fri, Fri, Mon
    e = pd.Series(pd.to_datetime(["2025-01-06", "2025-01-10", "2025-01-06"]))      # Mon, next Fri, same day
    assert list(rb.trading_days_to_expiry(d, e, holidays=pd.DatetimeIndex([]))) == [1, 5, 0]
    hol = pd.DatetimeIndex(["2025-01-08"])
    assert list(rb.trading_days_to_expiry(d, e, holidays=hol)) == [1, 4, 0]


def test_known_nse_holiday_is_found():
    assert pd.Timestamp("2025-03-14") in rb.trading_holidays()          # Holi 2025, a Friday
