"""Project-wide constants: the single place to change settings (PROJECT_PLAN.md §2).

Everything else imports from here; no module hard-codes a date, rate, filter or path.
"""
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Baseline contract (proposal Section 1)
# ---------------------------------------------------------------------------
S0, K0, R0, SIGMA0, T0, Q0 = 100.0, 100.0, 0.05, 0.20, 1.0, 0.0
BS_CALL_REF, BS_PUT_REF = 10.4506, 5.5735  # used in tests

SEED = 42

# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
UNDERLYINGS = {"NIFTY": "index", "RELIANCE": "stock", "HINDUNILVR": "stock"}

# Regimes chosen from NIFTY 30-day realised volatility.
# 10 Apr 2025 was an NSE holiday (Mahavir Jayanti), hence four April dates.
REGIMES = {
    "calm": ["2025-07-14", "2025-07-15", "2025-07-16", "2025-07-17", "2025-07-18"],   # ~8.7%
    "normal": ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"],  # ~13%
    "volatile": ["2025-04-07", "2025-04-08", "2025-04-09", "2025-04-11"],               # ~19-21%
}
DATE_TO_REGIME = {d: reg for reg, ds in REGIMES.items() for d in ds}

# RBI 91-day T-bill cut-off yield (YTM, % p.a.) from the auction just before each regime.
# Sources: RBI press releases 2024-2025/1836 (1 Jan 2025), 2025-2026/15 (2 Apr 2025),
# 2025-2026/681 (9 Jul 2025); PDFs kept in data/raw/rbi/.
TBILL_YTM = {"normal": 6.5961, "volatile": 6.3017, "calm": 5.3872}
# Continuously compounded rate used by every model: r = ln(1 + y).
RATE_BY_REGIME = {reg: math.log(1 + y / 100) for reg, y in TBILL_YTM.items()}
RISK_FREE = {d: RATE_BY_REGIME[reg] for d, reg in DATE_TO_REGIME.items()}

# Spot / OHLC history files in data/raw (glob patterns; raw names are kept unchanged).
SPOT_FILES = {
    "NIFTY": "NIFTY 50_Historical_PR_*.csv",
    "RELIANCE": "Quote-Equity-RELIANCE-EQ-*.csv",
    "HINDUNILVR": "Quote-Equity-HINDUNILVR-EQ-*.csv",
}
CORP_ACTION_FILES = {
    "RELIANCE": "CF-CA-equities-RELIANCE-*.csv",
    "HINDUNILVR": "CF-CA-equities-HINDUNILVR-*.csv",
}
# Dividends announced after the corporate-action download (31 Jul 2025) window.
# Reliance FY25 final dividend Rs 5.50, ex-date 14 Aug 2025 (NSE corporate announcement).
EXTRA_DIVIDENDS = [("RELIANCE", "2025-08-14", 5.50)]

# Splits / bonus issues: (first ex-date, ratio). Prices before the ex-date are divided by ratio.
SPLITS = {"RELIANCE": [("2024-10-28", 2.0)]}  # 1:1 bonus

# Short NSE disaster-recovery test sessions on Saturdays (about an hour of trading). Their high-low range is
# ~7% of a normal day's, so as full "days" they bias range-based volatility down; dropped by load_spot, so the
# move rolls into the next session's close-to-close return. (20 Jan 2024 and 1 Feb 2025 were full Saturday
# sessions and are kept.)
SHORT_SESSIONS = ["2024-03-02", "2024-05-18"]

# India VIX (NSE, Historical Data -> India VIX); used for reference, NIFTY only.
VIX_FILE = "hist_india_vix_*.csv"

# ---------------------------------------------------------------------------
# Cleaning filters
# ---------------------------------------------------------------------------
MIN_VOLUME, MIN_OI = 10, 50      # contracts
# Keep only prices set by trades: NSE reports a THEORETICAL settlement price (its own model) when a contract
# did not trade enough near the close; then SttlmPric != ClsPric. Comparing models with NSE's model is circular.
TRADED_SETTLE_ONLY = True
MIN_PRICE = 1.0                   # Rs; below this one tick (0.05) is more than 5% of the premium
# Static-arbitrage cleaning across strikes (monotone, convex, |dV/dK| <= e^{-rT}); violations smaller than
# max(ARB_TOL_ABS, ARB_TOL_REL * price) are treated as tick / closing-VWAP noise and kept.
ARB_TOL_ABS, ARB_TOL_REL = 0.10, 0.01
MONEYNESS = (0.80, 1.20)          # K / F
MIN_DAYS, MAX_EXPIRIES = 3, 3
CRR_STEPS_MARKET = 200            # tree size for market pricing

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"

# ---------------------------------------------------------------------------
# Figure style (PROJECT_PLAN.md §7)
# ---------------------------------------------------------------------------
GREEN, SLATE, GREY = "#00590D", "#334155", "#94A3B8"
FIG_SINGLE, FIG_FULL = 3.3, 6.8   # inches
