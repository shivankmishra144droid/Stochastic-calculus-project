# European Option Pricing: Binomial Trees vs Black–Scholes

MAC F244 project (Semester I, 2026–27). We compare the binomial tree (CRR, Jarrow–Rudd, Leisen–Reimer) with
the Black–Scholes model: convergence, Greeks, flexibility (discrete dividends, time-varying volatility), and fit
to real NSE option prices for NIFTY, Reliance and HUL in three volatility regimes. A machine-learning model then
asks whether it can choose a better volatility input than any single estimate.

**Central question.** For European options on Indian market underlyings, how do the binomial and Black–Scholes
models compare in accuracy, speed, flexibility and real-market fit, and does the choice of *model* or the choice
of *inputs* (especially volatility) matter more?

**Short answer.** On 6,052 cleaned NSE contracts, Black–Scholes and a 200-step tree differ by 0.02–0.12 bp of the
underlying, while switching between reasonable volatility estimates moves prices by 4–21 bp and the models miss
market prices by 5–35 bp. The tree is a discretisation of the same GBM model as Black–Scholes, so their gap is
numerical error: more than 100× smaller than the effect of the volatility input. Even the best constant σ (same-day ATM
implied vol) misses by 2–15 bp, the smile that both share; that is the real model error.

**Which model to choose** (report, Table "Decision rule"):

| Situation | Choose | Why |
|---|---|---|
| European, no cash dividend before expiry | Black–Scholes (Black-76 on the parity forward) | Exact, ~0.2 ms; a tree only adds 0.05 bp numerical error |
| Time-varying volatility | Black–Scholes with the average variance | Only total variance matters for Europeans |
| Known cash dividend, priced from spot | Tree with a cash drop at the ex-date | Escrowed BS underprices by up to 1.3% ATM |
| Dividend already in a market forward | Black–Scholes on that forward | On Reliance all treatments miss by the same Rs 0.7–1.1 |
| Early exercise / path dependence | Tree | No closed form (not tested: NSE options are European) |
| If you use a tree | Leisen–Reimer, odd N | 10⁻⁴ in 28–66 steps at every strike; CRR needs >25,000 |
| σ, option prices available | Same-day implied vol | 6.3 bp |
| σ, no option prices, regime seen before | ML blend (random forest) | 10.1 bp vs 12.7 (EWMA), 14.9 (30-day) |
| σ, no option prices, new regime | EWMA or 30-day + VIX regression | 12.2–12.7 bp; six-input ML overfits |

**Robustness checks** (report Section 7, `src/market/robustness.py`):
- Looser filters (volume ≥ 1, no OI limit: 6,271–6,547 contracts) and stricter ones (volume ≥ 100, OI ≥ 500: 5,343)
  leave the decomposition unchanged: numerical 0.05 bp, input choice 10.4–11.2 bp, smile 6.1–6.3 bp (input/numerical 215–222×).
- A trading-day clock (variance σ²·n_td/252) does not reduce errors at any maturity (+0.2 to +1.8 bp, CIs include 0
  or are slightly worse), so the calendar clock is kept.

**Limitations**:
- Only 14 trading days on 3 underlyings (42 rows for the ML), so the confidence intervals are rough.
- Settlements are closing VWAPs, not synchronous with the underlying.
- The T-bill rate is not the true carry for single stocks, and VIX measures NIFTY, not the stocks.
- Options are European only, and both models use one constant σ.

## Quick start
```bash
python -m venv .venv && .venv/Scripts/activate      # (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
python -m pytest -q                                 # 78 tests, ~10 s
python run_all.py                                   # every figure and table, ~7 min (4 of them: the runtime table)
python run_all.py --only convergence market         # just some parts
```
Outputs go to `results/figures/*.pdf`, `results/tables/*.csv` and `data/processed/` (all regenerated, not committed).

## Layout
| Path | What it holds | Owner |
|---|---|---|
| `config.py` | every constant: baseline contract, regimes, risk-free rates, filters, paths | all |
| `src/core/bs.py` | Black–Scholes price, Greeks, Black-76, implied vol (Newton + brentq) | Adwait |
| `src/core/binomial.py` | CRR / JR / LR trees, tree Greeks, time-varying σ (Amin), discrete dividends | Zaid |
| `src/core/vol.py` | historical, EWMA, Parkinson volatility; `as_of` (no look-ahead) | Adwait |
| `src/data/` | NSE bhavcopy loader (old + UDiFF), spot/dividend loaders, implied forward, cleaning | Shivank |
| `src/bs_analysis/` | price surface, Greeks, sensitivity, parity, limiting cases | Adwait |
| `src/tree_analysis/` | replication, N = 4 tree, CRR/JR/LR table, tree Greeks, arbitrage example | Zaid |
| `src/convergence/` | price vs N, log–log rates, error heatmaps, corrections, runtime | Asvin |
| `src/market/` | model vs market, smiles, forward fix, error tables, headline figure | Shivank |
| `src/model_risk/` | dividends, time-varying σ, σ-error vs N-error | Pradyumn |
| `src/stats.py` | bootstrap CIs over trading dates (cluster bootstrap) | all |
| `src/market/robustness.py` | Robustness: error decomposition under looser/stricter liquidity filters, and a trading-day clock | all |
| `src/ml/sigma_blend.py` | ML chooses σ: linear regression / random forest blend of historical vols + India VIX, tested on unseen days and regimes | all |
| `tests/` | pytest suite (engines, data pipeline on synthetic files, no-look-ahead checks) | all |
| `report/` | LaTeX report and viva notes | all |
| `docs/proposal.pdf` | submitted proposal | |

`PROJECT_PLAN.md` holds the full design (interfaces, maths, experiments); `TODO.md` is the checklist.

## Data (all in `data/raw/`, as downloaded, never edited)
| Data | Source | Downloaded |
|---|---|---|
| F&O bhavcopies, 14 trading days (Jan, Apr, Jul 2025) | nseindia.com → All Reports → Derivatives → Archives (UDiFF) | Jan: 31 Oct 2026; Apr, Jul: 6 Nov 2026 |
| NIFTY 50 OHLC, 2 Jan 2024 – 1 Aug 2025 | niftyindices.com → Historical Data | 31 Oct 2026 |
| RELIANCE, HINDUNILVR OHLC, 1 Dec 2023 – 31 Jul 2025 | nseindia.com → Security-wise price volume archives | 6 Nov 2026 |
| Corporate actions (dividends, Reliance 1:1 bonus) | nseindia.com → Corporate Actions | 6 Nov 2026 |
| 91-day T-bill cut-off yields (1 Jan, 2 Apr, 9 Jul 2025) | RBI press releases, `data/raw/rbi/` | 6 Nov 2026 |
| India VIX, 2 Dec 2024 – 31 Jul 2025 | nseindia.com → Historical Data → India VIX | 6 Nov 2026 |

Data audit (all fixes are in the loaders and `config.py`; raw files are untouched):
- **Theoretical settlements.** For contracts that barely trade, NSE's `SttlmPric` is its own model price, not a trade
  (settlement ≠ close). Only trade-set prices are kept (`config.TRADED_SETTLE_ONLY`); on expiry day `SttlmPric`
  is the underlying price, which the ≥ 3-day filter removes.
- **Static arbitrage.** Closing-half-hour VWAPs of neighbouring strikes can break monotonicity/convexity (worst on
  7 Apr 2025); `clean.strike_arbitrage_mask` drops the lowest-volume offender until no violation exceeds
  max(2 ticks, 1 %). Prices below the discounted intrinsic value on the forward, and premiums under ₹1, are dropped.
- **Units.** Reliance's ₹10 dividend of 19 Aug 2024 predates the 1:1 bonus, so it is ₹5 per current share
  (`load_dividends`, column `amount`; `amount_declared` keeps the announced figure).
- **Volatility inputs.** Close-to-close estimators use dividend-adjusted closes (`total_return_close`); the two
  one-hour Saturday DR-test sessions (2 Mar, 18 May 2024) are dropped (`config.SHORT_SESSIONS`).
- **India VIX.** `Open` is just the previous close and closes are rounded to 2 decimals from Feb 2025; only `Close`
  is used (`load_vix`).

Reliance prices before the 28 Oct 2024 bonus are divided by 2 (`config.SPLITS`). The Reliance ₹5.50 dividend
(ex 14 Aug 2025) falls after the corporate-action download window and is entered in `config.EXTRA_DIVIDENDS`.
NIFTY history starts 2 Jan 2024, so the 252-day volatility exists only from early January 2025.

## How we work
- Branch per task (`name/task`), pull request, one reviewer, then merge into `main`.
- Every public function's docstring names the proposal equation it implements.
- Every figure is produced by code; `python run_all.py` must work from a fresh clone.
