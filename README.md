# European Option Pricing: Binomial Trees vs Black–Scholes

MAC F244 project (Semester I, 2026–27). We compare the binomial tree (CRR, Jarrow–Rudd, Leisen–Reimer) with
the Black–Scholes model: convergence, Greeks, flexibility (discrete dividends, time-varying volatility), and fit
to real NSE option prices for NIFTY, Reliance and HUL in three volatility regimes. Each workstream is benchmarked
against a machine-learning model.

**Central question.** For European options on Indian market underlyings, how do the binomial and Black–Scholes
models compare in accuracy, speed, flexibility and real-market fit, and does the choice of *model* or the choice
of *inputs* (especially volatility) matter more?

**Short answer.** On 7,718 NSE contracts, Black–Scholes and a 200-step tree differ by 0.02–0.12 bp of the
underlying, while switching between reasonable volatility estimates moves prices by 3.6–18 bp and the models miss
market prices by 5–31 bp. Inputs matter about 100× more than the model.

## Quick start
```bash
python -m venv .venv && .venv/Scripts/activate      # (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
python -m pytest -q                                 # 59 tests, ~6 s
python run_all.py                                   # every figure and table, ~2 min
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
| `src/ml/` | MLP (Adwait), decision tree (Zaid), error regression (Asvin), IV random forest (Shivank), HAR (Pradyumn) | each |
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

Notes: Reliance prices before the 28 Oct 2024 bonus are divided by 2 (`config.SPLITS`). The Reliance ₹5.50 dividend
(ex 14 Aug 2025) falls after the corporate-action download window and is entered in `config.EXTRA_DIVIDENDS`.
NIFTY history starts 2 Jan 2024, so the 252-day volatility exists only from early January 2025.

## How we work
- Branch per task (`name/task`), pull request, one reviewer, then merge into `main`.
- Every public function's docstring names the proposal equation it implements.
- Every figure is produced by code; `python run_all.py` must work from a fresh clone.
