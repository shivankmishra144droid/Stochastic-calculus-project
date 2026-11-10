# Project To-Do: European Option Pricing (MAC F244)

Status as of **10 Nov 2026**. Final submission **15 Nov 2026**.
Detailed method, formulas and code design are in `PROJECT_PLAN.md`; results and how to run are in `README.md`.

---

## 1. Done

- **Data.** 14 bhavcopies (Jan, Apr, Jul 2025) in `data/raw/`; NIFTY, Reliance, HUL history; corporate actions
  (Reliance 1:1 bonus in `config.SPLITS`, Rs 5.50 dividend in `config.EXTRA_DIVIDENDS`); RBI T-bill rates; India VIX.
- **Repo structure.** `config.py` at the root, `src/{core,data,bs_analysis,tree_analysis,convergence,market,model_risk,ml}`,
  `tests/`, `report/`; `data/raw/` committed; `pytest.ini` limits pytest to `tests/`.
- **Engines.** `src/core/bs.py` (price, Greeks, Black-76, implied vol), `src/core/binomial.py` (CRR/JR/LR, tree Greeks,
  Amin time-varying sigma, discrete and cash dividends), `src/core/vol.py` (historical, EWMA, Parkinson; no look-ahead).
- **Experiments.** BS sweeps, tree comparison, convergence (price vs N, log-log rates, corrections, runtime),
  model vs market (implied forward, smiles, error tables, headline decomposition), model risk (dividends,
  time-varying sigma, sigma-error vs N-error), robustness (liquidity filters, trading-day clock).
- **ML.** Replaced the per-person ML plans with one study: `src/ml/sigma_blend.py` (LR / random forest blend of
  historical vols + India VIX, tested on unseen days and unseen regimes).
- **Pipeline.** `python run_all.py` regenerates every figure and table (~7 min); test suite passes.
- **Report.** `report/main.tex` / `main.pdf` with decision rule, limitations, robustness checks and references;
  `report/viva_notes.md`. Pradyumn's ID (`2023B5AD0973G`) and the spelling Asvin confirmed.

---

## 2. Still to do

### Admin
- [ ] Make the GitHub repo **private** (Settings → General → Danger Zone → Change visibility)
- [ ] Add everyone as collaborators (Settings → Collaborators)
- [ ] Check the contributors list no longer shows Claude
- [ ] Merge the working branch into `main` and push (remote `main` is still at the initial commit)

### Before submission
- [ ] Fresh-clone check: `pip install -r requirements.txt` → `python -m pytest -q` → `python run_all.py`
- [ ] Each person rereads their section of the report against the current `results/tables/`
- [ ] Names and IDs on the title page
- [ ] Viva rehearsal (checklist below)
- [ ] **Sun 15 Nov:** submit report + code

---

## 3. Viva checklist (everyone)

- [ ] Explain risk-neutral q and why it isn't the real probability
- [ ] Why d < e^{rΔt} < u
- [ ] Derive the one-period replicating portfolio
- [ ] Meaning of Φ(d₁) and Φ(d₂) in the BS formula
- [ ] Why the tree converges to BS; why CRR oscillates between even and odd N
- [ ] What implied volatility is; why there is a smile/skew
- [ ] Why historical σ misprices market options
- [ ] Why NSE options are European and why that matters
- [ ] The σ-blend ML study: what it learns, why the "unseen day" test is a best case, why it fails in a new regime
- [ ] Your own data choices: filters, rate used, which dates and why (regimes from NIFTY realised vol)
