# Project To-Do: European Option Pricing (MAC F244)

Status as of **6 Nov 2026**. Final submission **15 Nov 2026**.
Detailed method, formulas and code design are in `PROJECT_PLAN.md`. This file is the checklist.

---

## 1. Where we are

### Data: almost complete

| Item | Status | Notes |
|---|---|---|
| F&O bhavcopies, January 2025 (2, 3, 6, 7, 8 Jan) | ✅ Done | In `Videos\stochastic` and in `option_project/data/raw/` |
| F&O bhavcopies, April 2025 (7, 8, 9, 11 Apr) | ⬜ Check | You've downloaded them; move into `Videos\stochastic`, then `data/raw/` |
| F&O bhavcopies, July 2025 (14–18 Jul) | ⬜ Check | Same as April |
| NIFTY 50 history (Jan 2024 → Aug 2025) | ✅ Done | Optional: add Dec 2023 so 252-day vol works for January |
| RELIANCE history (Dec 2023 → Jul 2025) | ✅ Done | **1:1 bonus on 28 Oct 2024 adjusted in `config.SPLITS`** |
| HINDUNILVR history (Dec 2023 → Jul 2025) | ✅ Done | No split/bonus |
| Dividends | ✅ Done | `data/raw/dividends.csv`. HUL ₹24 ex 23 Jun 2025; Reliance ₹5.50 ex 14 Aug 2025 |
| Risk-free rates | ✅ Done | RBI 91-day T-bill: 6.5961% (Jan), 6.3017% (Apr), 5.3872% (Jul), in `config.py` |

### Code
- `option_project.zip`: working NSE pipeline (load → clean → price → implied vol → smiles), 5 tests passing.
  Ran on all five January dates: 2,845 contracts after cleaning, implied vol solved for 95%.
- GitHub repo `Stochastic-calculus-project`: only `baseline_convergence.py`, `README.md`, `requirements.txt`, `.gitignore`.
- `PROJECT_PLAN.md`: full design. Not yet in the repo.

### Findings so far (keep these for the report)
- Black–Scholes and the binomial tree differ by at most 0.41 NIFTY points (₹0.06–0.09 for the stocks),
  while both miss market prices by 15 points on NIFTY (₹1.6–3.1 for the stocks). **Inputs matter more than the model.**
- NIFTY shows a clear volatility smile/skew; puts are priced at higher implied vol than calls.
- The three assets behave differently: NIFTY follows the regimes, Reliance spikes with the market in April,
  HUL is calmer in April and most volatile in July (its own news). Good for "effects of changes in assets".

---

## 2. Admin (today, ~15 minutes)

- [ ] Make the GitHub repo **private** (Settings → General → Danger Zone → Change visibility)
- [ ] Collect everyone's GitHub username in the group chat; add them as collaborators (Settings → Collaborators)
- [ ] Check the contributors list no longer shows Claude (if it still does, the force-push didn't go through)
- [ ] Push `PROJECT_PLAN.md` and this `TODO.md` to the repo
- [x] Confirmed: Pradyumn's ID is `2023B5AD0973G`; spelling is Asvin

---

## 3. Data (today)

- [ ] Put the 9 April/July bhavcopies in `Videos\stochastic` (names start `BhavCopy_NSE_FO_0_0_0_2025…`)
- [ ] Copy all 14 bhavcopies into `data/raw/` of the project
- [ ] Add the April and July dates to `DATES` in `src/config.py`
- [ ] Run `python run_workflow.py` and check: each date loads, implied vol solves for ≥ 90%
- [ ] If any date is missing (holiday), use the next trading day and update `config.py`

---

## 4. Set up the repo structure (Shivank or Pradyumn, today/tomorrow)

- [ ] Create the folders from `PROJECT_PLAN.md` §1: `src/core/`, `src/data/`, `src/convergence/`, `src/market/`,
      `src/model_risk/`, `src/ml/`, `tests/`, `results/figures/`, `results/tables/`, `report/`
- [ ] Copy the NSE pipeline from `option_project.zip` into `src/data/` (load.py, clean.py) and its tests into `tests/`
- [ ] Add `config.py` at the root (baseline contract, regimes, rates, splits, filters)
- [ ] Decide how data is shared: either commit `data/raw/` (files are small, ~1 MB each; remove `data/` and `*.zip`
      from `.gitignore`) or keep it on Google Drive
- [ ] Each person creates their branch: `name/task` (e.g. `zaid/crr-tree`)

---

## 5. Work per person

### Adwait: Black–Scholes engine and theory
- [ ] `src/core/bs.py`: `bs_price`, `bs_greeks`, `black76_price`, `implied_vol`, `implied_vol_forward`
- [ ] `src/core/vol.py`: historical, EWMA, Parkinson volatility (no look-ahead)
- [ ] Tests: baseline 10.4506 / 5.5735, parity, Greeks vs finite differences, implied-vol round trip
- [ ] Figures: price surface, Greeks vs S for 3 maturities, sensitivity to σ / r / T
- [ ] ML: `MLPRegressor` learning C/K from (S/K, T, r, σ); extrapolation test; no-arbitrage check
- [ ] Report: GBM → Itô → BS PDE → formula (half page)

### Zaid: Binomial tree engine
- [ ] `src/core/binomial.py`: CRR, Jarrow–Rudd, Leisen–Reimer; tree Greeks; time-varying σ and discrete-dividend versions
- [ ] Tests: 1- and 2-step trees by hand, convergence to BS, parity in the tree, 0 < p < 1
- [ ] Figures: N = 4 tree diagram, CRR/JR/LR comparison table, tree Greeks vs BS
- [ ] ML: `DecisionTreeRegressor` on tree prices; show flat steps ⇒ Δ, Γ ≈ 0
- [ ] Report: one-period replication, risk-neutral q, backward induction (half page)

### Asvin: Convergence analysis
- [ ] Price vs every N from 1 to 1000 (even/odd colours); log–log error with fitted slope
- [ ] CRR vs JR vs LR; error vs moneyness, σ, T
- [ ] Corrections: N/N+1 averaging, BS smoothing, Richardson; runtime vs accuracy table
- [ ] ML: linear regression of error on 1/N, (−1)^N/N, strike position λ
- [ ] Report: CLT argument, why CRR oscillates, results (half page)

### Shivank: Real NSE data and model vs market
- [ ] All 14 dates through the pipeline; data-cleaning summary table
- [ ] **Implied forward from put–call parity** (fixes the jump in the smile at K/S = 1)
- [ ] Raise `MIN_VOLUME` until smiles are smooth; note the value used
- [ ] Smiles per regime (NIFTY), then Reliance and HUL
- [ ] Pricing-error tables by moneyness × maturity × regime; **headline chart: model choice vs input choice vs market error**
- [ ] ML: quadratic smile fit (baseline: level, skew, curvature) + `RandomForestRegressor` for implied vol, split by time
- [ ] Report: data, smiles, errors (half page)

### Pradyumn: Model risk and integration
- [ ] Dividends: BS escrowed vs BS continuous yield vs tree with ex-date drop
      - HUL April options expiring end-June (₹24 ex 23 Jun 2025)
      - Reliance July options expiring Aug/Sep (₹5.50 ex 14 Aug 2025)
- [ ] Time-varying σ in the tree (Amin's method) vs BS with average variance
- [ ] σ-error vs N-error bar chart
- [ ] ML: HAR regression volatility forecast → BS and tree
- [ ] `run_all.py`; fresh-clone reproducibility check; merge the report
- [ ] Report: introduction, model-risk section, conclusion

---

## 6. Timeline

| Date | Must be done |
|---|---|
| **Fri 6 Nov** | Admin (§2), data complete (§3), repo structure (§4) |
| **Sat 7 Nov** | `src/core/` (bs.py, binomial.py, vol.py) merged with tests: everyone depends on this |
| Sun 8 – Tue 10 Nov | All experiments running; first versions of every figure |
| Tue 10 – Wed 11 Nov | ML models trained and evaluated |
| Thu 12 Nov | Final figures; each person pushes their half-page to `report/` |
| Fri 13 Nov | Report merged and trimmed to 3–4 pages; references checked |
| Sat 14 Nov | Fresh-clone check (`pip install -r requirements.txt` → `python run_all.py`); viva rehearsal |
| **Sun 15 Nov** | Submit report + code |

---

## 7. Report checklist

- [ ] Introduction states the central question and says we extended beyond the proposal
      (real NSE option prices, implied vol, ML benchmarks)
- [ ] Every equation in the report maps to a function in the code
- [ ] 5–6 strongest figures only; the rest go in the repo
- [ ] Data sources with download dates: NSE bhavcopies, NSE equity history, niftyindices.com, NSE corporate actions, RBI T-bill auctions
- [ ] References: Black–Scholes (1973), Cox–Ross–Rubinstein (1979), Jarrow–Rudd (1983), Leisen–Reimer (1996), Hull,
      Hutchinson–Lo–Poggio (1994), Breiman (2001), Corsi (2009), scikit-learn (2011)
- [ ] Names and IDs correct

---

## 8. Viva checklist (everyone)

- [ ] Explain risk-neutral q and why it isn't the real probability
- [ ] Why d < e^{rΔt} < u
- [ ] Derive the one-period replicating portfolio
- [ ] Meaning of Φ(d₁) and Φ(d₂) in the BS formula
- [ ] Why the tree converges to BS; why CRR oscillates between even and odd N
- [ ] What implied volatility is; why there is a smile/skew
- [ ] Why historical σ misprices market options
- [ ] Why NSE options are European and why that matters
- [ ] Your own ML model: what it learns, where it fails, how it compares with the maths
- [ ] Your own data choices: filters, rate used, which dates and why (regimes from NIFTY realised vol)
