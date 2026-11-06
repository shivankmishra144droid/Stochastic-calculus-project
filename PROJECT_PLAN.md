# European Option Pricing: Binomial Trees vs Black–Scholes
## Complete coding plan (MAC F244, final submission 15 Nov 2026)

This file is the single reference for building the project. It covers the repository layout, the shared
interfaces everyone codes against, the maths each function implements, the data pipeline (including the NSE
quirks we already found), every person's experiments and ML model, the tests, the figures, and the
timeline.

**Central question** (state it in the report introduction and answer it in the conclusion):

> For European options on Indian market underlyings, how do the binomial and Black–Scholes models compare in
> accuracy, speed, flexibility and real-market fit, and does the choice of *model* or the choice of *inputs*
> (especially volatility) matter more?

First real evidence (NIFTY, 2 Jan 2025): BS and a 200-step CRR tree agree to within 0.39 index points on every
contract, while both miss market prices by 10–18 points because one historical σ (13.5%) cannot match a smile
running from about 12.5% to 25%. **Inputs matter more than the model.** The rest of the project tests this
properly.

---

## 0. Ground rules

| Rule | Why |
|---|---|
| Python 3.10+, `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`, `pytest` | One stack, nothing exotic |
| All pricing code in `src/` as importable modules; notebooks only for exploration | Others must be able to `import` your code |
| Every public function has a docstring naming the proposal equation it implements | "Clear connection between theory and implementation" is graded |
| Vectorise with NumPy; no Python loops over nodes inside a time step | A 1000-step tree must run in milliseconds |
| Fixed random seed (`SEED = 42`) wherever randomness appears (ML splits, model init) | Reproducible figures |
| Every figure is produced by a script, never by hand | `python run_all.py` regenerates the whole report |
| Branch per task (`name/task`), pull request, one reviewer, merge into `main` often | Avoids a giant merge the night before |
| Each person edits mainly their own folder | Prevents merge conflicts |

---

## 1. Repository layout

```
Stochastic-calculus-project/
├── README.md
├── PROJECT_PLAN.md               ← this file
├── requirements.txt
├── config.py                     ← all constants: baseline contract, dates, rates, filters, paths
├── run_all.py                    ← regenerates every figure and table in the report
│
├── src/
│   ├── core/                     ← SHARED. Persons 1 + 2. Everyone imports from here.
│   │   ├── bs.py                 ← Black–Scholes price, Greeks, implied vol      (Person 1)
│   │   ├── binomial.py           ← CRR / JR / LR trees, tree Greeks               (Person 2)
│   │   └── vol.py                ← historical / EWMA / Parkinson volatility       (Person 1)
│   │
│   ├── data/                     ← Person 4 (Person 5 helps)
│   │   ├── load.py               ← NSE bhavcopy (old + UDiFF) → common table; spot loader
│   │   ├── clean.py              ← attach S, T, r, σ; liquidity and moneyness filters
│   │   └── forward.py            ← implied forward from put–call parity
│   │
│   ├── bs_analysis/              ← Person 1: sweeps, surfaces, Greek curves
│   ├── tree_analysis/            ← Person 2: tree vs BS Greeks, parametrisation comparison
│   ├── convergence/              ← Person 3: N-sweeps, rates, corrections, runtime
│   ├── market/                   ← Person 4: model vs market, smiles, regimes
│   ├── model_risk/               ← Person 5: dividends, time-varying σ, σ-error vs N-error
│   └── ml/                       ← one file per person
│       ├── mlp_bs.py             (Person 1)
│       ├── tree_regressor.py     (Person 2)
│       ├── error_regression.py   (Person 3)
│       ├── iv_forest.py          (Person 4)
│       └── har.py                (Person 5)
│
├── tests/                        ← pytest; each person adds tests for their module
├── data/
│   ├── raw/                      ← bhavcopy zips, spot CSVs, dividend CSVs (as downloaded)
│   └── processed/                ← cleaned tables (generated)
├── results/
│   ├── figures/                  ← <person>_<what>.pdf  e.g. conv_loglog_error.pdf
│   └── tables/                   ← CSV tables that go into the report
└── report/                       ← LaTeX report, one .tex section per person
```

Existing code to reuse:
- `baseline_convergence.py` (already in the repo) contains a working `bs_price` and `crr_price`. Persons 1 and 2
  start by moving these into `src/core/` and extending them.
- The NSE workflow from 31 Oct (`option_project.zip`: `load.py`, `clean.py`, `pricing.py`, tests) has been run
  on real data and works. Person 4 copies it into `src/data/` instead of starting from scratch.

---

## 2. `config.py` (one place for every constant)

```python
# Baseline contract (proposal Section 1)
S0, K0, R0, SIGMA0, T0, Q0 = 100.0, 100.0, 0.05, 0.20, 1.0, 0.0
BS_CALL_REF, BS_PUT_REF = 10.4506, 5.5735          # used in tests

SEED = 42

# Market data
UNDERLYINGS = {"NIFTY": "index", "RELIANCE": "stock", "HINDUNILVR": "stock"}
REGIMES = {                                        # chosen from NIFTY 30-day realised vol
    "calm":     ["2025-07-14", "2025-07-15", "2025-07-16", "2025-07-17", "2025-07-18"],   # ~8.7%
    "normal":   ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"],   # ~13%
    "volatile": ["2025-04-07", "2025-04-08", "2025-04-09", "2025-04-11"],                 # ~19-21%
}
RISK_FREE = {   # RBI 91-day T-bill yield, continuously compounded: r = ln(1 + y)
    # "2025-01-02": ...,  "2025-04-07": ...,  "2025-07-14": ...      ← fill from dbie.rbi.org.in
}

# Cleaning filters
MIN_VOLUME, MIN_OI = 10, 50          # contracts; raise if smiles are jagged
MONEYNESS = (0.80, 1.20)             # K / F
MIN_DAYS, MAX_EXPIRIES = 3, 3
```

**Rate convention:** RBI quotes T-bill yields as simple annual yields. Convert once: `r = np.log(1 + y)`.
Use the same r for BS and the tree.

---

## 3. Shared interfaces (agree first, do not change without telling the group)

All functions accept NumPy arrays where it makes sense (vectorised over contracts), and use
`option_type in {"C", "P"}`, time in **years**, rates and vols as **decimals**, continuous dividend yield `q`.

### 3.1 `src/core/bs.py` (Person 1)

```python
def bs_price(S, K, T, r, sigma, option_type="C", q=0.0) -> np.ndarray
def bs_greeks(S, K, T, r, sigma, option_type="C", q=0.0) -> dict   # delta, gamma, vega, theta, rho
def black76_price(F, K, T, r, sigma, option_type="C") -> np.ndarray # price from forward F
def implied_vol(price, S, K, T, r, option_type="C", q=0.0) -> float # NaN if no solution
def implied_vol_forward(price, F, K, T, r, option_type="C") -> float
```

Maths:

- $d_1 = \dfrac{\ln(S/K) + (r - q + \tfrac12\sigma^2)T}{\sigma\sqrt T}$, $d_2 = d_1 - \sigma\sqrt T$
- Call $= S e^{-qT}\Phi(d_1) - K e^{-rT}\Phi(d_2)$; Put $= K e^{-rT}\Phi(-d_2) - S e^{-qT}\Phi(-d_1)$
- Greeks (call; put via parity): $\Delta = e^{-qT}\Phi(d_1)$, $\Gamma = \dfrac{e^{-qT}\varphi(d_1)}{S\sigma\sqrt T}$,
  $\text{Vega} = S e^{-qT}\varphi(d_1)\sqrt T$,
  $\Theta = -\dfrac{S e^{-qT}\varphi(d_1)\sigma}{2\sqrt T} + qSe^{-qT}\Phi(d_1) - rKe^{-rT}\Phi(d_2)$,
  $\rho = KTe^{-rT}\Phi(d_2)$. Report Θ per year in code; divide by 365 for "per day" in plots.
- Black-76: $C = e^{-rT}[F\Phi(d_1) - K\Phi(d_2)]$, $d_1 = \dfrac{\ln(F/K) + \tfrac12\sigma^2T}{\sigma\sqrt T}$.

Implied volatility, step by step:
1. Check no-arbitrage bounds. Call: $\max(Se^{-qT} - Ke^{-rT}, 0) < C < Se^{-qT}$; put: $\max(Ke^{-rT} - Se^{-qT}, 0) < P < Ke^{-rT}$. Outside them → return `NaN`.
2. Newton: start at $\sigma_0 = \sqrt{2|\ln(S/K) + rT|/T}$ (Brenner–Subrahmanyam-style start), clipped to [0.05, 1].
   Update $\sigma \leftarrow \sigma - (\text{BS}(\sigma) - \text{price})/\text{Vega}(\sigma)$, stop when |diff| < 1e-8.
3. If Vega < 1e-8 or σ leaves [1e-4, 5], fall back to bisection on [1e-4, 5] (`scipy.optimize.brentq` is fine).
4. Record how many contracts fail, and report that number in the report.

### 3.2 `src/core/binomial.py` (Person 2)

```python
def binomial_price(S, K, T, r, sigma, N, option_type="C", q=0.0, method="crr") -> float
def binomial_greeks(S, K, T, r, sigma, N, option_type="C", q=0.0, method="crr") -> dict
def binomial_tree_values(S, K, T, r, sigma, N, ...) -> list[np.ndarray]  # small N only, for drawing
def binomial_price_timevarying(S, K, T, r_path, sigma_path, ...) -> float # used by Person 5
def binomial_price_discrete_div(S, K, T, r, sigma, N, divs, ...) -> float # used by Person 5
```

Parametrisations (Δt = T/N, b = r − q):

| Method | u, d | probability p |
|---|---|---|
| CRR | $u = e^{\sigma\sqrt{\Delta t}}$, $d = 1/u$ | $p = \dfrac{e^{b\Delta t} - d}{u - d}$ |
| Jarrow–Rudd | $u = e^{(b - \sigma^2/2)\Delta t + \sigma\sqrt{\Delta t}}$, $d = e^{(b - \sigma^2/2)\Delta t - \sigma\sqrt{\Delta t}}$ | $p = \dfrac{e^{b\Delta t} - d}{u - d}$ (≈ ½; use this exact form so the tree is arbitrage-free) |
| Leisen–Reimer | see below | see below |

Leisen–Reimer (use **odd N**; if N is even, set N ← N + 1 and say so in the docstring):
- Peizer–Pratt inversion: $h(z) = \tfrac12 + \tfrac12\,\text{sign}(z)\sqrt{1 - \exp\!\Big[-\big(\tfrac{z}{N + 1/3 + 0.1/(N+1)}\big)^2\big(N + \tfrac16\big)\Big]}$
- $p = h(d_2)$, $p' = h(d_1)$ (using the BS $d_1, d_2$ of the option)
- $u = e^{b\Delta t}\,p'/p$, $d = (e^{b\Delta t} - p\,u)/(1-p)$

Backward induction (all methods):
```python
j  = np.arange(N, -1, -1)
ST = S * u**j * d**(N - j)
V  = payoff(ST)
disc = np.exp(-r * dt)
for _ in range(N):
    V = disc * (p * V[:-1] + (1 - p) * V[1:])
```

Tree Greeks (keep the node arrays at steps 1 and 2 during induction):
- $\Delta = \dfrac{V_{1,u} - V_{1,d}}{S u - S d}$
- $\Gamma = \dfrac{\dfrac{V_{2,uu} - V_{2,ud}}{Su^2 - Sud} - \dfrac{V_{2,ud} - V_{2,dd}}{Sud - Sd^2}}{\tfrac12(Su^2 - Sd^2)}$
- $\Theta = \dfrac{V_{2,ud} - V_0}{2\Delta t}$ (exact for CRR where ud = 1)
- Vega and ρ by central finite differences: bump σ by ±0.001 or r by ±0.0001 and reprice with the same N.

Time-varying σ (for Person 5): to keep the tree recombining, choose step lengths so that $\sigma_i^2\Delta t_i = \text{const}$
(Amin's method). Then $u = e^{\sqrt{c}}$ is constant, and only $p_i = (e^{b_i\Delta t_i} - d)/(u-d)$ and
the discount $e^{-r_i\Delta t_i}$ change per step. Requirement: $\sum_i \Delta t_i = T$, which fixes c from
$\int_0^T \sigma(t)^2dt = Nc$.

Discrete dividends (for Person 5): proportional drop at the ex-date step. If dividend yield $\delta_i = D_i/S_{\text{ex}}$
is applied at step $k$, multiply all node prices at steps ≥ k by $(1 - \delta_i)$. This keeps recombination.
Estimate $\delta_i$ as $D_i / S$ at the ex-date (from the spot file).

### 3.3 `src/core/vol.py` (Person 1)

```python
def log_returns(close: pd.Series) -> pd.Series
def hist_vol(close, window=30, trading_days=252) -> pd.Series         # rolling std × √252
def ewma_vol(close, lam=0.94, trading_days=252) -> pd.Series          # RiskMetrics
def parkinson_vol(high, low, window=30, trading_days=252) -> pd.Series # range-based, needs High/Low
```
- EWMA: $\sigma_t^2 = \lambda\sigma_{t-1}^2 + (1-\lambda)R_{t-1}^2$
- Parkinson: $\sigma^2 = \dfrac{1}{4\ln 2}\cdot\dfrac{1}{n}\sum \ln^2(H_t/L_t)$, annualise by 252.
- **No look-ahead:** the volatility used on option date t must come from data up to t − 1 or t. Use
  `series.reindex(dates, method="ffill")`.

---

## 4. Data pipeline (Person 4; Person 5 helps with downloading)

### 4.1 What to download (all manual, browser)

| Data | Where | Range / dates |
|---|---|---|
| F&O bhavcopy (UDiFF Common Bhavcopy Final, zip) | nseindia.com → All Reports → Derivatives → **Archives** tab | Every date in `REGIMES` |
| NIFTY 50 history | niftyindices.com → Historical Data → Equity → Broad Market Indices → NIFTY 50 | 01-Dec-2023 → 31-Jul-2025 (already have Jan 2024 → Aug 2025) |
| RELIANCE, HINDUNILVR history | nseindia.com → Historical Reports → Capital Market → Security-wise Price Volume Archives (series EQ) | 01-Dec-2023 → 31-Jul-2025 |
| Dividends | nseindia.com → Get Quote → Corporate Actions | 2024–2025, ex-date + amount |
| 91-day T-bill yield | dbie.rbi.org.in | Auctions nearest each regime |

Save everything **unchanged** into `data/raw/`. Never edit raw files by hand.

### 4.2 Loading (`src/data/load.py`)

Two bhavcopy formats. The loader detects which one by the presence of `FinInstrmTp`.

| Field | UDiFF (from 8 Jul 2024) | Old format |
|---|---|---|
| option filter | `FinInstrmTp` ∈ {IDO, STO} | `INSTRUMENT` ∈ {OPTIDX, OPTSTK} |
| symbol | `TckrSymb` | `SYMBOL` |
| expiry | `XpryDt` (YYYY-MM-DD) | `EXPIRY_DT` (DD-Mon-YYYY) |
| strike | `StrkPric` | `STRIKE_PR` |
| type | `OptnTp` CE/PE | `OPTION_TYP` CE/PE |
| settlement | `SttlmPric` | `SETTLE_PR` |
| volume | `TtlTradgVol` | `CONTRACTS` |
| open interest | `OpnIntrst` | `OPEN_INT` |
| underlying | `UndrlygPric` | — (take from spot file) |

Output columns: `date, symbol, instrument, expiry, strike, option_type (C/P), close, settle, volume, open_interest, underlying`.

Spot loader: NSE CSVs have trailing spaces in column names, commas in numbers, and dates like `02-JAN-2025`.
Strip and lower-case column names, remove commas, parse with `dayfirst=True`. Several files per symbol
(`spot_NIFTY_part1.csv`, `..._part2.csv`) are concatenated and de-duplicated.

### 4.3 Cleaning (`src/data/clean.py`)

Order matters; print how many rows each step removes (that table goes into the report):
1. Keep only symbols in `UNDERLYINGS`.
2. Attach S (bhavcopy `underlying`, else spot close on that date), `days = expiry − date`, `T = days/365`, r.
3. Attach volatility estimates (30/60/252-day historical, EWMA, Parkinson), forward-filled, no look-ahead.
4. Filters: volume ≥ MIN_VOLUME, OI ≥ MIN_OI, settle > 0, days ≥ MIN_DAYS, nearest MAX_EXPIRIES expiries.
5. Compute the implied forward (4.4), then moneyness = K/F, and filter on MONEYNESS.
6. Keep **OTM options only** for smiles (puts with K < F, calls with K ≥ F); keep both for pricing-error tables.

### 4.4 Implied forward (`src/data/forward.py`): fixes the jump at K/S = 1

Problem seen on 2 Jan 2025: the put and call smiles do not meet at K/S = 1. Using spot S ignores the
carry (rates minus dividends) and NSE settlement timing. Fix:

For each (symbol, date, expiry), using strikes where both a call and a put exist near the money:
$$F_K = K + e^{rT}(C_K - P_K)$$
Take the **median** of $F_K$ over the 5 strikes closest to spot. Then price with Black-76 on F, and compute
implied vols with `implied_vol_forward`. The smile should become continuous; show the before/after plot,
which is a good viva point.

### 4.5 Known pitfalls

- Use the **settlement price**, not the last traded price. Far OTM strikes with tiny volume have stale settlements;
  that's why the smile was jagged. Raise `MIN_VOLUME` if needed.
- Deep ITM options often violate the no-arbitrage bounds after rounding to ₹0.05 → IV is NaN. Expected.
- NSE weekly NIFTY expiries changed weekday during 2025; don't hard-code expiry weekdays, read `expiry` from the file.
- Stock options on NSE are physically settled, but still European: fine for the models.
- Market holidays: if a bhavcopy is missing, use the next trading day and update `REGIMES`.

---

## 5. Person-by-person build plan

Each section lists: what to code, the experiments, the figures/tables, the ML model, and the tests.
Figure names are fixed so `run_all.py` and the report can reference them.

### Person 1: Adwait (Black–Scholes theory, engine, sensitivity)

**Code:** `src/core/bs.py`, `src/core/vol.py`, `src/bs_analysis/sweeps.py`.

**Experiments**
1. Price surface: call price over S ∈ [60, 140] × T ∈ [0.01, 2] at baseline r, σ. `bs_price_surface.pdf` (3D or heatmap).
2. Greek curves vs S for T ∈ {1 week, 1 month, 1 year}: Δ, Γ, Vega, Θ. Explain Γ spiking at the strike as T → 0. `bs_greeks_vs_S.pdf`.
3. Sensitivity: price vs σ ∈ [0.05, 0.6], vs r ∈ [0, 0.12], vs T. Show call price is increasing and convex in σ near ATM. `bs_sensitivity.pdf`.
4. Put–call parity residual on 10,000 random parameter sets (should be ~1e-14).
5. Limiting cases table: σ → 0 (price → discounted intrinsic of the forward), T → 0 (→ payoff), S → ∞ (call ≈ S − Ke^{−rT}).

**ML model: neural network surrogate (`src/ml/mlp_bs.py`)**
- Data: 50,000 random samples. Use homogeneity: inputs $(m = S/K,\ T,\ r,\ \sigma)$, target $C/K$.
  Ranges: m ∈ [0.7, 1.3], T ∈ [0.02, 2], r ∈ [0, 0.1], σ ∈ [0.05, 0.6].
- Model: `MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", max_iter=500, early_stopping=True, random_state=SEED)`; scale inputs with `StandardScaler` (in a `Pipeline`).
- Evaluate on a held-out 20%: MAE, max error, error vs moneyness.
- **Extrapolation test:** evaluate on σ ∈ [0.6, 0.9] and T ∈ [2, 3] (outside training). Errors should blow up; plot it.
- **No-arbitrage check:** on a grid, count where predicted price decreases in S or in σ, or is non-convex in K.
- Figures: `ml_mlp_error_vs_moneyness.pdf`, `ml_mlp_extrapolation.pdf`.

**Tests:** baseline values 10.4506 / 5.5735 (±1e-4); parity; Greeks vs finite differences of `bs_price` (rel. error < 1e-5); IV round-trip for 100 random cases; IV returns NaN below intrinsic.

---

### Person 2: Zaid (Binomial tree engine)

**Code:** `src/core/binomial.py`, `src/tree_analysis/`.

**Experiments**
1. One-period replication worked example (by hand in the report, checked by code): Δ shares + B bond replicate
   the payoff in both states; show $V_0 = \Delta S + B$ equals the risk-neutral formula.
2. Tree diagram for N = 4 (stock and option value at each node). `tree_diagram_N4.pdf` (use matplotlib text and lines).
3. CRR vs JR vs LR prices at N ∈ {25, 51, 101, 201} for ITM/ATM/OTM. Table.
4. Tree Greeks vs BS Greeks: Δ, Γ, Θ against N; show Γ also oscillates. `tree_greeks_vs_bs.pdf`.
5. Check $0 < p < 1$ condition: show a parameter set (huge r, tiny σ, large Δt) where CRR gives p > 1, and explain why (arbitrage).

**ML model: decision tree regressor (`src/ml/tree_regressor.py`)**
- Same input/target design as Person 1, but target = CRR price with N = 500.
- `DecisionTreeRegressor(max_depth=d)` for d ∈ {4, 8, 12, 20}; plot accuracy vs depth.
- Key result: predictions are piecewise constant, so Δ and Γ from finite differences are 0 almost everywhere and spike at split boundaries. Plot predicted price vs S with BS overlaid, and the resulting "Δ". `ml_dtree_steps.pdf`, `ml_dtree_delta.pdf`.
- Talking point: a model can be accurate in price and useless for hedging.

**Tests:** 1-step and 2-step trees by hand (r = 0 case is easy); CRR → BS within 2e-3 at N = 2000; JR and LR also converge; LR at N = 101 within 1e-4 of BS (actual ≈ 3.4e-5); put–call parity holds in the tree exactly for every N; p in (0, 1) for baseline.

---

### Person 3: Asvin (Convergence analysis)

**Code:** `src/convergence/` (imports `binomial_price`, `bs_price`).

**Theory to write up** (report + viva)
- In CRR, $\ln S_N = \ln S_0 + \sum_{i=1}^N X_i$ with i.i.d. $X_i \in \{\pm\sigma\sqrt{\Delta t}\}$. Mean and variance
  match risk-neutral GBM to $O(\Delta t)$; CLT ⇒ $\ln S_T$ → normal ⇒ tree price → BS.
- Error is $O(1/N)$ with an oscillating term: the strike falls at a different relative position between two terminal
  nodes as N changes. For an ATM option, the strike sits exactly on a node for even N and between nodes for odd N.
- Strike-position variable: terminal node j has price $S_0e^{(2j-N)\sigma\sqrt{\Delta t}}$, so the strike sits at
  $j^* = \dfrac{\ln(K/S_0) + N\sigma\sqrt{\Delta t}}{2\sigma\sqrt{\Delta t}}$. Its fractional part
  $\lambda = j^* - \lfloor j^* \rfloor \in [0, 1)$ is the strike's position between two terminal nodes.
  The error behaves like a smooth function of λ divided by N.

**Experiments**
1. Price vs N for every N in 1..1000; even/odd in different colours. `conv_price_vs_N.pdf`.
2. Log–log |error| vs N, fitted slope with 95% CI (`np.polyfit` on N ≥ 50, or `scipy.stats.linregress`). `conv_loglog_error.pdf`.
3. CRR vs JR vs LR on the same log–log plot. LR should show ~O(1/N²).
4. Error vs moneyness K/S ∈ {0.8, 0.9, 1.0, 1.1, 1.2}, vs σ, vs T (heatmaps of N·|error|). `conv_error_heatmap.pdf`.
5. Corrections:
   - average of N and N+1: $V = \tfrac12(V_N + V_{N+1})$
   - BS-smoothing: at step N−1 replace node values with the BS price for the last Δt
   - Richardson extrapolation: $V = 2V_{2N} - V_N$ (use even N with same parity)
   Plot all on the log–log chart. `conv_corrections.pdf`.
6. Runtime vs accuracy: for each method, N needed to reach |error| < 1e-2, 1e-3, 1e-4, and its runtime
   (`time.perf_counter`, median of 5 runs). Table `conv_runtime.csv`.

**ML model: linear regression of the error (`src/ml/error_regression.py`)**
- Data: CRR error for N ∈ [20, 500] and several (K, σ, T).
- Features: $1/N$, $(-1)^N/N$, and $\lambda/N$, $\lambda^2/N$ (strike position). Target: error.
- `LinearRegression()`; report coefficients and R². With only $1/N$ and $(-1)^N/N$ for ATM, the fit is near-perfect.
- Use $V_{\text{corrected}} = V_N - \widehat{\text{error}}$; compare against corrections in experiment 5.
- Talking point: the regression "rediscovers" the theory, since the important features are exactly the ones the theory predicts.

**Tests:** fitted slope for CRR ATM is in [−1.1, −0.9]; averaging reduces max error over N ∈ [100, 200] by ≥ 5×.

---

### Person 4: Shivank (Real NSE data, implied vol, model vs market)

**Code:** `src/data/` (whole pipeline) and `src/market/`.

**Experiments**
1. Data summary table: contracts per regime, per symbol, after each filter.
2. Volatility estimates per regime: 30/60/252-day historical, EWMA, Parkinson vs ATM implied vol. Table + bar chart. `mkt_vol_estimates.pdf`.
3. Smiles: IV vs K/F per expiry, one panel per regime (NIFTY). Before/after implied-forward fix. `mkt_smile_<regime>.pdf`, `mkt_forward_fix.pdf`.
4. Pricing error: BS and CRR (N = 200) with each σ estimate vs settlement price. Metrics: mean error (bias), MAE, RMSE, mean relative error, by moneyness bucket × maturity bucket × regime. Table `mkt_errors.csv`; heatmap `mkt_error_heatmap.pdf`.
5. **Central-question chart:** |BS − CRR| (model choice) vs |BS(σ₁) − BS(σ₂)| over σ estimates (input choice) vs |BS − market|, per regime. `mkt_model_vs_input.pdf`. This is the project's headline figure.
6. Repeat 3–4 for RELIANCE and HINDUNILVR (fewer liquid strikes; note it).

**ML model: random forest for implied vol (`src/ml/iv_forest.py`) with a quadratic baseline**
- Baseline: per (date, expiry), fit $\sigma(m) = a + b(m-1) + c(m-1)^2$ with `np.polyfit`. Report a (level), b (skew), c (curvature) per regime; b and c should jump in April 2025.
- Random forest: inputs $(\ln(K/F),\ T,\ \sigma_{30\text{d}},\ \text{regime-free})$, target IV.
  `RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED)`.
- **Split by time:** train on earlier dates, test on later dates. Never a random split (leakage across strikes on the same day).
- Evaluate: IV RMSE, and the BS pricing error when σ = predicted IV, vs constant σ and vs quadratic baseline.
- Feature importance plot. `ml_rf_iv_fit.pdf`, `ml_rf_importance.pdf`.
- Talking point: a random forest's surface is step-shaped (averaged trees); mention it before you're asked.

**Tests:** loader on a 5-row sample of each format; forward recovers F to < 0.1% on synthetic parity-consistent data; filters never return contracts with days < MIN_DAYS.

---

### Person 5: Pradyumn (Dividends, time-varying σ, model risk, integration)

**Code:** `src/model_risk/`, `run_all.py`, report assembly.

**Experiments**
1. **Discrete dividends** (HINDUNILVR, a dividend payer, and/or RELIANCE):
   - Model A: BS with escrowed dividend, $S^* = S - \sum D_ie^{-rt_i}$ over dividends before expiry.
   - Model B: BS with continuous yield q = annual dividend / S.
   - Model C: tree with proportional drop at the ex-date step (Section 3.2).
   - Compare prices on a synthetic case (S = 100, D = 3 at t = 0.5) as function of strike, then vs market settlement for HUL options spanning an ex-date. `mr_dividends.pdf`.
2. **Time-varying σ:** take the NIFTY ATM implied-vol term structure on a date (IV by expiry). Build σ(t) piecewise; price a 3-month option with (a) BS using $\bar\sigma = \sqrt{\frac1T\int\sigma^2dt}$ and (b) the time-varying tree. They should nearly agree for Europeans (variance is what matters); show it and explain why. Then show where they differ (only through rate/step discretisation). Honest result either way.
3. **σ-error vs N-error:** at baseline, the price change from σ ± 1%, ± 2%, ± 5% (relative) vs the CRR error at N = 10, 50, 200. Bar chart `mr_sigma_vs_N.pdf`. Expected: σ error dominates for any N ≥ 50.

**ML model: HAR volatility forecast (`src/ml/har.py`)**
- Daily variance proxy: Parkinson $RV_t = \frac{\ln^2(H_t/L_t)}{4\ln 2}$ (from NSE High/Low; less noisy than $R_t^2$).
- $RV^{(w)}_t$ = mean of last 5 days, $RV^{(m)}_t$ = mean of last 22 days.
- Direct h-step model (h = days to expiry, use h ∈ {5, 22}):
  $\overline{RV}_{t+1:t+h} = \beta_0 + \beta_d RV_t + \beta_w RV^{(w)}_t + \beta_m RV^{(m)}_t + \varepsilon$
- `LinearRegression()` on a rolling or expanding window, training only on data before the option date.
- Forecast σ = √(252 × forecast) → feed into BS and the tree; compare pricing error vs 30-day historical σ and vs EWMA.
- Figures: `ml_har_forecast.pdf` (forecast vs realised), `ml_har_pricing.pdf`.

**Integration duties:** keep `main` runnable; `run_all.py` calls each person's `make_figures()`; final check that a fresh clone + `pip install -r requirements.txt` + `python run_all.py` reproduces everything.

**Tests:** escrowed BS with D = 0 equals plain BS; time-varying tree with constant σ equals CRR; HAR forecast uses no future data (assert max training date < option date).

---

## 6. `run_all.py` contract

Every analysis module exposes one function:
```python
def make_figures(cfg) -> list[str]:   # returns paths written
```
`run_all.py`:
```python
from src.bs_analysis import sweeps
from src.tree_analysis import compare
from src.convergence import experiments
from src.market import analysis
from src.model_risk import experiments as mr
from src.ml import mlp_bs, tree_regressor, error_regression, iv_forest, har

for mod in [sweeps, compare, experiments, analysis, mr,
            mlp_bs, tree_regressor, error_regression, iv_forest, har]:
    paths = mod.make_figures(config)
    print(mod.__name__, "->", len(paths), "files")
```
Use `--only convergence` style flags if a full run gets slow.

---

## 7. Figure style (one look across the report)

```python
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
GREEN, SLATE, GREY = "#00590D", "#334155", "#94A3B8"   # proposal colours
```
- Save as PDF, width 3.3 in (single column) or 6.8 in (full width).
- Axis labels with units; legend without frame; no titles inside figures (captions go in LaTeX).

---

## 8. Testing

```bash
python -m pytest -q
```
Minimum tests per person are listed in Section 5. Add a test for every bug you fix. CI is optional; just run
tests before every pull request.

---

## 9. Report mapping (3–4 pages)

| Report section | Owner | Key figures |
|---|---|---|
| 1. Introduction + central question | Pradyumn | – |
| 2. Models: GBM, BS, CRR, convergence theory | Adwait, Zaid, Asvin | tree diagram |
| 3. Numerical results: convergence, corrections, Greeks | Asvin, Zaid | `conv_loglog_error`, `conv_corrections`, `tree_greeks_vs_bs` |
| 4. Real market: smiles, errors, model vs input | Shivank | `mkt_smile_*`, `mkt_model_vs_input` |
| 5. Flexibility & model risk | Pradyumn | `mr_dividends`, `mr_sigma_vs_N` |
| 6. ML benchmarks (one paragraph each) | everyone | one figure total, or a summary table |
| 7. Conclusion | Pradyumn | – |
| References (models, algorithms, data sources with download dates) | everyone | – |

Page limit is tight: pick the 5–6 strongest figures; the rest go in an appendix or the repo.
Add one line in the introduction: *"Beyond the proposal, we extended the study to real NSE option prices,
implied volatility and machine-learning benchmarks."*

Confirmed: Pradyumn's ID is 2023B5AD0973G; the spelling is Asvin.

---

## 10. Timeline (6 → 15 November)

| Date | Milestone |
|---|---|
| **6–7 Nov** | `src/core/` done and tested (bs.py, binomial.py, vol.py); data downloaded; loader + cleaning on all regimes |
| 8–10 Nov | All experiments in Section 5 running; first versions of every figure |
| 10–11 Nov | ML models trained and evaluated |
| 12 Nov | Figures final; each person writes their half-page and pushes it to `report/` |
| 13 Nov | Report merged and trimmed to 3–4 pages; references checked |
| 14 Nov | Fresh-clone reproducibility check; viva rehearsal (each person 3–4 min + cross-questions) |
| **15 Nov** | Submit report + code |

Persons 1 and 2 are the critical path: until `src/core/` is merged, others should code against the functions
in `baseline_convergence.py` with the same signatures, then switch the import.

---

## 11. Viva preparation (everyone must be able to answer)

- What is the risk-neutral probability, and why isn't it the real probability of an up move?
- Why must $d < e^{r\Delta t} < u$?
- Derive the one-period replicating portfolio.
- What does each term in the BS formula mean? (Φ(d₂) = risk-neutral probability of exercise; S·Φ(d₁) = PV of receiving the stock given exercise.)
- Why does the tree converge to BS? Why does CRR oscillate between even and odd N?
- What is implied volatility, and why is there a smile/skew?
- Why does historical σ misprice market options?
- Why are NSE options European, and why does that matter?
- For your ML model: what it learns, where it fails, and how it compares with the mathematical model.

Per person, prepare the derivation or argument behind your own section on paper.
