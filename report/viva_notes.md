# Viva answer sheet (everyone)

Numbers come from `results/tables/` after `python run_all.py`.

**Risk-neutral probability q.** q = (e^{(r−δ)Δt} − d)/(u − d) is the weight that makes the discounted stock a
martingale, so pricing by expectation under q equals the cost of replication. It is not the real up-probability:
the real drift μ never enters, because the option is priced by hedging, not by forecasting.

**Why d < e^{rΔt} < u.** If e^{rΔt} ≥ u the bond beats the stock in every state (short stock, buy bond); if
e^{rΔt} ≤ d the stock beats the bond (borrow, buy stock). Both are arbitrages, and either makes q ∉ (0,1).
Example in `tree_arbitrage_example.csv`: r = 50 %, σ = 5 %, N = 1 gives q = 6.97.

**One-period replication.** Hold Δ shares and B in the bond with ΔSu + Be^{rΔt} = V_u and ΔSd + Be^{rΔt} = V_d.
So Δ = (V_u − V_d)/(Su − Sd), B = e^{−rΔt}(uV_d − dV_u)/(u − d), and V_0 = ΔS + B = e^{−rΔt}[qV_u + (1−q)V_d].
The worked baseline numbers are in `tree_replication_example.csv`: Δ = 0.550, B = −42.82, V_0 = 12.16.

**Terms of the BS formula.** Φ(d₂) is the risk-neutral probability of exercise, so Ke^{−rT}Φ(d₂) is the present value
of paying K. SΦ(d₁) is the present value of receiving the stock if exercised (stock-measure probability). Φ(d₁) is also
the call's delta.

**Why the tree converges to BS.** ln S_N is a sum of N i.i.d. ±σ√Δt steps with the GBM mean and variance to O(Δt).
By the CLT, ln S_T becomes normal, so the terminal distribution becomes lognormal, and the price (an expectation of a
bounded-growth payoff) converges to BS.

**Why CRR oscillates between even and odd N.** Terminal nodes are spaced 2σ√Δt apart in log-price. The strike's
position λ between two nodes changes with N. At the money, λ = 0 for even N (the strike is on a node) and ½ for odd N.
The error is (A + Bλ(1−λ))/N plus higher order, so it zig-zags. Our regression recovers λ(1−λ) with R² ≥ 0.996.
Fixes: average N and N+1, Richardson on even N, BS-smoothing, or Leisen–Reimer (O(1/N²)).

**Implied volatility, smile and skew.** Implied vol is the σ that makes BS equal the market price. A constant-σ
model would give a flat line. Real smiles come from fat tails and jumps, and the equity skew from crash fear
(demand for OTM puts) and volatility rising when prices fall. NIFTY skew b went from −0.21 (Jan) to −0.64 (Apr 2025).

**Why historical σ misprices options.** It looks backwards and is a single number. The market prices the expected
*future* volatility under Q plus a risk premium, and different levels per strike (the smile). In the April 2025 regime the
NIFTY ATM IV averaged 23 %, against 17 % for 30-day historical (`mkt_vol_estimates.csv`).

**Why NSE options are European, and why it matters.** NSE index and stock options can be exercised only at expiry
(since 2010–11 for stock options). So early exercise is impossible, BS applies directly, and the tree's
early-exercise advantage is irrelevant. That is why the two models agree to 0.1 bp on our data.

**Our data choices.** Regimes were chosen from NIFTY 30-day realised vol: calm Jul 2025 (~9 %), normal Jan 2025
(~13 %), volatile Apr 2025 (~17–21 %, the tariff shock). The rate is the RBI 91-day cut-off YTM before each regime,
r = ln(1+y). We use settlement prices, keep the nearest 3 expiries with ≥ 3 days, volume ≥ 10, OI ≥ 50 and
0.8 ≤ K/F ≤ 1.2, which leaves 28,114 → 7,718 contracts. The implied forward from parity removes the put/call jump at
K/S = 1.

## Your own ML model (prepare 1 minute each)
| Person | What it learns | Where it fails | Versus the maths |
|---|---|---|---|
| Adwait (MLP) | C/K from (S/K, T, r, σ); MAE 0.18 per ₹100 | outside the training range (4–5× worse); no-arbitrage shape broken | BS is exact everywhere, at no training cost |
| Zaid (decision tree) | CRR prices; MAE ≈ 1 per ₹100 at depth 20 | piecewise constant ⇒ Δ = 0 almost everywhere | tree/BS Greeks are smooth and hedgeable |
| Asvin (error regression) | e_N on 1/N, (−1)^N/N, λ/N, λ²/N; R² ≥ 0.996 | without λ features, R² ≈ 0 off the money | it rediscovers the theory's error term |
| Shivank (random forest IV) | the smile shape from ln K/F, T, σ₃₀ | cannot learn a new regime's level (worse pricing on Jul) | the smile is an input problem, not a model problem |
| Pradyumn (HAR) | future realised vol from daily/weekly/monthly RV | ties with the naive average; Parkinson misses overnight gaps | better σ inputs help more than more tree steps |
