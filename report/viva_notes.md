# Viva answer sheet (everyone)

Numbers come from `results/tables/` after `python run_all.py`.

**"So which model should I use?"** (the project's answer, report Table "Decision rule").
Black–Scholes for European options: it is exact and ~0.2 ms, and a tree only adds numerical error (0.05 bp on NSE
data). Same with a volatility term structure: use BS with the average variance. Use a tree when the contract has a
known cash dividend you model from spot (escrowed BS underprices by up to 1.3 % ATM), early exercise or path
dependence (not tested: NSE options are European). If you use a tree, use Leisen–Reimer with odd N (10⁻⁴ in 28–66
steps at every strike). The bigger decision is σ: implied vol if option prices exist (6.3 bp), else an ML blend in a
familiar regime (10.1 bp) or EWMA / 30-day + VIX in a new one (12.2–12.7 bp). If the dividend is already in a market
forward, BS on that forward is enough: on Reliance all treatments missed by the same Rs 0.7–1.1.

**"What are the weaknesses?"** Say them first, before the examiner does: 14 trading days and 3 underlyings
(bootstrap intervals over so few dates are rough); settlements are closing VWAPs; the T-bill rate is not the stock carry (that is why the
dividend cannot be identified); VIX is NIFTY volatility used for stocks; no American options, so the tree's
early-exercise advantage is argued, not measured; and the smile (6.3 bp) is outside both models.

**"Do the results depend on your filters or day count?"** No, we checked both (`rob_liquidity`, `rob_clock`).
Looser filters (volume ≥ 1, no OI) or stricter ones (volume ≥ 100, OI ≥ 500) keep numerical at 0.05 bp, input at
10.4–11.2 bp and the smile at 6.1–6.3 bp. Measuring time in trading days (σ²·n_td/252) does not reduce errors at any
maturity (+0.2 to +1.8 bp), so we keep calendar days.

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
Fixes: average N and N+1, Richardson on even N, BS-smoothing, or Leisen–Reimer (O(1/N²)). Only LR is robust at
every strike: Richardson works at the money only, because there λ = 0 for every even N; off the money it is no
better than CRR (Table 1).

**Implied volatility, smile and skew.** Implied vol is the σ that makes BS equal the market price. A constant-σ
model would give a flat line. Real smiles come from fat tails and jumps, and the equity skew from crash fear
(demand for OTM puts) and volatility rising when prices fall. NIFTY skew b went from −0.21 (calm, Jul 2025) and −0.25 (Jan) to −0.69 (Apr 2025).

**Why historical σ misprices options.** It looks backwards and is a single number. The market prices the expected
*future* volatility under Q plus a risk premium, and different levels per strike (the smile). In the April 2025 regime the
NIFTY ATM IV averaged 23 %, against 17 % for 30-day historical (`mkt_vol_estimates.csv`).

**Why NSE options are European, and why it matters.** NSE index and stock options can be exercised only at expiry
(since 2010–11 for stock options). So early exercise is impossible, BS applies directly, and the tree's
early-exercise advantage is irrelevant. Careful: that is *not* why they agree to 0.1 bp on our data. They agree
because the CRR tree is a discretisation of the same GBM model with the same σ, so the gap is numerical error
(O(1/N); at N = 200 it is tiny). The real model error is the smile, which neither can produce.

**Our data choices.** Regimes were chosen from NIFTY 30-day realised vol: calm Jul 2025 (~9 %), normal Jan 2025
(~13 %), volatile Apr 2025 (~17–21 %, the tariff shock). The rate is the RBI 91-day cut-off YTM before each regime,
r = ln(1+y). We use settlement prices **set by trades** (when a contract barely trades, NSE publishes a theoretical
settlement from its own model, visible as settlement ≠ close; using it would compare our model with NSE's). We keep
the nearest 3 expiries with ≥ 3 days, volume ≥ 10, OI ≥ 50, premium ≥ Rs 1, 0.8 ≤ K/F ≤ 1.2, and drop static
arbitrage (below discounted intrinsic on F; non-monotone, non-convex or too steep across strikes beyond 2 ticks or
1 %, lowest-volume contract dropped). That leaves 28,114 → 6,052 contracts, all with an implied vol. Spot histories:
Reliance pre-bonus prices halved (no fake −69 % log return on 28 Oct 2024), dividends in the same split-adjusted
units, close-to-close vols on dividend-adjusted closes, and the two one-hour Saturday test sessions dropped. The implied forward from parity removes the put/call jump at
K/S = 1.

## The ML part: can ML choose a better σ? (everyone)
**Question.** σ drives pricing error, so can ML combine our volatility estimates into a better σ than any single one?
**Inputs:** 30/60/252-day historical, EWMA, Parkinson, India VIX (all known at the close, no option prices).
**Target:** that day's ATM implied vol (nearest expiry ≥ 7 days). **Data:** 3 underlyings × 14 days = 42 rows.
**Models:** linear regression and random forest on all six inputs; linear regression on 30-day + VIX only.
**Testing:** never on a day it trained on. "Unseen day" = trained on the other 13 days (it has seen the regime:
best case); "unseen regime" = trained on the other two regimes.

| σ used | Pricing error (bp) |
|---|---|
| 30-day historical | 14.9 |
| EWMA (best single) | 12.7 |
| India VIX | 16.8 |
| Random forest, unseen day | **10.1** (−4.8 vs 30-day, CI [−7.4, −2.5]; −2.6 vs EWMA, CI [−4.7, −0.7]) |
| Random forest / regression (all inputs), unseen regime | 19.5 / 21.7 (worse) |
| Regression on 30-day + VIX, unseen regime | 12.2 (level with EWMA) |
| Same-day ATM implied vol (uses option prices) | 6.3 |

**Answer.** ML picks a better σ (21–32% less error) when it has seen similar market conditions; in a brand-new
regime the six-input models overfit (42 rows), and the simple 30-day + VIX blend or EWMA alone is safer.
**Likely questions.** *Why so few rows?* ATM IV exists only on our 14 bhavcopy days; more days would help most.
*Is "unseen day" cheating?* It never sees that day, but it does see later days of the same regime, so it is a best
case; "unseen regime" is the strict test. *Why was 30-day + VIX chosen?* One backward-looking and one forward-looking
input; it was added after the larger models overfit, so we present it as indicative.
