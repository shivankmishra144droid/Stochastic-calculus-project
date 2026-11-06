"""
MAC F244 Project - Baseline: CRR convergence to Black-Scholes
European option pricing: CRR binomial tree vs Black-Scholes

What this script does:
  1. Prices the baseline European call and put with the Black-Scholes formula.
  2. Checks put-call parity.
  3. Prices the same options with a CRR binomial tree for several N and
     reports the error against Black-Scholes and the runtime.
  4. Plots the tree price against N (even/odd oscillation) and the error on a
     log-log scale, and estimates the convergence rate (expected slope -1).

Run:  python baseline_convergence.py      (needs numpy, scipy, matplotlib)
Output: printed table + convergence.pdf
"""
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Baseline contract (Section 1 of the proposal) ----------------
S0 = 100.0     # spot price
K = 100.0      # strike
r = 0.05       # risk-free rate (continuously compounded)
sigma = 0.20   # volatility
T = 1.0        # maturity in years


# Pricers live in src/core (single source of truth for the whole project).
from src.core.binomial import binomial_price
from src.core.bs import bs_price


def crr_price(S, K, T, r, sigma, N, option_type="C"):
    """N-step Cox-Ross-Rubinstein price (proposal Section 2.1); thin wrapper over src.core.binomial."""
    return binomial_price(S, K, T, r, sigma, N, option_type, method="crr")


# ---------------- 1. Black-Scholes values and put-call parity ----------------
C_bs = bs_price(S0, K, T, r, sigma, "C")
P_bs = bs_price(S0, K, T, r, sigma, "P")
parity_gap = (C_bs - P_bs) - (S0 - K * np.exp(-r * T))   # should be 0
print(f"Black-Scholes call = {C_bs:.4f}, put = {P_bs:.4f}")
print(f"Put-call parity gap = {parity_gap:.2e}\n")

# ---------------- 2. CRR prices, errors and runtime ----------------
print(f"{'N':>5} {'CRR call':>10} {'|error|':>10} {'CRR put':>10} {'runtime':>10}")
for N in [10, 50, 100, 101, 500, 1000]:
    t0 = time.perf_counter()
    c = crr_price(S0, K, T, r, sigma, N, "C")
    runtime_ms = (time.perf_counter() - t0) * 1000
    p = crr_price(S0, K, T, r, sigma, N, "P")
    print(f"{N:>5} {c:>10.4f} {abs(c - C_bs):>10.2e} {p:>10.4f} {runtime_ms:>8.2f} ms")

# ---------------- 3. Convergence for every N from 1 to 300 ----------------
Ns = np.arange(1, 301)
err = np.array([crr_price(S0, K, T, r, sigma, n, "C") - C_bs for n in Ns])
even, odd = Ns % 2 == 0, Ns % 2 == 1

# convergence rate: slope of log|error| against log N (fit on N >= 51)
slope = np.polyfit(np.log(Ns[50:]), np.log(np.abs(err[50:])), 1)[0]
print(f"\nEstimated convergence rate: error ~ N^({slope:.2f})  (theory: N^-1)")

# ---------------- 4. Plot ----------------
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
green, grey = "#00590D", "#94A3B8"
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))

ax[0].plot(Ns[even], C_bs + err[even], color=green, lw=1.2, label="even N")
ax[0].plot(Ns[odd], C_bs + err[odd], color=grey, lw=1.2, label="odd N")
ax[0].axhline(C_bs, color="k", ls="--", lw=1, label="Black–Scholes")
ax[0].set_ylim(C_bs - 0.25, C_bs + 0.25)
ax[0].set_xlabel("N")
ax[0].set_ylabel("Call price")
ax[0].legend(frameon=False, fontsize=8)

ax[1].loglog(Ns[even], np.abs(err[even]), ".", ms=3, color=green, label="even N")
ax[1].loglog(Ns[odd], np.abs(err[odd]), ".", ms=3, color=grey, label="odd N")
ax[1].loglog(Ns, 0.8 / Ns, "k:", lw=1, label=r"$O(1/N)$")
ax[1].set_xlabel("N")
ax[1].set_ylabel("|error|")
ax[1].legend(frameon=False, fontsize=8)

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("convergence.pdf")
print("Saved convergence.pdf")
