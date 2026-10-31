# European Option Pricing: Binomial Trees vs Black–Scholes

MAC F244 project (Semester I, 2026–27). This project compares the CRR binomial tree with the Black–Scholes
model: convergence, Greeks, flexibility (dividends, time-varying volatility), and fit to real NSE option
prices. Each part is benchmarked against a machine-learning model.

## Quick start
```bash
pip install -r requirements.txt
python baseline_convergence.py
```
`baseline_convergence.py` prices the baseline contract (S0 = K = 100, r = 5%, σ = 20%, T = 1) with
Black–Scholes and a CRR tree, checks put–call parity, and plots how the tree converges to the BS price.

## Team and workstreams
| Member | Workstream | ML model |
|---|---|---|
| Adwait Eklavya | Black–Scholes theory, pricer, Greeks, sensitivity | Neural network (MLPRegressor) learning BS price |
| Mohammad Zaid Agha | Binomial engine (CRR, JR, LR), tree Greeks, tests | Decision-tree regressor on tree prices |
| Asvin Sunderiyal | Convergence analysis, corrections, runtime | Linear regression of the error on 1/N, (−1)^N/N |
| Shivank Mishra | Real NSE data, implied vol, smile, model vs market | Random forest for implied vol (quadratic fit baseline) |
| Pradyumn Venkata Rayasam | Dividends, time-varying σ, model risk, integration | HAR regression volatility forecast |

## How we work
- **Never push directly to `main`.** Create a branch (`git checkout -b adwait-bs-greeks`), push it,
  and open a pull request. Someone else reviews it before merging.
- Persons 1 and 2: working pricers by **3 November**. Everyone else imports them.
- Put shared code in Python modules, not notebooks, so it can be imported. Notebooks are fine for exploration.
- Every function gets a docstring saying which equation of the proposal it implements.
