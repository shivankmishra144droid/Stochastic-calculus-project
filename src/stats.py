"""Uncertainty for market results: bootstrap that resamples whole trading dates.

Contracts on the same date share the spot, the forward, the volatility inputs and the smile, so they are not
independent: resampling single contracts would give far too narrow intervals. We resample dates with replacement
(a cluster bootstrap). With only 4-5 dates per regime the intervals are themselves rough (and percentile
intervals from few clusters tend to be too narrow), so they are a lower bound on the real uncertainty.
"""
import numpy as np
import pandas as pd

import config

N_BOOT = 2000


def cluster_bootstrap(df: pd.DataFrame, stat, cluster: str = "date", n_boot: int = N_BOOT,
                      seed: int = config.SEED, level: float = 0.95):
    """Point estimate and percentile CI of `stat(df)` when whole `cluster` groups are resampled.

    `stat` maps a DataFrame to a float (or a 1-d array of floats). Returns (point, lo, hi) as floats or arrays.
    """
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby(cluster, sort=True)]
    point = np.asarray(stat(df), dtype=float)
    if len(groups) < 2:
        return _out(point, np.full_like(point, np.nan), np.full_like(point, np.nan))
    boots = np.empty((n_boot,) + point.shape)
    for b in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        boots[b] = stat(pd.concat([groups[i] for i in idx], ignore_index=True))
    a = (1 - level) / 2
    lo, hi = np.nanpercentile(boots, [100 * a, 100 * (1 - a)], axis=0)
    return _out(point, lo, hi)


def cluster_mean_ci(df: pd.DataFrame, cols, cluster: str = "date", n_boot: int = N_BOOT,
                    seed: int = config.SEED, level: float = 0.95) -> pd.DataFrame:
    """Fast cluster bootstrap for contract-level means of several columns (NaNs skipped per column).

    The mean over contracts is a ratio of sums, so per-cluster sums and counts are resampled.
    Returns a DataFrame indexed by column with mean, lo, hi, n (contracts) and n_clusters.
    """
    cols = list(cols)
    g = df.groupby(cluster)
    sums = g[cols].sum(min_count=1).fillna(0.0).to_numpy()
    cnts = g[cols].count().to_numpy().astype(float)
    k = len(sums)
    rng = np.random.default_rng(seed)
    point = sums.sum(0) / cnts.sum(0)
    if k < 2:
        lo = hi = np.full(len(cols), np.nan)
    else:
        idx = rng.integers(0, k, (n_boot, k))
        with np.errstate(invalid="ignore", divide="ignore"):
            boots = sums[idx].sum(1) / cnts[idx].sum(1)
        a = (1 - level) / 2
        lo, hi = np.nanpercentile(boots, [100 * a, 100 * (1 - a)], axis=0)
    return pd.DataFrame({"mean": point, "lo": lo, "hi": hi, "n": cnts.sum(0).astype(int), "n_clusters": k},
                        index=cols)


def _out(point, lo, hi):
    if point.ndim == 0:
        return float(point), float(lo), float(hi)
    return point, lo, hi
