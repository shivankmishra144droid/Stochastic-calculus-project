"""Bootstrap helpers: cluster resampling must respect within-date correlation."""
import numpy as np
import pandas as pd
import pytest

from src.stats import cluster_bootstrap, cluster_mean_ci


def _clustered(n_dates=30, per_date=200, date_sd=1.0, noise_sd=0.1, seed=0):
    rng = np.random.default_rng(seed)
    d = np.repeat(np.arange(n_dates), per_date)
    x = rng.normal(0, date_sd, n_dates)[d] + rng.normal(0, noise_sd, len(d))
    return pd.DataFrame({"date": d, "x": x, "y": 2 * x})


def test_cluster_mean_ci_matches_slow_bootstrap_and_is_wide():
    df = _clustered()
    fast = cluster_mean_ci(df, ["x"], "date").loc["x"]
    slow = cluster_bootstrap(df, lambda g: g["x"].mean(), "date", n_boot=1000)
    assert fast["mean"] == pytest.approx(df["x"].mean())
    assert fast["lo"] == pytest.approx(slow[1], abs=0.08) and fast["hi"] == pytest.approx(slow[2], abs=0.08)
    # with a common date effect the CI width is ~ 2*1.96*1/sqrt(30) = 0.72, far wider than the naive iid width
    assert 0.5 < fast["hi"] - fast["lo"] < 1.0
    naive = 2 * 1.96 * df["x"].std() / np.sqrt(len(df))
    assert fast["hi"] - fast["lo"] > 5 * naive
    assert fast["n"] == len(df) and fast["n_clusters"] == 30


def test_cluster_mean_ci_handles_nans_and_one_cluster():
    df = _clustered(n_dates=5, per_date=10)
    df.loc[df.index[::3], "y"] = np.nan
    ci = cluster_mean_ci(df, ["x", "y"], "date")
    assert ci.loc["y", "mean"] == pytest.approx(df["y"].mean())
    assert ci.loc["y", "n"] == df["y"].notna().sum()
    one = cluster_mean_ci(df[df["date"] == 0], ["x"], "date")
    assert np.isnan(one.loc["x", "lo"])

