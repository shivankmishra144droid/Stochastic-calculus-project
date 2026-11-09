"""ML sigma blend: the test day / regime must never be seen in training, and the regression must recover a known
blend of the inputs."""
import numpy as np
import pandas as pd
import pytest

from src.ml import sigma_blend as sb


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=12)
    rows = []
    for i, d in enumerate(dates):
        for sym in ("A", "B", "C"):
            x = {k: rng.uniform(0.08, 0.3) for k in sb.ALL_INPUTS}
            rows.append(dict(symbol=sym, date=d, regime=["calm", "normal", "volatile"][i // 4], **x,
                             atm_iv=0.02 + 0.5 * x["hv30"] + 0.4 * x["vix"]))
    return pd.DataFrame(rows)


def test_held_out_group_never_used_in_training():
    t = _synthetic()
    make, inputs = sb.MODELS["LR (all)"]
    for group in ("date", "regime"):
        base = sb.out_of_sample(t, make, inputs, group)
        g0 = t[group].iloc[0]
        t2 = t.copy()
        t2.loc[t2[group] == g0, "atm_iv"] += 5.0          # corrupt the held-out targets
        moved = sb.out_of_sample(t2, make, inputs, group)
        held = t[group] == g0
        assert np.allclose(base[held], moved[held])       # predictions for that group are unchanged
        assert not np.allclose(base[~held], moved[~held])  # (other groups do train on it)


def test_two_input_regression_recovers_known_blend():
    t = _synthetic()
    make, inputs = sb.MODELS["LR (30d+VIX)"]
    pred = sb.out_of_sample(t, make, inputs, "regime")
    assert np.allclose(pred, t["atm_iv"], atol=1e-10)     # exact linear target is learned out of sample


def test_daily_table_is_complete_and_point_in_time():
    df = sb.load_priced()
    t = sb.daily_table(df)
    assert not t.duplicated(["symbol", "date"]).any()
    assert t[sb.ALL_INPUTS + ["atm_iv"]].notna().all().all()
    assert len(t) == df.groupby(["symbol", "date"]).ngroups
    # inputs are the same-day values the cleaned option table carries (computed with as_of: no look-ahead)
    first = df.groupby(["symbol", "date"])["hv30"].first().reset_index()
    m = t.merge(first, on=["symbol", "date"], suffixes=("", "_opt"))
    assert np.allclose(m["hv30"], m["hv30_opt"])
    # hv252 is filled (with hv60) only where it was missing
    assert (t.loc[~t["hv252_filled"], "hv252"].to_numpy() ==
            df.groupby(["symbol", "date"])["hv252"].first().dropna().to_numpy()).all()
