"""Unit tests for the data conditioning utilities.

These tests pin only shape / NaN behaviour / sign invariants — not the
specific numeric outputs — so the suite stays robust under future
implementation tweaks (different rolling backend, different FFD weight
caching, etc.).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.conditioning import (
    categorize_adx,
    dow_one_hot,
    fractional_diff,
    hour_one_hot,
    log_returns,
    robust_scale,
    rolling_zscore,
    spread_atr_ratio,
    vol_normalize,
    winsorize,
)


def test_log_returns_shape_and_first_nan():
    prices = np.array([1.0, 1.01, 1.02, 1.03, 1.04])
    r = log_returns(prices)
    assert r.shape == prices.shape
    assert np.isnan(r[0])
    assert np.allclose(r[1], np.log(1.01 / 1.0))
    # short input
    assert log_returns(np.array([1.0])).shape == (1,)
    assert np.isnan(log_returns(np.array([1.0]))[0])


def test_log_returns_handles_zero_and_negative():
    prices = np.array([1.0, 0.0, -1.0, 1.0])
    r = log_returns(prices)
    assert r.shape == prices.shape
    # zero and negative-ratio cases must be NaN, not -inf
    assert np.isnan(r[1])
    assert np.isnan(r[2])


def test_winsorize_clips_outliers():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    x[0] = 1e6
    x[1] = -1e6
    w = winsorize(x, p_low=1, p_high=99)
    assert w.max() < 1e6
    assert w.min() > -1e6
    # the 1st percentile must be the new floor (to within ties)
    assert w.min() >= np.percentile(x, 1) - 1e-9


def test_winsorize_preserves_nans():
    x = np.array([1.0, np.nan, 2.0, 3.0, 100.0])
    out = winsorize(x, p_low=10, p_high=90)
    assert np.isnan(out[1])


def test_vol_normalize_short_input_all_nan():
    out = vol_normalize(np.array([0.01, -0.02, 0.005]), window=50)
    assert np.all(np.isnan(out))


def test_vol_normalize_unit_std_after_norm():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(500) * 0.01
    vn = vol_normalize(r, window=50)
    valid = vn[~np.isnan(vn)]
    # rolling-normalised series should have order-1 magnitude
    assert valid.std() == pytest.approx(1.0, abs=0.5)


def test_rolling_zscore_window_warmup():
    x = np.arange(200, dtype=float)
    z = rolling_zscore(x, window=50)
    assert np.all(np.isnan(z[:49]))
    assert not np.isnan(z[100])


def test_rolling_zscore_constant_yields_nan():
    x = np.ones(120, dtype=float)
    z = rolling_zscore(x, window=50)
    # zero variance -> divide-by-zero -> NaN by contract
    assert np.all(np.isnan(z[60:]))


def test_fractional_diff_shape_and_warmup():
    rng = np.random.default_rng(2)
    # cumulative sum of returns gives a non-stationary "price" series
    x = np.cumsum(rng.standard_normal(500) * 0.01) + 100.0
    out = fractional_diff(x, d=0.4, thres=1e-3)
    assert out.shape == x.shape
    # the warmup region is NaN
    assert np.isnan(out[0])
    # later values are finite
    assert np.isfinite(out[-1])


def test_fractional_diff_d_zero_is_identity():
    x = np.arange(50, dtype=float)
    out = fractional_diff(x, d=0.0, thres=1e-3)
    valid = ~np.isnan(out)
    assert np.allclose(out[valid], x[valid])


def test_categorize_adx_buckets():
    adx = np.array([10.0, 20.0, 30.0, 40.0, 50.0, np.nan])
    cats = categorize_adx(adx)
    assert cats.shape == (6, 3)
    # 10 -> weak; 20 -> moderate; 30 -> moderate; 40 -> strong; 50 -> strong
    assert cats[0].tolist() == [1, 0, 0]
    assert cats[1].tolist() == [0, 1, 0]
    assert cats[2].tolist() == [0, 1, 0]
    assert cats[3].tolist() == [0, 0, 1]
    assert cats[4].tolist() == [0, 0, 1]
    # NaN row is all zeros
    assert cats[5].tolist() == [0, 0, 0]
    # exactly one-hot for finite rows
    finite = cats[:-1]
    assert (finite.sum(axis=1) == 1).all()


def test_hour_one_hot_shape_and_unique():
    ts = pd.Series(pd.to_datetime(
        ["2026-05-13 00:30", "2026-05-13 09:00", "2026-05-13 23:59"], utc=True))
    out = hour_one_hot(ts)
    assert out.shape == (3, 24)
    assert (out.sum(axis=1) == 1).all()
    assert out["h_0"].iloc[0] == 1
    assert out["h_9"].iloc[1] == 1
    assert out["h_23"].iloc[2] == 1


def test_dow_one_hot_only_weekdays():
    ts = pd.Series(pd.to_datetime(
        ["2026-05-18 10:00", "2026-05-19 10:00", "2026-05-22 10:00"], utc=True))  # Mon, Tue, Fri
    out = dow_one_hot(ts)
    assert out.shape == (3, 5)
    assert out["dow_0"].iloc[0] == 1
    assert out["dow_1"].iloc[1] == 1
    assert out["dow_4"].iloc[2] == 1


def test_spread_atr_ratio_scale_invariance():
    # if we scale both spread and ATR by the same factor, the ratio is invariant
    r1 = spread_atr_ratio(np.array([1.0, 2.0]), np.array([10.0, 20.0]))
    r2 = spread_atr_ratio(np.array([10.0, 20.0]), np.array([100.0, 200.0]))
    assert np.allclose(r1, r2)


def test_spread_atr_ratio_zero_atr_is_nan():
    out = spread_atr_ratio(np.array([1.0]), np.array([0.0]))
    assert np.isnan(out[0])


def test_robust_scale_unit_iqr():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(1000)
    s = robust_scale(x)
    q75, q25 = np.nanpercentile(s, [75, 25])
    assert q75 - q25 == pytest.approx(1.0, abs=0.05)
    assert abs(np.nanmedian(s)) < 1e-9


def test_robust_scale_constant_returns_zeros():
    out = robust_scale(np.ones(10))
    assert np.allclose(out, 0)


def test_robust_scale_external_stats():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = robust_scale(x, median=3.0, iqr=2.0)
    assert np.allclose(out, (x - 3.0) / 2.0)
