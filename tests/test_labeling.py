from __future__ import annotations

import pandas as pd

from src.data.labeling import TripleBarrierConfig, triple_barrier_label, binary_from_result


def _ohlc():
    idx = pd.date_range("2026-01-01", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open":  [1.0] * 10,
        "high":  [1.0, 1.0, 1.0, 1.0, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0],
        "low":   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "close": [1.0] * 10,
    }, index=idx)
    return df


def test_triple_barrier_upper_hit():
    df = _ohlc()
    label = triple_barrier_label(df, df.index[0], atr=0.05, config=TripleBarrierConfig(k_up=1.0, k_lo=1.0, n_bars=6, side=1))
    assert label == 1


def test_triple_barrier_timeout():
    df = _ohlc()
    label = triple_barrier_label(df, df.index[0], atr=10.0, config=TripleBarrierConfig(k_up=1.0, k_lo=1.0, n_bars=3, side=1))
    assert label == 0


def test_binary_from_result():
    s = pd.Series(["WIN", "LOSS", "WIN"])
    y = binary_from_result(s)
    assert list(y) == [1, 0, 1]
