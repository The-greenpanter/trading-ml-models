from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import filter_weekend, temporal_split, make_target


def _toy_df() -> pd.DataFrame:
    ts = pd.to_datetime(
        [
            "2026-05-13 10:00",  # Wed
            "2026-05-14 10:00",  # Thu
            "2026-05-15 10:00",  # Fri
            "2026-05-16 10:00",  # Sat -> drop
            "2026-05-17 10:00",  # Sun -> drop
            "2026-05-18 10:00",  # Mon
        ],
        utc=True,
    )
    return pd.DataFrame({"timestamp": ts, "result": ["WIN", "LOSS"] * 3})


def test_filter_weekend_drops_sat_sun():
    df = _toy_df()
    out = filter_weekend(df)
    assert len(out) == 4
    assert set(out["timestamp"].dt.dayofweek.unique()) <= {0, 2, 3, 4}


def test_temporal_split_is_ordered():
    df = _toy_df().sort_values("timestamp").reset_index(drop=True)
    s = temporal_split(df, train_frac=0.5, val_frac=0.25)
    assert len(s.train) + len(s.val) + len(s.test) == len(df)
    assert s.train["timestamp"].max() <= s.val["timestamp"].min()
    assert s.val["timestamp"].max() <= s.test["timestamp"].min()


def test_make_target_binary():
    df = _toy_df()
    y = make_target(df)
    assert set(y.tolist()) <= {0, 1}
    assert len(y) == len(df)
