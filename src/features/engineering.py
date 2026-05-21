"""Feature engineering for the alert snapshot.

Inputs are rows of the rule-based scanner (one per alert). We build:
  - 7 vote booleans (already binary in the CSV)
  - ADX, ATR, score (numerical)
  - direction one-hot (long/short)
  - regime one-hot (TRENDING/RANGING)
  - symbol one-hot
  - hour-of-day one-hot (UTC)
  - day-of-week one-hot (Mon-Fri only after weekend filter)
  - lag_1_result: previous trade's WIN/LOSS (global) -- autocorrelation probe
  - lag_1_symbol_result: previous trade's result for the same symbol
  - lag_1_symbol_dir_result: previous trade's result for the same symbol+direction

The lag features are computed in time order. They must be re-fit per fold to
avoid leakage across the train/val/test boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

VOTE_COLS = [
    "vote_ema_dir",
    "vote_ema200",
    "vote_macd_mom",
    "vote_rsi7",
    "vote_bb",
    "vote_stoch",
    "vote_vol",
]

SYMBOLS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCAD"]
HOURS = list(range(24))
DOWS = [0, 1, 2, 3, 4]  # Mon-Fri only after weekend filter
REGIMES = ["TRENDING", "RANGING"]
DIRECTIONS = ["LONG", "SHORT"]


@dataclass
class FeatureSet:
    X: pd.DataFrame
    feature_names: List[str]


def _onehot(series: pd.Series, categories: list, prefix: str) -> pd.DataFrame:
    out = pd.DataFrame(index=series.index)
    for cat in categories:
        out[f"{prefix}_{cat}"] = (series == cat).astype(int)
    return out


def _lag_result(df: pd.DataFrame, group_cols: list | None) -> np.ndarray:
    """Shifted previous result (1=WIN, 0=LOSS, NaN if no prior)."""
    res = (df["result"] == "WIN").astype(float)
    if group_cols is None:
        prev = res.shift(1)
    else:
        prev = res.groupby([df[c] for c in group_cols]).shift(1)
    return prev.fillna(0.5).values  # neutral prior when unknown


def build_features(df: pd.DataFrame) -> FeatureSet:
    """Return a feature DataFrame ready for XGBoost.

    `df` must be sorted by timestamp ascending and already weekend-filtered.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    parts: list[pd.DataFrame] = []

    # 1. votes (binary) + numerical
    parts.append(df[VOTE_COLS].astype(float))
    parts.append(df[["score", "adx", "atr"]].astype(float))

    # 2. direction / regime / symbol one-hots
    parts.append(_onehot(df["direction"].str.upper(), DIRECTIONS, "dir"))
    parts.append(_onehot(df["regime"].str.upper(), REGIMES, "regime"))
    parts.append(_onehot(df["symbol"].str.upper().str.replace("/", "", regex=False), SYMBOLS, "sym"))

    # 3. hour / day-of-week one-hot
    ts = df["timestamp"]
    parts.append(_onehot(ts.dt.hour, HOURS, "h"))
    parts.append(_onehot(ts.dt.dayofweek, DOWS, "dow"))

    # 4. lag features (use 0.5 as neutral prior; later folds will recompute)
    lags = pd.DataFrame({
        "lag1_result": _lag_result(df, None),
        "lag1_symbol_result": _lag_result(df, ["symbol"]),
        "lag1_symbol_dir_result": _lag_result(df, ["symbol", "direction"]),
    }, index=df.index)
    parts.append(lags)

    X = pd.concat(parts, axis=1)
    X = X.astype(float)
    return FeatureSet(X=X, feature_names=list(X.columns))
