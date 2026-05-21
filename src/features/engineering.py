"""Feature engineering for the alert snapshot.

Inputs are rows of the rule-based scanner (one per alert). We build:
  - 7 vote booleans (already binary in the CSV)
  - score (discrete numerical, kept raw)
  - ADX -> 3 one-hot buckets (weak / moderate / strong) using Wilder's
    classic thresholds (see ``src.data.conditioning.categorize_adx``)
  - ATR -> relative ATR = ATR / entry (scale-invariant across symbols)
  - direction / regime / symbol one-hot
  - hour-of-day one-hot (UTC) — non-ordinal so intraday seasonality can be
    captured in a single split per category
  - day-of-week one-hot (Mon-Fri only after weekend filter)
  - per-symbol log returns of the entry price, lags 1/2/5 (zero-imputed on
    insufficient history). Note: the snapshot is per-alert (irregular in
    time), so these are "inter-alert log price diffs per symbol", not
    bar returns. Documented in ``docs/DATA_CONDITIONING.md``.
  - vol-normalized log returns (window=50) — comparable across regimes
  - rolling z-score of log returns (window=50) — regime-shift aware
  - spread/ATR proxy: sl_pips / (atr*1e4) — execution-friction in
    volatility units. We use ``sl_pips`` as a proxy because the CSV has
    no broker spread column.
  - lag_1_result / lag_1_symbol_result / lag_1_symbol_dir_result
    (autocorrelation probes; computed in time order)

Per-fold scaling
----------------
The transforms in ``conditioning.py`` are split into two families:

1. *Trailing* (vol_normalize, rolling_zscore, log_returns) — use only the
   past, so they are leakage-safe even when computed on the full series.
2. *Global stat* (winsorize percentiles, robust_scale median/IQR) — must
   be fit on the training fold only. The pipeline exposes ``fit_scalers``
   and ``apply_scalers`` so ``train.py`` can fit on ``train_idx`` and apply
   to ``test_idx`` inside each walk-forward fold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.conditioning import (
    categorize_adx,
    dow_one_hot,
    hour_one_hot,
    log_returns,
    robust_scale,
    rolling_zscore,
    spread_atr_ratio,
    vol_normalize,
    winsorize,
)

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
DOWS = [0, 1, 2, 3, 4]
REGIMES = ["TRENDING", "RANGING"]
DIRECTIONS = ["LONG", "SHORT"]

# Features that benefit from a global robust scaler. These are the
# continuous-ratio features that, while bounded in principle, can still
# carry occasional fat tails (we want their median/IQR fit on the training
# fold only).
SCALABLE_FEATURES = [
    "atr_rel",
    "spread_atr_ratio",
    "vol_normalized_atr",
]


@dataclass
class FeatureSet:
    X: pd.DataFrame
    feature_names: List[str]


@dataclass
class FoldScalers:
    """Per-fold fitted statistics (median, IQR, winsorize percentiles).

    Bundles all global stats needed to scale a held-out test fold without
    leakage. Pass to ``apply_scalers`` after fitting on the train fold.
    """

    robust: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    winsor: Dict[str, Tuple[float, float]] = field(default_factory=dict)


def _onehot(series: pd.Series, categories: list, prefix: str) -> pd.DataFrame:
    out = pd.DataFrame(index=series.index)
    for cat in categories:
        out[f"{prefix}_{cat}"] = (series == cat).astype(int)
    return out


def _lag_result(df: pd.DataFrame, group_cols: list | None) -> np.ndarray:
    """Shifted previous result (1=WIN, 0=LOSS, 0.5 neutral if no prior)."""
    res = (df["result"] == "WIN").astype(float)
    if group_cols is None:
        prev = res.shift(1)
    else:
        prev = res.groupby([df[c] for c in group_cols]).shift(1)
    return prev.fillna(0.5).values


def _per_symbol_transform(
    df: pd.DataFrame,
    value_col: str,
    func,
    **kwargs,
) -> np.ndarray:
    """Apply a 1-D transform per symbol, preserving the original ordering.

    The snapshot is already sorted by global timestamp; here we group by
    symbol, sort within group, apply ``func``, then reassemble in the
    original index order.
    """
    out = np.full(len(df), np.nan, dtype=float)
    for sym, grp in df.groupby("symbol", sort=False):
        sub = grp.sort_values("timestamp")
        vals = func(sub[value_col].to_numpy(dtype=float), **kwargs)
        out[sub.index.to_numpy()] = vals
    return out


def _per_symbol_lag(values: np.ndarray, df: pd.DataFrame, k: int) -> np.ndarray:
    """Shift ``values`` by ``k`` rows within each symbol (zero-imputed)."""
    s = pd.Series(values, index=df.index)
    shifted = s.groupby(df["symbol"]).shift(k)
    return shifted.fillna(0.0).to_numpy()


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_features(df: pd.DataFrame) -> FeatureSet:
    """Return a feature DataFrame ready for XGBoost.

    ``df`` must be sorted by timestamp ascending and already weekend-filtered.
    Trailing transforms (rolling z-score, vol-normalize, log returns) are
    computed here; global-stat scaling is left to ``fit_scalers`` +
    ``apply_scalers`` so the caller can fit per fold.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    parts: list[pd.DataFrame] = []

    # 1. votes (binary)
    parts.append(df[VOTE_COLS].astype(float))

    # 2. score (discrete) -- kept raw, it's already on a small ordinal scale
    parts.append(df[["score"]].astype(float))

    # 3. ADX -> 3-bucket one-hot (replaces raw ADX)
    adx_cats = categorize_adx(df["adx"].to_numpy(dtype=float))
    parts.append(pd.DataFrame(
        adx_cats,
        columns=["adx_weak", "adx_moderate", "adx_strong"],
        index=df.index,
    ).astype(float))

    # 4. ATR -> relative ATR (scale-invariant across symbols)
    atr_rel = (df["atr"].to_numpy(dtype=float) / df["entry"].to_numpy(dtype=float))
    parts.append(pd.DataFrame({"atr_rel": atr_rel}, index=df.index))

    # 5. categorical one-hots
    parts.append(_onehot(df["direction"].str.upper(), DIRECTIONS, "dir"))
    parts.append(_onehot(df["regime"].str.upper(), REGIMES, "regime"))
    parts.append(
        _onehot(df["symbol"].str.upper().str.replace("/", "", regex=False), SYMBOLS, "sym")
    )

    # 6. hour / day-of-week one-hot
    ts = df["timestamp"]
    parts.append(hour_one_hot(ts))
    parts.append(dow_one_hot(ts))

    # 7. per-symbol log-returns (inter-alert) + lags 1/2/5
    base_ret = _per_symbol_transform(df, "entry", log_returns)
    # zero-impute the leading-NaN row of each symbol so the lag pipeline
    # below produces dense columns suitable for the tree booster.
    base_ret_filled = np.where(np.isfinite(base_ret), base_ret, 0.0)
    log_ret_lag1 = _per_symbol_lag(base_ret_filled, df, 1)
    log_ret_lag2 = _per_symbol_lag(base_ret_filled, df, 2)
    log_ret_lag5 = _per_symbol_lag(base_ret_filled, df, 5)
    parts.append(pd.DataFrame({
        "log_returns_lag1": log_ret_lag1,
        "log_returns_lag2": log_ret_lag2,
        "log_returns_lag5": log_ret_lag5,
    }, index=df.index))

    # 8. vol-normalized log return (window=50) + rolling z-score (window=50)
    vn = _per_symbol_transform(df, "entry", _vol_norm_helper, window=50)
    rz = _per_symbol_transform(df, "entry", _zscore_helper, window=50)
    parts.append(pd.DataFrame({
        "vol_normalized_atr": np.where(np.isfinite(vn), vn, 0.0),
        "rolling_zscore_returns_50": np.where(np.isfinite(rz), rz, 0.0),
    }, index=df.index))

    # 9. spread/ATR ratio (proxy via sl_pips). Both in pips.
    if "sl_pips" in df.columns:
        # ATR is in price units -> convert to pips. For JPY pairs the pip
        # factor is 1e2, for the rest 1e4. Keep it simple and consistent
        # with the scanner's pip convention.
        pip_factor = np.where(df["symbol"].str.contains("JPY"), 1e2, 1e4)
        atr_pips = df["atr"].to_numpy(dtype=float) * pip_factor
        ratio = spread_atr_ratio(df["sl_pips"].to_numpy(dtype=float), atr_pips)
        parts.append(pd.DataFrame(
            {"spread_atr_ratio": np.where(np.isfinite(ratio), ratio, 0.0)},
            index=df.index,
        ))

    # 10. lag features (use 0.5 as neutral prior)
    lags = pd.DataFrame({
        "lag1_result": _lag_result(df, None),
        "lag1_symbol_result": _lag_result(df, ["symbol"]),
        "lag1_symbol_dir_result": _lag_result(df, ["symbol", "direction"]),
    }, index=df.index)
    parts.append(lags)

    X = pd.concat(parts, axis=1)
    X = X.astype(float)
    # final NaN safeguard (e.g. early-warmup rows of rolling_zscore)
    X = X.fillna(0.0)
    return FeatureSet(X=X, feature_names=list(X.columns))


def _vol_norm_helper(prices: np.ndarray, window: int = 50) -> np.ndarray:
    return vol_normalize(log_returns(prices), window=window)


def _zscore_helper(prices: np.ndarray, window: int = 50) -> np.ndarray:
    return rolling_zscore(log_returns(prices), window=window)


# ---------------------------------------------------------------------------
# Per-fold scaling (fit on train, apply to test)
# ---------------------------------------------------------------------------


def fit_scalers(X_train: pd.DataFrame) -> FoldScalers:
    """Fit winsorize percentiles and robust-scale stats on the training fold.

    Only columns in ``SCALABLE_FEATURES`` are scaled; binary/one-hot
    features are left alone (scaling would just inject noise into a tree).
    """
    s = FoldScalers()
    for col in SCALABLE_FEATURES:
        if col not in X_train.columns:
            continue
        vals = X_train[col].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        finite = np.isfinite(vals)
        if not finite.any():
            continue
        lo = float(np.nanpercentile(vals[finite], 1.0))
        hi = float(np.nanpercentile(vals[finite], 99.0))
        s.winsor[col] = (lo, hi)
        clipped = np.clip(vals[finite], lo, hi)
        median = float(np.median(clipped))
        q75, q25 = np.percentile(clipped, [75, 25])
        iqr = float(q75 - q25)
        s.robust[col] = (median, iqr)
    return s


def apply_scalers(X: pd.DataFrame, scalers: FoldScalers) -> pd.DataFrame:
    """Apply fitted winsorize-then-robust-scale to the matched columns."""
    out = X.copy()
    for col, (lo, hi) in scalers.winsor.items():
        if col not in out.columns:
            continue
        vals = out[col].to_numpy(dtype=float)
        vals = np.clip(vals, lo, hi)
        median, iqr = scalers.robust.get(col, (0.0, 1.0))
        if iqr == 0:
            out[col] = 0.0
        else:
            out[col] = (vals - median) / iqr
    return out
