"""Data conditioning utilities for finance ML.

Pure numpy/pandas helpers that turn raw, scale-heterogeneous financial
features into well-behaved inputs for tree boosters and linear models.

Design notes (see docs/DATA_CONDITIONING.md):

* Each function is a pure array utility. Stateful "fit/transform" wrappers
  belong in the feature pipeline that calls them — that pipeline is
  responsible for fitting global statistics (percentiles, median/IQR,
  fractional-diff weights) on the *training fold only* and applying them to
  the test fold, to avoid look-ahead leakage.
* Rolling / trailing transforms (vol_normalize, rolling_zscore,
  fractional_diff) only use past data at each row, so they are safe to
  compute on the full series — but ``log_returns`` between consecutive
  alerts is only meaningful when the array is already sorted by timestamp
  *within a single symbol*. The caller is responsible for grouping.
* The conditioning playbook is derived from López de Prado, *Advances in
  Financial Machine Learning* (AFML, 2018), chapters 3 (labelling) and 5
  (fractional differentiation), and from Mantegna & Stanley, *Hierarchical
  structure in financial markets*, Eur. Phys. J. B 11, 193 (1999), which
  motivates working with log-returns (or fractionally-differentiated
  prices) rather than raw prices.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Stationarity / returns
# ---------------------------------------------------------------------------


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Log returns ``log(P_t / P_{t-1})`` with a leading NaN.

    The first row has no predecessor; we return NaN there so the caller can
    decide how to impute (a neutral zero is usually fine for tree models).
    """
    prices = np.asarray(prices, dtype=float)
    out = np.full_like(prices, np.nan, dtype=float)
    if prices.size < 2:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = prices[1:] / prices[:-1]
        out[1:] = np.log(ratio)
    out[~np.isfinite(out)] = np.nan
    return out


def winsorize(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Clip values to the ``[p_low, p_high]`` percentile band.

    Defaults to 1/99 — a standard fat-tail mitigation that retains rank
    information while preventing a handful of extreme observations from
    dominating a linear fit or a tree split.

    NaNs are preserved (``np.nanpercentile``).
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    finite = np.isfinite(x)
    if not finite.any():
        return x.copy()
    lo = float(np.nanpercentile(x[finite], p_low))
    hi = float(np.nanpercentile(x[finite], p_high))
    out = x.copy()
    out[finite] = np.clip(x[finite], lo, hi)
    return out


def vol_normalize(returns: np.ndarray, window: int = 50) -> np.ndarray:
    """Divide returns by their trailing rolling standard deviation.

    Renders returns comparable across volatility regimes: a +1% move in a
    quiet regime carries more information than a +1% move during a vol
    spike. Implemented via ``pd.Series.rolling`` with ``min_periods=window``
    so early rows return NaN.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return returns
    s = pd.Series(returns)
    sd = s.rolling(window=window, min_periods=window).std(ddof=1)
    out = np.array((s / sd).to_numpy(), dtype=float, copy=True)
    out[~np.isfinite(out)] = np.nan
    return out


def rolling_zscore(x: np.ndarray, window: int = 100) -> np.ndarray:
    """Trailing rolling z-score: ``(x - mean_w) / std_w``.

    Adapts to slow regime drift by re-centering on the local mean. Standard
    on macro and trend-following features (e.g. distance to a moving
    average) per AFML §3.6.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    s = pd.Series(x)
    mu = s.rolling(window=window, min_periods=window).mean()
    sd = s.rolling(window=window, min_periods=window).std(ddof=1)
    out = np.array(((s - mu) / sd).to_numpy(), dtype=float, copy=True)
    out[~np.isfinite(out)] = np.nan
    return out


# ---------------------------------------------------------------------------
# Fractional differentiation (López de Prado, AFML ch. 5)
# ---------------------------------------------------------------------------


def _ffd_weights(d: float, thres: float) -> np.ndarray:
    """Fixed-Width Fractional Differentiation weights (AFML eq. 5.8).

    Generates the binomial expansion of (1-L)^d truncated at the first
    weight whose absolute value falls below ``thres``.
    """
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
        if k > 10000:  # safety guard
            break
    return np.array(w[::-1])  # reverse so we convolve naturally


def fractional_diff(x: np.ndarray, d: float = 0.4, thres: float = 1e-2) -> np.ndarray:
    """Fixed-Width Fractional Differentiation (FFD).

    López de Prado AFML chapter 5: differentiate a series by a real order
    ``d in (0, 1)`` to obtain a (typically) stationary version while
    retaining maximum memory. ``d=0`` is the identity, ``d=1`` is the first
    difference. ``d=0.4`` is the canonical first try.

    Implementation: convolve with the truncated binomial-expansion weights.
    The first ``len(weights)-1`` rows are NaN (insufficient history).
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    w = _ffd_weights(d, thres)
    width = len(w)
    out = np.full_like(x, np.nan, dtype=float)
    if x.size < width:
        return out
    for i in range(width - 1, x.size):
        window = x[i - width + 1 : i + 1]
        if not np.all(np.isfinite(window)):
            continue
        out[i] = float(np.dot(w, window))
    return out


# ---------------------------------------------------------------------------
# Categorical / one-hot encoders
# ---------------------------------------------------------------------------


# Wilder's classic ADX thresholds, used industry-wide.
ADX_WEAK = 20.0
ADX_STRONG = 40.0


def categorize_adx(adx: np.ndarray) -> np.ndarray:
    """Bin ADX into 3 columns of one-hot: weak (<20), moderate (20-40), strong (>=40).

    Returns an ``(n, 3)`` int array with columns ``[weak, moderate, strong]``.
    NaNs map to all-zero rows (the model can still split on the indicator).
    """
    adx = np.asarray(adx, dtype=float)
    weak = (adx < ADX_WEAK).astype(int)
    strong = (adx >= ADX_STRONG).astype(int)
    moderate = ((adx >= ADX_WEAK) & (adx < ADX_STRONG)).astype(int)
    nan_mask = ~np.isfinite(adx)
    weak[nan_mask] = 0
    moderate[nan_mask] = 0
    strong[nan_mask] = 0
    return np.stack([weak, moderate, strong], axis=1)


def hour_one_hot(timestamps: pd.Series) -> pd.DataFrame:
    """Hour-of-day (UTC) -> 24 one-hot columns ``h_0`` ... ``h_23``.

    Trees can split on ordinal hours, but a one-hot encoding (i) lets the
    model capture non-monotonic intraday patterns (London open, NY close)
    in a single split per category, and (ii) makes feature importance
    interpretable per hour.
    """
    ts = pd.to_datetime(timestamps, utc=True)
    hours = ts.dt.hour
    out = pd.DataFrame(
        {f"h_{h}": (hours == h).astype(int).values for h in range(24)},
        index=timestamps.index,
    )
    return out


def dow_one_hot(timestamps: pd.Series) -> pd.DataFrame:
    """Day-of-week (Mon=0 ... Fri=4) -> 5 one-hot columns. Excludes weekend.

    Caller must have weekend-filtered the data; Sat/Sun rows would yield
    all-zero rows here.
    """
    ts = pd.to_datetime(timestamps, utc=True)
    dow = ts.dt.dayofweek
    out = pd.DataFrame(
        {f"dow_{d}": (dow == d).astype(int).values for d in range(5)},
        index=timestamps.index,
    )
    return out


# ---------------------------------------------------------------------------
# Scale-invariant ratios + robust scaling
# ---------------------------------------------------------------------------


def spread_atr_ratio(spread_pips: np.ndarray, atr: np.ndarray) -> np.ndarray:
    """Scale-invariant proxy for execution friction: ``spread / ATR``.

    Both must be in commensurate units (pips, or both in price units).
    Divides through the symbol's volatility scale so EURUSD and USDJPY rows
    live on the same axis.
    """
    spread_pips = np.asarray(spread_pips, dtype=float)
    atr = np.asarray(atr, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.array(spread_pips / atr, dtype=float, copy=True)
    out[~np.isfinite(out)] = np.nan
    return out


def robust_scale(
    x: np.ndarray,
    median: Optional[float] = None,
    iqr: Optional[float] = None,
) -> np.ndarray:
    """``(x - median) / IQR`` — robust to outliers vs StandardScaler.

    If ``median`` and ``iqr`` are passed, they are used directly (the
    "transform" path; fit them on the training fold of a walk-forward
    split and pass them in when scaling the test fold). Otherwise both are
    estimated from ``x`` (the "fit_transform" convenience path).
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    if median is None:
        median = float(np.nanmedian(x))
    if iqr is None:
        q75, q25 = np.nanpercentile(x, [75, 25])
        iqr = float(q75 - q25)
    if iqr == 0:
        return np.zeros_like(x)
    return (x - median) / iqr
