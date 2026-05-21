"""Triple-barrier labeling (López de Prado, AFML ch.3).

The current snapshot already encodes the outcome with a single barrier
(next-candle close vs entry). For OHLCV historical data we'll implement the
canonical triple-barrier method:

    for each event t with entry price p_t and ATR_t:
        upper = p_t + k_up * ATR_t       (take-profit barrier)
        lower = p_t - k_lo * ATR_t       (stop-loss barrier)
        timeout = t + N bars             (vertical barrier)
        label = +1 if upper hit first
              = -1 if lower hit first
              =  0 if timeout hit first

This module exposes a function that, given an OHLC DataFrame indexed by
timestamp + a list of event timestamps, returns the triple-barrier label.
The snapshot path uses `binary_from_result` as a fallback.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TripleBarrierConfig:
    k_up: float = 1.0
    k_lo: float = 1.0
    n_bars: int = 24  # 2h on M5
    side: int = 1  # +1 long, -1 short — flip barriers accordingly


def triple_barrier_label(
    ohlc: pd.DataFrame,
    event_ts: pd.Timestamp,
    atr: float,
    config: TripleBarrierConfig,
) -> int:
    """Return +1/-1/0 for a single event.

    `ohlc` must have columns: open, high, low, close. Indexed by timestamp,
    sorted ascending. `event_ts` must be present in the index.
    """
    if event_ts not in ohlc.index:
        raise KeyError(f"event_ts {event_ts} not in ohlc index")
    i = ohlc.index.get_loc(event_ts)
    entry = ohlc.iloc[i]["close"]
    window = ohlc.iloc[i + 1 : i + 1 + config.n_bars]
    if window.empty:
        return 0
    if config.side == 1:
        upper = entry + config.k_up * atr
        lower = entry - config.k_lo * atr
    else:
        upper = entry - config.k_up * atr
        lower = entry + config.k_lo * atr
    for _, row in window.iterrows():
        hi, lo = row["high"], row["low"]
        if config.side == 1:
            if hi >= upper:
                return 1
            if lo <= lower:
                return -1
        else:
            if lo <= upper:
                return 1
            if hi >= lower:
                return -1
    return 0


def binary_from_result(result_series: pd.Series) -> np.ndarray:
    """Fallback labeler used on the current snapshot (single-candle outcome)."""
    return (result_series.values == "WIN").astype(int)
