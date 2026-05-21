"""Walk-forward validation (rolling, no shuffle).

For the current ~1k-row snapshot we use simple expanding-window folds.
The proper Purged K-Fold + embargo (AFML ch.7) becomes important once the
labeling horizon stretches over multiple bars; for the next-candle-binary
fallback labeling there is effectively zero overlap, so embargo=0 is fine.

This module returns index splits, leaving fitting to the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class Fold:
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def walk_forward_splits(
    n_samples: int,
    n_folds: int = 5,
    min_train_frac: float = 0.4,
    embargo: int = 0,
) -> Iterator[Fold]:
    """Yield expanding-window folds.

    The first fold trains on the initial `min_train_frac` of the data and
    tests on the next chunk; each subsequent fold expands the training window
    by one chunk. The final chunk is the test set of the last fold.

    `embargo` rows are skipped between train end and test start (set > 0 when
    labels can leak across the boundary).
    """
    if n_folds < 2:
        raise ValueError("need at least 2 folds")
    start_train_end = int(n_samples * min_train_frac)
    remaining = n_samples - start_train_end
    if remaining <= n_folds:
        raise ValueError("not enough samples for the requested folds")
    chunk = remaining // n_folds

    train_end = start_train_end
    for fold_id in range(n_folds):
        test_start = train_end + embargo
        test_end = test_start + chunk if fold_id < n_folds - 1 else n_samples
        if test_start >= n_samples:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(test_idx) == 0:
            break
        yield Fold(fold_id=fold_id, train_idx=train_idx, test_idx=test_idx)
        train_end = test_end
