"""LSTM model placeholder.

Will be implemented only if the XGBoost baseline cannot reach precision >= 0.60
on walk-forward validation. See ARCHITECTURE.md for the planned topology
(2 stacked LSTM layers, 64-128 units, dropout 0.3-0.5, sequence length
60-100 M5 bars).
"""
from __future__ import annotations


def build_lstm(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(
        "LSTM not implemented yet. Promote to deep learning only after XGBoost "
        "baseline is validated on walk-forward (see results/run-001/REPORT.md)."
    )
