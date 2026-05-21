"""Meta-labeling skeleton (López de Prado AFML ch.3.6).

Two-stage classifier:
  side  -- predicts the *direction* of the bet (long/short/none).
  size  -- given the side, predicts whether to take the bet at all
           (binary: profitable / not). This is what we wire into the trader.

The current rule-based engine already proposes a side, so on this dataset the
"size" model is the one with edge-finding leverage. Implementation is deferred
to `notebooks/03_xgboost_baseline.ipynb` once the primary model lands.
"""
from __future__ import annotations


class MetaLabeler:  # pragma: no cover - placeholder
    def __init__(self, side_model=None, size_model=None):
        self.side_model = side_model
        self.size_model = size_model

    def predict(self, X):
        raise NotImplementedError("MetaLabeler is a placeholder for run-001")
