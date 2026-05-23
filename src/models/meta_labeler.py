"""Meta-labeling / stacked-secondary classifier.

López de Prado, AFML chapter 3.6 describes meta-labeling as a two-stage
classifier:

    side  -- predicts the *direction* of a bet (long/short/none).
    size  -- given the side, predicts whether to *take* the bet at all
             (binary: profitable / not).

Honest framing for this repo (see ``results/run-003/REPORT.md``)
================================================================
The rule-based scanner is the *side* model: it proposes long/short on every
alert. The run-002 XGBoost was already the *size* model in AFML's sense:
it gates the rule-based side. What we build here is a **stacked secondary**
on top of the run-002 primary -- not a fresh AFML meta-labeler. The
nomenclature "meta-labeling" is preserved because Juan Diego's task brief
uses it, but the technique is a model-of-models stack: secondary learns
when to trust the run-002 primary's positive predictions.

Triple-barrier note
-------------------
AFML §3.4 prescribes triple-barrier labels (TP / SL / vertical timeout) for
the secondary's training labels. Our snapshot has *no OHLCV history* -- only
a per-row outcome (next-candle close vs entry) and the entry/SL/TP price
levels. Without bar-level highs/lows we cannot run a real triple-barrier.
We therefore use the single-bar binary ``result`` as the secondary's label
and document this as the dominant limitation of run-003. Run-004 should
re-derive labels from Dukascopy ticks (12-24 months).

Models implemented
------------------
Two pre-registered candidates for the secondary, both fit only on rows where
the primary predicted positive:

  * ``LogisticSecondary`` -- pure numpy IRLS / gradient-step logistic
    regression on a small set of features. Cannot overfit ~30-110 training
    rows the way a deep boosted tree would.
  * ``XGBoostSecondary`` -- intentionally tiny XGBoost (``max_depth=2,
    n_estimators=50, min_child_weight=10``). Pre-registered so the choice
    of architecture is not post-hoc.

The two are reported side-by-side in REPORT.md; the user picks. No
threshold sweep is used to *select* a model -- thresholds are reported as a
full curve.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xgboost as xgb


# ---------------------------------------------------------------------------
# Pure-numpy logistic regression (zero extra dependencies)
# ---------------------------------------------------------------------------


@dataclass
class LogisticSecondary:
    """L2-regularised logistic regression fit by gradient descent.

    Pure numpy. Small enough to be safe on ~30-110 training rows.
    """

    learning_rate: float = 0.05
    n_iter: int = 2000
    l2: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # numerically stable sigmoid
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def _standardise_fit(self, X: np.ndarray) -> np.ndarray:
        self._mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        std[std == 0] = 1.0
        self._std = std
        return (X - self._mean) / self._std

    def _standardise_apply(self, X: np.ndarray) -> np.ndarray:
        if self._mean is None or self._std is None:
            raise RuntimeError("model not fitted")
        return (X - self._mean) / self._std

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "LogisticSecondary":
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n, p = X.shape
        Xs = self._standardise_fit(X)
        if sample_weight is None:
            sample_weight = np.ones(n, dtype=float)
        sw = sample_weight / sample_weight.sum() * n  # normalise mean=1
        w = rng.normal(0.0, 0.01, size=p)
        b = 0.0
        for _ in range(self.n_iter):
            z = Xs @ w + b
            p_hat = self._sigmoid(z)
            err = (p_hat - y) * sw
            grad_w = Xs.T @ err / n + self.l2 * w / n
            grad_b = float(err.sum() / n)
            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b
        self.coef_ = w
        self.intercept_ = b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model not fitted")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Xs = self._standardise_apply(X)
        p1 = self._sigmoid(Xs @ self.coef_ + self.intercept_)
        return np.stack([1 - p1, p1], axis=1)


# ---------------------------------------------------------------------------
# Tiny XGBoost secondary -- pre-registered hyperparameters
# ---------------------------------------------------------------------------


XGB_SECONDARY_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 2,
    "eta": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 2.0,
    "min_child_weight": 10,
    "seed": 42,
}
XGB_SECONDARY_NUM_ROUND = 50


@dataclass
class XGBoostSecondary:
    """Intentionally small XGBoost secondary.

    Designed to *not* overfit on the ~30-110 row training sets that arise
    from filtering to rows where the primary predicted positive.
    """

    booster: xgb.Booster | None = None
    params: dict[str, Any] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "XGBoostSecondary":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        p = dict(XGB_SECONDARY_PARAMS)
        pos = float(np.sum(y == 1))
        neg = float(np.sum(y == 0))
        if pos > 0 and neg > 0:
            p.setdefault("scale_pos_weight", neg / pos)
        dtrain = xgb.DMatrix(X, label=y, weight=sample_weight)
        booster = xgb.train(
            p,
            dtrain,
            num_boost_round=XGB_SECONDARY_NUM_ROUND,
            verbose_eval=False,
        )
        self.booster = booster
        self.params = p
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("model not fitted")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        p1 = np.asarray(self.booster.predict(xgb.DMatrix(X))).reshape(-1)
        return np.stack([1 - p1, p1], axis=1)


# ---------------------------------------------------------------------------
# Feature selection helper for the logistic secondary
# ---------------------------------------------------------------------------


def select_top_features_by_gain(
    booster: xgb.Booster,
    feature_names: Sequence[str],
    k: int = 5,
) -> list[str]:
    """Return the top-k feature names by XGBoost gain.

    XGBoost returns gain keyed by f0, f1, ... when fed a numpy array. We map
    back to the user-provided feature_names by index. Missing features
    (those XGBoost never split on) are silently dropped.
    """
    gain = booster.get_score(importance_type="gain")
    # gain keys look like 'f0', 'f1', ...
    parsed: list[tuple[float, str]] = []
    for key, val in gain.items():
        if not key.startswith("f"):
            continue
        try:
            idx = int(key[1:])
        except ValueError:
            continue
        if 0 <= idx < len(feature_names):
            parsed.append((float(val), feature_names[idx]))
    parsed.sort(reverse=True)
    return [name for _, name in parsed[:k]]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_logistic(model: LogisticSecondary, path: str | Path) -> None:
    """Persist a LogisticSecondary as a small JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if model.coef_ is None or model._mean is None or model._std is None:
        raise RuntimeError("logistic not fitted")
    payload = {
        "coef_": model.coef_.tolist(),
        "intercept_": float(model.intercept_),
        "mean_": model._mean.tolist(),
        "std_": model._std.tolist(),
        "learning_rate": model.learning_rate,
        "n_iter": model.n_iter,
        "l2": model.l2,
        "seed": model.seed,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_logistic(path: str | Path) -> LogisticSecondary:
    with open(path) as f:
        payload = json.load(f)
    m = LogisticSecondary(
        learning_rate=payload["learning_rate"],
        n_iter=payload["n_iter"],
        l2=payload["l2"],
        seed=payload["seed"],
    )
    m.coef_ = np.asarray(payload["coef_"], dtype=float)
    m.intercept_ = float(payload["intercept_"])
    m._mean = np.asarray(payload["mean_"], dtype=float)
    m._std = np.asarray(payload["std_"], dtype=float)
    return m
