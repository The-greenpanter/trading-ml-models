"""XGBoost baseline classifier.

Uses the native xgboost Booster API (xgb.train) so the package works without
scikit-learn installed. A thin wrapper exposes `.fit/.predict_proba` for
convenience.

Conservative hyperparameters chosen for a small (~1k) imbalanced dataset.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb


DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 4,
    "eta": 0.05,                # learning rate
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
    "seed": 42,
}
DEFAULT_NUM_BOOST_ROUND = 300
DEFAULT_EARLY_STOPPING = 30


class XGBBoosterModel:
    """Thin wrapper around xgb.Booster with a sklearn-ish surface."""

    def __init__(self, booster: xgb.Booster, params: dict, best_iteration: int | None = None):
        self.booster = booster
        self.params = params
        self.best_iteration = best_iteration

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dmat = xgb.DMatrix(X)
        if self.best_iteration is not None:
            p1 = self.booster.predict(dmat, iteration_range=(0, self.best_iteration + 1))
        else:
            p1 = self.booster.predict(dmat)
        p1 = np.asarray(p1).reshape(-1)
        return np.stack([1 - p1, p1], axis=1)

    def save_model(self, path: str | Path) -> None:
        self.booster.save_model(str(path))


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    params: dict | None = None,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
) -> XGBBoosterModel:
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    if pos > 0 and neg > 0:
        p.setdefault("scale_pos_weight", neg / pos)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    evals: list[tuple[xgb.DMatrix, str]] = [(dtrain, "train")]
    early = None
    if X_val is not None and y_val is not None and len(X_val) > 0:
        dval = xgb.DMatrix(X_val, label=y_val)
        evals.append((dval, "val"))
        early = DEFAULT_EARLY_STOPPING

    booster = xgb.train(
        p,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early,
        verbose_eval=False,
    )
    best_it = getattr(booster, "best_iteration", None)
    return XGBBoosterModel(booster=booster, params=p, best_iteration=best_it)


def save_model(model: XGBBoosterModel, out_dir: str | Path, feature_names: list[str]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(out / "model.json")
    with open(out / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    params = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
              for k, v in model.params.items()}
    params["best_iteration"] = model.best_iteration
    with open(out / "params.json", "w") as f:
        json.dump(params, f, indent=2)
