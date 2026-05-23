"""Tests for the stacked-secondary meta-labeler (run-003)."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.meta_labeler import (
    LogisticSecondary,
    XGBoostSecondary,
    load_logistic,
    save_logistic,
    select_top_features_by_gain,
)


@pytest.fixture(scope="module")
def linearly_separable_data():
    """Two-feature linearly separable dataset."""
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # logits = 2 * x1 - x2, deterministic-ish labels
    logits = 2 * x1 - x2
    y = (logits + rng.normal(scale=0.3, size=n) > 0).astype(int)
    X = np.column_stack([x1, x2])
    return X, y


def test_logistic_separates_easy_data(linearly_separable_data):
    X, y = linearly_separable_data
    m = LogisticSecondary(n_iter=3000, learning_rate=0.05, l2=0.01).fit(X, y)
    proba = m.predict_proba(X)[:, 1]
    acc = float(((proba >= 0.5).astype(int) == y).mean())
    assert acc >= 0.85, f"easy data should be learnable, got acc={acc:.3f}"


def test_logistic_predict_proba_shape(linearly_separable_data):
    X, y = linearly_separable_data
    m = LogisticSecondary().fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    # rows sum to 1
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_logistic_sigmoid_numerical_stability():
    # extreme positive and negative logits shouldn't overflow
    big = np.array([1000.0, -1000.0, 0.0])
    s = LogisticSecondary._sigmoid(big)
    assert np.all(np.isfinite(s))
    assert 0.0 <= s.min() and s.max() <= 1.0


def test_logistic_roundtrip_save_load(tmp_path, linearly_separable_data):
    X, y = linearly_separable_data
    m = LogisticSecondary(n_iter=500).fit(X, y)
    save_logistic(m, tmp_path / "logreg.json")
    m2 = load_logistic(tmp_path / "logreg.json")
    # identical predictions to numerical precision
    p1 = m.predict_proba(X)[:, 1]
    p2 = m2.predict_proba(X)[:, 1]
    assert np.allclose(p1, p2, atol=1e-10)


def test_xgboost_secondary_runs(linearly_separable_data):
    X, y = linearly_separable_data
    m = XGBoostSecondary().fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.all(np.isfinite(proba))
    # rows sum ~= 1
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_xgboost_secondary_small_sample():
    """A guard: secondary must train on tiny datasets (~30 rows)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 4))
    y = (X[:, 0] + rng.normal(scale=0.5, size=30) > 0).astype(int)
    m = XGBoostSecondary().fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (30, 2)


def test_select_top_features_by_gain():
    """Train a tiny XGB on a 3-feature dataset where x0 is the only signal,
    then check the top-1 gain feature is x0.
    """
    rng = np.random.default_rng(2)
    n = 300
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x0 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    X = np.column_stack([x0, x1, x2])
    m = XGBoostSecondary().fit(X, y)
    top = select_top_features_by_gain(m.booster, ["x0", "x1", "x2"], k=1)
    assert top == ["x0"]
