from __future__ import annotations

import numpy as np

from src.validation.metrics import (
    win_rate,
    profit_factor,
    sharpe_ratio,
    deflated_sharpe_ratio,
    trade_returns_from_predictions,
)


def test_win_rate_precision():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 1, 1, 0, 0])
    # taken = idx 0,1,2 -> WIN, LOSS, WIN -> 2/3
    assert abs(win_rate(y_true, y_pred) - 2 / 3) < 1e-9


def test_profit_factor():
    rets = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    assert abs(profit_factor(rets) - 3 / 2) < 1e-9


def test_sharpe_zero_std():
    assert sharpe_ratio(np.array([0.5, 0.5, 0.5])) == 0.0


def test_dsr_in_unit_interval():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.01, 1.0, 200)
    p = deflated_sharpe_ratio(rets, n_trials=5)
    assert 0.0 <= p <= 1.0


def test_trade_returns_filters_negatives():
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 1, 0])
    rets = trade_returns_from_predictions(y_true, y_pred, rr=2.0)
    assert list(rets) == [2.0, -1.0]
