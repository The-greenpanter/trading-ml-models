"""Trading + statistical metrics.

WR, profit factor, Sharpe, Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class TradeMetrics:
    n_trades: int
    win_rate: float
    profit_factor: float
    sharpe: float
    dsr: float


def win_rate(y_true: np.ndarray, y_pred: np.ndarray | None = None) -> float:
    """If y_pred is given, compute precision (WR on predicted positives).
    Otherwise compute base rate of y_true.
    """
    if y_pred is None:
        return float(np.mean(y_true == 1))
    mask = y_pred == 1
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(y_true[mask] == 1))


def profit_factor(returns: np.ndarray) -> float:
    """sum(positive returns) / |sum(negative returns)|."""
    pos = returns[returns > 0].sum()
    neg = -returns[returns < 0].sum()
    if neg <= 0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / neg)


def sharpe_ratio(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    """Annualised Sharpe assuming i.i.d. returns. Risk-free rate = 0."""
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return 0.0
    mu = returns.mean()
    sd = returns.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(mu / sd * math.sqrt(periods_per_year))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int = 1,
    periods_per_year: float = 252.0,
) -> float:
    """Probabilistic Sharpe Ratio deflated by the number of trials.

    Bailey & López de Prado (2014), eq. 8-10. Returns a probability in [0,1]
    that the *true* Sharpe is above zero, accounting for skew, kurtosis,
    sample size and the multiplicity of strategy trials tested.

    Reference: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
    """
    returns = np.asarray(returns, dtype=float)
    n = returns.size
    if n < 4:
        return 0.0
    mu = returns.mean()
    sd = returns.std(ddof=1)
    if sd == 0:
        return 0.0
    sr = mu / sd  # non-annualised, per-period Sharpe
    # higher moments
    m3 = ((returns - mu) ** 3).mean() / sd ** 3
    m4 = ((returns - mu) ** 4).mean() / sd ** 4
    # expected maximum SR under N independent strategies (Bailey 2014 eq. 6)
    if n_trials <= 1:
        sr0 = 0.0
    else:
        gamma = 0.5772156649  # Euler-Mascheroni
        e_max = (1 - gamma) * norm.ppf(1 - 1.0 / n_trials) + gamma * norm.ppf(
            1 - 1.0 / (n_trials * math.e)
        )
        sr0 = e_max / math.sqrt(n)  # under H0 sigma(SR_hat) = 1/sqrt(n)
    # PSR / DSR (probabilistic)
    denom = math.sqrt(max(1e-12, 1 - m3 * sr + (m4 - 1) / 4 * sr ** 2))
    z = (sr - sr0) * math.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def trade_returns_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rr: float = 1.0,
) -> np.ndarray:
    """Translate (label, prediction) pairs into per-trade returns.

    Only predicted-positive trades are 'taken'. WIN -> +rr, LOSS -> -1.
    """
    mask = y_pred == 1
    out = np.zeros(mask.sum(), dtype=float)
    if mask.sum() == 0:
        return out
    out[y_true[mask] == 1] = rr
    out[y_true[mask] == 0] = -1.0
    return out


def summarise(y_true: np.ndarray, y_pred: np.ndarray, rr: float = 1.0, n_trials: int = 1) -> TradeMetrics:
    rets = trade_returns_from_predictions(y_true, y_pred, rr=rr)
    return TradeMetrics(
        n_trades=int((y_pred == 1).sum()),
        win_rate=win_rate(y_true, y_pred),
        profit_factor=profit_factor(rets),
        sharpe=sharpe_ratio(rets),
        dsr=deflated_sharpe_ratio(rets, n_trials=n_trials),
    )
