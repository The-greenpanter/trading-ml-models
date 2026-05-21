# run-002 vs run-001 — XGBoost with data conditioning

Same dataset (`_alerts_snapshot.csv`, n=1036 after weekend filter), same
walk-forward configuration (5 expanding folds, `min_train_frac=0.4`),
same XGBoost hyper-parameters (`max_depth=4, eta=0.05, n_estimators=300,
subsample=0.8, colsample_bytree=0.8, min_child_weight=5, seed=42,
scale_pos_weight` set per fold), same decision threshold (0.5).

The single variable changed between the two runs is the feature
pipeline:

* run-001: raw ADX, raw ATR, plain hour/dow one-hots, lag-result probes.
  51 features.
* run-002: ADX bucketed (weak/moderate/strong); ATR replaced by
  ATR/entry; new conditioned features `log_returns_lag1/2/5` (per-symbol
  inter-alert log returns), `vol_normalized_atr` and
  `rolling_zscore_returns_50` (trailing window=50, leakage-safe);
  `spread_atr_ratio` proxy via `sl_pips / atr_pips`. The 3 continuous
  ratio features (`atr_rel`, `vol_normalized_atr`, `spread_atr_ratio`)
  are winsorised at 1/99 and robust-scaled with statistics fit **per
  walk-forward fold on the training portion only** (no look-ahead).
  59 features.

## Aggregate

| Metric                        |    run-001 |    run-002 |   delta |
|-------------------------------|-----------:|-----------:|--------:|
| Trades taken (predicted +)    |        214 |        182 |    -32  |
| WR on taken                   |  **0.491** |  **0.544** | **+5.3pp** |
| WR Wilson 95% CI              | [0.424, 0.557] | [0.471, 0.615] | shifted up |
| Profit factor (RR=1)          |       0.96 |       1.19 |  +0.23  |
| Sharpe (per-trade, x252)      |     -0.296 |      1.397 |  +1.69  |
| Deflated Sharpe Ratio         |  **0.072** |  **0.498** | **+0.426** |

Confusion matrices:

run-001: `[[359, 109], [49, 105]]` (TN=359, FP=109, FN=49, TP=105)
run-002: `[[385,  83], [55,  99]]` (TN=385, FP= 83, FN=55, TP= 99)

The model becomes **more selective** (-32 takes) and the false-positive
count drops by 24% (109 → 83). The price is 6 missed wins (TP 105 → 99),
which is the natural precision/recall trade.

## Per-fold breakdown

| fold | n_test | base_rate | run-001 WR | run-002 WR | delta |
|---:|---:|---:|---:|---:|---:|
| 0 | 124 | 0.145 | 0.467 | 0.400 | -0.067 |
| 1 | 124 | 0.161 | 0.242 | 0.222 | -0.020 |
| 2 | 124 | 0.468 | 0.678 | 0.712 | +0.034 |
| 3 | 124 | 0.218 | 0.370 | 0.487 | +0.117 |
| 4 | 126 | 0.246 | 0.605 | 0.645 | +0.040 |

Folds 0 and 1 actually got **worse** under the conditioned features
(-6.7pp and -2.0pp). Those are the two folds where the test base rate
collapsed to 14-16%. The lift is therefore not uniform across regimes:
conditioning pays in folds 2-4 (where the base rate is in the 22-47%
range) and slightly hurts in the bottom regime. Fold 3 is the cleanest
improvement: same base rate as run-001, +11.7pp WR.

Read together with the dominant gain features (lag1_symbol_dir_result
46.96, lag1_symbol_result 26.66), the model is still mostly betting on
"this symbol+direction just won → bet again". Conditioning improved
the surrounding context features but didn't change the dominant signal.
This is exactly the diagnosis for which meta-labeling (AFML §3.6) is
prescribed: a second classifier learns *when to trust* the primary
model's signal as a function of regime / hour / volatility, rather than
trying to invent a new primary signal.

## Top 15 features by gain (run-002 final model)

| rank | gain | feature |
|---:|---:|---|
|  1 | 46.96 | `lag1_symbol_dir_result` |
|  2 | 26.66 | `lag1_symbol_result` |
|  3 | 15.15 | `sym_USDCAD` |
|  4 | 13.57 | `sym_GBPUSD` |
|  5 | 12.90 | `sym_AUDUSD` |
|  6 | 11.84 | `sym_USDJPY` |
|  7 | 11.63 | `h_9` |
|  8 | 11.45 | `sym_EURUSD` |
|  9 | 10.73 | `dow_4` |
| 10 |  9.62 | `h_14` |
| 11 |  8.87 | `dir_LONG` |
| 12 |  8.28 | `spread_atr_ratio` *(new)* |
| 13 |  8.24 | `h_16` |
| 14 |  7.54 | `dow_2` |
| 15 |  6.39 | `atr_rel` *(new)* |

Two of the new conditioned features make it into the top 15 by gain
(`spread_atr_ratio`, `atr_rel`). The autocorrelation lag features
remain the dominant signal — symbol+direction memory is by far the
strongest single predictor.

## Verdict

WR moves from 49.1% to 54.4% — a +5.3pp lift, within the
hypothesised 5-10pp band, but **below the 55% / 60% deployment
threshold**. DSR jumps from 0.072 (effectively noise) to 0.498
(roughly even odds that the true Sharpe is positive). PF crosses
1.0 to 1.19, so on RR=1 trades the strategy is no longer
loss-making in expectation — but the Wilson CI [47.1%, 61.5%]
still includes 50%, so this is not yet a "verified edge".

**Next step (recommendation):** meta-labeling. Train an XGBoost
meta-classifier on the (proba, fold, base-rate) of the
run-002 model to learn when to trade vs skip — AFML chapter 3
calls this exactly the right move when a primary model has
positive precision but insufficient confidence to act
unconditionally. An HMM regime-state ensemble is the heavier-
weight alternative; meta-labeling is cheaper and matches the
data we already have.
