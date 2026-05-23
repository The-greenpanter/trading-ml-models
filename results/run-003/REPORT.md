# run-003 — Stacked secondary over run-002 primary

**Honest framing.** The rule-based scanner is the AFML §3.6 *side* model
(it proposes long/short). The run-002 XGBoost is already the AFML *size*
model -- it gates the rule-based side and was the run-001 → run-002
improvement. What we add here is a **stacked secondary** on top of run-002,
not a fresh AFML meta-labeler. The technique is a model-of-models stack:
secondary learns *when to trust* the run-002 primary's positive
predictions.

## Known limitation: no true triple-barrier

AFML §3.4 prescribes triple-barrier labels (TP / SL / vertical timeout) for
the secondary's training set. The snapshot has **no OHLCV history** --
only per-row outcomes (next-candle close vs entry) and entry/SL/TP price
levels. Without bar-level highs/lows we cannot apply real triple-barrier
labelling. We use the snapshot's binary `result` as the secondary's
label. **Run-004 should re-derive labels from Dukascopy ticks (12-24
months)** if this run does not hit the WR >= 60% target.

## Dataset

| | |
|---|---|
| Source CSV | `data\_alerts_snapshot.csv` |
| Rows raw | 1253 |
| Weekend rows removed | 217 |
| Rows clean | 1036 |
| Base WIN rate (clean) | 0.275 |
| Features (primary) | 59 |
| Primary threshold | 0.5 |
| Secondary default threshold | 0.5 |
| Outer walk-forward folds | 5 |
| Inner CV folds (for OOF primary) | 3 |

## Pre-registered configuration

* **Primary**: identical to run-002. XGBoost with `max_depth=4, eta=0.05,
  n_estimators=300, subsample=0.8, colsample=0.8, min_child_weight=5`.
  Decision threshold 0.5.
* **Logistic secondary**: pure-numpy L2 logistic regression on
  `[primary_proba] + top-5` features from the primary's gain ranking.
* **XGBoost secondary**: tiny `max_depth=2, n_estimators=50,
  min_child_weight=10`, fit on the full primary feature set + primary_proba.
* **Secondary headline threshold**: 0.50 (pre-registered, identical to
  run-002 / run-001). A 7-point sweep is reported as a curve, NOT selected
  after the fact.
* **Inner CV**: 3 expanding-window folds inside the outer-train portion
  to generate OOF primary probabilities (the data on which the secondary
  is trained).

## Walk-forward results (per fold)

`n_2nd` = number of rows the secondary saw at training time (rows where
inner-OOF primary predicted positive within the outer-train portion).

| fold | n_train | n_test | n_2nd | base_rate | primary_taken | primary_WR | logit_taken | logit_WR | xgb_taken | xgb_WR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 414 | 124 | 82 | 0.145 | 35 | 0.400 | 27 | 0.370 | 0 | 0.000 |
| 1 | 538 | 124 | 102 | 0.161 | 18 | 0.222 | 11 | 0.273 | 4 | 0.000 |
| 2 | 662 | 124 | 102 | 0.468 | 59 | 0.712 | 42 | 0.738 | 38 | 0.684 |
| 3 | 786 | 124 | 147 | 0.218 | 39 | 0.487 | 12 | 0.750 | 11 | 0.545 |
| 4 | 910 | 126 | 167 | 0.246 | 31 | 0.645 | 26 | 0.731 | 22 | 0.818 |

## Aggregate (default threshold = 0.50)

| metric | primary only | logistic secondary | xgboost secondary |
|---|---:|---:|---:|
| trades taken | 182 | 118 | 75 |
| WR | 0.544 | 0.610 | 0.667 |
| WR 95% CI | [0.471, 0.615] | [0.520, 0.693] | [0.554, 0.763] |
| precision | 0.544 | 0.610 | 0.667 |
| recall | 0.643 | 0.468 | 0.325 |
| F1 | 0.589 | 0.529 | 0.437 |
| profit factor | 1.19 | 1.57 | 2.00 |
| Sharpe (x252) | 1.397 | 3.571 | 5.575 |
| DSR | 0.498 | 0.882 | 0.949 |

### Confusion matrices

**Primary only** (TN, FP / FN, TP):

|              | pred LOSS | pred WIN |
|---           |---:       |---:      |
| actual LOSS  | 385 | 83 |
| actual WIN   | 55 | 99 |

**Logistic secondary @ 0.50**:

|              | pred LOSS | pred WIN |
|---           |---:       |---:      |
| actual LOSS  | 422 | 46 |
| actual WIN   | 82 | 72 |

**XGBoost secondary @ 0.50**:

|              | pred LOSS | pred WIN |
|---           |---:       |---:      |
| actual LOSS  | 443 | 25 |
| actual WIN   | 104 | 50 |

## Comparison vs run-002

run-002 aggregate (reference): WR 0.544 over 182 trades, PF 1.19,
Sharpe 1.397, DSR 0.498, CI95 [0.471, 0.615].

run-003 primary-only column above is the run-002 model retrained from
the same code path -- treat any tiny delta as the result of the inner-CV
inflating the training schedule and stochastic XGBoost. The interesting
columns are the two secondaries.

## Secondary threshold sweep (observational only)

The sweep is reported as a curve. The headline numbers above use the
pre-registered threshold 0.50; the rows below are NOT a model-selection
mechanism -- they exist so you can see the shape of the precision/recall
trade-off without re-running.

### Logistic secondary

| threshold | n_taken | WR | WR 95% CI | precision | recall | F1 | PF | Sharpe | DSR |
|---:|---:|---:|:--|---:|---:|---:|---:|---:|---:|
| 0.45 | 127 | 0.614 | [0.527, 0.694] | 0.614 | 0.506 | 0.555 | 1.59 | 3.709 | 0.913 |
| 0.50 | 118 | 0.610 | [0.520, 0.693] | 0.610 | 0.468 | 0.529 | 1.57 | 3.571 | 0.882 |
| 0.55 | 85 | 0.647 | [0.541, 0.740] | 0.647 | 0.357 | 0.460 | 1.83 | 4.856 | 0.931 |
| 0.60 | 76 | 0.711 | [0.600, 0.800] | 0.711 | 0.351 | 0.470 | 2.45 | 7.320 | 0.990 |
| 0.65 | 62 | 0.774 | [0.656, 0.860] | 0.774 | 0.312 | 0.444 | 3.43 | 10.326 | 0.997 |
| 0.70 | 41 | 0.854 | [0.716, 0.931] | 0.854 | 0.227 | 0.359 | 5.83 | 15.689 | 0.995 |
| 0.75 | 33 | 0.939 | [0.804, 0.983] | 0.939 | 0.201 | 0.332 | 15.50 | 28.787 | 0.984 |

### XGBoost secondary

| threshold | n_taken | WR | WR 95% CI | precision | recall | F1 | PF | Sharpe | DSR |
|---:|---:|---:|:--|---:|---:|---:|---:|---:|---:|
| 0.45 | 137 | 0.555 | [0.471, 0.635] | 0.555 | 0.494 | 0.522 | 1.25 | 1.742 | 0.536 |
| 0.50 | 75 | 0.667 | [0.554, 0.763] | 0.667 | 0.325 | 0.437 | 2.00 | 5.575 | 0.949 |
| 0.55 | 62 | 0.710 | [0.587, 0.808] | 0.710 | 0.286 | 0.407 | 2.44 | 7.274 | 0.976 |
| 0.60 | 32 | 0.656 | [0.483, 0.796] | 0.656 | 0.136 | 0.226 | 1.91 | 5.140 | 0.716 |
| 0.65 | 15 | 0.600 | [0.357, 0.802] | 0.600 | 0.058 | 0.107 | 1.50 | 3.130 | 0.345 |
| 0.70 | 3 | 1.000 | [0.438, 1.000] | 1.000 | 0.019 | 0.038 | inf | 0.000 | 0.000 |
| 0.75 | 2 | 1.000 | [0.342, 1.000] | 1.000 | 0.013 | 0.026 | inf | 0.000 | 0.000 |


## Verdict

**Target hit on the point estimate.** At the pre-registered secondary threshold 0.50, both secondaries' point WR estimates clear 0.60: logit 0.610 on n=118, xgb 0.667 on n=75. However the Wilson 95% CI lower bounds (logit 0.520, xgb 0.554) do not reach 0.60 -- so "WR >= 0.60 with 95% confidence" is NOT yet supported; what IS supported is "point WR >= 0.60 and the CI excludes 0.50". Also read the regime diagnostic below before declaring victory: if most of the edge comes from the high-base-rate folds, live deployment will only behave like the backtest while the underlying WIN regime persists. Recommended next step: shadow-mode paper-trade on the VPS for 2 weeks alongside run-002, then compare live WR / drawdown.

## Diagnostic: is the secondary just doing regime selection?

Folds 0-1 are the low-base-rate regime (14-16% WIN). Folds 2-4 are the
higher-base-rate regime (22-47%). If a secondary's `n_taken` collapses
on folds 0-1 while folds 2-4 carry the headline WR, the "edge" is mostly
regime selection rather than model-of-models insight.

| regime | n_folds | primary_taken | primary_WR | logit_taken | logit_WR | xgb_taken | xgb_WR |
|---|---:|---:|---:|---:|---:|---:|---:|
| low base rate (folds with base<0.20) | 2 | 53 | 0.340 | 38 | 0.342 | 4 | 0.000 |
| high base rate (folds with base>=0.20) | 3 | 129 | 0.628 | 80 | 0.738 | 71 | 0.704 |

## Files

* `model/primary/model.json` -- run-002-equivalent primary XGBoost.
* `model/logistic_secondary.json` -- weights of the LogisticSecondary.
* `model/xgboost_secondary.json` -- the tiny XGBoost secondary booster.
* `model/secondary_meta.json` -- pre-registered config + top-k feature
  list used by the logistic secondary.
* `summary.json` -- full metrics (per-fold + aggregate + sweep).
