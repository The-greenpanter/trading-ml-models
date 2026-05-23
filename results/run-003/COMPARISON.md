# run-003 vs run-002 — stacked secondary over the XGBoost primary

Same dataset (`_alerts_snapshot.csv`, 1036 rows after weekend filter), same
outer walk-forward (5 expanding folds, `min_train_frac=0.4`), same primary
hyper-parameters (run-002 XGBoost), same primary decision threshold (0.5).

The single variable changed between the two runs is the **addition of a
stacked secondary classifier** that decides bet/no-bet on the
primary-positive rows. Two secondaries were pre-registered and reported
side-by-side: a numpy logistic regression and a tiny XGBoost.

## Aggregate (default secondary threshold = 0.50)

| Metric                | run-002 (primary only) | run-003 logit | run-003 xgb |
|-----------------------|----------------------:|--------------:|------------:|
| Trades taken          | 182                   | 118           | 75          |
| WR                    | **0.544**             | **0.610**     | **0.667**   |
| WR Wilson 95% CI      | [0.471, 0.615]        | [0.520, 0.693]| [0.554, 0.763] |
| precision             | 0.544                 | 0.610         | 0.667       |
| recall                | 0.643                 | 0.468         | 0.325       |
| F1                    | 0.589                 | 0.529         | 0.437       |
| Profit factor (RR=1)  | 1.19                  | 1.57          | 2.00        |
| Sharpe (per-trade x252)| 1.397                | 3.571         | 5.575       |
| Deflated Sharpe       | 0.498                 | **0.882**     | **0.949**   |

Both secondaries push WR above the 60% deployment threshold at the
pre-registered decision threshold (0.50). The Wilson 95% CI for the
logistic secondary [0.520, 0.693] still includes 0.55 but **no longer
includes 0.50** — under standard backtest-statistics framing, this is
the first run where "the strategy has positive expected WR on RR=1
trades" is supportable at the 95% confidence level.

## Confusion matrices

run-002 primary: `[[385, 83], [55, 99]]` (TN=385, FP=83, FN=55, TP=99)
run-003 logit:   `[[422, 46], [82, 72]]` (TN=422, FP=46, FN=82, TP=72)
run-003 xgb:     `[[443, 25], [104, 50]]` (TN=443, FP=25, FN=104, TP=50)

The logistic secondary cuts false positives from 83 → 46 (-45%); the
XGBoost secondary cuts them to 25 (-70%). The price is increased false
negatives (recall drops). This is the precision/recall trade-off that
meta-labeling is designed to control.

## Per-fold breakdown

| fold | n_test | base_rate | run-002 WR | run-003 logit WR | run-003 xgb WR |
|---:|---:|---:|---:|---:|---:|
| 0 | 124 | 0.145 | 0.400 (n=35) | 0.370 (n=27) | 0.000 (n=0)  |
| 1 | 124 | 0.161 | 0.222 (n=18) | 0.273 (n=11) | 0.000 (n=4)  |
| 2 | 124 | 0.468 | 0.712 (n=59) | 0.738 (n=42) | 0.684 (n=38) |
| 3 | 124 | 0.218 | 0.487 (n=39) | 0.750 (n=12) | 0.545 (n=11) |
| 4 | 126 | 0.246 | 0.645 (n=31) | 0.731 (n=26) | 0.818 (n=22) |

## Regime selection: the most important caveat

| regime                                  | n_folds | primary_taken | primary_WR | logit_taken | logit_WR | xgb_taken | xgb_WR |
|---|---:|---:|---:|---:|---:|---:|---:|
| low base rate (folds with base<0.20)    | 2 | 53  | 0.340 | 38 | 0.342 | 4   | 0.000 |
| high base rate (folds with base>=0.20)  | 3 | 129 | 0.628 | 80 | 0.738 | 71  | 0.704 |

**Both secondaries derive their headline edge from the high-base-rate
folds.** The XGBoost secondary essentially refuses to trade in the
low-regime (4 trades over 2 folds, 0% WR), while the logistic secondary
trades through both regimes but without an edge in the low regime
(34.2% WR, indistinguishable from run-002's primary).

What this means in practice
---------------------------
* The run-003 headline numbers are conditioned on the dataset's mix of
  regimes (about 60/40 high/low). If live trading runs into an extended
  low-base-rate regime, the XGBoost secondary will go very quiet (good,
  if low-base-rate truly maps to "don't trade") and the logistic
  secondary will continue to take roughly half the rule-based alerts
  with no precision lift over run-002 (less good).
* Recommendation: deploy both as a **paper-trade pair** alongside the
  rule-based bot. Track which regime live trading is in, and have a
  pre-committed rule for when each secondary's projection becomes
  valid (e.g., logit on weeks with rolling-30-day WIN base rate >= 0.20,
  xgb regardless).

## Top features picked by the logistic secondary

The top-5 features by primary's XGBoost gain (per fold, then assembled
in `summary.json::fold_results[*].top_features_used_by_logistic`) are
dominated by:

  * `lag1_symbol_dir_result` (consistent with run-002)
  * `sym_USDJPY`, `sym_AUDUSD`, `sym_EURUSD` (per-symbol effects)
  * `dir_LONG` and selected hour-of-day features

Plus the engineered `primary_proba` column that connects secondary to
primary. The logistic uses the smaller feature set by design; the
XGBoost secondary has access to all 59 + primary_proba but its tiny
depth=2 / 50-round configuration prevents the over-fit that a deeper
booster would suffer on the 30-110 row training sets per fold.

## Verdict

The pre-registered secondary threshold 0.50 produces:
  * Logistic WR = 0.610, DSR = 0.882, n = 118 — modest but credible lift.
  * XGBoost  WR = 0.667, DSR = 0.949, n = 75  — bigger lift, fewer trades.

Both clear the WR >= 0.60 deployment target on the aggregate. The regime
diagnostic shows the lift is concentrated in high-base-rate folds, which
caps how confident we can be that the edge generalises out-of-sample.

**Recommended next move (Juan Diego):**
1. Promote both secondaries to shadow / paper-trade on the VPS for ~14
   days alongside the rule-based bot, logging decisions vs outcomes.
2. Track live WR per regime bucket (rolling 30-day base rate).
3. Only promote to real-money the secondary whose live regime-conditional
   WR matches the backtest within ±5pp.
4. Schedule run-004 only if the paper-trade dies: re-derive triple-barrier
   labels from Dukascopy ticks and add an HMM regime-state ensemble.
