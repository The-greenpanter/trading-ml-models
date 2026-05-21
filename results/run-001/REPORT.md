# run-001 — XGBoost baseline

## Dataset

| | |
|---|---|
| Source CSV | `data\_alerts_snapshot.csv` |
| Rows raw | 1253 |
| Weekend rows removed | 217 |
| Rows clean | 1036 |
| Base WIN rate (clean) | 0.275 |
| Features | 51 |
| Threshold | 0.5 |
| Walk-forward folds | 5 |

## Walk-forward results

| fold | n_train | n_test | base_rate | taken | WR_on_taken | PF | DSR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 414 | 124 | 0.145 | 30 | 0.467 | 0.88 | 0.064 |
| 1 | 538 | 124 | 0.161 | 33 | 0.242 | 0.32 | 0.000 |
| 2 | 662 | 124 | 0.468 | 59 | 0.678 | 2.11 | 0.931 |
| 3 | 786 | 124 | 0.218 | 54 | 0.370 | 0.59 | 0.002 |
| 4 | 910 | 126 | 0.246 | 38 | 0.605 | 1.53 | 0.544 |

## Aggregate

- Trades taken (predicted positive): **214**
- WR on taken: **0.491**  (Wilson 95% CI: [0.424, 0.557])
- Profit factor: **0.96**
- Sharpe (per-trade, annualised x252): **-0.296**
- Deflated Sharpe Ratio (PSR-style probability, n_trials = n_folds): **0.072**

### Confusion matrix (cols = predicted, rows = actual)

|              | pred LOSS | pred WIN |
|---           |---:       |---:      |
| actual LOSS  | 359 | 109 |
| actual WIN   | 49 | 105 |

## Conclusion

XGBoost baseline **does not** reach precision >= 0.60 on walk-forward. Next step per ARCHITECTURE.md: triple-barrier labeling on Dukascopy ticks (12-24 months) and an LSTM/GRU experiment.

## Hyperparameters

See `model/params.json`. Conservative defaults: `n_estimators=300, max_depth=4, lr=0.05, subsample=0.8, colsample=0.8`. `scale_pos_weight` set per fold from class balance.

## Features

`vote_ema_dir`, `vote_ema200`, `vote_macd_mom`, `vote_rsi7`, `vote_bb`, `vote_stoch`, `vote_vol`, `score`, `adx`, `atr`, `dir_LONG`, `dir_SHORT`, `regime_TRENDING`, `regime_RANGING`, `sym_EURUSD`, `sym_GBPUSD`, `sym_AUDUSD`, `sym_USDJPY`, `sym_USDCAD`, `h_0`, `h_1`, `h_2`, `h_3`, `h_4`, `h_5`, `h_6`, `h_7`, `h_8`, `h_9`, `h_10`, `h_11`, `h_12`, `h_13`, `h_14`, `h_15`, `h_16`, `h_17`, `h_18`, `h_19`, `h_20`, `h_21`, `h_22`, `h_23`, `dow_0`, `dow_1`, `dow_2`, `dow_3`, `dow_4`, `lag1_result`, `lag1_symbol_result`, `lag1_symbol_dir_result`
