# run-001 — XGBoost baseline

## Dataset

| | |
|---|---|
| Source CSV | `data\_alerts_snapshot.csv` |
| Rows raw | 1253 |
| Weekend rows removed | 217 |
| Rows clean | 1036 |
| Base WIN rate (clean) | 0.275 |
| Features | 59 |
| Threshold | 0.5 |
| Walk-forward folds | 5 |

## Walk-forward results

| fold | n_train | n_test | base_rate | taken | WR_on_taken | PF | DSR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 414 | 124 | 0.145 | 35 | 0.400 | 0.67 | 0.012 |
| 1 | 538 | 124 | 0.161 | 18 | 0.222 | 0.29 | 0.003 |
| 2 | 662 | 124 | 0.468 | 59 | 0.712 | 2.47 | 0.974 |
| 3 | 786 | 124 | 0.218 | 39 | 0.487 | 0.95 | 0.091 |
| 4 | 910 | 126 | 0.246 | 31 | 0.645 | 1.82 | 0.665 |

## Aggregate

- Trades taken (predicted positive): **182**
- WR on taken: **0.544**  (Wilson 95% CI: [0.471, 0.615])
- Profit factor: **1.19**
- Sharpe (per-trade, annualised x252): **1.397**
- Deflated Sharpe Ratio (PSR-style probability, n_trials = n_folds): **0.498**

### Confusion matrix (cols = predicted, rows = actual)

|              | pred LOSS | pred WIN |
|---           |---:       |---:      |
| actual LOSS  | 385 | 83 |
| actual WIN   | 55 | 99 |

## Conclusion

XGBoost baseline **does not** reach precision >= 0.60 on walk-forward. Next step per ARCHITECTURE.md: triple-barrier labeling on Dukascopy ticks (12-24 months) and an LSTM/GRU experiment.

## Hyperparameters

See `model/params.json`. Conservative defaults: `n_estimators=300, max_depth=4, lr=0.05, subsample=0.8, colsample=0.8`. `scale_pos_weight` set per fold from class balance.

## Features

`vote_ema_dir`, `vote_ema200`, `vote_macd_mom`, `vote_rsi7`, `vote_bb`, `vote_stoch`, `vote_vol`, `score`, `adx_weak`, `adx_moderate`, `adx_strong`, `atr_rel`, `dir_LONG`, `dir_SHORT`, `regime_TRENDING`, `regime_RANGING`, `sym_EURUSD`, `sym_GBPUSD`, `sym_AUDUSD`, `sym_USDJPY`, `sym_USDCAD`, `h_0`, `h_1`, `h_2`, `h_3`, `h_4`, `h_5`, `h_6`, `h_7`, `h_8`, `h_9`, `h_10`, `h_11`, `h_12`, `h_13`, `h_14`, `h_15`, `h_16`, `h_17`, `h_18`, `h_19`, `h_20`, `h_21`, `h_22`, `h_23`, `dow_0`, `dow_1`, `dow_2`, `dow_3`, `dow_4`, `log_returns_lag1`, `log_returns_lag2`, `log_returns_lag5`, `vol_normalized_atr`, `rolling_zscore_returns_50`, `spread_atr_ratio`, `lag1_result`, `lag1_symbol_result`, `lag1_symbol_dir_result`
