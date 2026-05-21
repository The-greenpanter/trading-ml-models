# Data conditioning playbook

> "The vast majority of papers in the finance ML literature fail because
> they feed non-stationary, scale-heterogeneous, fat-tailed series into
> ML algorithms designed for the opposite."
> — Marcos López de Prado, *Advances in Financial Machine Learning*,
> 2018, p. 75.

This document records *why* each transform in
`src/data/conditioning.py` exists, what assumption it relaxes, and how
it is applied per walk-forward fold to avoid look-ahead leakage.

## Why conditioning matters here

Run-001 of the XGBoost baseline fed the model raw `ADX`, raw `ATR`,
ordinal hour, and a handful of vote booleans. It reached only 49.1%
precision on taken trades (DSR 0.072, profit factor 0.96) — i.e. no
better than the unconditional base rate, and with a Sharpe of -0.30.

The hypothesis behind run-002 was that the **raw inputs violate the
canonical preconditions of supervised learning on finance data**:

1. Non-stationarity (raw price, raw ATR).
2. Scale heterogeneity across symbols (EURUSD pips ≠ USDJPY pips,
   AUDUSD volatility ≠ GBPUSD volatility).
3. Fat tails / outliers that pull the median statistic of a split.
4. Ordinal encoding of categorical-by-meaning features (e.g. ADX
   regime; 25 isn't "1.25× more trendy" than 20 — it's a different
   regime).

The conditioning playbook below is a translation of the
recommendations in AFML chapters 3 (labelling and weights) and 5
(fractional differentiation), plus Mantegna & Stanley (1999) on the
necessity of log-returns / fractionally-differentiated prices when
computing cross-asset correlation distances.

## Transforms

### `log_returns(prices)`

The first and simplest stationarity step:
`r_t = log(P_t / P_{t-1})`.

* Raw `P_t` is unit-root non-stationary. A model trained on prices
  from one period cannot generalise to a different price regime.
* Log returns are approximately stationary and approximately
  symmetric. They also have the additive-over-time property
  `r_{t→t+k} = sum r_{t+i}`, which simplifies aggregation.
* In this codebase we apply log returns **per symbol**, after sorting
  by timestamp. The snapshot CSV is per-alert (irregular), so these
  are *inter-alert* log-price diffs — a noisy proxy for bar returns
  but still a useful feature for tree splits.

References:
- AFML chapter 3, §3.1 (labelling rationale uses log-returns).
- Mantegna & Stanley, *Hierarchical structure in financial markets*,
  Eur. Phys. J. B **11**, 193 (1999),
  <https://link.springer.com/article/10.1007/s100510050929>.

### `winsorize(x, p_low=1, p_high=99)`

Clips values to the empirical 1st / 99th percentile band.

* Finance returns have heavier tails than the Gaussian — the
  empirical kurtosis of daily FX returns is routinely > 6 (López de
  Prado, AFML §3.3).
* Untreated, a handful of 4-sigma days dominate the L2-style
  objectives used by regularised models and pull tree-split
  thresholds.
* Winsorising preserves rank but bounds influence. We do **not**
  trim-drop, because the rest of the row's features (votes, score,
  hour) are still informative even when the ATR was extreme.

### `vol_normalize(returns, window=50)`

Divides each return by its trailing rolling standard deviation.

* A +1% move in a quiet regime is not the same news as a +1% move
  during a vol spike. A model that conflates the two will mis-learn
  the conditional distribution of the next outcome.
* AFML §3.4 calls this *volatility-adjusted returns* and uses it as
  the natural unit for the triple-barrier labelling threshold.

### `rolling_zscore(x, window=100)`

Trailing rolling z-score: `(x - mean_w) / std_w`.

* Adapts to slow regime drift by re-centering on the local mean.
* Useful for features that have an absolute interpretation but a
  drifting baseline (e.g. RSI-7 averages 50 in trendless markets but
  can sit at 70 for weeks in a strong trend).

### `fractional_diff(x, d=0.4, thres=1e-2)`

López de Prado's flagship contribution (AFML chapter 5): a
real-order differentiation of a series, `(1-L)^d`, that produces a
(typically) stationary series while retaining **maximum memory** of
the original — unlike the first difference (`d=1`), which destroys
all level information.

* `d ∈ (0, 1)` is the canonical interval. `d=0.4` is the textbook
  first try.
* Implemented via the Fixed-Width Fractional Differentiation (FFD)
  estimator, AFML §5.4: convolve with the truncated binomial
  expansion weights, stop adding weights once `|w_k| < thres`.
* Not currently materialised as a feature (the snapshot has no long
  contiguous price series), but the utility is available for the
  next iteration that will use Dukascopy minute bars.

### `categorize_adx(adx)` — Wilder thresholds

ADX is a regime indicator, not a continuous predictor. Wilder's
original thresholds (J. Welles Wilder, *New Concepts in Technical
Trading Systems*, 1978) are:

| Range  | Regime    |
|--------|-----------|
| < 20   | Weak      |
| 20-40  | Moderate  |
| ≥ 40   | Strong    |

We bucket and one-hot rather than feeding the raw value, because the
relationship between ADX and forward returns is non-monotonic — a
single split on `adx > 22.5` cannot capture the "weak vs moderate vs
strong" structure.

### `hour_one_hot(timestamps)` / `dow_one_hot(timestamps)`

24 columns for hour-of-day (UTC); 5 columns for day-of-week
(Mon-Fri only — Sat/Sun are filtered upstream by `filter_weekend`).

Trees *can* split on ordinal hours, but they need two splits
(`h ≥ 8` AND `h ≤ 17`) to isolate the London session. One-hot
encoding makes intraday seasonality reachable in a single split per
category and makes feature importance per hour interpretable.

### `spread_atr_ratio(spread_pips, atr)`

Execution friction in volatility units — scale-invariant across
symbols. In the snapshot we have no broker `spread` column, so the
production proxy is `sl_pips / atr_pips`: a wider SL relative to
ATR signals either a more demanding setup or a thinner liquidity
environment.

### `robust_scale(x)`

`(x - median) / IQR`. Robust replacement for `StandardScaler`
(`(x - mean) / std`), which is itself dominated by the fat tails
we just argued against in `winsorize`. The two are usually composed:
winsorise first, then `robust_scale` on the clipped values.

## Per-fold fitting (anti-leakage)

The conditioning utilities are split by leakage profile:

| Family                    | Functions                                     | Leakage-safe on full series? |
|---------------------------|-----------------------------------------------|------------------------------|
| Trailing (uses past only) | `log_returns`, `vol_normalize`, `rolling_zscore`, `fractional_diff` | Yes |
| Per-row (no past needed)  | `categorize_adx`, `hour_one_hot`, `dow_one_hot`, `spread_atr_ratio` | Yes |
| Global statistic          | `winsorize`, `robust_scale`                    | **No** — fit per fold |

`src/features/engineering.py` exposes `fit_scalers(X_train)` and
`apply_scalers(X, scalers)`. `src/train.py` calls them inside the
walk-forward loop, fitting on `train_idx` only and applying to
`test_idx`. The 3 currently-scaled features are `atr_rel`,
`spread_atr_ratio`, `vol_normalized_atr`.

## Feature-by-feature delta (run-001 → run-002)

| Original (run-001) | Conditioned (run-002)                                                              | Rationale                                              |
|--------------------|-------------------------------------------------------------------------------------|--------------------------------------------------------|
| `adx` (raw float)  | `adx_weak`, `adx_moderate`, `adx_strong` (one-hot)                                  | Non-monotonic regime indicator                         |
| `atr` (raw float)  | `atr_rel = atr / entry`, winsorised + robust-scaled per fold                        | Scale-invariant across symbols                          |
| —                  | `log_returns_lag1`, `log_returns_lag2`, `log_returns_lag5` (per symbol)             | Stationary, captures recent micro-momentum             |
| —                  | `vol_normalized_atr` (window=50)                                                    | Comparable across vol regimes                          |
| —                  | `rolling_zscore_returns_50`                                                         | Regime-shift aware return signal                       |
| —                  | `spread_atr_ratio = sl_pips / atr_pips`, winsorised + robust-scaled per fold        | Execution-friction in volatility units                 |
| `h_0`…`h_23`       | `h_0`…`h_23` (unchanged)                                                            | Already correct: non-ordinal one-hot                   |
| `dow_0`…`dow_4`    | `dow_0`…`dow_4` (unchanged)                                                         | Already correct                                        |
| Vote booleans      | Vote booleans (unchanged)                                                           | Already binary                                          |
| `score` (discrete) | `score` (unchanged)                                                                 | Small ordinal scale already; no benefit from scaling   |

## Results

| Metric                | run-001 | run-002 | Δ |
|-----------------------|---------|---------|---|
| n features            | 51      | 59      | +8 |
| WR on taken           | 0.491   | 0.544   | **+5.3pp** |
| Profit factor (RR=1)  | 0.96    | 1.19    | +0.23 |
| Sharpe (x252)         | -0.296  | 1.397   | +1.69 |
| Deflated Sharpe Ratio | 0.072   | 0.498   | +0.426 |

See `results/run-002/COMPARISON.md` for the per-fold breakdown and
the top-15 feature gain table. Two conditioned features
(`spread_atr_ratio`, `atr_rel`) make the top 15 by gain.

## Limitations and next step

* The snapshot is irregular in time (one row per alert, not per
  bar). "Log returns" in this dataset are inter-alert price diffs,
  not true bar returns. The transform is still useful as a
  short-memory momentum probe but should be replaced by proper bar
  returns once Dukascopy minute data is ingested.
* `fractional_diff` is implemented but unused in run-002; it
  becomes useful once we have a contiguous price series.
* WR 54.4% with CI [47.1%, 61.5%] still includes 50%. Conditioning
  alone is unlikely to lift WR above the 55% / 60% deployment
  thresholds. The natural follow-up is **meta-labeling** (AFML
  chapter 3.6): train a second XGBoost on the (probability, fold,
  base_rate_estimate) of the run-002 model to learn when to take
  vs skip its own signal.
