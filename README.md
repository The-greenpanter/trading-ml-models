# trading-ml-models

ML research pipeline for an algorithmic forex scalping system. Companion repository to the private `trading-algoritmico` project (5 forex pairs · M5 · scoring-based rule engine).

**Goal:** replace / complement a rule-based scoring system (baseline WR ~35% on a contaminated dataset, ~27.5% after cleaning) with a probabilistic ML model targeting **walk-forward precision >= 0.60**.

**Author:** Juan Diego Peña Castillo ([The-greenpanter](https://github.com/The-greenpanter))
**License:** MIT (see `LICENSE`)

---

## Design philosophy (López de Prado, AFML)

1. **No deep learning shortcuts.** XGBoost baseline first; only escalate to LSTM/GRU/Transformer if the tabular baseline fails to clear the precision target on walk-forward.
2. **Triple-barrier labeling** instead of next-candle WIN/LOSS.
3. **Purged K-Fold + embargo** + walk-forward validation. No `shuffle=True`.
4. **Deflated Sharpe Ratio** (Bailey & López de Prado 2014) as the promotion gate, not raw Sharpe.
5. **Filter the weekend bug** before any training: the source dataset contains 217 Sat/Sun trades from a 24/7 data feed that must be removed (see `memory/research/data_quality_weekend_bug_2026-05-21.md` in the private repo).

See `ARCHITECTURE.md` in the private `trading-algoritmico/ml_models/` directory for the full design rationale.

---

## Quickstart

```bash
# 1. clone
git clone https://github.com/The-greenpanter/trading-ml-models.git
cd trading-ml-models

# 2. install
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. drop the snapshot CSV into data/ (it is gitignored)
#    expected schema: id,timestamp,symbol,direction,entry,sl,tp,...,vote_*,result,...
cp /path/to/_alerts_snapshot.csv data/

# 4. train the XGBoost baseline
python -m src.train --csv data/_alerts_snapshot.csv --out results/run-001/

# 5. inspect the report
cat results/run-001/REPORT.md
```

---

## Layout

```
src/
  data/loader.py        # CSV load + weekend filter + temporal split
  data/labeling.py      # triple-barrier (single-candle fallback used here)
  features/engineering.py
  models/xgboost_model.py
  models/lstm_model.py  # placeholder
  models/meta_labeler.py
  validation/walk_forward.py
  validation/metrics.py # WR, profit factor, DSR
  train.py              # CLI entrypoint
tests/                  # pytest
notebooks/              # 01_eda → 04_lstm_experiment
results/                # gitignored (only REPORT.md and summary.json committed per run)
```

---

## Status

- `run-001` — XGBoost baseline on the cleaned snapshot (1036 trades after weekend filter). See `results/run-001/REPORT.md`.

---

## Related work

- Private companion: `trading-algoritmico` (signal engine, MT5 bridge, VPS bots).
- Architecture rationale: `ml_models/ARCHITECTURE.md` (private repo).
- Data quality note: `memory/research/data_quality_weekend_bug_2026-05-21.md` (private repo).
