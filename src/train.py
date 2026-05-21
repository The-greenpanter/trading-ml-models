"""CLI training entrypoint.

Usage:
    python -m src.train --csv data/_alerts_snapshot.csv --out results/run-001/
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from src.data.loader import load_snapshot, filter_weekend, make_target, temporal_split


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    """2x2 confusion matrix [[TN, FP], [FN, TP]] for binary labels in {0,1}."""
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return [[tn, fp], [fn, tp]]
from src.features.engineering import build_features
from src.models.xgboost_model import train_xgb, save_model
from src.validation.walk_forward import walk_forward_splits
from src.validation.metrics import summarise, win_rate


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def run(csv_path: Path, out_dir: Path, threshold: float = 0.5, n_folds: int = 5) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_snapshot(csv_path)
    n_raw = len(raw)
    clean = filter_weekend(raw)
    n_clean = len(clean)
    n_weekend = n_raw - n_clean

    fs = build_features(clean)
    X = fs.X.values
    y = make_target(clean)

    # --- walk-forward ---
    fold_results = []
    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []
    all_y_prob: list[np.ndarray] = []

    for fold in walk_forward_splits(len(clean), n_folds=n_folds, min_train_frac=0.4):
        X_tr, y_tr = X[fold.train_idx], y[fold.train_idx]
        X_te, y_te = X[fold.test_idx], y[fold.test_idx]
        # carve a small validation tail off the train block for early stopping
        cut = int(len(X_tr) * 0.85)
        X_fit, y_fit = X_tr[:cut], y_tr[:cut]
        X_val, y_val = X_tr[cut:], y_tr[cut:]

        clf = train_xgb(X_fit, y_fit, X_val, y_val)
        y_prob = clf.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        m = summarise(y_te, y_pred, rr=1.0, n_trials=n_folds)
        base_wr = float(np.mean(y_te))
        fold_results.append({
            "fold_id": fold.fold_id,
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "base_rate_test": base_wr,
            "n_predicted_positive": int((y_pred == 1).sum()),
            "win_rate_on_taken": m.win_rate,
            "profit_factor": m.profit_factor,
            "sharpe": m.sharpe,
            "dsr": m.dsr,
        })
        all_y_true.append(y_te)
        all_y_pred.append(y_pred)
        all_y_prob.append(y_prob)

    y_true_cat = np.concatenate(all_y_true)
    y_pred_cat = np.concatenate(all_y_pred)
    y_prob_cat = np.concatenate(all_y_prob)

    overall = summarise(y_true_cat, y_pred_cat, rr=1.0, n_trials=n_folds)
    n_taken = int((y_pred_cat == 1).sum())
    ci_lo, ci_hi = _wilson_ci(overall.win_rate, n_taken)
    cm = _confusion_matrix(y_true_cat, y_pred_cat)

    # --- final model on full temporal split (60/20/20) for export ---
    split = temporal_split(clean)
    Xtr = build_features(split.train).X.values
    ytr = make_target(split.train)
    Xv = build_features(split.val).X.values
    yv = make_target(split.val)
    final_clf = train_xgb(Xtr, ytr, Xv, yv)
    save_model(final_clf, out_dir / "model", fs.feature_names)

    summary = {
        "csv": str(csv_path),
        "n_rows_raw": n_raw,
        "n_rows_weekend_removed": int(n_weekend),
        "n_rows_clean": n_clean,
        "base_rate_clean": float(np.mean(y)),
        "n_features": len(fs.feature_names),
        "threshold": threshold,
        "n_folds": n_folds,
        "fold_results": fold_results,
        "overall": {
            "n_predicted_positive": n_taken,
            "win_rate_on_taken": overall.win_rate,
            "win_rate_ci95": [ci_lo, ci_hi],
            "profit_factor": overall.profit_factor,
            "sharpe": overall.sharpe,
            "dsr": overall.dsr,
            "confusion_matrix": cm,
        },
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _write_report(out_dir, summary, fs.feature_names)
    return summary


def _write_report(out_dir: Path, summary: dict, feature_names: list[str]) -> None:
    cm = summary["overall"]["confusion_matrix"]
    fold_table = "\n".join(
        f"| {r['fold_id']} | {r['n_train']} | {r['n_test']} | {r['base_rate_test']:.3f} | "
        f"{r['n_predicted_positive']} | {r['win_rate_on_taken']:.3f} | "
        f"{r['profit_factor']:.2f} | {r['dsr']:.3f} |"
        for r in summary["fold_results"]
    )
    ci = summary["overall"]["win_rate_ci95"]
    target_hit = summary["overall"]["win_rate_on_taken"] >= 0.60
    conclusion = (
        "XGBoost baseline **reaches** the precision >= 0.60 target on walk-forward. "
        "Promote to shadow mode on the VPS (log-only) for 2 weeks before enabling live orders."
        if target_hit
        else "XGBoost baseline **does not** reach precision >= 0.60 on walk-forward. "
             "Next step per ARCHITECTURE.md: triple-barrier labeling on Dukascopy ticks "
             "(12-24 months) and an LSTM/GRU experiment."
    )
    report = f"""# run-001 — XGBoost baseline

## Dataset

| | |
|---|---|
| Source CSV | `{summary['csv']}` |
| Rows raw | {summary['n_rows_raw']} |
| Weekend rows removed | {summary['n_rows_weekend_removed']} |
| Rows clean | {summary['n_rows_clean']} |
| Base WIN rate (clean) | {summary['base_rate_clean']:.3f} |
| Features | {summary['n_features']} |
| Threshold | {summary['threshold']} |
| Walk-forward folds | {summary['n_folds']} |

## Walk-forward results

| fold | n_train | n_test | base_rate | taken | WR_on_taken | PF | DSR |
|---:|---:|---:|---:|---:|---:|---:|---:|
{fold_table}

## Aggregate

- Trades taken (predicted positive): **{summary['overall']['n_predicted_positive']}**
- WR on taken: **{summary['overall']['win_rate_on_taken']:.3f}**  (Wilson 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])
- Profit factor: **{summary['overall']['profit_factor']:.2f}**
- Sharpe (per-trade, annualised x252): **{summary['overall']['sharpe']:.3f}**
- Deflated Sharpe Ratio (PSR-style probability, n_trials = n_folds): **{summary['overall']['dsr']:.3f}**

### Confusion matrix (cols = predicted, rows = actual)

|              | pred LOSS | pred WIN |
|---           |---:       |---:      |
| actual LOSS  | {cm[0][0]} | {cm[0][1]} |
| actual WIN   | {cm[1][0]} | {cm[1][1]} |

## Conclusion

{conclusion}

## Hyperparameters

See `model/params.json`. Conservative defaults: `n_estimators=300, max_depth=4, lr=0.05, subsample=0.8, colsample=0.8`. `scale_pos_weight` set per fold from class balance.

## Features

{', '.join(f'`{n}`' for n in feature_names)}
"""
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    summary = run(args.csv, args.out, threshold=args.threshold, n_folds=args.folds)
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
