"""
scripts/train_xgboost_baseline.py

AegisFin-AI Phase 2 — Step 5: Production-Compatible XGBoost Baseline Model.

Trains an XGBoost baseline classifier strictly on chronological training data
(data/behavioral/splits/train.csv) using the frozen 62 production features.

Evaluates on the chronological validation split (data/behavioral/splits/validation.csv).
Calibration, policy, and final test splits remain strictly UNTOUCHED.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# Ensure project root in python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.production_feature_definitions import (
    VALID_FEATURE_NAMES,
    parse_utc_timestamp,
)

SPLITS_DIR = BASE_DIR / "data" / "behavioral" / "splits"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"


def load_dataset(
    csv_path: Path,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Loads feature split CSV and extracts X, y, and transaction_ids."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {csv_path}")

    df = pd.read_csv(csv_path)

    # 1. Verification of columns
    assert "transaction_id" in df.columns, "Missing transaction_id in split CSV"
    assert "fraud_label" in df.columns, "Missing fraud_label in split CSV"

    # 2. Verify all 62 production features are present in the exact schema order
    for feat in VALID_FEATURE_NAMES:
        assert feat in df.columns, f"Missing feature in split CSV: {feat}"

    # Verify no unexpected feature columns or scenario columns
    feature_cols = [c for c in df.columns if c not in ("transaction_id", "fraud_label")]
    assert len(feature_cols) == len(VALID_FEATURE_NAMES), (
        f"Feature count mismatch: found {len(feature_cols)}, expected {len(VALID_FEATURE_NAMES)}"
    )
    assert feature_cols == VALID_FEATURE_NAMES, "Feature column ordering does not match VALID_FEATURE_NAMES"

    for col in df.columns:
        assert "scenario" not in col.lower(), f"Forbidden scenario column found in dataset: {col}"

    # Extract X, y, and ids
    tx_ids = df["transaction_id"].tolist()
    y = df["fraud_label"].values.astype(int)
    X = df[VALID_FEATURE_NAMES]

    # Verify data cleanliness
    assert not X.isna().any().any(), "NaN values detected in features!"
    assert not np.isinf(X.values).any(), "Inf values detected in features!"
    assert set(np.unique(y)).issubset({0, 1}), f"Non-binary fraud label values: {np.unique(y)}"

    return X, y, tx_ids


def train_and_evaluate(
    train_csv: Path,
    val_csv: Path,
) -> Dict[str, Any]:
    """Fits XGBoost baseline model on train split and evaluates on validation split."""
    print("=" * 80)
    print("AegisFin Phase 2: Train XGBoost Baseline Model")
    print("=" * 80)

    # 1. Load Data
    print("\n[Step 1/5] Loading training and validation splits...")
    X_train, y_train, train_ids = load_dataset(train_csv)
    X_val, y_val, val_ids = load_dataset(val_csv)

    n_train = len(X_train)
    n_val = len(X_val)
    train_fraud = int(np.sum(y_train))
    train_legit = n_train - train_fraud
    val_fraud = int(np.sum(y_val))
    val_legit = n_val - val_fraud

    print(f"  Training rows   : {n_train:,} (Fraud: {train_fraud:,} [{train_fraud/n_train*100:.2f}%], Legit: {train_legit:,})")
    print(f"  Validation rows : {n_val:,} (Fraud: {val_fraud:,} [{val_fraud/n_val*100:.2f}%], Legit: {val_legit:,})")
    print(f"  Feature count   : {len(VALID_FEATURE_NAMES)} production features")

    # 2. Calculate Scale Pos Weight ONLY from Training Data
    print("\n[Step 2/5] Calculating class imbalance weighting from training data...")
    scale_pos_weight = float(train_legit / train_fraud)
    print(f"  scale_pos_weight = {train_legit} / {train_fraud} = {scale_pos_weight:.6f}")

    # 3. Model Configuration & Training
    print("\n[Step 3/5] Fitting XGBoost Baseline Classifier...")
    model_params = {
        "random_state": 42,
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "n_jobs": -1,
    }

    model = XGBClassifier(**model_params)

    t0 = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    fit_duration_sec = round(time.time() - t0, 2)
    print(f"  XGBoost training completed in {fit_duration_sec} seconds.")

    # 4. Validation Evaluation
    print("\n[Step 4/5] Evaluating model on validation split (uncalibrated baseline)...")
    val_probs = model.predict_proba(X_val)[:, 1]

    # Metrics
    pr_auc = float(average_precision_score(y_val, val_probs))
    roc_auc = float(roc_auc_score(y_val, val_probs))
    loss = float(log_loss(y_val, val_probs))
    brier = float(brier_score_loss(y_val, val_probs))

    # Diagnostic threshold at 0.50
    thresh_diag = 0.50
    y_pred_50 = (val_probs >= thresh_diag).astype(int)
    prec_50 = float(precision_score(y_val, y_pred_50, zero_division=0))
    rec_50 = float(recall_score(y_val, y_pred_50, zero_division=0))
    f1_50 = float(f1_score(y_val, y_pred_50, zero_division=0))
    cm_50 = confusion_matrix(y_val, y_pred_50)
    tn_50, fp_50, fn_50, tp_50 = cm_50.ravel()

    print(f"  PR-AUC (Primary Metric)   : {pr_auc:.4f}")
    print(f"  ROC-AUC                   : {roc_auc:.4f}")
    print(f"  Log Loss                  : {loss:.4f}")
    print(f"  Brier Score               : {brier:.4f}")
    print(f"  Diagnostic Threshold 0.50 :")
    print(f"    Precision               : {prec_50:.4f}")
    print(f"    Recall                  : {rec_50:.4f}")
    print(f"    F1 Score                : {f1_50:.4f}")
    print(f"    Confusion Matrix        : TP={tp_50}, FP={fp_50}, TN={tn_50}, FN={fn_50}")
    print("  * Note: Threshold 0.50 is purely diagnostic. Operational threshold tuning will be done in Policy.")

    # 5. Feature Importance Analysis
    importances = model.feature_importances_
    feat_imp_pairs = sorted(
        zip(VALID_FEATURE_NAMES, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    top_15_features = [
        {"rank": i + 1, "feature_name": f, "importance": float(round(imp, 6))}
        for i, (f, imp) in enumerate(feat_imp_pairs[:15])
    ]

    print("\nTop 15 Most Important Features (Gain / Split Importance):")
    for item in top_15_features:
        print(f"  {item['rank']:>2}. {item['feature_name']:<32} : {item['importance']:.6f}")

    # 6. Save Artifacts
    print("\n[Step 5/5] Saving model artifact and metadata...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "aegisfin_xgboost_baseline.pkl"
    meta_path = MODELS_DIR / "aegisfin_xgboost_baseline_metadata.json"
    rep_json_path = REPORTS_DIR / "xgboost_baseline_report.json"
    rep_md_path = REPORTS_DIR / "xgboost_baseline_report.md"

    # Save model artifact with pickle
    artifact_payload = {
        "model": model,
        "model_type": "xgboost.XGBClassifier",
        "feature_names": VALID_FEATURE_NAMES,
        "feature_count": len(VALID_FEATURE_NAMES),
        "target_col": "fraud_label",
        "scale_pos_weight": scale_pos_weight,
        "model_params": model_params,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "trained_on": str(train_csv),
        "validation_metrics": {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "log_loss": loss,
            "brier_score": brier,
            "precision_at_0_50": prec_50,
            "recall_at_0_50": rec_50,
            "f1_at_0_50": f1_50,
        },
    }

    with open(model_path, "wb") as f:
        pickle.dump(artifact_payload, f)
    print(f"  -> Model artifact saved to: {model_path}")

    # Save metadata JSON
    meta_payload = {
        "model_name": "aegisfin_xgboost_baseline",
        "model_type": "xgboost.XGBClassifier",
        "created_at_utc": artifact_payload["created_at_utc"],
        "training_data": {
            "path": str(train_csv),
            "rows": n_train,
            "fraud_count": train_fraud,
            "legitimate_count": train_legit,
            "fraud_rate_pct": round(train_fraud / n_train * 100, 4),
            "scale_pos_weight": scale_pos_weight,
        },
        "validation_data": {
            "path": str(val_csv),
            "rows": n_val,
            "fraud_count": val_fraud,
            "legitimate_count": val_legit,
            "fraud_rate_pct": round(val_fraud / n_val * 100, 4),
        },
        "feature_count": len(VALID_FEATURE_NAMES),
        "feature_names": VALID_FEATURE_NAMES,
        "model_params": model_params,
        "training_duration_seconds": fit_duration_sec,
        "validation_metrics": artifact_payload["validation_metrics"],
        "diagnostic_confusion_matrix_at_0_50": {
            "true_positive": int(tp_50),
            "false_positive": int(fp_50),
            "true_negative": int(tn_50),
            "false_negative": int(fn_50),
        },
        "top_15_features": top_15_features,
        "untouched_splits": ["calibration.csv", "policy.csv", "test.csv"],
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)
    print(f"  -> Model metadata saved to: {meta_path}")

    # Save Report JSON
    with open(rep_json_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)
    print(f"  -> Report JSON saved to: {rep_json_path}")

    # Generate Markdown Report
    top_15_table_rows = "\n".join(
        f"| {item['rank']} | `{item['feature_name']}` | {item['importance']:.6f} |"
        for item in top_15_features
    )

    md_content = f"""# AegisFin-AI Phase 2 — XGBoost Baseline Model Report

**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Model Name:** `aegisfin_xgboost_baseline`  
**Model Type:** `xgboost.XGBClassifier`  
**Status:** Baseline Training & Evaluation Complete (Uncalibrated)

---

## 1. Executive Summary & Core Results

The first production-compatible baseline fraud detection model for AegisFin Phase 2 was successfully trained using the frozen 62 production features.

Training was performed strictly on chronological training data (`train.csv`), and evaluated on the chronological validation split (`validation.csv`). Calibration, policy, and final test splits were **not used** and remain 100% untouched.

| Metric | Baseline Value | Interpretation |
|---|---|---|
| **PR-AUC (Primary Metric)** | **{pr_auc:.4f}** | Area under Precision-Recall curve on chronological validation |
| **ROC-AUC** | **{roc_auc:.4f}** | Discriminative ranking capability |
| **Log Loss** | **{loss:.4f}** | Cross-entropy loss (uncalibrated probabilities) |
| **Brier Score** | **{brier:.4f}** | Mean squared probability error |
| **Precision @ 0.50** | **{prec_50:.4f}** | Diagnostic precision at default 0.50 cut-off |
| **Recall @ 0.50** | **{rec_50:.4f}** | Diagnostic recall at default 0.50 cut-off |
| **F1 Score @ 0.50** | **{f1_50:.4f}** | Diagnostic harmonic mean at default 0.50 cut-off |

> [!IMPORTANT]
> **Threshold Notice:** The 0.50 threshold is purely diagnostic. Operational threshold optimization and risk bands will be established in subsequent Policy steps.

---

## 2. Dataset & Split Specifications

| Dataset Split | Rows | Fraud Count | Legit Count | Fraud Rate | Role |
|---|---|---|---|---|---|
| **TRAIN** | {n_train:,} | {train_fraud:,} | {train_legit:,} | {train_fraud / n_train * 100:.2f}% | Model parameter fitting |
| **VALIDATION** | {n_val:,} | {val_fraud:,} | {val_legit:,} | {val_fraud / n_val * 100:.2f}% | Baseline performance evaluation |
| **CALIBRATION** | 10,000 | 1,084 | 8,916 | 10.84% | **Untouched** (Reserved for Platt/Isotonic) |
| **POLICY** | 10,000 | 1,059 | 8,941 | 10.59% | **Untouched** (Reserved for Threshold/Rules) |
| **FINAL TEST** | 10,000 | 877 | 9,123 | 8.77% | **Untouched** (Reserved for final benchmark) |

### Class Imbalance Handling
Class imbalance was addressed using `scale_pos_weight`, calculated strictly from the **training partition only**:
$$\\text{{scale\\_pos\\_weight}} = \\frac{{\\text{{Legitimate}}_{{\\text{{train}}}}}}{{\\text{{Fraud}}_{{\\text{{train}}}}}} = \\frac{{{train_legit}}}{{{train_fraud}}} = {scale_pos_weight:.6f}$$

---

## 3. Model Hyperparameters & Configuration

```json
{json.dumps(model_params, indent=2)}
```

- **Feature Count:** Exactly 62 production features (from `VALID_FEATURE_NAMES`).
- **Feature Ordering:** Exactly preserved as defined in `app/production_feature_definitions.py`.
- **Identifiers Excluded:** `transaction_id` excluded from feature matrix.
- **Scenario Metadata:** Zero synthetic scenario columns used.

---

## 4. Confusion Matrix (Diagnostic Threshold = 0.50)

| | Predicted Legit (0) | Predicted Fraud (1) | Total Actual |
|---|---|---|---|
| **Actual Legit (0)** | {tn_50:,} (TN) | {fp_50:,} (FP) | {val_legit:,} |
| **Actual Fraud (1)** | {fn_50:,} (FN) | {tp_50:,} (TP) | {val_fraud:,} |
| **Total Predicted** | {tn_50 + fn_50:,} | {tp_50 + fp_50:,} | {n_val:,} |

---

## 5. Top 15 Feature Importances (Gain)

| Rank | Feature Name | Importance (Gain) |
|---|---|---|
{top_15_table_rows}

*Note: All 62 production features are retained in the model. No features were dropped based on importance.*

---

## 6. Output Artifacts

- **Model Pickle:** `models/aegisfin_xgboost_baseline.pkl`
- **Model Metadata JSON:** `models/aegisfin_xgboost_baseline_metadata.json`
- **Report JSON:** `reports/xgboost_baseline_report.json`
- **Report Markdown:** `reports/xgboost_baseline_report.md`

---

## 7. Governance & Anti-Leakage Verifications

1. **Zero Pre-Processing Leakage:** No SMOTE, undersampling, oversampling, scaling, or feature selection applied.
2. **Strict Chronological Sequence:** Model fit strictly on `train.csv` (ended `2026-01-28 11:02:38 UTC`), evaluated on `validation.csv` (started `2026-01-28 11:03:14 UTC`).
3. **Partition Isolation:** Calibration, Policy, and Final Test datasets were never loaded or inspected during this step.
4. **Schema Parity:** Model inputs strictly conform to the 62 features generated by the live production feature engine.
"""

    with open(rep_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  -> Report Markdown saved to: {rep_md_path}")

    return meta_payload


def main():
    parser = argparse.ArgumentParser(description="Train AegisFin Phase 2 XGBoost Baseline")
    parser.add_argument(
        "--train-csv",
        type=str,
        default=str(SPLITS_DIR / "train.csv"),
        help="Path to training split CSV",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default=str(SPLITS_DIR / "validation.csv"),
        help="Path to validation split CSV",
    )
    args = parser.parse_args()

    train_and_evaluate(
        train_csv=Path(args.train_csv),
        val_csv=Path(args.val_csv),
    )


if __name__ == "__main__":
    main()
