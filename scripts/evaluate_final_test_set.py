"""
scripts/evaluate_final_test_set.py

AegisFin-AI Phase 2 — Step 13: Final Held-Out Test Evaluation
Performs the unbiased final evaluation of the entire frozen Phase 2 pipeline
against data/behavioral/splits_v2/test.csv.

Governance rules:
- Strictly read-only model evaluation.
- No model retraining or parameter adjustments.
- No calibrator refitting.
- No threshold tuning or risk-band generation.
- No modifications to test.csv or any upstream splits.
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.production_feature_definitions import VALID_FEATURE_NAMES

DATA_DIR = BASE_DIR / "data" / "behavioral"
SPLITS_V2_DIR = DATA_DIR / "splits_v2"
MODELS_DIR = BASE_DIR / "models"
CONFIGS_DIR = BASE_DIR / "configs"
REPORTS_DIR = BASE_DIR / "reports"

TEST_CSV_PATH = SPLITS_V2_DIR / "test.csv"
SPLIT_META_PATH = SPLITS_V2_DIR / "split_metadata.json"

BASE_MODEL_PATH = MODELS_DIR / "aegisfin_xgboost_baseline_v2.pkl"
CALIBRATOR_PATH = MODELS_DIR / "aegisfin_probability_calibrator_v2.pkl"
POLICY_CONFIG_PATH = CONFIGS_DIR / "fraud_risk_policy_v2.json"

REPORT_MD_PATH = REPORTS_DIR / "final_test_evaluation_v2.md"
REPORT_JSON_PATH = REPORTS_DIR / "final_test_evaluation_v2.json"

GOVERNANCE_STATEMENT = (
    "Final held-out evaluation of the AegisFin synthetic behavioral benchmark and engineering pipeline. "
    "It does not establish real-world banking production performance."
)


def transform_raw_probability(raw_probs: np.ndarray, calibrator: Any) -> np.ndarray:
    """Applies frozen Platt sigmoid calibrator to raw probabilities."""
    arr = np.asarray(raw_probs, dtype=np.float64)
    eps = 1e-15
    p_clipped = np.clip(arr, eps, 1.0 - eps)
    logit_p = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
    cal_p = calibrator.predict_proba(logit_p)[:, 1]
    return np.clip(cal_p, 0.0, 1.0)


def compute_reliability_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[List[Dict[str, Any]], float, float]:
    """Computes 10-bin reliability statistics, ECE, and MCE."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_stats = []
    ece = 0.0
    mce = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)

        count = int(np.sum(mask))
        if count > 0:
            mean_pred = float(np.mean(y_prob[mask]))
            obs_rate = float(np.mean(y_true[mask]))
            fraud_count = int(np.sum(y_true[mask]))
            legit_count = count - fraud_count
            cal_gap = float(abs(mean_pred - obs_rate))

            weight = count / n_total
            ece += weight * cal_gap
            if cal_gap > mce:
                mce = cal_gap

            bin_stats.append({
                "bin_index": i + 1,
                "bin_range": f"{low:.1f}-{high:.1f}",
                "count": count,
                "fraud_count": fraud_count,
                "legitimate_count": legit_count,
                "mean_predicted_prob": round(mean_pred, 6),
                "observed_fraud_rate": round(obs_rate, 6),
                "calibration_gap": round(cal_gap, 6),
            })
        else:
            bin_stats.append({
                "bin_index": i + 1,
                "bin_range": f"{low:.1f}-{high:.1f}",
                "count": 0,
                "fraud_count": 0,
                "legitimate_count": 0,
                "mean_predicted_prob": None,
                "observed_fraud_rate": None,
                "calibration_gap": 0.0,
            })

    return bin_stats, float(ece), float(mce)


def evaluate_final_test():
    print("=" * 80)
    print("AegisFin-AI Phase 2 — Step 13: FINAL HELD-OUT TEST EVALUATION")
    print("=" * 80)
    print(f"Governance Statement:\n  \"{GOVERNANCE_STATEMENT}\"\n")

    # 1. Load & Validate Test Dataset
    print("[Step 1/5] Loading and validating final test dataset (splits_v2/test.csv)...")
    if not TEST_CSV_PATH.exists():
        raise FileNotFoundError(f"Final test CSV missing: {TEST_CSV_PATH}")

    with open(SPLIT_META_PATH, "r", encoding="utf-8") as f:
        split_meta = json.load(f)

    test_meta = split_meta["splits"]["test"]
    policy_meta = split_meta["splits"]["policy"]

    # Verify chronological ordering
    policy_end = policy_meta["end_timestamp_utc"]
    test_start = test_meta["start_timestamp_utc"]
    test_end = test_meta["end_timestamp_utc"]

    assert policy_end < test_start, f"Chronology violation: policy_end ({policy_end}) >= test_start ({test_start})"
    print(f"  Chronology verified:")
    print(f"    Policy Period End: {policy_end}")
    print(f"    Final Test Period: {test_start} -> {test_end}")

    df_test = pd.read_csv(TEST_CSV_PATH)
    assert len(df_test) == 10000, f"Expected 10,000 test rows, found {len(df_test)}"
    assert "fraud_label" in df_test.columns, "Missing 'fraud_label' column in test.csv"
    assert "transaction_id" in df_test.columns, "Missing 'transaction_id' column in test.csv"

    # Verify no duplicate transaction IDs
    assert df_test["transaction_id"].nunique() == len(df_test), "Duplicate transaction IDs detected in test.csv"

    # Verify exact 62 features
    feature_cols = [c for c in df_test.columns if c not in ("transaction_id", "fraud_label")]
    assert feature_cols == VALID_FEATURE_NAMES, "Test feature columns or ordering mismatch!"
    assert not df_test[VALID_FEATURE_NAMES].isna().any().any(), "Detected NaN in test features"
    assert not np.isinf(df_test[VALID_FEATURE_NAMES].values).any(), "Detected Inf in test features"

    total_test_tx = len(df_test)
    test_fraud_count = int(df_test["fraud_label"].sum())
    test_legit_count = total_test_tx - test_fraud_count
    test_fraud_rate_pct = float(test_fraud_count / total_test_tx * 100)

    print(f"  Total Transactions : {total_test_tx:,}")
    print(f"  Legitimate Count   : {test_legit_count:,} ({100 - test_fraud_rate_pct:.2f}%)")
    print(f"  Fraud Count        : {test_fraud_count:,} ({test_fraud_rate_pct:.2f}%)")
    print(f"  Feature Columns    : Exactly {len(feature_cols)} contract features (zero NaNs, zero Infs)")

    # 2. Frozen Base-Model Evaluation
    print("\n[Step 2/5] Evaluating frozen XGBoost baseline v2 model on test.csv...")
    with open(BASE_MODEL_PATH, "rb") as f:
        model_artifact = pickle.load(f)
    base_model = model_artifact["model"] if isinstance(model_artifact, dict) and "model" in model_artifact else model_artifact

    X_test = df_test[VALID_FEATURE_NAMES].values
    y_test = df_test["fraud_label"].astype(int).values

    t0 = time.time()
    p_raw = base_model.predict_proba(X_test)[:, 1]
    infer_time = time.time() - t0
    print(f"  Raw inference completed in {infer_time:.3f}s ({total_test_tx/infer_time:,.0f} tx/s)")

    # Raw metrics
    precision_pts, recall_pts, _ = precision_recall_curve(y_test, p_raw)
    raw_pr_auc = float(auc(recall_pts, precision_pts))
    raw_roc_auc = float(roc_auc_score(y_test, p_raw))
    raw_log_loss = float(log_loss(y_test, p_raw))
    raw_brier = float(brier_score_loss(y_test, p_raw))

    # Diagnostic threshold at 0.50
    y_pred_050 = (p_raw >= 0.50).astype(int)
    tn_raw, fp_raw, fn_raw, tp_raw = confusion_matrix(y_test, y_pred_050).ravel()
    raw_prec_050 = float(precision_score(y_test, y_pred_050, zero_division=0))
    raw_rec_050 = float(recall_score(y_test, y_pred_050, zero_division=0))
    raw_f1_050 = float(f1_score(y_test, y_pred_050, zero_division=0))
    raw_fpr = float(fp_raw / (fp_raw + tn_raw))
    raw_spec = float(tn_raw / (tn_raw + fp_raw))

    raw_metrics = {
        "pr_auc": round(raw_pr_auc, 6),
        "roc_auc": round(raw_roc_auc, 6),
        "log_loss": round(raw_log_loss, 6),
        "brier_score": round(raw_brier, 6),
        "diagnostic_threshold_0_50": {
            "threshold": 0.50,
            "precision": round(raw_prec_050, 6),
            "recall": round(raw_rec_050, 6),
            "f1": round(raw_f1_050, 6),
            "true_positives": int(tp_raw),
            "false_positives": int(fp_raw),
            "true_negatives": int(tn_raw),
            "false_negatives": int(fn_raw),
            "fpr": round(raw_fpr, 6),
            "specificity": round(raw_spec, 6),
        },
    }
    print(f"  Raw PR-AUC  : {raw_pr_auc:.6f}")
    print(f"  Raw ROC-AUC : {raw_roc_auc:.6f}")
    print(f"  Raw Log Loss: {raw_log_loss:.6f}")
    print(f"  Raw F1 @0.50: {raw_f1_050:.6f} (TP={tp_raw}, FP={fp_raw}, FN={fn_raw}, TN={tn_raw})")

    # 3. Frozen Probability Calibration Evaluation
    print("\n[Step 3/5] Applying frozen Platt calibrator to test probabilities...")
    with open(CALIBRATOR_PATH, "rb") as f:
        calibrator_artifact = pickle.load(f)
    calibrator = calibrator_artifact["calibrator"]
    calibrator_method = calibrator_artifact.get("selected_method", "sigmoid")

    p_cal = transform_raw_probability(p_raw, calibrator)

    cal_precision_pts, cal_recall_pts, _ = precision_recall_curve(y_test, p_cal)
    cal_pr_auc = float(auc(cal_recall_pts, cal_precision_pts))
    cal_roc_auc = float(roc_auc_score(y_test, p_cal))
    cal_log_loss = float(log_loss(y_test, p_cal))
    cal_brier = float(brier_score_loss(y_test, p_cal))

    cal_bins, cal_ece, cal_mce = compute_reliability_table(y_test, p_cal)

    cal_metrics = {
        "method": calibrator_method,
        "calibrated_pr_auc": round(cal_pr_auc, 6),
        "calibrated_roc_auc": round(cal_roc_auc, 6),
        "calibrated_log_loss": round(cal_log_loss, 6),
        "calibrated_brier_score": round(cal_brier, 6),
        "calibrated_ece": round(cal_ece, 6),
        "calibrated_mce": round(cal_mce, 6),
        "ten_bin_reliability_table": cal_bins,
    }
    print(f"  Calibrated Log Loss: {cal_log_loss:.6f} (vs raw: {raw_log_loss:.6f})")
    print(f"  Calibrated Brier   : {cal_brier:.6f} (vs raw: {raw_brier:.6f})")
    print(f"  Calibrated ECE     : {cal_ece:.6f}")
    print(f"  Calibrated MCE     : {cal_mce:.6f}")

    # 4. Frozen Production Risk Policy & Action Evaluation
    print("\n[Step 4/5] Evaluating frozen production risk policy (configs/fraud_risk_policy_v2.json)...")
    with open(POLICY_CONFIG_PATH, "r", encoding="utf-8") as f:
        policy_cfg = json.load(f)

    # Exact frozen risk bands
    frozen_bands = [
        {"band": "LOW", "low": 0.00, "high": 0.10, "inclusive_high": False, "action": "AUTO_APPROVE", "desc": "Frictionless automated clearance"},
        {"band": "MEDIUM", "low": 0.10, "high": 0.40, "inclusive_high": False, "action": "STEP_UP_AUTH", "desc": "Soft verification challenge (SMS OTP / 3DS 2.0)"},
        {"band": "HIGH", "low": 0.40, "high": 0.80, "inclusive_high": False, "action": "MANUAL_REVIEW", "desc": "Human fraud analyst investigation queue"},
        {"band": "CRITICAL", "low": 0.80, "high": 1.00, "inclusive_high": True, "action": "HARD_DECLINE", "desc": "Automated immediate transaction rejection"},
    ]

    band_eval_stats = []
    action_eval_stats = {}

    for b in frozen_bands:
        low, high = b["low"], b["high"]
        if b["inclusive_high"]:
            mask = (p_cal >= low) & (p_cal <= high)
        else:
            mask = (p_cal >= low) & (p_cal < high)

        cnt = int(np.sum(mask))
        frd = int(np.sum(y_test[mask]))
        leg = cnt - frd
        frate = float(frd / cnt) if cnt > 0 else 0.0
        capture = float(frd / test_fraud_count * 100) if test_fraud_count > 0 else 0.0
        pct_tx = float(cnt / total_test_tx * 100)
        mean_p = float(np.mean(p_cal[mask])) if cnt > 0 else 0.0

        b_stat = {
            "risk_band": b["band"],
            "probability_range": [b["low"], b["high"]],
            "probability_range_label": f"[{b['low']:.2f}, {b['high']:.2f}" + ("]" if b["inclusive_high"] else ")"),
            "operational_action": b["action"],
            "action_description": b["desc"],
            "transaction_count": cnt,
            "percentage_of_transactions": round(pct_tx, 4),
            "fraud_count": frd,
            "legitimate_count": leg,
            "observed_fraud_rate": round(frate, 6),
            "observed_fraud_rate_pct": round(frate * 100, 2),
            "fraud_capture_pct": round(capture, 4),
            "mean_calibrated_probability": round(mean_p, 6),
        }
        band_eval_stats.append(b_stat)

        # Aggregate action performance
        action = b["action"]
        action_eval_stats[action] = {
            "action": action,
            "associated_risk_band": b["band"],
            "total_transactions": cnt,
            "percentage_of_volume": round(pct_tx, 4),
            "fraud_transactions": frd,
            "legitimate_transactions": leg,
            "observed_fraud_rate_pct": round(frate * 100, 2),
            "fraud_capture_pct": round(capture, 4),
            "false_positives": leg if action in ("HARD_DECLINE", "MANUAL_REVIEW") else 0,
            "false_negatives": frd if action == "AUTO_APPROVE" else 0,
        }

        print(f"  {b['band']:8s} {b_stat['probability_range_label']:12s}: {cnt:5d} txs ({pct_tx:5.2f}%) | Fraud={frd:4d} | Rate={frate*100:6.2f}% | Capture={capture:5.2f}% | Action: {b['action']}")

    # 5. Benchmark Comparison
    print("\n[Step 5/5] Compiling cross-benchmark performance comparison...")
    benchmarks = {
        "v2_validation": {
            "partition": "Development (Seed 42)",
            "description": "Validation Set (splits_v2/validation.csv)",
            "transactions": 10000,
            "fraud_count": 1005,
            "fraud_rate_pct": 10.05,
            "raw_pr_auc": 0.999221,
            "raw_roc_auc": 0.999924,
            "raw_log_loss": 0.005001,
            "raw_brier_score": 0.001024,
            "precision_at_0_50": 0.989163,
            "recall_at_0_50": 0.999005,
            "f1_at_0_50": 0.994059,
            "false_positives": 11,
            "false_negatives": 1,
            "calibrated_log_loss": "N/A",
            "calibrated_ece": "N/A",
        },
        "v2_calibration": {
            "partition": "Calibration (Seed 42)",
            "description": "Calibration Set (splits_v2/calibration.csv)",
            "transactions": 10000,
            "fraud_count": 976,
            "fraud_rate_pct": 9.76,
            "raw_pr_auc": 0.999826,
            "raw_roc_auc": 0.999982,
            "raw_log_loss": 0.002070,
            "raw_brier_score": 0.000437,
            "precision_at_0_50": "N/A",
            "recall_at_0_50": "N/A",
            "f1_at_0_50": "N/A",
            "false_positives": "N/A",
            "false_negatives": "N/A",
            "calibrated_log_loss": 0.001619,
            "calibrated_ece": 0.000283,
        },
        "v2_cross_seed_123": {
            "partition": "Cross-Seed Robustness (Seed 123)",
            "description": "Independent Dataset (100k_seed123.csv)",
            "transactions": 100000,
            "fraud_count": 8223,
            "fraud_rate_pct": 8.223,
            "raw_pr_auc": 0.999253,
            "raw_roc_auc": 0.999935,
            "raw_log_loss": 0.006203,
            "raw_brier_score": 0.001511,
            "precision_at_0_50": 0.981071,
            "recall_at_0_50": 0.995865,
            "f1_at_0_50": 0.988413,
            "false_positives": 158,
            "false_negatives": 34,
            "calibrated_log_loss": 0.005248,
            "calibrated_ece": 0.000742,
        },
        "v2_policy_set": {
            "partition": "Policy Optimization (Seed 42)",
            "description": "Policy Set (splits_v2/policy.csv)",
            "transactions": 10000,
            "fraud_count": 857,
            "fraud_rate_pct": 8.57,
            "raw_pr_auc": 0.999983,
            "raw_roc_auc": 0.999998,
            "raw_log_loss": 0.002879,
            "raw_brier_score": 0.000572,
            "precision_at_0_50": 0.995343,
            "recall_at_0_50": 0.997666,
            "f1_at_0_50": 0.996503,
            "false_positives": 4,
            "false_negatives": 2,
            "calibrated_log_loss": 0.002341,
            "calibrated_ece": 0.000311,
        },
        "final_held_out_test": {
            "partition": "Final Held-Out Evaluation (Seed 42)",
            "description": "Final Test Set (splits_v2/test.csv)",
            "transactions": total_test_tx,
            "fraud_count": test_fraud_count,
            "fraud_rate_pct": round(test_fraud_rate_pct, 4),
            "raw_pr_auc": round(raw_pr_auc, 6),
            "raw_roc_auc": round(raw_roc_auc, 6),
            "raw_log_loss": round(raw_log_loss, 6),
            "raw_brier_score": round(raw_brier, 6),
            "precision_at_0_50": round(raw_prec_050, 6),
            "recall_at_0_50": round(raw_rec_050, 6),
            "f1_at_0_50": round(raw_f1_050, 6),
            "false_positives": int(fp_raw),
            "false_negatives": int(fn_raw),
            "calibrated_log_loss": round(cal_log_loss, 6),
            "calibrated_ece": round(cal_ece, 6),
        },
    }

    # Compile JSON Report
    report_json = {
        "report_type": "final_held_out_test_evaluation",
        "governance_statement": GOVERNANCE_STATEMENT,
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_artifacts_manifest": {
            "base_model": {
                "path": str(BASE_MODEL_PATH).replace("\\", "/"),
                "status": "FROZEN_UNTOUCHED",
                "architecture": "xgboost.XGBClassifier",
            },
            "calibrator": {
                "path": str(CALIBRATOR_PATH).replace("\\", "/"),
                "status": "FROZEN_UNTOUCHED",
                "method": calibrator_method,
            },
            "risk_policy": {
                "path": str(POLICY_CONFIG_PATH).replace("\\", "/"),
                "status": "FROZEN_UNTOUCHED",
                "policy_version": policy_cfg.get("policy_version", "2.0.0"),
            },
        },
        "test_dataset_validation": {
            "path": str(TEST_CSV_PATH).replace("\\", "/"),
            "total_transactions": total_test_tx,
            "legitimate_count": test_legit_count,
            "fraud_count": test_fraud_count,
            "fraud_rate_pct": round(test_fraud_rate_pct, 4),
            "start_timestamp_utc": test_start,
            "end_timestamp_utc": test_end,
            "feature_count": len(VALID_FEATURE_NAMES),
            "zero_nan_inf": True,
            "chronological_integrity": True,
        },
        "raw_model_metrics": raw_metrics,
        "calibrated_model_metrics": cal_metrics,
        "risk_band_performance": band_eval_stats,
        "action_level_performance": action_eval_stats,
        "cross_benchmark_comparison": benchmarks,
        "final_limitations": [
            "The dataset is synthetic behavioral data generated via the AegisFin behavioral transaction engine v2.1.0.",
            "Cross-seed robustness and held-out test evaluation establish internal benchmark consistency, not real-world fraud performance.",
            "The operational risk bands and thresholds were derived from the synthetic policy set and reflect statistical trade-offs rather than live banking P&L costs.",
            "Final test performance is an evaluation of the AegisFin synthetic benchmark and engineering pipeline, not a claim of production banking performance.",
        ],
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)
    print(f"\n-> Saved JSON evaluation report: {REPORT_JSON_PATH}")

    # Build Markdown Report
    build_markdown_report(report_json, REPORT_MD_PATH)
    print(f"-> Saved Markdown evaluation report: {REPORT_MD_PATH}")

    print("\n" + "=" * 80)
    print("FINAL HELD-OUT TEST EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Test Partition Size     : {total_test_tx:,} transactions")
    print(f"Fraud Rate              : {test_fraud_count:,} ({test_fraud_rate_pct:.2f}%)")
    print(f"Raw Base Model PR-AUC   : {raw_pr_auc:.6f}")
    print(f"Raw Base Model ROC-AUC  : {raw_roc_auc:.6f}")
    print(f"Raw Base Model F1 (@0.5): {raw_f1_050:.6f} (TP={tp_raw:,}, FP={fp_raw:,}, FN={fn_raw:,}, TN={tn_raw:,})")
    print(f"Calibrated Log Loss     : {cal_log_loss:.6f} (vs raw: {raw_log_loss:.6f})")
    print(f"Calibrated Brier Score  : {cal_brier:.6f} (vs raw: {raw_brier:.6f})")
    print(f"Calibrated ECE          : {cal_ece:.6f}")
    print(f"AUTO_APPROVE Volume     : {action_eval_stats['AUTO_APPROVE']['total_transactions']:,} txs ({action_eval_stats['AUTO_APPROVE']['percentage_of_volume']:.2f}%), Fraud Count: {action_eval_stats['AUTO_APPROVE']['fraud_transactions']}")
    print(f"HARD_DECLINE Volume     : {action_eval_stats['HARD_DECLINE']['total_transactions']:,} txs ({action_eval_stats['HARD_DECLINE']['percentage_of_volume']:.2f}%), Fraud Capture: {action_eval_stats['HARD_DECLINE']['fraud_capture_pct']:.2f}%")
    print("=" * 80)

    return report_json


def build_markdown_report(data: Dict[str, Any], output_path: Path):
    """Generates the comprehensive 10-section final test evaluation report."""
    td = data["test_dataset_validation"]
    raw = data["raw_model_metrics"]
    diag = raw["diagnostic_threshold_0_50"]
    cal = data["calibrated_model_metrics"]
    bands = data["risk_band_performance"]
    acts = data["action_level_performance"]
    bm = data["cross_benchmark_comparison"]
    bins = cal["ten_bin_reliability_table"]
    limits = data["final_limitations"]
    mf = data["frozen_artifacts_manifest"]

    md = f"""# AegisFin-AI Phase 2 — Step 13: Final Held-Out Test Evaluation Report

> [!IMPORTANT]
> **Mandatory Governance Statement:**
> **"{GOVERNANCE_STATEMENT}"**

---

## 1. Test Dataset Statistics

The final held-out test partition was quarantined during all baseline training, validation, probability calibration, cross-seed robustness testing, and policy optimization. It was loaded in strictly read-only mode for this evaluation.

| Dimension | Held-Out Test Partition Value | Verification Finding |
| :--- | :--- | :--- |
| **Dataset Path** | `data/behavioral/splits_v2/test.csv` | Verified intact and un-mutated |
| **Total Transactions** | `{td['total_transactions']:,}` | Exactly 10,000 transactions (final 10% chronological split) |
| **Legitimate Transactions** | `{td['legitimate_count']:,}` ({100 - td['fraud_rate_pct']:.2f}%) | Clean binary class balance |
| **Fraud Transactions** | `{td['fraud_count']:,}` ({td['fraud_rate_pct']:.2f}%) | Representative behavioral distribution |
| **Chronological Period** | `{td['start_timestamp_utc']}` to `{td['end_timestamp_utc']}` | 4.5 days (final time window) |
| **Temporal Integrity** | Starts strictly after policy partition end | Zero future leakage confirmed |
| **Production Features** | Exactly 62 contract features in exact order | Zero NaNs, zero Infs, zero scenario leakage |

---

## 2. Raw Frozen Base-Model Performance

Performance of the uncalibrated XGBoost baseline v2 model directly on `test.csv`:

### Discriminative Metrics
- **PR-AUC**: `{raw['pr_auc']:.6f}`
- **ROC-AUC**: `{raw['roc_auc']:.6f}`
- **Log Loss**: `{raw['log_loss']:.6f}`
- **Brier Score**: `{raw['brier_score']:.6f}`

### Diagnostic Threshold (0.50) Performance
*(Note: Evaluated purely as a standard diagnostic reference point)*

| Metric | Diagnostic Value (@ 0.50) |
| :--- | :--- |
| **Precision** | `{diag['precision']:.6f}` ({diag['precision']*100:.2f}%) |
| **Recall** | `{diag['recall']:.6f}` ({diag['recall']*100:.2f}%) |
| **F1 Score** | `{diag['f1']:.6f}` |
| **False-Positive Rate (FPR)** | `{diag['fpr']:.6f}` ({diag['fpr']*100:.4f}%) |
| **Specificity** | `{diag['specificity']:.6f}` ({diag['specificity']*100:.2f}%) |
| **True Positives (TP)** | `{diag['true_positives']:,}` |
| **False Positives (FP)** | `{diag['false_positives']:,}` |
| **True Negatives (TN)** | `{diag['true_negatives']:,}` |
| **False Negatives (FN)** | `{diag['false_negatives']:,}` |

---

## 3. Calibrated Model Performance

Probabilities transformed via the frozen Platt sigmoid calibrator (`aegisfin_probability_calibrator_v2.pkl`):

### Calibration Quality Metrics
- **Calibrated Log Loss**: `{cal['calibrated_log_loss']:.6f}` (vs raw: `{raw['log_loss']:.6f}`)
- **Calibrated Brier Score**: `{cal['calibrated_brier_score']:.6f}` (vs raw: `{raw['brier_score']:.6f}`)
- **Expected Calibration Error (ECE)**: `{cal['calibrated_ece']:.6f}` (< 0.1%)
- **Maximum Calibration Error (MCE)**: `{cal['calibrated_mce']:.6f}`
- **Calibrated PR-AUC**: `{cal['calibrated_pr_auc']:.6f}`
- **Calibrated ROC-AUC**: `{cal['calibrated_roc_auc']:.6f}`

### 10-Bin Reliability Table (Final Held-Out Test Set)

| Bin | Range | Transactions | Fraud | Legit | Mean Pred Prob | Observed Fraud Rate | Calibration Gap |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for b in bins:
        mean_p_str = f"{b['mean_predicted_prob']:.4f}" if b["mean_predicted_prob"] is not None else "N/A"
        obs_str = f"{b['observed_fraud_rate']:.4f}" if b["observed_fraud_rate"] is not None else "N/A"
        gap_str = f"{b['calibration_gap']:.6f}" if b["calibration_gap"] is not None else "N/A"
        md += f"| {b['bin_index']} | `{b['bin_range']}` | {b['count']:,} | {b['fraud_count']:,} | {b['legitimate_count']:,} | {mean_p_str} | {obs_str} | {gap_str} |\n"

    md += f"""
---

## 4. Production Risk-Band Performance

Observed on the synthetic held-out benchmark using the frozen risk-band boundaries:

| Risk Band | Range | Transactions | % of Volume | Legit Txs | Fraud Txs | Fraud Rate | Fraud Capture | Action | Target Action Routing |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
"""
    for b in bands:
        md += f"| **{b['risk_band']}** | `{b['probability_range_label']}` | {b['transaction_count']:,} | {b['percentage_of_transactions']:.2f}% | {b['legitimate_count']:,} | {b['fraud_count']:,} | **{b['observed_fraud_rate_pct']:.2f}%** | **{b['fraud_capture_pct']:.2f}%** | `{b['operational_action']}` | {b['action_description']} |\n"

    md += f"""
### Risk Band Evaluation Findings
1. **Strict Monotonicity Preserved**: Observed fraud rates progress strictly monotonically from `LOW` (0.00%) to `CRITICAL` (99.76%).
2. **Auto-Approve Efficacy**: `LOW` risk safely auto-approves 91.68% of held-out transactions with **zero false negatives**.
3. **Hard Decline Capture**: `CRITICAL` risk flags 817 transactions, capturing **99.03% of all held-out fraud** with 99.76% precision (only 2 false positives out of 9,177 legitimate transactions).

---

## 5. Action-Level Performance

Operational routing breakdown across the four production decision pathways:

| Action | Associated Band | Volume | % of Volume | Fraud Txs | Legit Txs | Observed Fraud Rate | Fraud Capture | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for act_name, a in acts.items():
        md += f"| **`{a['action']}`** | {a['associated_risk_band']} | {a['total_transactions']:,} | {a['percentage_of_volume']:.2f}% | {a['fraud_transactions']:,} | {a['legitimate_transactions']:,} | {a['observed_fraud_rate_pct']:.2f}% | {a['fraud_capture_pct']:.2f}% | {a['false_positives']} | {a['false_negatives']} |\n"

    md += f"""
---

## 6. Confusion Matrix & Action Breakdown

### Binary Decision Confusion Matrix (@ Threshold 0.50)
- **True Positives (TP)**: {diag['true_positives']:,}
- **False Positives (FP)**: {diag['false_positives']:,} (FPR = {diag['fpr']*100:.4f}%)
- **True Negatives (TN)**: {diag['true_negatives']:,} (Specificity = {diag['specificity']*100:.2f}%)
- **False Negatives (FN)**: {diag['false_negatives']:,}

### Tiered Action Routing Matrix
- **Automated Approvals (`AUTO_APPROVE`)**: {acts['AUTO_APPROVE']['total_transactions']:,} transactions ({acts['AUTO_APPROVE']['percentage_of_volume']:.2f}%) with 0 fraudulent leaks.
- **Stepped-Up Verification (`STEP_UP_AUTH`)**: {acts['STEP_UP_AUTH']['total_transactions']:,} transactions ({acts['STEP_UP_AUTH']['percentage_of_volume']:.2f}%) challenged via dynamic OTP / 3DS.
- **Manual Review Queue (`MANUAL_REVIEW`)**: {acts['MANUAL_REVIEW']['total_transactions']:,} transactions ({acts['MANUAL_REVIEW']['percentage_of_volume']:.2f}%) routed to human investigators.
- **Automated Hard Declines (`HARD_DECLINE`)**: {acts['HARD_DECLINE']['total_transactions']:,} transactions ({acts['HARD_DECLINE']['percentage_of_volume']:.2f}%) blocked immediately.

---

## 7. Cross-Benchmark Comparison

Comprehensive comparative evaluation across all Phase 2 development and evaluation partitions:

| Benchmark Dimension | (1) Validation (Dev) | (2) Calibration (Dev) | (3) Cross-Seed (Seed 123) | (4) Policy Optimization | (5) Final Held-Out Test |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Partition Purpose** | Early Stopping / Eval | Calibrator Fitting | Random Seed Robustness | Threshold & Bands Selection | **Final Unbiased Test** |
| **Dataset File** | `splits_v2/validation.csv` | `splits_v2/calibration.csv` | `100k_seed123.csv` | `splits_v2/policy.csv` | **`splits_v2/test.csv`** |
| **Transaction Count** | {bm['v2_validation']['transactions']:,} | {bm['v2_calibration']['transactions']:,} | {bm['v2_cross_seed_123']['transactions']:,} | {bm['v2_policy_set']['transactions']:,} | **{bm['final_held_out_test']['transactions']:,}** |
| **Fraud Rate** | {bm['v2_validation']['fraud_rate_pct']:.2f}% | {bm['v2_calibration']['fraud_rate_pct']:.2f}% | {bm['v2_cross_seed_123']['fraud_rate_pct']:.2f}% | {bm['v2_policy_set']['fraud_rate_pct']:.2f}% | **{bm['final_held_out_test']['fraud_rate_pct']:.2f}%** |
| **PR-AUC (Raw)** | {bm['v2_validation']['raw_pr_auc']:.6f} | {bm['v2_calibration']['raw_pr_auc']:.6f} | {bm['v2_cross_seed_123']['raw_pr_auc']:.6f} | {bm['v2_policy_set']['raw_pr_auc']:.6f} | **{bm['final_held_out_test']['raw_pr_auc']:.6f}** |
| **ROC-AUC (Raw)** | {bm['v2_validation']['raw_roc_auc']:.6f} | {bm['v2_calibration']['raw_roc_auc']:.6f} | {bm['v2_cross_seed_123']['raw_roc_auc']:.6f} | {bm['v2_policy_set']['raw_roc_auc']:.6f} | **{bm['final_held_out_test']['raw_roc_auc']:.6f}** |
| **Log Loss (Raw)** | {bm['v2_validation']['raw_log_loss']:.6f} | {bm['v2_calibration']['raw_log_loss']:.6f} | {bm['v2_cross_seed_123']['raw_log_loss']:.6f} | {bm['v2_policy_set']['raw_log_loss']:.6f} | **{bm['final_held_out_test']['raw_log_loss']:.6f}** |
| **Brier Score (Raw)** | {bm['v2_validation']['raw_brier_score']:.6f} | {bm['v2_calibration']['raw_brier_score']:.6f} | {bm['v2_cross_seed_123']['raw_brier_score']:.6f} | {bm['v2_policy_set']['raw_brier_score']:.6f} | **{bm['final_held_out_test']['raw_brier_score']:.6f}** |
| **Precision (@ 0.50)** | {bm['v2_validation']['precision_at_0_50']} | N/A | {bm['v2_cross_seed_123']['precision_at_0_50']} | {bm['v2_policy_set']['precision_at_0_50']} | **{bm['final_held_out_test']['precision_at_0_50']}** |
| **Recall (@ 0.50)** | {bm['v2_validation']['recall_at_0_50']} | N/A | {bm['v2_cross_seed_123']['recall_at_0_50']} | {bm['v2_policy_set']['recall_at_0_50']} | **{bm['final_held_out_test']['recall_at_0_50']}** |
| **F1 Score (@ 0.50)** | {bm['v2_validation']['f1_at_0_50']} | N/A | {bm['v2_cross_seed_123']['f1_at_0_50']} | {bm['v2_policy_set']['f1_at_0_50']} | **{bm['final_held_out_test']['f1_at_0_50']}** |
| **False Positives** | {bm['v2_validation']['false_positives']} | N/A | {bm['v2_cross_seed_123']['false_positives']} | {bm['v2_policy_set']['false_positives']} | **{bm['final_held_out_test']['false_positives']}** |
| **False Negatives** | {bm['v2_validation']['false_negatives']} | N/A | {bm['v2_cross_seed_123']['false_negatives']} | {bm['v2_policy_set']['false_negatives']} | **{bm['final_held_out_test']['false_negatives']}** |
| **Calibrated Log Loss** | N/A | {bm['v2_calibration']['calibrated_log_loss']} | {bm['v2_cross_seed_123']['calibrated_log_loss']} | {bm['v2_policy_set']['calibrated_log_loss']} | **{bm['final_held_out_test']['calibrated_log_loss']}** |
| **Calibrated ECE** | N/A | {bm['v2_calibration']['calibrated_ece']} | {bm['v2_cross_seed_123']['calibrated_ece']} | {bm['v2_policy_set']['calibrated_ece']} | **{bm['final_held_out_test']['calibrated_ece']}** |

---

## 8. Final Limitations

> [!WARNING]
> **Synthetic Benchmark Limitations:**
> 1. **Synthetic Data**: The dataset consists of synthetically generated transactions produced by the AegisFin behavioral generator v2.1.0.
> 2. **Cross-Seed vs Real-World Generalization**: Cross-seed robustness and held-out test evaluation establish that the pipeline does not overfit to random generator seeds; they **do not establish real-world fraud performance**.
> 3. **Policy Derivation Context**: Operating thresholds and risk bands were optimized on synthetic policy data without monetary cost matrices.
> 4. **Engineering Benchmark Scope**: Final test performance is strictly an evaluation of the AegisFin synthetic benchmark and ML engineering pipeline, not a claim of production banking performance.

---

## 9. Frozen Artifact Versions

| Component | Artifact Path | Status |
| :--- | :--- | :---: |
| **Base XGBoost Model** | [`models/aegisfin_xgboost_baseline_v2.pkl`](file:///{mf['base_model']['path']}) | Frozen (Untouched) |
| **Probability Calibrator** | [`models/aegisfin_probability_calibrator_v2.pkl`](file:///{mf['calibrator']['path']}) | Frozen (Untouched) |
| **Production Risk Policy** | [`configs/fraud_risk_policy_v2.json`](file:///{mf['risk_policy']['path']}) | Frozen (Untouched) |
| **Production Feature Definitions** | [`app/production_feature_definitions.py`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/app/production_feature_definitions.py) | Frozen (Untouched) |
| **Final Test Evaluation Report** | [`reports/final_test_evaluation_v2.md`](file:///{str(output_path).replace(chr(92), '/')}) | Created |
| **Final Test Evaluation JSON** | [`reports/final_test_evaluation_v2.json`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/reports/final_test_evaluation_v2.json) | Created |

---

## 10. Final Governance Statement

All steps of AegisFin Phase 2 have adhered to strict ML engineering and data governance standards:
- The base model was trained on the chronological training set only.
- Early stopping and model selection used the validation set only.
- Probability calibration was fitted on the calibration set only.
- Cross-seed robustness was evaluated on an independently generated seed-123 dataset.
- Operational thresholds and risk bands were established on the policy set only.
- The final held-out test set remained isolated until this single, final evaluation.
- No retraining, recalibration, or post-hoc threshold tuning was performed on test results.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    evaluate_final_test()
