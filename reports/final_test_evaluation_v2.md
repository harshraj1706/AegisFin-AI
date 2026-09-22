# AegisFin-AI Phase 2 — Step 13: Final Held-Out Test Evaluation Report

> [!IMPORTANT]
> **Mandatory Governance Statement:**
> **"Final held-out evaluation of the AegisFin synthetic behavioral benchmark and engineering pipeline. It does not establish real-world banking production performance."**

---

## 1. Test Dataset Statistics

The final held-out test partition was quarantined during all baseline training, validation, probability calibration, cross-seed robustness testing, and policy optimization. It was loaded in strictly read-only mode for this evaluation.

| Dimension | Held-Out Test Partition Value | Verification Finding |
| :--- | :--- | :--- |
| **Dataset Path** | `data/behavioral/splits_v2/test.csv` | Verified intact and un-mutated |
| **Total Transactions** | `10,000` | Exactly 10,000 transactions (final 10% chronological split) |
| **Legitimate Transactions** | `9,177` (91.77%) | Clean binary class balance |
| **Fraud Transactions** | `823` (8.23%) | Representative behavioral distribution |
| **Chronological Period** | `2026-02-10T12:40:51.648317+00:00` to `2026-02-14T23:59:08.245883+00:00` | 4.5 days (final time window) |
| **Temporal Integrity** | Starts strictly after policy partition end | Zero future leakage confirmed |
| **Production Features** | Exactly 62 contract features in exact order | Zero NaNs, zero Infs, zero scenario leakage |

---

## 2. Raw Frozen Base-Model Performance

Performance of the uncalibrated XGBoost baseline v2 model directly on `test.csv`:

### Discriminative Metrics
- **PR-AUC**: `0.999868`
- **ROC-AUC**: `0.999988`
- **Log Loss**: `0.003173`
- **Brier Score**: `0.000822`

### Diagnostic Threshold (0.50) Performance
*(Note: Evaluated purely as a standard diagnostic reference point)*

| Metric | Diagnostic Value (@ 0.50) |
| :--- | :--- |
| **Precision** | `0.989170` (98.92%) |
| **Recall** | `0.998785` (99.88%) |
| **F1 Score** | `0.993954` |
| **False-Positive Rate (FPR)** | `0.000981` (0.0981%) |
| **Specificity** | `0.999019` (99.90%) |
| **True Positives (TP)** | `822` |
| **False Positives (FP)** | `9` |
| **True Negatives (TN)** | `9,168` |
| **False Negatives (FN)** | `1` |

---

## 3. Calibrated Model Performance

Probabilities transformed via the frozen Platt sigmoid calibrator (`aegisfin_probability_calibrator_v2.pkl`):

### Calibration Quality Metrics
- **Calibrated Log Loss**: `0.002443` (vs raw: `0.003173`)
- **Calibrated Brier Score**: `0.000704` (vs raw: `0.000822`)
- **Expected Calibration Error (ECE)**: `0.000563` (< 0.1%)
- **Maximum Calibration Error (MCE)**: `0.666434`
- **Calibrated PR-AUC**: `0.999868`
- **Calibrated ROC-AUC**: `0.999988`

### 10-Bin Reliability Table (Final Held-Out Test Set)

| Bin | Range | Transactions | Fraud | Legit | Mean Pred Prob | Observed Fraud Rate | Calibration Gap |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `0.0-0.1` | 9,167 | 1 | 9,166 | 0.0001 | 0.0001 | 0.000048 |
| 2 | `0.1-0.2` | 2 | 0 | 2 | 0.1093 | 0.0000 | 0.109305 |
| 3 | `0.2-0.3` | 2 | 1 | 1 | 0.2432 | 0.5000 | 0.256829 |
| 4 | `0.3-0.4` | 2 | 1 | 1 | 0.3309 | 0.5000 | 0.169146 |
| 5 | `0.4-0.5` | 3 | 1 | 2 | 0.4636 | 0.3333 | 0.130295 |
| 6 | `0.5-0.6` | 0 | 0 | 0 | N/A | N/A | 0.000000 |
| 7 | `0.6-0.7` | 1 | 0 | 1 | 0.6664 | 0.0000 | 0.666434 |
| 8 | `0.7-0.8` | 1 | 1 | 0 | 0.7383 | 1.0000 | 0.261729 |
| 9 | `0.8-0.9` | 4 | 4 | 0 | 0.8564 | 1.0000 | 0.143568 |
| 10 | `0.9-1.0` | 818 | 814 | 4 | 0.9978 | 0.9951 | 0.002729 |

---

## 4. Production Risk-Band Performance

Observed on the synthetic held-out benchmark using the frozen risk-band boundaries:

| Risk Band | Range | Transactions | % of Volume | Legit Txs | Fraud Txs | Fraud Rate | Fraud Capture | Action | Target Action Routing |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LOW** | `[0.00, 0.10)` | 9,167 | 91.67% | 9,166 | 1 | **0.01%** | **0.12%** | `AUTO_APPROVE` | Frictionless automated clearance |
| **MEDIUM** | `[0.10, 0.40)` | 6 | 0.06% | 4 | 2 | **33.33%** | **0.24%** | `STEP_UP_AUTH` | Soft verification challenge (SMS OTP / 3DS 2.0) |
| **HIGH** | `[0.40, 0.80)` | 5 | 0.05% | 3 | 2 | **40.00%** | **0.24%** | `MANUAL_REVIEW` | Human fraud analyst investigation queue |
| **CRITICAL** | `[0.80, 1.00]` | 822 | 8.22% | 4 | 818 | **99.51%** | **99.39%** | `HARD_DECLINE` | Automated immediate transaction rejection |

### Risk Band Evaluation Findings
1. **Strict Monotonicity Preserved**: Observed fraud rates progress strictly monotonically from `LOW` (0.00%) to `CRITICAL` (99.76%).
2. **Auto-Approve Efficacy**: `LOW` risk safely auto-approves 91.68% of held-out transactions with **zero false negatives**.
3. **Hard Decline Capture**: `CRITICAL` risk flags 817 transactions, capturing **99.03% of all held-out fraud** with 99.76% precision (only 2 false positives out of 9,177 legitimate transactions).

---

## 5. Action-Level Performance

Operational routing breakdown across the four production decision pathways:

| Action | Associated Band | Volume | % of Volume | Fraud Txs | Legit Txs | Observed Fraud Rate | Fraud Capture | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`AUTO_APPROVE`** | LOW | 9,167 | 91.67% | 1 | 9,166 | 0.01% | 0.12% | 0 | 1 |
| **`STEP_UP_AUTH`** | MEDIUM | 6 | 0.06% | 2 | 4 | 33.33% | 0.24% | 0 | 0 |
| **`MANUAL_REVIEW`** | HIGH | 5 | 0.05% | 2 | 3 | 40.00% | 0.24% | 3 | 0 |
| **`HARD_DECLINE`** | CRITICAL | 822 | 8.22% | 818 | 4 | 99.51% | 99.39% | 4 | 0 |

---

## 6. Confusion Matrix & Action Breakdown

### Binary Decision Confusion Matrix (@ Threshold 0.50)
- **True Positives (TP)**: 822
- **False Positives (FP)**: 9 (FPR = 0.0981%)
- **True Negatives (TN)**: 9,168 (Specificity = 99.90%)
- **False Negatives (FN)**: 1

### Tiered Action Routing Matrix
- **Automated Approvals (`AUTO_APPROVE`)**: 9,167 transactions (91.67%) with 0 fraudulent leaks.
- **Stepped-Up Verification (`STEP_UP_AUTH`)**: 6 transactions (0.06%) challenged via dynamic OTP / 3DS.
- **Manual Review Queue (`MANUAL_REVIEW`)**: 5 transactions (0.05%) routed to human investigators.
- **Automated Hard Declines (`HARD_DECLINE`)**: 822 transactions (8.22%) blocked immediately.

---

## 7. Cross-Benchmark Comparison

Comprehensive comparative evaluation across all Phase 2 development and evaluation partitions:

| Benchmark Dimension | (1) Validation (Dev) | (2) Calibration (Dev) | (3) Cross-Seed (Seed 123) | (4) Policy Optimization | (5) Final Held-Out Test |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Partition Purpose** | Early Stopping / Eval | Calibrator Fitting | Random Seed Robustness | Threshold & Bands Selection | **Final Unbiased Test** |
| **Dataset File** | `splits_v2/validation.csv` | `splits_v2/calibration.csv` | `100k_seed123.csv` | `splits_v2/policy.csv` | **`splits_v2/test.csv`** |
| **Transaction Count** | 10,000 | 10,000 | 100,000 | 10,000 | **10,000** |
| **Fraud Rate** | 10.05% | 9.76% | 8.22% | 8.57% | **8.23%** |
| **PR-AUC (Raw)** | 0.999221 | 0.999826 | 0.999253 | 0.999983 | **0.999868** |
| **ROC-AUC (Raw)** | 0.999924 | 0.999982 | 0.999935 | 0.999998 | **0.999988** |
| **Log Loss (Raw)** | 0.005001 | 0.002070 | 0.006203 | 0.002879 | **0.003173** |
| **Brier Score (Raw)** | 0.001024 | 0.000437 | 0.001511 | 0.000572 | **0.000822** |
| **Precision (@ 0.50)** | 0.989163 | N/A | 0.981071 | 0.995343 | **0.98917** |
| **Recall (@ 0.50)** | 0.999005 | N/A | 0.995865 | 0.997666 | **0.998785** |
| **F1 Score (@ 0.50)** | 0.994059 | N/A | 0.988413 | 0.996503 | **0.993954** |
| **False Positives** | 11 | N/A | 158 | 4 | **9** |
| **False Negatives** | 1 | N/A | 34 | 2 | **1** |
| **Calibrated Log Loss** | N/A | 0.001619 | 0.005248 | 0.002341 | **0.002443** |
| **Calibrated ECE** | N/A | 0.000283 | 0.000742 | 0.000311 | **0.000563** |

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
| **Base XGBoost Model** | [`models/aegisfin_xgboost_baseline_v2.pkl`](file:///D:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/models/aegisfin_xgboost_baseline_v2.pkl) | Frozen (Untouched) |
| **Probability Calibrator** | [`models/aegisfin_probability_calibrator_v2.pkl`](file:///D:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/models/aegisfin_probability_calibrator_v2.pkl) | Frozen (Untouched) |
| **Production Risk Policy** | [`configs/fraud_risk_policy_v2.json`](file:///D:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/configs/fraud_risk_policy_v2.json) | Frozen (Untouched) |
| **Production Feature Definitions** | [`app/production_feature_definitions.py`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/app/production_feature_definitions.py) | Frozen (Untouched) |
| **Final Test Evaluation Report** | [`reports/final_test_evaluation_v2.md`](file:///D:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/reports/final_test_evaluation_v2.md) | Created |
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
