# AegisFin-AI Phase 2 — Step 9: XGBoost Baseline v2 Model Report

**Date:** 2026-09-21 04:30:06 UTC  
**Model Name:** `aegisfin_xgboost_baseline_v2`  
**Model Type:** `xgboost.XGBClassifier`  
**Training Split:** [`data/behavioral/splits_v2/train.csv`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/splits_v2/train.csv)  
**Validation Split:** [`data/behavioral/splits_v2/validation.csv`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/splits_v2/validation.csv)  
**Status:** Baseline Training & Evaluation Complete (Uncalibrated)

---

## 1. Executive Summary & Direct v1 vs v2 Comparison

The baseline XGBoost classifier was retrained on the newly generated, realistic **behavioral dataset `v2`** using the exact same hyperparameters and 62 production features.

### Direct Performance Comparison (v1 vs v2)

| Metric | v1 Baseline (Artificial Separation) | v2 Baseline (Realistic Overlap) | Notes / Delta |
| :--- | :---: | :---: | :--- |
| Train Fraud Rate | 7.3583% | 7.6283% | Natural class balance shift |
| Validation Fraud Rate | 10.6500% | 10.0500% | Realistic chronological test distribution |
| `scale_pos_weight` | 12.590034 | 12.109023 | Recalculated strictly on v2 train |
| **PR-AUC (Primary)** | **1.0000** | **0.9992** | Primary ranking metric |
| **ROC-AUC** | **1.0000** | **0.9999** | Global discrimination metric |
| **Log Loss** | 0.000051 | 0.005001 | Probability cross-entropy |
| **Brier Score** | 0.000002 | 0.001024 | Mean squared probability calibration error |
| Precision @ 0.50 | 1.0000 | 0.9892 | Diagnostic cut-off precision |
| Recall @ 0.50 | 1.0000 | 0.9990 | Diagnostic cut-off recall |
| **F1 Score @ 0.50** | **1.0000** | **0.9941** | Harmonic mean at 0.50 |
| True Positives (TP) | 1,065 | 1,004 | Correctly detected fraud cases |
| False Positives (FP) | 0 | 11 | Legitimate transactions flagged as fraud |
| True Negatives (TN) | 8,935 | 8,984 | Correctly approved legitimate transactions |
| False Negatives (FN) | 0 | 1 | Fraud transactions missed |

> [!IMPORTANT]
> **Interpretation Rule:** A lower score does NOT indicate model failure. In v1, the model achieved an artificial 1.0000 due to synthetic voids (zero legitimate velocity noise, $250–$800 amount void). In v2, legitimate micro-purchases, shared household devices, corporate NAT IPs, and stealth ATO/card testing introduce realistic behavioral noise and boundary ambiguity.

---

## 2. Dataset & Split Specifications (v2)

| Dataset Split | Rows | Fraud Count | Legit Count | Fraud Rate | Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **TRAIN** | 60,000 | 4,577 | 55,423 | 7.63% | Model parameter fitting |
| **VALIDATION** | 10,000 | 1,005 | 8,995 | 10.05% | Baseline performance evaluation |
| **CALIBRATION** | 10,000 | 976 | 9,024 | 9.76% | **Untouched** (Reserved for Platt/Isotonic) |
| **POLICY** | 10,000 | 857 | 9,143 | 8.57% | **Untouched** (Reserved for Threshold/Rules) |
| **FINAL TEST** | 10,000 | 823 | 9,177 | 8.23% | **Untouched** (Reserved for final benchmark) |

### Class Imbalance Handling
Class imbalance was addressed using `scale_pos_weight`, calculated strictly from the **v2 training partition only**:
$$\text{scale\_pos\_weight} = \frac{\text{Legitimate}_{\text{train}}}{\text{Fraud}_{\text{train}}} = \frac{55423}{4577} = 12.109023$$

---

## 3. Model Hyperparameters & Configuration

```json
{
  "random_state": 42,
  "n_estimators": 500,
  "learning_rate": 0.05,
  "max_depth": 6,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "objective": "binary:logistic",
  "eval_metric": "aucpr",
  "scale_pos_weight": 12.109023377758357,
  "tree_method": "hist",
  "n_jobs": -1
}
```

- **Feature Count:** Exactly 62 production features (`VALID_FEATURE_NAMES`).
- **Feature Ordering:** Exactly preserved as defined in `app/production_feature_definitions.py`.
- **Identifiers Excluded:** `transaction_id` excluded from feature matrix.
- **Scenario Metadata:** Zero synthetic scenario columns used.

---

## 4. Confusion Matrix (Diagnostic Threshold = 0.50)

| | Predicted Legit (0) | Predicted Fraud (1) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Legit (0)** | 8,984 (TN) | 11 (FP) | 8,995 |
| **Actual Fraud (1)** | 1 (FN) | 1,004 (TP) | 1,005 |
| **Total Predicted** | 8,985 | 1,015 | 10,000 |

---

## 5. Top 15 Feature Importances (Gain)

| Rank | Feature Name | Importance (Gain) |
| :---: | :--- | :---: |
| 1 | `device_tx_count_1h` | 0.314687 |
| 2 | `ip_tx_count_1h` | 0.160881 |
| 3 | `card_device_seen_before` | 0.118283 |
| 4 | `customer_device_seen_before` | 0.093559 |
| 5 | `merchant_tx_count_1h` | 0.057652 |
| 6 | `ip_is_new` | 0.036277 |
| 7 | `product_code_freq` | 0.030949 |
| 8 | `card_ip_seen_before` | 0.028489 |
| 9 | `customer_ip_seen_before` | 0.015605 |
| 10 | `card_is_new` | 0.013650 |
| 11 | `amount_sum_last_1h` | 0.013163 |
| 12 | `customer_is_new` | 0.011959 |
| 13 | `customer_tx_count_5m` | 0.009128 |
| 14 | `ip_unique_customers_24h` | 0.008559 |
| 15 | `customer_tx_count_1h` | 0.008123 |

*Note: All 62 production features are retained in the model. No feature selection or elimination was performed.*

---

## 6. Output Artifacts

- **Model Pickle:** [`models/aegisfin_xgboost_baseline_v2.pkl`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/models/aegisfin_xgboost_baseline_v2.pkl)
- **Model Metadata JSON:** [`models/aegisfin_xgboost_baseline_v2_metadata.json`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/models/aegisfin_xgboost_baseline_v2_metadata.json)
- **Report JSON:** [`reports/xgboost_baseline_v2_report.json`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/reports/xgboost_baseline_v2_report.json)
- **Report Markdown:** [`reports/xgboost_baseline_v2_report.md`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/reports/xgboost_baseline_v2_report.md)

---

## 7. Governance & Anti-Leakage Verifications

1. **Zero Pre-Processing Leakage:** No SMOTE, undersampling, oversampling, scaling, or feature selection applied.
2. **Strict Chronological Sequence:** Model fit strictly on `train.csv` (ended `2026-01-28 05:18:19 UTC`), evaluated on `validation.csv` (started `2026-01-28 05:18:49 UTC`).
3. **Partition Isolation:** Calibration, Policy, and Final Test datasets were never loaded or inspected during this step.
4. **Schema Parity:** Model inputs strictly conform to the 62 features generated by the live production feature engine.
5. **Original Artifact Safety:** v1 model artifact and datasets remain completely unmodified.
