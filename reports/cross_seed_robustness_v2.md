# AegisFin-AI Phase 2 — Step 11: Cross-Seed Robustness Evaluation Report

> [!IMPORTANT]
> **Mandatory Governance Statement:**
> **"This is a cross-seed generalization test within the AegisFin synthetic behavioral benchmark. It does not establish real-world fraud performance."**

---

## 1. Objective

The objective of this evaluation is to determine whether the existing frozen v2 XGBoost baseline model (`models/aegisfin_xgboost_baseline_v2.pkl`) and its associated production probability calibrator (`models/aegisfin_probability_calibrator_v2.pkl`) generalize effectively to an independently generated 100,000-transaction behavioral dataset produced with an entirely distinct random seed (`seed = 123`).

This test answers a critical question in synthetic ML benchmarks:
**Did the model learn generalizable representations of behavioral fraud mechanisms (velocity, bursts, device sharing, entity unfamiliarity), or did it inadvertently overfit to the pseudo-random sampling paths, specific entity IDs, or synthetic artifacts of seed 42?**

---

## 2. Dataset Generation

The independent evaluation dataset was generated using the identical behavioral transaction generator version 2.1.0 with engine version `v2`.

| Parameter | Value |
| :--- | :--- |
| **Generator Version** | `2.1.0` (Realistic Behavioral Engine v2) |
| **Random Seed** | `123` (Independent from training seed `42`) |
| **Total Transactions** | `100,000` |
| **Legitimate Transactions** | `91,777` (91.78%) |
| **Fraudulent Transactions** | `8,223` (8.22%) |
| **Simulation Timespan** | 45.0 days |
| **Raw Dataset File** | [`raw_transactions_100k_seed123.csv`](file:///D:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/raw_transactions_100k_seed123.csv) |
| **Production Features File** | [`production_features_100k_seed123.csv`](file:///D:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/production_features_100k_seed123.csv) |
| **Metadata File** | [`behavioral_dataset_metadata_seed123.json`](file:///D:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/behavioral_dataset_metadata_seed123.json) |

No generator behavioral parameters, weights, or code logic were altered to influence or optimize evaluation performance.

---

## 3. Dataset Validation

The seed-123 dataset underwent the full 11-step automated behavioral validation suite (`scripts/validate_behavioral_dataset.py`) prior to model evaluation.

### Validation Results Summary

| Validation Check | Status | Finding |
| :--- | :---: | :--- |
| **1. Basic Data Integrity** | **PASS** | Exactly 100,000 rows, 0 duplicate transaction IDs, 0 NaN, 0 Inf. |
| **2. Feature Schema Compliance** | **PASS** | Exactly 62 production features in identical contract order. All numeric. |
| **3. Entity Diversity** | **PASS** | 8,000 customers, 9,598 cards, 15,071 devices, 1,200 merchants, 13,007 IPs. |
| **4. Class Balance** | **PASS** | Fraud rate: 8.22% (8,223 fraud / 91,777 legit). |
| **5. High-Value Legitimacy** | **PASS** | 11,531 legitimate transactions >= $500 verified without false-positive triggers. |
| **6. Behavioral Contrast** | **PASS** | Expected contrast directions preserved (e.g. velocity, burst, entity age). |
| **7. Scenario Sanity** | **PASS** | All 7 behavioral scenarios simulated with distinctive profiles. |
| **8. Anti-Leakage Point-in-Time** | **PASS** | Strict point-in-time inequality verified independently; future invariance = 100%. |
| **9. Raw to Feature Parity** | **PASS** | 100,000 / 100,000 rows aligned with 100% parity across ID, label, and amount. |
| **10. Behavioral Red Flags** | **PASS** | No trivial single-feature shortcuts detected; realistic overlap confirmed. |
| **11. Feature Contract Isolation** | **PASS** | Zero scenario columns in feature matrix; `X` contains strictly the 62 contract features. |

---

## 4. Frozen Model Information

The evaluation was executed in strictly read-only inference mode. Neither the base model nor the probability calibrator was retrained, refitted, or modified in any way.

- **Base Model Artifact**: `models/aegisfin_xgboost_baseline_v2.pkl`
- **Base Model Architecture**: `xgboost.XGBClassifier` (`hist` tree method, depth 6, n_estimators 500, lr 0.05, `scale_pos_weight = 12.109`)
- **Base Model Training Set**: `data/behavioral/splits_v2/train.csv` (Seed 42, 60,000 rows)
- **Calibrator Artifact**: `models/aegisfin_probability_calibrator_v2.pkl`
- **Calibrator Architecture**: Platt Sigmoid Scaling (`sklearn.linear_model.LogisticRegression` on logit transform)
- **Calibrator Fitting Set**: `data/behavioral/splits_v2/calibration.csv` (Seed 42, 10,000 rows)

---

## 5. Raw Model Metrics

Metrics computed directly on raw uncalibrated probabilities `p_raw = model.predict_proba(X_seed123)[:, 1]`:

### Global Discriminative Metrics
- **PR-AUC**: `0.999253`
- **ROC-AUC**: `0.999935`
- **Log Loss**: `0.006203`
- **Brier Score**: `0.001511`
- **Expected Calibration Error (ECE)**: `0.001365`
- **Maximum Calibration Error (MCE)**: `0.337436`

### Diagnostic Threshold (0.50) Performance
> *Note: Threshold 0.50 is reported purely as a diagnostic reference. No threshold optimization was performed.*

| Metric | Diagnostic Value (@ 0.50) |
| :--- | :--- |
| **Precision** | `0.981071` (98.11%) |
| **Recall** | `0.995865` (99.59%) |
| **F1 Score** | `0.988413` |
| **True Positives (TP)** | `8,189` |
| **False Positives (FP)** | `158` |
| **True Negatives (TN)** | `91,619` |
| **False Negatives (FN)** | `34` |

### Raw Predicted Probability Distribution
- **Min**: `0.000000`
- **1st Percentile**: `0.000000`
- **25th Percentile**: `0.000001`
- **Median**: `0.000003`
- **Mean**: `0.083595`
- **75th Percentile**: `0.000017`
- **99th Percentile**: `0.999995`
- **Max**: `0.999999`

---

## 6. Calibrated Model Metrics

Metrics computed after transforming predictions with the frozen production Platt calibrator (`aegisfin_probability_calibrator_v2.pkl`):

### Post-Calibration Global Metrics
- **Calibrated Log Loss**: `0.005248` (vs raw: `0.006203`)
- **Calibrated Brier Score**: `0.001333` (vs raw: `0.001511`)
- **Calibrated ECE**: `0.000742`
- **Calibrated MCE**: `0.209556`
- **Calibrated PR-AUC**: `0.999253`
- **Calibrated ROC-AUC**: `0.999935`

### 10-Bin Reliability Table (Independent Seed-123 Dataset)

| Bin | Range | Transactions | Fraud | Legit | Mean Pred Prob | Observed Fraud Rate | Calibration Gap |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `0.0-0.1` | 91,623 | 29 | 91,594 | 0.0001 | 0.0003 | 0.000201 |
| 2 | `0.1-0.2` | 42 | 8 | 34 | 0.1448 | 0.1905 | 0.045633 |
| 3 | `0.2-0.3` | 34 | 11 | 23 | 0.2501 | 0.3235 | 0.073400 |
| 4 | `0.3-0.4` | 32 | 15 | 17 | 0.3407 | 0.4688 | 0.128055 |
| 5 | `0.4-0.5` | 22 | 7 | 15 | 0.4537 | 0.3182 | 0.135562 |
| 6 | `0.5-0.6` | 24 | 13 | 11 | 0.5427 | 0.5417 | 0.000995 |
| 7 | `0.6-0.7` | 29 | 25 | 4 | 0.6525 | 0.8621 | 0.209556 |
| 8 | `0.7-0.8` | 44 | 28 | 16 | 0.7555 | 0.6364 | 0.119148 |
| 9 | `0.8-0.9` | 76 | 60 | 16 | 0.8587 | 0.7895 | 0.069253 |
| 10 | `0.9-1.0` | 8,074 | 8,027 | 47 | 0.9976 | 0.9942 | 0.003429 |

---

## 7. Scenario-Level Diagnostics

Performance evaluated across distinct behavioral scenario labels in the independent seed-123 dataset. Scenario labels were extracted strictly from `raw_transactions_100k_seed123.csv` for post-hoc auditing and were **never** accessible to the model feature matrix.

| Scenario | Total Txs | Fraud Txs | Fraud Rate | PR-AUC | Recall (@0.50) | Precision (@0.50) | F1 (@0.50) | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Account Takeover** | 1,529 | 1,529 | 100.00% | N/A | 0.9954 | 1.0000 | 0.9977 | 1,522 | 0 | 7 | 0 |
| **Burst Fraud** | 1,500 | 1,500 | 100.00% | N/A | 0.9987 | 1.0000 | 0.9993 | 1,498 | 0 | 2 | 0 |
| **Card Testing** | 1,699 | 1,699 | 100.00% | N/A | 0.9959 | 1.0000 | 0.9979 | 1,692 | 0 | 7 | 0 |
| **Legitimate High-Value** | 3,500 | 0 | 0.00% | N/A | N/A | 0.0000 | N/A | 0 | 17 | 0 | 3,483 |
| **Legitimate Stable** | 88,277 | 0 | 0.00% | N/A | N/A | 0.0000 | N/A | 0 | 141 | 0 | 88,136 |
| **Shared Device / IP Ring** | 1,995 | 1,995 | 100.00% | N/A | 1.0000 | 1.0000 | 1.0000 | 1,995 | 0 | 0 | 0 |
| **Stolen Card** | 1,500 | 1,500 | 100.00% | N/A | 0.9880 | 1.0000 | 0.9940 | 1,482 | 0 | 18 | 0 |

---

## 8. Comparison with Seed-42 Results

To assess generalization robustness, we compare the independent seed-123 performance against the established seed-42 validation and calibration benchmarks:

| Benchmark Dimension | (A) Seed-42 Validation Set | (B) Seed-42 Calibration Set | (C) Seed-123 Independent Dataset |
| :--- | :--- | :--- | :--- |
| **Transaction Count** | 10,000 | 10,000 | **100,000** |
| **Fraud Count (Rate)** | 1,005 (10.05%) | 976 (9.76%) | **8,223 (8.22%)** |
| **PR-AUC (Raw)** | 0.999221 | 0.999826 | **0.999253** |
| **ROC-AUC (Raw)** | 0.999924 | 0.999982 | **0.999935** |
| **Log Loss (Raw)** | 0.005001 | 0.002070 | **0.006203** |
| **Brier Score (Raw)** | 0.001024 | 0.000437 | **0.001511** |
| **Precision (@ 0.50)** | 0.989163 | N/A | **0.981071** |
| **Recall (@ 0.50)** | 0.999005 | N/A | **0.995865** |
| **F1 Score (@ 0.50)** | 0.994059 | N/A | **0.988413** |
| **False Positive Count** | 11 | N/A | **158** |
| **False Negative Count** | 1 | N/A | **34** |
| **Calibrated Log Loss** | N/A | 0.001619 | **0.005248** |
| **Calibrated Brier Score** | N/A | 0.000340 | **0.001333** |
| **Calibrated ECE** | N/A | 0.000283 | **0.000742** |

---

## 9. Interpretation

1. **Cross-Seed Stability**: The base model maintains exceptionally high discriminative stability across random seeds (PR-AUC: 0.999253 on seed 123 vs 0.999221 on seed 42 validation). ROC-AUC remains virtually identical at 0.999935.
2. **Probability Calibration Portability**: The frozen Platt sigmoid calibrator fitted exclusively on seed-42 calibration data successfully transfers to seed 123 without re-estimation, yielding a calibrated ECE of 0.000742 and reducing Log Loss and Brier Score consistently.
3. **Absence of Seed memorization**: Because seed 123 instantiates completely different customer IDs, card numbers, device IDs, and IP addresses, the model's sustained performance demonstrates that it relies on behavioral dynamics (rolling velocity aggregations, ratios, and entity-linkage graphs) rather than memorizing entity tokens.
4. **Scenario Coverage**: Across all 5 fraudulent scenarios (Shared Device Ring, Card Testing, Account Takeover, Stolen Card, Burst Fraud), the model achieves high recall and discriminative power. Legitimate High-Value transactions are correctly distinguished from high-value fraud with minimal false alarms.

---

## 10. Limitations

> [!WARNING]
> **Synthetic Benchmark Limitations:**
> 1. **Synthetic Data Boundaries**: This cross-seed robustness test evaluates generalization *across random seeds within the AegisFin synthetic behavioral data generation engine*. It proves that the model is robust to generator sampling variance, entity ID permutation, and temporal scheduling shifts within this generator.
> 2. **Real-World Non-Equivalence**: This test **does NOT** establish performance on real-world banking transaction data. Real-world payment rails contain unmodeled confounding factors, complex social engineering tactics, evolving merchant categories, seasonality, network latency, and delayed fraud chargeback reporting that are not fully captured by the synthetic generator.
> 3. **Adversarial Drift**: Synthetic fraud scenarios follow fixed statistical rules. In production environments, fraud rings continuously adapt their tactics in response to active anti-fraud rules (adversarial drift).

---

## 11. Artifact Manifest

| Artifact File | Description | Status |
| :--- | :--- | :---: |
| [`data/behavioral/raw_transactions_100k_seed123.csv`](file:///D:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/raw_transactions_100k_seed123.csv) | 100,000 raw transactions generated with seed = 123 | Created & Validated |
| [`data/behavioral/production_features_100k_seed123.csv`](file:///D:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/production_features_100k_seed123.csv) | Exactly 62 production features aligned row-for-row | Created & Validated |
| [`data/behavioral/behavioral_dataset_metadata_seed123.json`](file:///D:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/behavioral_dataset_metadata_seed123.json) | Generation configuration and summary metadata | Created & Verified |
| [`models/aegisfin_xgboost_baseline_v2.pkl`](file:///D:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/models/aegisfin_xgboost_baseline_v2.pkl) | Frozen v2 base XGBoost model | Untouched & Frozen |
| [`models/aegisfin_probability_calibrator_v2.pkl`](file:///D:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/models/aegisfin_probability_calibrator_v2.pkl) | Frozen v2 Platt sigmoid probability calibrator | Untouched & Frozen |
| [`reports/cross_seed_robustness_v2.json`](file:///D:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/reports/cross_seed_robustness_v2.json) | Structured evaluation payload and metrics | Created |
| [`reports/cross_seed_robustness_v2.md`](file:///D:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/reports/cross_seed_robustness_v2.md) | Complete 11-section governance and evaluation report | Created |
| [`tests/test_cross_seed_robustness_v2.py`](file:///D:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/tests/test_cross_seed_robustness_v2.py) | Automated pytest regression test suite for Step 11 | Planned / Active |

