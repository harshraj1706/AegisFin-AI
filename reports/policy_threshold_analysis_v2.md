# AegisFin-AI Phase 2 — Step 12: Policy Threshold Analysis & Risk Bands Report

> [!IMPORTANT]
> **Mandatory Governance Statement:**
> **"Provisional AegisFin policy based on synthetic behavioral policy data. It does not establish real-world banking performance."**

---

## 1. Objective

The objective of Step 12 is to establish operational decision policies and risk-band governance for the AegisFin fraud detection pipeline.
Using the **calibrated posterior fraud probabilities** derived from the frozen XGBoost baseline v2 model and frozen Platt sigmoid calibrator, we evaluate operational threshold trade-offs and delineate four production-ready risk bands (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).

**Strict Data Isolation Guarantee:**
All analysis, threshold sweeps, candidate selections, and risk band definitions were conducted **strictly and exclusively** on the chronological `policy.csv` partition (10,000 transactions). The final test partition (`data/behavioral/splits_v2/test.csv`) remains 100% quarantined and uninspected.

---

## 2. Policy Dataset Validation

The chronological policy partition was validated prior to analysis:

| Parameter | Policy Dataset Value | Validation Finding |
| :--- | :--- | :--- |
| **Dataset Path** | `data/behavioral/splits_v2/policy.csv` | Exists and verified |
| **Total Transactions** | `10,000` | Exactly 10,000 transactions |
| **Legitimate Transactions** | `9,143` (91.43%) | Clean binary class balance |
| **Fraud Transactions** | `857` (8.57%) | Representative behavioral distribution |
| **Chronological Period** | `2026-02-06T02:12:50.525511+00:00` to `2026-02-10T12:40:40.879628+00:00` | Exactly 4.4 days |
| **Temporal Precedence** | Starts strictly after calibration end | Zero future leakage confirmed |
| **Temporal Isolation** | Ends strictly before test set start | Test set completely isolated |
| **Feature Space** | Exactly 62 production features in contract order | Zero NaN, zero Inf values |

---

## 3. Threshold Sweep Summary

A high-resolution threshold sweep across 1,001 equidistant operating points ($t \in [0.000, 1.000]$, step = $0.001$) was conducted on the calibrated fraud probabilities.

### Key Checkpoints Table

| Threshold | Precision | Recall | F1 Score | FPR | Specificity | Flagged Count | Flagged Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.05** | 0.9873 | 1.0000 | 0.9936 | 0.001203 | 0.9988 | 868.0 | 8.68% |
| **0.10** | 0.9896 | 1.0000 | 0.9948 | 0.000984 | 0.9990 | 866.0 | 8.66% |
| **0.25** | 0.9930 | 1.0000 | 0.9965 | 0.000656 | 0.9993 | 863.0 | 8.63% |
| **0.30** | 0.9942 | 1.0000 | 0.9971 | 0.000547 | 0.9995 | 862.0 | 8.62% |
| **0.40** | 0.9942 | 0.9977 | 0.9959 | 0.000547 | 0.9995 | 860.0 | 8.60% |
| **0.50** | 0.9953 | 0.9977 | 0.9965 | 0.000437 | 0.9996 | 859.0 | 8.59% |
| **0.60** | 0.9953 | 0.9977 | 0.9965 | 0.000437 | 0.9996 | 859.0 | 8.59% |
| **0.70** | 0.9965 | 0.9953 | 0.9959 | 0.000328 | 0.9997 | 856.0 | 8.56% |
| **0.80** | 0.9977 | 0.9930 | 0.9953 | 0.000219 | 0.9998 | 853.0 | 8.53% |
| **0.90** | 0.9976 | 0.9837 | 0.9906 | 0.000219 | 0.9998 | 845.0 | 8.45% |
| **0.95** | 1.0000 | 0.9732 | 0.9864 | 0.000000 | 1.0000 | 834.0 | 8.34% |

---

## 4. Candidate Operational Operating Points

Because real-world banking operations face diverse risk appetites and manual review cost profiles, three candidate operating points were identified:

| Policy Dimension | (Candidate A) Maximum Recall | (Candidate B) Balanced Midpoint (Provisional) | (Candidate C) Low False-Positive Rate |
| :--- | :---: | :---: | :---: |
| **Target Threshold** | **`t = 0.31`** | **`t = 0.50`** | **`t = 0.80`** |
| **Fraud Recall** | **100.00%** (857.0 / 857) | **99.77%** (855.0 / 857) | **99.30%** (851.0 / 857) |
| **Precision** | 99.42% | **99.53%** | **99.77%** |
| **F1 Score** | 0.9971 | **0.9965** | 0.9953 |
| **False-Positive Rate** | 0.0547% (5.0 false alarms) | **0.0437%** (4.0 false alarms) | **0.0219%** (2.0 false alarms) |
| **Fraud Capture Rate** | **100.00%** | **99.77%** | **99.30%** |
| **Review / Flag Rate** | 8.62% (862.0 txs) | **8.59%** (859.0 txs) | **8.53%** (853.0 txs) |
| **Recommended Operational Use** | Strict zero-tolerance compliance | General default production decisioning | Automated hard declines with zero customer friction |

---

## 5. Production Risk Bands

Four mutually exclusive, collectively exhaustive, and strictly monotonic risk bands were established based on calibrated posterior probabilities:

| Risk Band | Calibrated Range | Volume | % of Txs | Fraud Count | Observed Fraud Rate | Operational Action | Target Action Description |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LOW** | `[0.00, 0.10)` | 9,134 | 91.34% | 0 | **0.00%** | `AUTO_APPROVE` | Immediate clearing with frictionless customer experience. |
| **MEDIUM** | `[0.10, 0.40)` | 6 | 0.06% | 2 | **33.33%** | `STEP_UP_AUTH` | Automated step-up challenge (SMS OTP, biometrics, 3DS 2.0). |
| **HIGH** | `[0.40, 0.80)` | 7 | 0.07% | 4 | **57.14%** | `MANUAL_REVIEW` | Routing to fraud risk analyst queue for investigation. |
| **CRITICAL** | `[0.80, 1.00]` | 853 | 8.53% | 851 | **99.77%** | `HARD_DECLINE` | Automated immediate transaction rejection and account safeguard lock. |

### Risk Band Properties & Monotonicity
- **Mutual Exclusivity**: Every possible probability $p \in [0.0, 1.0]$ maps to exactly one risk band.
- **Collective Exhaustiveness**: Complete unit interval coverage (`[0.0, 0.10) U [0.10, 0.40) U [0.40, 0.80) U [0.80, 1.00] = [0.0, 1.0]`).
- **Strict Risk Monotonicity**: 
  - `LOW`: 0.00% fraud rate (mean $p = 0.00005$)
  - `MEDIUM`: 33.33% fraud rate (mean $p = 0.26155$)
  - `HIGH`: 57.14% fraud rate (mean $p = 0.65633$)
  - `CRITICAL`: 99.77% fraud rate (mean $p = 0.99623$)
- Higher probability **never** maps to a lower risk band.

---

## 6. Risk-Band Confusion & Operational Behavior

Detailed breakdown of how transaction volume and fraud capture distribute across the risk tiers:

| Risk Band | Legit Txs | Fraud Txs | Total Txs | Fraud Rate | Fraud Capture (% of Total Fraud) | Volume Share (% of Total Txs) | Mean Calibrated Prob |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LOW** | 9,134 | 0 | 9,134 | 0.00% | **0.00%** | 91.34% | 0.00005 |
| **MEDIUM** | 4 | 2 | 6 | 33.33% | **0.23%** | 0.06% | 0.26155 |
| **HIGH** | 3 | 4 | 7 | 57.14% | **0.47%** | 0.07% | 0.65633 |
| **CRITICAL** | 2 | 851 | 853 | 99.77% | **99.30%** | 8.53% | 0.99623 |

### Key Operational Insights
1. **Frictionless Base (91.34%)**: 9,134 transactions are classified as `LOW` risk with **0 false negatives**. 100% of these transactions can be auto-approved instantly.
2. **High-Efficiency Triage (0.13%)**: The intermediate zone (`MEDIUM` + `HIGH`) comprises only 13 transactions (0.13% of volume), yet contains 6 confirmed frauds. This ultra-lean review queue makes human manual review and dynamic 2FA challenges exceptionally scalable.
3. **Decisive Block Rate (8.53%)**: The `CRITICAL` band captures **99.30% of all fraud** (851/857) with an observed fraud purity of 99.77% (only 2 false positives out of 9,143 legitimate transactions).

---

## 7. Selected Provisional Production Policy

- **Provisional Primary Decision Threshold**: **`t = 0.50`**
- **Action Mapping**:
  - Probability < 0.10: `AUTO_APPROVE` (Frictionless clearing)
  - 0.10 <= Probability < 0.40: `STEP_UP_AUTH` (Dynamic 2FA challenge)
  - 0.40 <= Probability < 0.80: `MANUAL_REVIEW` (Human risk review queue)
  - Probability >= 0.80: `HARD_DECLINE` (Immediate decline and alert)

### Selection Rationale:
1. **Defensible Probabilistic Standard**: $t = 0.50$ represents the natural Bayesian decision boundary where the calibrated posterior probability indicates that a transaction is more likely fraudulent than legitimate.
2. **Exceptional Empirical Performance**: At $t = 0.50$, the policy achieves **99.77% fraud recall** (only 2 missed frauds out of 857) while generating only **4 false alarms** across 9,143 legitimate transactions (FPR = 0.0437%).
3. **Operational Feasibility**: Requiring manual intervention on only 0.07% of transactions (`HIGH`) and step-up auth on 0.06% (`MEDIUM`) ensures minimal overhead for compliance and risk teams.

---

## 8. Policy Visualizations

All visual artifacts have been generated using matplotlib and saved to `reports/figures/`:

1. [`reports/figures/policy_precision_recall_f1_vs_threshold.png`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/reports/figures/policy_precision_recall_f1_vs_threshold.png)
2. [`reports/figures/policy_fpr_vs_threshold.png`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/reports/figures/policy_fpr_vs_threshold.png)
3. [`reports/figures/policy_fraud_rate_by_risk_band.png`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/reports/figures/policy_fraud_rate_by_risk_band.png)
4. [`reports/figures/policy_tx_distribution_by_risk_band.png`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/reports/figures/policy_tx_distribution_by_risk_band.png)
5. [`reports/figures/policy_dashboard_summary.png`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/reports/figures/policy_dashboard_summary.png)

---

## 9. Policy Limitations

> [!WARNING]
> **Synthetic Benchmark Limitations:**
> 1. **Synthetic Cost Absence**: Because no monetary cost matrix (e.g., chargeback fee, customer churn cost, manual review hourly cost) was specified, these operational points optimize statistical trade-offs rather than financial P&L.
> 2. **Benchmark Specificity**: These thresholds reflect calibrated probabilities within the AegisFin synthetic behavioral engine. Real-world financial payment networks require empirical recalibration against live chargeback and manual review telemetry.
> 3. **Adversarial Adaptation**: Synthetic transactions operate under static generative rules. In production, fraud rings adapt velocity and credential techniques to exploit fixed thresholds.

---

## 10. Artifact Manifest

| Artifact File | Description | Status |
| :--- | :--- | :---: |
| [`configs/fraud_risk_policy_v2.json`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/configs/fraud_risk_policy_v2.json) | Machine-readable production policy configuration | Created |
| [`reports/policy_threshold_analysis_v2.json`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/reports/policy_threshold_analysis_v2.json) | Structured policy evaluation metrics and sweep checkpoints | Created |
| [`reports/policy_threshold_analysis_v2.md`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/reports/policy_threshold_analysis_v2.md) | Comprehensive policy report and governance documentation | Created |
| [`reports/figures/policy_dashboard_summary.png`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/reports/figures/policy_dashboard_summary.png) | 4-panel policy trade-off and risk-band dashboard | Created |
| [`tests/test_policy_thresholds_v2.py`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/tests/test_policy_thresholds_v2.py) | Automated test suite verifying risk band integrity | Planned |

