# AegisFin-AI Phase 2 — Step 10: Probability Calibration Report

**Date:** 2026-09-21 05:40:46 UTC  
**Base Model:** [`models/aegisfin_xgboost_baseline_v2.pkl`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/models/aegisfin_xgboost_baseline_v2.pkl)  
**Calibration Dataset:** [`data/behavioral/splits_v2/calibration.csv`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/splits_v2/calibration.csv)  
**Selected Method:** **Platt Scaling (Sigmoid Calibration)**  
**Selected Calibrator Artifact:** [`models/aegisfin_probability_calibrator_v2.pkl`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/models/aegisfin_probability_calibrator_v2.pkl)  

---

## 1. Executive Summary & Calibration Method Comparison

Probability calibration was conducted exclusively on the chronological calibration split (`splits_v2/calibration.csv`), strictly after training and validation. The base XGBoost v2 model was **not retrained**.

### Side-by-Side Calibration Performance

| Calibration Method | Log Loss | Brier Score | ECE (Expected Error) | MCE (Max Error) | PR-AUC | ROC-AUC | Status / Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Raw XGBoost Baseline** | 0.002070 | 0.000437 | 0.001854 | 0.003120 | 0.9998 | 1.0000 | Uncalibrated baseline |
| **Sigmoid (Platt Scaling)** | **0.001619** | **0.000340** | **0.000830** | **0.001450** | **0.9998** | **1.0000** | **SELECTED FOR PRODUCTION** |
| **Isotonic Regression** | 0.001234 | 0.000293 | 0.000494 | 0.000920 | 0.9999 | 1.0000 | Non-parametric benchmark |

### Relative Improvements of Sigmoid vs Raw Baseline
- **Log Loss Reduction:** -21.79% (0.002070 -> 0.001619)
- **Brier Score Reduction:** -22.20% (0.000437 -> 0.000340)
- **Expected Calibration Error (ECE) Reduction:** -55.23% (0.001854 -> 0.000830)
- **Ranking Preservation:** PR-AUC and ROC-AUC remain identical (0.9998 and 1.0000), confirming that calibration refined probability reliability without distorting relative ranking order.

---

## 2. Selection Rationale: Why Platt Scaling (Sigmoid) was Selected

While Isotonic regression achieved marginally lower empirical error on the calibration sample, **Platt Scaling (Sigmoid)** was selected as the production calibrator for four fundamental reasons:
1. **Strict Parametric Monotonicity:** Sigmoid calibration applies a continuous logistic function:
   $$P(\text{fraud}|p) = \frac{1}{1 + \exp(-(A \cdot \text{logit}(p) + B))}$$
   with learned parameters $A = 1.0605$ and $B = -1.5920$. This guarantees that higher raw risk always produces strictly higher calibrated risk.
2. **Elimination of Probability Ties / Step Plateaus:** Isotonic regression is a piecewise constant step function that produces identical output probabilities across adjacent raw scores, causing ties that complicate downstream decision threshold optimization.
3. **Out-of-Distribution Robustness:** Parametric sigmoid scaling avoids overfitting to local density variations in the calibration set and extrapolates smoothly to unobserved raw margins.
4. **Contract Parity with Live API:** Directly matches the production contract of AegisFin's live inference service.

---

## 3. Calibration Dataset Statistics

| Metric | Calibration Split Value |
| :--- | :--- |
| **Total Rows** | 10,000 |
| **Legitimate Transactions** | 9,024 (90.24%) |
| **Fraud Transactions** | 976 (9.76%) |
| **Chronological Start (UTC)** | 2026-02-01T15:22:48.459352+00:00 |
| **Chronological End (UTC)** | 2026-02-06T02:12:07.838265+00:00 |
| **Features Evaluated** | Exactly 62 production features in standardized contract order |
| **Isolation Status** | Policy and Test partitions were completely isolated and never loaded |

---

## 4. Reliability Tables & Calibration Curves (10 Bins)

### 4.1 Raw XGBoost Baseline (Uncalibrated)

| Probability Bin | Total Count | Fraud Count | Legit Count | Mean Pred Prob | Observed Fraud Rate | Calibration Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.0-0.1 | 9,014 | 0 | 9,014 | 0.0002 | 0.0000 | 0.0002 |
| 0.1-0.2 | 1 | 0 | 1 | 0.1144 | 0.0000 | 0.1144 |
| 0.2-0.3 | 2 | 0 | 2 | 0.2611 | 0.0000 | 0.2611 |
| 0.3-0.4 | 1 | 0 | 1 | 0.3401 | 0.0000 | 0.3401 |
| 0.4-0.5 | 0 | 0 | 0 | N/A | N/A | 0.0000 |
| 0.5-0.6 | 1 | 0 | 1 | 0.5591 | 0.0000 | 0.5591 |
| 0.6-0.7 | 2 | 1 | 1 | 0.6381 | 0.5000 | 0.1381 |
| 0.7-0.8 | 1 | 0 | 1 | 0.7002 | 0.0000 | 0.7002 |
| 0.8-0.9 | 2 | 1 | 1 | 0.8710 | 0.5000 | 0.3710 |
| 0.9-1.0 | 976 | 974 | 2 | 0.9992 | 0.9980 | 0.0012 |

### 4.2 Sigmoid Calibrated (Platt Scaling — Selected)

| Probability Bin | Total Count | Fraud Count | Legit Count | Mean Pred Prob | Observed Fraud Rate | Calibration Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.0-0.1 | 9,018 | 0 | 9,018 | 0.0001 | 0.0000 | 0.0001 |
| 0.1-0.2 | 0 | 0 | 0 | N/A | N/A | 0.0000 |
| 0.2-0.3 | 3 | 1 | 2 | 0.2504 | 0.3333 | 0.0829 |
| 0.3-0.4 | 1 | 0 | 1 | 0.3335 | 0.0000 | 0.3335 |
| 0.4-0.5 | 0 | 0 | 0 | N/A | N/A | 0.0000 |
| 0.5-0.6 | 1 | 1 | 0 | 0.5971 | 1.0000 | 0.4029 |
| 0.6-0.7 | 1 | 0 | 1 | 0.6166 | 0.0000 | 0.6166 |
| 0.7-0.8 | 2 | 2 | 0 | 0.7766 | 1.0000 | 0.2234 |
| 0.8-0.9 | 1 | 1 | 0 | 0.8127 | 1.0000 | 0.1873 |
| 0.9-1.0 | 973 | 971 | 2 | 0.9978 | 0.9979 | 0.0001 |

### 4.3 Isotonic Calibrated

| Probability Bin | Total Count | Fraud Count | Legit Count | Mean Pred Prob | Observed Fraud Rate | Calibration Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.0-0.1 | 9,020 | 0 | 9,020 | 0.0000 | 0.0000 | 0.0000 |
| 0.1-0.2 | 0 | 0 | 0 | N/A | N/A | 0.0000 |
| 0.2-0.3 | 0 | 0 | 0 | N/A | N/A | 0.0000 |
| 0.3-0.4 | 0 | 0 | 0 | N/A | N/A | 0.0000 |
| 0.4-0.5 | 0 | 0 | 0 | N/A | N/A | 0.0000 |
| 0.5-0.6 | 4 | 2 | 2 | 0.5000 | 0.5000 | 0.0000 |
| 0.6-0.7 | 0 | 0 | 0 | N/A | N/A | 0.0000 |
| 0.7-0.8 | 0 | 0 | 0 | N/A | N/A | 0.0000 |
| 0.8-0.9 | 0 | 0 | 0 | N/A | N/A | 0.0000 |
| 0.9-1.0 | 976 | 974 | 2 | 0.9980 | 0.9980 | 0.0000 |

---

## 5. Artifacts Created

1. **Calibrator Model Pickle:** [`models/aegisfin_probability_calibrator_v2.pkl`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/models/aegisfin_probability_calibrator_v2.pkl)
   - Contains serialized `AegisProbabilityCalibrator` with `.transform(p_raw)` API.
2. **Calibrator Metadata JSON:** [`models/aegisfin_probability_calibrator_v2_metadata.json`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/models/aegisfin_probability_calibrator_v2_metadata.json)
3. **Calibration Report JSON:** [`reports/probability_calibration_v2.json`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/reports/probability_calibration_v2.json)
4. **Calibration Report Markdown:** [`reports/probability_calibration_v2.md`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/reports/probability_calibration_v2.md)

---

## 6. Next Step Readiness (Step 11: Policy & Operating Thresholds)

- The probability calibrator is fitted, saved, and validated.
- Calibrated probabilities are statistically grounded and ready for operating threshold optimization, business cost matrices, and risk band construction strictly on [`data/behavioral/splits_v2/policy.csv`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/splits_v2/policy.csv).
- The final test partition remains 100% untouched.
