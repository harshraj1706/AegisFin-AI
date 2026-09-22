# AegisFin-AI Phase 2 — Step 8: Behavioral Dataset v2 Chronological Splits Report

**Document Version:** 1.0.0  
**Generated At:** 2026-09-21T04:21:00Z  
**Source Dataset:** [`data/behavioral/production_features_100k_v2.csv`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/production_features_100k_v2.csv)  
**Raw Transactions:** [`data/behavioral/raw_transactions_100k_v2.csv`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/raw_transactions_100k_v2.csv)  
**Output Splits Directory:** [`data/behavioral/splits_v2/`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/splits_v2/)  
**Split Metadata:** [`data/behavioral/splits_v2/split_metadata.json`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/splits_v2/split_metadata.json)  

---

## 1. Executive Summary

In accordance with AegisFin-AI Phase 2 — Step 8 requirements, the validated 100,000-row behavioral dataset `v2` has been partitioned into **5 strictly chronological splits** without random shuffling. The partitioning methodology matches the validated v1 architecture, isolating the final test set and separating training, validation, calibration, and policy decision spaces.

All 16 mandatory verification checks passed with zero errors, zero warnings, and zero data leakage. All original `v1` datasets and split files remain 100% untouched, verified by SHA-256 cryptographic hashes.

---

## 2. Chronological Split Summary

The dataset covers a 45.0-day period from `2026-01-01T00:00:02.926862+00:00` to `2026-02-14T23:59:08.245883+00:00`. The rows were sorted chronologically prior to splitting.

| Partition | Proportion | Row Count | Fraud Count | Legit Count | Fraud Rate | Start Timestamp (UTC) | End Timestamp (UTC) | Duration |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 60.0% | 60,000 | 4,577 | 55,423 | 7.6283% | 2026-01-01T00:00:02.926862 | 2026-01-28T05:18:19.404436 | 27.22 days |
| **VALIDATION** | 10.0% | 10,000 | 1,005 | 8,995 | 10.0500% | 2026-01-28T05:18:49.163519 | 2026-02-01T15:22:47.295308 | 4.42 days |
| **CALIBRATION** | 10.0% | 10,000 | 976 | 9,024 | 9.7600% | 2026-02-01T15:22:48.459352 | 2026-02-06T02:12:07.838265 | 4.45 days |
| **POLICY** | 10.0% | 10,000 | 857 | 9,143 | 8.5700% | 2026-02-06T02:12:50.525511 | 2026-02-10T12:40:40.879628 | 4.44 days |
| **TEST** | 10.0% | 10,000 | 823 | 9,177 | 8.2300% | 2026-02-10T12:40:51.648317 | 2026-02-14T23:59:08.245883 | 4.47 days |
| **TOTAL** | **100.0%** | **100,000** | **8,238** | **91,762** | **8.2380%** | **2026-01-01T00:00:02.926862** | **2026-02-14T23:59:08.245883** | **45.00 days** |

---

## 3. Strict Temporal Boundary Inequalities

Point-in-time boundaries strictly enforce temporal causality. The latest timestamp of partition $i$ is strictly strictly earlier than the earliest timestamp of partition $i+1$:

$$\max(T_{\text{TRAIN}}) < \min(T_{\text{VALIDATION}}) < \min(T_{\text{CALIBRATION}}) < \min(T_{\text{POLICY}}) < \min(T_{\text{TEST}})$$

- **Train $\to$ Validation Boundary:**
  - $\max(T_{\text{TRAIN}}) = \text{2026-01-28 05:18:19.404436+00:00}$
  - $\min(T_{\text{VALIDATION}}) = \text{2026-01-28 05:18:49.163519+00:00}$
  - **Temporal Gap:** $+29.759$ seconds (Strict inequality verified: $\max < \min$)
- **Validation $\to$ Calibration Boundary:**
  - $\max(T_{\text{VALIDATION}}) = \text{2026-02-01 15:22:47.295308+00:00}$
  - $\min(T_{\text{CALIBRATION}}) = \text{2026-02-01 15:22:48.459352+00:00}$
  - **Temporal Gap:** $+1.164$ seconds (Strict inequality verified: $\max < \min$)
- **Calibration $\to$ Policy Boundary:**
  - $\max(T_{\text{CALIBRATION}}) = \text{2026-02-06 02:12:07.838265+00:00}$
  - $\min(T_{\text{POLICY}}) = \text{2026-02-06 02:12:50.525511+00:00}$
  - **Temporal Gap:** $+42.687$ seconds (Strict inequality verified: $\max < \min$)
- **Policy $\to$ Test Boundary:**
  - $\max(T_{\text{POLICY}}) = \text{2026-02-10 12:40:40.879628+00:00}$
  - $\min(T_{\text{TEST}}) = \text{2026-02-10 12:40:51.648317+00:00}$
  - **Temporal Gap:** $+10.768$ seconds (Strict inequality verified: $\max < \min$)

---

## 4. Mandatory Validation Checks (16/16 Passed)

| # | Validation Item | Specification | Observed Result | Status |
| :-: | :--- | :--- | :--- | :-: |
| **1** | Total Row Count | Exactly 100,000 rows | 100,000 rows | **PASSED** |
| **2** | Train Row Count | Exactly 60,000 rows (60.0%) | 60,000 rows | **PASSED** |
| **3** | Validation Row Count | Exactly 10,000 rows (10.0%) | 10,000 rows | **PASSED** |
| **4** | Calibration Row Count | Exactly 10,000 rows (10.0%) | 10,000 rows | **PASSED** |
| **5** | Policy Row Count | Exactly 10,000 rows (10.0%) | 10,000 rows | **PASSED** |
| **6** | Final Test Row Count | Exactly 10,000 rows (10.0%) | 10,000 rows | **PASSED** |
| **7** | Disjoint Transaction IDs | Zero ID overlap between splits | 0 overlapping IDs across all pairs; 100,000 unique IDs | **PASSED** |
| **8** | Chronological Ordering | Strictly monotonic partitions | $\max(T_i) < \min(T_{i+1})$ for all adjacent splits | **PASSED** |
| **9** | Zero NaN Values | 0 NaN values across all features | 0 NaN detected across all 62 features in all splits | **PASSED** |
| **10** | Zero Inf Values | 0 Inf values across all features | 0 Inf detected across all 62 features in all splits | **PASSED** |
| **11** | Exact 62 Features | Exact 62 production feature names | Exactly 62 production features in contract order | **PASSED** |
| **12** | Binary Fraud Label | $\text{fraud\_label} \in \{0, 1\}$ | All labels are strictly 0 or 1 | **PASSED** |
| **13** | Both Classes Present | 0 and 1 in every split | Both classes present in Train, Val, Calib, Policy, Test | **PASSED** |
| **14** | No Scenario Columns | Zero scenario/metadata features | No `scenario` columns exist; `transaction_id` excluded | **PASSED** |
| **15** | v1 Splits Unmodified | Zero changes to v1 splits | All 6 files match pre-execution SHA-256 hashes | **PASSED** |
| **16** | Final Test Isolated | Untouched final test partition | Test set isolated; zero model fitting, calibration, or policy tuning | **PASSED** |

---

## 5. Cryptographic Verification of v1 Splits (Zero Modification)

To ensure strict regression safety, SHA-256 hashes of all files in `data/behavioral/splits/` were computed before and after the Step 8 execution:

| File Name | SHA-256 Checksum | Pre-Run Status | Post-Run Status |
| :--- | :--- | :---: | :---: |
| `calibration.csv` | `009ab87a9d38e8e6676fce95f22cc565af5f853a1ccea5b4af29565de1f50bf0` | Unmodified | **MATCH** |
| `policy.csv` | `35a2001f55078affd91856eda29e215a4a84387573ea6afcacda3858aa28ec74` | Unmodified | **MATCH** |
| `split_metadata.json` | `77ff37b98bd95c0541f1bac9e56e2694484de00757a8f0de3f251a1fab49d7dd` | Unmodified | **MATCH** |
| `test.csv` | `a4e370ffbfd77443ddd86a2334f42a2ad714b73ab0ecc4e847898c6e9019022e` | Unmodified | **MATCH** |
| `train.csv` | `61cca17da3a85f636c12e0a4e00df7f7ddf029397a9d591d8dddab8bec988773` | Unmodified | **MATCH** |
| `validation.csv` | `da3bf6f73b928c7e43f15cf3d9aca7ce0f47b11289f7d1777781a5f0f3dcde0f` | Unmodified | **MATCH** |

---

## 6. Schema & Feature Matrix Compliance

Each split CSV contains exactly **64 columns**:
1. `transaction_id` (String identifier, strictly excluded from the ML feature matrix $X$)
2. `fraud_label` (Binary target $y \in \{0, 1\}$)
3. Columns 3–64: Exactly the **62 validated production features** in standardized contract order:
   - `amount`, `product_code_freq`, `card_network_freq`, `card_type_freq`, `email_domain_freq`, `missing_fields_count`
   - `hour`, `weekday_index`, `is_weekend`, `is_night`, `day_of_year`, `time_since_midnight_sec`
   - `amount_log`, `amount_cents`, `is_round_amount`, `is_zero_cents`
   - `customer_tx_count_5m`, `customer_tx_count_1h`, `customer_tx_count_24h`, `customer_amount_mean`, `customer_amount_std`, `customer_amount_zscore`, `customer_amount_ratio`, `customer_is_new`, `customer_unique_merchants_24h`, `customer_unique_devices_24h`
   - `card_tx_count_5m`, `card_tx_count_1h`, `card_tx_count_24h`, `card_amount_mean`, `card_amount_std`, `card_amount_ratio`, `card_is_new`
   - `device_tx_count_5m`, `device_tx_count_1h`, `device_tx_count_24h`, `device_unique_customers_24h`, `device_unique_cards_24h`, `device_is_new`
   - `merchant_tx_count_1h`, `merchant_tx_count_24h`, `merchant_amount_mean`, `merchant_amount_std`, `customer_merchant_tx_count`, `customer_merchant_is_new`
   - `ip_tx_count_5m`, `ip_tx_count_1h`, `ip_tx_count_24h`, `ip_unique_customers_24h`, `ip_unique_cards_24h`, `ip_unique_devices_24h`, `ip_is_new`
   - `customer_device_seen_before`, `customer_card_seen_before`, `customer_merchant_seen_before`, `customer_ip_seen_before`, `card_device_seen_before`, `card_ip_seen_before`
   - `rapid_transaction_flag`, `transactions_last_15m`, `amount_sum_last_1h`, `amount_sum_last_24h`

---

## 7. Artifact Manifest

The following split files were generated in [`data/behavioral/splits_v2/`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/splits_v2/):

| File Path | Size | Description |
| :--- | :---: | :--- |
| [`data/behavioral/splits_v2/train.csv`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/splits_v2/train.csv) | 23.55 MB | 60,000 rows (4,577 fraud, 55,423 legit) |
| [`data/behavioral/splits_v2/validation.csv`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/splits_v2/validation.csv) | 4.20 MB | 10,000 rows (1,005 fraud, 8,995 legit) |
| [`data/behavioral/splits_v2/calibration.csv`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/splits_v2/calibration.csv) | 4.21 MB | 10,000 rows (976 fraud, 9,024 legit) |
| [`data/behavioral/splits_v2/policy.csv`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/splits_v2/policy.csv) | 4.22 MB | 10,000 rows (857 fraud, 9,143 legit) |
| [`data/behavioral/splits_v2/test.csv`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/splits_v2/test.csv) | 4.23 MB | 10,000 rows (823 fraud, 9,177 legit) — **UNTOUCHED & ISOLATED** |
| [`data/behavioral/splits_v2/split_metadata.json`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/splits_v2/split_metadata.json) | 4.44 KB | Comprehensive metadata and verification records |
| [`reports/behavioral_splits_v2_report.json`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/reports/behavioral_splits_v2_report.json) | 4.54 KB | Structured machine-readable report |
| [`reports/behavioral_splits_v2_report.md`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/reports/behavioral_splits_v2_report.md) | This Document | Human-readable audit report |

---

## 8. Next Step Readiness

The `v2` chronological splits are verified, validated, and ready for model training in Step 9:
- **`train.csv`**: Used exclusively to fit the new XGBoost baseline classifier.
- **`validation.csv`**: Used exclusively for early stopping, hyperparameter tuning, and initial validation diagnostics.
- **`calibration.csv`**: Strictly reserved for probability calibration (Platt / Isotonic).
- **`policy.csv`**: Strictly reserved for decision threshold and operating risk band optimization.
- **`test.csv`**: Strictly isolated and untouched until final deployment audit.
