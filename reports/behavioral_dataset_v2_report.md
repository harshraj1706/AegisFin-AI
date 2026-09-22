# AegisFin Phase 2: Behavioral Dataset v2 Quality & Realism Report

**Generation Date:** 2026-09-21 04:16:30 UTC  
**Generator Engine:** `scripts/generate_behavioral_dataset.py` (Version 2.1.0)  
**Dataset Suffix:** `100k_v2`  
**Status:** Validated, Feature-Engine Aligned, Ready for Chronological Partitioning

---

## 1. Executive Summary & Dataset Overview

A new 100,000-row behavioral payment dataset has been generated to replace the overly-separable v1 dataset with realistic behavioral overlap, continuous monetary distributions, and natural Bayes error.

All 62 production features were generated strictly through the frozen production feature engine [app/production_feature_definitions.py](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/app/production_feature_definitions.py). No features or contracts were altered.

| Metric | v2 Dataset Value | Target / Reference |
|---|---|---|
| **Total Rows Generated** | **100,000** | 100,000 |
| **Legitimate Transactions** | **91,762 (91.76%)** | Proportional baseline |
| **Fraud Transactions** | **8,238 (8.24%)** | ~8.50% target fraud rate |
| **Production Features** | **Exactly 62 features** | `VALID_FEATURE_NAMES` (100% frozen) |
| **NaN / Inf Values** | **0 (Clean)** | 0 |
| **Chronological Ordering** | **Strictly Monotonic** | 2026-01-01 00:00:02 to 2026-02-14 23:59:08 UTC |
| **Duplicate IDs** | **0** | 0 |
| **Validation Scorecard** | **ALL 10 CHECKS PASSED** | [scripts/validate_behavioral_dataset.py](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/scripts/validate_behavioral_dataset.py) |

---

## 2. Realism Diagnostics & Behavioral Overlap (Items A - J)

| # | Behavioral Metric | v2 Value | v1 Baseline Comparison | Behavioral Significance |
|---|---|:---:|:---:|---|
| **A** | **Legit with `device_tx_count_1h > 0`** | **3.80%** (3,487 txs) | 0.88% (802 txs) | Household shared tablets and flash-sale bursts eliminate zero-velocity shortcuts |
| **B** | **Legit with `ip_tx_count_1h > 0`** | **5.60%** (5,137 txs) | 0.98% (894 txs) | Corporate NAT and campus Wi-Fi sharing introduce natural IP velocity overlap |
| **C** | **Legit with `customer_ip_seen_before = 0`** | **27.00%** (24,776 txs) | 14.50% (13,278 txs) | Traveling, mobile dynamic subnets, and office checkouts ensure IP novelty != fraud |
| **D** | **Legit with `customer_device_seen_before = 0`** | **19.82%** (18,183 txs) | 13.40% (12,274 txs) | New phones, computer upgrades, and browser resets ensure device novelty != fraud |
| **E** | **Legit with `amount < $5`** | **3.78%** (3,465 txs) | 0.00% (0 txs) | Micro-purchases ($0.99–$4.99) overlap directly with card testing |
| **F** | **Legit with `amount $250 - $800`** | **9.64%** (8,845 txs) | 0.00% (0 txs) | **Eliminated the artificial gap**: electronics, travel, and appliances now span $250–$800 |
| **G** | **Fraud with `device_tx_count_1h = 0`** | **31.51%** (2,596 txs) | 18.20% (1,547 txs) | Stealth ATO and low-and-slow card testing execute without velocity bursts |
| **H** | **Fraud with `ip_tx_count_1h = 0`** | **29.84%** (2,458 txs) | 18.20% (1,547 txs) | Rotating residential proxies prevent single-IP velocity rules from catching all fraud |
| **I** | **Fraud with `amount > $5`** | **81.38%** (6,704 txs) | 80.00% (6,800 txs) | Preserves full spectrum from micro-testing to major ATO attacks |
| **J** | **Fraud with `amount $250 - $800`** | **38.49%** (3,171 txs) | 38.80% (3,298 txs) | Fraud and legitimate transactions heavily share the intermediate amount band |

---

## 3. Telemetry Missingness & Entity Diversity

| Entity Type | Unique Count in v2 | Missingness / Nulls | Realistic Mechanism |
|---|:---:|:---:|---|
| **Customers** | **8,000** | 0 | Bank user accounts |
| **Cards** | **9,541** | 0 | Primary and secondary debit/credit cards |
| **Devices** | **15,166** | **4,820 nulls (4.82%)** | Incognito mode, ad-blockers, and Safari privacy masking |
| **Merchants** | **1,200** | 0 | Retail, digital, travel, restaurant pos |
| **IP Addresses** | **13,029** | 0 | Residential, mobile, corporate NAT, and proxy pools |

> [!NOTE]
> **No Telemetry Shortcut:** Missing `device_id` occurs in both legitimate transactions (**4.86%**) and fraud transactions (**4.38%**). The model cannot use missing telemetry as a label shortcut.

---

## 4. Scenario Breakdown (v2 Dataset)

| Scenario Name | Type | Transactions | Share | Mean Amount | Median Amount |
|---|---|:---:|:---:|:---:|:---:|
| **Legitimate Stable** | Legit | 88,262 | 88.26% | $196.39 | $80.50 |
| **Legitimate High-Value** | Legit | 3,500 | 3.50% | $2,009.63 | $1,996.97 |
| **Shared Device / IP Ring** | Fraud | 1,995 | 2.00% | $532.73 | $530.00 |
| **Card Testing** | Fraud | 1,699 | 1.70% | $2.97 | $2.00 |
| **Account Takeover** | Fraud | 1,544 | 1.54% | $1,004.98 | $743.36 |
| **Stolen Card** | Fraud | 1,500 | 1.50% | $994.89 | $999.13 |
| **Burst Fraud** | Fraud | 1,500 | 1.50% | $697.59 | $700.24 |

---

## 5. Artifact Paths & Safety Confirmations

- **Raw Transactions CSV:** [data/behavioral/raw_transactions_100k_v2.csv](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/raw_transactions_100k_v2.csv)
- **Production Features CSV:** [data/behavioral/production_features_100k_v2.csv](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/production_features_100k_v2.csv)
- **Dataset Metadata JSON:** [data/behavioral/behavioral_dataset_metadata_v2.json](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/data/behavioral/behavioral_dataset_metadata_v2.json)
- **Report Markdown:** [reports/behavioral_dataset_v2_report.md](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/reports/behavioral_dataset_v2_report.md)
- **Report JSON:** [reports/behavioral_dataset_v2_report.json](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/reports/behavioral_dataset_v2_report.json)

> [!IMPORTANT]
> **Strict Governance Confirmed:**
> 1. The original v1 dataset files (`raw_transactions_100k.csv`, `production_features_100k.csv`, `splits/`) are intact and unmodified.
> 2. No ML models were trained on v2.
> 3. No calibration, threshold tuning, or policy rules have been evaluated on v2.
> 4. All 62 production feature definitions in [app/production_feature_definitions.py](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/app/production_feature_definitions.py) remained 100% frozen.
