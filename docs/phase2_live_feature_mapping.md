# AegisFin Phase 2: Live Transaction Feature Mapping & Schema Reference

## Overview

The AegisFin Phase 2 Fraud Detection champion model was trained on the IEEE-CIS Fraud Detection dataset and expects an exact feature vector of **459 columns** in a strict mathematical order.

In live production, incoming transaction payloads contain standard payment gateway and identity attributes. This document provides a transparent, comprehensive audit mapping showing how live transaction attributes, historical database records, and training-time preprocessor artifacts map to the **459 model input features**.

---

## High-Level Feature Composition Summary

| Feature Category | Feature Count | Description | Primary Data Source |
| :--- | :--- | :--- | :--- |
| **Historical Entity Features** | 15 | Customer, card, and card-address aggregate statistics strictly computed prior to transaction timestamp. | `fraud_entity_state` / `fraud_transactions` |
| **Live & Derived Features** | 29 | Raw transaction attributes, time-based indicators, log/cents transforms, missingness count, and interaction frequencies. | Live payload (`transaction`) |
| **Unavailable IEEE-CIS Features** | 415 | Anonymized V-features, C-counts, D-deltas, identity attributes, and non-collected match flags. | Preprocessor training medians & default frequencies |
| **Total Model Input Schema** | **459** | Complete feature vector passed to XGBoost champion artifact. | Reconstructed Feature Vector |

---

## 1. Historical Features (15 Features)

Historical features capture entity behavior over time. All historical aggregations adhere to strict **anti-leakage principles**: only transactions where `transaction_timestamp < current_transaction_timestamp` are considered.

| Feature Name | Type | Entity Scope | Source / Computation |
| :--- | :--- | :--- | :--- |
| `uid_past_count` | float | Customer (`customer_id`) | Total prior transactions completed by customer |
| `uid_past_mean_amt` | float | Customer (`customer_id`) | Mean transaction amount across prior transactions (defaults to current amount if new) |
| `uid_past_std_amt` | float | Customer (`customer_id`) | Standard deviation of prior amounts ($0.0$ if $N \le 1$) |
| `uid_amount_ratio` | float | Customer (`customer_id`) | $\frac{\text{current\_amount}}{\text{uid\_past\_mean\_amt} + 10^{-5}}$ ($1.0$ if new) |
| `uid_is_new` | float | Customer (`customer_id`) | $1.0$ if `uid_past_count == 0`, else $0.0$ |
| `card1_past_count` | float | Card (`card_id` / `card1`) | Total prior transactions on this card |
| `card1_past_mean_amt` | float | Card (`card_id` / `card1`) | Mean prior transaction amount on this card |
| `card1_past_std_amt` | float | Card (`card_id` / `card1`) | Standard deviation of prior amounts on this card |
| `card1_amount_ratio` | float | Card (`card_id` / `card1`) | $\frac{\text{current\_amount}}{\text{card1\_past\_mean\_amt} + 10^{-5}}$ ($1.0$ if new) |
| `card1_is_new` | float | Card (`card_id` / `card1`) | $1.0$ if `card1_past_count == 0`, else $0.0$ |
| `card1_addr1_past_count` | float | Combination (`card_id` + `address_id`) | Total prior transactions matching this card & billing address |
| `card1_addr1_past_mean_amt` | float | Combination (`card_id` + `address_id`) | Mean prior amount for card & address combination |
| `card1_addr1_past_std_amt` | float | Combination (`card_id` + `address_id`) | Standard deviation of prior amounts for card & address |
| `card1_addr1_amount_ratio` | float | Combination (`card_id` + `address_id`) | Ratio of current amount to historical combination mean |
| `card1_addr1_is_new` | float | Combination (`card_id` + `address_id`) | $1.0$ if combination not seen before, else $0.0$ |

---

## 2. Live & Derived Features (29 Features)

These features are extracted directly from the live payload or derived deterministically from transaction fields:

### A. Raw & Direct Numeric Features
- `TransactionAmt`: Raw transaction amount (USD).
- `TransactionDT`: Transaction timestamp represented in elapsed integer seconds.
- `card1`: Numeric identifier of payment card issuing bank (defaults to median $9633.0$).
- `addr1`: Numeric billing region / zip code (defaults to median $299.0$).
- `card2`, `card3`, `card5`, `addr2`, `dist1`, `dist2`: Optional live inputs, imputed via saved training medians when omitted.

### B. Derived Date, Time & Amount Features
- `hour`: Transaction hour in UTC ($0.0 - 23.0$).
- `weekday_index`: Day of the week ($0.0 = \text{Monday}, 6.0 = \text{Sunday}$).
- `is_weekend`: Indicator flag ($1.0$ if Saturday or Sunday, else $0.0$).
- `is_night`: Indicator flag ($1.0$ if transaction occurs before 06:00 UTC, else $0.0$).
- `day_index`: Day of the year ($1.0 - 366.0$).
- `TransactionAmt_log`: $\ln(1 + \text{TransactionAmt})$.
- `TransactionAmt_cents`: Fractional component of transaction amount ($\text{amount} \pmod 1$).
- `missing_count`: Count of features requiring imputation.

### C. Live Categorical Frequency Encoded Features
Categorical inputs are mapped to training frequencies using `preprocessor['frequency_maps']`:
- `ProductCD__freq`: Encodes product code (`W`, `H`, `C`, `S`, `R`).
- `card4__freq`: Encodes card network (`visa`, `mastercard`, `discover`, `american express`).
- `card6__freq`: Encodes card type (`debit`, `credit`).
- `P_emaildomain__freq`: Encodes purchaser email domain (e.g., `gmail.com`, `yahoo.com`).
- `R_emaildomain__freq`: Encodes recipient email domain.
- `DeviceType__freq`: Encodes client device type (`desktop`, `mobile`).
- `DeviceInfo__freq`: Encodes client device identifier string.
- `card1_addr1__freq`: Composite frequency for `card1` + `addr1`.
- `card1_email__freq`: Composite frequency for `card1` + `P_emaildomain`.
- `card1_ProductCD__freq`: Composite frequency for `card1` + `ProductCD`.
- `uid__freq`: Composite interaction key frequency.

---

## 3. Unavailable IEEE-CIS Features (415 Features)

The IEEE-CIS competition dataset contained 415 anonymized or unavailable features that are not provided in real-time retail transaction streams.

### Breakdown of Unavailable Features:
1. **V-Features (339 features: `V1` to `V339`)**:
   - Proprietary payment network counter features, velocity aggregations, and anonymized security signals engineered by Vesta.
2. **C-Features (14 features: `C1` to `C14`)**:
   - Anonymized count aggregations (e.g. how many addresses are associated with the card, etc.).
3. **D-Features (15 features: `D1` to `D15`)**:
   - Timedelta features (days since previous transaction, days since card registration, etc.).
4. **M-Features (9 features: `M1__freq` to `M9__freq`)**:
   - Match flags (e.g., name on card matches billing name, address match flags).
5. **Identity Features (38 features: `id_01` to `id_38`)**:
   - Browser device fingerprint attributes, screen resolutions, OS versions, IP flags.

### Strict Non-Fabrication Handling Strategy:
- **Do NOT hallucinate or synthesize live telemetry**: Fabricating fake velocity counters would introduce distribution shift and bias the XGBoost trees.
- **Replay Saved Preprocessor Medians**: For all numeric features, `FraudFeatureService` uses the exact median values calculated on the training partition and saved inside `models/aegisfin_phase2_fraud_champion.pkl` (`preprocessor['numeric_fill']`).
- **Replay Saved Frequency Map Defaults**: For categorical missing columns, the service maps to `__MISSING__` frequency weights calculated during training.
- **Result**: Guarantees zero NaN values, zero infinite values, and exact numerical alignment with the tree split thresholds learned by the champion model artifact.
