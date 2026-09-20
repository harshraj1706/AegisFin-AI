# AegisFin Phase 2: Fraud Feature Engineering Service Architecture

## 1. System Overview

The Phase 2 Fraud Feature Engine (`app/fraud_feature_service.py`) is responsible for transforming raw incoming transaction payloads and historical database states into a standardized **459-dimensional numeric feature vector** for fraud evaluation.

```
Incoming Transaction Payload
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                 FraudFeatureService                         │
│                                                             │
│ 1. Parse UTC Timestamp & Transaction Attributes             │
│ 2. Point-in-time Query: public.fraud_entity_state           │
│    (Anti-leakage: timestamp < current_timestamp)            │
│ 3. Compute 15 Historical Entity Aggregates                  │
│ 4. Derive Date/Time, Cents, Log & Interaction Frequencies   │
│ 5. Impute Missing IEEE-CIS Features using Saved Medians     │
│ 6. Validate (1, 459) Shape, Order, Zero NaNs/Infs          │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
459-Dimensional Feature DataFrame (Ready for Model Inference)
           │
           ▼ (Future Step: Predict Risk & Log Result)
           │
┌─────────────────────────────────────────────────────────────┐
│                 Post-Prediction Lifecycle                   │
│                                                             │
│ 1. Insert transaction into public.fraud_transactions        │
│ 2. Call update_entity_state() for customer, card, & addr    │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Strict Anti-Leakage Architecture

In production fraud systems, feature leakage occurs when a model feature observes data from the current transaction or future transactions. AegisFin guarantees absolute point-in-time isolation:

1. **Transaction Isolation**:
   The current transaction is **NEVER** inserted into `public.fraud_transactions` or `public.fraud_entity_state` before feature generation is complete.
2. **Temporal Query Filters**:
   When querying historical transactions or entity states, the service strictly enforces:
   ```sql
   WHERE transaction_timestamp < current_transaction_timestamp
   ```
   Transactions occurring at `transaction_timestamp >= current_transaction_timestamp` are unconditionally excluded.
3. **Point-in-Time Entity State Checks**:
   The `fraud_entity_state` table maintains a `last_seen` timestamp. If `last_seen < current_timestamp`, the service utilizes $O(1)$ pre-aggregated state (`transaction_count`, `amount_sum`, `amount_sq_sum`). If the entity state contains subsequent transactions (e.g. in out-of-order backfill processing), the engine falls back to a point-in-time query on `fraud_transactions`.

---

## 3. Entity History & Pre-Aggregation

To avoid scanning large transaction tables on every live API request, historical aggregates are cached in `public.fraud_entity_state`:

- **Entity Types**:
  - `customer`: Aggregates grouped by `customer_id`.
  - `card`: Aggregates grouped by `card_id` / `card1`.
  - `card_addr`: Aggregates grouped by composite `card_id` + `address_id`.

- **Maintained Quantities**:
  - `transaction_count`: Number of past transactions ($N$).
  - `amount_sum`: $\sum x_i$
  - `amount_sq_sum`: $\sum x_i^2$
  - `last_seen`: Timestamp of latest historical transaction.
  - `last_amount`: Amount of latest historical transaction.

- **Derived Statistical Moments**:
  - $\text{mean} = \frac{\sum x_i}{N}$
  - $\text{variance} = \max\left(0, \frac{\sum x_i^2}{N} - \text{mean}^2\right)$
  - $\text{std} = \sqrt{\text{variance}}$
  - $\text{amount\_ratio} = \frac{\text{current\_amount}}{\text{mean} + 10^{-5}}$

---

## 4. Reusing Saved Training Preprocessor Artifacts

The champion model artifact located at `backend/models/aegisfin_phase2_fraud_champion.pkl` contains the exact serialized preprocessor state created during model training:

- **`feature_names`**: 459 exact feature identifiers in ordered sequence.
- **`frequency_maps`**: Pre-calculated relative frequency values for categorical columns (`ProductCD`, `card4`, `card6`, `P_emaildomain`, `R_emaildomain`, `DeviceType`, `DeviceInfo`, interaction keys). Unseen categories map to `__MISSING__`.
- **`numeric_fill`**: Pre-calculated median values for 424 numeric features (including 339 V-features, 14 C-features, 15 D-features, and identity fields).

By reading these values directly from the artifact, the engine guarantees deterministic alignment with the training distribution without manual hardcoding.

---

## 5. Entity State Update Lifecycle

The function `update_entity_state(transaction)` is decoupled from `build_fraud_features`:

```python
def update_entity_state(transaction: Dict[str, Any]) -> Dict[str, Any]:
    # Updates public.fraud_entity_state AFTER fraud prediction is made
```

This ensures:
1. No mutation occurs during read-only feature extraction.
2. If feature generation or prediction fails, the database state remains unpolluted.
3. Asynchronous or queued database updates can be performed safely.

---

## 6. Developer Debug Endpoint

A testing endpoint is provided on the FastAPI backend for development and integration validation:

- **Path**: `POST /api/v1/fraud/features/test`
- **Request Body**:
  ```json
  {
    "customer_id": "CUST_9918",
    "amount": 450.00,
    "transaction_timestamp": "2026-09-20T12:00:00Z",
    "card_id": "9633",
    "address_id": "299.0",
    "product_code": "W",
    "card_network": "visa",
    "card_type": "debit"
  }
  ```
- **Response**:
  ```json
  {
    "feature_count": 459,
    "expected_feature_count": 459,
    "feature_schema_valid": true,
    "sample_features": {
      "TransactionAmt": 450.0,
      "hour": 12.0,
      "weekday_index": 6.0,
      "is_weekend": 1.0,
      "is_night": 0.0,
      "TransactionAmt_log": 6.111467,
      "TransactionAmt_cents": 0.0,
      "uid_past_count": 0.0,
      "uid_is_new": 1.0,
      "ProductCD__freq": 0.74,
      ...
    }
  }
  ```
- **Security**: Suppresses all secret keys and sensitive database connection strings.

---

## 7. Automated Test Suite

A comprehensive test suite is implemented in [`tests/test_fraud_feature_service.py`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/tests/test_fraud_feature_service.py):

| Test Case | Objective | Result |
| :--- | :--- | :--- |
| `test_a_first_transaction_for_new_customer` | Asserts `past_count == 0.0`, `is_new == 1.0`, `mean_amt == current_amount`. | **PASSED** |
| `test_b_second_transaction_for_same_customer` | Asserts `past_count == 1.0`, `is_new == 0.0`, `mean_amt == first_amount`. | **PASSED** |
| `test_c_transaction_at_12_must_not_see_transaction_at_13` | Asserts future transactions are strictly excluded from history. | **PASSED** |
| `test_d_current_transaction_not_included_in_own_statistics` | Asserts current transaction does not enter its own historical mean. | **PASSED** |
| `test_e_missing_optional_fields` | Verifies graceful fallback to training medians and default frequencies. | **PASSED** |
| `test_f_repeated_requests_consistency` | Asserts deterministic identical feature output for repeated runs. | **PASSED** |
| `test_g_exact_feature_column_ordering` | Asserts strict equality between generated column order and champion model schema. | **PASSED** |
| `test_h_exact_feature_count` | Asserts output shape is strictly `(1, 459)`. | **PASSED** |
| `test_i_no_nan_or_inf_values` | Asserts zero `NaN` and zero infinite values in feature matrix. | **PASSED** |
| `test_j_entity_state_update_correctness` | Asserts persistent updates to `fraud_entity_state` in Supabase. | **PASSED** |
| `test_debug_endpoint_fraud_features_test` | Tests FastAPI `POST /api/v1/fraud/features/test` integration. | **PASSED** |
