# AegisFin Phase 2: Fraud Prediction API (`POST /api/v1/fraud/predict`)

## 1. Overview

The `POST /api/v1/fraud/predict` endpoint is the production API for Phase 2 real-time fraud risk assessment. It coordinates the full transaction lifecycle:

```
Incoming Request (Bearer Token + Transaction Payload)
                    │
                    ▼
      [Step 1] Authenticate User (Supabase Auth Bearer)
                    │
                    ▼
      [Step 2] Validate Request Schema (amount >= 0, IDs present)
                    │
                    ▼
      [Step 3] Duplicate Check (fraud_transactions.transaction_id)
                    │ (If duplicate: Return HTTP 409 Conflict)
                    ▼
      [Step 4] Temporal Anti-Leakage Feature Extraction
               (strictly WHERE transaction_timestamp < current_timestamp)
                    │
                    ▼
      [Step 5] Replay Preprocessor & Impute Missing IEEE-CIS Fields
               (zero NaNs, 459 ordered model-ready features)
                    │
                    ▼
      [Step 6] XGBoost Champion Model Inference
                    │
                    ▼
      [Step 7] Platt Probability Calibration
                    │
                    ▼
      [Step 8] Dynamic Policy Evaluation (LOW, REVIEW, HIGH)
                    │
                    ▼
      [Step 9] Database Persistence & State Update:
               1. Insert raw transaction -> public.fraud_transactions
               2. Insert prediction result -> public.fraud_predictions
               3. Update aggregates -> public.fraud_entity_state
                    │
                    ▼
      [Step 10] Return Response (HTTP 200)
```

---

## 2. Authentication

- **Scheme**: Bearer Token Authentication via Supabase Auth
- **Header**: `Authorization: Bearer <Supabase Access Token>`
- **Validation**: Token is cryptographically validated using Supabase Auth. The authenticated user ID is bound to audit logs.
- **Failures**:
  - Missing token: `HTTP 401 Unauthorized`
  - Invalid/expired token: `HTTP 401 Unauthorized`

---

## 3. Request Schema (`FraudPredictionRequest`)

```json
{
  "transaction_id": "TX_LIVE_90210",
  "amount": 1250.00,
  "transaction_timestamp": "2026-09-20T10:15:30Z",
  "customer_id": "CUST_98231",
  "card_id": "9633",
  "device_id": "DEV_FINGERPRINT_A9",
  "merchant_id": "MERCH_BESTBUY",
  "email_domain": "gmail.com",
  "address_id": "299.0",
  "product_code": "W",
  "card_network": "visa",
  "card_type": "debit",
  "ip_address": "192.168.1.100",
  "country": "US"
}
```

### Field Definitions:
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `transaction_id` | string | **Yes** | Unique transaction identifier. Must be unique across all transactions. |
| `amount` | float | **Yes** | Transaction amount in USD ($\ge 0.0$). Alias: `transaction_amount`. |
| `transaction_timestamp` | string | Optional | ISO-8601 UTC timestamp. Defaults to current UTC time if omitted. Alias: `timestamp`. |
| `customer_id` | string | **Yes** | Customer / account entity ID (maps to historical entity state). |
| `card_id` | string | **Yes** | Card number or card identifier (maps to `card1`). |
| `device_id` | string | Optional | Device fingerprint / hardware ID. |
| `merchant_id` | string | Optional | Merchant identifier. |
| `email_domain` | string | Optional | Purchaser email domain (maps to `P_emaildomain`). |
| `address_id` | string | Optional | Billing zip or location identifier (maps to `addr1`). |
| `product_code` | string | Optional | Product code (`W`, `H`, `C`, `S`, `R`). Defaults to `"W"`. |
| `card_network` | string | Optional | Card network (`visa`, `mastercard`, etc.). |
| `card_type` | string | Optional | Card type (`debit`, `credit`). |
| `ip_address` | string | Optional | Client IP address. |
| `country` | string | Optional | Two-letter ISO country code. |

---

## 4. Response Schema (`FraudPredictionResponse`)

```json
{
  "transaction_id": "TX_LIVE_90210",
  "fraud_probability": 0.2384,
  "fraud_band": "LOW",
  "decision": "ALLOW",
  "model_name": "XGBoost",
  "model_version": "phase2-xgb-v1",
  "calibration_version": "phase2-platt-v1",
  "policy_version": "phase2-policy-v1",
  "prediction_latency_ms": 25.412
}
```

### Field Definitions:
| Field | Type | Description |
| :--- | :--- | :--- |
| `transaction_id` | string | Confirmed transaction ID. |
| `fraud_probability` | float | Calibrated positive-class probability of fraud ($0.0 \le p \le 1.0$). |
| `fraud_band` | string | Policy risk band: `"LOW"`, `"REVIEW"`, or `"HIGH"`. |
| `decision` | string | Automated decision: `"ALLOW"`, `"MANUAL_REVIEW"`, or `"BLOCK"`. |
| `model_name` | string | Model family (`"XGBoost"`). |
| `model_version` | string | Trained model version (`"phase2-xgb-v1"`). |
| `calibration_version` | string | Calibrator version (`"phase2-platt-v1"`). |
| `policy_version` | string | Risk policy threshold version (`"phase2-policy-v1"`). |
| `prediction_latency_ms` | float | Inference and calibration latency in milliseconds. |

---

## 5. Duplicate Transaction Handling

`transaction_id` is unique. If a transaction with the same `transaction_id` is submitted again:
- The API intercepts the submission before feature creation or inference.
- It returns **`HTTP 409 Conflict`**:
  ```json
  {
    "detail": "Transaction 'TX_LIVE_90210' has already been processed."
  }
  ```
- No duplicate records are written to `fraud_transactions` or `fraud_predictions`.

---

## 6. Database Storage Flow

All database writes use the server-side Supabase client (`get_supabase_admin_client()`):
1. **`public.fraud_transactions`**:
   Stores raw transaction inputs (`transaction_id`, `customer_id`, `card_id`, `device_id`, `merchant_id`, `amount`, `transaction_timestamp`, `email_domain`, `address_id`, `product_code`, `ip_address`, `country`).
2. **`public.fraud_predictions`**:
   Stores model outputs (`transaction_id`, `model_name`, `model_version`, `calibration_version`, `policy_version`, `fraud_probability`, `fraud_band`, `decision`, `prediction_latency_ms`).
3. **`public.fraud_entity_state`**:
   Maintains running historical aggregates for customer, card, and card+address entities (`transaction_count`, `amount_sum`, `amount_sq_sum`, `last_seen`, `last_amount`).

---

## 7. Example cURL Request

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/fraud/predict" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_SUPABASE_ACCESS_TOKEN" \
  -d '{
    "transaction_id": "TXN_PROD_1001",
    "amount": 850.00,
    "transaction_timestamp": "2026-09-20T10:00:00Z",
    "customer_id": "CUST_501",
    "card_id": "9633",
    "address_id": "299.0",
    "product_code": "W",
    "email_domain": "gmail.com",
    "card_network": "visa",
    "card_type": "debit"
  }'
```

### Example 200 OK Response:
```json
{
  "transaction_id": "TXN_PROD_1001",
  "fraud_probability": 0.2359,
  "fraud_band": "LOW",
  "decision": "ALLOW",
  "model_name": "XGBoost",
  "model_version": "phase2-xgb-v1",
  "calibration_version": "phase2-platt-v1",
  "policy_version": "phase2-policy-v1",
  "prediction_latency_ms": 26.14
}
```
