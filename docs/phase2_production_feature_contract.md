# AegisFin Phase 2: Production Feature Contract & Formal Validation Report

**Author:** AegisFin Advanced AI Engineering Team  
**Date:** September 2026  
**Status:** VALIDATED & FROZEN (Pre-Retraining Stage)  
**Associated Outputs:**
- [phase2_production_feature_validation.csv](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/docs/phase2_production_feature_validation.csv)
- [phase2_production_feature_schema.json](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/docs/phase2_production_feature_schema.json)
- [production_feature_definitions.py](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/app/production_feature_definitions.py)

---

## 1. Executive Summary & Audit Tally

To eliminate dependence on the 391 legacy anonymous benchmark features ($V_1$–$V_{339}$, $C_1$–$C_{14}$, $M_1$–$M_9$, $id_{01}$–$id_{29}$) without destabilizing the running production environment or modifying frozen artifacts, we audited **78 candidate features**. Each candidate feature was evaluated against the 14 live input fields in Streamlit/FastAPI and the historical tables in Supabase (`public.fraud_transactions`, `public.fraud_entity_state`).

Every candidate feature was assigned exactly one validation status:
- **`VALID`**: The feature can be computed with 100% mathematical and temporal parity during both historical training and live AegisFin inference.
- **`NEEDS_DATA`**: The feature represents a high-value fraud indicator, but requires schema extensions, frontend form additions, or third-party service subscriptions not currently active.
- **`INVALID`**: The feature was rejected due to data-type instability, alphanumeric casting fragility, or ambiguous multi-entity collinearity.

### Audit Summary Tally

| Classification | Count | % of Candidate Pool | Production Candidate Status |
|---|:---:|:---:|---|
| **VALID Features** | **62** | 79.49% | **Approved for Production Model Retraining** |
| **NEEDS_DATA Features** | **13** | 16.67% | **Deferred to Phase 2.5 Enrichment** |
| **INVALID Features** | **3** | 3.85% | **Permanently Rejected / Excluded** |
| **Total Features Audited** | **78** | **100.0%** | — |

```
Candidate Pool (78 Audited Features)
├── VALID Production Candidate Features (62)
│   ├── Direct Transaction Features (6)
│   ├── Time & Calendar Features (6)
│   ├── Amount Transformation Features (4)
│   ├── Customer Behavioral History (10)
│   ├── Card Behavioral History (7)
│   ├── Device Fingerprint History (6)
│   ├── Merchant Risk & History (6)
│   ├── Client IP Network History (7)
│   ├── Cross-Entity Relationship Features (6)
│   └── Velocity & Burst Features (4)
├── NEEDS_DATA Features (13)  ── [Deferred to Phase 2.5]
└── INVALID Features (3)      ── [Permanently Excluded]
```

---

## 2. Live Input & Database Source of Truth

### 2.1 The 14 Live Raw Inputs (Frontend & FastAPI)
Confirmed via [`app/schemas.py`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/app/schemas.py) and [`app/streamlit_fraud_view.py`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/app/streamlit_fraud_view.py):

| # | Field Name | Data Type | Required | Streamlit Widget | Description |
|:---:|:---|:---:|:---:|:---|:---|
| 1 | `transaction_id` | `str` | Yes | Text Input | Unique transaction identifier |
| 2 | `amount` | `float` | Yes | Number Input | Monetary amount in USD |
| 3 | `transaction_timestamp` | `str` | Optional | ISO-8601 UTC Text Input | Event timestamp (defaults to current UTC) |
| 4 | `customer_id` | `str` | Yes | Text Input | Customer or account ID |
| 5 | `card_id` | `str` | Yes | Text Input | Payment card identifier (maps to `card1`) |
| 6 | `device_id` | `str` | Optional | Text Input | Hardware fingerprint identifier |
| 7 | `merchant_id` | `str` | Optional | Text Input | Merchant / store identifier |
| 8 | `product_code` | `str` | Optional | Selectbox (`W,H,C,S,R`) | Transaction category |
| 9 | `card_network` | `str` | Optional | Selectbox (`visa,mastercard...`) | Card network scheme |
| 10 | `card_type` | `str` | Optional | Selectbox (`debit,credit`) | Card funding classification |
| 11 | `email_domain` | `str` | Optional | Text Input | Purchaser email domain |
| 12 | `address_id` | `str` | Optional | Text Input | Billing zip / location ID |
| 13 | `ip_address` | `str` | Optional | Text Input | Client IP address |
| 14 | `country` | `str` | Optional | Text Input | Two-letter ISO country code |

### 2.2 Supabase Historical Schema (`public.fraud_transactions` & `public.fraud_entity_state`)
Historical features are populated strictly from transactions recorded in `public.fraud_transactions` prior to the current transaction timestamp:
- **Columns Available in `public.fraud_transactions`**: `transaction_id`, `customer_id`, `card_id`, `device_id`, `merchant_id`, `amount`, `transaction_timestamp`, `email_domain`, `address_id`, `product_code`, `ip_address`, `country`, `created_at`.
- **Columns Missing from Supabase**: No confirmed fraud chargeback labels (`is_fraud_confirmed`), no gateway decline events (`gateway_decline_code`), no customer registration timestamp (`account_created_at`), and no multi-timescale 30-day rolling baselines.

---

## 3. Strict Anti-Leakage & Formula Equivalence Guarantees

### 3.1 Point-in-Time Temporal Constraint
To prevent data leakage, both training feature engineering and live inference feature engineering enforce the invariant:
$$\text{Filter: } t_{\text{historical}} < t_{\text{current}}$$
- The current transaction is **never** included in its own historical features.
- Any concurrent or future transactions ($t \ge t_{\text{curr}}$) in backfill logs or asynchronous event streams are strictly filtered out before computing count, mean, standard deviation, ratio, or z-score.

### 3.2 Lookback Window Definitions

| Lookback Window | Lower Bound Filter | Upper Bound Filter | Description |
|---|---|---|---|
| **5 Minutes (`5m`)** | $t_{\text{curr}} - 300\text{ seconds} \le t$ | $t < t_{\text{curr}}$ | Rapid bot burst window |
| **15 Minutes (`15m`)** | $t_{\text{curr}} - 900\text{ seconds} \le t$ | $t < t_{\text{curr}}$ | High-frequency cluster window |
| **1 Hour (`1h`)** | $t_{\text{curr}} - 3600\text{ seconds} \le t$ | $t < t_{\text{curr}}$ | Intraday velocity window |
| **24 Hours (`24h`)** | $t_{\text{curr}} - 86400\text{ seconds} \le t$ | $t < t_{\text{curr}}$ | Daily volume & diversity window |
| **Lifetime** | $-\infty < t$ | $t < t_{\text{curr}}$ | Historical baseline for entity |

### 3.3 Zero & Null Handling Protocol
1. **Division by Zero Protection**: All ratio and z-score denominators add $\epsilon = 10^{-5}$ ($1e-5$).
2. **Standard Deviation Variance Clamp**: $\text{var} = \max\left(0.0, \frac{1}{N}\sum(a_i - \mu)^2\right)$. If $N < 2$, standard deviation is defined as $0.0$.
3. **Z-Score Normalization**: If $N < 2$ or $\sigma == 0.0$, $Z = 0.0$.
4. **Amount Ratio Cold Start**: If entity count $N == 0$, $\text{ratio} = 1.0$.

---

## 4. Analysis of the 13 `NEEDS_DATA` Features

The following 13 features were identified as valuable, but cannot be calculated in the current AegisFin production stack without additional data sources, frontend fields, or external services:

| Feature Name | Functional Group | What is Missing? | How to Obtain It? | Add to Frontend or Backend? | Third-Party Source Required? |
|---|---|---|---|:---:|:---:|
| `device_risk_score` | Device Intelligence | Commercial device risk reputation score | Subscribe to FingerprintJS Pro or Sift Science Device API | Backend | **Yes** (FingerprintJS / Sift) |
| `merchant_fraud_rate_historical` | Merchant Risk | Confirmed chargeback and dispute feedback labels | Add `is_fraud_disputed` column to `fraud_transactions` with chargeback ingestion job | Backend / Supabase | **Yes** (Acquiring bank dispute file) |
| `ip_is_vpn_or_proxy` | IP Network | Autonomous System (ASN) and VPN/proxy detection flags | Query IPQualityScore or MaxMind GeoIP2 Precision Insights | Backend | **Yes** (IPQualityScore / MaxMind) |
| `ip_country_mismatch` | IP Network | IP geolocation resolution engine | Host MaxMind GeoIP2 City database locally or query MaxMind API | Backend | **Yes** (MaxMind GeoIP2) |
| `consecutive_declines_24h` | Velocity & Burst | Gateway decline event stream | Stream ISO 8583 decline webhooks from Stripe/Adyen/CyberSource | Backend / Supabase | **Yes** (Payment Gateway decline events) |
| `card_days_active` | Card History | Card token issuance timestamp | Fetch card created/tokenized timestamp from issuer API or tokenization service | Backend | **Yes** (Card Issuer / Gateway Token API) |
| `account_age_days` | Customer History | Customer registration timestamp | Join `customer_id` against `auth.users.created_at` or `profiles.created_at` | Backend / Supabase | No (Internal Supabase join) |
| `transaction_channel` | Direct Transaction | Channel enum (`WEB`, `MOBILE_APP`, `POS`, `API`) | Add `channel` dropdown widget to Streamlit and field to `FraudPredictionRequest` | Frontend + Backend API | No |
| `shipping_country` | Direct Transaction | Physical delivery destination country | Add `shipping_country` text input to Streamlit and field to `FraudPredictionRequest` | Frontend + Backend API | No |
| `browser_user_agent` | Device Intelligence | Client HTTP User-Agent header | Capture `request.headers.get('user-agent')` in FastAPI and parse via `ua-parser` | Backend | No |
| `card_velocity_surge` | Velocity & Burst | 30-day baseline rolling hourly velocity | Add scheduled rollup cron computing 30-day moving average velocity in Supabase | Backend / Supabase | No |
| `device_velocity_surge` | Velocity & Burst | 30-day baseline rolling device velocity | Add scheduled rollup cron computing 30-day moving average velocity in Supabase | Backend / Supabase | No |
| `country_freq` | Direct Transaction | Empirical country frequency distribution table | Generate global reference transaction volume table by ISO country code | Backend | No (Offline census/reference table) |

---

## 5. Analysis of the 3 `INVALID` Features

The following 3 features were rejected and permanently excluded from the production feature candidate set:

1. **`address_id_numeric`**:
   - *Reason for Rejection:* In AegisFin, `address_id` is an arbitrary string or alphanumeric postal identifier (e.g., `"ZIP_94103"`, `"NY"`, `"ADDR_88"`). Casting this directly to a numeric float (`float(address_id)`) raises a `ValueError` on non-numeric strings and induces arbitrary, non-physical ordinal splits in decision trees.
   - *Remedy:* Address associations are properly captured via cross-entity relational flags (`customer_card_seen_before`, `card_device_seen_before`).
2. **`transactions_last_5m`**:
   - *Reason for Rejection:* Defined ambiguously as `"COUNT(*) across customer/card in 5m"`. This conflates two distinct entities into an ill-defined union query that creates double-counting when a customer uses multiple cards or when a card is shared. Furthermore, it is directly collinear with `customer_tx_count_5m` and `card_tx_count_5m`.
   - *Remedy:* Excluded in favor of the clean, entity-specific features `customer_tx_count_5m` and `card_tx_count_5m`.
3. **`transactions_last_1h`**:
   - *Reason for Rejection:* Suffer from the same ambiguous multi-entity formulation as `transactions_last_5m` and is a redundant collinear duplicate of `customer_tx_count_1h`.
   - *Remedy:* Excluded in favor of `customer_tx_count_1h`.

---

## 6. The 62 Approved Production Features

The 62 valid production features are implemented in [`app/production_feature_definitions.py`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/app/production_feature_definitions.py) and partitioned across 10 functional groups:

1. **Direct Transaction Features (6)**: `amount`, `product_code_freq`, `card_network_freq`, `card_type_freq`, `email_domain_freq`, `missing_fields_count`
2. **Time & Calendar Features (6)**: `hour`, `weekday_index`, `is_weekend`, `is_night`, `day_of_year`, `time_since_midnight_sec`
3. **Amount Transformation Features (4)**: `amount_log`, `amount_cents`, `is_round_amount`, `is_zero_cents`
4. **Customer Behavioral History (10)**: `customer_tx_count_5m`, `customer_tx_count_1h`, `customer_tx_count_24h`, `customer_amount_mean`, `customer_amount_std`, `customer_amount_zscore`, `customer_amount_ratio`, `customer_is_new`, `customer_unique_merchants_24h`, `customer_unique_devices_24h`
5. **Card Behavioral History (7)**: `card_tx_count_5m`, `card_tx_count_1h`, `card_tx_count_24h`, `card_amount_mean`, `card_amount_std`, `card_amount_ratio`, `card_is_new`
6. **Device Fingerprint History (6)**: `device_tx_count_5m`, `device_tx_count_1h`, `device_tx_count_24h`, `device_unique_customers_24h`, `device_unique_cards_24h`, `device_is_new`
7. **Merchant Risk & History (6)**: `merchant_tx_count_1h`, `merchant_tx_count_24h`, `merchant_amount_mean`, `merchant_amount_std`, `customer_merchant_tx_count`, `customer_merchant_is_new`
8. **Client IP Network History (7)**: `ip_tx_count_5m`, `ip_tx_count_1h`, `ip_tx_count_24h`, `ip_unique_customers_24h`, `ip_unique_cards_24h`, `ip_unique_devices_24h`, `ip_is_new`
9. **Cross-Entity Relationship Features (6)**: `customer_device_seen_before`, `customer_card_seen_before`, `customer_merchant_seen_before`, `customer_ip_seen_before`, `card_device_seen_before`, `card_ip_seen_before`
10. **Velocity & Burst Features (4)**: `rapid_transaction_flag`, `transactions_last_15m`, `amount_sum_last_1h`, `amount_sum_last_24h`

**Total Approved Production Feature Vector Dimension: 62**
