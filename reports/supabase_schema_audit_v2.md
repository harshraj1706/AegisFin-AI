# AegisFin-AI Phase 2 — Step 14.5: Supabase Schema Audit & Migration Review Report

> [!IMPORTANT]
> **Mandatory Governance Statement:**
> **"Audit of current Supabase database architecture, SQL schemas, and migration readiness for AegisFin-AI Phase 2 Fraud Detection. No live database modifications executed."**

---

## 1. Executive Summary

This final review completes **Step 14.5: Supabase Migration Review**, conducting a rigorous verification of the prepared PostgreSQL migration script against current FastAPI microservices, entity-state aggregation pipelines, and model inference services.

All Phase 2 ML artifacts remain strictly **FROZEN**:
- Base Model: `models/aegisfin_xgboost_baseline_v2.pkl`
- Calibrator: `models/aegisfin_probability_calibrator_v2.pkl`
- Risk Policy: `configs/fraud_risk_policy_v2.json`
- Feature Definitions: `app/production_feature_definitions.py`
- Production Contract: 62 numeric features, contract version `2.1.0`

### Core Audit Outcomes:
1. **Migration Safety Confirmed**: The migration file [`supabase/migrations/20260921000001_phase2_fraud_schema_update.sql`](file:///d:/AegisFin-AI_main_folder/AegisFin-AI-Phase1-Rich-Integration-Updated/AegisFin-AI-Phase1-Rich-Integration/supabase/migrations/20260921000001_phase2_fraud_schema_update.sql) contains **zero destructive statements** (`DROP TABLE`, `DROP COLUMN`, `TRUNCATE`, `DELETE`).
2. **Column-Count Inconsistency Resolved**: The previous audit reported "6 required columns" while proposing 7 additive columns. The correct count is verified as **7 additive columns** (`raw_fraud_probability`, `calibrated_fraud_probability`, `risk_band`, `recommended_action`, `feature_contract_version`, `calibrator_type`, `scored_at`).
3. **Probability Precision Verified**: Both probability columns utilize `NUMERIC(8, 7)`, safely and precisely accommodating probabilities in $[0.0, 1.0]$ with 7 decimal digits of precision ($1 \times 10^{-7}$ resolution), matching existing schema standards.
4. **Trigger Compatibility Clarified**: The compatibility trigger `trg_sync_fraud_predictions_compatibility` is recognized and documented as an **intentionally lossy compatibility projection** (mapping 4 canonical risk tiers into 3 legacy bands, and 4 canonical actions into 3 legacy decisions).
5. **Feature Snapshots Status**: **"Schema exists but application persistence is not yet integrated."** Table DDL and indexes are defined, including `model_version`, while router integration is deferred to runtime cutover.
6. **Persistence Flow Traced**: The current FastAPI router still routes requests through the legacy 459-feature IEEE-CIS champion service. The missing integration points for runtime cutover to the v2 62-feature pipeline are explicitly mapped.

---

## 2. Current Schema Inventory

The platform currently interacts with four core PostgreSQL tables in the `public` schema:

| Table Name | Role | Primary Key | Foreign Keys | RLS | Components Reading / Writing |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **`public.profiles`** | Analyst identity & roles | `id UUID` | `id -> auth.users(id)` ON DELETE CASCADE | Enabled | Read: `app/auth.py`<br>Written: `app/auth.py (signup)` |
| **`public.fraud_transactions`** | Raw transaction ingestion ledger | `id BIGSERIAL` | `transaction_id VARCHAR(64) UNIQUE` | Enabled | Read: `fraud_router.py`, `fraud_feature_service.py`<br>Written: `fraud_router.py` |
| **`public.fraud_predictions`** | Model scoring & decision logs | `id BIGSERIAL` | `transaction_id -> fraud_transactions` ON DELETE CASCADE | Enabled | Read: `fraud_router.py (/history, /high-risk)`<br>Written: `fraud_router.py (predict_fraud)` |
| **`public.fraud_entity_state`** | Anti-leakage entity running sums | `id BIGSERIAL` | `(entity_type, entity_key) UNIQUE` | Enabled | Read: `fraud_feature_service.py`<br>Written: `fraud_feature_service.py (update_entity_state)` |

---

## 3. Resolution of Column-Count Inconsistency

The previous audit summary stated: *"All 6 required Phase 2 columns present"*.
However, Section 2 of `20260921000001_phase2_fraud_schema_update.sql` actually defines **7 additive columns**:

```sql
-- 1. Uncalibrated raw probability
ALTER TABLE public.fraud_predictions ADD COLUMN IF NOT EXISTS raw_fraud_probability NUMERIC(8, 7);

-- 2. Calibrated posterior probability
ALTER TABLE public.fraud_predictions ADD COLUMN IF NOT EXISTS calibrated_fraud_probability NUMERIC(8, 7);

-- 3. Canonical 4-tier risk band
ALTER TABLE public.fraud_predictions ADD COLUMN IF NOT EXISTS risk_band VARCHAR(20);

-- 4. Canonical 4-tier operational action
ALTER TABLE public.fraud_predictions ADD COLUMN IF NOT EXISTS recommended_action VARCHAR(30);

-- 5. Production feature contract version
ALTER TABLE public.fraud_predictions ADD COLUMN IF NOT EXISTS feature_contract_version VARCHAR(20) DEFAULT '2.1.0';

-- 6. Calibrator type
ALTER TABLE public.fraud_predictions ADD COLUMN IF NOT EXISTS calibrator_type VARCHAR(30) DEFAULT 'sigmoid';

-- 7. Explicit scoring timestamp
ALTER TABLE public.fraud_predictions ADD COLUMN IF NOT EXISTS scored_at TIMESTAMPTZ DEFAULT now();
```

- **Root Cause**: The audit report grouped `calibrator_type` into general metadata rather than listing it in the primary column count.
- **Resolution**: The audit reports (`.md` and `.json`) and test suites (`scripts/verify_supabase_migration.py` and `tests/test_supabase_migration_v2.py`) have been updated to explicitly verify all **7 additive columns**.

---

## 4. Probability Precision & Constraint Verification

Both `raw_fraud_probability` and `calibrated_fraud_probability` use PostgreSQL `NUMERIC(8, 7)`:
- **Precision**: 8 total significant digits.
- **Scale**: 7 fractional digits after the decimal point.
- **Integer Digits**: $8 - 7 = 1$ digit.
- **Numerical Range**: $[-9.9999999, +9.9999999]$.
- **Unit Interval Compliance**:
  - `0.0000000`: Minimum value (valid).
  - `1.0000000`: Maximum value (1 integer digit, 7 fractional digits; valid without overflow).
  - Resolution: `0.0000001` ($10^{-7}$ or $0.00001\%$), ensuring sub-basis-point discrimination for high-confidence predictions.
- **Domain Constraints**:
  ```sql
  CHECK (calibrated_fraud_probability IS NULL OR (
      calibrated_fraud_probability >= 0.0 AND calibrated_fraud_probability <= 1.0
  ))
  CHECK (raw_fraud_probability IS NULL OR (
      raw_fraud_probability >= 0.0 AND raw_fraud_probability <= 1.0
  ))
  ```
  Both constraints correctly enforce mathematical probability bounds $[0, 1]$ while allowing `NULL` during pre-population states.

---

## 5. Legacy Trigger Compatibility: Intentionally Lossy Projection

> [!CAUTION]
> **Lossy Compatibility Disclosure:**
> The trigger `trg_sync_fraud_predictions_compatibility` is an **intentionally lossy compatibility projection**, NOT a lossless bijection. Because Phase 2 defines 4 risk tiers and 4 operational actions, projecting them into legacy 3-band and 3-decision columns inevitably collapses distinct operational states.

### 5.1 Canonical $\rightarrow$ Legacy Risk Band Mapping (Lossy Collapse)

| Canonical Phase 2 `risk_band` (4 Tiers) | Synchronized Legacy `fraud_band` (3 Tiers) | Information Status |
| :--- | :--- | :--- |
| **`LOW`** ($[0.00, 0.10)$) | **`LOW`** | Lossless (1-to-1) |
| **`MEDIUM`** ($[0.10, 0.40)$) | **`REVIEW`** | **Lossy Collapse**: Distinct MFA challenge tier collapsed with analyst review. |
| **`HIGH`** ($[0.40, 0.80)$) | **`REVIEW`** | **Lossy Collapse**: Collapsed into same legacy slot as `MEDIUM`. |
| **`CRITICAL`** ($[0.80, 1.00]$) | **`HIGH`** | **Semantics Shifted**: Canonical `CRITICAL` is mapped to legacy `HIGH`. |

*Result: Any consumer reading only legacy `fraud_band` cannot distinguish whether a transaction in `REVIEW` was `MEDIUM` ($0.10 \le p < 0.40$) or `HIGH` ($0.40 \le p < 0.80$).*

### 5.2 Canonical $\rightarrow$ Legacy Operational Action Mapping (Lossy Collapse)

| Canonical Phase 2 `recommended_action` (4 Actions) | Synchronized Legacy `decision` (3 Decisions) | Information Status |
| :--- | :--- | :--- |
| **`AUTO_APPROVE`** | **`ALLOW`** | Lossless (1-to-1) |
| **`STEP_UP_AUTH`** | **`MANUAL_REVIEW`** | **Lossy Collapse**: Automated step-up MFA challenge collapsed with human review. |
| **`MANUAL_REVIEW`** | **`MANUAL_REVIEW`** | **Lossy Collapse**: Collapsed into same legacy slot as `STEP_UP_AUTH`. |
| **`HARD_DECLINE`** | **`BLOCK`** | Lossless semantic equivalent. |

*Result: Any consumer reading only legacy `decision` seeing `MANUAL_REVIEW` cannot tell if the transaction required automated MFA (`STEP_UP_AUTH`) or human analyst queue routing (`MANUAL_REVIEW`).*

### 5.3 Legacy $\rightarrow$ Canonical Reverse Fallback Mapping
If legacy code writes to the legacy columns without providing canonical fields:
- `fraud_band = 'LOW'` $\rightarrow$ `risk_band = 'LOW'`
- `fraud_band = 'REVIEW'` $\rightarrow$ `risk_band = 'HIGH'` *(Ambiguous fallback: assumes `HIGH`, losing `MEDIUM`)*
- `fraud_band = 'HIGH'` $\rightarrow$ `risk_band = 'CRITICAL'`
- `decision = 'ALLOW'` $\rightarrow$ `recommended_action = 'AUTO_APPROVE'`
- `decision = 'MANUAL_REVIEW'` $\rightarrow$ `recommended_action = 'MANUAL_REVIEW'` *(Loses `STEP_UP_AUTH`)*
- `decision = 'BLOCK'` $\rightarrow$ `recommended_action = 'HARD_DECLINE'`

**Conclusion**: This lossy projection is necessary to keep legacy frontends (e.g., Streamlit views expecting `ALLOW`/`MANUAL_REVIEW`/`BLOCK` and `LOW`/`REVIEW`/`HIGH`) operational during transition. However, canonical Phase 2 analytics must read the canonical columns (`risk_band`, `recommended_action`) to access full 4-tier fidelity.

---

## 6. Feature Snapshots Integration Status

**Status Statement**:
> **"Schema exists but application persistence is not yet integrated."**

### 6.1 Database Schema Definition
The table definition in `20260921000001_phase2_fraud_schema_update.sql` provides all necessary audit attributes:
```sql
CREATE TABLE IF NOT EXISTS public.fraud_feature_snapshots (
    id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL UNIQUE REFERENCES public.fraud_transactions(transaction_id) ON DELETE CASCADE,
    feature_contract_version VARCHAR(20) NOT NULL DEFAULT '2.1.0',
    model_version VARCHAR(50) NOT NULL DEFAULT '2.1.0',
    feature_count INTEGER NOT NULL DEFAULT 62,
    features JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_fraud_feature_snapshots_txn
ON public.fraud_feature_snapshots(transaction_id);

CREATE INDEX IF NOT EXISTS idx_fraud_feature_snapshots_created_at
ON public.fraud_feature_snapshots(created_at DESC);
```

### 6.2 Application Integration Audit
- **Current State**: Neither `app/fraud_router.py` nor `app/fraud_feature_service.py` currently writes to `public.fraud_feature_snapshots`.
- **Reason**: The FastAPI router currently runs the legacy 459-feature IEEE-CIS pipeline. Feature snapshot logging will be integrated during the subsequent runtime cutover step when the 62-feature production engine is connected.

---

## 7. Prediction Persistence Flow & Missing Integration Points

Tracing the end-to-end transaction scoring lifecycle:

```
[1] Incoming Request (POST /api/v1/fraud/predict)
         │  (Handled by app/fraud_router.py)
         ▼
[2] Supabase Auth Validation
         │  (Handled by app/auth.py via Supabase GoTrue)
         ▼
[3] Duplicate Transaction Check
         │  (Handled by app/fraud_router.py -> fraud_transactions)
         ▼
[4] Anti-Leakage Point-in-Time State Query
         │  (Handled by app/fraud_feature_service.py -> fraud_entity_state)
         ▼
[5] Feature Vector Construction
         │  ⚠️ CURRENT: app/fraud_feature_service.py produces 459 legacy features
         │  🔧 MISSING INTEGRATION: Must cut over to app/production_feature_definitions.py (62 features)
         ▼
[6] Model Inference
         │  ⚠️ CURRENT: app/fraud_model_service.py loads legacy champion artifact
         │  🔧 MISSING INTEGRATION: Must cut over to models/aegisfin_xgboost_baseline_v2.pkl
         ▼
[7] Probability Calibration
         │  ⚠️ CURRENT: Handled by legacy champion artifact calibrator
         │  🔧 MISSING INTEGRATION: Must cut over to models/aegisfin_probability_calibrator_v2.pkl
         ▼
[8] Policy & Risk Band Assignment
         │  ⚠️ CURRENT: app/fraud_model_service.py evaluates legacy 3-tier thresholds
         │  🔧 MISSING INTEGRATION: Must cut over to configs/fraud_risk_policy_v2.json (4 bands, 4 actions)
         ▼
[9] Persistence: Raw Transaction
         │  (Handled by app/fraud_router.py -> fraud_transactions)
         ▼
[10] Persistence: Prediction Record
         │  ⚠️ CURRENT: app/fraud_router.py writes only legacy fields (fraud_probability, fraud_band, decision)
         │  🔧 MISSING INTEGRATION: Must explicitly supply raw_fraud_probability, calibrated_fraud_probability,
         │                           risk_band, recommended_action, feature_contract_version
         ▼
[11] Persistence: Entity State Aggregates
         │  (Handled by app/fraud_feature_service.py -> fraud_entity_state)
         ▼
[12] Persistence: Feature Snapshot
         │  ⚠️ CURRENT: NOT PERFORMED
         │  🔧 MISSING INTEGRATION: Router must insert 62-feature vector into fraud_feature_snapshots
```

---

## 8. Version Metadata Manifest

The database migration and application metadata preserve the genuine, existing repository versions:

| Version Key | Canonical Value | Source of Truth |
| :--- | :--- | :--- |
| **`model_name`** | `aegisfin_xgboost_baseline_v2` | `configs/fraud_risk_policy_v2.json` |
| **`model_version`** | `2.1.0` | `configs/fraud_risk_policy_v2.json` |
| **`calibration_version`** | `2.1.0` | `configs/fraud_risk_policy_v2.json` |
| **`calibrator_type`** | `sigmoid` | `models/aegisfin_probability_calibrator_v2.pkl` |
| **`policy_version`** | `2.0.0` | `configs/fraud_risk_policy_v2.json` |
| **`feature_contract_version`** | `2.1.0` | `docs/phase2_production_feature_schema.json` |
| **`feature_count`** | `62` | `app/production_feature_definitions.py` |

---

## 9. Verification & Safety Test Results

The migration script and verification suite were executed via automated static analysis and pytest:

1. **`scripts/verify_supabase_migration.py`**:
   - `[PASS]` Safety Check: Zero destructive statements (no DROP, TRUNCATE, or DELETE).
   - `[PASS]` Idempotency Check: All DDL statements employ safe `IF NOT EXISTS` clauses.
   - `[PASS]` Schema Contract: All 7 additive Phase 2 columns present.
   - `[PASS]` Domain Constraints: 4 risk bands and 4 operational actions properly constrained.
   - `[PASS]` Governance Table: Dedicated `public.fraud_feature_snapshots` defined with `model_version`.
   - `[PASS]` Backward Compatibility: Bidirectional synchronization trigger validated.
   - `[PASS]` Logic Simulation: Trigger mapping simulation verified as an intentionally lossy compatibility projection.
   - **Result**: **7 out of 7 checks PASSED**.

2. **`tests/test_supabase_migration_v2.py`**:
   - **7 passed in 0.03s**.

3. **Phase 2 ML Regression Test Suite**:
   - **27 passed in 3.03s** (splits, models, calibration, cross-seed, policy, held-out test).

---

## 10. Final Migration Assessment

| Assessment Dimension | Status | Notes |
| :--- | :---: | :--- |
| **Migration SQL Safety** | **SAFE** | 100% additive, non-destructive, zero DROPs. |
| **Column Count** | **7 COLUMNS** | Verified and reconciled across all reports and test suites. |
| **Legacy Trigger Compatibility** | **LOSSY PROJECTION** | Intentionally lossy to maintain 3-tier legacy frontend compatibility. |
| **Feature Snapshots Integration** | **PENDING RUNTIME CUTOVER** | Table DDL ready; application logging deferred to next step. |
| **Prediction Persistence Integration** | **PENDING RUNTIME CUTOVER** | Router currently executes legacy 459-feature pipeline. |
| **Row-Level Security (RLS)** | **ENFORCED** | Read-only for authenticated analysts; bypass for backend service role. |
| **Ready to Apply** | **READY** | Migration is safe to execute in Supabase upon operator authorization. |

---

*Step 14.5 is complete. In accordance with instructions, execution has stopped and no live database modifications were made.*
