-- =============================================================================
-- AegisFin-AI Phase 2: Supabase Schema Migration
-- Migration: 20260921000001_phase2_fraud_schema_update.sql
-- Description: Non-destructive, additive migration to align Supabase persistence
--              with the Phase 2 production feature contract, calibrated probabilities,
--              canonical risk bands, operational routing decisions, and audit snapshots.
-- Governance: SAFE / ADDITIVE ONLY - Preserves all existing data, RLS, and Phase 1 models.
-- =============================================================================

-- Enable UUID extension if not already present
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- -----------------------------------------------------------------------------
-- 1. BASELINE TABLE BOOTSTRAP (Idempotent: IF NOT EXISTS)
-- Ensures local development environments or fresh databases initialize cleanly
-- without impacting existing populated tables.
-- -----------------------------------------------------------------------------

-- 1.1 Profiles table (linked to auth.users)
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    full_name TEXT,
    role VARCHAR(50) DEFAULT 'analyst',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- 1.2 Raw Fraud Transactions table
CREATE TABLE IF NOT EXISTS public.fraud_transactions (
    id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL UNIQUE,
    customer_id VARCHAR(64) NOT NULL,
    card_id VARCHAR(64) NOT NULL,
    device_id VARCHAR(64),
    merchant_id VARCHAR(64),
    amount NUMERIC(12, 2) NOT NULL,
    transaction_timestamp TIMESTAMPTZ NOT NULL,
    email_domain VARCHAR(100),
    address_id VARCHAR(64),
    product_code VARCHAR(10) DEFAULT 'W',
    card_network VARCHAR(50),
    card_type VARCHAR(50),
    ip_address VARCHAR(45),
    country VARCHAR(10),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 1.3 Baseline Fraud Predictions table
CREATE TABLE IF NOT EXISTS public.fraud_predictions (
    id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL UNIQUE REFERENCES public.fraud_transactions(transaction_id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL DEFAULT 'XGBoost',
    model_version VARCHAR(50) NOT NULL DEFAULT 'phase2-xgb-v1',
    calibration_version VARCHAR(50) NOT NULL DEFAULT 'phase2-platt-v1',
    policy_version VARCHAR(50) NOT NULL DEFAULT 'phase2-policy-v1',
    fraud_probability NUMERIC(8, 7) NOT NULL,
    fraud_band VARCHAR(20) NOT NULL DEFAULT 'LOW',
    decision VARCHAR(30) NOT NULL DEFAULT 'ALLOW',
    prediction_latency_ms NUMERIC(8, 4),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 1.4 Historical Entity Aggregates table (Anti-Leakage State Engine)
CREATE TABLE IF NOT EXISTS public.fraud_entity_state (
    id BIGSERIAL PRIMARY KEY,
    entity_type VARCHAR(32) NOT NULL,
    entity_key VARCHAR(128) NOT NULL,
    transaction_count INTEGER NOT NULL DEFAULT 0,
    amount_sum NUMERIC(16, 2) NOT NULL DEFAULT 0.0,
    amount_sq_sum NUMERIC(24, 4) NOT NULL DEFAULT 0.0,
    first_seen TIMESTAMPTZ,
    last_seen TIMESTAMPTZ,
    last_amount NUMERIC(12, 2),
    unique_merchant_count INTEGER DEFAULT 0,
    unique_device_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT uq_entity_type_key UNIQUE (entity_type, entity_key)
);

-- -----------------------------------------------------------------------------
-- 2. PHASE 2 ADDITIVE SCHEMA EVOLUTION FOR public.fraud_predictions
-- Adds explicit probability separation, canonical 4-tier risk bands,
-- operational action routing, and formal version tracking.
-- Strictly non-destructive (ALTER TABLE ... ADD COLUMN IF NOT EXISTS).
-- -----------------------------------------------------------------------------

-- 2.1 Uncalibrated raw XGBoost probability
ALTER TABLE public.fraud_predictions
ADD COLUMN IF NOT EXISTS raw_fraud_probability NUMERIC(8, 7);

-- 2.2 Calibrated posterior fraud probability (canonical Phase 2 field)
ALTER TABLE public.fraud_predictions
ADD COLUMN IF NOT EXISTS calibrated_fraud_probability NUMERIC(8, 7);

-- 2.3 Canonical Phase 2 Risk Band ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
ALTER TABLE public.fraud_predictions
ADD COLUMN IF NOT EXISTS risk_band VARCHAR(20);

-- 2.4 Canonical Phase 2 Operational Decision ('AUTO_APPROVE', 'STEP_UP_AUTH', 'MANUAL_REVIEW', 'HARD_DECLINE')
ALTER TABLE public.fraud_predictions
ADD COLUMN IF NOT EXISTS recommended_action VARCHAR(30);

-- 2.5 Phase 2 62-feature production contract version
ALTER TABLE public.fraud_predictions
ADD COLUMN IF NOT EXISTS feature_contract_version VARCHAR(20) DEFAULT '2.1.0';

-- 2.6 Calibrator and base model metadata
ALTER TABLE public.fraud_predictions
ADD COLUMN IF NOT EXISTS calibrator_type VARCHAR(30) DEFAULT 'sigmoid';

-- 2.7 Scored timestamp (canonical audit timestamp)
ALTER TABLE public.fraud_predictions
ADD COLUMN IF NOT EXISTS scored_at TIMESTAMPTZ DEFAULT now();

-- -----------------------------------------------------------------------------
-- 3. DOMAIN INTEGRITY & CHECK CONSTRAINTS
-- Safe conditional addition of validation constraints.
-- -----------------------------------------------------------------------------

DO $$
BEGIN
    -- Risk Band domain constraint
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_fraud_predictions_risk_band'
    ) THEN
        ALTER TABLE public.fraud_predictions
        ADD CONSTRAINT chk_fraud_predictions_risk_band
        CHECK (risk_band IS NULL OR risk_band IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'));
    END IF;

    -- Operational Action domain constraint
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_fraud_predictions_recommended_action'
    ) THEN
        ALTER TABLE public.fraud_predictions
        ADD CONSTRAINT chk_fraud_predictions_recommended_action
        CHECK (recommended_action IS NULL OR recommended_action IN (
            'AUTO_APPROVE', 'STEP_UP_AUTH', 'MANUAL_REVIEW', 'HARD_DECLINE'
        ));
    END IF;

    -- Calibrated Probability range constraint
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_fraud_predictions_calibrated_prob_range'
    ) THEN
        ALTER TABLE public.fraud_predictions
        ADD CONSTRAINT chk_fraud_predictions_calibrated_prob_range
        CHECK (calibrated_fraud_probability IS NULL OR (
            calibrated_fraud_probability >= 0.0 AND calibrated_fraud_probability <= 1.0
        ));
    END IF;

    -- Raw Probability range constraint
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_fraud_predictions_raw_prob_range'
    ) THEN
        ALTER TABLE public.fraud_predictions
        ADD CONSTRAINT chk_fraud_predictions_raw_prob_range
        CHECK (raw_fraud_probability IS NULL OR (
            raw_fraud_probability >= 0.0 AND raw_fraud_probability <= 1.0
        ));
    END IF;
END $$;

-- -----------------------------------------------------------------------------
-- 4. TWO-WAY BACKWARD COMPATIBILITY SYNCHRONIZATION TRIGGER
-- Ensures that writes using either legacy Phase 1/Phase 2 fields
-- (fraud_probability, fraud_band, decision) or canonical Phase 2 fields
-- (calibrated_fraud_probability, risk_band, recommended_action)
-- are automatically bidirectionally populated without breaking frontend views.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION public.fn_sync_fraud_predictions_compatibility()
RETURNS TRIGGER AS $$
BEGIN
    -- 1. Sync Probabilities
    IF NEW.calibrated_fraud_probability IS NOT NULL AND NEW.fraud_probability IS NULL THEN
        NEW.fraud_probability := NEW.calibrated_fraud_probability;
    ELSIF NEW.fraud_probability IS NOT NULL AND NEW.calibrated_fraud_probability IS NULL THEN
        NEW.calibrated_fraud_probability := NEW.fraud_probability;
    END IF;

    -- 2. Sync Risk Bands
    -- Canonical -> Legacy mapping
    IF NEW.risk_band IS NOT NULL AND NEW.fraud_band IS NULL THEN
        IF NEW.risk_band = 'LOW' THEN
            NEW.fraud_band := 'LOW';
        ELSIF NEW.risk_band = 'MEDIUM' THEN
            NEW.fraud_band := 'REVIEW';
        ELSIF NEW.risk_band = 'HIGH' THEN
            NEW.fraud_band := 'REVIEW';
        ELSIF NEW.risk_band = 'CRITICAL' THEN
            NEW.fraud_band := 'HIGH';
        END IF;
    -- Legacy -> Canonical mapping
    ELSIF NEW.fraud_band IS NOT NULL AND NEW.risk_band IS NULL THEN
        IF NEW.fraud_band = 'LOW' THEN
            NEW.risk_band := 'LOW';
        ELSIF NEW.fraud_band = 'REVIEW' THEN
            NEW.risk_band := 'HIGH';
        ELSIF NEW.fraud_band = 'HIGH' THEN
            NEW.risk_band := 'CRITICAL';
        END IF;
    END IF;

    -- 3. Sync Operational Actions / Decisions
    -- Canonical -> Legacy mapping
    IF NEW.recommended_action IS NOT NULL AND NEW.decision IS NULL THEN
        IF NEW.recommended_action = 'AUTO_APPROVE' THEN
            NEW.decision := 'ALLOW';
        ELSIF NEW.recommended_action = 'STEP_UP_AUTH' THEN
            NEW.decision := 'MANUAL_REVIEW';
        ELSIF NEW.recommended_action = 'MANUAL_REVIEW' THEN
            NEW.decision := 'MANUAL_REVIEW';
        ELSIF NEW.recommended_action = 'HARD_DECLINE' THEN
            NEW.decision := 'BLOCK';
        END IF;
    -- Legacy -> Canonical mapping
    ELSIF NEW.decision IS NOT NULL AND NEW.recommended_action IS NULL THEN
        IF NEW.decision = 'ALLOW' THEN
            NEW.recommended_action := 'AUTO_APPROVE';
        ELSIF NEW.decision = 'MANUAL_REVIEW' THEN
            NEW.recommended_action := 'MANUAL_REVIEW';
        ELSIF NEW.decision = 'BLOCK' THEN
            NEW.recommended_action := 'HARD_DECLINE';
        END IF;
    END IF;

    -- 4. Sync Timestamp
    IF NEW.scored_at IS NULL THEN
        NEW.scored_at := COALESCE(NEW.created_at, now());
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_sync_fraud_predictions_compatibility ON public.fraud_predictions;
CREATE TRIGGER trg_sync_fraud_predictions_compatibility
BEFORE INSERT OR UPDATE ON public.fraud_predictions
FOR EACH ROW
EXECUTE FUNCTION public.fn_sync_fraud_predictions_compatibility();

-- -----------------------------------------------------------------------------
-- 5. OPTIONAL FEATURE SNAPSHOT TABLE: public.fraud_feature_snapshots
-- Regulatory & Governance Architecture (SR 11-7 / Model Governance)
-- Stores the 62 engineered feature values in compact JSONB format.
-- Prevents bloating the transactional fraud_predictions table while guaranteeing
-- reproducible model auditing, SHAP retrospectives, and model drift analysis.
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.fraud_feature_snapshots (
    id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(64) NOT NULL UNIQUE REFERENCES public.fraud_transactions(transaction_id) ON DELETE CASCADE,
    feature_contract_version VARCHAR(20) NOT NULL DEFAULT '2.1.0',
    model_version VARCHAR(50) NOT NULL DEFAULT '2.1.0',
    feature_count INTEGER NOT NULL DEFAULT 62,
    features JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- -----------------------------------------------------------------------------
-- 6. PERFORMANCE & QUERY OPTIMIZATION INDEXES
-- Creates compound and btree indexes to ensure sub-10ms query performance
-- for real-time history lookups, high-risk analyst dashboards, and entity aggregation.
-- -----------------------------------------------------------------------------

-- Predictions Indexes
CREATE INDEX IF NOT EXISTS idx_fraud_predictions_txn_id
ON public.fraud_predictions(transaction_id);

CREATE INDEX IF NOT EXISTS idx_fraud_predictions_risk_band
ON public.fraud_predictions(risk_band);

CREATE INDEX IF NOT EXISTS idx_fraud_predictions_action
ON public.fraud_predictions(recommended_action);

CREATE INDEX IF NOT EXISTS idx_fraud_predictions_scored_at
ON public.fraud_predictions(scored_at DESC);

CREATE INDEX IF NOT EXISTS idx_fraud_predictions_prob_desc
ON public.fraud_predictions(calibrated_fraud_probability DESC);

-- Transactions Historical Lookback Indexes (Anti-Leakage Query Optimization)
CREATE INDEX IF NOT EXISTS idx_fraud_transactions_customer_time
ON public.fraud_transactions(customer_id, transaction_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_fraud_transactions_card_time
ON public.fraud_transactions(card_id, transaction_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_fraud_transactions_device_time
ON public.fraud_transactions(device_id, transaction_timestamp DESC);

-- Entity State Lookup Index
CREATE INDEX IF NOT EXISTS idx_fraud_entity_state_lookup
ON public.fraud_entity_state(entity_type, entity_key);

-- Feature Snapshots Indexes
CREATE INDEX IF NOT EXISTS idx_fraud_feature_snapshots_txn
ON public.fraud_feature_snapshots(transaction_id);

CREATE INDEX IF NOT EXISTS idx_fraud_feature_snapshots_created_at
ON public.fraud_feature_snapshots(created_at DESC);

-- -----------------------------------------------------------------------------
-- 7. ROW LEVEL SECURITY (RLS) & ACCESS CONTROL
-- Enables Supabase RLS while allowing authenticated analyst reads
-- and server-side service role write operations.
-- -----------------------------------------------------------------------------

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.fraud_transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.fraud_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.fraud_entity_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.fraud_feature_snapshots ENABLE ROW LEVEL SECURITY;

-- 7.1 Profiles RLS
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE tablename = 'profiles' AND policyname = 'Allow authenticated users read all profiles'
    ) THEN
        CREATE POLICY "Allow authenticated users read all profiles"
        ON public.profiles FOR SELECT
        TO authenticated
        USING (true);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE tablename = 'profiles' AND policyname = 'Allow users update own profile'
    ) THEN
        CREATE POLICY "Allow users update own profile"
        ON public.profiles FOR UPDATE
        TO authenticated
        USING (auth.uid() = id);
    END IF;
END $$;

-- 7.2 Fraud Transactions RLS (Analysts can view; backend service role inserts/updates)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE tablename = 'fraud_transactions' AND policyname = 'Allow authenticated analysts read transactions'
    ) THEN
        CREATE POLICY "Allow authenticated analysts read transactions"
        ON public.fraud_transactions FOR SELECT
        TO authenticated
        USING (true);
    END IF;
END $$;

-- 7.3 Fraud Predictions RLS (Analysts can view history & high risk alerts)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE tablename = 'fraud_predictions' AND policyname = 'Allow authenticated analysts read predictions'
    ) THEN
        CREATE POLICY "Allow authenticated analysts read predictions"
        ON public.fraud_predictions FOR SELECT
        TO authenticated
        USING (true);
    END IF;
END $$;

-- 7.4 Feature Snapshots RLS
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE tablename = 'fraud_feature_snapshots' AND policyname = 'Allow authenticated analysts read snapshots'
    ) THEN
        CREATE POLICY "Allow authenticated analysts read snapshots"
        ON public.fraud_feature_snapshots FOR SELECT
        TO authenticated
        USING (true);
    END IF;
END $$;

-- =============================================================================
-- End of Migration: 20260921000001_phase2_fraud_schema_update.sql
-- =============================================================================
