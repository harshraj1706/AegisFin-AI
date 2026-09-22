from __future__ import annotations

import uuid
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import SUPABASE_SECRET_KEY
from app.supabase_client import get_supabase_admin_client
from app.fraud_feature_service import (
    FraudFeatureService,
    build_fraud_features,
    get_fraud_feature_service,
    update_entity_state,
    parse_timestamp,
)


@pytest.fixture(scope="module")
def feature_service() -> FraudFeatureService:
    return get_fraud_feature_service()


def test_service_initialization(feature_service: FraudFeatureService):
    """Verify that FraudFeatureService loads 459 features and preprocessor state."""
    assert len(feature_service.feature_names) == 459
    assert "preprocessor" in dir(feature_service)
    assert len(feature_service.frequency_maps) > 0
    assert len(feature_service.numeric_fill) > 0


# -----------------------------------------------------------------------------
# Test A: First transaction for a new customer
# -----------------------------------------------------------------------------
def test_a_first_transaction_for_new_customer(feature_service: FraudFeatureService):
    """
    A. First transaction for a new customer:
    Expected: past_count = 0, new_entity = true (uid_is_new = 1.0)
    """
    txn = {
        "customer_id": f"TEST_NEW_{uuid.uuid4().hex[:8]}",
        "amount": 250.0,
        "transaction_timestamp": "2026-09-20T10:00:00Z",
        "card_id": "9999",
        "address_id": "300.0",
        "product_code": "W",
        "card_network": "visa",
        "card_type": "debit",
    }

    df = feature_service.build_fraud_features(txn, history=[])
    assert df.shape == (1, 459)
    assert df["uid_past_count"].iloc[0] == 0.0
    assert df["uid_is_new"].iloc[0] == 1.0
    assert df["uid_past_mean_amt"].iloc[0] == 250.0
    assert df["uid_amount_ratio"].iloc[0] == 1.0
    assert df["uid_past_std_amt"].iloc[0] == 0.0


# -----------------------------------------------------------------------------
# Test B: Second transaction for same customer
# -----------------------------------------------------------------------------
def test_b_second_transaction_for_same_customer(feature_service: FraudFeatureService):
    """
    B. Second transaction for same customer:
    Expected: past_count increases, historical mean uses ONLY first transaction.
    """
    cust_id = f"TEST_REPEAT_{uuid.uuid4().hex[:8]}"

    txn1 = {
        "customer_id": cust_id,
        "amount": 500.0,
        "transaction_timestamp": "2026-09-20T10:00:00Z",
        "card_id": "9999",
        "address_id": "300.0",
    }

    txn2 = {
        "customer_id": cust_id,
        "amount": 1000.0,
        "transaction_timestamp": "2026-09-20T11:00:00Z",
        "card_id": "9999",
        "address_id": "300.0",
    }

    # Pass txn1 in history for txn2
    history = [txn1]
    df = feature_service.build_fraud_features(txn2, history=history)

    assert df["uid_past_count"].iloc[0] == 1.0
    assert df["uid_is_new"].iloc[0] == 0.0
    assert df["uid_past_mean_amt"].iloc[0] == 500.0
    assert df["uid_amount_ratio"].iloc[0] == pytest.approx(1000.0 / 500.0, rel=1e-3)


# -----------------------------------------------------------------------------
# Test C: Transaction at 12:00 must NOT see transaction at 13:00
# -----------------------------------------------------------------------------
def test_c_transaction_at_12_must_not_see_transaction_at_13(feature_service: FraudFeatureService):
    """
    C. Transaction at 12:00 must NOT see transaction at 13:00 (strict anti-leakage).
    """
    cust_id = f"TEST_TIMETRAVEL_{uuid.uuid4().hex[:8]}"

    future_txn = {
        "customer_id": cust_id,
        "amount": 9999.0,
        "transaction_timestamp": "2026-09-20T13:00:00Z",
    }

    current_txn = {
        "customer_id": cust_id,
        "amount": 150.0,
        "transaction_timestamp": "2026-09-20T12:00:00Z",
    }

    df = feature_service.build_fraud_features(current_txn, history=[future_txn])

    # Future transaction at 13:00 must NOT be in the history of transaction at 12:00
    assert df["uid_past_count"].iloc[0] == 0.0
    assert df["uid_is_new"].iloc[0] == 1.0
    assert df["uid_past_mean_amt"].iloc[0] == 150.0


# -----------------------------------------------------------------------------
# Test D: Current transaction must NOT be included in its own statistics
# -----------------------------------------------------------------------------
def test_d_current_transaction_not_included_in_own_statistics(feature_service: FraudFeatureService):
    """
    D. Current transaction must NOT be included in its own statistics.
    """
    cust_id = f"TEST_OWN_STATS_{uuid.uuid4().hex[:8]}"

    current_txn = {
        "customer_id": cust_id,
        "amount": 300.0,
        "transaction_timestamp": "2026-09-20T12:00:00Z",
    }

    # Even if history mistakenly includes the exact same current transaction
    identical_in_history = {
        "customer_id": cust_id,
        "amount": 300.0,
        "transaction_timestamp": "2026-09-20T12:00:00Z",
    }

    # Strict filter: timestamp < current_timestamp (>= is discarded)
    df = feature_service.build_fraud_features(current_txn, history=[identical_in_history])
    assert df["uid_past_count"].iloc[0] == 0.0
    assert df["uid_is_new"].iloc[0] == 1.0


# -----------------------------------------------------------------------------
# Test E: Missing optional fields
# -----------------------------------------------------------------------------
def test_e_missing_optional_fields(feature_service: FraudFeatureService):
    """
    E. Missing optional fields:
    Verifies that sparse/minimal live input successfully produces complete 459-feature vector.
    """
    sparse_txn = {
        "amount": 85.50,
        # missing customer_id, card_id, address_id, product_code, emails, device info, etc.
    }

    df = feature_service.build_fraud_features(sparse_txn, history=[])
    assert df.shape == (1, 459)
    assert not df.isna().any().any()
    assert not np.isinf(df.values).any()
    assert df["missing_count"].iloc[0] > 400.0


# -----------------------------------------------------------------------------
# Test F: Repeated requests produce identical results
# -----------------------------------------------------------------------------
def test_f_repeated_requests_consistency(feature_service: FraudFeatureService):
    """
    F. Repeated requests:
    Verifies deterministic execution.
    """
    txn = {
        "customer_id": "CUST_STABILITY_TEST",
        "amount": 120.0,
        "transaction_timestamp": "2026-09-20T14:30:00Z",
        "card_id": "1234",
        "address_id": "200.0",
        "product_code": "W",
    }

    df1 = feature_service.build_fraud_features(txn, history=[])
    df2 = feature_service.build_fraud_features(txn, history=[])

    pd.testing.assert_frame_equal(df1, df2)


# -----------------------------------------------------------------------------
# Test G: Exact feature column ordering
# -----------------------------------------------------------------------------
def test_g_exact_feature_column_ordering(feature_service: FraudFeatureService):
    """
    G. Exact feature column ordering matches the saved champion feature_names.
    """
    txn = {"amount": 42.0}
    df = feature_service.build_fraud_features(txn, history=[])

    assert list(df.columns) == feature_service.feature_names


# -----------------------------------------------------------------------------
# Test H: Exact feature count
# -----------------------------------------------------------------------------
def test_h_exact_feature_count(feature_service: FraudFeatureService):
    """
    H. Exact feature count: exactly 459 features.
    """
    txn = {"amount": 42.0}
    df = feature_service.build_fraud_features(txn, history=[])

    assert df.shape[1] == 459
    assert len(feature_service.feature_names) == 459


# -----------------------------------------------------------------------------
# Test I: No NaN/inf values where the model does not permit them
# -----------------------------------------------------------------------------
def test_i_no_nan_or_inf_values(feature_service: FraudFeatureService):
    """
    I. No NaN/inf values across all engineered features.
    """
    test_cases = [
        {"amount": 0.0},
        {"amount": 1000000.0},
        {"amount": 55.25, "customer_id": "UNKNOWN", "card_id": "invalid_num"},
        {"amount": 33.10, "transaction_timestamp": "2026-01-01T00:00:00Z"},
    ]

    for tc in test_cases:
        df = feature_service.build_fraud_features(tc, history=[])
        assert not df.isna().any().any(), f"NaN detected in test case: {tc}"
        assert not np.isinf(df.values).any(), f"Inf detected in test case: {tc}"


# -----------------------------------------------------------------------------
# Test J: Entity-state update correctness
# -----------------------------------------------------------------------------
def test_j_entity_state_update_correctness():
    """
    J. Entity-state update correctness:
    Verifies that update_entity_state safely inserts and updates fraud_entity_state.
    """
    admin = get_supabase_admin_client()
    test_cust_id = f"TEST_ES_{uuid.uuid4().hex[:8]}"

    txn = {
        "customer_id": test_cust_id,
        "amount": 200.0,
        "transaction_timestamp": "2026-09-20T10:00:00Z",
    }

    try:
        # 1. First update -> Insert record
        res1 = update_entity_state(txn)
        assert res1["status"] == "success"
        assert res1["entities_updated"] >= 1

        state1 = (
            admin.table("fraud_entity_state")
            .select("*")
            .eq("entity_type", "customer")
            .eq("entity_key", test_cust_id)
            .execute()
            .data
        )
        assert len(state1) == 1
        assert state1[0]["transaction_count"] == 1
        assert state1[0]["amount_sum"] == 200.0
        assert state1[0]["amount_sq_sum"] == 40000.0
        assert state1[0]["last_amount"] == 200.0

        # 2. Second update -> Update existing record
        txn2 = {
            "customer_id": test_cust_id,
            "amount": 300.0,
            "transaction_timestamp": "2026-09-20T11:00:00Z",
        }
        res2 = update_entity_state(txn2)
        assert res2["status"] == "success"

        state2 = (
            admin.table("fraud_entity_state")
            .select("*")
            .eq("entity_type", "customer")
            .eq("entity_key", test_cust_id)
            .execute()
            .data
        )
        assert len(state2) == 1
        assert state2[0]["transaction_count"] == 2
        assert state2[0]["amount_sum"] == 500.0
        assert state2[0]["amount_sq_sum"] == 40000.0 + 90000.0  # 130000.0
        assert state2[0]["last_amount"] == 300.0

    finally:
        # Cleanup
        admin.table("fraud_entity_state").delete().eq("entity_type", "customer").eq("entity_key", test_cust_id).execute()


# -----------------------------------------------------------------------------
# Debug Endpoint Test
# -----------------------------------------------------------------------------
def test_debug_endpoint_fraud_features_test():
    """Verify POST /api/v1/fraud/features/test returns valid schema status."""
    client = TestClient(app)

    payload = {
        "customer_id": "TEST_CUST_API",
        "amount": 750.50,
        "transaction_timestamp": "2026-09-20T12:00:00Z",
        "card_id": "9633",
        "address_id": "299.0",
        "product_code": "W",
        "card_network": "visa",
        "card_type": "debit",
    }

    response = client.post("/api/v1/fraud/features/test", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["feature_count"] == 62
    assert data["expected_feature_count"] == 62
    assert data["feature_schema_valid"] is True
    assert "sample_features" in data
    assert data["sample_features"]["amount"] == 750.50
    assert data["sample_features"]["is_weekend"] == 1.0  # 2026-09-20 is Sunday
    assert data["sample_features"]["hour"] == 12.0

    # Ensure secret key is never in the response
    assert SUPABASE_SECRET_KEY not in str(data)
