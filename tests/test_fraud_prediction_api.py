from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Generator

import pytest
from fastapi.testclient import TestClient

from app.auth import login_user, signup_user
from app.config import SUPABASE_SECRET_KEY
from app.main import app
from app.supabase_client import get_supabase_admin_client


@pytest.fixture(scope="module")
def auth_headers() -> Generator[Dict[str, str], None, None]:
    """Creates a temporary test user, logs in, and yields valid Bearer Authorization headers."""
    unique_id = uuid.uuid4().hex[:8]
    email = f"fraud_api_test_{unique_id}@gmail.com"
    password = f"AegisFinPass_{unique_id}!123"
    full_name = f"Fraud Test Analyst {unique_id}"

    # Sign up
    signup_res = signup_user(email=email, password=password, full_name=full_name)
    token = signup_res.get("access_token")

    # If signup didn't return access_token directly, log in
    if not token:
        login_res = login_user(email=email, password=password)
        token = login_res.get("access_token")

    user_id = signup_res.get("user_id")
    headers = {"Authorization": f"Bearer {token}"}

    yield headers

    # Teardown: purge user from auth and profiles
    admin = get_supabase_admin_client()
    try:
        if user_id:
            admin.table("profiles").delete().eq("id", user_id).execute()
            admin.auth.admin.delete_user(user_id)
    except Exception:
        pass


@pytest.fixture
def cleanup_transactions():
    """Tracks created test transaction IDs and purges them from Supabase in teardown."""
    created_txns: list[str] = []
    created_entities: list[tuple[str, str]] = []

    yield {"txns": created_txns, "entities": created_entities}

    admin = get_supabase_admin_client()
    for txn_id in created_txns:
        try:
            admin.table("fraud_predictions").delete().eq("transaction_id", txn_id).execute()
        except Exception:
            pass
        try:
            admin.table("fraud_transactions").delete().eq("transaction_id", txn_id).execute()
        except Exception:
            pass

    for etype, ekey in created_entities:
        try:
            admin.table("fraud_entity_state").delete().eq("entity_type", etype).eq("entity_key", ekey).execute()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# 1. Valid Authenticated Prediction (Requirement 12.1, 12.6, 12.7)
# -----------------------------------------------------------------------------
def test_valid_authenticated_prediction(auth_headers, cleanup_transactions):
    """Verify successful end-to-end prediction with authenticated user."""
    client = TestClient(app)
    txn_id = f"TEST_API_{uuid.uuid4().hex[:8]}"
    cust_id = f"CUST_{uuid.uuid4().hex[:8]}"
    cleanup_transactions["txns"].append(txn_id)
    cleanup_transactions["entities"].append(("customer", cust_id))

    payload = {
        "transaction_id": txn_id,
        "amount": 750.50,
        "transaction_timestamp": "2026-09-20T10:00:00Z",
        "customer_id": cust_id,
        "card_id": "9633",
        "device_id": "DEV_MOBILE_01",
        "merchant_id": "MERCH_AMAZON",
        "email_domain": "gmail.com",
        "address_id": "299.0",
        "product_code": "W",
        "card_network": "visa",
        "card_type": "debit",
    }

    response = client.post("/api/v1/fraud/predict", json=payload, headers=auth_headers)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert data["transaction_id"] == txn_id
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["fraud_band"] in {"LOW", "REVIEW", "HIGH"}
    assert data["model_name"] == "XGBoost"
    assert data["model_version"] == "2.1.0"
    assert data["calibration_version"] == "2.1.0"
    assert data["policy_version"] == "2.0.0"
    assert data["feature_contract_version"] == "2.1.0"
    assert data["prediction_latency_ms"] > 0.0

    # Canonical Phase 2 fields
    assert "raw_fraud_probability" in data
    assert "calibrated_fraud_probability" in data
    assert data["risk_band"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert data["recommended_action"] in {"AUTO_APPROVE", "STEP_UP_AUTH", "MANUAL_REVIEW", "HARD_DECLINE"}

    # Ensure SUPABASE_SECRET_KEY is never leaked
    assert SUPABASE_SECRET_KEY not in str(data)


# -----------------------------------------------------------------------------
# 2. Missing Authentication (Requirement 12.2)
# -----------------------------------------------------------------------------
def test_missing_authentication():
    """Verify endpoint rejects requests without Authorization header (HTTP 401)."""
    client = TestClient(app)
    payload = {
        "transaction_id": "TEST_NO_AUTH",
        "amount": 100.0,
        "customer_id": "CUST_1",
        "card_id": "9633",
    }
    response = client.post("/api/v1/fraud/predict", json=payload)
    assert response.status_code == 401
    assert "Missing authentication credentials" in response.json().get("detail", "")


# -----------------------------------------------------------------------------
# 3. Invalid Authentication (Requirement 12.3)
# -----------------------------------------------------------------------------
def test_invalid_authentication():
    """Verify endpoint rejects invalid/forged bearer token (HTTP 401)."""
    client = TestClient(app)
    payload = {
        "transaction_id": "TEST_BAD_AUTH",
        "amount": 100.0,
        "customer_id": "CUST_1",
        "card_id": "9633",
    }
    headers = {"Authorization": "Bearer invalid_forged_token_xyz"}
    response = client.post("/api/v1/fraud/predict", json=payload, headers=headers)
    assert response.status_code == 401


# -----------------------------------------------------------------------------
# 4. Invalid Request Body (Requirement 12.4)
# -----------------------------------------------------------------------------
def test_invalid_request_body(auth_headers):
    """Verify negative amounts and missing required identifiers fail schema validation."""
    client = TestClient(app)

    # Negative amount
    res_neg = client.post(
        "/api/v1/fraud/predict",
        json={"transaction_id": "T1", "amount": -10.0, "customer_id": "C1", "card_id": "9633"},
        headers=auth_headers,
    )
    assert res_neg.status_code == 422

    # Missing transaction_id
    res_missing_id = client.post(
        "/api/v1/fraud/predict",
        json={"amount": 100.0, "customer_id": "C1", "card_id": "9633"},
        headers=auth_headers,
    )
    assert res_missing_id.status_code == 422


# -----------------------------------------------------------------------------
# 5. Duplicate Transaction Submission (Requirement 12.5)
# -----------------------------------------------------------------------------
def test_duplicate_transaction_handling(auth_headers, cleanup_transactions):
    """Verify submitting the same transaction_id twice raises HTTP 409 Conflict."""
    client = TestClient(app)
    txn_id = f"TEST_DUP_{uuid.uuid4().hex[:8]}"
    cust_id = f"CUST_DUP_{uuid.uuid4().hex[:8]}"
    cleanup_transactions["txns"].append(txn_id)
    cleanup_transactions["entities"].append(("customer", cust_id))

    payload = {
        "transaction_id": txn_id,
        "amount": 200.0,
        "customer_id": cust_id,
        "card_id": "9633",
    }

    # 1. First submission -> 200 OK
    res1 = client.post("/api/v1/fraud/predict", json=payload, headers=auth_headers)
    assert res1.status_code == 200

    # 2. Second submission with same transaction_id -> 409 Conflict
    res2 = client.post("/api/v1/fraud/predict", json=payload, headers=auth_headers)
    assert res2.status_code == 409
    assert "already been processed" in res2.json().get("detail", "")


# -----------------------------------------------------------------------------
# 6. Database Storage & Entity State Updates (Requirement 12.8, 12.9, 12.10)
# -----------------------------------------------------------------------------
def test_database_persistence_and_entity_updates(auth_headers, cleanup_transactions):
    """Verify row created in fraud_transactions, fraud_predictions, and fraud_entity_state."""
    client = TestClient(app)
    admin = get_supabase_admin_client()

    txn_id = f"TEST_PERSIST_{uuid.uuid4().hex[:8]}"
    cust_id = f"CUST_PERSIST_{uuid.uuid4().hex[:8]}"
    cleanup_transactions["txns"].append(txn_id)
    cleanup_transactions["entities"].append(("customer", cust_id))

    payload = {
        "transaction_id": txn_id,
        "amount": 450.00,
        "transaction_timestamp": "2026-09-20T10:00:00Z",
        "customer_id": cust_id,
        "card_id": "9633",
        "device_id": "DEV_TEST_99",
        "merchant_id": "MERCH_TEST",
        "email_domain": "outlook.com",
        "address_id": "299.0",
        "product_code": "W",
    }

    # Execute prediction
    res = client.post("/api/v1/fraud/predict", json=payload, headers=auth_headers)
    assert res.status_code == 200
    pred_data = res.json()

    # Verify fraud_transactions row
    tx_db = admin.table("fraud_transactions").select("*").eq("transaction_id", txn_id).execute().data
    assert len(tx_db) == 1
    assert tx_db[0]["transaction_id"] == txn_id
    assert tx_db[0]["customer_id"] == cust_id
    assert float(tx_db[0]["amount"]) == 450.00
    assert tx_db[0]["device_id"] == "DEV_TEST_99"

    # Verify fraud_predictions row
    pred_db = admin.table("fraud_predictions").select("*").eq("transaction_id", txn_id).execute().data
    assert len(pred_db) == 1
    assert pred_db[0]["transaction_id"] == txn_id
    assert pred_db[0]["model_name"] == "XGBoost"
    assert pred_db[0]["fraud_band"] == pred_data["fraud_band"]
    assert pred_db[0]["decision"] == pred_data["decision"]
    assert float(pred_db[0]["fraud_probability"]) == pytest.approx(pred_data["fraud_probability"], rel=1e-4)

    # Verify fraud_entity_state row updated
    entity_db = (
        admin.table("fraud_entity_state")
        .select("*")
        .eq("entity_type", "customer")
        .eq("entity_key", cust_id)
        .execute()
        .data
    )
    assert len(entity_db) >= 1
    assert entity_db[0]["transaction_count"] >= 1
    assert float(entity_db[0]["amount_sum"]) >= 450.00


# -----------------------------------------------------------------------------
# 7. Anti-Leakage: Current Transaction Excluded from History (Requirement 12.11)
# -----------------------------------------------------------------------------
def test_anti_leakage_current_transaction_excluded(auth_headers, cleanup_transactions):
    """
    Verify that a brand new customer's first transaction has uid_is_new=1.0 and
    uid_past_count=0.0 during feature calculation (current txn not yet in history).
    """
    client = TestClient(app)
    txn_id = f"TEST_LEAK_PREV_{uuid.uuid4().hex[:8]}"
    cust_id = f"CUST_FRESH_{uuid.uuid4().hex[:8]}"
    cleanup_transactions["txns"].append(txn_id)
    cleanup_transactions["entities"].append(("customer", cust_id))

    payload = {
        "transaction_id": txn_id,
        "amount": 999.00,
        "transaction_timestamp": "2026-09-20T10:00:00Z",
        "customer_id": cust_id,
        "card_id": "9633",
        "address_id": "299.0",
        "product_code": "W",
    }

    res = client.post("/api/v1/fraud/predict", json=payload, headers=auth_headers)
    assert res.status_code == 200

    # Test debug endpoint to confirm how feature engine evaluates brand new customer
    feat_res = client.post("/api/v1/fraud/features/test", json=payload)
    assert feat_res.status_code == 200
    features = feat_res.json()["features"]
    assert features["customer_is_new"] == 1.0
    assert features["customer_tx_count_5m"] == 0.0
    assert features["customer_tx_count_24h"] == 0.0


# -----------------------------------------------------------------------------
# 8. Phase 1 Endpoints Untouched (Requirement 12.12)
# -----------------------------------------------------------------------------
def test_phase1_endpoints_regression():
    """Verify Phase 1 /api/v1/predict and /health endpoints remain fully functional."""
    client = TestClient(app)

    # 1. Health check reports Phase 1 info AND fraud_model_loaded
    res_health = client.get("/health")
    assert res_health.status_code == 200
    h_data = res_health.json()
    assert h_data["status"] == "ok"
    assert h_data["model_name"] == "XGBoost"
    assert h_data["model_version"] == "credit-xgb-v1.0.0"
    assert h_data["fraud_model_loaded"] is True

    # 2. Phase 1 prediction
    sample_path = Path(__file__).resolve().parents[1] / "sample_request.json"
    req_data = json.loads(sample_path.read_text())

    res_pred = client.post("/api/v1/predict", json=req_data)
    assert res_pred.status_code == 200
    p_data = res_pred.json()
    assert "default_probability" in p_data
    assert "risk_band" in p_data
    assert p_data["risk_band"] in {"LOW", "MODERATE", "ELEVATED", "HIGH"}
    assert p_data["feature_count"] == 183
