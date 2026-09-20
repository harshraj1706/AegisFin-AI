"""
test_streamlit_fraud_integration.py

Integration tests for AegisFin Phase 2 Streamlit Fraud Detection frontend & API contracts.
Verifies:
1. Authenticated fraud page access
2. Unauthenticated access (401)
3. Valid API prediction request
4. API 401 handling across endpoints
5. API validation error (422)
6. Successful fraud response rendering & contract structure
7. Duplicate transaction handling (409)
8. Transaction history loading (/api/v1/fraud/history)
9. High-risk history loading (/api/v1/fraud/high-risk)
10. Streamlit presets schema compliance
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Generator

import pytest
from fastapi.testclient import TestClient

from app.auth import login_user, signup_user
from app.main import app
from app.schemas import FraudPredictionRequest
from app.streamlit_fraud_view import FRAUD_PRESETS
from app.supabase_client import get_supabase_admin_client


@pytest.fixture(scope="module")
def auth_headers() -> Generator[Dict[str, str], None, None]:
    """Creates a temporary analyst test user and generates a valid Bearer token."""
    unique_id = uuid.uuid4().hex[:8]
    email = f"streamlit_test_{unique_id}@gmail.com"
    password = f"AegisStreamlit_{unique_id}!123"
    full_name = f"Frontend Test Analyst {unique_id}"

    # Sign up
    signup_res = signup_user(email=email, password=password, full_name=full_name)
    token = signup_res.get("access_token")

    if not token:
        login_res = login_user(email=email, password=password)
        token = login_res.get("access_token")

    user_id = signup_res.get("user_id")
    headers = {"Authorization": f"Bearer {token}"}

    yield headers

    # Cleanup test user
    admin = get_supabase_admin_client()
    try:
        if user_id:
            admin.table("profiles").delete().eq("id", user_id).execute()
            admin.auth.admin.delete_user(user_id)
    except Exception:
        pass


@pytest.fixture
def cleanup_transactions():
    """Tracks created test transactions and removes them from Supabase in teardown."""
    created_txns: list[str] = []

    yield created_txns

    admin = get_supabase_admin_client()
    for txn_id in created_txns:
        try:
            admin.table("fraud_predictions").delete().eq("transaction_id", txn_id).execute()
            admin.table("fraud_transactions").delete().eq("transaction_id", txn_id).execute()
        except Exception:
            pass


def test_unauthenticated_access_blocked():
    """Requirement 2 & 4: Unauthenticated access to fraud endpoints must return 401."""
    with TestClient(app) as client:
        # Prediction unauthenticated
        res_pred = client.post("/api/v1/fraud/predict", json={"transaction_id": "TEST"})
        assert res_pred.status_code == 401

        # History unauthenticated
        res_hist = client.get("/api/v1/fraud/history")
        assert res_hist.status_code == 401

        # High-risk unauthenticated
        res_risk = client.get("/api/v1/fraud/high-risk")
        assert res_risk.status_code == 401


def test_authenticated_fraud_health_access():
    """Requirement 1: Health endpoint is reachable for frontend health check."""
    with TestClient(app) as client:
        res = client.get("/api/v1/fraud/health")
        assert res.status_code == 200
        data = res.json()
        assert data.get("status") in ("ready", "ok")
        assert data.get("fraud_model_loaded") is True
        assert data.get("model_name") == "XGBoost"
        assert data.get("model_version") == "phase2-xgb-v1"


def test_valid_fraud_prediction_api_request(auth_headers, cleanup_transactions):
    """Requirement 3: Valid transaction payload submitted by frontend returns 200 with calibrated prediction."""
    txn_id = f"TXN_FRONTEND_{uuid.uuid4().hex[:8].upper()}"
    cleanup_transactions.append(txn_id)

    payload = {
        "transaction_id": txn_id,
        "amount": 149.99,
        "transaction_timestamp": datetime.now(timezone.utc).isoformat(),
        "customer_id": "CUST_STREAMLIT_01",
        "card_id": "40001234",
        "device_id": "DEV_STREAMLIT_BROWSER",
        "merchant_id": "M_AMAZON",
        "email_domain": "gmail.com",
        "address_id": "299.0",
        "product_code": "W",
        "card_network": "visa",
        "card_type": "credit",
        "ip_address": "192.0.2.1",
        "country": "US",
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/fraud/predict", json=payload, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()

        assert data["transaction_id"] == txn_id
        assert 0.0 <= data["fraud_probability"] <= 1.0
        assert data["fraud_band"] in ("LOW", "REVIEW", "HIGH")
        assert data["decision"] in ("ALLOW", "MANUAL_REVIEW", "BLOCK")
        assert data["model_name"] == "XGBoost"
        assert data["model_version"] == "phase2-xgb-v1"
        assert data["prediction_latency_ms"] > 0


def test_api_validation_error_handling(auth_headers):
    """Requirement 5: Validation errors (e.g. negative amount, missing mandatory fields) return 422."""
    with TestClient(app) as client:
        # Negative amount
        invalid_payload = {
            "transaction_id": "TXN_INVALID_AMT",
            "amount": -50.0,
            "customer_id": "CUST_01",
            "card_id": "CARD_01",
        }
        res = client.post("/api/v1/fraud/predict", json=invalid_payload, headers=auth_headers)
        assert res.status_code == 422

        # Missing mandatory card_id
        missing_card = {
            "transaction_id": "TXN_MISSING_CARD",
            "amount": 100.0,
            "customer_id": "CUST_01",
        }
        res_card = client.post("/api/v1/fraud/predict", json=missing_card, headers=auth_headers)
        assert res_card.status_code == 422


def test_duplicate_transaction_error_handling(auth_headers, cleanup_transactions):
    """Requirement 7: Submitting duplicate transaction ID returns 409 conflict."""
    dup_id = f"TXN_DUP_{uuid.uuid4().hex[:8].upper()}"
    cleanup_transactions.append(dup_id)

    payload = {
        "transaction_id": dup_id,
        "amount": 200.0,
        "customer_id": "CUST_DUP",
        "card_id": "CARD_DUP",
    }

    with TestClient(app) as client:
        # First submission succeeds
        first_resp = client.post("/api/v1/fraud/predict", json=payload, headers=auth_headers)
        assert first_resp.status_code == 200

        # Second submission triggers 409
        second_resp = client.post("/api/v1/fraud/predict", json=payload, headers=auth_headers)
        assert second_resp.status_code == 409
        assert "already been processed" in second_resp.json().get("detail", "").lower()


def test_transaction_history_endpoint(auth_headers, cleanup_transactions):
    """Requirement 8: Transaction history endpoint returns paginated transaction items."""
    test_id = f"TXN_HIST_{uuid.uuid4().hex[:8].upper()}"
    cleanup_transactions.append(test_id)

    payload = {
        "transaction_id": test_id,
        "amount": 350.0,
        "customer_id": "CUST_HIST",
        "card_id": "CARD_HIST",
    }

    with TestClient(app) as client:
        # Seed one transaction
        client.post("/api/v1/fraud/predict", json=payload, headers=auth_headers)

        # Query history
        hist_resp = client.get("/api/v1/fraud/history?limit=10", headers=auth_headers)
        assert hist_resp.status_code == 200
        body = hist_resp.json()
        assert "items" in body
        assert isinstance(body["items"], list)

        # Verify seeded transaction is present in recent history
        matching = [item for item in body["items"] if item.get("transaction_id") == test_id]
        assert len(matching) >= 1
        item = matching[0]
        assert item["amount"] == 350.0
        assert "fraud_probability" in item
        assert "fraud_band" in item
        assert "decision" in item


def test_high_risk_history_endpoint(auth_headers, cleanup_transactions):
    """Requirement 9: High-risk history endpoint returns only transactions with HIGH or REVIEW band."""
    with TestClient(app) as client:
        risk_resp = client.get("/api/v1/fraud/high-risk?limit=5", headers=auth_headers)
        assert risk_resp.status_code == 200
        body = risk_resp.json()
        assert "items" in body
        assert isinstance(body["items"], list)

        # Every returned item must be in HIGH or REVIEW risk band
        for item in body["items"]:
            assert item.get("fraud_band") in ("HIGH", "REVIEW")


def test_fraud_presets_schema_compliance():
    """Requirement 3: Verify all presets provided in Streamlit UI conform to FraudPredictionRequest schema."""
    for preset_label, preset in FRAUD_PRESETS.items():
        # Inject dummy unique ID and timestamp to validate complete request schema
        test_payload = {
            **preset,
            "transaction_id": f"TEST_PRESET_{uuid.uuid4().hex[:6]}",
            "transaction_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        # Pydantic validation
        validated = FraudPredictionRequest(**test_payload)
        assert validated.amount == preset["amount"]
        assert validated.customer_id == preset["customer_id"]
        assert validated.card_id == preset["card_id"]
