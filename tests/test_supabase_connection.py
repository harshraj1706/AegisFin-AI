from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.supabase_client import (
    get_supabase_client,
    get_supabase_admin_client,
    check_supabase_connection,
    check_database_health,
    test_fraud_transactions_crud as run_fraud_transactions_crud,
)
from app.config import SUPABASE_SECRET_KEY


def test_supabase_client_singleton():
    """Verify that get_supabase_client returns a reusable singleton instance."""
    client1 = get_supabase_client()
    client2 = get_supabase_client()
    assert client1 is client2, "get_supabase_client must return the same cached instance"


def test_supabase_admin_client_singleton():
    """Verify that get_supabase_admin_client returns a reusable singleton instance."""
    admin1 = get_supabase_admin_client()
    admin2 = get_supabase_admin_client()
    assert admin1 is admin2, "get_supabase_admin_client must return the same cached instance"


def test_supabase_connection_check():
    """Verify that check_supabase_connection successfully connects to Supabase."""
    result = check_supabase_connection()
    assert result["status"] == "connected"
    assert result["publishable_key_configured"] is True
    assert result["secret_key_configured"] is True
    assert result.get("auth_service") == "healthy"
    assert isinstance(result.get("latency_ms"), (int, float))

    # CRITICAL SECURITY CHECK: Ensure secret key is never leaked in the response
    assert SUPABASE_SECRET_KEY not in str(result), "SUPABASE_SECRET_KEY must never be exposed!"
    assert "secret_key" not in result or result.get("secret_key") is None or isinstance(result.get("secret_key"), bool)


def test_supabase_database_health():
    """Verify that check_database_health confirms accessibility to fraud_transactions table."""
    result = check_database_health()
    assert result["status"] == "healthy"
    assert result["database"] == "connected"
    assert result["table"] == "fraud_transactions"
    assert result["accessible"] is True
    assert SUPABASE_SECRET_KEY not in str(result), "SUPABASE_SECRET_KEY leaked in db health response!"


def test_fraud_transactions_crud_cycle():
    """
    Step 3 Verification:
    Connect -> Insert dummy row -> Read row back -> Delete row -> Verify database clean.
    """
    result = run_fraud_transactions_crud(
        transaction_id="TEST_TXN_001",
        customer_id="TEST_CUSTOMER",
        card_id="TEST_CARD",
        amount=100.00,
    )

    assert result["success"] is True, f"CRUD test failed: {result}"
    assert result["steps"]["connect"] is True
    assert result["steps"]["insert"] is True
    assert result["steps"]["read"] is True
    assert result["steps"]["delete"] is True
    assert result["steps"]["verified_clean"] is True

    verified = result["data_verified"]
    assert verified["transaction_id"] == "TEST_TXN_001"
    assert verified["customer_id"] == "TEST_CUSTOMER"
    assert verified["card_id"] == "TEST_CARD"
    assert verified["amount"] == 100.00

    # Ensure secret key is never present in the report
    assert SUPABASE_SECRET_KEY not in str(result), "SUPABASE_SECRET_KEY leaked in CRUD report!"


def test_fastapi_supabase_health_endpoint():
    """Verify FastAPI /health/supabase endpoint response and status code."""
    client = TestClient(app)
    response = client.get("/health/supabase")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "connected"
    assert data["publishable_key_configured"] is True
    assert data["secret_key_configured"] is True
    assert SUPABASE_SECRET_KEY not in str(data), "SUPABASE_SECRET_KEY leaked in API response!"


def test_fastapi_supabase_db_health_endpoint():
    """Verify FastAPI /health/supabase/db endpoint response and status code."""
    client = TestClient(app)
    response = client.get("/health/supabase/db")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["accessible"] is True
    assert SUPABASE_SECRET_KEY not in str(data), "SUPABASE_SECRET_KEY leaked in API response!"
