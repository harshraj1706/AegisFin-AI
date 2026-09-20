from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from supabase import Client, create_client

from .config import (
    SUPABASE_PUBLISHABLE_KEY,
    SUPABASE_SECRET_KEY,
    SUPABASE_URL,
)

# Reusable client singletons to avoid creating new clients per request
_supabase_client: Optional[Client] = None
_supabase_admin_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """
    Returns a reusable singleton Supabase client configured for normal Data API operations.
    Uses SUPABASE_PUBLISHABLE_KEY, respecting Row-Level Security (RLS).
    """
    global _supabase_client
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_PUBLISHABLE_KEY:
            raise ValueError(
                "Missing Supabase configuration. Ensure SUPABASE_URL and "
                "SUPABASE_PUBLISHABLE_KEY are defined in .env."
            )
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY)
    return _supabase_client


def get_supabase_admin_client() -> Client:
    """
    Returns a reusable singleton Supabase client configured for server-side admin operations.
    Uses SUPABASE_SECRET_KEY to bypass RLS for administrative tasks.
    CRITICAL: SUPABASE_SECRET_KEY must strictly remain server-side and never be exposed.
    """
    global _supabase_admin_client
    if _supabase_admin_client is None:
        if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
            raise ValueError(
                "Missing Supabase admin configuration. Ensure SUPABASE_URL and "
                "SUPABASE_SECRET_KEY are defined in .env."
            )
        _supabase_admin_client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
    return _supabase_admin_client


def check_supabase_connection() -> Dict[str, Any]:
    """
    Backend health check verifying connectivity and authentication with Supabase.
    Pings the Supabase Auth health endpoint using the initialized client session.
    Never exposes raw secrets or tokens.
    """
    if not SUPABASE_URL:
        return {
            "status": "error",
            "message": "SUPABASE_URL is not set",
            "publishable_key_configured": bool(SUPABASE_PUBLISHABLE_KEY),
            "secret_key_configured": bool(SUPABASE_SECRET_KEY),
        }

    if not SUPABASE_PUBLISHABLE_KEY:
        return {
            "status": "error",
            "message": "SUPABASE_PUBLISHABLE_KEY is not set",
            "publishable_key_configured": False,
            "secret_key_configured": bool(SUPABASE_SECRET_KEY),
        }

    try:
        t0 = time.perf_counter()
        client = get_supabase_client()

        # Ping the Supabase GoTrue Auth health check with client credentials
        resp = client.postgrest.session.get(
            f"{SUPABASE_URL.rstrip('/')}/auth/v1/health",
            timeout=10.0,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        if resp.status_code == 200:
            return {
                "status": "connected",
                "supabase_url": SUPABASE_URL,
                "latency_ms": latency_ms,
                "publishable_key_configured": True,
                "secret_key_configured": bool(SUPABASE_SECRET_KEY),
                "auth_service": "healthy",
            }
        else:
            return {
                "status": "degraded",
                "http_status": resp.status_code,
                "supabase_url": SUPABASE_URL,
                "latency_ms": latency_ms,
                "publishable_key_configured": True,
                "secret_key_configured": bool(SUPABASE_SECRET_KEY),
            }

    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "supabase_url": SUPABASE_URL,
            "publishable_key_configured": bool(SUPABASE_PUBLISHABLE_KEY),
            "secret_key_configured": bool(SUPABASE_SECRET_KEY),
        }


def check_database_health() -> Dict[str, Any]:
    """
    Verifies backend communication with the Supabase database.
    Checks table access on fraud_transactions via the admin client.
    """
    try:
        admin = get_supabase_admin_client()
        res = admin.table("fraud_transactions").select("id").limit(1).execute()
        return {
            "status": "healthy",
            "database": "connected",
            "table": "fraud_transactions",
            "accessible": True,
        }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "database": "error",
            "error": str(exc),
        }


def test_fraud_transactions_crud(
    transaction_id: str = "TEST_TXN_001",
    customer_id: str = "TEST_CUSTOMER",
    card_id: str = "TEST_CARD",
    amount: float = 100.00,
) -> Dict[str, Any]:
    """
    Executes a complete Connect -> Insert -> Read -> Delete CRUD test on fraud_transactions
    using safe dummy data.
    Ensures that cleanup occurs in a finally block so the database remains clean.
    Never exposes SUPABASE_SECRET_KEY.
    """
    admin = get_supabase_admin_client()
    now_utc = datetime.now(timezone.utc).isoformat()

    dummy_row = {
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "card_id": card_id,
        "amount": amount,
        "transaction_timestamp": now_utc,
    }

    report: Dict[str, Any] = {
        "transaction_id": transaction_id,
        "steps": {
            "connect": False,
            "insert": False,
            "read": False,
            "delete": False,
            "verified_clean": False,
        },
        "success": False,
    }

    try:
        # Step 1: Connect
        if not admin:
            raise ConnectionError("Failed to obtain Supabase admin client")
        report["steps"]["connect"] = True

        # Pre-cleanup in case a previously aborted test left a row
        try:
            admin.table("fraud_transactions").delete().eq("transaction_id", transaction_id).execute()
        except Exception:
            pass

        # Step 2: Insert temporary test row
        t0_insert = time.perf_counter()
        ins_res = admin.table("fraud_transactions").insert(dummy_row).execute()
        insert_latency = round((time.perf_counter() - t0_insert) * 1000, 2)
        if not ins_res.data:
            raise RuntimeError(f"Insert returned no data: {ins_res}")
        report["steps"]["insert"] = True
        report["insert_latency_ms"] = insert_latency

        # Step 3: Read that row back
        t0_read = time.perf_counter()
        read_res = (
            admin.table("fraud_transactions")
            .select("*")
            .eq("transaction_id", transaction_id)
            .execute()
        )
        read_latency = round((time.perf_counter() - t0_read) * 1000, 2)
        if not read_res.data:
            raise RuntimeError(f"Read returned no data for transaction_id={transaction_id}")

        row_read = read_res.data[0]
        assert row_read["transaction_id"] == transaction_id
        assert row_read["customer_id"] == customer_id
        assert row_read["card_id"] == card_id
        assert float(row_read["amount"]) == amount
        report["steps"]["read"] = True
        report["read_latency_ms"] = read_latency
        report["data_verified"] = {
            "id": row_read.get("id"),
            "transaction_id": row_read.get("transaction_id"),
            "customer_id": row_read.get("customer_id"),
            "card_id": row_read.get("card_id"),
            "amount": float(row_read.get("amount")),
            "transaction_timestamp": row_read.get("transaction_timestamp"),
        }

    finally:
        # Step 4: Delete the test row so database remains clean
        try:
            t0_del = time.perf_counter()
            admin.table("fraud_transactions").delete().eq("transaction_id", transaction_id).execute()
            report["steps"]["delete"] = True
            report["delete_latency_ms"] = round((time.perf_counter() - t0_del) * 1000, 2)
        except Exception as exc:
            report["delete_error"] = str(exc)

        # Step 5: Verify clean
        try:
            clean_res = (
                admin.table("fraud_transactions")
                .select("id")
                .eq("transaction_id", transaction_id)
                .execute()
            )
            if len(clean_res.data) == 0:
                report["steps"]["verified_clean"] = True
        except Exception as exc:
            report["verify_clean_error"] = str(exc)

    report["success"] = all(report["steps"].values())
    return report


if __name__ == "__main__":
    print("==================================================")
    print("AegisFin-AI: Supabase Connection & Database Test")
    print("==================================================")
    
    print("\n1. Testing Supabase Service Connection...")
    conn_result = check_supabase_connection()
    for k, v in conn_result.items():
        print(f"   {k}: {v}")

    print("\n2. Testing Database Health...")
    db_result = check_database_health()
    for k, v in db_result.items():
        print(f"   {k}: {v}")

    print("\n3. Testing fraud_transactions CRUD (Connect -> Insert -> Read -> Delete)...")
    crud_result = test_fraud_transactions_crud()
    print("   Steps:")
    for step, passed in crud_result["steps"].items():
        status = "PASSED" if passed else "FAILED"
        print(f"     • {step.ljust(16)}: {status}")
    print(f"   Overall CRUD Success: {crud_result['success']}")
    if "data_verified" in crud_result:
        print("   Verified Record:")
        for k, v in crud_result["data_verified"].items():
            print(f"     {k}: {v}")
    print("==================================================")
