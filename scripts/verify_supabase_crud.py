"""
AegisFin-AI: Step 3 — Supabase Database Connectivity and CRUD Test Script.

Validates:
AegisFin FastAPI / Backend
      ↓
Supabase client
      ↓
fraud_transactions
      ↓
insert/read/delete
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure backend root is in sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.supabase_client import (
    check_supabase_connection,
    check_database_health,
    test_fraud_transactions_crud,
)


def main() -> int:
    print("=" * 60)
    print("AegisFin-AI: Step 3 — Supabase Database Connectivity & CRUD Test")
    print("=" * 60)

    # 1. Check Service Connectivity
    print("\n[1/3] Verifying Supabase API connection...")
    conn = check_supabase_connection()
    print(f"      Status: {conn.get('status')}")
    print(f"      URL: {conn.get('supabase_url')}")
    print(f"      Auth Service: {conn.get('auth_service', 'unknown')}")
    print(f"      Latency: {conn.get('latency_ms', 0)} ms")
    if conn.get("status") != "connected":
        print("[-] Supabase connection failed!")
        return 1

    # 2. Check Database Table Health
    print("\n[2/3] Checking fraud_transactions table access...")
    db_health = check_database_health()
    print(f"      Database: {db_health.get('database')}")
    print(f"      Table: {db_health.get('table')}")
    print(f"      Accessible: {db_health.get('accessible', False)}")
    if db_health.get("status") != "healthy":
        print(f"[-] Database table check failed: {db_health.get('error')}")
        return 1

    # 3. Perform CRUD Cycle (Connect -> Insert -> Read -> Delete)
    print("\n[3/3] Executing CRUD cycle on fraud_transactions...")
    crud = test_fraud_transactions_crud()
    for step, passed in crud["steps"].items():
        icon = "[PASS]" if passed else "[FAIL]"
        print(f"      {icon} {step.ljust(18)}")

    if crud.get("success"):
        print("\n[+] Verification successful!")
        verified = crud.get("data_verified", {})
        print(f"    - Transaction ID : {verified.get('transaction_id')}")
        print(f"    - Customer ID    : {verified.get('customer_id')}")
        print(f"    - Card ID        : {verified.get('card_id')}")
        print(f"    - Amount         : ${verified.get('amount'):.2f}")
        print(f"    - Timestamp      : {verified.get('transaction_timestamp')}")
        print("    - Cleaned up     : Database confirmed clean")
        print("=" * 60)
        return 0
    else:
        print("\n[-] CRUD test failed!")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
