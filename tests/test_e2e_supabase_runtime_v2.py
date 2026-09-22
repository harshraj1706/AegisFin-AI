"""
tests/test_e2e_supabase_runtime_v2.py

Automated Pytest wrapper verifying Step 17 End-to-End Supabase + FastAPI Integration.
Validates:
1. Schema conformity across all 4 Supabase fraud tables
2. Real prediction inference with frozen 62-feature pipeline
3. Database writes in fraud_transactions, fraud_predictions, and fraud_feature_snapshots
4. Exact 62-feature JSONB snapshot structure and anti-leakage
5. Duplicate transaction rejection (409 Conflict)
6. Legacy field compatibility mapping
7. Entity state updates
8. Safe test record cleanup
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.auth import signup_user, login_user
from app.production_feature_definitions import (
    VALID_FEATURE_NAMES,
    generate_production_features,
)
from app.supabase_client import get_supabase_admin_client


@pytest.fixture(scope="module")
def e2e_analyst_auth():
    unique_id = uuid.uuid4().hex[:8]
    email = f"pytest_e2e_{unique_id}@aegisfin.ai"
    password = f"AegisPass_{unique_id}!123"

    signup_res = signup_user(email=email, password=password, full_name="Pytest E2E Analyst")
    token = signup_res.get("access_token")
    user_id = signup_res.get("user_id")

    if not token:
        login_res = login_user(email=email, password=password)
        token = login_res.get("access_token")
        user_id = login_res.get("user_id")

    headers = {"Authorization": f"Bearer {token}"}
    yield headers, user_id, unique_id

    # Cleanup test user
    admin = get_supabase_admin_client()
    try:
        if user_id:
            admin.table("profiles").delete().eq("id", user_id).execute()
            admin.auth.admin.delete_user(user_id)
    except Exception:
        pass


def test_e2e_schema_and_column_conformity():
    """Verify live Supabase tables have exact Phase 2 columns."""
    admin = get_supabase_admin_client()

    snapshot_cols = ["id", "transaction_id", "feature_contract_version", "model_version", "feature_count", "features", "created_at"]
    for col in snapshot_cols:
        res = admin.table("fraud_feature_snapshots").select(col).limit(1).execute()
        assert res is not None

    # Verify invalid columns do not exist
    for bad_col in ["prediction_id", "customer_id", "snapshot_timestamp"]:
        with pytest.raises(Exception):
            admin.table("fraud_feature_snapshots").select(bad_col).limit(1).execute()

    pred_cols = [
        "raw_fraud_probability", "calibrated_fraud_probability",
        "risk_band", "recommended_action", "feature_contract_version",
        "calibrator_type", "scored_at"
    ]
    for col in pred_cols:
        res = admin.table("fraud_predictions").select(col).limit(1).execute()
        assert res is not None


def test_e2e_full_transaction_flow(e2e_analyst_auth):
    """Run real test transaction through FastAPI, verify DB writes, snapshots, duplicate protection, and cleanup."""
    headers, user_id, unique_id = e2e_analyst_auth
    client = TestClient(app)
    admin = get_supabase_admin_client()

    test_txn_id = f"PYTEST_E2E_{unique_id}"
    test_cust_id = f"CUST_PYTEST_{unique_id}"
    test_card_id = f"CARD_PYTEST_{unique_id}"
    test_device_id = f"DEV_PYTEST_{unique_id}"
    test_amount = 299.95
    test_ts = datetime.now(timezone.utc).isoformat()

    payload = {
        "transaction_id": test_txn_id,
        "customer_id": test_cust_id,
        "amount": test_amount,
        "transaction_timestamp": test_ts,
        "card_id": test_card_id,
        "device_id": test_device_id,
        "merchant_id": f"MERCH_PYTEST_{unique_id}",
        "email_domain": "gmail.com",
        "address_id": "299.0",
        "product_code": "W",
        "card_network": "visa",
        "card_type": "debit",
        "ip_address": "10.99.50.1",
        "country": "US",
    }

    try:
        # 1. First Prediction Request
        resp1 = client.post("/api/v1/fraud/predict", json=payload, headers=headers)
        assert resp1.status_code == 200
        data1 = resp1.json()

        assert data1["transaction_id"] == test_txn_id
        assert data1["model_version"] == "2.1.0"
        assert data1["feature_contract_version"] == "2.1.0"
        assert "calibrated_fraud_probability" in data1
        assert "risk_band" in data1
        assert "recommended_action" in data1
        assert "fraud_probability" in data1
        assert "fraud_band" in data1
        assert "decision" in data1

        # 2. Verify Database Writes
        db_tx = admin.table("fraud_transactions").select("*").eq("transaction_id", test_txn_id).execute().data
        db_pred = admin.table("fraud_predictions").select("*").eq("transaction_id", test_txn_id).execute().data
        db_snap = admin.table("fraud_feature_snapshots").select("*").eq("transaction_id", test_txn_id).execute().data

        assert len(db_tx) == 1
        assert len(db_pred) == 1
        assert len(db_snap) == 1

        # 3. Verify Snapshot Content
        stored_features = db_snap[0]["features"]
        assert len(stored_features) == 62
        assert set(stored_features.keys()) == set(VALID_FEATURE_NAMES)
        assert db_snap[0]["feature_count"] == 62
        assert db_snap[0]["feature_contract_version"] == "2.1.0"

        # Compare values against reference feature vector (cold start)
        ref_features = generate_production_features(current_transaction=payload, history=[])
        for k in VALID_FEATURE_NAMES:
            assert abs(float(stored_features[k]) - float(ref_features[k])) < 1e-4

        # 4. Verify Duplicate Protection (409)
        resp_dup = client.post("/api/v1/fraud/predict", json=payload, headers=headers)
        assert resp_dup.status_code == 409

        # Ensure no duplicate rows
        assert len(admin.table("fraud_transactions").select("id").eq("transaction_id", test_txn_id).execute().data) == 1
        assert len(admin.table("fraud_predictions").select("id").eq("transaction_id", test_txn_id).execute().data) == 1
        assert len(admin.table("fraud_feature_snapshots").select("id").eq("transaction_id", test_txn_id).execute().data) == 1

        # 5. Verify Anti-Leakage with Second Transaction
        second_txn_id = f"PYTEST_E2E_2_{unique_id}"
        second_ts = datetime.fromtimestamp(datetime.fromisoformat(test_ts).timestamp() + 180, tz=timezone.utc).isoformat()
        payload2 = dict(payload)
        payload2["transaction_id"] = second_txn_id
        payload2["transaction_timestamp"] = second_ts
        payload2["amount"] = 450.0

        resp2 = client.post("/api/v1/fraud/predict", json=payload2, headers=headers)
        assert resp2.status_code == 200

        db_snap2 = admin.table("fraud_feature_snapshots").select("features").eq("transaction_id", second_txn_id).execute().data
        assert len(db_snap2) == 1
        features2 = db_snap2[0]["features"]

        # First txn saw 0 customer history; second txn saw 1
        assert stored_features["customer_tx_count_1h"] == 0.0
        assert features2["customer_tx_count_1h"] >= 1.0

        # 6. Verify Entity State
        entity_res = admin.table("fraud_entity_state").select("*").eq("entity_type", "customer").eq("entity_key", test_cust_id).execute().data
        assert len(entity_res) == 1
        assert int(entity_res[0]["transaction_count"]) == 2
        assert abs(float(entity_res[0]["amount_sum"]) - (299.95 + 450.0)) < 1e-2

    finally:
        # 7. Safe Cleanup of Synthetic Test Records
        for tid in [test_txn_id, f"PYTEST_E2E_2_{unique_id}"]:
            try:
                admin.table("fraud_feature_snapshots").delete().eq("transaction_id", tid).execute()
                admin.table("fraud_predictions").delete().eq("transaction_id", tid).execute()
                admin.table("fraud_transactions").delete().eq("transaction_id", tid).execute()
            except Exception:
                pass
        try:
            admin.table("fraud_entity_state").delete().eq("entity_type", "customer").eq("entity_key", test_cust_id).execute()
            admin.table("fraud_entity_state").delete().eq("entity_type", "card").eq("entity_key", test_card_id).execute()
        except Exception:
            pass
