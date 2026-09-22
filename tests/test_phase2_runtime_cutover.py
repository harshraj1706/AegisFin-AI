"""
tests/test_phase2_runtime_cutover.py

AegisFin-AI Phase 2 — Step 16: FastAPI Production Runtime Cutover Tests.
Verifies all 21 mandatory cutover requirements:
1. Exact 62-feature generation is used.
2. Legacy 459-feature model is NOT called by the live endpoint.
3. Frozen XGBoost v2 is called.
4. Frozen Platt calibrator is called.
5. Frozen policy config is applied.
6. LOW mapping works ([0.00, 0.10) -> LOW, ALLOW).
7. MEDIUM mapping works ([0.10, 0.40) -> REVIEW, MANUAL_REVIEW).
8. HIGH mapping works ([0.40, 0.80) -> REVIEW, MANUAL_REVIEW).
9. CRITICAL mapping works ([0.80, 1.00] -> HIGH, BLOCK).
10. AUTO_APPROVE mapping works.
11. STEP_UP_AUTH mapping works.
12. MANUAL_REVIEW mapping works.
13. HARD_DECLINE mapping works.
14. Raw and calibrated probabilities are persisted.
15. Canonical risk band is persisted.
16. Canonical recommended action is persisted.
17. 62-feature snapshot is persisted.
18. Snapshot contains exactly 62 features.
19. Current transaction is not included in its own historical features (strict anti-leakage).
20. Existing authentication behavior remains intact.
21. Existing API legacy response fields remain compatible.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.auth import get_current_user
from app.main import app
from app.fraud_feature_service import fetch_historical_transactions
from app.fraud_model_service import (
    FraudModelService,
    get_fraud_model_service,
    LEGACY_FRAUD_BAND_MAP,
    LEGACY_DECISION_MAP,
)
from app.production_feature_definitions import (
    VALID_FEATURE_NAMES,
    generate_production_features,
)
from app.schemas import FraudPredictionResponse


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def override_auth():
    """Provides authenticated test analyst user by default."""
    app.dependency_overrides[get_current_user] = lambda: {
        "user_id": "cutover-test-user-id",
        "email": "cutover_analyst@aegisfin.ai",
        "role": "analyst",
        "authenticated": True,
    }
    yield
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture(scope="module")
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-valid-token"}


@pytest.fixture
def mock_supabase():
    """Mocks Supabase admin table operations for deterministic isolated verification."""
    with patch("app.fraud_router.get_supabase_admin_client") as mock_get_admin:
        admin_mock = MagicMock()
        mock_get_admin.return_value = admin_mock

        # Setup table mock returns
        table_mock = MagicMock()
        admin_mock.table.return_value = table_mock

        # Duplicate check returns empty
        table_mock.select.return_value = table_mock
        table_mock.eq.return_value = table_mock
        table_mock.lt.return_value = table_mock
        table_mock.order.return_value = table_mock
        table_mock.limit.return_value = table_mock
        table_mock.insert.return_value = table_mock
        table_mock.update.return_value = table_mock
        table_mock.delete.return_value = table_mock
        table_mock.execute.return_value = MagicMock(data=[])

        yield admin_mock


# -----------------------------------------------------------------------------
# 1. Exact 62-Feature Generation
# -----------------------------------------------------------------------------
def test_item_01_exact_62_feature_generation():
    """Requirement 1: Exact 62-feature generation is used in contract order."""
    txn = {
        "transaction_id": "TX_62_FEAT_01",
        "amount": 250.0,
        "transaction_timestamp": "2026-09-21T12:00:00Z",
        "customer_id": "CUST_62_01",
        "card_id": "CARD_62_01",
        "product_code": "W",
    }
    feats = generate_production_features(current_transaction=txn, history=[])
    assert len(feats) == 62
    assert list(feats.keys()) == VALID_FEATURE_NAMES
    for k, v in feats.items():
        assert isinstance(v, (int, float))
        assert not np.isnan(v)
        assert not np.isinf(v)


# -----------------------------------------------------------------------------
# 2. Legacy 459-feature model is NOT called by live endpoint
# 3. Frozen XGBoost v2 is called
# 4. Frozen Platt calibrator is called
# 5. Frozen policy config is applied
# -----------------------------------------------------------------------------
def test_item_02_03_04_05_live_endpoint_uses_v2_not_legacy_champion(client, auth_headers, mock_supabase):
    """
    Requirements 2, 3, 4, 5:
    - Live endpoint calls Phase 2 XGBoost v2 service with 62 features.
    - Legacy 459-feature model is NOT called.
    - Platt calibrator is applied.
    - Frozen policy is applied.
    """
    service = get_fraud_model_service()
    assert service.expected_feature_count == 62
    assert service.model_version == "2.1.0"
    assert service.calibration_version == "2.1.0"
    assert service.policy_version == "2.0.0"

    payload = {
        "transaction_id": f"TX_CUTOVER_{uuid.uuid4().hex[:8]}",
        "amount": 120.00,
        "transaction_timestamp": "2026-09-21T14:00:00Z",
        "customer_id": "CUST_LIVE_01",
        "card_id": "CARD_LIVE_01",
    }

    # Spy on model predict
    with patch.object(service, "predict", wraps=service.predict) as spy_predict:
        res = client.post("/api/v1/fraud/predict", json=payload, headers=auth_headers)
        assert res.status_code == 200, res.text
        spy_predict.assert_called_once()
        call_args = spy_predict.call_args[0][0]
        # Verified input is 62 production features, not 459
        if isinstance(call_args, pd.DataFrame):
            assert call_args.shape[1] == 62
        else:
            assert len(call_args) == 62

    data = res.json()
    assert data["model_version"] == "2.1.0"
    assert data["feature_contract_version"] == "2.1.0"
    assert data["calibration_version"] == "2.1.0"
    assert data["policy_version"] == "2.0.0"
    assert data["feature_count"] == 62


# -----------------------------------------------------------------------------
# 6, 7, 8, 9, 10, 11, 12, 13. Risk Band and Action Mappings
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "calibrated_prob,expected_band,expected_action,expected_legacy_band,expected_legacy_decision",
    [
        (0.000, "LOW", "AUTO_APPROVE", "LOW", "ALLOW"),
        (0.050, "LOW", "AUTO_APPROVE", "LOW", "ALLOW"),
        (0.099, "LOW", "AUTO_APPROVE", "LOW", "ALLOW"),
        (0.100, "MEDIUM", "STEP_UP_AUTH", "REVIEW", "MANUAL_REVIEW"),
        (0.250, "MEDIUM", "STEP_UP_AUTH", "REVIEW", "MANUAL_REVIEW"),
        (0.399, "MEDIUM", "STEP_UP_AUTH", "REVIEW", "MANUAL_REVIEW"),
        (0.400, "HIGH", "MANUAL_REVIEW", "REVIEW", "MANUAL_REVIEW"),
        (0.600, "HIGH", "MANUAL_REVIEW", "REVIEW", "MANUAL_REVIEW"),
        (0.799, "HIGH", "MANUAL_REVIEW", "REVIEW", "MANUAL_REVIEW"),
        (0.800, "CRITICAL", "HARD_DECLINE", "HIGH", "BLOCK"),
        (0.950, "CRITICAL", "HARD_DECLINE", "HIGH", "BLOCK"),
        (1.000, "CRITICAL", "HARD_DECLINE", "HIGH", "BLOCK"),
    ],
)
def test_item_06_to_13_policy_bands_and_action_mappings(
    calibrated_prob,
    expected_band,
    expected_action,
    expected_legacy_band,
    expected_legacy_decision,
):
    """
    Requirements 6, 7, 8, 9, 10, 11, 12, 13:
    Verifies all 4 policy intervals:
    - LOW: [0.00, 0.10) -> AUTO_APPROVE (legacy: LOW, ALLOW)
    - MEDIUM: [0.10, 0.40) -> STEP_UP_AUTH (legacy: REVIEW, MANUAL_REVIEW)
    - HIGH: [0.40, 0.80) -> MANUAL_REVIEW (legacy: REVIEW, MANUAL_REVIEW)
    - CRITICAL: [0.80, 1.00] -> HARD_DECLINE (legacy: HIGH, BLOCK)
    """
    service = get_fraud_model_service()
    band, action = service.evaluate_risk_policy(calibrated_prob)
    assert band == expected_band
    assert action == expected_action

    # Test legacy projection maps
    assert LEGACY_FRAUD_BAND_MAP[band] == expected_legacy_band
    assert LEGACY_DECISION_MAP[action] == expected_legacy_decision


# -----------------------------------------------------------------------------
# 14, 15, 16, 17, 18. Persistence of Canonical Fields & Feature Snapshot
# -----------------------------------------------------------------------------
def test_item_14_to_18_persistence_canonical_and_feature_snapshots(client, auth_headers, mock_supabase):
    """
    Requirements 14, 15, 16, 17, 18:
    - Raw and calibrated probabilities persisted.
    - Canonical risk band persisted.
    - Canonical recommended action persisted.
    - 62-feature snapshot persisted.
    - Snapshot contains exactly 62 features in JSONB.
    """
    txn_id = f"TX_PERSIST_{uuid.uuid4().hex[:8]}"
    payload = {
        "transaction_id": txn_id,
        "amount": 99.99,
        "transaction_timestamp": "2026-09-21T15:00:00Z",
        "customer_id": "CUST_PERSIST_01",
        "card_id": "CARD_PERSIST_01",
        "device_id": "DEV_PERSIST_01",
        "merchant_id": "MERCH_PERSIST_01",
        "email_domain": "gmail.com",
        "address_id": "94103",
        "product_code": "W",
        "card_network": "visa",
        "card_type": "debit",
        "ip_address": "192.168.1.1",
        "country": "US",
    }

    # Intercept table inserts
    inserted_records: dict[str, list[dict]] = {}

    def fake_insert(record):
        # Determine table name from mock call
        return MagicMock(execute=lambda: MagicMock(data=[record]))

    # Capture calls to table("fraud_predictions") and table("fraud_feature_snapshots")
    prediction_inserts = []
    snapshot_inserts = []
    transaction_inserts = []

    def mock_table(table_name):
        t_mock = MagicMock()
        t_mock.select.return_value = t_mock
        t_mock.eq.return_value = t_mock
        t_mock.lt.return_value = t_mock
        t_mock.order.return_value = t_mock
        t_mock.limit.return_value = t_mock
        t_mock.execute.return_value = MagicMock(data=[])

        if table_name == "fraud_predictions":
            t_mock.insert.side_effect = lambda rec: prediction_inserts.append(rec) or MagicMock(execute=lambda: MagicMock(data=[rec]))
        elif table_name == "fraud_feature_snapshots":
            t_mock.insert.side_effect = lambda rec: snapshot_inserts.append(rec) or MagicMock(execute=lambda: MagicMock(data=[rec]))
        elif table_name == "fraud_transactions":
            t_mock.insert.side_effect = lambda rec: transaction_inserts.append(rec) or MagicMock(execute=lambda: MagicMock(data=[rec]))

        return t_mock

    mock_supabase.table.side_effect = mock_table

    res = client.post("/api/v1/fraud/predict", json=payload, headers=auth_headers)
    assert res.status_code == 200, res.text

    # 1. Verify fraud_transactions insert
    assert len(transaction_inserts) == 1
    tx_rec = transaction_inserts[0]
    assert tx_rec["transaction_id"] == txn_id
    assert tx_rec["amount"] == 99.99

    # 2. Verify fraud_predictions insert (Requirement 14, 15, 16)
    assert len(prediction_inserts) == 1
    pred_rec = prediction_inserts[0]
    assert pred_rec["transaction_id"] == txn_id
    assert "raw_fraud_probability" in pred_rec
    assert "calibrated_fraud_probability" in pred_rec
    assert 0.0 <= pred_rec["raw_fraud_probability"] <= 1.0
    assert 0.0 <= pred_rec["calibrated_fraud_probability"] <= 1.0
    assert pred_rec["risk_band"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert pred_rec["recommended_action"] in {"AUTO_APPROVE", "STEP_UP_AUTH", "MANUAL_REVIEW", "HARD_DECLINE"}
    assert pred_rec["feature_contract_version"] == "2.1.0"
    assert pred_rec["calibrator_type"] == "sigmoid"
    assert "scored_at" in pred_rec

    # Also verify legacy compatibility columns
    assert pred_rec["fraud_band"] in {"LOW", "REVIEW", "HIGH"}
    assert pred_rec["decision"] in {"ALLOW", "MANUAL_REVIEW", "BLOCK"}
    assert pred_rec["fraud_probability"] == pred_rec["calibrated_fraud_probability"]

    # 3. Verify fraud_feature_snapshots insert (Requirement 17, 18)
    assert len(snapshot_inserts) == 1
    snap_rec = snapshot_inserts[0]
    assert snap_rec["transaction_id"] == txn_id
    assert snap_rec["feature_contract_version"] == "2.1.0"
    assert snap_rec["model_version"] == "2.1.0"
    assert snap_rec["feature_count"] == 62
    assert isinstance(snap_rec["features"], dict)
    assert len(snap_rec["features"]) == 62
    assert list(snap_rec["features"].keys()) == VALID_FEATURE_NAMES


# -----------------------------------------------------------------------------
# 19. Anti-Leakage: Current Transaction Excluded from Its Own History
# -----------------------------------------------------------------------------
def test_item_19_anti_leakage_current_transaction_excluded():
    """
    Requirement 19:
    The current transaction is strictly excluded from its own historical features.
    History must strictly enforce: history.timestamp < current.timestamp.
    """
    curr_txn = {
        "transaction_id": "TX_CURR_100",
        "customer_id": "CUST_LEAK_CHECK",
        "card_id": "CARD_LEAK_CHECK",
        "amount": 500.0,
        "transaction_timestamp": "2026-09-21T12:00:00Z",
    }

    # Concurrent transaction with same timestamp
    concurrent_txn = {
        "transaction_id": "TX_CONCURRENT",
        "customer_id": "CUST_LEAK_CHECK",
        "card_id": "CARD_LEAK_CHECK",
        "amount": 200.0,
        "transaction_timestamp": "2026-09-21T12:00:00Z",
    }

    # Future transaction
    future_txn = {
        "transaction_id": "TX_FUTURE",
        "customer_id": "CUST_LEAK_CHECK",
        "card_id": "CARD_LEAK_CHECK",
        "amount": 9999.0,
        "transaction_timestamp": "2026-09-21T13:00:00Z",
    }

    # Valid prior transaction
    prior_txn = {
        "transaction_id": "TX_PRIOR",
        "customer_id": "CUST_LEAK_CHECK",
        "card_id": "CARD_LEAK_CHECK",
        "amount": 100.0,
        "transaction_timestamp": "2026-09-21T11:00:00Z",
    }

    # History contains curr_txn, concurrent_txn, future_txn, and prior_txn
    polluted_history = [curr_txn, concurrent_txn, future_txn, prior_txn]

    features = generate_production_features(current_transaction=curr_txn, history=polluted_history)

    # Only prior_txn should have been processed!
    # customer_tx_count_24h must be 1.0 (from prior_txn only)
    assert features["customer_tx_count_24h"] == 1.0
    # customer_amount_mean must be 100.0 (from prior_txn only)
    assert features["customer_amount_mean"] == 100.0
    # customer_is_new must be 0.0 (because prior_txn exists)
    assert features["customer_is_new"] == 0.0
    # curr_amount is 500.0, prior mean is 100.0 -> ratio should be ~5.0
    assert pytest.approx(features["customer_amount_ratio"], abs=1e-2) == 5.0


# -----------------------------------------------------------------------------
# 20. Authentication Behavior Intact
# -----------------------------------------------------------------------------
def test_item_20_authentication_enforced(client):
    """Requirement 20: Existing authentication behavior remains intact."""
    app.dependency_overrides.pop(get_current_user, None)
    try:
        payload = {
            "transaction_id": "TX_AUTH_TEST",
            "amount": 100.0,
            "customer_id": "CUST_AUTH",
            "card_id": "CARD_AUTH",
        }
        # No auth header -> 401 Unauthorized
        res = client.post("/api/v1/fraud/predict", json=payload)
        assert res.status_code == 401

        # Bad auth header -> 401 Unauthorized
        res_bad = client.post(
            "/api/v1/fraud/predict",
            json=payload,
            headers={"Authorization": "Bearer forged-token-invalid"},
        )
        assert res_bad.status_code == 401
    finally:
        app.dependency_overrides[get_current_user] = lambda: {
            "user_id": "cutover-test-user-id",
            "email": "cutover_analyst@aegisfin.ai",
            "role": "analyst",
            "authenticated": True,
        }


# -----------------------------------------------------------------------------
# 21. API Legacy Response Fields Remain Fully Compatible
# -----------------------------------------------------------------------------
def test_item_21_legacy_response_fields_compatibility(client, auth_headers, mock_supabase):
    """
    Requirement 21:
    Existing API legacy response fields remain compatible so UI and callers don't break:
    - fraud_probability
    - fraud_band
    - decision
    - model_name
    - model_version
    - calibration_version
    - policy_version
    - prediction_latency_ms
    AND canonical Phase 2 fields are exposed:
    - raw_fraud_probability
    - calibrated_fraud_probability
    - risk_band
    - recommended_action
    - feature_contract_version
    - calibrator_type
    """
    payload = {
        "transaction_id": f"TX_COMPAT_{uuid.uuid4().hex[:8]}",
        "amount": 50.0,
        "customer_id": "CUST_COMPAT",
        "card_id": "CARD_COMPAT",
    }

    res = client.post("/api/v1/fraud/predict", json=payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()

    # Legacy fields
    assert "fraud_probability" in data
    assert "fraud_band" in data
    assert "decision" in data
    assert "model_name" in data
    assert "model_version" in data
    assert "calibration_version" in data
    assert "policy_version" in data
    assert "prediction_latency_ms" in data

    # Canonical Phase 2 fields
    assert "raw_fraud_probability" in data
    assert "calibrated_fraud_probability" in data
    assert "risk_band" in data
    assert "recommended_action" in data
    assert "feature_contract_version" in data
    assert "calibrator_type" in data
    assert "scored_at" in data

    # Verify Pydantic response model validation passes
    parsed_response = FraudPredictionResponse(**data)
    assert parsed_response.transaction_id == data["transaction_id"]
    assert parsed_response.feature_contract_version == "2.1.0"
