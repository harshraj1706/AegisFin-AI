from __future__ import annotations

import copy
import uuid
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.fraud_feature_service import build_fraud_features, get_fraud_feature_service
from app.fraud_model_service import (
    FraudModelService,
    FraudModelSchemaError,
    FraudModelArtifactError,
    FraudModelCalibrationError,
    get_fraud_model_service,
    predict_fraud_risk,
    get_fraud_band,
    get_fraud_decision,
)


@pytest.fixture(scope="module")
def model_service() -> FraudModelService:
    return get_fraud_model_service()


# -----------------------------------------------------------------------------
# 1. Model Service Initialization and Metadata Tests
# -----------------------------------------------------------------------------
def test_model_service_initialization(model_service: FraudModelService):
    """Verify singleton initialization, metadata, and dynamic schema resolution."""
    assert model_service.model_name == "XGBoost"
    assert model_service.model_version == "phase2-xgb-v1"
    assert model_service.calibration_version == "phase2-platt-v1"
    assert model_service.policy_version == "phase2-policy-v1"
    assert model_service.calibration_method == "Platt"
    assert model_service.expected_feature_count == 459
    assert len(model_service.feature_names) == 459

    # Verify policy thresholds loaded dynamically
    assert model_service.review_threshold == pytest.approx(0.24520720672829832, rel=1e-5)
    assert model_service.high_threshold == pytest.approx(0.3544080190457745, rel=1e-5)
    assert model_service.decision_threshold == 0.11


def test_model_service_singleton():
    """Verify get_fraud_model_service returns the cached singleton."""
    s1 = get_fraud_model_service()
    s2 = get_fraud_model_service()
    assert s1 is s2


# -----------------------------------------------------------------------------
# 2. Integration: Feature Engine -> Model Service (Section 8)
# -----------------------------------------------------------------------------
def test_feature_engine_to_model_service_pipeline(model_service: FraudModelService):
    """
    Pass deterministic test transaction through:
    transaction -> feature engine -> model service.
    """
    txn = {
        "transaction_id": "TEST_PHASE2_MODEL_001",
        "customer_id": "TEST_CUSTOMER_001",
        "card_id": "9633",
        "device_id": "TEST_DEVICE_001",
        "merchant_id": "TEST_MERCHANT_001",
        "amount": 1000.00,
        "transaction_timestamp": "2026-09-20T10:00:00Z",
        "address_id": "299.0",
        "product_code": "W",
        "card_network": "visa",
        "card_type": "debit",
    }

    # Step 5: Feature Engine
    df = build_fraud_features(txn, history=[])
    assert df.shape == (1, 459)

    # Step 6: Model Service
    res = model_service.predict(df)

    # Verify inference output structure
    assert "raw_probability" in res
    assert "fraud_probability" in res
    assert "fraud_band" in res
    assert "decision" in res
    assert "inference_latency_ms" in res
    assert "model_inference_latency_ms" in res

    # Numerical validity
    assert 0.0 <= res["raw_probability"] <= 1.0
    assert 0.0 <= res["fraud_probability"] <= 1.0
    assert res["fraud_band"] in {"LOW", "REVIEW", "HIGH"}
    assert res["decision"] in {"ALLOW", "MANUAL_REVIEW", "BLOCK"}
    assert res["feature_count"] == 459
    assert res["model_name"] == "XGBoost"
    assert res["model_version"] == "phase2-xgb-v1"
    assert res["inference_latency_ms"] > 0.0


# -----------------------------------------------------------------------------
# 3. Determinism Test (Section 8)
# -----------------------------------------------------------------------------
def test_deterministic_predictions(model_service: FraudModelService):
    """Repeated identical inputs must produce deterministic outputs."""
    txn = {
        "customer_id": "CUST_DET_001",
        "amount": 420.50,
        "transaction_timestamp": "2026-09-20T12:00:00Z",
        "card_id": "9633",
        "address_id": "299.0",
    }

    df1 = build_fraud_features(txn, history=[])
    df2 = build_fraud_features(txn, history=[])

    res1 = model_service.predict(df1)
    res2 = model_service.predict(df2)

    assert res1["raw_probability"] == res2["raw_probability"]
    assert res1["fraud_probability"] == res2["fraud_probability"]
    assert res1["fraud_band"] == res2["fraud_band"]
    assert res1["decision"] == res2["decision"]


# -----------------------------------------------------------------------------
# 4. Historical Behavior & Anti-Leakage (Section 9)
# -----------------------------------------------------------------------------
def test_historical_behavior_and_anti_leakage(model_service: FraudModelService):
    """
    Verify historical influence on model prediction:
    - Transaction A (earlier, $500)
    - Transaction B (later, $1000) sees A -> past_count=1
    - Transaction C (earlier) vs Future Transaction D ($99,999) -> D does not affect C
    """
    cust_id = f"TEST_HIST_{uuid.uuid4().hex[:8]}"

    txn_a = {
        "customer_id": cust_id,
        "amount": 500.0,
        "transaction_timestamp": "2026-09-20T10:00:00Z",
    }
    txn_b = {
        "customer_id": cust_id,
        "amount": 1000.0,
        "transaction_timestamp": "2026-09-20T11:00:00Z",
    }

    # B can use A's historical state
    df_b_with_a = build_fraud_features(txn_b, history=[txn_a])
    res_b = model_service.predict(df_b_with_a)
    assert df_b_with_a["uid_past_count"].iloc[0] == 1.0
    assert 0.0 <= res_b["fraud_probability"] <= 1.0

    # Anti-leakage: C at 12:00 must NOT see future transaction D at 13:00
    txn_c = {
        "customer_id": cust_id,
        "amount": 300.0,
        "transaction_timestamp": "2026-09-20T12:00:00Z",
    }
    txn_d = {
        "customer_id": cust_id,
        "amount": 99999.0,
        "transaction_timestamp": "2026-09-20T13:00:00Z",
    }

    df_c_with_d = build_fraud_features(txn_c, history=[txn_d])
    res_c = model_service.predict(df_c_with_d)
    assert df_c_with_d["uid_past_count"].iloc[0] == 0.0
    assert df_c_with_d["uid_is_new"].iloc[0] == 1.0
    assert 0.0 <= res_c["fraud_probability"] <= 1.0


# -----------------------------------------------------------------------------
# 5. Calibration Method & Boundary Checks (Section 4)
# -----------------------------------------------------------------------------
def test_calibration_bounds_and_monotonicity(model_service: FraudModelService):
    """Verify Platt calibrator strictly maintains probabilities in [0, 1]."""
    test_raw_probs = [0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.99, 1.0]
    cal_probs = [model_service.calibrate_probability(p) for p in test_raw_probs]

    for p in cal_probs:
        assert 0.0 <= p <= 1.0

    # Platt scaling logistic curve is strictly monotonically increasing
    for i in range(len(cal_probs) - 1):
        assert cal_probs[i] <= cal_probs[i + 1]


def test_calibration_invalid_input(model_service: FraudModelService):
    """Calibrator must raise on out-of-bounds inputs."""
    with pytest.raises(FraudModelCalibrationError):
        model_service.calibrate_probability(-0.01)

    with pytest.raises(FraudModelCalibrationError):
        model_service.calibrate_probability(1.05)


def test_platt_calibration_known_raw_probability_agreement(model_service: FraudModelService):
    """
    Verify known raw probability (p = 0.0025):
    - Live calibration MUST apply logit transform: logit(p) = log(p / (1 - p))
    - Then call saved Platt LogisticRegression calibrator.
    - Yields calibrated probability approximately 0.0020 (specifically ~0.002009), NOT 0.2253.
    - Backend service agrees precisely with saved calibrator object.
    """
    raw_p = 0.0025
    cal_prob = model_service.calibrate_probability(raw_p)

    # 1. Verify calibration output matches expected ~0.0020
    assert pytest.approx(cal_prob, abs=1e-4) == 0.0020
    assert pytest.approx(cal_prob, abs=1e-6) == 0.002009

    # 2. Verify it does NOT evaluate to the buggy un-transformed ~0.2253
    assert abs(cal_prob - 0.2253) > 0.15

    # 3. Verify backend calibration agrees precisely with the saved calibrator object
    logit_val = float(np.log(raw_p / (1.0 - raw_p)))
    direct_cal_prob = float(model_service.calibrator.predict_proba(np.array([[logit_val]]))[0, 1])
    assert cal_prob == pytest.approx(direct_cal_prob, abs=1e-9)


def test_platt_calibration_pipeline_transformation_stages(model_service: FraudModelService):
    """
    Verifies full calibration pipeline:
    raw probability
    -> clip to (0, 1)
    -> logit transformation
    -> saved Platt LogisticRegression
    -> calibrated probability.
    """
    # Verify calibrator coefficients from saved champion artifact
    assert hasattr(model_service.calibrator, "coef_")
    assert hasattr(model_service.calibrator, "intercept_")
    coef = float(model_service.calibrator.coef_[0, 0])
    intercept = float(model_service.calibrator.intercept_[0])
    assert pytest.approx(coef, abs=1e-2) == 0.8300
    assert pytest.approx(intercept, abs=1e-2) == -1.2370

    test_points = [0.0001, 0.0025, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95, 0.999]
    eps = 1e-12

    for p in test_points:
        # Step 1: Clip
        p_clipped = float(np.clip(p, eps, 1.0 - eps))
        # Step 2: Logit
        logit_p = float(np.log(p_clipped / (1.0 - p_clipped)))
        # Step 3: Reshape to (-1, 1)
        arr = np.array([[logit_p]])
        # Step 4: Call saved calibrator
        expected_cal = float(model_service.calibrator.predict_proba(arr)[0, 1])
        # Step 5: Service calibrated probability
        actual_cal = model_service.calibrate_probability(p)

        assert actual_cal == pytest.approx(expected_cal, abs=1e-9)


# -----------------------------------------------------------------------------
# 6. Fraud Risk Bands & Decisions (Section 5)
# -----------------------------------------------------------------------------
def test_fraud_risk_bands_and_decisions(model_service: FraudModelService):
    """Verify dynamic mapping using artifact thresholds."""
    rev_th = model_service.review_threshold
    high_th = model_service.high_threshold

    # LOW Band -> ALLOW
    assert model_service.get_fraud_band(rev_th - 0.01) == "LOW"
    assert model_service.get_fraud_decision("LOW") == "ALLOW"

    # REVIEW Band -> MANUAL_REVIEW
    assert model_service.get_fraud_band(rev_th + 0.01) == "REVIEW"
    assert model_service.get_fraud_decision("REVIEW") == "MANUAL_REVIEW"

    # HIGH Band -> BLOCK
    assert model_service.get_fraud_band(high_th + 0.01) == "HIGH"
    assert model_service.get_fraud_decision("HIGH") == "BLOCK"

    # Standalone module-level helpers
    assert get_fraud_band(rev_th - 0.01) == "LOW"
    assert get_fraud_decision("LOW") == "ALLOW"


# -----------------------------------------------------------------------------
# 7. Schema Validation & Error Handling (Section 3 & 11)
# -----------------------------------------------------------------------------
def test_schema_validation_column_count_mismatch(model_service: FraudModelService):
    """Service must raise FraudModelSchemaError if column count != 459."""
    txn = {"amount": 100.0}
    df = build_fraud_features(txn, history=[])

    # Drop 1 column -> 458 columns
    invalid_df = df.iloc[:, :-1]
    assert invalid_df.shape[1] == 458

    with pytest.raises(FraudModelSchemaError) as exc_info:
        model_service.predict(invalid_df)

    assert "Fraud model feature schema mismatch" in str(exc_info.value)
    assert "459" in str(exc_info.value)
    assert "458" in str(exc_info.value)


def test_schema_validation_column_names_or_order_mismatch(model_service: FraudModelService):
    """Service must raise FraudModelSchemaError if columns are swapped or renamed."""
    txn = {"amount": 100.0}
    df = build_fraud_features(txn, history=[])

    # Swap first two columns
    cols = list(df.columns)
    cols[0], cols[1] = cols[1], cols[0]
    shuffled_df = df[cols]

    with pytest.raises(FraudModelSchemaError) as exc_info:
        model_service.predict(shuffled_df)

    assert "Fraud model feature schema mismatch" in str(exc_info.value)


def test_schema_validation_nan_and_inf_detection(model_service: FraudModelService):
    """Service must reject DataFrames containing NaNs or Infs."""
    txn = {"amount": 100.0}
    df = build_fraud_features(txn, history=[])

    # Introduce NaN
    nan_df = df.copy()
    nan_df.iloc[0, 0] = np.nan
    with pytest.raises(FraudModelSchemaError) as exc_info:
        model_service.predict(nan_df)
    assert "NaN values" in str(exc_info.value)

    # Introduce Inf
    inf_df = df.copy()
    inf_df.iloc[0, 0] = np.inf
    with pytest.raises(FraudModelSchemaError) as exc_info:
        model_service.predict(inf_df)
    assert "infinite values" in str(exc_info.value)


def test_schema_validation_non_numeric_type(model_service: FraudModelService):
    """Service must reject non-numeric feature types."""
    txn = {"amount": 100.0}
    df = build_fraud_features(txn, history=[])

    str_df = df.copy().astype(object)
    str_df.iloc[0, 0] = "string_value"
    with pytest.raises(FraudModelSchemaError) as exc_info:
        model_service.predict(str_df)
    assert "non-numeric" in str(exc_info.value)


# -----------------------------------------------------------------------------
# 8. Backend Startup Validation Test (Section 12)
# -----------------------------------------------------------------------------
def test_backend_startup_validation():
    """Verify backend startup event initializes and validates the fraud model."""
    client = TestClient(app)
    # Ping /health to trigger startup event
    response = client.get("/health")
    assert response.status_code == 200

    service = get_fraud_model_service()
    info = service.info()
    assert info["status"] == "ready"
    assert info["model_name"] == "XGBoost"
    assert info["expected_feature_count"] == 459
