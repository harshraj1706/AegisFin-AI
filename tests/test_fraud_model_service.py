from __future__ import annotations

import copy
import uuid
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.production_feature_definitions import (
    VALID_FEATURE_NAMES,
    generate_production_features,
)
from app.fraud_model_service import (
    FraudModelService,
    FraudModelSchemaError,
    FraudModelArtifactError,
    FraudModelCalibrationError,
    LegacyFraudModelService,
    get_fraud_model_service,
    get_legacy_fraud_model_service,
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
    """Verify singleton initialization, metadata, and 62-feature production contract."""
    assert model_service.model_name == "XGBoost"
    assert model_service.model_version == "2.1.0"
    assert model_service.calibration_version == "2.1.0"
    assert model_service.policy_version == "2.0.0"
    assert model_service.feature_contract_version == "2.1.0"
    assert model_service.calibration_method == "Platt"
    assert model_service.calibrator_type == "sigmoid"
    assert model_service.expected_feature_count == 62
    assert len(model_service.feature_names) == 62
    assert model_service.feature_names == VALID_FEATURE_NAMES

    # Verify policy thresholds loaded dynamically from configs/fraud_risk_policy_v2.json
    assert model_service.review_threshold == 0.10
    assert model_service.high_threshold == 0.80
    assert model_service.decision_threshold == 0.50


def test_model_service_singleton():
    """Verify get_fraud_model_service returns the cached singleton."""
    s1 = get_fraud_model_service()
    s2 = get_fraud_model_service()
    assert s1 is s2


# -----------------------------------------------------------------------------
# 2. Integration: Feature Engine -> Model Service (62 Features)
# -----------------------------------------------------------------------------
def test_feature_engine_to_model_service_pipeline(model_service: FraudModelService):
    """
    Pass deterministic test transaction through:
    transaction -> 62 production feature generator -> model service.
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
        "ip_address": "192.168.1.1",
        "country": "US",
    }

    # Step 5: Feature Engine (62 features)
    features = generate_production_features(txn, history=[])
    assert len(features) == 62

    # Step 6: Model Service
    res = model_service.predict(features)

    # Verify canonical Phase 2 fields
    assert "raw_fraud_probability" in res
    assert "calibrated_fraud_probability" in res
    assert "risk_band" in res
    assert "recommended_action" in res
    assert "feature_contract_version" in res
    assert "calibrator_type" in res
    assert "scored_at" in res

    # Verify legacy compatibility fields
    assert "raw_probability" in res
    assert "fraud_probability" in res
    assert "fraud_band" in res
    assert "decision" in res
    assert "inference_latency_ms" in res
    assert "model_inference_latency_ms" in res

    # Numerical validity
    assert 0.0 <= res["raw_fraud_probability"] <= 1.0
    assert 0.0 <= res["calibrated_fraud_probability"] <= 1.0
    assert res["risk_band"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert res["recommended_action"] in {"AUTO_APPROVE", "STEP_UP_AUTH", "MANUAL_REVIEW", "HARD_DECLINE"}
    assert res["fraud_band"] in {"LOW", "REVIEW", "HIGH"}
    assert res["decision"] in {"ALLOW", "MANUAL_REVIEW", "BLOCK"}
    assert res["feature_count"] == 62
    assert res["model_name"] == "XGBoost"
    assert res["model_version"] == "2.1.0"
    assert res["inference_latency_ms"] > 0.0


# -----------------------------------------------------------------------------
# 3. Determinism Test
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

    f1 = generate_production_features(txn, history=[])
    f2 = generate_production_features(txn, history=[])

    res1 = model_service.predict(f1)
    res2 = model_service.predict(f2)

    assert res1["raw_fraud_probability"] == res2["raw_fraud_probability"]
    assert res1["calibrated_fraud_probability"] == res2["calibrated_fraud_probability"]
    assert res1["risk_band"] == res2["risk_band"]
    assert res1["recommended_action"] == res2["recommended_action"]
    assert res1["fraud_band"] == res2["fraud_band"]
    assert res1["decision"] == res2["decision"]


# -----------------------------------------------------------------------------
# 4. Historical Behavior & Anti-Leakage
# -----------------------------------------------------------------------------
def test_historical_behavior_and_anti_leakage(model_service: FraudModelService):
    """
    Verify historical influence on model prediction:
    - Transaction A (earlier, $500)
    - Transaction B (later, $1000) sees A -> customer_tx_count_24h=1
    - Transaction C (earlier) vs Future Transaction D ($99,999) -> D does not affect C
    """
    cust_id = f"TEST_HIST_{uuid.uuid4().hex[:8]}"

    txn_a = {
        "customer_id": cust_id,
        "card_id": "9633",
        "amount": 500.0,
        "transaction_timestamp": "2026-09-20T10:00:00Z",
    }
    txn_b = {
        "customer_id": cust_id,
        "card_id": "9633",
        "amount": 1000.0,
        "transaction_timestamp": "2026-09-20T11:00:00Z",
    }

    # B can use A's historical state
    f_b_with_a = generate_production_features(txn_b, history=[txn_a])
    res_b = model_service.predict(f_b_with_a)
    assert f_b_with_a["customer_tx_count_24h"] == 1.0
    assert f_b_with_a["customer_amount_mean"] == 500.0
    assert 0.0 <= res_b["calibrated_fraud_probability"] <= 1.0

    # Anti-leakage: C at 12:00 must NOT see future transaction D at 13:00
    txn_c = {
        "customer_id": cust_id,
        "card_id": "9633",
        "amount": 300.0,
        "transaction_timestamp": "2026-09-20T12:00:00Z",
    }
    txn_d = {
        "customer_id": cust_id,
        "card_id": "9633",
        "amount": 99999.0,
        "transaction_timestamp": "2026-09-20T13:00:00Z",
    }

    f_c_with_d = generate_production_features(txn_c, history=[txn_d])
    res_c = model_service.predict(f_c_with_d)
    assert f_c_with_d["customer_tx_count_24h"] == 0.0
    assert f_c_with_d["customer_is_new"] == 1.0
    assert 0.0 <= res_c["calibrated_fraud_probability"] <= 1.0


# -----------------------------------------------------------------------------
# 5. Calibration Method & Boundary Checks
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
    - Live calibration applies logit transform: logit(p) = log(p / (1 - p))
    - Then calls saved Platt LogisticRegression calibrator.
    - Yields calibrated probability approximately 0.000355.
    - Agrees precisely with saved calibrator object.
    """
    raw_p = 0.0025
    cal_prob = model_service.calibrate_probability(raw_p)

    # Verify backend calibration agrees precisely with the saved calibrator object
    logit_val = float(np.log(raw_p / (1.0 - raw_p)))
    direct_cal_prob = float(model_service.calibrator.predict_proba(np.array([[logit_val]]))[0, 1])
    assert cal_prob == pytest.approx(direct_cal_prob, abs=1e-9)
    assert 0.0 <= cal_prob <= 1.0


def test_platt_calibration_pipeline_transformation_stages(model_service: FraudModelService):
    """
    Verifies full calibration pipeline:
    raw probability
    -> clip to (0, 1)
    -> logit transformation
    -> saved Platt LogisticRegression
    -> calibrated probability.
    """
    # Verify calibrator coefficients from saved Phase 2 artifact
    assert hasattr(model_service.calibrator, "coef_")
    assert hasattr(model_service.calibrator, "intercept_")
    coef = float(model_service.calibrator.coef_[0, 0])
    intercept = float(model_service.calibrator.intercept_[0])
    assert pytest.approx(coef, abs=1e-2) == 1.06
    assert pytest.approx(intercept, abs=1e-2) == -1.59

    test_points = [0.0001, 0.0025, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95, 0.999]
    eps = 1e-15

    for p in test_points:
        p_clipped = float(np.clip(p, eps, 1.0 - eps))
        logit_p = float(np.log(p_clipped / (1.0 - p_clipped)))
        arr = np.array([[logit_p]])
        expected_cal = float(model_service.calibrator.predict_proba(arr)[0, 1])
        actual_cal = model_service.calibrate_probability(p)
        assert actual_cal == pytest.approx(expected_cal, abs=1e-9)


# -----------------------------------------------------------------------------
# 6. Fraud Risk Bands & Decisions
# -----------------------------------------------------------------------------
def test_fraud_risk_bands_and_decisions(model_service: FraudModelService):
    """Verify dynamic mapping using Phase 2 policy thresholds."""
    # LOW Band -> AUTO_APPROVE (legacy ALLOW)
    band_low, act_low = model_service.evaluate_risk_policy(0.05)
    assert band_low == "LOW"
    assert act_low == "AUTO_APPROVE"
    assert model_service.get_fraud_band(0.05) == "LOW"
    assert model_service.get_fraud_decision("LOW") == "ALLOW"

    # MEDIUM Band -> STEP_UP_AUTH (legacy MANUAL_REVIEW)
    band_med, act_med = model_service.evaluate_risk_policy(0.25)
    assert band_med == "MEDIUM"
    assert act_med == "STEP_UP_AUTH"
    assert model_service.get_fraud_band(0.25) == "REVIEW"
    assert model_service.get_fraud_decision("MEDIUM") == "MANUAL_REVIEW"

    # HIGH Band -> MANUAL_REVIEW (legacy MANUAL_REVIEW)
    band_hi, act_hi = model_service.evaluate_risk_policy(0.60)
    assert band_hi == "HIGH"
    assert act_hi == "MANUAL_REVIEW"
    assert model_service.get_fraud_band(0.60) == "REVIEW"
    assert model_service.get_fraud_decision("HIGH") == "MANUAL_REVIEW"

    # CRITICAL Band -> HARD_DECLINE (legacy BLOCK)
    band_crit, act_crit = model_service.evaluate_risk_policy(0.95)
    assert band_crit == "CRITICAL"
    assert act_crit == "HARD_DECLINE"
    assert model_service.get_fraud_band(0.95) == "HIGH"
    assert model_service.get_fraud_decision("CRITICAL") == "BLOCK"

    # Standalone module-level helpers
    assert get_fraud_band(0.05) == "LOW"
    assert get_fraud_decision("LOW") == "ALLOW"
    assert get_fraud_band(0.95) == "HIGH"
    assert get_fraud_decision("CRITICAL") == "BLOCK"


# -----------------------------------------------------------------------------
# 7. Schema Validation & Error Handling (62 Features)
# -----------------------------------------------------------------------------
def test_schema_validation_column_count_mismatch(model_service: FraudModelService):
    """Service must raise FraudModelSchemaError if column count != 62."""
    txn = {"amount": 100.0, "customer_id": "c1", "card_id": "card1"}
    feats = generate_production_features(txn, history=[])
    df = pd.DataFrame([feats])[VALID_FEATURE_NAMES]

    # Drop 1 column -> 61 columns
    invalid_df = df.iloc[:, :-1]
    assert invalid_df.shape[1] == 61

    with pytest.raises(FraudModelSchemaError) as exc_info:
        model_service.predict(invalid_df)

    assert "Fraud model feature schema mismatch" in str(exc_info.value)
    assert "62" in str(exc_info.value)
    assert "61" in str(exc_info.value)


def test_schema_validation_column_names_or_order_mismatch(model_service: FraudModelService):
    """Service must raise FraudModelSchemaError if columns are swapped or renamed."""
    txn = {"amount": 100.0, "customer_id": "c1", "card_id": "card1"}
    feats = generate_production_features(txn, history=[])
    df = pd.DataFrame([feats])[VALID_FEATURE_NAMES]

    # Swap first two columns
    cols = list(df.columns)
    cols[0], cols[1] = cols[1], cols[0]
    shuffled_df = df[cols]

    with pytest.raises(FraudModelSchemaError) as exc_info:
        model_service.predict(shuffled_df)

    assert "Fraud model feature schema mismatch" in str(exc_info.value)


def test_schema_validation_nan_and_inf_detection(model_service: FraudModelService):
    """Service must reject DataFrames containing NaNs or Infs."""
    txn = {"amount": 100.0, "customer_id": "c1", "card_id": "card1"}
    feats = generate_production_features(txn, history=[])
    df = pd.DataFrame([feats])[VALID_FEATURE_NAMES]

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
    txn = {"amount": 100.0, "customer_id": "c1", "card_id": "card1"}
    feats = generate_production_features(txn, history=[])
    df = pd.DataFrame([feats])[VALID_FEATURE_NAMES]

    str_df = df.copy().astype(object)
    str_df.iloc[0, 0] = "string_value"
    with pytest.raises(FraudModelSchemaError) as exc_info:
        model_service.predict(str_df)
    assert "non-numeric" in str(exc_info.value)


# -----------------------------------------------------------------------------
# 8. Backend Startup Validation Test
# -----------------------------------------------------------------------------
def test_backend_startup_validation():
    """Verify backend startup event initializes and validates the Phase 2 fraud model."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200

    service = get_fraud_model_service()
    info = service.info()
    assert info["status"] == "ready"
    assert info["model_name"] == "XGBoost"
    assert info["model_version"] == "2.1.0"
    assert info["expected_feature_count"] == 62


# -----------------------------------------------------------------------------
# 9. Legacy 459-Feature Benchmark Model Service (Offline Reference)
# -----------------------------------------------------------------------------
def test_legacy_459_feature_model_service():
    """
    Verify legacy 459-feature IEEE-CIS model artifact remains loaded and available
    in LegacyFraudModelService for offline reference benchmarking, but is separate
    from the live Phase 2 service.
    """
    legacy_service = get_legacy_fraud_model_service()
    assert legacy_service.expected_feature_count == 459
    assert len(legacy_service.feature_names) == 459
    assert legacy_service.model_version == "phase2-xgb-legacy-459"
    assert hasattr(legacy_service.model, "predict_proba")
