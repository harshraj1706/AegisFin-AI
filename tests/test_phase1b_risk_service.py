import json
import pytest
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.risk_service import Phase1RiskService, classify_risk_band, safe_logit
from app.mapper import request_to_dataframe
from app.schemas import PredictionRequest
from app.main import app, MODEL_PATH


@pytest.fixture
def risk_service():
    return Phase1RiskService(MODEL_PATH)


@pytest.fixture
def sample_payload():
    sample_path = Path(__file__).resolve().parents[1] / "sample_request.json"
    return json.loads(sample_path.read_text(encoding="utf-8"))


# 1. Phase 1B bundle loading test
def test_phase1b_bundle_loading(risk_service):
    assert risk_service is not None
    assert hasattr(risk_service, "bundle")
    expected_keys = {
        "model",
        "preprocessing_state",
        "selected_features",
        "categorical_features",
        "calibrator",
        "calibration_method",
        "risk_policy",
        "metadata",
    }
    assert expected_keys.issubset(set(risk_service.bundle.keys()))


# 2. Model loading test
def test_frozen_xgboost_model_loading(risk_service):
    model = risk_service.model
    assert model is not None
    # Verify model class is XGBoost
    assert type(model).__name__ in {"XGBClassifier", "Booster"}
    assert len(risk_service.selected_features) == 183


# 3. Calibrator loading test
def test_phase1b_calibrator_loading(risk_service):
    calibrator = risk_service.calibrator
    assert calibrator is not None
    assert type(calibrator).__name__ == "LogisticRegression"
    assert risk_service.calibration_method == "sigmoid_platt_scaling"
    # Ensure coefficient and intercept exist from Platt scaling
    assert hasattr(calibrator, "coef_")
    assert hasattr(calibrator, "intercept_")
    assert calibrator.coef_.shape == (1, 1)


# 4. Preprocessing replay test
def test_preprocessing_replay(risk_service, sample_payload):
    req = PredictionRequest(**sample_payload)
    df_raw = request_to_dataframe(req)
    df_transformed = risk_service.transform(df_raw)

    # Feature space and ordering must match selected_features exactly
    assert list(df_transformed.columns) == risk_service.selected_features
    assert len(df_transformed.columns) == 183

    # Check that indicator missing flags were properly constructed
    assert "EXT_SOURCE_1__MISSING" in df_transformed.columns
    # Check that categorical columns have category dtype
    for cat_col in risk_service.preprocessing_state["categorical_cols"]:
        if cat_col in df_transformed.columns:
            assert isinstance(df_transformed[cat_col].dtype, pd.CategoricalDtype)


# 5. Deterministic prediction test
def test_deterministic_prediction(risk_service, sample_payload):
    req = PredictionRequest(**sample_payload)
    df_raw = request_to_dataframe(req)

    res1 = risk_service.predict_risk(df_raw)
    res2 = risk_service.predict_risk(df_raw)

    assert res1["default_probability"] == res2["default_probability"]
    assert res1["raw_probability"] == res2["raw_probability"]
    assert res1["risk_band"] == res2["risk_band"]


# 6. Probability is between 0 and 1
def test_probability_in_range(risk_service, sample_payload):
    req = PredictionRequest(**sample_payload)
    df_raw = request_to_dataframe(req)

    res = risk_service.predict_risk(df_raw)
    cal_prob = res["default_probability"]
    raw_prob = res["raw_probability"]

    assert 0.0 <= cal_prob <= 1.0
    assert 0.0 <= raw_prob <= 1.0


# 7. Risk-band boundary behavior test
def test_risk_band_boundary_behavior():
    thresholds = {"LOW_MAX": 0.15, "MODERATE_MAX": 0.40, "ELEVATED_MAX": 0.70}

    # Explicit boundary tests required by Phase 1B specification:
    boundary_cases = [
        (0.149999, "LOW"),
        (0.150000, "MODERATE"),
        (0.399999, "MODERATE"),
        (0.400000, "ELEVATED"),
        (0.699999, "ELEVATED"),
        (0.700000, "HIGH"),
        (0.0, "LOW"),
        (0.999999, "HIGH"),
        (1.0, "HIGH"),
    ]

    for p_val, expected_band in boundary_cases:
        actual_band = classify_risk_band(p_val, thresholds)
        assert actual_band == expected_band, f"Failed at {p_val}: expected {expected_band}, got {actual_band}"

    # Invalid range bounds should raise ValueError
    with pytest.raises(ValueError):
        classify_risk_band(-0.01)

    with pytest.raises(ValueError):
        classify_risk_band(1.01)


# 8. API response schema test
def test_api_response_schema(sample_payload):
    client = TestClient(app)
    response = client.post("/api/v1/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()

    assert "default_probability" in data
    assert "risk_band" in data
    assert "model_name" in data
    assert "model_version" in data
    assert "calibration_version" in data
    assert "policy_version" in data
    assert "feature_count" in data

    assert data["risk_band"] in {"LOW", "MODERATE", "ELEVATED", "HIGH"}
    assert 0.0 <= data["default_probability"] <= 1.0


# 9. Model / calibration / policy version metadata test
def test_version_metadata(risk_service, sample_payload):
    info = risk_service.info()
    assert info["model_name"] == "XGBoost"
    assert info["model_version"] == "credit-xgb-v1.0.0"
    assert info["calibration_version"] == "credit-calibration-v1.0.0"
    assert info["policy_version"] == "credit-risk-policy-v1.0.0"
    assert info["feature_count"] == 183
    assert info["calibration_method"] == "sigmoid_platt_scaling"

    client = TestClient(app)
    health = client.get("/health").json()
    assert health["model_version"] == "credit-xgb-v1.0.0"
    assert health["calibration_version"] == "credit-calibration-v1.0.0"
    assert health["policy_version"] == "credit-risk-policy-v1.0.0"


# 10. Missing / invalid input handling test
def test_invalid_input_handling():
    client = TestClient(app)

    # Empty payload
    res_empty = client.post("/api/v1/predict", json={})
    assert res_empty.status_code == 422

    # Negative loan amount
    bad_payload = {
        "application": {
            "contract_type": "Cash loans",
            "credit_amount": -5000.0,
        }
    }
    res_bad = client.post("/api/v1/predict", json=bad_payload)
    assert res_bad.status_code == 422

    # Unknown unexpected field (extra forbidden in CustomerApplication)
    extra_field_payload = {
        "application": {
            "non_existent_field": 12345,
        }
    }
    res_extra = client.post("/api/v1/predict", json=extra_field_payload)
    assert res_extra.status_code == 422
