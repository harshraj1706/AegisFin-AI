import json
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app, get_service
from app.mapper import request_to_dataframe
from app.schemas import PredictionRequest
from app.feature_engineering import engineer_features


def test_model_service_direct_inference():
    service = get_service()
    info = service.info()
    assert info["model_name"] == "XGBoost"
    assert info["model_version"] == "credit-xgb-v1.0.0"
    assert info["feature_count"] == 183

    sample_path = Path(__file__).resolve().parents[1] / "sample_request.json"
    req_data = json.loads(sample_path.read_text())
    req = PredictionRequest(**req_data)
    df = request_to_dataframe(req)
    
    prob = service.predict_probability(df)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_fastapi_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_name"] == "XGBoost"
    assert data["model_version"] == "credit-xgb-v1.0.0"
    assert data["calibration_version"] == "credit-calibration-v1.0.0"


def test_fastapi_predict_endpoint():
    client = TestClient(app)
    sample_path = Path(__file__).resolve().parents[1] / "sample_request.json"
    req_data = json.loads(sample_path.read_text())

    response = client.post("/api/v1/predict", json=req_data)
    assert response.status_code == 200
    res = response.json()
    assert "default_probability" in res
    assert 0.0 <= res["default_probability"] <= 1.0
    assert "risk_band" in res
    assert res["risk_band"] in {"LOW", "MODERATE", "ELEVATED", "HIGH"}
    assert res["model_name"] == "XGBoost"
    assert res["model_version"] == "credit-xgb-v1.0.0"
    assert res["calibration_version"] == "credit-calibration-v1.0.0"
    assert res["policy_version"] == "credit-risk-policy-v1.0.0"
    assert res["feature_count"] == 183


def test_feature_engineering_ratios():
    sample_path = Path(__file__).resolve().parents[1] / "sample_request.json"
    req_data = json.loads(sample_path.read_text())
    req = PredictionRequest(**req_data)
    df_raw = request_to_dataframe(req)
    df_eng = engineer_features(df_raw)

    assert "CREDIT_INCOME_RATIO" in df_eng.columns
    assert "ANNUITY_INCOME_RATIO" in df_eng.columns
    assert "EXT_SOURCE_MEAN" in df_eng.columns
    assert df_eng.loc[0, "CREDIT_INCOME_RATIO"] == 800000 / 450000
    assert df_eng.loc[0, "ANNUITY_INCOME_RATIO"] == 35000 / 450000
