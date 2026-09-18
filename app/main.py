from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException

from .mapper import request_to_dataframe
from .risk_service import Phase1RiskService
from .schemas import PredictionRequest, PredictionResponse


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.getenv(
        "AEGISFIN_MODEL_PATH",
        BASE_DIR / "models" / "aegisfin_phase1b_calibrated_model.pkl",
    )
)

app = FastAPI(
    title="AegisFin-AI Phase 1 API",
    version="1.0.0",
)

_service: Optional[Phase1RiskService] = None


def get_service() -> Phase1RiskService:
    global _service
    if _service is None:
        _service = Phase1RiskService(MODEL_PATH)
    return _service


@app.get("/health")
def health():
    try:
        service = get_service()
        return {
            "status": "ok",
            "model_loaded": True,
            **service.info(),
        }
    except Exception as exc:
        return {
            "status": "degraded",
            "model_loaded": False,
            "error": str(exc),
        }


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):
    try:
        service = get_service()
        raw_df = request_to_dataframe(request)
        result = service.predict_risk(raw_df)

        return PredictionResponse(
            default_probability=result["default_probability"],
            risk_band=result["risk_band"],
            model_name=result["model_name"],
            model=result["model_name"],
            model_version=result["model_version"],
            calibration_version=result["calibration_version"],
            policy_version=result["policy_version"],
            feature_count=result["feature_count"],
            raw_probability=result.get("raw_probability"),
            calibration_method=result.get("calibration_method"),
        )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Prediction failed: {exc}",
        ) from exc
