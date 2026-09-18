from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .mapper import request_to_dataframe
from .model_service import Phase1ModelService
from .schemas import PredictionRequest, PredictionResponse


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.getenv(
        "AEGISFIN_MODEL_PATH",
        BASE_DIR / "models" / "aegisfin_phase1_final_model.pkl",
    )
)

app = FastAPI(
    title="AegisFin-AI Phase 1 API",
    version="1.0.0",
)

_service = None


def get_service() -> Phase1ModelService:
    global _service
    if _service is None:
        _service = Phase1ModelService(MODEL_PATH)
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
        probability = service.predict_probability(raw_df)
        info = service.info()

        return PredictionResponse(
            default_probability=probability,
            model=info["model_name"],
            model_version=info["model_version"],
            feature_count=info["feature_count"],
        )

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Prediction failed: {exc}",
        ) from exc
