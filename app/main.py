from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Depends, status

from .mapper import request_to_dataframe
from .risk_service import Phase1RiskService
from .schemas import PredictionRequest, PredictionResponse
from .supabase_client import check_supabase_connection, check_database_health
from .auth import get_current_user, signup_user, login_user, SignupRequest, LoginRequest


from contextlib import asynccontextmanager

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.getenv(
        "AEGISFIN_MODEL_PATH",
        BASE_DIR / "models" / "aegisfin_phase1b_calibrated_model.pkl",
    )
)

_service: Optional[Phase1RiskService] = None


def get_service() -> Phase1RiskService:
    global _service
    if _service is None:
        _service = Phase1RiskService(MODEL_PATH)
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize & validate Phase 1 Risk Service
    get_service()
    # 2. Initialize & validate Phase 2 Fraud Model Service (artifact, schema, calibration, policy)
    from .fraud_model_service import get_fraud_model_service
    get_fraud_model_service()
    yield


app = FastAPI(
    title="AegisFin-AI Phase 1 & 2 API",
    version="1.0.0",
    lifespan=lifespan,
)

from .fraud_router import router as fraud_router

app.include_router(fraud_router)


@app.get("/health")
def health():
    try:
        service = get_service()
        fraud_loaded = False
        try:
            from .fraud_model_service import get_fraud_model_service
            get_fraud_model_service()
            fraud_loaded = True
        except Exception:
            fraud_loaded = False

        return {
            "status": "ok",
            "model_loaded": True,
            "fraud_model_loaded": fraud_loaded,
            **service.info(),
        }
    except Exception as exc:
        return {
            "status": "degraded",
            "model_loaded": False,
            "fraud_model_loaded": False,
            "error": str(exc),
        }


@app.get("/health/supabase")
def health_supabase():
    result = check_supabase_connection()
    if result.get("status") != "connected":
        raise HTTPException(status_code=503, detail=result)
    return result


@app.get("/health/supabase/db")
def health_supabase_db():
    result = check_database_health()
    if result.get("status") != "healthy":
        raise HTTPException(status_code=503, detail=result)
    return result


# -----------------------------------------------------------------------------
# Supabase Authentication Endpoints
# -----------------------------------------------------------------------------
@app.post("/api/v1/auth/signup", status_code=status.HTTP_201_CREATED)
def auth_signup(request: SignupRequest):
    try:
        res = signup_user(request.email, request.password, request.full_name)
        return res
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Signup failed: {exc}") from exc


@app.post("/api/v1/auth/login")
def auth_login(request: LoginRequest):
    try:
        res = login_user(request.email, request.password)
        return res
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Login failed: {exc}") from exc


@app.get("/api/v1/auth/me")
def auth_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    return {
        "user_id": current_user["user_id"],
        "email": current_user["email"],
        "authenticated": True,
        "full_name": current_user.get("full_name"),
        "role": current_user.get("role", "analyst"),
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
