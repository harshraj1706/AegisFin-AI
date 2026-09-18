"""
AegisFin-AI - Phase 1 ML Integration
====================================
Single-file VS Code integration for the final Phase 1 model.

Run API:
    uvicorn ml_integration_phase1:app --reload

Expected model file:
    models/aegisfin_phase1_final_model.pkl

The .pkl bundle must contain:
    model
    preprocessing_state
    selected_features
    metadata

Important:
- The customer-facing schema is human-readable.
- Internal/bureau enrichment can be supplied separately.
- raw_overrides is intended only for trusted backend fields.
- The API returns model probability only; approve/reject logic belongs to a
  later business-policy layer.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


# ============================================================================
# 1. PATHS / APP
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.getenv(
        "AEGISFIN_MODEL_PATH",
        BASE_DIR / "models" / "aegisfin_phase1_final_model.pkl",
    )
)

app = FastAPI(
    title="AegisFin-AI Phase 1 ML API",
    version="1.0.0",
)


# ============================================================================
# 2. NEW RICH PRODUCTION-FACING SCHEMA
# ============================================================================

class CustomerApplication(BaseModel):
    """Information that a customer can reasonably provide."""

    model_config = ConfigDict(extra="forbid")

    # Loan / application
    contract_type: Optional[str] = "Cash loans"
    credit_amount: Optional[float] = Field(None, ge=0)
    annuity_amount: Optional[float] = Field(None, ge=0)
    goods_price: Optional[float] = Field(None, ge=0)

    # Personal / family
    age_years: Optional[float] = Field(None, ge=18, le=100)
    gender: Optional[str] = None
    children: Optional[int] = Field(None, ge=0)
    family_members: Optional[float] = Field(None, ge=0)
    family_status: Optional[str] = None

    # Education / income / employment
    education_type: Optional[str] = None
    income_type: Optional[str] = None
    annual_income: Optional[float] = Field(None, ge=0)
    occupation_type: Optional[str] = None
    employment_years: Optional[float] = Field(None, ge=0, le=100)
    organization_type: Optional[str] = None

    # Housing / ownership
    housing_type: Optional[str] = None
    owns_car: Optional[bool] = None
    owns_realty: Optional[bool] = None
    own_car_age_years: Optional[float] = Field(None, ge=0, le=100)

    # Contact / process
    work_phone: Optional[bool] = None
    weekday_application: Optional[str] = None


class EnrichmentData(BaseModel):
    """
    Important model variables that should normally come from trusted
    bank/internal/bureau systems rather than being manually typed by a customer.

    These fields were added to make the integration schema richer, using the
    important variables identified in the existing Phase 1 feature-importance /
    SHAP work.
    """

    model_config = ConfigDict(extra="allow")

    # External risk-source features
    ext_source_1: Optional[float] = Field(None, ge=0, le=1)
    ext_source_2: Optional[float] = Field(None, ge=0, le=1)
    ext_source_3: Optional[float] = Field(None, ge=0, le=1)

    # Customer timeline / region
    days_id_publish: Optional[float] = None
    days_last_phone_change: Optional[float] = None
    days_registration: Optional[float] = None
    region_rating_client_w_city: Optional[float] = None
    region_population_relative: Optional[float] = None
    reg_city_not_live_city: Optional[bool] = None

    # Bureau activity
    credit_bureau_quarter: Optional[float] = Field(None, ge=0)
    credit_bureau_year: Optional[float] = Field(None, ge=0)

    # Social-circle risk indicators
    def_30_cnt_social_circle: Optional[float] = Field(None, ge=0)
    def_60_cnt_social_circle: Optional[float] = Field(None, ge=0)

    # Document / phone indicators
    flag_document_3: Optional[bool] = None
    flag_work_phone: Optional[bool] = None

    def to_raw_dict(self) -> Dict[str, Any]:
        mapping = {
            "ext_source_1": "EXT_SOURCE_1",
            "ext_source_2": "EXT_SOURCE_2",
            "ext_source_3": "EXT_SOURCE_3",
            "days_id_publish": "DAYS_ID_PUBLISH",
            "days_last_phone_change": "DAYS_LAST_PHONE_CHANGE",
            "days_registration": "DAYS_REGISTRATION",
            "region_rating_client_w_city": "REGION_RATING_CLIENT_W_CITY",
            "region_population_relative": "REGION_POPULATION_RELATIVE",
            "reg_city_not_live_city": "REG_CITY_NOT_LIVE_CITY",
            "credit_bureau_quarter": "AMT_REQ_CREDIT_BUREAU_QRT",
            "credit_bureau_year": "AMT_REQ_CREDIT_BUREAU_YEAR",
            "def_30_cnt_social_circle": "DEF_30_CNT_SOCIAL_CIRCLE",
            "def_60_cnt_social_circle": "DEF_60_CNT_SOCIAL_CIRCLE",
            "flag_document_3": "FLAG_DOCUMENT_3",
            "flag_work_phone": "FLAG_WORK_PHONE",
        }

        source = self.model_dump(exclude_none=True)
        result: Dict[str, Any] = {}

        for key, value in source.items():
            raw_name = mapping[key]
            if key in {
                "reg_city_not_live_city",
                "flag_document_3",
                "flag_work_phone",
            }:
                result[raw_name] = int(bool(value))
            else:
                result[raw_name] = value

        return result


class PredictionRequest(BaseModel):
    """Complete Phase 1 prediction request."""

    model_config = ConfigDict(extra="forbid")

    application: CustomerApplication
    enrichment: Optional[EnrichmentData] = None
    raw_overrides: Dict[str, Any] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    default_probability: float
    model: str
    model_version: str
    feature_count: int


# Resolve postponed type annotations explicitly. This also makes the module
# robust when loaded dynamically by VS Code / test runners.
CustomerApplication.model_rebuild()
EnrichmentData.model_rebuild()
PredictionRequest.model_rebuild()
PredictionResponse.model_rebuild()


# ============================================================================
# 3. CUSTOMER SCHEMA -> TRAINING RAW COLUMN MAPPING
# ============================================================================

CUSTOMER_TO_RAW = {
    "contract_type": "NAME_CONTRACT_TYPE",
    "credit_amount": "AMT_CREDIT",
    "annuity_amount": "AMT_ANNUITY",
    "goods_price": "AMT_GOODS_PRICE",
    "gender": "CODE_GENDER",
    "children": "CNT_CHILDREN",
    "family_members": "CNT_FAM_MEMBERS",
    "family_status": "NAME_FAMILY_STATUS",
    "education_type": "NAME_EDUCATION_TYPE",
    "income_type": "NAME_INCOME_TYPE",
    "annual_income": "AMT_INCOME_TOTAL",
    "occupation_type": "OCCUPATION_TYPE",
    "organization_type": "ORGANIZATION_TYPE",
    "housing_type": "NAME_HOUSING_TYPE",
    "weekday_application": "WEEKDAY_APPR_PROCESS_START",
}


def request_to_raw_dict(request: PredictionRequest) -> Dict[str, Any]:
    app_data = request.application
    raw: Dict[str, Any] = {}

    for field_name, raw_name in CUSTOMER_TO_RAW.items():
        value = getattr(app_data, field_name)
        if value is not None:
            raw[raw_name] = value

    # Training representation: categorical ownership flags are Y/N.
    if app_data.owns_car is not None:
        raw["FLAG_OWN_CAR"] = "Y" if app_data.owns_car else "N"

    if app_data.owns_realty is not None:
        raw["FLAG_OWN_REALTY"] = "Y" if app_data.owns_realty else "N"

    # Numeric binary training column.
    if app_data.work_phone is not None:
        raw["FLAG_WORK_PHONE"] = int(bool(app_data.work_phone))

    # Convert human-friendly years into the signed day-count format
    # used by the original training data.
    if app_data.age_years is not None:
        raw["DAYS_BIRTH"] = -float(app_data.age_years) * 365.25

    if app_data.employment_years is not None:
        raw["DAYS_EMPLOYED"] = -float(app_data.employment_years) * 365.25

    if app_data.own_car_age_years is not None:
        raw["OWN_CAR_AGE"] = float(app_data.own_car_age_years)

    if request.enrichment is not None:
        raw.update(request.enrichment.to_raw_dict())

    # Backend-only escape hatch for additional trusted raw columns.
    raw.update(request.raw_overrides)
    return raw


def request_to_dataframe(request: PredictionRequest) -> pd.DataFrame:
    return pd.DataFrame([request_to_raw_dict(request)])


# ============================================================================
# 4. SAME FEATURE ENGINEERING USED DURING INTEGRATION
# ============================================================================


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    out = pd.Series(np.nan, index=a.index, dtype="float64")
    valid = a.notna() & b.notna() & (b != 0)
    out.loc[valid] = a.loc[valid] / b.loc[valid]
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED_ANOMALY"] = (
            df["DAYS_EMPLOYED"] == 365243
        ).astype("int8")
        df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

    if "DAYS_EMPLOYED" in df.columns:
        df["EMPLOYMENT_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["CREDIT_INCOME_RATIO"] = safe_divide(
            df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"]
        )

    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["ANNUITY_INCOME_RATIO"] = safe_divide(
            df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"]
        )

    if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(df.columns):
        df["CREDIT_ANNUITY_RATIO"] = safe_divide(
            df["AMT_CREDIT"], df["AMT_ANNUITY"]
        )

    if {"AMT_GOODS_PRICE", "AMT_CREDIT"}.issubset(df.columns):
        df["GOODS_CREDIT_RATIO"] = safe_divide(
            df["AMT_GOODS_PRICE"], df["AMT_CREDIT"]
        )

    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(df.columns):
        df["INCOME_PER_FAMILY_MEMBER"] = safe_divide(
            df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"] + 1
        )

    if {"AMT_INCOME_TOTAL", "CNT_CHILDREN"}.issubset(df.columns):
        df["INCOME_PER_CHILD"] = safe_divide(
            df["AMT_INCOME_TOTAL"], df["CNT_CHILDREN"] + 1
        )

    ext_cols = [
        c
        for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        if c in df.columns
    ]

    if ext_cols:
        df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
        df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)
        df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)
        df["EXT_SOURCE_MAX"] = df[ext_cols].max(axis=1)

    return df


# ============================================================================
# 5. MODEL SERVICE
# ============================================================================

class Phase1ModelService:
    """Loads the saved Phase 1 model bundle and reproduces its preprocessing."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        with self.model_path.open("rb") as f:
            self.bundle: Dict[str, Any] = pickle.load(f)

        required = {
            "model",
            "preprocessing_state",
            "selected_features",
            "metadata",
        }
        missing = required - set(self.bundle)
        if missing:
            raise ValueError(
                f"Invalid Phase 1 model bundle. Missing keys: {sorted(missing)}"
            )

        self.model = self.bundle["model"]
        self.preprocessing_state = self.bundle["preprocessing_state"]
        self.selected_features = list(self.bundle["selected_features"])
        self.metadata = self.bundle["metadata"]

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        X = engineer_features(raw_df.copy())
        state = self.preprocessing_state

        X = X.drop(columns=state["drop_cols"], errors="ignore")

        # Recreate the missingness indicators used by training.
        for col in state["indicator_cols"]:
            if col in X.columns:
                X[f"{col}__MISSING"] = X[col].isna().astype("int8")

        for col in state["numeric_cols"]:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")

        for col in state["categorical_cols"]:
            if col in X.columns:
                values = X[col].astype("string").fillna("__MISSING__")
                X[col] = pd.Categorical(
                    values,
                    categories=state["category_vocab"][col],
                )

        # Exact same feature order/count as training.
        X = X.reindex(columns=self.selected_features)

        # Recreate categorical dtype after reindexing so newly missing columns
        # retain the training category representation.
        for col in state["categorical_cols"]:
            if col in X.columns:
                X[col] = pd.Categorical(
                    X[col].astype("string").fillna("__MISSING__"),
                    categories=state["category_vocab"][col],
                )

        return X

    def predict_probability(self, raw_df: pd.DataFrame) -> float:
        X = self.transform(raw_df)
        probability = self.model.predict_proba(X)[0, 1]
        return float(probability)

    def info(self) -> Dict[str, Any]:
        return {
            "model_name": self.metadata.get("model_name", "XGBoost"),
            "model_version": self.metadata.get("model_version", "phase1"),
            "feature_count": len(self.selected_features),
        }


# ============================================================================
# 6. LAZY MODEL LOADING
# ============================================================================

_service: Optional[Phase1ModelService] = None


def get_service() -> Phase1ModelService:
    global _service
    if _service is None:
        _service = Phase1ModelService(MODEL_PATH)
    return _service


# ============================================================================
# 7. API ENDPOINTS
# ============================================================================

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


@app.post("/api/v1/predict", response_model=PredictionResponse)
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


# ============================================================================
# 8. SAMPLE REQUEST
# ============================================================================

SAMPLE_REQUEST = {
    "application": {
        "contract_type": "Cash loans",
        "gender": "M",
        "owns_car": True,
        "owns_realty": True,
        "age_years": 32,
        "children": 1,
        "family_members": 3,
        "annual_income": 450000,
        "credit_amount": 800000,
        "annuity_amount": 35000,
        "goods_price": 700000,
        "income_type": "Working",
        "education_type": "Higher education",
        "family_status": "Married",
        "housing_type": "House / apartment",
        "occupation_type": "Managers",
        "employment_years": 6,
        "organization_type": "Business Entity Type 3",
        "own_car_age_years": 4,
        "work_phone": True,
        "weekday_application": "WEDNESDAY",
    },
    "enrichment": {
        "ext_source_1": 0.52,
        "ext_source_2": 0.62,
        "ext_source_3": 0.71,
        "days_id_publish": -1200,
        "days_last_phone_change": -250,
        "days_registration": -3000,
        "region_rating_client_w_city": 2,
        "region_population_relative": 0.02,
        "reg_city_not_live_city": False,
        "credit_bureau_quarter": 0,
        "credit_bureau_year": 1,
        "def_30_cnt_social_circle": 0,
        "def_60_cnt_social_circle": 0,
        "flag_document_3": True,
        "flag_work_phone": True,
    },
    "raw_overrides": {},
}


if __name__ == "__main__":
    # Convenience mode for VS Code: `python ml_integration_phase1.py`
    # starts the API using uvicorn.
    import uvicorn

    uvicorn.run("ml_integration_phase1:app", host="127.0.0.1", port=8000, reload=True)
