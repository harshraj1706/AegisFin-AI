from __future__ import annotations
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class CustomerApplication(BaseModel):
    """Information a customer can reasonably provide."""

    model_config = ConfigDict(extra="forbid")

    # Highest-value loan/application fields
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

    # Process / contact
    work_phone: Optional[bool] = None
    weekday_application: Optional[str] = None


class EnrichmentData(BaseModel):
    """
    High-value model features that should normally come from
    trusted bank/internal/bureau systems rather than customer typing.
    """

    model_config = ConfigDict(extra="allow")

    # Strong SHAP features in the existing Phase 1 notebook
    ext_source_1: Optional[float] = Field(None, ge=0, le=1)
    ext_source_2: Optional[float] = Field(None, ge=0, le=1)
    ext_source_3: Optional[float] = Field(None, ge=0, le=1)

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

    # Document / contact indicators
    flag_document_3: Optional[bool] = None
    flag_work_phone: Optional[bool] = None

    def to_raw_dict(self) -> Dict[str, Any]:
        # These names are the original Home-Credit-style raw training columns.
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
    """
    Full production-facing request:
    customer-known application + trusted enrichment + optional raw overrides.
    """

    model_config = ConfigDict(extra="forbid")

    application: CustomerApplication
    enrichment: Optional[EnrichmentData] = None

    # Escape hatch for trusted backend fields that are not yet explicit
    # in EnrichmentData.
    raw_overrides: Dict[str, Any] = Field(default_factory=dict)


from typing import Literal

class PredictionResponse(BaseModel):
    default_probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated default probability")
    risk_band: Literal["LOW", "MODERATE", "ELEVATED", "HIGH"] = Field(..., description="Deterministic risk band")
    model_name: str = Field(..., description="Base model family")
    model_version: str = Field(..., description="Base model version")
    calibration_version: str = Field(..., description="Probability calibration version")
    policy_version: str = Field(..., description="Deterministic risk policy version")
    feature_count: int = Field(..., description="Number of engineered features in model")
    
    # Backwards compatibility and technical audit fields
    model: Optional[str] = Field(None, description="Alias for model_name")
    raw_probability: Optional[float] = Field(None, description="Uncalibrated raw XGBoost probability")
    calibration_method: Optional[str] = Field(None, description="Calibration algorithm")


class FraudPredictionRequest(BaseModel):
    """
    Production-facing Phase 2 Fraud Prediction Request.
    Accepts raw transaction parameters required by the feature engine.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="allow",
    )

    transaction_id: str = Field(..., min_length=1, description="Unique transaction identifier")
    amount: float = Field(..., ge=0.0, description="Transaction amount in USD", alias="transaction_amount")
    transaction_timestamp: Optional[str] = Field(
        default=None,
        alias="timestamp",
        description="ISO-8601 UTC timestamp of transaction. Defaults to current time if omitted.",
    )
    customer_id: str = Field(..., min_length=1, description="Customer or account identifier")
    card_id: str = Field(..., min_length=1, description="Card identifier (maps to card1)")
    device_id: Optional[str] = Field(None, description="Device fingerprint / identifier")
    merchant_id: Optional[str] = Field(None, description="Merchant identifier")
    email_domain: Optional[str] = Field(None, description="Purchaser email domain (maps to P_emaildomain)")
    address_id: Optional[str] = Field(None, description="Billing zip or address identifier (maps to addr1)")
    product_code: Optional[str] = Field("W", description="Transaction product code (W, H, C, S, R)")
    card_network: Optional[str] = Field(None, description="Card network (visa, mastercard, etc.)")
    card_type: Optional[str] = Field(None, description="Card type (debit, credit)")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    country: Optional[str] = Field(None, description="Country code (e.g. US, IN)")


class FraudPredictionResponse(BaseModel):
    """
    Phase 2 Fraud Prediction Response.
    Returns calibrated fraud probability, policy band, operational decision, and model metadata.
    Supports both Phase 2 canonical fields and legacy compatibility fields.
    """

    model_config = ConfigDict(populate_by_name=True)

    transaction_id: str = Field(..., description="Unique transaction identifier")
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated fraud probability (legacy alias)")
    fraud_band: Literal["LOW", "REVIEW", "HIGH"] = Field(..., description="Legacy projected risk band")
    decision: Literal["ALLOW", "MANUAL_REVIEW", "BLOCK"] = Field(..., description="Legacy projected policy decision")
    model_name: str = Field(..., description="Model family (XGBoost)")
    model_version: str = Field(..., description="Trained model version")
    calibration_version: str = Field(..., description="Calibration version")
    policy_version: str = Field(..., description="Risk policy version")
    prediction_latency_ms: float = Field(..., description="Inference latency in milliseconds")

    # Canonical Phase 2 Fields (Source of Truth)
    raw_fraud_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Canonical raw uncalibrated fraud probability")
    calibrated_fraud_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Canonical calibrated fraud probability")
    risk_band: Optional[Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]] = Field(None, description="Canonical 4-tier Phase 2 risk band")
    recommended_action: Optional[Literal["AUTO_APPROVE", "STEP_UP_AUTH", "MANUAL_REVIEW", "HARD_DECLINE"]] = Field(None, description="Canonical Phase 2 operational action")
    feature_contract_version: Optional[str] = Field("2.1.0", description="Phase 2 feature contract version")
    calibrator_type: Optional[str] = Field("sigmoid", description="Calibrator type")
    scored_at: Optional[str] = Field(None, description="ISO-8601 UTC timestamp of scoring event")

    # Optional technical / audit fields
    raw_probability: Optional[float] = Field(None, description="Uncalibrated raw model probability")
    calibration_method: Optional[str] = Field(None, description="Calibration algorithm (Platt)")
    feature_count: Optional[int] = Field(None, description="Number of model features evaluated")
