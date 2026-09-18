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


class PredictionResponse(BaseModel):
    default_probability: float
    model: str
    model_version: str
    feature_count: int
