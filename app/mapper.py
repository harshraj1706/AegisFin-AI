from __future__ import annotations
from typing import Any, Dict
import pandas as pd
from .schemas import PredictionRequest

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
    app = request.application
    raw: Dict[str, Any] = {}

    for field_name, raw_name in CUSTOMER_TO_RAW.items():
        value = getattr(app, field_name)
        if value is not None:
            raw[raw_name] = value

    # Keep categorical flags in the same representation as training data.
    if app.owns_car is not None:
        raw["FLAG_OWN_CAR"] = "Y" if app.owns_car else "N"

    if app.owns_realty is not None:
        raw["FLAG_OWN_REALTY"] = "Y" if app.owns_realty else "N"

    # These are numeric binary columns in the original training schema.
    if app.work_phone is not None:
        raw["FLAG_WORK_PHONE"] = int(bool(app.work_phone))

    # The training data stores age/employment as negative day counts.
    if app.age_years is not None:
        raw["DAYS_BIRTH"] = -float(app.age_years) * 365.25

    if app.employment_years is not None:
        raw["DAYS_EMPLOYED"] = -float(app.employment_years) * 365.25

    if app.own_car_age_years is not None:
        raw["OWN_CAR_AGE"] = float(app.own_car_age_years)

    if request.enrichment is not None:
        raw.update(request.enrichment.to_raw_dict())

    raw.update(request.raw_overrides)
    return raw


def request_to_dataframe(request: PredictionRequest) -> pd.DataFrame:
    return pd.DataFrame([request_to_raw_dict(request)])
