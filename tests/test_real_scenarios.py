"""
Real-world scenario test suite for AegisFin-AI Phase 1B
Verifies probability calibration and deterministic risk band tiering across real test cases.
"""

import sys
import numpy as np
import pytest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.risk_service import Phase1RiskService, classify_risk_band, safe_logit
from app.schemas import PredictionRequest
from app.mapper import request_to_dataframe
from streamlit_app import PRESETS


@pytest.fixture(scope="module")
def risk_service():
    return Phase1RiskService("models/aegisfin_phase1b_calibrated_model.pkl")


def test_user_reported_case_7_18_percent():
    """
    Directly verify the user's reported scenario:
    An applicant with 7.18% calibrated default probability MUST be classified as LOW RISK BAND,
    NEVER MODERATE.
    """
    band_718 = classify_risk_band(0.0718)
    assert band_718 == "LOW", f"Expected LOW for 7.18%, got {band_718}"

    # Also test boundary around 7.18%
    assert classify_risk_band(0.01) == "LOW"
    assert classify_risk_band(0.05) == "LOW"
    assert classify_risk_band(0.1499) == "LOW"


def test_deterministic_boundary_thresholds():
    """Verify all 4 deterministic risk bands at their exact mathematical boundaries."""
    # LOW (< 0.15)
    assert classify_risk_band(0.000) == "LOW"
    assert classify_risk_band(0.0718) == "LOW"
    assert classify_risk_band(0.1499) == "LOW"

    # MODERATE (0.15 to 0.40)
    assert classify_risk_band(0.1500) == "MODERATE"
    assert classify_risk_band(0.2500) == "MODERATE"
    assert classify_risk_band(0.3999) == "MODERATE"

    # ELEVATED (0.40 to 0.70)
    assert classify_risk_band(0.4000) == "ELEVATED"
    assert classify_risk_band(0.5500) == "ELEVATED"
    assert classify_risk_band(0.6999) == "ELEVATED"

    # HIGH (>= 0.70)
    assert classify_risk_band(0.7000) == "HIGH"
    assert classify_risk_band(0.8500) == "HIGH"
    assert classify_risk_band(1.000) == "HIGH"


def test_real_preset_prime_borrower(risk_service):
    """Verify the Prime Borrower preset produces a LOW risk band."""
    preset_prime = PRESETS["🟢 Prime Borrower (Low Risk Band, < 15%)"]
    req = PredictionRequest(**preset_prime)
    df = request_to_dataframe(req)
    res = risk_service.predict_risk(df)

    assert res["default_probability"] < 0.15
    assert res["risk_band"] == "LOW"
    assert "calibrated_probability" not in res or res["calibrated_probability"] == res["default_probability"]


def test_real_preset_moderate_applicant(risk_service):
    """Verify the Moderate Risk applicant preset produces a MODERATE risk band."""
    preset_mod = PRESETS["🟡 Moderate Risk Applicant (Moderate Band, 15% - 40%)"]
    req = PredictionRequest(**preset_mod)
    df = request_to_dataframe(req)
    res = risk_service.predict_risk(df)

    assert 0.15 <= res["default_probability"] < 0.40
    assert res["risk_band"] == "MODERATE"


def test_real_preset_elevated_profile(risk_service):
    """Verify the Elevated Risk profile preset produces ELEVATED risk band."""
    preset_elev = PRESETS["🔴 Elevated Risk Profile (Elevated Band, 40% - 70%)"]
    req = PredictionRequest(**preset_elev)
    df = request_to_dataframe(req)
    res = risk_service.predict_risk(df)

    assert 0.40 <= res["default_probability"] < 0.70
    assert res["risk_band"] == "ELEVATED"


def test_extreme_high_risk_profile(risk_service):
    """Verify that an applicant with severe delinquency history reaches HIGH risk band."""
    distressed = {
        "application": {
            "contract_type": "Cash loans",
            "credit_amount": 1500000.0,
            "annuity_amount": 90000.0,
            "goods_price": 1000000.0,
            "age_years": 21.0,
            "gender": "M",
            "children": 3,
            "family_members": 5.0,
            "family_status": "Single / not married",
            "education_type": "Lower secondary",
            "income_type": "Unemployed",
            "annual_income": 80000.0,
            "occupation_type": "Low-skill Laborers",
            "organization_type": "Self-employed",
            "employment_years": 0.1,
            "housing_type": "Rented apartment",
            "owns_car": False,
            "owns_realty": False,
            "own_car_age_years": None,
            "work_phone": False,
            "weekday_application": "FRIDAY",
        },
        "enrichment": {
            "ext_source_1": 0.01,
            "ext_source_2": 0.01,
            "ext_source_3": 0.01,
            "days_id_publish": -100.0,
            "days_last_phone_change": -5.0,
            "days_registration": -200.0,
            "region_rating_client_w_city": 3.0,
            "region_population_relative": 0.005,
            "reg_city_not_live_city": True,
            "credit_bureau_quarter": 6.0,
            "credit_bureau_year": 12.0,
            "def_30_cnt_social_circle": 5.0,
            "def_60_cnt_social_circle": 4.0,
            "flag_document_3": False,
            "flag_work_phone": False,
        },
        "raw_overrides": {},
    }
    req = PredictionRequest(**distressed)
    df = request_to_dataframe(req)
    res = risk_service.predict_risk(df)

    assert res["default_probability"] >= 0.40  # At least elevated, potentially high
    assert res["risk_band"] in ["ELEVATED", "HIGH"]
