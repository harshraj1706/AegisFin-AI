"""
AegisFin-AI — Phase 1 Credit Risk Assessment Web Application
Interactive Streamlit frontend integrated with Phase 1 ML Model & FastAPI Backend.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests
import streamlit as st

# Set page config with modern wide layout and custom title
st.set_page_config(
    page_title="AegisFin-AI | Credit Default Risk Assessment",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, premium financial dashboard styling
st.markdown(
    """
    <style>
    /* Global styles and clean typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 26px;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 14px;
        opacity: 0.85;
        margin: 0;
        font-weight: 400;
    }
    
    /* Risk Result Badges */
    .risk-badge {
        padding: 16px 20px;
        border-radius: 12px;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
        border: 1px solid transparent;
    }
    .risk-low {
        background-color: rgba(16, 185, 129, 0.12);
        color: #059669;
        border-color: rgba(16, 185, 129, 0.3);
    }
    .risk-moderate {
        background-color: rgba(245, 158, 11, 0.12);
        color: #d97706;
        border-color: rgba(245, 158, 11, 0.3);
    }
    .risk-elevated {
        background-color: rgba(249, 115, 22, 0.12);
        color: #ea580c;
        border-color: rgba(249, 115, 22, 0.3);
    }
    .risk-high {
        background-color: rgba(239, 68, 68, 0.12);
        color: #dc2626;
        border-color: rgba(239, 68, 68, 0.3);
    }
    
    /* Ratio Card Metric Containers */
    .metric-card {
        background: rgba(248, 250, 252, 0.8);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-label {
        font-size: 12px;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 20px;
        font-weight: 700;
        color: #0f172a;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: #f1f5f9;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 1. APPLICATION SETUP & PRESET CONFIGURATIONS
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.getenv(
        "AEGISFIN_MODEL_PATH",
        BASE_DIR / "models" / "aegisfin_phase1_final_model.pkl",
    )
)

PRESETS = {
    "🟢 Prime Borrower (Low Risk)": {
        "application": {
            "contract_type": "Cash loans",
            "credit_amount": 500000.0,
            "annuity_amount": 22000.0,
            "goods_price": 450000.0,
            "age_years": 42.0,
            "gender": "F",
            "children": 0,
            "family_members": 2.0,
            "family_status": "Married",
            "education_type": "Higher education",
            "income_type": "Working",
            "annual_income": 650000.0,
            "occupation_type": "Managers",
            "organization_type": "Business Entity Type 3",
            "employment_years": 12.0,
            "housing_type": "House / apartment",
            "owns_car": True,
            "owns_realty": True,
            "own_car_age_years": 3.0,
            "work_phone": True,
            "weekday_application": "TUESDAY",
        },
        "enrichment": {
            "ext_source_1": 0.65,
            "ext_source_2": 0.72,
            "ext_source_3": 0.78,
            "days_id_publish": -3200.0,
            "days_last_phone_change": -800.0,
            "days_registration": -6500.0,
            "region_rating_client_w_city": 1.0,
            "region_population_relative": 0.025,
            "reg_city_not_live_city": False,
            "credit_bureau_quarter": 0.0,
            "credit_bureau_year": 1.0,
            "def_30_cnt_social_circle": 0.0,
            "def_60_cnt_social_circle": 0.0,
            "flag_document_3": True,
            "flag_work_phone": True,
        },
        "raw_overrides": {},
    },
    "🟡 Moderate Risk Applicant": {
        "application": {
            "contract_type": "Cash loans",
            "credit_amount": 800000.0,
            "annuity_amount": 35000.0,
            "goods_price": 700000.0,
            "age_years": 32.0,
            "gender": "M",
            "children": 1,
            "family_members": 3.0,
            "family_status": "Married",
            "education_type": "Higher education",
            "income_type": "Working",
            "annual_income": 450000.0,
            "occupation_type": "Managers",
            "organization_type": "Business Entity Type 3",
            "employment_years": 6.0,
            "housing_type": "House / apartment",
            "owns_car": True,
            "owns_realty": True,
            "own_car_age_years": 4.0,
            "work_phone": True,
            "weekday_application": "MONDAY",
        },
        "enrichment": {
            "ext_source_1": 0.42,
            "ext_source_2": 0.62,
            "ext_source_3": 0.71,
            "days_id_publish": -2500.0,
            "days_last_phone_change": -300.0,
            "days_registration": -5000.0,
            "region_rating_client_w_city": 2.0,
            "region_population_relative": 0.0188,
            "reg_city_not_live_city": False,
            "credit_bureau_quarter": 1.0,
            "credit_bureau_year": 2.0,
            "def_30_cnt_social_circle": 0.0,
            "def_60_cnt_social_circle": 0.0,
            "flag_document_3": True,
            "flag_work_phone": True,
        },
        "raw_overrides": {},
    },
    "🔴 High Risk Profile": {
        "application": {
            "contract_type": "Cash loans",
            "credit_amount": 950000.0,
            "annuity_amount": 55000.0,
            "goods_price": 850000.0,
            "age_years": 23.0,
            "gender": "M",
            "children": 2,
            "family_members": 4.0,
            "family_status": "Single / not married",
            "education_type": "Secondary / secondary special",
            "income_type": "Working",
            "annual_income": 200000.0,
            "occupation_type": "Laborers",
            "organization_type": "Self-employed",
            "employment_years": 0.8,
            "housing_type": "Rented apartment",
            "owns_car": False,
            "owns_realty": False,
            "own_car_age_years": None,
            "work_phone": False,
            "weekday_application": "FRIDAY",
        },
        "enrichment": {
            "ext_source_1": 0.15,
            "ext_source_2": 0.22,
            "ext_source_3": 0.18,
            "days_id_publish": -800.0,
            "days_last_phone_change": -45.0,
            "days_registration": -1200.0,
            "region_rating_client_w_city": 3.0,
            "region_population_relative": 0.008,
            "reg_city_not_live_city": True,
            "credit_bureau_quarter": 4.0,
            "credit_bureau_year": 8.0,
            "def_30_cnt_social_circle": 2.0,
            "def_60_cnt_social_circle": 1.0,
            "flag_document_3": False,
            "flag_work_phone": False,
        },
        "raw_overrides": {},
    },
}

# -----------------------------------------------------------------------------
# 2. IN-PROCESS MODEL SERVICE CACHE
# -----------------------------------------------------------------------------
@st.cache_resource
def load_local_service():
    """Loads and caches the Phase1ModelService in memory."""
    try:
        from app.model_service import Phase1ModelService
        if not MODEL_PATH.exists():
            return None, f"Model file not found at {MODEL_PATH}"
        service = Phase1ModelService(MODEL_PATH)
        return service, None
    except Exception as exc:
        return None, str(exc)


# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS & BACKEND INTEGRATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine & Integration")
    
    execution_mode = st.radio(
        "Execution Mode",
        options=["Direct Engine (Local)", "FastAPI Backend (REST)"],
        help="Direct mode runs predictions directly in Python. REST mode sends requests to the running FastAPI server.",
    )
    
    api_url = "http://127.0.0.1:8000"
    if execution_mode == "FastAPI Backend (REST)":
        api_url = st.text_input("FastAPI Base URL", value="http://127.0.0.1:8000")
        
        # Test Connection button
        if st.button("📡 Check API Health", use_container_width=True):
            try:
                res = requests.get(f"{api_url}/health", timeout=3)
                if res.status_code == 200:
                    st.success(f"Connected! Status: {res.json().get('status')}")
                else:
                    st.warning(f"API Returned HTTP {res.status_code}")
            except Exception as e:
                st.error(f"Failed to reach API: {e}")
    
    st.markdown("---")
    st.markdown("### 📦 Quick Scenario Presets")
    selected_preset_name = st.selectbox(
        "Load Preset Profile",
        options=list(PRESETS.keys()),
        index=1,
    )
    
    if st.button("⚡ Apply Preset to Form", use_container_width=True):
        preset_data = PRESETS[selected_preset_name]
        st.session_state["current_payload"] = preset_data
        st.success(f"Loaded '{selected_preset_name}'!")
        st.rerun()

    st.markdown("---")
    # Model info card
    service, service_err = load_local_service()
    if service:
        info = service.info()
        st.markdown(
            f"""
            <div style="background:#e0e7ff; padding:12px; border-radius:10px; font-size:12px; color:#3730a3;">
                <strong>Active Model:</strong> {info['model_name']} ({info['model_version']})<br>
                <strong>Feature Space:</strong> {info['feature_count']} engineered features<br>
                <strong>Engine:</strong> XGBoost 3.2.0
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif service_err:
        st.warning(f"Local Model Notice: {service_err}")


# Initialize session state payload if absent
if "current_payload" not in st.session_state:
    st.session_state["current_payload"] = PRESETS["🟡 Moderate Risk Applicant"]

current_app = st.session_state["current_payload"].get("application", {})
current_enr = st.session_state["current_payload"].get("enrichment", {})

# -----------------------------------------------------------------------------
# 4. MAIN HEADER & HERO
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">
            <span>🛡️ AegisFin-AI</span>
            <span style="font-size:13px; background:rgba(255,255,255,0.2); padding:4px 10px; border-radius:20px; font-weight:500;">Phase 1 Production Release</span>
        </div>
        <p class="hero-subtitle">
            Next-generation Credit Default Probability & Financial Risk Intelligence Dashboard powered by XGBoost & SHAP Feature Engineering.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tab Navigation
tab_form, tab_json, tab_about = st.tabs(["📝 Loan Assessment Form", "💻 Raw JSON Payload / API Test", "ℹ️ Model Specs & Schema"])

# -----------------------------------------------------------------------------
# 5. TAB 1: INTERACTIVE FORM
# -----------------------------------------------------------------------------
with tab_form:
    # Helpful introductory guide explaining parameters
    with st.expander("💡 Parameter & Terminology Guide (Click to expand)", expanded=False):
        st.markdown(
            """
            <div style="font-size: 13px; line-height: 1.65; color: #334155; padding: 4px;">
                <p style="margin-top:0; font-weight:600; font-size:14px; color:#1e293b;">📘 Quick Reference for Input Parameters:</p>
                <ul style="margin-bottom: 8px;">
                    <li><strong>Contract Type:</strong> <em>Cash loans</em> (fixed lump-sum personal/business term loans) vs. <em>Revolving loans</em> (credit lines/cards where money can be repeatedly borrowed and repaid).</li>
                    <li><strong>Credit Amount Requested:</strong> The total loan principal amount requested from the lender.</li>
                    <li><strong>Loan Annuity / EMI:</strong> The monthly or periodic repayment installment amount required to service the loan.</li>
                    <li><strong>Goods / Asset Price:</strong> The purchase invoice price or fair market value of the item/asset being financed. For unsecured cash loans, this is typically equal to or slightly lower than the credit amount.</li>
                    <li><strong>Total Annual Income:</strong> Total gross annual earnings across salary, business, investments, and other declared sources.</li>
                    <li><strong>Employment Tenure:</strong> Number of consecutive years at current employer. Longer tenure signifies employment stability.</li>
                    <li><strong>External Scores (EXT_SOURCE 1 / 2 / 3):</strong> Normalized credit bureau rating scores (scale 0.0 to 1.0; higher score = lower default risk).</li>
                    <li><strong>Social Circle Overdue Defaults:</strong> Number of known contacts/associates with 30+ or 60+ days past-due payment defaults.</li>
                </ul>
                <p style="margin-bottom:0; font-style:italic; color:#64748b;">Tip: Hover over the <strong>(?)</strong> tooltip icon beside any input field for specific field guidance.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### 1. Loan & Demographics Profile")
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            contract_type = st.selectbox(
                "Contract Type",
                options=["Cash loans", "Revolving loans"],
                index=0 if current_app.get("contract_type") == "Cash loans" else 1,
                help="Type of credit agreement: 'Cash loans' (fixed term lump-sum loan) or 'Revolving loans' (open-ended credit line / credit card).",
            )
            credit_amount = st.number_input(
                "Credit Amount Requested ($ / ₹)",
                min_value=1000.0,
                max_value=10000000.0,
                value=float(current_app.get("credit_amount", 800000.0)),
                step=10000.0,
                help="Total loan principal amount requested by the borrower.",
            )
            annuity_amount = st.number_input(
                "Loan Annuity / EMI Amount ($ / ₹)",
                min_value=100.0,
                max_value=1000000.0,
                value=float(current_app.get("annuity_amount", 35000.0)),
                step=1000.0,
                help="Periodic repayment installment (EMI) required to service the requested loan.",
            )
            goods_price = st.number_input(
                "Goods / Asset Price ($ / ₹)",
                min_value=0.0,
                max_value=10000000.0,
                value=float(current_app.get("goods_price", 700000.0)),
                step=10000.0,
                help="The purchase price of the item/goods/property being financed. For personal cash loans, set this equal to or close to the requested credit amount.",
            )

        with col2:
            annual_income = st.number_input(
                "Total Annual Income ($ / ₹)",
                min_value=10000.0,
                max_value=50000000.0,
                value=float(current_app.get("annual_income", 450000.0)),
                step=25000.0,
                help="Total gross annual earnings of the applicant across employment, business, pensions, or other declared sources.",
            )
            age_years = st.slider(
                "Applicant Age (Years)",
                min_value=18,
                max_value=90,
                value=int(current_app.get("age_years", 32)),
                help="Applicant's current age in years. Used to evaluate credit lifecycle and remaining working years.",
            )
            gender = st.selectbox(
                "Gender",
                options=["M", "F", "XNA"],
                index=["M", "F", "XNA"].index(current_app.get("gender", "M")) if current_app.get("gender") in ["M", "F", "XNA"] else 0,
                help="Applicant's gender (M = Male, F = Female, XNA = Not Specified).",
            )
            education_options = [
                "Higher education",
                "Secondary / secondary special",
                "Incomplete higher",
                "Lower secondary",
                "Academic degree",
            ]
            cur_edu = current_app.get("education_type", "Higher education")
            education_type = st.selectbox(
                "Education Level",
                options=education_options,
                index=education_options.index(cur_edu) if cur_edu in education_options else 0,
                help="Highest level of formal education completed by the applicant.",
            )

        with col3:
            family_options = [
                "Married",
                "Single / not married",
                "Civil marriage",
                "Separated",
                "Widow",
            ]
            cur_fam = current_app.get("family_status", "Married")
            family_status = st.selectbox(
                "Family Status",
                options=family_options,
                index=family_options.index(cur_fam) if cur_fam in family_options else 0,
                help="Legal marital and domestic status of the applicant.",
            )
            children = st.number_input(
                "Number of Children",
                min_value=0,
                max_value=15,
                value=int(current_app.get("children", 1)),
                help="Count of dependent children supported by the applicant.",
            )
            family_members = st.number_input(
                "Total Family Members",
                min_value=1.0,
                max_value=20.0,
                value=float(current_app.get("family_members", 3.0)),
                help="Total number of people living in the applicant's household (used to compute per-capita disposable income).",
            )
            housing_options = [
                "House / apartment",
                "With parents",
                "Municipal apartment",
                "Rented apartment",
                "Office apartment",
                "Co-op apartment",
            ]
            cur_house = current_app.get("housing_type", "House / apartment")
            housing_type = st.selectbox(
                "Housing Type",
                options=housing_options,
                index=housing_options.index(cur_house) if cur_house in housing_options else 0,
                help="Primary residential ownership/living arrangement of the applicant.",
            )

    st.markdown("#### 2. Employment & Asset Portfolio")
    with st.container():
        ecol1, ecol2, ecol3 = st.columns(3)
        
        with ecol1:
            income_types = [
                "Working",
                "Commercial associate",
                "Pensioner",
                "State servant",
                "Unemployed",
                "Student",
                "Businessman",
                "Maternity leave",
            ]
            cur_inc_type = current_app.get("income_type", "Working")
            income_type = st.selectbox(
                "Income Type",
                options=income_types,
                index=income_types.index(cur_inc_type) if cur_inc_type in income_types else 0,
                help="Primary source/classification of employment income.",
            )
            
            occupation_types = [
                "Managers",
                "Core staff",
                "Laborers",
                "Drivers",
                "Sales staff",
                "High skill tech staff",
                "Accountants",
                "Medicine staff",
                "Security staff",
                "Cleaning staff",
                "Private service staff",
                "Low-skill Laborers",
                "Waiters/barmen staff",
                "Secretaries",
                "Realty agents",
                "IT staff",
                "HR staff",
                "Cooking staff",
            ]
            cur_occ = current_app.get("occupation_type", "Managers")
            occupation_type = st.selectbox(
                "Occupation Category",
                options=occupation_types,
                index=occupation_types.index(cur_occ) if cur_occ in occupation_types else 0,
                help="Specific job function or professional specialization.",
            )

        with ecol2:
            employment_years = st.slider(
                "Employment Tenure (Years)",
                min_value=0.0,
                max_value=50.0,
                value=float(current_app.get("employment_years", 6.0)),
                step=0.5,
                help="Total consecutive years employed at the current company/organization. Higher tenure reflects stability.",
            )
            organization_type = st.text_input(
                "Organization Type",
                value=str(current_app.get("organization_type", "Business Entity Type 3")),
                help="Industry category or corporate classification of the employer organization.",
            )

        with ecol3:
            owns_car = st.checkbox(
                "Owns Car",
                value=bool(current_app.get("owns_car", True)),
                help="Indicates whether the applicant owns one or more personal vehicles.",
            )
            own_car_age = None
            if owns_car:
                own_car_age = st.number_input(
                    "Car Age (Years)",
                    min_value=0.0,
                    max_value=60.0,
                    value=float(current_app.get("own_car_age_years", 4.0) or 4.0),
                    help="Age of applicant's primary vehicle in years.",
                )
            owns_realty = st.checkbox(
                "Owns Realty / Property",
                value=bool(current_app.get("owns_realty", True)),
                help="Indicates whether the applicant owns real estate (house, apartment, or land).",
            )
            work_phone = st.checkbox(
                "Work Phone Registered",
                value=bool(current_app.get("work_phone", True)),
                help="Indicates whether a verified workplace contact phone number was provided.",
            )

    with st.expander("🛡️ Advanced Bureau & Risk Enrichment Signals (Optional / Automated)", expanded=False):
        st.markdown(
            """
            <div style="font-size: 12px; color: #64748b; margin-bottom: 12px;">
                These signals are typically enriched automatically from credit bureaus (e.g. CIBIL, Experian, Equifax) and institutional registries.
            </div>
            """,
            unsafe_allow_html=True,
        )
        bcol1, bcol2, bcol3 = st.columns(3)
        
        with bcol1:
            ext_1 = st.slider(
                "External Score 1 (EXT_SOURCE_1)",
                0.0,
                1.0,
                float(current_enr.get("ext_source_1", 0.42) or 0.42),
                0.01,
                help="Normalized credit bureau rating score from External Agency 1 (scale 0.0 to 1.0; higher = safer borrower).",
            )
            ext_2 = st.slider(
                "External Score 2 (EXT_SOURCE_2)",
                0.0,
                1.0,
                float(current_enr.get("ext_source_2", 0.62) or 0.62),
                0.01,
                help="Normalized credit bureau rating score from External Agency 2 (scale 0.0 to 1.0; key predictive default signal).",
            )
            ext_3 = st.slider(
                "External Score 3 (EXT_SOURCE_3)",
                0.0,
                1.0,
                float(current_enr.get("ext_source_3", 0.71) or 0.71),
                0.01,
                help="Normalized credit bureau rating score from External Agency 3 (scale 0.0 to 1.0; independent agency metric).",
            )
            
        with bcol2:
            days_id = st.number_input(
                "Days Since ID Publish",
                value=float(current_enr.get("days_id_publish", -2500.0)),
                help="Days elapsed since applicant's national identity document was issued/renewed (negative value relative to application date).",
            )
            days_phone = st.number_input(
                "Days Since Last Phone Change",
                value=float(current_enr.get("days_last_phone_change", -300.0)),
                help="Days elapsed since applicant changed their primary contact phone number (higher negative numbers = longer stability).",
            )
            days_reg = st.number_input(
                "Days Since Registration",
                value=float(current_enr.get("days_registration", -5000.0)),
                help="Days elapsed since applicant registered their residential address.",
            )
            region_rating = st.selectbox(
                "Region Rating Client With City",
                options=[1, 2, 3],
                index=int(current_enr.get("region_rating_client_w_city", 2)) - 1,
                help="Regional banking credit risk tier of applicant's city/region (1 = Prime Metropolitan / Lowest Risk, 3 = High Risk / Rural).",
            )

        with bcol3:
            bureau_qrt = st.number_input(
                "Bureau Queries (Quarter)",
                min_value=0.0,
                value=float(current_enr.get("credit_bureau_quarter", 1.0)),
                help="Number of credit inquiries recorded by credit bureaus for this applicant in the last 3 months.",
            )
            bureau_yr = st.number_input(
                "Bureau Queries (Year)",
                min_value=0.0,
                value=float(current_enr.get("credit_bureau_year", 2.0)),
                help="Number of credit inquiries recorded by credit bureaus for this applicant in the last 12 months.",
            )
            def_30 = st.number_input(
                "Social Circle Def 30 Count",
                min_value=0.0,
                value=float(current_enr.get("def_30_cnt_social_circle", 0.0)),
                help="Count of applicant's social contacts/associates who had payment defaults past due by 30+ days.",
            )
            def_60 = st.number_input(
                "Social Circle Def 60 Count",
                min_value=0.0,
                value=float(current_enr.get("def_60_cnt_social_circle", 0.0)),
                help="Count of applicant's social contacts/associates who had payment defaults past due by 60+ days.",
            )
            flag_doc3 = st.checkbox(
                "Flag Document 3 Provided",
                value=bool(current_enr.get("flag_document_3", True)),
                help="KYC verification flag: whether primary national identity proof (Document 3) was submitted.",
            )

    # Assemble structured payload
    constructed_request = {
        "application": {
            "contract_type": contract_type,
            "credit_amount": credit_amount,
            "annuity_amount": annuity_amount,
            "goods_price": goods_price,
            "age_years": float(age_years),
            "gender": gender,
            "children": int(children),
            "family_members": float(family_members),
            "family_status": family_status,
            "education_type": education_type,
            "income_type": income_type,
            "annual_income": annual_income,
            "occupation_type": occupation_type,
            "employment_years": float(employment_years),
            "organization_type": organization_type,
            "housing_type": housing_type,
            "owns_car": owns_car,
            "owns_realty": owns_realty,
            "own_car_age_years": own_car_age,
            "work_phone": work_phone,
            "weekday_application": "MONDAY",
        },
        "enrichment": {
            "ext_source_1": ext_1,
            "ext_source_2": ext_2,
            "ext_source_3": ext_3,
            "days_id_publish": days_id,
            "days_last_phone_change": days_phone,
            "days_registration": days_reg,
            "region_rating_client_w_city": float(region_rating),
            "region_population_relative": 0.0188,
            "reg_city_not_live_city": False,
            "credit_bureau_quarter": bureau_qrt,
            "credit_bureau_year": bureau_yr,
            "def_30_cnt_social_circle": def_30,
            "def_60_cnt_social_circle": def_60,
            "flag_document_3": flag_doc3,
            "flag_work_phone": work_phone,
        },
        "raw_overrides": {},
    }

    st.markdown("---")
    predict_clicked = st.button("🚀 Calculate Default Risk & Analyze", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# 6. TAB 2: RAW JSON PAYLOAD EDITOR
# -----------------------------------------------------------------------------
with tab_json:
    st.markdown("#### Test or Inspect Raw API Request Payload")
    json_str_input = st.text_area(
        "Request JSON (Pydantic / FastAPI schema)",
        value=json.dumps(st.session_state.get("current_payload", PRESETS["🟡 Moderate Risk Applicant"]), indent=2),
        height=380,
        help="Paste a complete JSON payload matching the Pydantic PredictionRequest schema to test the model pipeline.",
    )
    json_predict_clicked = st.button("🚀 Execute Prediction on Custom JSON", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# 7. TAB 3: SPECIFICATIONS & ARCHITECTURE
# -----------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
        ### 📖 AegisFin-AI Architecture Overview
        
        - **Model Engine**: Gradient Boosted Decision Trees (`XGBoost 3.2.0`) with Scikit-Learn API integration.
        - **Feature Engineering Pipeline**: 
          - DTI Ratios (`CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `GOODS_CREDIT_RATIO`)
          - Per-capita Ratios (`INCOME_PER_FAMILY_MEMBER`, `INCOME_PER_CHILD`)
          - Composite Bureau aggregations (`EXT_SOURCE_MEAN`, `EXT_SOURCE_STD`, `EXT_SOURCE_MIN`, `EXT_SOURCE_MAX`)
          - Day count normalization and categorical missing imputation.
        - **Schema Design**: Two-tier ingestion separating clean customer self-reported attributes from institutional enrichment records.
        
        ---
        
        ### 📋 Input Parameter Definitions & Risk Impact
        
        | Parameter | Domain | Description & Impact on Underwriting |
        | :--- | :--- | :--- |
        | **Contract Type** | Demographics / Loan | Specifies whether the facility is a lump-sum term loan (*Cash loans*) or a reusable revolving line (*Revolving loans*). |
        | **Credit Amount** | Financial ($ / ₹) | Total principal amount requested. Higher credit amounts relative to income elevate the Debt-to-Income (DTI) ratio. |
        | **Loan Annuity** | Financial ($ / ₹) | Periodic repayment obligation (EMI). Used to compute Debt Service Coverage & Annuity Burden (% of monthly earnings). |
        | **Goods / Asset Price** | Financial ($ / ₹) | Value or purchase price of the underlying asset/goods financed. Determines Loan-to-Value (LTV) coverage. |
        | **Total Annual Income** | Financial ($ / ₹) | Declared annual gross earnings. Forms the denominator for debt service and per-capita household capacity. |
        | **Employment Tenure** | Employment | Years spent with current employer. Strong negative correlation with early default risk. |
        | **External Scores (1/2/3)** | Bureau Signal | Normalized credit bureau scores (0.0 to 1.0). Aggregated into statistical mean/dispersion features with high predictive power. |
        | **Social Circle Defaults** | Risk Network | Count of social circle contacts delinquent by 30+ or 60+ days, capturing clustered network risk. |
        """
    )


# -----------------------------------------------------------------------------
# 8. PREDICTION EXECUTION LOGIC
# -----------------------------------------------------------------------------
target_request_dict = None
if predict_clicked:
    target_request_dict = constructed_request
elif json_predict_clicked:
    try:
        target_request_dict = json.loads(json_str_input)
    except Exception as exc:
        st.error(f"Invalid JSON format: {exc}")

if target_request_dict:
    st.markdown("---")
    st.markdown("## 📊 Risk Intelligence & Assessment Results")
    
    with st.spinner("Processing features through AegisFin pipeline..."):
        prediction_result = None
        error_msg = None
        
        if execution_mode == "Direct Engine (Local)":
            try:
                from app.schemas import PredictionRequest
                from app.mapper import request_to_dataframe
                
                req_obj = PredictionRequest(**target_request_dict)
                df_raw = request_to_dataframe(req_obj)
                
                srv, srv_err = load_local_service()
                if srv is None:
                    error_msg = f"Failed to load model service: {srv_err}"
                else:
                    prob = srv.predict_probability(df_raw)
                    info = srv.info()
                    prediction_result = {
                        "default_probability": prob,
                        "model": info["model_name"],
                        "model_version": info["model_version"],
                        "feature_count": info["feature_count"],
                    }
            except Exception as exc:
                error_msg = f"Direct execution error: {exc}"
                
        else: # FastAPI REST Mode
            try:
                res = requests.post(f"{api_url}/api/v1/predict", json=target_request_dict, timeout=10)
                if res.status_code == 200:
                    prediction_result = res.json()
                else:
                    error_msg = f"API Error (HTTP {res.status_code}): {res.text}"
            except Exception as exc:
                error_msg = f"Could not connect to FastAPI server at {api_url}: {exc}"

    if error_msg:
        st.error(f"❌ {error_msg}")
    elif prediction_result:
        prob = float(prediction_result.get("default_probability", 0.0))
        pct_prob = prob * 100.0
        
        # Determine risk classification
        if pct_prob < 15.0:
            badge_class = "risk-low"
            risk_label = "LOW DEFAULT RISK — HIGH APPROVAL CONFIDENCE"
            risk_icon = "🟢"
            risk_desc = "Strong financial health, positive bureau indicators, and conservative debt obligations."
        elif pct_prob < 40.0:
            badge_class = "risk-moderate"
            risk_label = "MODERATE RISK — STANDARD UNDERWRITING"
            risk_icon = "🟡"
            risk_desc = "Balanced profile with acceptable debt service capacity. Standard terms recommended."
        elif pct_prob < 70.0:
            badge_class = "risk-elevated"
            risk_label = "ELEVATED RISK — CONDITIONAL APPROVAL"
            risk_icon = "🟠"
            risk_desc = "Elevated risk profile. Consider requesting additional collateral, co-signers, or lower loan amount."
        else:
            badge_class = "risk-high"
            risk_label = "HIGH RISK — CAUTION / POLICY DECLINE"
            risk_icon = "🔴"
            risk_desc = "Significant indicators of default likelihood based on historical credit trends."

        # Risk Banner
        st.markdown(
            f"""
            <div class="risk-badge {badge_class}">
                <div>
                    <div style="font-size: 18px;">{risk_icon} <strong>{risk_label}</strong></div>
                    <div style="font-size: 13px; font-weight: 400; margin-top: 4px;">{risk_desc}</div>
                </div>
                <div style="font-size: 32px; font-weight: 800; text-align: right;">
                    {pct_prob:.2f}%
                    <div style="font-size: 11px; font-weight: 500; text-transform: uppercase;">Default Probability</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Key Financial & Bureau Ratio Cards
        app_data = target_request_dict.get("application", {})
        enr_data = target_request_dict.get("enrichment", {})
        
        inc = float(app_data.get("annual_income") or 1.0)
        crd = float(app_data.get("credit_amount") or 0.0)
        ann = float(app_data.get("annuity_amount") or 0.0)
        gds = float(app_data.get("goods_price") or crd)
        fam = float(app_data.get("family_members") or 1.0)
        
        dti = crd / inc if inc > 0 else 0.0
        annuity_burden = (ann / inc) * 100.0 if inc > 0 else 0.0
        ltv = (crd / gds) * 100.0 if gds > 0 else 100.0
        inc_per_capita = inc / (fam + 1)
        
        ext_scores = [enr_data.get(k) for k in ["ext_source_1", "ext_source_2", "ext_source_3"] if enr_data.get(k) is not None]
        avg_ext = sum(ext_scores) / len(ext_scores) if ext_scores else 0.5

        rcol1, rcol2, rcol3, rcol4, rcol5 = st.columns(5)
        with rcol1:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Credit / Income (DTI)</div>
                    <div class="metric-val">{dti:.2f}x</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with rcol2:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Annuity Burden</div>
                    <div class="metric-val">{annuity_burden:.1f}%</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with rcol3:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Loan to Goods Value</div>
                    <div class="metric-val">{ltv:.1f}%</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with rcol4:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Income / Household</div>
                    <div class="metric-val">${inc_per_capita:,.0f}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with rcol5:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Avg Bureau Score</div>
                    <div class="metric-val">{avg_ext:.2f}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Risk Signal Comparison Chart
        c_left, c_right = st.columns([1, 1])
        with c_left:
            st.markdown("##### 📈 External Bureau Rating Signals")
            chart_data = pd.DataFrame({
                "Source": ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "Benchmark Low-Risk"],
                "Score (0-1)": [
                    float(enr_data.get("ext_source_1") or 0.0),
                    float(enr_data.get("ext_source_2") or 0.0),
                    float(enr_data.get("ext_source_3") or 0.0),
                    0.65,
                ]
            })
            st.bar_chart(chart_data, x="Source", y="Score (0-1)", color="#4f46e5")

        with c_right:
            st.markdown("##### 💼 Debt Service & Leverage Ratios")
            stress_data = pd.DataFrame({
                "Metric": ["Annuity / Income (%)", "Safe Debt Limit (%)"],
                "Value": [annuity_burden, 25.0]
            })
            st.bar_chart(stress_data, x="Metric", y="Value", color="#f59e0b")

        # Technical Payload & Audit Export
        with st.expander("🔍 Assessment Audit Log & Technical Payloads"):
            st.json({
                "request_payload": target_request_dict,
                "model_response": prediction_result,
                "calculated_metrics": {
                    "credit_income_ratio": round(dti, 4),
                    "annuity_income_percentage": round(annuity_burden, 2),
                    "goods_credit_ratio": round(ltv, 2),
                    "income_per_family_member": round(inc_per_capita, 2),
                    "ext_source_mean": round(avg_ext, 4),
                },
            })
            
            # Download report button
            report_data = {
                "assessment_id": "AEGIS-PHASE1-DEMO",
                "default_probability": prob,
                "risk_category": risk_label,
                "model_metadata": prediction_result,
                "input_data": target_request_dict,
            }
            st.download_button(
                label="📥 Download Assessment Report (JSON)",
                data=json.dumps(report_data, indent=2),
                file_name="credit_risk_assessment_report.json",
                mime="application/json",
            )
