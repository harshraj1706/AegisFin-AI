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
from supabase import create_client, Client

from app.config import SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, API_BASE_URL
from app.streamlit_fraud_view import render_fraud_detection_page

# Set page config with modern wide layout and custom title
st.set_page_config(
    page_title="AegisFin-AI | Risk & Fraud Intelligence",
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
        BASE_DIR / "models" / "aegisfin_phase1b_calibrated_model.pkl",
    )
)

PRESETS = {
    "🟢 Prime Borrower (Low Risk Band, < 15%)": {
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
    "🟡 Moderate Risk Applicant (Moderate Band, 15% - 40%)": {
        "application": {
            "contract_type": "Cash loans",
            "credit_amount": 850000.0,
            "annuity_amount": 38000.0,
            "goods_price": 750000.0,
            "age_years": 30.0,
            "gender": "M",
            "children": 1,
            "family_members": 3.0,
            "family_status": "Married",
            "education_type": "Secondary / secondary special",
            "income_type": "Working",
            "annual_income": 280000.0,
            "occupation_type": "Laborers",
            "organization_type": "Business Entity Type 3",
            "employment_years": 1.5,
            "housing_type": "House / apartment",
            "owns_car": False,
            "owns_realty": True,
            "own_car_age_years": None,
            "work_phone": True,
            "weekday_application": "MONDAY",
        },
        "enrichment": {
            "ext_source_1": 0.18,
            "ext_source_2": 0.22,
            "ext_source_3": 0.20,
            "days_id_publish": -1800.0,
            "days_last_phone_change": -120.0,
            "days_registration": -3500.0,
            "region_rating_client_w_city": 2.0,
            "region_population_relative": 0.0188,
            "reg_city_not_live_city": False,
            "credit_bureau_quarter": 2.0,
            "credit_bureau_year": 4.0,
            "def_30_cnt_social_circle": 1.0,
            "def_60_cnt_social_circle": 0.0,
            "flag_document_3": True,
            "flag_work_phone": True,
        },
        "raw_overrides": {},
    },
    "🔴 Elevated Risk Profile (Elevated Band, 40% - 70%)": {
        "application": {
            "contract_type": "Cash loans",
            "credit_amount": 950000.0,
            "annuity_amount": 55000.0,
            "goods_price": 850000.0,
            "age_years": 22.0,
            "gender": "M",
            "children": 2,
            "family_members": 4.0,
            "family_status": "Single / not married",
            "education_type": "Lower secondary",
            "income_type": "Working",
            "annual_income": 180000.0,
            "occupation_type": "Low-skill Laborers",
            "organization_type": "Self-employed",
            "employment_years": 0.3,
            "housing_type": "Rented apartment",
            "owns_car": False,
            "owns_realty": False,
            "own_car_age_years": None,
            "work_phone": False,
            "weekday_application": "FRIDAY",
        },
        "enrichment": {
            "ext_source_1": 0.08,
            "ext_source_2": 0.10,
            "ext_source_3": 0.07,
            "days_id_publish": -500.0,
            "days_last_phone_change": -30.0,
            "days_registration": -800.0,
            "region_rating_client_w_city": 3.0,
            "region_population_relative": 0.008,
            "reg_city_not_live_city": True,
            "credit_bureau_quarter": 4.0,
            "credit_bureau_year": 7.0,
            "def_30_cnt_social_circle": 3.0,
            "def_60_cnt_social_circle": 2.0,
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
    """Loads and caches the Phase1RiskService in memory."""
    try:
        from app.risk_service import Phase1RiskService
        if not MODEL_PATH.exists():
            return None, f"Phase 1B model file not found at {MODEL_PATH}"
        service = Phase1RiskService(MODEL_PATH)
        return service, None
    except Exception as exc:
        return None, str(exc)


# -----------------------------------------------------------------------------
# 2.5 SUPABASE AUTHENTICATION & SESSION GATEKEEPER
# -----------------------------------------------------------------------------
@st.cache_resource
def get_supabase_frontend_client() -> Optional[Client]:
    """Provides a cached Supabase client using the safe publishable key."""
    if not SUPABASE_URL or not SUPABASE_PUBLISHABLE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY)
    except Exception:
        return None


# Initialize session state credentials
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["access_token"] = None
    st.session_state["user"] = None

# Unauthenticated Gatekeeper: redirect and block access until logged in
if not st.session_state.get("authenticated", False):
    st.markdown(
        """
        <div style="text-align: center; margin-top: 25px; margin-bottom: 24px;">
            <div style="display:inline-flex; align-items:center; gap:10px; margin-bottom:8px;">
                <span style="font-size: 36px;">🛡️</span>
                <span style="font-size: 30px; font-weight: 800; color: #1e1b4b; letter-spacing: -0.5px;">AegisFin-AI</span>
            </div>
            <p style="color: #64748b; font-size: 14px; margin: 0; font-weight: 500;">
                Institutional Financial Intelligence & Credit Risk Assessment Platform
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_l, col_center, col_r = st.columns([1, 2, 1])
    with col_center:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%); color: white; padding: 22px 24px; border-radius: 14px 14px 0 0; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: #a5b4fc; font-weight: 600; margin-bottom: 4px;">Security Gatekeeper</div>
                <h3 style="margin: 0; font-size: 20px; font-weight: 700;">Analyst Portal Authentication</h3>
                <p style="margin: 6px 0 0 0; font-size: 12px; color: #c7d2fe; opacity: 0.9;">Supabase Auth • RBAC Protection • Session Persistence</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_login, tab_signup = st.tabs(["🔐 Sign In", "📝 Create Account"])

        with tab_login:
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            login_email = st.text_input("Analyst Email", key="auth_login_email", placeholder="analyst@example.com")
            login_pwd = st.text_input("Password", type="password", key="auth_login_pwd", placeholder="••••••••")

            if st.button("🚀 Sign In to AegisFin", type="primary", use_container_width=True, key="btn_signin"):
                if not login_email or not login_pwd:
                    st.error("Please enter both email and password.")
                else:
                    with st.spinner("Verifying credentials with Supabase..."):
                        success = False
                        err_msg = ""
                        client = get_supabase_frontend_client()

                        # 1. Direct Supabase client sign in with publishable key
                        if client:
                            try:
                                res = client.auth.sign_in_with_password({
                                    "email": login_email.strip(),
                                    "password": login_pwd,
                                })
                                if res and res.session:
                                    st.session_state["authenticated"] = True
                                    st.session_state["access_token"] = res.session.access_token
                                    user_meta = res.user.user_metadata or {}
                                    st.session_state["user"] = {
                                        "user_id": str(res.user.id),
                                        "email": res.user.email,
                                        "full_name": user_meta.get("full_name") or login_email.split("@")[0].capitalize(),
                                        "role": "analyst",
                                    }
                                    success = True
                            except Exception as exc:
                                err_msg = str(exc)

                        # 2. Fallback via backend FastAPI /api/v1/auth/login
                        if not success:
                            try:
                                api_res = requests.post(
                                    f"{API_BASE_URL.rstrip('/')}/api/v1/auth/login",
                                    json={"email": login_email.strip(), "password": login_pwd},
                                    timeout=6,
                                )
                                if api_res.status_code == 200:
                                    data = api_res.json()
                                    st.session_state["authenticated"] = True
                                    st.session_state["access_token"] = data["access_token"]
                                    st.session_state["user"] = {
                                        "user_id": data["user_id"],
                                        "email": data["email"],
                                        "full_name": data.get("full_name"),
                                        "role": data.get("role", "analyst"),
                                    }
                                    success = True
                                else:
                                    err_msg = api_res.json().get("detail", "Invalid email or password.")
                            except Exception as exc:
                                if not err_msg:
                                    err_msg = str(exc)

                        if success:
                            st.success("✅ Access Granted! Loading workspace...")
                            st.rerun()
                        else:
                            st.error(f"❌ Access Denied: {err_msg}")

        with tab_signup:
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            signup_name = st.text_input("Full Name", key="auth_signup_name", placeholder="Jane Doe")
            signup_email = st.text_input("Work Email", key="auth_signup_email", placeholder="jane@company.com")
            signup_pwd = st.text_input("Password", type="password", key="auth_signup_pwd", placeholder="Minimum 6 characters")

            if st.button("✨ Create Analyst Account", type="primary", use_container_width=True, key="btn_signup"):
                if not signup_email or not signup_pwd:
                    st.error("Please provide both email and password.")
                elif len(signup_pwd) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    with st.spinner("Provisioning user & public.profiles in Supabase..."):
                        try:
                            signup_resp = requests.post(
                                f"{API_BASE_URL.rstrip('/')}/api/v1/auth/signup",
                                json={
                                    "email": signup_email.strip(),
                                    "password": signup_pwd,
                                    "full_name": signup_name.strip() if signup_name else None,
                                },
                                timeout=10,
                            )
                            if signup_resp.status_code in (200, 201):
                                # Auto-login immediately upon creation
                                login_resp = requests.post(
                                    f"{API_BASE_URL.rstrip('/')}/api/v1/auth/login",
                                    json={"email": signup_email.strip(), "password": signup_pwd},
                                    timeout=6,
                                )
                                if login_resp.status_code == 200:
                                    login_data = login_resp.json()
                                    st.session_state["authenticated"] = True
                                    st.session_state["access_token"] = login_data["access_token"]
                                    st.session_state["user"] = {
                                        "user_id": login_data["user_id"],
                                        "email": login_data["email"],
                                        "full_name": login_data.get("full_name") or signup_name or "Analyst",
                                        "role": login_data.get("role", "analyst"),
                                    }
                                    st.success("🎉 Account created and profile provisioned! Entering workspace...")
                                    st.rerun()
                                else:
                                    st.success("✅ Account created successfully! Please sign in using the Sign In tab.")
                            else:
                                err_detail = signup_resp.json().get("detail", signup_resp.text)
                                st.error(f"❌ Registration Failed: {err_detail}")
                        except Exception as exc:
                            st.error(f"❌ Could not connect to backend for signup: {exc}")

    with st.sidebar:
        st.markdown("### 🔒 AegisFin Access Portal")
        st.info("Please sign in or create an account to unlock the credit risk modeling workspace.")

    st.stop()


# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS & BACKEND INTEGRATION
# -----------------------------------------------------------------------------
with st.sidebar:
    # Authenticated User Badge & Logout
    user_info = st.session_state.get("user") or {}
    user_name = user_info.get("full_name") or "Risk Analyst"
    user_email = user_info.get("email") or ""
    user_role = user_info.get("role", "analyst").capitalize()

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%); color: white; padding: 14px 16px; border-radius: 12px; margin-bottom: 15px; border: 1px solid rgba(255,255,255,0.1);">
            <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #a5b4fc; font-weight: 600;">Authenticated Analyst</div>
            <div style="font-size: 15px; font-weight: 700; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">👤 {user_name}</div>
            <div style="font-size: 11px; color: #cbd5e1; margin-top: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{user_email}</div>
            <div style="margin-top: 8px; display: inline-block; background: rgba(99, 102, 241, 0.25); border: 1px solid rgba(165, 180, 252, 0.3); padding: 2px 8px; border-radius: 6px; font-size: 10px; font-weight: 600; color: #c7d2fe;">
                🛡️ {user_role}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("🚪 Sign Out / Lock Session", use_container_width=True, key="sidebar_signout_btn"):
        client = get_supabase_frontend_client()
        if client:
            try:
                client.auth.sign_out()
            except Exception:
                pass
        st.session_state["authenticated"] = False
        st.session_state["access_token"] = None
        st.session_state["user"] = None
        st.rerun()

    st.markdown("---")
    st.markdown("### 🧭 Platform Navigation")
    platform_module = st.radio(
        "Select Module",
        options=["Credit Risk Assessment", "Fraud Detection"],
        index=0 if st.session_state.get("platform_module") != "Fraud Detection" else 1,
        key="platform_module_nav",
        help="Switch between Phase 1 Credit Risk Analysis and Phase 2 Real-Time Fraud Detection.",
    )
    st.session_state["platform_module"] = platform_module

    if platform_module == "Fraud Detection":
        st.markdown("---")
        st.markdown("### ⚙️ Fraud Engine Configuration")
        fraud_api_url = st.text_input("FastAPI Base URL", value=API_BASE_URL, key="fraud_api_url_input")

        if st.button("📡 Check Fraud API Health", use_container_width=True, key="btn_fraud_api_health"):
            try:
                res = requests.get(f"{fraud_api_url.rstrip('/')}/api/v1/fraud/health", timeout=3)
                if res.status_code == 200:
                    h = res.json()
                    st.success(f"Connected! Model: {h.get('model_name')} ({h.get('model_version')}) | Status: {h.get('status')}")
                else:
                    st.warning(f"Fraud API returned HTTP {res.status_code}")
            except Exception as exc:
                st.error(f"Cannot reach Fraud API: {exc}")

        st.markdown("---")
        st.markdown(
            """
            <div style="background:#e0f2fe; padding:12px; border-radius:10px; font-size:12px; color:#0369a1; line-height: 1.5;">
                <strong>Active Model:</strong> XGBoost (phase2-xgb-v1)<br>
                <strong>Calibration:</strong> Platt Scaling (phase2-platt-v1)<br>
                <strong>Risk Policy:</strong> phase2-policy-v1<br>
                <strong>Feature Space:</strong> 17 Ingested & Historical Features<br>
                <strong>Database:</strong> Supabase PostgreSQL
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown("---")
        st.markdown("### ⚙️ Engine & Integration")
        
        execution_mode = st.radio(
            "Execution Mode",
            options=["FastAPI Backend (REST)", "Direct Engine (Local)"],
            index=0,
            help="REST mode communicates with the FastAPI service (source of truth). Direct mode runs the same canonical risk service locally.",
        )
        
        api_url = API_BASE_URL
        if execution_mode == "FastAPI Backend (REST)":
            api_url = st.text_input("FastAPI Base URL", value=API_BASE_URL)
            
            # Test Connection button
            if st.button("📡 Check API Health", use_container_width=True):
                try:
                    res = requests.get(f"{api_url}/health", timeout=3)
                    if res.status_code == 200:
                        health_data = res.json()
                        st.success(f"Connected! Model: {health_data.get('model_version')} | Cal: {health_data.get('calibration_version')}")
                    else:
                        st.warning(f"API Returned HTTP {res.status_code}")
                except Exception as e:
                    st.error(f"Failed to reach API: {e}")

            # Test Protected Auth Endpoint button
            if st.button("🔑 Verify Session (/api/v1/auth/me)", use_container_width=True):
                try:
                    token = st.session_state.get("access_token", "")
                    res = requests.get(
                        f"{api_url}/api/v1/auth/me",
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=4,
                    )
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"Verified! UID: {data.get('user_id')[:8]}... | Email: {data.get('email')}")
                    else:
                        st.warning(f"Auth check returned HTTP {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"Failed to reach API: {e}")
        
        st.markdown("---")
        st.markdown("### 📦 Quick Scenario Presets")
        
        preset_names = list(PRESETS.keys())
        
        def on_preset_select():
            sel = st.session_state.get("preset_dropdown_selector")
            if sel and sel in PRESETS:
                st.session_state["current_payload"] = PRESETS[sel]
                st.session_state["preset_nonce"] = st.session_state.get("preset_nonce", 0) + 1

        selected_preset_name = st.selectbox(
            "Load Preset Profile",
            options=preset_names,
            index=0,
            key="preset_dropdown_selector",
            on_change=on_preset_select,
            help="Select a benchmark profile to instantly load its financial parameters into the assessment form.",
        )
        
        if st.button("⚡ Apply Preset to Form", use_container_width=True):
            cur_sel = st.session_state.get("preset_dropdown_selector", selected_preset_name)
            preset_data = PRESETS[cur_sel]
            st.session_state["current_payload"] = preset_data
            st.session_state["preset_nonce"] = st.session_state.get("preset_nonce", 0) + 1
            st.success(f"Loaded '{cur_sel}'!")
            st.rerun()

        st.markdown("---")
        # Model info card
        service, service_err = load_local_service()
        if service:
            info = service.info()
            st.markdown(
                f"""
                <div style="background:#e0e7ff; padding:12px; border-radius:10px; font-size:12px; color:#3730a3; line-height: 1.5;">
                    <strong>Active Model:</strong> {info['model_name']} ({info['model_version']})<br>
                    <strong>Calibration:</strong> {info.get('calibration_version', 'N/A')}<br>
                    <strong>Risk Policy:</strong> {info.get('policy_version', 'N/A')}<br>
                    <strong>Feature Space:</strong> {info['feature_count']} engineered features<br>
                    <strong>Engine:</strong> Frozen XGBoost + Platt Scaling
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif service_err:
            st.warning(f"Local Model Notice: {service_err}")


# Routing: Render Fraud Detection Page if selected
if st.session_state.get("platform_module") == "Fraud Detection":
    fraud_target_url = st.session_state.get("fraud_api_url_input", API_BASE_URL)
    render_fraud_detection_page(api_base_url=fraud_target_url)
    st.stop()


# Initialize session state payload if absent
if "current_payload" not in st.session_state:
    first_preset = list(PRESETS.keys())[0]
    st.session_state["current_payload"] = PRESETS[first_preset]
    st.session_state["preset_nonce"] = 0

current_app = st.session_state["current_payload"].get("application", {})
current_enr = st.session_state["current_payload"].get("enrichment", {})
nonce = st.session_state.get("preset_nonce", 0)

# -----------------------------------------------------------------------------
# 4. MAIN HEADER & HERO
# -----------------------------------------------------------------------------
active_user = st.session_state.get("user") or {}
active_name = active_user.get("full_name") or "Risk Analyst"
active_email = active_user.get("email") or ""

st.markdown(
    f"""
    <div class="hero-card">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
            <div class="hero-title">
                <span>🛡️ AegisFin-AI</span>
                <span style="font-size:13px; background:rgba(255,255,255,0.2); padding:4px 10px; border-radius:20px; font-weight:500;">Phase 1B Calibrated Release</span>
            </div>
            <div style="background:rgba(16, 185, 129, 0.25); border:1px solid rgba(52, 211, 153, 0.5); padding:5px 14px; border-radius:20px; font-size:12px; color:#d1fae5; font-weight:600; display:flex; align-items:center; gap:6px;">
                <span>🟢 Logged in:</span> <span>{active_name}</span> <span style="opacity:0.8; font-size:11px;">({active_email})</span>
            </div>
        </div>
        <p class="hero-subtitle">
            Next-generation Credit Default Probability & Financial Risk Intelligence Dashboard powered by Calibrated XGBoost & Advanced Feature Engineering.
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
    with st.expander("💡 Parameter & Terminology Guide (Plain-English Explanations)", expanded=True):
        st.markdown(
            """
            <div style="font-size: 13px; line-height: 1.65; color: #334155; padding: 4px;">
                <p style="margin-top:0; font-weight:600; font-size:14px; color:#1e293b;">📘 Plain-English Guide to Input Parameters & Financial Ratios:</p>
                <ul style="margin-bottom: 8px;">
                    <li><strong>Contract Type:</strong> <em>Cash loans</em> (fixed lump-sum term loans disbursed once and repaid via EMI) vs. <em>Revolving loans</em> (flexible credit lines or credit cards where funds can be repeatedly borrowed and repaid).</li>
                    <li><strong>Credit Amount Requested (₹):</strong> The total loan principal amount requested by the applicant.</li>
                    <li><strong>Loan Annuity / EMI Amount (₹):</strong> The scheduled repayment installment (EMI) needed to service the loan. Used to evaluate your repayment burden.</li>
                    <li><strong>Goods / Asset Price (₹):</strong> The invoice or purchase price of the item/property being financed. For unsecured personal cash loans, set this equal to or close to the Credit Amount Requested.</li>
                    <li><strong>Total Annual Income (₹):</strong> Total gross annual income across salary, business, investments, and other verifiable sources.</li>
                    <li><strong>Employment Tenure:</strong> Continuous years with current employer. Longer tenure signifies higher job stability.</li>
                    <li><strong>External Scores (EXT_SOURCE 1 / 2 / 3):</strong> Independent credit bureau rating scores (normalized from 0.0 to 1.0, similar to CIBIL or Experian scores). Higher scores indicate stronger repayment history and lower risk.</li>
                    <li><strong>Social Circle Overdue Defaults:</strong> Count of acquaintances or social circle contacts who have defaulted on loans by 30+ or 60+ days.</li>
                </ul>
                <p style="margin-bottom:0; font-style:italic; color:#64748b;">Tip: All financial amounts are in Indian Rupees (₹). Hover over any <strong>(?)</strong> icon for field-specific guidance.</p>
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
                key=f"contract_type_{nonce}",
                help="Type of credit agreement: 'Cash loans' (fixed term lump-sum loan) or 'Revolving loans' (open-ended credit line / credit card).",
            )
            credit_amount = st.number_input(
                "Credit Amount Requested (₹)",
                min_value=1000.0,
                max_value=10000000.0,
                value=float(current_app.get("credit_amount", 500000.0)),
                step=10000.0,
                key=f"credit_amount_{nonce}",
                help="Total loan principal amount requested by the borrower in Rupees (₹).",
            )
            st.caption("ℹ️ Total principal amount borrowed from the lender.")
            annuity_amount = st.number_input(
                "Loan Annuity / EMI Amount (₹)",
                min_value=100.0,
                max_value=1000000.0,
                value=float(current_app.get("annuity_amount", 22000.0)),
                step=1000.0,
                key=f"annuity_amount_{nonce}",
                help="Periodic repayment installment (EMI) required to service the requested loan in Rupees (₹).",
            )
            st.caption("ℹ️ Repayment installment amount (EMI) to service the loan.")
            goods_price = st.number_input(
                "Goods / Asset Price (₹)",
                min_value=0.0,
                max_value=10000000.0,
                value=float(current_app.get("goods_price", 450000.0)),
                step=10000.0,
                key=f"goods_price_{nonce}",
                help="The purchase price of the item/goods/property being financed in Rupees (₹). For personal cash loans, set this equal or close to the requested credit amount.",
            )
            st.caption("ℹ️ Value of item/asset financed (for cash loans, match credit amount).")

        with col2:
            annual_income = st.number_input(
                "Total Annual Income (₹)",
                min_value=10000.0,
                max_value=50000000.0,
                value=float(current_app.get("annual_income", 650000.0)),
                step=25000.0,
                key=f"annual_income_{nonce}",
                help="Total gross annual earnings of the applicant in Rupees (₹) across employment, business, pensions, or other sources.",
            )
            st.caption("ℹ️ Total gross yearly income across all declared sources.")
            age_years = st.slider(
                "Applicant Age (Years)",
                min_value=18,
                max_value=90,
                value=int(current_app.get("age_years", 42)),
                key=f"age_years_{nonce}",
                help="Applicant's current age in years. Used to evaluate credit lifecycle and remaining working years.",
            )
            gender = st.selectbox(
                "Gender",
                options=["M", "F", "XNA"],
                index=["M", "F", "XNA"].index(current_app.get("gender", "F")) if current_app.get("gender") in ["M", "F", "XNA"] else 0,
                key=f"gender_{nonce}",
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
                key=f"education_type_{nonce}",
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
                key=f"family_status_{nonce}",
                help="Legal marital and domestic status of the applicant.",
            )
            children = st.number_input(
                "Number of Children",
                min_value=0,
                max_value=15,
                value=int(current_app.get("children", 0)),
                key=f"children_{nonce}",
                help="Count of dependent children supported by the applicant.",
            )
            family_members = st.number_input(
                "Total Family Members",
                min_value=1.0,
                max_value=20.0,
                value=float(current_app.get("family_members", 2.0)),
                key=f"family_members_{nonce}",
                help="Total number of people living in the applicant's household (used to compute per-capita disposable income).",
            )
            st.caption("ℹ️ Household size used to estimate per-capita income.")
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
                key=f"housing_type_{nonce}",
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
                key=f"income_type_{nonce}",
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
                key=f"occupation_type_{nonce}",
                help="Specific job function or professional specialization.",
            )

        with ecol2:
            employment_years = st.slider(
                "Employment Tenure (Years)",
                min_value=0.0,
                max_value=50.0,
                value=float(current_app.get("employment_years", 12.0)),
                step=0.5,
                key=f"employment_years_{nonce}",
                help="Total consecutive years employed at the current company/organization. Higher tenure reflects stability.",
            )
            st.caption("ℹ️ Consecutive years at current employer (reflects job stability).")
            organization_type = st.text_input(
                "Organization Type",
                value=str(current_app.get("organization_type", "Business Entity Type 3")),
                key=f"organization_type_{nonce}",
                help="Industry category or corporate classification of the employer organization.",
            )

        with ecol3:
            owns_car = st.checkbox(
                "Owns Car",
                value=bool(current_app.get("owns_car", True)),
                key=f"owns_car_{nonce}",
                help="Indicates whether the applicant owns one or more personal vehicles.",
            )
            own_car_age = None
            if owns_car:
                own_car_age = st.number_input(
                    "Car Age (Years)",
                    min_value=0.0,
                    max_value=60.0,
                    value=float(current_app.get("own_car_age_years", 3.0) or 3.0),
                    key=f"own_car_age_{nonce}",
                    help="Age of applicant's primary vehicle in years.",
                )
            owns_realty = st.checkbox(
                "Owns Realty / Property",
                value=bool(current_app.get("owns_realty", True)),
                key=f"owns_realty_{nonce}",
                help="Indicates whether the applicant owns real estate (house, apartment, or land).",
            )
            work_phone = st.checkbox(
                "Work Phone Registered",
                value=bool(current_app.get("work_phone", True)),
                key=f"work_phone_{nonce}",
                help="Indicates whether a verified workplace contact phone number was provided.",
            )

    with st.expander("🛡️ Advanced Bureau & Risk Enrichment Signals (Optional / Automated)", expanded=False):
        st.markdown(
            """
            <div style="font-size: 12px; color: #64748b; margin-bottom: 12px;">
                These signals are typically enriched automatically from credit bureaus (e.g. CIBIL, Experian, Equifax) and institutional registries.
                <strong>Scale:</strong> External Scores range from 0.00 (highest risk) to 1.00 (lowest risk / prime credit).
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
                float(current_enr.get("ext_source_1", 0.65) or 0.65),
                0.01,
                key=f"ext_1_{nonce}",
                help="Normalized credit bureau rating score from External Agency 1 (scale 0.0 to 1.0; higher = safer borrower).",
            )
            ext_2 = st.slider(
                "External Score 2 (EXT_SOURCE_2)",
                0.0,
                1.0,
                float(current_enr.get("ext_source_2", 0.72) or 0.72),
                0.01,
                key=f"ext_2_{nonce}",
                help="Normalized credit bureau rating score from External Agency 2 (scale 0.0 to 1.0; key predictive default signal).",
            )
            ext_3 = st.slider(
                "External Score 3 (EXT_SOURCE_3)",
                0.0,
                1.0,
                float(current_enr.get("ext_source_3", 0.78) or 0.78),
                0.01,
                key=f"ext_3_{nonce}",
                help="Normalized credit bureau rating score from External Agency 3 (scale 0.0 to 1.0; independent agency metric).",
            )
            st.caption("ℹ️ Bureau scores (0 to 1): Higher score indicates stronger credit history.")
            
        with bcol2:
            days_id = st.number_input(
                "Days Since ID Publish",
                value=float(current_enr.get("days_id_publish", -3200.0)),
                key=f"days_id_{nonce}",
                help="Days elapsed since applicant's national identity document was issued/renewed (negative value relative to application date).",
            )
            days_phone = st.number_input(
                "Days Since Last Phone Change",
                value=float(current_enr.get("days_last_phone_change", -800.0)),
                key=f"days_phone_{nonce}",
                help="Days elapsed since applicant changed their primary contact phone number (higher negative numbers = longer stability).",
            )
            days_reg = st.number_input(
                "Days Since Registration",
                value=float(current_enr.get("days_registration", -6500.0)),
                key=f"days_reg_{nonce}",
                help="Days elapsed since applicant registered their residential address.",
            )
            region_rating = st.selectbox(
                "Region Rating Client With City",
                options=[1, 2, 3],
                index=int(current_enr.get("region_rating_client_w_city", 1)) - 1,
                key=f"region_rating_{nonce}",
                help="Regional banking credit risk tier of applicant's city/region (1 = Prime Metropolitan / Lowest Risk, 3 = High Risk / Rural).",
            )

        with bcol3:
            bureau_qrt = st.number_input(
                "Bureau Queries (Quarter)",
                min_value=0.0,
                value=float(current_enr.get("credit_bureau_quarter", 0.0)),
                key=f"bureau_qrt_{nonce}",
                help="Number of credit inquiries recorded by credit bureaus for this applicant in the last 3 months.",
            )
            bureau_yr = st.number_input(
                "Bureau Queries (Year)",
                min_value=0.0,
                value=float(current_enr.get("credit_bureau_year", 1.0)),
                key=f"bureau_yr_{nonce}",
                help="Number of credit inquiries recorded by credit bureaus for this applicant in the last 12 months.",
            )
            def_30 = st.number_input(
                "Social Circle Def 30 Count",
                min_value=0.0,
                value=float(current_enr.get("def_30_cnt_social_circle", 0.0)),
                key=f"def_30_{nonce}",
                help="Count of applicant's social contacts/associates who had payment defaults past due by 30+ days.",
            )
            def_60 = st.number_input(
                "Social Circle Def 60 Count",
                min_value=0.0,
                value=float(current_enr.get("def_60_cnt_social_circle", 0.0)),
                key=f"def_60_{nonce}",
                help="Count of applicant's social contacts/associates who had payment defaults past due by 60+ days.",
            )
            st.caption("ℹ️ Social contacts with 30+/60+ days loan default history.")
            flag_doc3 = st.checkbox(
                "Flag Document 3 Provided",
                value=bool(current_enr.get("flag_document_3", True)),
                key=f"flag_doc3_{nonce}",
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
        value=json.dumps(st.session_state.get("current_payload", list(PRESETS.values())[0]), indent=2),
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
        ### 📖 AegisFin-AI Architecture Overview (Phase 1 Complete)
        
        - **Core Model**: Frozen XGBoost Classifier (`models/aegisfin_phase1_final_model.pkl`), selected via rigorous Phase 1A cross-validation and Optuna hyperparameter optimization.
        - **Probability Calibration**: Phase 1B Platt Scaling (`models/aegisfin_phase1b_calibrated_model.pkl`) mapping raw model margins into statistically grounded, reliable default probabilities.
        - **Risk Policy Tiering**: Deterministic, rule-based classification using explicit technical thresholds:
          - 🟢 **LOW RISK**: Calibrated default probability < 15%
          - 🟡 **MODERATE RISK**: Calibrated default probability 15% to 40%
          - 🟠 **ELEVATED RISK**: Calibrated default probability 40% to 70%
          - 🔴 **HIGH RISK**: Calibrated default probability ≥ 70%
          *(Note: These thresholds are provisional technical demo boundaries, not official banking credit approval/decline rules).*
        - **Domain Feature Engineering**: 183 input dimensions derived from Home Credit application records, including:
          - Leverage & Debt Ratios (`PAYMENT_RATE`, `CREDIT_TO_GOODS_RATIO`, `DEBT_TO_INCOME_RATIO`, `ANNUITY_INCOME_PERCENTAGE`)
          - Per-capita Ratios (`INCOME_PER_FAMILY_MEMBER`, `INCOME_PER_CHILD`)
          - Composite Bureau aggregations (`EXT_SOURCE_MEAN`, `EXT_SOURCE_STD`, `EXT_SOURCE_MIN`, `EXT_SOURCE_MAX`)
          - Day count normalization and categorical missing imputation.
        - **Schema Design**: Two-tier ingestion separating clean customer self-reported attributes from institutional enrichment records.
        - **Future Phases**: Model Explainability (SHAP) is planned for future Phase 3.
        
        ---
        
        ### 📋 Input Parameter Definitions & Risk Impact
        
        | Parameter | Domain | Description & Impact on Underwriting |
        | :--- | :--- | :--- |
        | **Contract Type** | Demographics / Loan | Specifies whether the facility is a lump-sum term loan (*Cash loans*) or a reusable revolving line (*Revolving loans*). |
        | **Credit Amount** | Financial (₹) | Total principal amount requested. Higher credit amounts relative to income elevate the Debt-to-Income (DTI) ratio. |
        | **Loan Annuity** | Financial (₹) | Periodic repayment obligation (EMI). Used to compute Debt Service Coverage & Annuity Burden (% of monthly earnings). |
        | **Goods / Asset Price** | Financial (₹) | Value or purchase price of the underlying asset/goods financed. Determines Loan-to-Value (LTV) coverage. |
        | **Total Annual Income** | Financial (₹) | Declared annual gross earnings in Indian Rupees (₹). Forms the denominator for debt service and per-capita household capacity. |
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
                    error_msg = f"Failed to load risk service: {srv_err}"
                else:
                    prediction_result = srv.predict_risk(df_raw)
            except Exception as exc:
                error_msg = f"Direct execution error: {exc}"
                
        else: # FastAPI REST Mode
            try:
                headers = {}
                token = st.session_state.get("access_token")
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                res = requests.post(
                    f"{api_url}/api/v1/predict",
                    json=target_request_dict,
                    headers=headers,
                    timeout=10,
                )
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
        
        # Enforce canonical deterministic Phase 1B policy
        # (< 0.15 = LOW, 0.15 - 0.40 = MODERATE, 0.40 - 0.70 = ELEVATED, >= 0.70 = HIGH)
        from app.risk_service import classify_risk_band
        risk_band = classify_risk_band(prob)
        
        model_version = prediction_result.get("model_version", "credit-xgb-v1.0.0")
        calibration_version = prediction_result.get("calibration_version", "credit-calibration-v1.0.0")
        policy_version = prediction_result.get("policy_version", "credit-risk-policy-v1.0.0")
        feature_count = prediction_result.get("feature_count", 183)
        
        # Display styling configuration for backend deterministic risk band (provisional technical thresholds)
        BAND_CONFIG = {
            "LOW": {
                "badge_class": "risk-low",
                "label": "LOW RISK BAND",
                "icon": "🟢",
                "desc": "Provisional Technical Demo Band: Calibrated default probability < 15%. Minimal risk of loan delinquency.",
            },
            "MODERATE": {
                "badge_class": "risk-moderate",
                "label": "MODERATE RISK BAND",
                "icon": "🟡",
                "desc": "Provisional Technical Demo Band: Calibrated default probability 15% - 40%. Standard applicant risk profile.",
            },
            "ELEVATED": {
                "badge_class": "risk-elevated",
                "label": "ELEVATED RISK BAND",
                "icon": "🟠",
                "desc": "Provisional Technical Demo Band: Calibrated default probability 40% - 70%. Heightened leverage or lower bureau signals.",
            },
            "HIGH": {
                "badge_class": "risk-high",
                "label": "HIGH RISK BAND",
                "icon": "🔴",
                "desc": "Provisional Technical Demo Band: Calibrated default probability ≥ 70%. Pronounced delinquency markers detected.",
            },
        }

        band_info = BAND_CONFIG.get(risk_band, BAND_CONFIG["MODERATE"])
        badge_class = band_info["badge_class"]
        risk_label = band_info["label"]
        risk_icon = band_info["icon"]
        risk_desc = band_info["desc"]

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
                    <div style="font-size: 11px; font-weight: 500; text-transform: uppercase;">Calibrated Default Probability</div>
                </div>
            </div>
            <div style="display: flex; gap: 12px; margin-top: -12px; margin-bottom: 16px; font-size: 12px; color: #64748b; flex-wrap: wrap;">
                <span style="background: #f1f5f9; padding: 4px 10px; border-radius: 6px;"><strong>Model:</strong> {model_version}</span>
                <span style="background: #f1f5f9; padding: 4px 10px; border-radius: 6px;"><strong>Calibration:</strong> {calibration_version}</span>
                <span style="background: #f1f5f9; padding: 4px 10px; border-radius: 6px;"><strong>Risk Policy:</strong> {policy_version}</span>
                <span style="background: #f1f5f9; padding: 4px 10px; border-radius: 6px;"><strong>Features:</strong> {feature_count}</span>
            </div>
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:14px 18px; margin-bottom:20px;">
                <div style="font-size:13px; font-weight:600; color:#1e293b; margin-bottom:6px;">📖 Understanding Your Assessment Results:</div>
                <div style="font-size:12px; color:#475569; line-height:1.65;">
                    • <strong>Calibrated Default Probability ({pct_prob:.2f}%):</strong> The statistically estimated probability that an applicant with these financial attributes will experience a 90+ days payment delinquency over the loan lifecycle.<br>
                    • <strong>Assigned Risk Band ({risk_label}):</strong> Classified objectively under Phase 1B technical thresholds:<br>
                      &nbsp;&nbsp;&nbsp;&nbsp;🟢 <em>Low Risk Band (&lt; 15%)</em>: Favorable creditworthiness, high stability, minimal likelihood of default.<br>
                      &nbsp;&nbsp;&nbsp;&nbsp;🟡 <em>Moderate Risk Band (15% - 40%)</em>: Standard applicant risk; acceptable repayment capacity under standard underwriting.<br>
                      &nbsp;&nbsp;&nbsp;&nbsp;🟠 <em>Elevated Risk Band (40% - 70%)</em>: Elevated risk signals (e.g. higher debt burden or lower bureau ratings).<br>
                      &nbsp;&nbsp;&nbsp;&nbsp;🔴 <em>High Risk Band (≥ 70%)</em>: Significant risk indicators; high statistical probability of repayment distress.
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
                    <div style="font-size:11px; color:#64748b; margin-top:4px;">Loan to Annual Income<br><strong>(Safer &lt; 2.5x)</strong></div>
                </div>""",
                unsafe_allow_html=True,
            )
        with rcol2:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Annuity Burden</div>
                    <div class="metric-val">{annuity_burden:.1f}%</div>
                    <div style="font-size:11px; color:#64748b; margin-top:4px;">EMI % of Annual Income<br><strong>(Safer &lt; 25%)</strong></div>
                </div>""",
                unsafe_allow_html=True,
            )
        with rcol3:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Loan to Goods Value</div>
                    <div class="metric-val">{ltv:.1f}%</div>
                    <div style="font-size:11px; color:#64748b; margin-top:4px;">Financing Coverage Ratio<br><strong>(100% = Full Loan)</strong></div>
                </div>""",
                unsafe_allow_html=True,
            )
        with rcol4:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Income / Household</div>
                    <div class="metric-val">₹{inc_per_capita:,.0f}</div>
                    <div style="font-size:11px; color:#64748b; margin-top:4px;">Annual Per-Capita Income<br><strong>(Higher is Safer)</strong></div>
                </div>""",
                unsafe_allow_html=True,
            )
        with rcol5:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Avg Bureau Score</div>
                    <div class="metric-val">{avg_ext:.2f}</div>
                    <div style="font-size:11px; color:#64748b; margin-top:4px;">Bureau Rating (0 to 1)<br><strong>(Higher is Safer)</strong></div>
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
                "assessment_id": "AEGIS-PHASE1B-DEMO",
                "default_probability": prob,
                "risk_band": risk_band,
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
