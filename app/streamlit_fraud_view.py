"""
streamlit_fraud_view.py

Streamlit view component for Phase 2 Fraud Detection.
Renders the authenticated transaction form, calls the FastAPI prediction API,
displays the calibrated risk result, and presents recent transaction history.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

from app.config import API_BASE_URL


FRAUD_PRESETS: Dict[str, Dict[str, Any]] = {
    "🟢 Low-Risk Retail Purchase ($45.00)": {
        "amount": 45.00,
        "customer_id": "CUST_RETAIL_01",
        "card_id": "9633",
        "device_id": "DEV_SAMSUNG_S23",
        "merchant_id": "MERCH_GROCERY_01",
        "email_domain": "gmail.com",
        "address_id": "299.0",
        "product_code": "W",
        "card_network": "visa",
        "card_type": "debit",
        "ip_address": "72.14.201.1",
        "country": "US",
    },
    "🟡 Moderate-Risk Electronics ($1,250.00)": {
        "amount": 1250.00,
        "customer_id": "CUST_ELEC_88",
        "card_id": "9633",
        "device_id": "DEV_NEW_CHROME_WIN",
        "merchant_id": "MERCH_BESTBUY",
        "email_domain": "yahoo.com",
        "address_id": "299.0",
        "product_code": "W",
        "card_network": "mastercard",
        "card_type": "credit",
        "ip_address": "198.51.100.42",
        "country": "US",
    },
    "🔴 High-Risk Rapid Wire Transfer ($25,000.00)": {
        "amount": 25000.00,
        "customer_id": "C100",
        "card_id": "CARD100",
        "device_id": "DEV20",
        "merchant_id": "M500",
        "email_domain": "protonmail.com",
        "address_id": "143.0",
        "product_code": "R",
        "card_network": "discover",
        "card_type": "credit",
        "ip_address": "203.0.113.195",
        "country": "CY",
    },
}


def render_fraud_detection_page(api_base_url: Optional[str] = None):
    """Renders the complete Phase 2 Fraud Detection Interface."""
    api_base_url = api_base_url or API_BASE_URL

    # 1. Page Header & Banner
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%); color: white; padding: 22px 28px; border-radius: 14px; margin-bottom: 22px; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 4px 20px rgba(0,0,0,0.15);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 28px;">🛡️</span>
                    <div>
                        <h2 style="margin: 0; font-size: 22px; font-weight: 700; color: #f8fafc; letter-spacing: -0.3px;">AegisFin AI — Fraud Detection Engine</h2>
                        <div style="font-size: 13px; color: #94a3b8; margin-top: 2px;">Real-Time Calibrated XGBoost Inference • Anti-Leakage Feature Pipeline • Supabase Entity History</div>
                    </div>
                </div>
                <div style="background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.35); padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; color: #38bdf8;">
                    ⚡ Phase 2 Champion Model Active
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Access Token Verification
    token = st.session_state.get("access_token")
    if not token:
        st.error("🔒 Authentication Required: No active session token found. Please sign in via the sidebar.")
        return

    auth_headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # 3. Top Action Bar: Presets & Controls
    col_preset, col_refresh = st.columns([3, 1])
    with col_preset:
        preset_names = list(FRAUD_PRESETS.keys())
        selected_preset = st.selectbox(
            "⚡ Load Benchmark Transaction Scenario",
            options=preset_names,
            index=0,
            key="fraud_preset_selector",
            help="Select a benchmark transaction to populate the evaluation form.",
        )
    with col_refresh:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset Transaction ID", use_container_width=True):
            st.session_state["fraud_txn_id_override"] = f"TXN_{uuid.uuid4().hex[:8].upper()}"
            st.rerun()

    preset_data = FRAUD_PRESETS[selected_preset]
    default_txn_id = st.session_state.get("fraud_txn_id_override") or f"TXN_{uuid.uuid4().hex[:8].upper()}"

    # 4. Interactive Transaction Form
    col_form, col_result = st.columns([1.1, 0.9])

    with col_form:
        st.markdown("### 📝 Transaction Evaluation Form")
        with st.form(key="fraud_evaluation_form"):
            col1, col2 = st.columns(2)
            with col1:
                txn_id_input = st.text_input(
                    "Transaction ID *",
                    value=default_txn_id,
                    help="Unique payment transaction reference (enforced strictly unique).",
                )
                customer_id_input = st.text_input(
                    "Customer ID *",
                    value=preset_data["customer_id"],
                    help="Unique identifier for customer/account.",
                )
                card_id_input = st.text_input(
                    "Card ID (card1) *",
                    value=preset_data["card_id"],
                    help="Payment card issuer ID (numeric or identifier).",
                )
                product_code_input = st.selectbox(
                    "Product Code",
                    options=["W", "H", "C", "S", "R"],
                    index=["W", "H", "C", "S", "R"].index(preset_data.get("product_code", "W")),
                    help="W=Web, H=Home, C=Commercial, S=Specialty, R=Retail",
                )
                email_domain_input = st.text_input(
                    "Email Domain",
                    value=preset_data.get("email_domain", "gmail.com"),
                    help="Purchaser email domain (maps to P_emaildomain).",
                )
                ip_address_input = st.text_input(
                    "Client IP Address",
                    value=preset_data.get("ip_address", "192.168.1.1"),
                )

            with col2:
                amount_input = st.number_input(
                    "Amount ($ USD) *",
                    min_value=0.0,
                    value=float(preset_data["amount"]),
                    step=10.0,
                    format="%.2f",
                    help="Transaction monetary value.",
                )
                device_id_input = st.text_input(
                    "Device ID / Fingerprint",
                    value=preset_data.get("device_id", "DEV20"),
                    help="Client hardware fingerprint or device string.",
                )
                merchant_id_input = st.text_input(
                    "Merchant ID",
                    value=preset_data.get("merchant_id", "M500"),
                    help="Merchant or POS identifier.",
                )
                address_id_input = st.text_input(
                    "Address / Zip ID (addr1)",
                    value=preset_data.get("address_id", "299.0"),
                    help="Billing location identifier (addr1).",
                )
                col_net, col_type = st.columns(2)
                with col_net:
                    card_network_input = st.selectbox(
                        "Card Network",
                        options=["visa", "mastercard", "discover", "american express"],
                        index=["visa", "mastercard", "discover", "american express"].index(
                            preset_data.get("card_network", "visa")
                        ),
                    )
                with col_type:
                    card_type_input = st.selectbox(
                        "Card Type",
                        options=["debit", "credit"],
                        index=["debit", "credit"].index(preset_data.get("card_type", "debit")),
                    )
                country_input = st.text_input(
                    "Country Code",
                    value=preset_data.get("country", "US"),
                )

            timestamp_input = st.text_input(
                "Timestamp (UTC ISO-8601)",
                value=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                help="Transaction event timestamp in UTC.",
            )

            submit_btn = st.form_submit_button(
                "⚡ Analyze Transaction",
                type="primary",
                use_container_width=True,
            )

        if submit_btn:
            if not txn_id_input.strip():
                st.error("Transaction ID cannot be empty.")
            elif not customer_id_input.strip():
                st.error("Customer ID cannot be empty.")
            elif not card_id_input.strip():
                st.error("Card ID cannot be empty.")
            elif amount_input < 0:
                st.error("Amount must be greater than or equal to 0.")
            else:
                payload = {
                    "transaction_id": txn_id_input.strip(),
                    "amount": float(amount_input),
                    "transaction_timestamp": timestamp_input.strip() if timestamp_input.strip() else datetime.now(timezone.utc).isoformat(),
                    "customer_id": customer_id_input.strip(),
                    "card_id": card_id_input.strip(),
                    "device_id": device_id_input.strip() if device_id_input else None,
                    "merchant_id": merchant_id_input.strip() if merchant_id_input else None,
                    "email_domain": email_domain_input.strip() if email_domain_input else None,
                    "address_id": address_id_input.strip() if address_id_input else None,
                    "product_code": product_code_input,
                    "card_network": card_network_input,
                    "card_type": card_type_input,
                    "ip_address": ip_address_input.strip() if ip_address_input else None,
                    "country": country_input.strip() if country_input else None,
                }

                with st.spinner("🤖 Consulting historical entity state & evaluating model risk..."):
                    try:
                        url = f"{api_base_url.rstrip('/')}/api/v1/fraud/predict"
                        resp = requests.post(url, json=payload, headers=auth_headers, timeout=12)

                        if resp.status_code == 200:
                            st.session_state["last_fraud_result"] = resp.json()
                            st.session_state["last_fraud_payload"] = payload
                            # Rotate ID for next submission
                            st.session_state["fraud_txn_id_override"] = f"TXN_{uuid.uuid4().hex[:8].upper()}"
                            st.success("✅ Transaction successfully analyzed and persisted!")
                        elif resp.status_code == 401:
                            st.error("🔒 Authentication Expired (HTTP 401): Please log in again to refresh your session.")
                        elif resp.status_code == 409:
                            detail = resp.json().get("detail", "Transaction ID has already been evaluated.") if resp.headers.get("content-type", "").startswith("application/json") else "Transaction ID has already been evaluated."
                            st.warning(f"⚠️ Duplicate Transaction (HTTP 409): {detail}")
                        elif resp.status_code in (400, 422):
                            detail = resp.json().get("detail", "Validation error occurred.") if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                            st.error(f"❌ Validation Error (HTTP {resp.status_code}): {detail}")
                        elif resp.status_code == 500:
                            st.error("❌ Internal Server Error (HTTP 500): The fraud service encountered an internal error. Please try again later.")
                        else:
                            detail = resp.json().get("detail") if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                            st.error(f"⚠️ Prediction Failed (HTTP {resp.status_code}): {detail}")
                    except Exception as exc:
                        st.error(f"❌ Connection Failure: Could not reach API at {api_base_url}: {exc}")

    # 5. Prediction Results Column
    with col_result:
        st.markdown("### 📊 Decision & Risk Assessment")
        last_res = st.session_state.get("last_fraud_result")
        last_payload = st.session_state.get("last_fraud_payload")

        if last_res:
            fraud_prob = float(last_res.get("calibrated_fraud_probability", last_res.get("fraud_probability", 0.0)))
            raw_prob = float(last_res.get("raw_fraud_probability", last_res.get("raw_probability", 0.0)))
            band = str(last_res.get("risk_band", last_res.get("fraud_band", "LOW"))).upper()
            decision = str(last_res.get("recommended_action", last_res.get("decision", "ALLOW"))).upper()
            prob_pct = f"{fraud_prob * 100:.2f}%"
            raw_pct = f"{raw_prob * 100:.2f}%"

            # Dynamic Band Badges & Color Palette (Phase 2 4-Tier Canonical + Legacy)
            badge_colors = {
                "LOW": {"bg": "rgba(16, 185, 129, 0.12)", "text": "#059669", "border": "#10b981", "icon": "🟢"},
                "MEDIUM": {"bg": "rgba(59, 130, 246, 0.12)", "text": "#2563eb", "border": "#3b82f6", "icon": "🔵"},
                "HIGH": {"bg": "rgba(245, 158, 11, 0.12)", "text": "#d97706", "border": "#f59e0b", "icon": "🟡"},
                "CRITICAL": {"bg": "rgba(239, 68, 68, 0.12)", "text": "#dc2626", "border": "#ef4444", "icon": "🔴"},
                "REVIEW": {"bg": "rgba(245, 158, 11, 0.12)", "text": "#d97706", "border": "#f59e0b", "icon": "🟡"},
            }
            colors = badge_colors.get(band, badge_colors["LOW"])

            # Render styled container card with unindented, clean HTML
            card_html = (
                f'<div style="background: white; border: 2px solid {colors["border"]}; border-radius: 14px; padding: 18px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 14px;">'
                f'<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 12px; margin-bottom: 14px;">'
                f'<div><div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #64748b; font-weight: 600;">Evaluated Transaction</div>'
                f'<div style="font-size: 17px; font-weight: 700; color: #0f172a;">{last_res.get("transaction_id")}</div></div>'
                f'<div style="background: {colors["bg"]}; color: {colors["text"]}; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 700; border: 1px solid {colors["border"]};">'
                f'{colors["icon"]} {band} RISK</div></div>'
                f'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px;">'
                f'<div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; text-align: center;">'
                f'<div style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase;">Calibrated Probability</div>'
                f'<div style="font-size: 26px; font-weight: 800; color: {colors["text"]}; margin-top: 2px;">{prob_pct}</div>'
                f'<div style="font-size: 10px; color: #94a3b8; margin-top: 2px;">Platt Calibrated</div></div>'
                f'<div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; text-align: center;">'
                f'<div style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase;">Recommended Action</div>'
                f'<div style="font-size: 20px; font-weight: 800; color: #0f172a; margin-top: 4px;">{decision}</div>'
                f'<div style="font-size: 10px; color: #94a3b8; margin-top: 2px;">Policy Action</div></div></div>'
                f'<div style="display: flex; justify-content: space-between; align-items: center; background: #faf5ff; border: 1px solid #e9d5ff; border-radius: 8px; padding: 8px 12px; margin-bottom: 12px; font-size: 12px;">'
                f'<span style="color: #6b21a8; font-weight: 600;">🌳 Raw XGBoost Tree Score:</span>'
                f'<span style="font-weight: 700; color: #581c87; font-size: 13px;">{raw_pct} (p_raw={raw_prob:.4f})</span></div>'
                f'<div style="background: #f1f5f9; border-radius: 10px; padding: 12px 14px; font-size: 12px; color: #475569; line-height: 1.6;">'
                f'<div style="display: flex; justify-content: space-between;"><strong>Model:</strong> <span>{last_res.get("model_name")} ({last_res.get("model_version")})</span></div>'
                f'<div style="display: flex; justify-content: space-between;"><strong>Calibration:</strong> <span>{last_res.get("calibration_version", "N/A")} ({last_res.get("calibration_method", "Platt")})</span></div>'
                f'<div style="display: flex; justify-content: space-between;"><strong>Policy Thresholds:</strong> <span>Med ≥ 10% | High ≥ 40% | Crit ≥ 80%</span></div>'
                f'<div style="display: flex; justify-content: space-between;"><strong>Features:</strong> <span>{last_res.get("feature_count", 62)} Production Features (v2.1.0)</span></div>'
                f'<div style="display: flex; justify-content: space-between;"><strong>Inference Latency:</strong> <span>⚡ {last_res.get("prediction_latency_ms", 0.0):.1f} ms</span></div></div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

            with st.expander("💡 Understanding Platt Probability Calibration", expanded=False):
                st.markdown(
                    """
                    - **Platt Probability Calibration Pipeline:**
                      The Phase 2 champion model calibrates raw XGBoost probabilities using the saved **Platt Logistic Regressor** fitted to validation log-odds:
                      $$\\text{logit}(P_{\\text{raw}}) = \\ln\\left(\\frac{P_{\\text{raw}}}{1 - P_{\\text{raw}}}\\right)$$
                      $$P_{\\text{calibrated}} = \\sigma\\left(\\beta_0 + \\beta_1 \\times \\text{logit}(P_{\\text{raw}})\\right)$$
                      where $\\beta_0 \\approx -1.2370$ (intercept) and $\\beta_1 \\approx 0.8300$ (coefficient).
                    - **Dynamic Model Calibration:**
                      - Low-risk standard purchase (e.g. $P_{\\text{raw}} = 0.0025$): $\\text{logit} \\approx -5.989 \\implies P_{\\text{calibrated}} \\approx \\mathbf{0.20\\%}$ (🟢 Low Risk)
                      - Medium-risk transaction (e.g. $P_{\\text{raw}} = 0.55$): $\\text{logit} \\approx +0.201 \\implies P_{\\text{calibrated}} \\approx \\mathbf{25.54\\%}$ (🟡 Review Risk)
                      - High-risk transaction (e.g. $P_{\\text{raw}} = 0.85$): $\\text{logit} \\approx +1.735 \\implies P_{\\text{calibrated}} \\approx \\mathbf{55.06\\%}$ (🔴 High Risk)
                    - **Risk Policy Thresholds (Saved in Model Artifact):**
                      - **🟢 LOW RISK**: Calibrated $< 24.52\\%$ $\\to$ Decision: `ALLOW`
                      - **🟡 REVIEW RISK**: $24.52\\% \\le \\text{Calibrated} < 35.44\\%$ $\\to$ Decision: `MANUAL_REVIEW`
                      - **🔴 HIGH RISK**: $\\text{Calibrated} \\ge 35.44\\%$ $\\to$ Decision: `BLOCK`
                    """
                )

            if last_payload:
                with st.expander("📄 Submitted Transaction Parameters", expanded=False):
                    st.json(last_payload)
        else:
            st.info("👈 Enter transaction parameters and click **Analyze Transaction** to view the live fraud prediction result.")

    st.markdown("---")

    # 6. Transaction History & High-Risk Section
    tab_hist, tab_high_risk = st.tabs(["📜 Transaction History", "🚨 Recent High-Risk Transactions"])

    with tab_hist:
        col_hdr, col_btn = st.columns([4, 1])
        with col_hdr:
            st.markdown("#### Live Ingested Transactions & Model Predictions")
        with col_btn:
            refresh_hist = st.button("🔄 Refresh Table", key="btn_refresh_history", use_container_width=True)

        try:
            hist_url = f"{api_base_url.rstrip('/')}/api/v1/fraud/history?limit=15"
            h_resp = requests.get(hist_url, headers=auth_headers, timeout=6)
            if h_resp.status_code == 200:
                items = h_resp.json().get("items", [])
                if items:
                    df_items = pd.DataFrame(items)
                    display_cols = [
                        "transaction_id",
                        "amount",
                        "customer_id",
                        "timestamp",
                        "calibrated_fraud_probability",
                        "risk_band",
                        "recommended_action",
                        "fraud_probability",
                        "fraud_band",
                        "decision",
                        "model_version",
                    ]
                    available_cols = [c for c in display_cols if c in df_items.columns]
                    df_display = df_items[available_cols].copy()
                    if "calibrated_fraud_probability" in df_display.columns:
                        df_display["calibrated_fraud_probability"] = df_display["calibrated_fraud_probability"].apply(
                            lambda p: f"{float(p)*100:.2f}%" if pd.notnull(p) else "N/A"
                        )
                    if "fraud_probability" in df_display.columns:
                        df_display["fraud_probability"] = df_display["fraud_probability"].apply(
                            lambda p: f"{float(p)*100:.2f}%" if pd.notnull(p) else "N/A"
                        )
                    if "amount" in df_display.columns:
                        df_display["amount"] = df_display["amount"].apply(lambda a: f"${float(a):,.2f}")
                    st.dataframe(df_display, use_container_width=True, height=280)
                else:
                    st.info("No transaction history records found in Supabase yet.")
            elif h_resp.status_code == 401:
                st.warning("🔒 Session expired (HTTP 401). Please re-authenticate via the sidebar.")
            else:
                st.warning(f"Could not load transaction history: {h_resp.status_code}")
        except Exception as exc:
            st.warning(f"Unable to load transaction history: {exc}")

    with tab_high_risk:
        st.markdown("#### Recent HIGH / REVIEW Risk Flagged Transactions")
        try:
            risk_url = f"{api_base_url.rstrip('/')}/api/v1/fraud/high-risk?limit=5"
            r_resp = requests.get(risk_url, headers=auth_headers, timeout=6)
            if r_resp.status_code == 200:
                risk_items = r_resp.json().get("items", [])
                if risk_items:
                    for item in risk_items:
                        r_band = item.get("risk_band") or item.get("fraud_band", "REVIEW")
                        r_act = item.get("recommended_action") or item.get("decision", "MANUAL_REVIEW")
                        r_prob = item.get("calibrated_fraud_probability") or item.get("fraud_probability", 0.0)
                        b_color = "#dc2626" if r_band in ("CRITICAL", "HIGH") else "#d97706"
                        st.markdown(
                            f"""
                            <div style="border-left: 4px solid {b_color}; background: #fff5f5; padding: 10px 14px; border-radius: 0 8px 8px 0; margin-bottom: 8px;">
                                <strong>⚠️ Alert:</strong> Transaction <code>{item.get('transaction_id')}</code> | Customer: <code>{item.get('customer_id')}</code> | Amount: <strong>${item.get('amount', 0):,.2f}</strong><br>
                                <span style="font-size:12px; color:#4b5563;">Calibrated Prob: <strong>{float(r_prob)*100:.2f}%</strong> | Risk Band: <strong>{r_band}</strong> | Action: <strong>{r_act}</strong></span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.success("🎉 No high-risk transactions detected recently!")
            elif r_resp.status_code == 401:
                st.warning("🔒 Session expired (HTTP 401). Please re-authenticate via the sidebar.")
            else:
                st.warning(f"Could not load high-risk alerts: {r_resp.status_code}")
        except Exception as exc:
            st.warning(f"Unable to fetch high-risk alerts: {exc}")
