# AegisFin-AI Phase 2 — Streamlit Fraud Detection Frontend Integration

## Overview

The AegisFin-AI web application provides institutional financial risk intelligence. **Step 8** integrates the dedicated **Fraud Detection** page into the existing Streamlit dashboard (`streamlit_app.py`), enabling analysts to evaluate payment transactions in real time with the Phase 2 champion XGBoost model, Platt probability calibration, risk policy assignment, and Supabase historical persistence.

---

## Architecture & Navigation

The Streamlit frontend provides unified navigation across platform modules via a sidebar selector:
1. **`Credit Risk Assessment` (Phase 1)**: Application default risk scoring based on applicant profile, loan details, and credit bureau history.
2. **`Fraud Detection` (Phase 2)**: Transaction-level real-time fraud assessment evaluating velocity, entity history, card/device mismatch, and risk bands.

Both modules reside within the single canonical Streamlit application (`streamlit_app.py`) and share the core security gatekeeper.

---

## Authentication & Security Flow

### 1. Gatekeeper & Session Persistence
- The Fraud Detection page is strictly protected and accessible only to authenticated users.
- Unauthenticated visitors are intercepted by the login/signup portal before any dashboard controls or prediction interfaces are rendered.
- Streamlit session state stores:
  - `st.session_state["authenticated"]`: Boolean access flag.
  - `st.session_state["access_token"]`: Valid Supabase JWT token.
  - `st.session_state["user"]`: Authenticated analyst profile metadata (`user_id`, `email`, `full_name`, `role`).

### 2. Zero Secret Key Leakage
- `SUPABASE_SECRET_KEY` is strictly server-side and is **never** imported, referenced, or exposed in Streamlit session state or frontend code.
- Streamlit interacts with Supabase using `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY`.
- Backend communication passes the authenticated access token in the standard HTTP header:
  ```http
  Authorization: Bearer <access_token>
  ```

---

## Transaction Evaluation Form & Required Fields

The form accepts the exact schema defined by `FraudPredictionRequest`:

| Field Name | Streamlit Control | Type | Description |
| :--- | :--- | :--- | :--- |
| **`transaction_id`** | `st.text_input` | `str` | Strictly unique transaction identifier |
| **`amount`** | `st.number_input` | `float` | Transaction amount in USD ($) |
| **`customer_id`** | `st.text_input` | `str` | Account/cardholder ID |
| **`card_id`** | `st.text_input` | `str` | Payment card identifier (maps to `card1`) |
| **`device_id`** | `st.text_input` | `str` | Device fingerprint / hardware identifier |
| **`merchant_id`** | `st.text_input` | `str` | Merchant identifier / POS reference |
| **`timestamp`** | `st.text_input` | `str` | ISO-8601 UTC timestamp of transaction |
| **`product_code`** | `st.selectbox` | `str` | `W` (Web), `H` (Home), `C` (Commercial), `S` (Specialty), `R` (Retail) |
| **`card_network`** | `st.selectbox` | `str` | `visa`, `mastercard`, `discover`, `american express` |
| **`card_type`** | `st.selectbox` | `str` | `debit`, `credit` |
| **`email_domain`** | `st.text_input` | `str` | Purchaser email domain (e.g. `gmail.com`, `protonmail.com`) |
| **`address_id`** | `st.text_input` | `str` | Billing address / zip identifier (`addr1`) |
| **`ip_address`** | `st.text_input` | `str` | Client IP address |
| **`country`** | `st.text_input` | `str` | Billing country code (`US`, `CY`, etc.) |

### Built-in Benchmark Scenarios
Three pre-configured institutional benchmark scenarios can be loaded with one click:
- **🟢 Low-Risk Retail Purchase ($45.00)**: Everyday consumer grocery transaction with matching history.
- **🟡 Moderate-Risk Electronics ($1,250.00)**: Higher value purchase with new device fingerprint.
- **🔴 High-Risk Rapid Wire Transfer ($25,000.00)**: Elevated velocity, offshore IP/country, high amount.

---

## Backend API Dependencies

The frontend delegates 100% of machine learning inference, feature engineering, and database queries to the FastAPI backend:

| Endpoint | Method | Purpose | Auth Required |
| :--- | :--- | :--- | :--- |
| **`/api/v1/fraud/predict`** | `POST` | Primary inference: computes features, runs XGBoost model, Platt calibration, applies policy, logs transaction and prediction in Supabase | Yes (`Bearer <token>`) |
| **`/api/v1/fraud/history`** | `GET` | Retrieves paginated recent transactions and evaluation scores | Yes (`Bearer <token>`) |
| **`/api/v1/fraud/high-risk`** | `GET` | Fetches transactions flagged with `HIGH` or `REVIEW` risk bands | Yes (`Bearer <token>`) |
| **`/api/v1/fraud/health`** | `GET` | Checks fraud model service readiness, versions, and policy | No |

---

## Result Card & Visual Hierarchy

Upon successful inference, the UI displays a clean visual hierarchy:
- **Fraud Probability**: Formatted percentage (e.g. `82.31%`).
- **Risk Band**: Direct from backend policy (`LOW`, `REVIEW`, `HIGH`), styled with distinct semantic color badges.
- **Decision**: Actionable policy determination (`ALLOW`, `MANUAL_REVIEW`, `BLOCK`).
- **Model Metadata**:
  - Model Name & Version: `XGBoost (phase2-xgb-v1)`
  - Calibration Version: `phase2-platt-v1`
  - Policy Version: `phase2-policy-v1`
  - Inference Latency: Displayed in milliseconds (`18 ms`).
- **Submitted Transaction Parameters**: Kept accessible in a collapsible viewer for analyst review.

---

## Error Handling

All backend responses are handled gracefully without exposing server stack traces:
- **`401 Unauthorized`**: Prompts the analyst: *"Session Expired: Please log in again via the sidebar."*
- **`400 / 422 Validation Error`**: Outlines specific missing or invalid parameters without crashing the UI.
- **`409 Conflict`**: Detects duplicates: *"Duplicate Transaction: Transaction ID has already been evaluated and processed."*
- **`500 Internal Error`**: Displays a safe generic error message directing the user to retry or check logs.

---

## Local Development & Startup Commands

### 1. Environment Setup
Ensure `.env` exists in the workspace root or project folder containing:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_PUBLISHABLE_KEY=your-publishable-key
SUPABASE_SECRET_KEY=your-secret-key
API_BASE_URL=http://127.0.0.1:8000
```

### 2. Start FastAPI Backend
```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
API Documentation available at `http://127.0.0.1:8000/docs`.

### 3. Start Streamlit Frontend
```powershell
python -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```
Analyst Portal available at `http://localhost:8501`.

### 4. Run Automated Regression Test Suite
```powershell
python -m pytest -v
```
Verifies all 76 tests (Phase 1, Supabase CRUD, Supabase Auth, Feature Engine, Model Service, API endpoints, and Streamlit integration).
