# AegisFin-AI

**AegisFin AI** is an end-to-end banking intelligence platform that combines credit risk scoring, real-time fraud detection, regulatory RAG, and multi-agent AI to generate explainable, auditable financial decisions with human oversight.

---

# Phase 1 — Rich Model Integration & Web Dashboard

This package contains the complete production integration for the **AegisFin-AI Phase 1 XGBoost Credit Default Model**, featuring a **Streamlit Web Application**, a **FastAPI REST API**, and a **Model Pipeline Service**.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application

To launch the interactive risk intelligence web dashboard:

```bash
streamlit run streamlit_app.py
```
*Or using the runner utility:*
```bash
python run_all.py
```

### 3. Launch the FastAPI Backend Server

To run the standalone REST API:

```bash
uvicorn app.main:app --reload
```
- **Swagger Documentation:** `http://127.0.0.1:8000/docs`
- **Health Check Endpoint:** `GET http://127.0.0.1:8000/health`
- **Prediction Endpoint:** `POST http://127.0.0.1:8000/api/v1/predict`

### 4. Run Both Frontend and Backend Concurrently

```bash
python run_all.py --all
```

---

## 🌟 Web Dashboard Features

- **Interactive Loan Assessment Form**: Comprehensive, organized inputs covering:
  - 💵 *Loan & Demographics* (Credit, Annuity, Goods Price, Income, Age, Gender, Education, Family)
  - 💼 *Employment & Assets* (Tenure, Occupation, Organization, Car/Realty Ownership)
  - 🛡️ *Bureau Enrichment Data* (`EXT_SOURCE_1/2/3`, Days ID/Phone/Registration, Bureau Queries, Social Circle Risk)
- **1-Click Preset Profiles**: Instantly load *Prime Low-Risk*, *Moderate-Risk*, or *High-Risk* applicant profiles.
- **Dual Execution Modes**:
  - **Direct Mode (Local)**: High-performance in-process inference without needing an external server.
  - **REST API Mode**: Client-server inference connected to FastAPI `http://127.0.0.1:8000`.
- **Intelligent Risk Dashboard**:
  - Dynamic Default Probability risk gauge & color-coded classification banners.
  - Computed credit ratios (DTI, Annuity Burden, LTV, Household Income per Capita).
  - Comparative visual charts for External Bureau ratings and Debt Service thresholds.
  - One-click JSON Assessment Report export and full technical audit log inspector.

---

## 🧪 Testing

Run the automated test suite with pytest:

```bash
python -m pytest
```

---

## 📦 Model File

The trained Phase 1 XGBoost model bundle resides at:
```text
models/aegisfin_phase1_final_model.pkl
```

The bundle contains:
- Trained XGBoost Booster
- Preprocessing state & Category Vocabularies
- 183 Selected features list
- Model metadata
