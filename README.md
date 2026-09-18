# AegisFin-AI

**AegisFin AI** is an end-to-end banking intelligence platform that combines credit risk scoring, real-time fraud detection, regulatory RAG, and multi-agent AI to generate explainable, auditable financial decisions with human oversight.

---

# Phase 1 — Credit Risk

Phase 1 delivers the production credit default risk scoring subsystem, consisting of Phase 1A (base model development & validation) and Phase 1B (probability calibration, deterministic risk banding, and application integration).

### Subphase Breakdown

* **Phase 1A — Model Development & Validation**:
  * Algorithm: Gradient Boosted Decision Trees (`XGBoost 3.2.0`, `credit-xgb-v1.0.0`)
  * Data: Home Credit application data with 183 engineered features (DTI ratios, per-capita metrics, bureau aggregates, missing indicators, category encoding)
  * Imbalance handling: Square-root positive weighting (`scale_pos_weight = 3.375`)
  * Validation: Optuna hyperparameter tuning; untouched test evaluation (`ROC-AUC = 0.7683`, `PR-AUC = 0.2509`, `Gini = 0.5366`, `KS = 0.4047`)
  * Preserved Artifact: `models/aegisfin_phase1_final_model.pkl` (FROZEN)

* **Phase 1B — Probability Calibration & Deterministic Risk Bands**:
  * Calibration: Sigmoid / Platt Scaling (`credit-calibration-v1.0.0`) fitted via logistic regression over log-odds (`safe_logit`) on out-of-fold validation predictions.
  * Calibration performance: Reduced test Brier score from **0.086270** (raw) down to **0.067362** (calibrated) while preserving perfect rank ordering.
  * Risk Policy: Deterministic risk-band function (`credit-risk-policy-v1.0.0`) with provisional technical/demo thresholds:
    * 🟢 **LOW**: Calibrated PD < 0.15
    * 🟡 **MODERATE**: 0.15 ≤ Calibrated PD < 0.40
    * 🟠 **ELEVATED**: 0.40 ≤ Calibrated PD < 0.70
    * 🔴 **HIGH**: Calibrated PD ≥ 0.70
    *(Note: These are provisional technical demo thresholds, not official banking lending policy or automated credit approval/decline rules).*
  * Production Artifact: `models/aegisfin_phase1b_calibrated_model.pkl` (PRIMARY INFERENCE ARTIFACT)

---

## 🔄 End-to-End Inference Pipeline

Inference follows a strictly deterministic, non-training pipeline executed in backend application code:

```text
Raw Application & Bureau Data
             ↓
Phase 1A Feature Engineering
             ↓
Phase 1A Preprocessing State (imputation, vocabularies, reindexing)
             ↓
Frozen XGBoost Model (predict_proba)
             ↓
Raw Default Probability
             ↓
Phase 1B Calibrator (Platt scaling via safe_logit)
             ↓
Calibrated Default Probability (PD)
             ↓
Deterministic Risk-Band Function
             ↓
FastAPI Backend (/api/v1/predict)
             ↓
Streamlit Dashboard (Display/Client Layer)
```

> **Note on Explainability**: SHAP explainability is planned for Phase 3. No SHAP execution occurs during Phase 1 inference.

---

## 📦 Model Artifacts

```text
models/
├── aegisfin_phase1_final_model.pkl         ← Phase 1A Frozen Base Bundle (UNCHANGED)
└── aegisfin_phase1b_calibrated_model.pkl  ← Phase 1B Calibrated Production Bundle (PRIMARY)
```

### Phase 1B Bundle Structure (`aegisfin_phase1b_calibrated_model.pkl`)

The bundle is a self-contained pickle package containing:
* `model`: Frozen `XGBClassifier`
* `preprocessing_state`: Dict with `category_vocab`, `indicator_cols`, `numeric_cols`, `drop_cols`
* `selected_features`: List of 183 selected feature names in exact column order
* `categorical_features`: List of 16 categorical feature names
* `calibrator`: Fitted `LogisticRegression` Platt calibrator
* `calibration_method`: `"sigmoid_platt_scaling"`
* `risk_policy`: Dict defining threshold boundaries (`LOW_MAX: 0.15`, `MODERATE_MAX: 0.40`, `ELEVATED_MAX: 0.70`)
* `metadata`: Complete model, calibration, and policy versions, random state, and evaluation metrics

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application

To launch the interactive risk intelligence web dashboard:

```bash
python -m streamlit run streamlit_app.py
```
*Or using the runner utility:*
```bash
python run_all.py
```

### 3. Launch the FastAPI Backend Server

To run the standalone REST API:

```bash
python -m uvicorn app.main:app --reload
```
- **Swagger Documentation:** `http://127.0.0.1:8000/docs`
- **Health Check Endpoint:** `GET http://127.0.0.1:8000/health`
- **Prediction Endpoint:** `POST http://127.0.0.1:8000/api/v1/predict`

### 4. Run Both Frontend and Backend Concurrently

```bash
python run_all.py --all
```

---

## 📡 API Contract Example

### Request (`POST /api/v1/predict`)
```json
{
  "application": {
    "contract_type": "Cash loans",
    "credit_amount": 800000,
    "annuity_amount": 35000,
    "goods_price": 700000,
    "age_years": 32,
    "gender": "M",
    "annual_income": 450000
  },
  "enrichment": {
    "ext_source_1": 0.42,
    "ext_source_2": 0.62,
    "ext_source_3": 0.71
  }
}
```

### Response (`200 OK`)
```json
{
  "default_probability": 0.012954,
  "risk_band": "LOW",
  "model_name": "XGBoost",
  "model_version": "credit-xgb-v1.0.0",
  "calibration_version": "credit-calibration-v1.0.0",
  "policy_version": "credit-risk-policy-v1.0.0",
  "feature_count": 183
}
```

---

## 🧪 Automated Testing

Run the full automated test suite with pytest:

```bash
python -m pytest
```

Test coverage includes:
1. Phase 1B bundle loading & key verification
2. Frozen XGBoost model loading
3. Platt calibrator loading & coefficients
4. Preprocessing replay & categorical category integrity
5. Deterministic prediction consistency
6. Probability bounds ($0 \le p \le 1$)
7. Risk-band boundary edge cases (`0.149999`, `0.150000`, `0.399999`, `0.400000`, `0.699999`, `0.700000`)
8. FastAPI prediction response schema
9. Model, calibration, and policy version metadata validation
10. Missing and invalid input handling
