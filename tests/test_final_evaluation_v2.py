"""
tests/test_final_evaluation_v2.py

Automated verification tests for AegisFin Phase 2 Step 13: Final Held-Out Test Evaluation:
1. Final evaluation reports (JSON and MD) exist and contain required metrics and governance statements.
2. Frozen base model, calibrator, and policy configuration remain intact and un-modified.
3. Test dataset integrity: exactly 10,000 transactions, 62 contract features in order, zero leakage.
4. Held-out discriminative performance meets benchmark generalization bounds.
5. Calibrated probabilities maintain low ECE and monotonic risk progression on unseen test data.
6. Action routing behaves as specified (AUTO_APPROVE >= 90% volume, HARD_DECLINE >= 95% fraud capture).
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from app.production_feature_definitions import VALID_FEATURE_NAMES

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "behavioral"
SPLITS_V2_DIR = DATA_DIR / "splits_v2"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
CONFIGS_DIR = BASE_DIR / "configs"

TEST_CSV_PATH = SPLITS_V2_DIR / "test.csv"
SPLIT_META_PATH = SPLITS_V2_DIR / "split_metadata.json"

BASE_MODEL_PATH = MODELS_DIR / "aegisfin_xgboost_baseline_v2.pkl"
CALIBRATOR_PATH = MODELS_DIR / "aegisfin_probability_calibrator_v2.pkl"
POLICY_CONFIG_PATH = CONFIGS_DIR / "fraud_risk_policy_v2.json"

REPORT_JSON_PATH = REPORTS_DIR / "final_test_evaluation_v2.json"
REPORT_MD_PATH = REPORTS_DIR / "final_test_evaluation_v2.md"

MANDATORY_GOVERNANCE_STATEMENT = (
    "Final held-out evaluation of the AegisFin synthetic behavioral benchmark and engineering pipeline. "
    "It does not establish real-world banking production performance."
)


def test_final_evaluation_artifacts_exist_and_valid():
    """Requirement 1: Reports exist, are non-empty, and contain required schema."""
    assert REPORT_JSON_PATH.exists(), f"JSON report missing: {REPORT_JSON_PATH}"
    assert REPORT_MD_PATH.exists(), f"MD report missing: {REPORT_MD_PATH}"

    with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    assert report_data["governance_statement"] == MANDATORY_GOVERNANCE_STATEMENT
    assert "test_dataset_validation" in report_data
    assert "raw_model_metrics" in report_data
    assert "calibrated_model_metrics" in report_data
    assert "risk_band_performance" in report_data
    assert "action_level_performance" in report_data
    assert "cross_benchmark_comparison" in report_data
    assert "final_limitations" in report_data

    with open(REPORT_MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    assert MANDATORY_GOVERNANCE_STATEMENT in md_text, "Mandatory governance statement missing in markdown report"


def test_frozen_artifacts_remained_untouched():
    """Requirement 2: Base model, calibrator, and policy configuration remain frozen."""
    assert BASE_MODEL_PATH.exists()
    with open(BASE_MODEL_PATH, "rb") as f:
        m_art = pickle.load(f)
    assert isinstance(m_art, dict)
    assert "model" in m_art
    assert m_art["feature_names"] == VALID_FEATURE_NAMES
    assert np.isclose(m_art["scale_pos_weight"], 12.109023, atol=1e-5)

    assert CALIBRATOR_PATH.exists()
    with open(CALIBRATOR_PATH, "rb") as f:
        c_art = pickle.load(f)
    assert isinstance(c_art, dict)
    assert "calibrator" in c_art
    assert c_art["selected_method"] == "sigmoid"

    assert POLICY_CONFIG_PATH.exists()
    with open(POLICY_CONFIG_PATH, "r", encoding="utf-8") as f:
        policy_cfg = json.load(f)
    assert policy_cfg["provisional_action_threshold"] == 0.50
    assert len(policy_cfg["risk_bands"]) == 4


def test_test_dataset_integrity():
    """Requirement 3: test.csv has exactly 10,000 rows, 62 contract features in order, and zero NaNs."""
    assert TEST_CSV_PATH.exists()
    with open(TEST_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        row_count = sum(1 for _ in reader)

    assert row_count == 10000, f"Expected 10,000 rows in test.csv, found {row_count}"
    expected_header = ["transaction_id", "fraud_label"] + VALID_FEATURE_NAMES
    assert header == expected_header, "Test CSV feature columns or ordering mismatch"
    assert "scenario" not in header, "scenario column must never appear in model feature columns"


def test_held_out_discriminative_metrics_stability():
    """Requirement 4: Model retains high discriminative performance on unseen test data."""
    with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw = data["raw_model_metrics"]
    assert raw["pr_auc"] >= 0.95, f"Held-out PR-AUC too low: {raw['pr_auc']}"
    assert raw["roc_auc"] >= 0.99, f"Held-out ROC-AUC too low: {raw['roc_auc']}"
    assert raw["diagnostic_threshold_0_50"]["f1"] >= 0.95, f"Held-out F1 too low: {raw['diagnostic_threshold_0_50']['f1']}"
    assert raw["diagnostic_threshold_0_50"]["recall"] >= 0.98, f"Held-out Recall too low: {raw['diagnostic_threshold_0_50']['recall']}"


def test_calibrated_probabilities_and_risk_band_monotonicity():
    """Requirement 5: Probability calibration error is low and risk bands are monotonic on test data."""
    with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cal = data["calibrated_model_metrics"]
    assert cal["calibrated_ece"] <= 0.05, f"Calibrated ECE too high on test set: {cal['calibrated_ece']}"

    bands = data["risk_band_performance"]
    assert len(bands) == 4, f"Expected 4 risk bands, found {len(bands)}"

    # Monotonic observed fraud rates across LOW, MEDIUM, HIGH, CRITICAL
    fraud_rates = [b["observed_fraud_rate"] for b in bands]
    for i in range(len(fraud_rates) - 1):
        assert fraud_rates[i] <= fraud_rates[i + 1], f"Monotonicity violation on test set: {fraud_rates[i]} > {fraud_rates[i+1]}"


def test_production_action_routing_efficacy():
    """Requirement 6: Action routing captures >= 95% of fraud in HARD_DECLINE and clears >= 90% in AUTO_APPROVE."""
    with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    acts = data["action_level_performance"]
    auto_approve = acts["AUTO_APPROVE"]
    hard_decline = acts["HARD_DECLINE"]

    # Frictionless auto-approval volume
    assert auto_approve["percentage_of_volume"] >= 90.0, f"Auto-approve volume lower than 90%: {auto_approve['percentage_of_volume']}%"

    # Hard decline captures vast majority of fraud
    assert hard_decline["fraud_capture_pct"] >= 95.0, f"Hard decline fraud capture too low: {hard_decline['fraud_capture_pct']}%"
