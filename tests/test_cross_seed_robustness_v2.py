"""
tests/test_cross_seed_robustness_v2.py

Automated verification tests for AegisFin Phase 2 Step 11: Cross-Seed Robustness Test:
1. Seed-123 dataset artifacts exist, are valid, and strictly adhere to the 62-feature contract.
2. Frozen XGBoost v2 base model and Platt calibrator v2 remain frozen and loadable.
3. Splits v2 (train, validation, calibration, policy, test) remain completely untouched.
4. Evaluation reports (JSON and MD) exist and contain the mandatory governance disclaimer.
5. Cross-seed evaluation stability: discriminative metrics and calibration errors meet robustness thresholds.
6. Scenario diagnostics verification: 7 scenarios audited without scenario feature leakage.
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.production_feature_definitions import VALID_FEATURE_NAMES

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "behavioral"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
SPLITS_V2_DIR = DATA_DIR / "splits_v2"

RAW_SEED123_PATH = DATA_DIR / "raw_transactions_100k_seed123.csv"
FEAT_SEED123_PATH = DATA_DIR / "production_features_100k_seed123.csv"
META_SEED123_PATH = DATA_DIR / "behavioral_dataset_metadata_seed123.json"

BASE_MODEL_PATH = MODELS_DIR / "aegisfin_xgboost_baseline_v2.pkl"
CALIBRATOR_PATH = MODELS_DIR / "aegisfin_probability_calibrator_v2.pkl"

REPORT_JSON_PATH = REPORTS_DIR / "cross_seed_robustness_v2.json"
REPORT_MD_PATH = REPORTS_DIR / "cross_seed_robustness_v2.md"

MANDATORY_DISCLAIMER = (
    "This is a cross-seed generalization test within the AegisFin synthetic behavioral benchmark. "
    "It does not establish real-world fraud performance."
)


def test_seed123_dataset_artifacts_exist_and_valid():
    """Requirement 1: Verify seed-123 raw, features, and metadata artifacts exist and are valid."""
    assert RAW_SEED123_PATH.exists(), f"Raw seed-123 CSV missing: {RAW_SEED123_PATH}"
    assert FEAT_SEED123_PATH.exists(), f"Feature seed-123 CSV missing: {FEAT_SEED123_PATH}"
    assert META_SEED123_PATH.exists(), f"Metadata seed-123 JSON missing: {META_SEED123_PATH}"

    with open(META_SEED123_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["random_seed"] == 123
    assert meta["total_transactions"] == 100000
    assert meta["production_feature_count"] == 62
    assert meta["is_chronological"] is True
    assert meta["has_nan_inf"] is False

    # Check feature CSV schema and exact columns
    with open(FEAT_SEED123_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        row_count = sum(1 for _ in reader)

    assert row_count == 100000, f"Expected 100,000 rows, found {row_count}"
    expected_header = ["transaction_id", "fraud_label"] + VALID_FEATURE_NAMES
    assert header == expected_header, "Production feature header or column order mismatch"
    assert "scenario" not in header, "scenario column must NEVER appear in feature dataset"


def test_v2_models_remain_untouched_and_frozen():
    """Requirement 2: Base XGBoost model and probability calibrator remain frozen and valid."""
    assert BASE_MODEL_PATH.exists(), f"Base model missing: {BASE_MODEL_PATH}"
    with open(BASE_MODEL_PATH, "rb") as f:
        model_artifact = pickle.load(f)

    assert isinstance(model_artifact, dict)
    assert "model" in model_artifact
    assert model_artifact["feature_names"] == VALID_FEATURE_NAMES
    assert np.isclose(model_artifact["scale_pos_weight"], 12.109023, atol=1e-5)

    assert CALIBRATOR_PATH.exists(), f"Calibrator missing: {CALIBRATOR_PATH}"
    with open(CALIBRATOR_PATH, "rb") as f:
        calibrator_artifact = pickle.load(f)

    assert isinstance(calibrator_artifact, dict)
    assert "calibrator" in calibrator_artifact
    assert calibrator_artifact["selected_method"] == "sigmoid"


def test_v2_splits_remain_untouched():
    """Requirement 3: Existing splits_v2 files remain untouched."""
    expected_files = {
        "train.csv": 60000,
        "validation.csv": 10000,
        "calibration.csv": 10000,
        "policy.csv": 10000,
        "test.csv": 10000,
    }
    for filename, expected_rows in expected_files.items():
        file_path = SPLITS_V2_DIR / filename
        assert file_path.exists(), f"Split file missing: {file_path}"
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            row_count = sum(1 for _ in reader)
        assert row_count == expected_rows, f"{filename} row count changed: {row_count} != {expected_rows}"


def test_cross_seed_reports_exist_and_contain_disclaimer():
    """Requirement 4: JSON and Markdown evaluation reports exist with mandatory disclaimer."""
    assert REPORT_JSON_PATH.exists(), f"JSON report missing: {REPORT_JSON_PATH}"
    assert REPORT_MD_PATH.exists(), f"Markdown report missing: {REPORT_MD_PATH}"

    with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    assert report_data["governance_disclaimer"] == MANDATORY_DISCLAIMER
    assert "benchmark_comparison" in report_data
    assert "scenario_diagnostics" in report_data

    with open(REPORT_MD_PATH, "r", encoding="utf-8") as f:
        md_content = f.read()

    assert MANDATORY_DISCLAIMER in md_content, "Mandatory disclaimer missing from Markdown report"


def test_cross_seed_evaluation_stability_thresholds():
    """Requirement 5: Model and calibrator generalizability meets stability thresholds on seed-123."""
    with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    raw_metrics = report_data["raw_model_metrics"]
    cal_metrics = report_data["calibrated_model_metrics"]

    # High discriminative stability
    assert raw_metrics["pr_auc"] >= 0.95, f"PR-AUC too low: {raw_metrics['pr_auc']}"
    assert raw_metrics["roc_auc"] >= 0.99, f"ROC-AUC too low: {raw_metrics['roc_auc']}"

    # ECE must remain low post-calibration
    assert cal_metrics["ece"] <= 0.05, f"Calibrated ECE too high: {cal_metrics['ece']}"

    # Stability vs seed-42 validation benchmark (PR-AUC 0.999221)
    pr_auc_diff = abs(raw_metrics["pr_auc"] - 0.999221)
    assert pr_auc_diff < 0.02, f"PR-AUC diverged excessively across seeds: diff={pr_auc_diff}"


def test_scenario_diagnostics_isolation():
    """Requirement 6: Scenario diagnostics audit all 7 scenarios with high recall and low FPR."""
    with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    scenarios = report_data["scenario_diagnostics"]
    assert len(scenarios) == 7, f"Expected 7 scenarios, found {len(scenarios)}"

    scenario_by_name = {sc["scenario"]: sc for sc in scenarios}

    # Fraudulent scenarios should have high recall
    fraud_scenarios = [
        "Account Takeover",
        "Burst Fraud",
        "Card Testing",
        "Shared Device / IP Ring",
        "Stolen Card",
    ]
    for name in fraud_scenarios:
        assert name in scenario_by_name, f"Missing scenario: {name}"
        assert scenario_by_name[name]["recall_at_0_50"] >= 0.95, f"Low recall in {name}: {scenario_by_name[name]['recall_at_0_50']}"

    # Legitimate scenarios should have low false positive rate (< 1%)
    for name in ["Legitimate Stable", "Legitimate High-Value"]:
        assert name in scenario_by_name, f"Missing scenario: {name}"
        fpr = scenario_by_name[name]["false_positives"] / scenario_by_name[name]["transaction_count"]
        assert fpr <= 0.01, f"Elevated false positive rate in {name}: {fpr:.4f}"
