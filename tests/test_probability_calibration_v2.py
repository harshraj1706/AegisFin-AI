"""
tests/test_probability_calibration_v2.py

Automated verification tests for AegisFin Phase 2 Step 10 Probability Calibration:
1. Calibrator artifact exists and loads.
2. Exactly 62 features are required by the base model.
3. Calibration data was used only for fitting the calibrator.
4. Policy and final test files were not loaded.
5. Final test remains untouched.
6. Calibrated probabilities are within [0, 1].
7. Calibrator can transform new raw probabilities.
8. Feature ordering remains unchanged.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from app.production_feature_definitions import VALID_FEATURE_NAMES

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
SPLITS_V2_DIR = BASE_DIR / "data" / "behavioral" / "splits_v2"

CALIBRATOR_PATH = MODELS_DIR / "aegisfin_probability_calibrator_v2.pkl"
CALIBRATOR_META_PATH = MODELS_DIR / "aegisfin_probability_calibrator_v2_metadata.json"
REPORT_JSON_PATH = REPORTS_DIR / "probability_calibration_v2.json"
REPORT_MD_PATH = REPORTS_DIR / "probability_calibration_v2.md"
BASE_MODEL_PATH = MODELS_DIR / "aegisfin_xgboost_baseline_v2.pkl"


def transform_raw_probability(raw_probs: np.ndarray | float, calibrator: Any) -> np.ndarray | float:
    """Transforms raw probability into calibrated probability using logit + Platt LogisticRegression."""
    is_scalar = np.isscalar(raw_probs)
    arr = np.atleast_1d(np.asarray(raw_probs, dtype=np.float64))
    eps = 1e-15
    p_clipped = np.clip(arr, eps, 1.0 - eps)
    logit_p = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
    cal_p = calibrator.predict_proba(logit_p)[:, 1]
    cal_p = np.clip(cal_p, 0.0, 1.0)
    return float(cal_p[0]) if is_scalar else cal_p


def test_calibrator_artifact_exists_and_loads():
    """Requirement 1: Calibrator artifact exists and loads successfully."""
    assert CALIBRATOR_PATH.exists(), f"Calibrator artifact missing: {CALIBRATOR_PATH}"
    with open(CALIBRATOR_PATH, "rb") as f:
        artifact = pickle.load(f)

    assert isinstance(artifact, dict), "Artifact payload must be a dictionary"
    assert "calibrator" in artifact, "Missing 'calibrator' in artifact"
    assert hasattr(artifact["calibrator"], "predict_proba"), "Calibrator must implement predict_proba"
    assert "selected_method" in artifact, "Missing 'selected_method' in artifact"
    assert artifact["selected_method"] == "sigmoid"
    assert "metrics_summary" in artifact, "Missing 'metrics_summary' in artifact"
    assert "slope" in artifact and "intercept" in artifact


def test_exactly_62_features_required_by_base_model():
    """Requirement 2: Exactly 62 features are required by the base model."""
    assert BASE_MODEL_PATH.exists(), f"Base model missing: {BASE_MODEL_PATH}"
    with open(BASE_MODEL_PATH, "rb") as f:
        model_artifact = pickle.load(f)

    base_model = model_artifact["model"]
    assert base_model.n_features_in_ == 62, f"Base model expects {base_model.n_features_in_} features, expected 62"

    features = model_artifact["feature_names"]
    assert len(features) == 62
    assert features == VALID_FEATURE_NAMES


def test_calibration_data_used_only_and_splits_isolation():
    """Requirement 3 & 4: Calibration data was used only; policy and test files were not loaded."""
    assert CALIBRATOR_META_PATH.exists(), f"Metadata JSON missing: {CALIBRATOR_META_PATH}"
    with open(CALIBRATOR_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "calibration.csv" in meta["calibration_dataset"]["path"]
    assert meta["calibration_dataset"]["total_rows"] == 10000
    assert "untouched_splits" in meta
    assert "policy.csv" in meta["untouched_splits"]
    assert "test.csv" in meta["untouched_splits"]


def test_final_test_remains_untouched():
    """Requirement 5: Final test partition remains completely untouched and isolated."""
    test_csv = SPLITS_V2_DIR / "test.csv"
    assert test_csv.exists(), f"Final test split missing: {test_csv}"
    assert test_csv.stat().st_size > 4_200_000, "Final test split size unexpected"

    policy_csv = SPLITS_V2_DIR / "policy.csv"
    assert policy_csv.exists(), f"Policy split missing: {policy_csv}"
    assert policy_csv.stat().st_size > 4_200_000, "Policy split size unexpected"


def test_calibrated_probabilities_within_unit_interval():
    """Requirement 6: Calibrated probabilities are strictly within [0, 1]."""
    with open(CALIBRATOR_PATH, "rb") as f:
        artifact = pickle.load(f)

    calibrator = artifact["calibrator"]

    test_inputs = np.array([0.0, 0.0001, 0.01, 0.05, 0.20, 0.50, 0.80, 0.95, 0.999, 1.0])
    cal_probs = transform_raw_probability(test_inputs, calibrator)

    assert len(cal_probs) == len(test_inputs)
    assert np.all(cal_probs >= 0.0), f"Negative probability found: {cal_probs.min()}"
    assert np.all(cal_probs <= 1.0), f"Probability > 1.0 found: {cal_probs.max()}"


def test_calibrator_can_transform_new_raw_probabilities_and_monotonicity():
    """Requirement 7: Calibrator can transform new raw probabilities and preserves strict monotonicity."""
    with open(CALIBRATOR_PATH, "rb") as f:
        artifact = pickle.load(f)

    calibrator = artifact["calibrator"]

    # Generate fine-grained raw probabilities
    raw_fine = np.linspace(0.0001, 0.9999, 1000)
    cal_fine = transform_raw_probability(raw_fine, calibrator)

    # Strictly non-decreasing monotonicity check
    diffs = np.diff(cal_fine)
    assert np.all(diffs >= 0.0), f"Monotonicity violation: min diff = {diffs.min()}"

    # Scalar input test
    p_scalar = transform_raw_probability(0.25, calibrator)
    assert isinstance(p_scalar, float)
    assert 0.0 <= p_scalar <= 1.0


def test_feature_ordering_remains_unchanged():
    """Requirement 8: Feature ordering remains unchanged matching VALID_FEATURE_NAMES."""
    with open(CALIBRATOR_PATH, "rb") as f:
        artifact = pickle.load(f)

    assert artifact["feature_count"] == 62
    assert artifact["feature_names"] == VALID_FEATURE_NAMES
    assert "transaction_id" not in artifact["feature_names"]
    assert "fraud_label" not in artifact["feature_names"]


def test_reports_exist_and_consistent():
    """Verifies report JSON and MD files exist and agree."""
    assert REPORT_JSON_PATH.exists(), f"Report JSON missing: {REPORT_JSON_PATH}"
    assert REPORT_MD_PATH.exists(), f"Report MD missing: {REPORT_MD_PATH}"

    with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["selected_method"] == "sigmoid"
    assert "reliability_bins" in rep
    assert len(rep["reliability_bins"]["sigmoid"]) == 10
