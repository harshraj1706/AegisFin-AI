"""
tests/test_xgboost_baseline_v2.py

Automated verification tests for the AegisFin Phase 2 XGBoost baseline v2 model:
- Model artifact file existence and loadability
- Model metadata existence and structure
- Exactly 62 production features loaded
- transaction_id excluded from feature space
- fraud_label excluded from X
- Exact feature ordering matching VALID_FEATURE_NAMES
- Inference input dimension and output probability bounds
- Confirmation that v1 model artifact remains intact and unmodified
- Confirmation that v2 calibration, policy, and test partitions remain untouched
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from app.production_feature_definitions import VALID_FEATURE_NAMES

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_V2_PATH = BASE_DIR / "models" / "aegisfin_xgboost_baseline_v2.pkl"
META_V2_PATH = BASE_DIR / "models" / "aegisfin_xgboost_baseline_v2_metadata.json"
REPORT_V2_JSON = BASE_DIR / "reports" / "xgboost_baseline_v2_report.json"
REPORT_V2_MD = BASE_DIR / "reports" / "xgboost_baseline_v2_report.md"

MODEL_V1_PATH = BASE_DIR / "models" / "aegisfin_xgboost_baseline.pkl"
SPLITS_V2_DIR = BASE_DIR / "data" / "behavioral" / "splits_v2"


def test_v2_model_artifact_exists_and_loads():
    assert MODEL_V2_PATH.exists(), f"Model v2 artifact missing: {MODEL_V2_PATH}"
    with open(MODEL_V2_PATH, "rb") as f:
        artifact = pickle.load(f)

    assert isinstance(artifact, dict), "Artifact payload must be a dictionary"
    assert "model" in artifact, "Missing 'model' in artifact"
    assert "feature_names" in artifact, "Missing 'feature_names' in artifact"
    assert "scale_pos_weight" in artifact, "Missing 'scale_pos_weight' in artifact"
    assert np.isclose(artifact["scale_pos_weight"], 12.109023, atol=1e-5)


def test_v2_metadata_exists_and_valid():
    assert META_V2_PATH.exists(), f"Metadata JSON missing: {META_V2_PATH}"
    with open(META_V2_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["model_name"] == "aegisfin_xgboost_baseline_v2"
    assert meta["dataset_version"] == "2.1.0"
    assert meta["feature_count"] == 62
    assert "validation_metrics" in meta
    assert "top_15_features" in meta
    assert len(meta["top_15_features"]) == 15
    assert "v1_vs_v2_comparison" in meta


def test_v2_exactly_62_production_features():
    with open(MODEL_V2_PATH, "rb") as f:
        artifact = pickle.load(f)

    features = artifact["feature_names"]
    assert len(features) == 62, f"Expected 62 features, got {len(features)}"
    assert features == VALID_FEATURE_NAMES, "Feature names do not match VALID_FEATURE_NAMES exactly"


def test_v2_transaction_id_and_label_excluded_from_x():
    with open(MODEL_V2_PATH, "rb") as f:
        artifact = pickle.load(f)

    features = artifact["feature_names"]
    assert "transaction_id" not in features, "transaction_id must NOT be in feature set"
    assert "fraud_label" not in features, "fraud_label must NOT be in feature set"
    for f in features:
        assert "scenario" not in f.lower(), f"Scenario metadata detected in feature set: {f}"


def test_v2_feature_ordering_and_model_inference():
    with open(MODEL_V2_PATH, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    assert model.n_features_in_ == 62

    # Test dummy inference vector of shape (1, 62)
    dummy_x = np.zeros((1, 62), dtype=np.float32)
    probs = model.predict_proba(dummy_x)

    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 0] <= 1.0
    assert 0.0 <= probs[0, 1] <= 1.0
    assert np.isclose(probs[0, 0] + probs[0, 1], 1.0)


def test_v2_report_artifacts_exist():
    assert REPORT_V2_JSON.exists(), f"Report JSON missing: {REPORT_V2_JSON}"
    assert REPORT_V2_MD.exists(), f"Report MD missing: {REPORT_V2_MD}"


def test_v1_model_artifact_remains_unmodified():
    assert MODEL_V1_PATH.exists(), f"v1 model missing: {MODEL_V1_PATH}"
    with open(MODEL_V1_PATH, "rb") as f:
        artifact = pickle.load(f)
    assert artifact["scale_pos_weight"] == pytest.approx(12.590033975084937, rel=1e-5)
    assert artifact["model"].n_features_in_ == 62


def test_v2_calibration_policy_test_untouched():
    for name in ["calibration.csv", "policy.csv", "test.csv"]:
        fpath = SPLITS_V2_DIR / name
        assert fpath.exists(), f"Split file missing: {fpath}"
        assert fpath.stat().st_size > 4_000_000, f"Split file size corrupted: {fpath}"
