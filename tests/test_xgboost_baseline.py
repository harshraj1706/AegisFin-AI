"""
tests/test_xgboost_baseline.py

Automated verification tests for the AegisFin Phase 2 XGBoost baseline model artifact:
- Model artifact file existence and loadability
- Model metadata existence and structure
- Exactly 62 production features loaded
- transaction_id excluded from feature space
- fraud_label excluded from X
- Exact feature ordering matching VALID_FEATURE_NAMES
- Inference input dimension and output probability bounds
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from app.production_feature_definitions import VALID_FEATURE_NAMES

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "aegisfin_xgboost_baseline.pkl"
META_PATH = BASE_DIR / "models" / "aegisfin_xgboost_baseline_metadata.json"
REPORT_PATH = BASE_DIR / "reports" / "xgboost_baseline_report.json"


def test_model_artifact_exists_and_loads():
    assert MODEL_PATH.exists(), f"Model artifact missing: {MODEL_PATH}"
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    assert isinstance(artifact, dict), "Artifact payload must be a dictionary"
    assert "model" in artifact, "Missing 'model' in artifact"
    assert "feature_names" in artifact, "Missing 'feature_names' in artifact"
    assert "scale_pos_weight" in artifact, "Missing 'scale_pos_weight' in artifact"


def test_metadata_exists_and_valid():
    assert META_PATH.exists(), f"Metadata JSON missing: {META_PATH}"
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["model_name"] == "aegisfin_xgboost_baseline"
    assert meta["feature_count"] == 62
    assert "validation_metrics" in meta
    assert "top_15_features" in meta
    assert len(meta["top_15_features"]) == 15


def test_exactly_62_production_features():
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    features = artifact["feature_names"]
    assert len(features) == 62, f"Expected 62 features, got {len(features)}"
    assert features == VALID_FEATURE_NAMES, "Feature names do not match VALID_FEATURE_NAMES exactly"


def test_transaction_id_and_label_excluded_from_x():
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    features = artifact["feature_names"]
    assert "transaction_id" not in features, "transaction_id must NOT be in feature set"
    assert "fraud_label" not in features, "fraud_label must NOT be in feature set"
    for f in features:
        assert "scenario" not in f.lower(), f"Scenario metadata detected in feature set: {f}"


def test_feature_ordering_and_model_inference():
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]

    # Verify n_features_in_
    assert model.n_features_in_ == 62

    # Dummy inference test with 62 features
    dummy_x = np.zeros((1, 62), dtype=np.float32)
    probs = model.predict_proba(dummy_x)

    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 0] <= 1.0
    assert 0.0 <= probs[0, 1] <= 1.0
    assert np.isclose(probs[0, 0] + probs[0, 1], 1.0)


def test_report_artifact_exists():
    assert REPORT_PATH.exists(), f"Report JSON missing: {REPORT_PATH}"
    report_md = BASE_DIR / "reports" / "xgboost_baseline_report.md"
    assert report_md.exists(), f"Report MD missing: {report_md}"
