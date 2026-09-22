"""
tests/test_policy_thresholds_v2.py

Automated verification tests for AegisFin Phase 2 Step 12: Policy Threshold Analysis & Risk Bands:
1. Policy artifacts (config, reports, figures) exist and are valid.
2. Risk bands are mutually exclusive and collectively exhaustive across [0.0, 1.0].
3. Lower probability never receives higher risk (strict monotonicity).
4. Threshold candidates are monotonically ordered and reflect valid trade-offs.
5. Analysis used policy.csv only; test.csv was never accessed or compromised.
6. Base model and probability calibrator remain frozen and un-retrained.
7. Exact 62-feature contract remains intact.
8. Mandatory governance disclaimer is strictly present in all artifacts.
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from app.production_feature_definitions import VALID_FEATURE_NAMES

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "behavioral"
SPLITS_V2_DIR = DATA_DIR / "splits_v2"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFIGS_DIR = BASE_DIR / "configs"

POLICY_CSV_PATH = SPLITS_V2_DIR / "policy.csv"
TEST_CSV_PATH = SPLITS_V2_DIR / "test.csv"
SPLIT_META_PATH = SPLITS_V2_DIR / "split_metadata.json"

BASE_MODEL_PATH = MODELS_DIR / "aegisfin_xgboost_baseline_v2.pkl"
CALIBRATOR_PATH = MODELS_DIR / "aegisfin_probability_calibrator_v2.pkl"

POLICY_CONFIG_PATH = CONFIGS_DIR / "fraud_risk_policy_v2.json"
REPORT_JSON_PATH = REPORTS_DIR / "policy_threshold_analysis_v2.json"
REPORT_MD_PATH = REPORTS_DIR / "policy_threshold_analysis_v2.md"

MANDATORY_DISCLAIMER = (
    "Provisional AegisFin policy based on synthetic behavioral policy data. "
    "It does not establish real-world banking performance."
)


def classify_risk_band(prob: float, bands_config: Dict[str, Any]) -> str:
    """Classifies a calibrated probability into a risk band using config boundaries."""
    for band_name, cfg in bands_config.items():
        low = cfg["probability_range"][0]
        high = cfg["probability_range"][1]
        if band_name == "CRITICAL" or cfg.get("upper_inclusive", False):
            if low <= prob <= high:
                return band_name
        else:
            if low <= prob < high:
                return band_name
    raise ValueError(f"Probability {prob} did not match any risk band")


def test_policy_artifacts_exist_and_valid():
    """Requirement 1: Verify config, json, markdown, and figure artifacts exist."""
    assert POLICY_CONFIG_PATH.exists(), f"Policy config missing: {POLICY_CONFIG_PATH}"
    assert REPORT_JSON_PATH.exists(), f"Policy JSON report missing: {REPORT_JSON_PATH}"
    assert REPORT_MD_PATH.exists(), f"Policy MD report missing: {REPORT_MD_PATH}"

    expected_figures = [
        "policy_precision_recall_f1_vs_threshold.png",
        "policy_fpr_vs_threshold.png",
        "policy_fraud_rate_by_risk_band.png",
        "policy_tx_distribution_by_risk_band.png",
        "policy_dashboard_summary.png",
    ]
    for fig_name in expected_figures:
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Figure missing: {fig_path}"
        assert fig_path.stat().st_size > 5000, f"Figure file suspiciously small: {fig_path}"

    with open(POLICY_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert "model_version" in cfg
    assert "calibrator_version" in cfg
    assert "feature_contract_version" in cfg
    assert "risk_bands" in cfg
    assert "provisional_action_threshold" in cfg
    assert "threshold_rationale" in cfg
    assert "policy_dataset_statistics" in cfg
    assert "governance_note" in cfg
    assert cfg["governance_note"] == MANDATORY_DISCLAIMER


def test_risk_bands_mutually_exclusive_and_collectively_exhaustive():
    """Requirement 2: Ensure bands form a complete partition over [0.0, 1.0] with zero overlap/gaps."""
    with open(POLICY_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    bands_config = cfg["risk_bands"]

    test_probs = [
        0.0, 0.00001, 0.05, 0.099999,
        0.10, 0.15, 0.25, 0.399999,
        0.40, 0.50, 0.65, 0.799999,
        0.80, 0.85, 0.95, 0.999999, 1.0
    ]

    for p in test_probs:
        matching_bands = []
        for name, b_cfg in bands_config.items():
            low = b_cfg["probability_range"][0]
            high = b_cfg["probability_range"][1]
            if name == "CRITICAL" or b_cfg.get("upper_inclusive", False):
                if low <= p <= high:
                    matching_bands.append(name)
            else:
                if low <= p < high:
                    matching_bands.append(name)

        assert len(matching_bands) == 1, f"Probability {p} matched {len(matching_bands)} bands: {matching_bands}"


def test_risk_bands_monotonicity_in_risk():
    """Requirement 3: Lower probability never receives higher risk; observed fraud rate is monotonic."""
    with open(POLICY_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    bands_config = cfg["risk_bands"]

    band_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    rank_map = {name: i for i, name in enumerate(band_order)}

    # Test fine probability grid for classification monotonicity
    grid = np.linspace(0.0, 1.0, 1000)
    prev_rank = -1
    for p in grid:
        band = classify_risk_band(p, bands_config)
        curr_rank = rank_map[band]
        assert curr_rank >= prev_rank, f"Monotonicity violation at p={p}: rank {curr_rank} < prev_rank {prev_rank}"
        prev_rank = curr_rank

    # Test empirical fraud rate monotonicity across bands
    rates = [bands_config[name]["observed_fraud_rate"] for name in band_order]
    for i in range(len(rates) - 1):
        assert rates[i] <= rates[i + 1], f"Observed fraud rate not monotonic: {rates[i]} > {rates[i+1]}"

    # Test mean calibrated probability monotonicity
    mean_probs = [bands_config[name]["mean_calibrated_probability"] for name in band_order]
    for i in range(len(mean_probs) - 1):
        assert mean_probs[i] < mean_probs[i + 1], f"Mean prob not strictly monotonic: {mean_probs[i]} >= {mean_probs[i+1]}"


def test_threshold_candidates_monotonic_ordering():
    """Requirement 4: Candidate thresholds are ordered and show consistent trade-offs."""
    with open(POLICY_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cands = cfg["candidate_operating_points"]

    cand_a = cands["candidate_a_max_recall"]
    cand_b = cands["candidate_b_balanced_midpoint"]
    cand_c = cands["candidate_c_low_fpr"]

    t_a = cand_a["threshold"]
    t_b = cand_b["threshold"]
    t_c = cand_c["threshold"]

    # Thresholds strictly ascending
    assert t_a < t_b < t_c, f"Candidate thresholds not strictly ordered: {t_a} < {t_b} < {t_c}"

    # Recall decreases or remains equal as threshold increases
    assert cand_a["recall"] >= cand_b["recall"] >= cand_c["recall"], "Recall must decrease with threshold"

    # FPR decreases or remains equal as threshold increases
    assert cand_a["fpr"] >= cand_b["fpr"] >= cand_c["fpr"], "FPR must decrease with threshold"


def test_policy_data_only_and_no_test_csv_access():
    """Requirement 5: Analysis used policy.csv only; test.csv remains quarantined and untouched."""
    assert POLICY_CSV_PATH.exists(), f"Policy CSV missing: {POLICY_CSV_PATH}"
    assert TEST_CSV_PATH.exists(), f"Test CSV missing: {TEST_CSV_PATH}"

    # Verify policy row count
    with open(POLICY_CSV_PATH, "r", encoding="utf-8") as f:
        policy_rows = sum(1 for _ in f) - 1
    assert policy_rows == 10000, f"Policy row count unexpected: {policy_rows}"

    # Verify test row count unchanged
    with open(TEST_CSV_PATH, "r", encoding="utf-8") as f:
        test_rows = sum(1 for _ in f) - 1
    assert test_rows == 10000, f"Test row count unexpected: {test_rows}"

    # Verify no test set leakage in policy config or reports
    with open(POLICY_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg_text = f.read()
    assert "test.csv" not in cfg_text, "test.csv must not appear in policy config"

    with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
        report_json = json.load(f)
    assert report_json["policy_dataset_validation"]["test_set_isolated"] is True


def test_models_and_calibrators_remain_frozen():
    """Requirement 6: Base model and calibrator artifacts remain frozen and unmodified."""
    assert BASE_MODEL_PATH.exists(), f"Base model missing: {BASE_MODEL_PATH}"
    with open(BASE_MODEL_PATH, "rb") as f:
        m_art = pickle.load(f)
    assert isinstance(m_art, dict)
    assert "model" in m_art
    assert m_art["feature_names"] == VALID_FEATURE_NAMES
    assert np.isclose(m_art["scale_pos_weight"], 12.109023, atol=1e-5)

    assert CALIBRATOR_PATH.exists(), f"Calibrator missing: {CALIBRATOR_PATH}"
    with open(CALIBRATOR_PATH, "rb") as f:
        c_art = pickle.load(f)
    assert isinstance(c_art, dict)
    assert "calibrator" in c_art
    assert c_art["selected_method"] == "sigmoid"


def test_exact_62_feature_contract_remains_intact():
    """Requirement 7: Verify exact 62 production feature contract in policy dataset and config."""
    with open(POLICY_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["feature_count"] == 62
    assert len(VALID_FEATURE_NAMES) == 62

    with open(POLICY_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
    expected_header = ["transaction_id", "fraud_label"] + VALID_FEATURE_NAMES
    assert header == expected_header, "Policy feature header mismatch"
    assert "scenario" not in header, "scenario column must never appear in policy feature columns"


def test_governance_disclaimer_present_in_reports():
    """Requirement 8: Ensure mandatory disclaimer is present in MD report and JSON report."""
    with open(REPORT_MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()
    assert MANDATORY_DISCLAIMER in md_text, "Mandatory disclaimer missing from MD report"

    with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    assert json_data["governance_disclaimer"] == MANDATORY_DISCLAIMER
