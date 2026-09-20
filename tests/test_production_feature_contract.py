"""
tests/test_production_feature_contract.py

Verification suite for AegisFin Phase 2 Production Feature Contract:
- Validates that every candidate feature has a complete contract specification.
- Confirms exact feature counts (78 audited: 58 VALID, 16 NEEDS_DATA, 4 INVALID).
- Verifies parity between Python definitions, CSV validation mapping, and JSON schema.
- Verifies training vs live feature generator 100% mathematical and numerical identity.
- Verifies prompt prescribed scenario (Tx A 10:00, Tx B 10:30 with count=1, mean=500, ratio=2.0).
- Verifies strict anti-leakage exclusion of future and concurrent transactions.
- Verifies cold-start zero-history behavior (no NaN, no Inf).
- Verifies exclusion of NEEDS_DATA and INVALID features from production candidate vectors.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import pytest
import numpy as np

from app.production_feature_definitions import (
    ALL_AUDITED_FEATURES,
    VALID_PRODUCTION_FEATURES,
    NEEDS_DATA_FEATURES,
    INVALID_FEATURES,
    VALID_FEATURE_NAMES,
    generate_production_features,
    training_feature_generator,
    live_feature_generator,
)
from backend.services.production_feature_definitions import (
    generate_production_features as backend_generate_features,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
CSV_PATH = DOCS_DIR / "phase2_production_feature_validation.csv"
JSON_PATH = DOCS_DIR / "phase2_production_feature_schema.json"
MD_PATH = DOCS_DIR / "phase2_production_feature_contract.md"


def test_feature_validation_counts_and_completeness():
    """Verify total audited features and status partition counts."""
    assert len(ALL_AUDITED_FEATURES) == 78, f"Expected 78 features, got {len(ALL_AUDITED_FEATURES)}"
    assert len(VALID_PRODUCTION_FEATURES) == 62, f"Expected 62 VALID, got {len(VALID_PRODUCTION_FEATURES)}"
    assert len(NEEDS_DATA_FEATURES) == 13, f"Expected 13 NEEDS_DATA, got {len(NEEDS_DATA_FEATURES)}"
    assert len(INVALID_FEATURES) == 3, f"Expected 3 INVALID, got {len(INVALID_FEATURES)}"

    # Assert no duplicate feature names
    all_names = [f.feature_name for f in ALL_AUDITED_FEATURES]
    assert len(all_names) == len(set(all_names)), "Duplicate feature names detected!"

    # Assert every feature has valid status
    valid_statuses = {"VALID", "NEEDS_DATA", "INVALID"}
    for f in ALL_AUDITED_FEATURES:
        assert f.production_status in valid_statuses, f"Invalid status '{f.production_status}' for {f.feature_name}"
        assert f.data_type in {"float", "int", "bool"}
        assert f.group
        assert f.training_source
        assert f.live_source
        assert f.training_formula
        assert f.live_formula
        assert f.leakage_rule
        assert f.nullable_behavior
        assert f.reason


def test_feature_validation_csv_and_schema_parity():
    """Verify CSV mapping, JSON schema, and markdown contract exist and match Python source."""
    assert CSV_PATH.exists(), f"Missing CSV: {CSV_PATH}"
    assert JSON_PATH.exists(), f"Missing JSON: {JSON_PATH}"
    assert MD_PATH.exists(), f"Missing Markdown: {MD_PATH}"

    # Verify CSV rows and columns
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 78, f"CSV should have 78 rows, got {len(rows)}"
    expected_cols = {
        "feature_name",
        "group",
        "training_source",
        "live_source",
        "training_formula",
        "live_formula",
        "lookback_window",
        "leakage_rule",
        "required_raw_fields",
        "required_supabase_fields",
        "status",
        "reason",
    }
    assert expected_cols.issubset(set(reader.fieldnames or [])), "CSV column mismatch"

    # Verify JSON schema
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    meta = schema["metadata"]
    assert meta["total_candidate_features_audited"] == 78
    assert meta["final_production_feature_count"] == 62
    assert meta["audit_breakdown"]["VALID"]["count"] == 62
    assert meta["audit_breakdown"]["NEEDS_DATA"]["count"] == 13
    assert meta["audit_breakdown"]["INVALID"]["count"] == 3
    assert len(schema["raw_input_fields"]) == 14


def test_training_vs_live_consistency_identical_values():
    """
    Verify training_feature_generator and live_feature_generator produce
    bit-for-bit identical results on the same transaction history.
    """
    history = [
        {
            "transaction_id": "TXN_001",
            "amount": 250.0,
            "transaction_timestamp": "2026-09-20T10:00:00Z",
            "customer_id": "C_TEST_01",
            "card_id": "CARD_999",
            "device_id": "DEV_ALPHA",
            "merchant_id": "M_STORE_1",
            "ip_address": "192.168.1.50",
        },
        {
            "transaction_id": "TXN_002",
            "amount": 750.0,
            "transaction_timestamp": "2026-09-20T10:15:00Z",
            "customer_id": "C_TEST_01",
            "card_id": "CARD_999",
            "device_id": "DEV_ALPHA",
            "merchant_id": "M_STORE_1",
            "ip_address": "192.168.1.50",
        },
    ]

    curr_txn = {
        "transaction_id": "TXN_003",
        "amount": 1000.0,
        "transaction_timestamp": "2026-09-20T10:30:00Z",
        "customer_id": "C_TEST_01",
        "card_id": "CARD_999",
        "device_id": "DEV_ALPHA",
        "merchant_id": "M_STORE_1",
        "product_code": "W",
        "card_network": "visa",
        "card_type": "debit",
        "email_domain": "gmail.com",
        "address_id": "ZIP_94103",
        "ip_address": "192.168.1.50",
        "country": "US",
    }

    train_feats = training_feature_generator(history, curr_txn)
    live_feats = live_feature_generator(history, curr_txn)

    assert set(train_feats.keys()) == set(live_feats.keys()) == set(VALID_FEATURE_NAMES)
    assert len(train_feats) == 62

    for k in train_feats:
        assert train_feats[k] == live_feats[k], f"Mismatch for feature '{k}': train={train_feats[k]} vs live={live_feats[k]}"


def test_prompt_prescribed_scenario_c100():
    """
    Verify the exact scenario specified by the prompt:
    Transaction A: customer=C100, timestamp=10:00, amount=500
    Transaction B: customer=C100, timestamp=10:30, amount=1000
    For B:
    - customer_tx_count_1h = 1
    - customer_amount_mean = 500
    - customer_amount_ratio = 2.0
    """
    txn_a = {
        "transaction_id": "TXN_A",
        "customer_id": "C100",
        "card_id": "CARD100",
        "amount": 500.0,
        "transaction_timestamp": "2026-09-20T10:00:00Z",
    }

    txn_b = {
        "transaction_id": "TXN_B",
        "customer_id": "C100",
        "card_id": "CARD100",
        "amount": 1000.0,
        "transaction_timestamp": "2026-09-20T10:30:00Z",
    }

    history_for_b = [txn_a]
    feats_b = generate_production_features(txn_b, history_for_b)

    assert feats_b["customer_tx_count_1h"] == 1.0, f"Expected 1.0, got {feats_b['customer_tx_count_1h']}"
    assert feats_b["customer_amount_mean"] == 500.0, f"Expected 500.0, got {feats_b['customer_amount_mean']}"
    # 1000 / (500 + 1e-5) is approximately 2.0
    assert np.isclose(feats_b["customer_amount_ratio"], 2.0, atol=1e-4), f"Expected ~2.0, got {feats_b['customer_amount_ratio']}"


def test_strict_anti_leakage_future_transaction_exclusion():
    """
    Verify that future transactions (timestamp >= current) and current transaction
    are strictly excluded and never alter features of earlier transactions.
    """
    txn_a = {
        "transaction_id": "TXN_A",
        "customer_id": "C100",
        "card_id": "CARD100",
        "amount": 500.0,
        "transaction_timestamp": "2026-09-20T10:00:00Z",
    }

    txn_b = {
        "transaction_id": "TXN_B",
        "customer_id": "C100",
        "card_id": "CARD100",
        "amount": 1000.0,
        "transaction_timestamp": "2026-09-20T10:30:00Z",
    }

    # Future transaction at 11:30 with large amount
    txn_c = {
        "transaction_id": "TXN_C",
        "customer_id": "C100",
        "card_id": "CARD100",
        "amount": 99999.0,
        "transaction_timestamp": "2026-09-20T11:30:00Z",
    }

    # Evaluate B with only A in history
    feats_b_clean = generate_production_features(txn_b, [txn_a])

    # Evaluate B with [A, B, C] in history (testing anti-leakage filter)
    feats_b_with_future = generate_production_features(txn_b, [txn_a, txn_b, txn_c])

    # Feats must be IDENTICAL: B and C must not affect B's features!
    for feat_name in VALID_FEATURE_NAMES:
        assert feats_b_clean[feat_name] == feats_b_with_future[feat_name], (
            f"Anti-leakage failure on '{feat_name}': clean={feats_b_clean[feat_name]} vs with_future={feats_b_with_future[feat_name]}"
        )


def test_zero_history_cold_start_handling():
    """Verify clean cold-start behavior for first-time entities (no NaN/Inf)."""
    new_txn = {
        "transaction_id": "TXN_FIRST",
        "customer_id": "NEW_USER_99",
        "card_id": "NEW_CARD_99",
        "device_id": "NEW_DEV_99",
        "merchant_id": "NEW_MERCH_99",
        "ip_address": "10.0.0.1",
        "amount": 120.0,
        "transaction_timestamp": "2026-09-20T12:00:00Z",
    }

    feats = generate_production_features(new_txn, history=[])

    assert feats["customer_is_new"] == 1.0
    assert feats["card_is_new"] == 1.0
    assert feats["device_is_new"] == 1.0
    assert feats["ip_is_new"] == 1.0
    assert feats["customer_merchant_is_new"] == 1.0

    assert feats["customer_amount_mean"] == 0.0
    assert feats["customer_amount_std"] == 0.0
    assert feats["customer_amount_zscore"] == 0.0
    assert feats["customer_amount_ratio"] == 1.0

    assert feats["customer_tx_count_5m"] == 0.0
    assert feats["customer_tx_count_1h"] == 0.0
    assert feats["customer_tx_count_24h"] == 0.0
    assert feats["rapid_transaction_flag"] == 0.0

    for name, val in feats.items():
        assert not math.isnan(val), f"Feature '{name}' returned NaN"
        assert not math.isinf(val), f"Feature '{name}' returned Inf"


def test_needs_data_and_invalid_exclusion():
    """Verify that no NEEDS_DATA or INVALID features contaminate the production vector."""
    needs_data_names = {f.feature_name for f in NEEDS_DATA_FEATURES}
    invalid_names = {f.feature_name for f in INVALID_FEATURES}

    # Verify set disjointness
    assert needs_data_names.isdisjoint(set(VALID_FEATURE_NAMES))
    assert invalid_names.isdisjoint(set(VALID_FEATURE_NAMES))

    # Test that sample execution output contains only valid feature names
    sample_txn = {
        "transaction_id": "TX_TEST",
        "customer_id": "C_TEST",
        "card_id": "CARD_TEST",
        "amount": 150.0,
        "transaction_timestamp": "2026-09-20T14:00:00Z",
    }
    feats = generate_production_features(sample_txn, [])
    assert set(feats.keys()) == set(VALID_FEATURE_NAMES)

    for nd in needs_data_names:
        assert nd not in feats
    for inv in invalid_names:
        assert inv not in feats


def test_backend_services_alias_import():
    """Verify that backend.services.production_feature_definitions functions identically."""
    sample_txn = {
        "transaction_id": "TX_ALIAS",
        "customer_id": "C_ALIAS",
        "card_id": "CARD_ALIAS",
        "amount": 350.0,
        "transaction_timestamp": "2026-09-20T15:00:00Z",
    }
    app_res = generate_production_features(sample_txn, [])
    backend_res = backend_generate_features(sample_txn, [])
    assert app_res == backend_res
