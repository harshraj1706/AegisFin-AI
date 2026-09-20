"""
tests/test_ieee_training_compatibility.py

Verification suite for AegisFin Phase 2 Training-Data Compatibility Audit:
- Verifies CSV artifact exists and contains all 62 VALID production features.
- Validates partition counts: exactly 22 IEEE_TRAINABLE, 18 IEEE_UNAVAILABLE, 22 IEEE_AMBIGUOUS.
- Validates entity-level constraints (Card, Customer, Merchant, IP, Device).
- Validates TransactionDT temporal feasibility mappings.
- Validates markdown report structure and strategic recommendation.
"""

from __future__ import annotations

import csv
from pathlib import Path
import pytest

from app.production_feature_definitions import (
    VALID_PRODUCTION_FEATURES,
    VALID_FEATURE_NAMES,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
CSV_PATH = DOCS_DIR / "phase2_ieee_training_compatibility.csv"
MD_PATH = DOCS_DIR / "phase2_ieee_training_compatibility.md"


def test_csv_artifact_exists_and_row_count():
    """Verify that phase2_ieee_training_compatibility.csv exists and has 62 features."""
    assert CSV_PATH.exists(), f"Compatibility CSV not found at {CSV_PATH}"
    
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    expected_headers = [
        "feature_name",
        "production_group",
        "ieee_source",
        "mapping_type",
        "trainable",
        "mapping_quality",
        "reason",
    ]
    assert reader.fieldnames == expected_headers, f"CSV headers mismatch: {reader.fieldnames}"
    assert len(rows) == 62, f"Expected exactly 62 rows in CSV, found {len(rows)}"
    
    # Assert every row corresponds to a VALID production feature
    csv_feature_names = [r["feature_name"] for r in rows]
    assert set(csv_feature_names) == set(VALID_FEATURE_NAMES), "Feature names in CSV do not match VALID_FEATURE_NAMES"
    assert len(csv_feature_names) == len(set(csv_feature_names)), "Duplicate feature name in CSV!"


def test_compatibility_partition_counts():
    """Verify exact partition counts: 22 TRAINABLE, 18 UNAVAILABLE, 22 AMBIGUOUS."""
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        
    trainable = [r for r in rows if r["trainable"] == "IEEE_TRAINABLE"]
    unavailable = [r for r in rows if r["trainable"] == "IEEE_UNAVAILABLE"]
    ambiguous = [r for r in rows if r["trainable"] == "IEEE_AMBIGUOUS"]
    
    assert len(trainable) == 22, f"Expected 22 IEEE_TRAINABLE, got {len(trainable)}: {[r['feature_name'] for r in trainable]}"
    assert len(unavailable) == 18, f"Expected 18 IEEE_UNAVAILABLE, got {len(unavailable)}: {[r['feature_name'] for r in unavailable]}"
    assert len(ambiguous) == 22, f"Expected 22 IEEE_AMBIGUOUS, got {len(ambiguous)}: {[r['feature_name'] for r in ambiguous]}"
    assert len(trainable) + len(unavailable) + len(ambiguous) == 62


def test_merchant_features_marked_unavailable():
    """Verify all 6 merchant features are strictly UNAVAILABLE due to missing merchant_id."""
    merchant_features = [
        "merchant_tx_count_1h",
        "merchant_tx_count_24h",
        "merchant_amount_mean",
        "merchant_amount_std",
        "customer_merchant_tx_count",
        "customer_merchant_is_new",
    ]
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        rows = {r["feature_name"]: r for r in csv.DictReader(f)}
        
    for feat in merchant_features:
        assert feat in rows, f"Missing merchant feature {feat}"
        assert rows[feat]["trainable"] == "IEEE_UNAVAILABLE"
        assert "merchant" in rows[feat]["reason"].lower()


def test_ip_features_marked_unavailable():
    """Verify all 7 IP features are strictly UNAVAILABLE due to missing ip_address."""
    ip_features = [
        "ip_tx_count_5m",
        "ip_tx_count_1h",
        "ip_tx_count_24h",
        "ip_unique_customers_24h",
        "ip_unique_cards_24h",
        "ip_unique_devices_24h",
        "ip_is_new",
    ]
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        rows = {r["feature_name"]: r for r in csv.DictReader(f)}
        
    for feat in ip_features:
        assert feat in rows, f"Missing IP feature {feat}"
        assert rows[feat]["trainable"] == "IEEE_UNAVAILABLE"
        assert "ip" in rows[feat]["reason"].lower()


def test_card_features_marked_trainable():
    """Verify all 7 Card behavioral features are marked TRAINABLE using card1."""
    card_features = [
        "card_tx_count_5m",
        "card_tx_count_1h",
        "card_tx_count_24h",
        "card_amount_mean",
        "card_amount_std",
        "card_amount_ratio",
        "card_is_new",
    ]
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        rows = {r["feature_name"]: r for r in csv.DictReader(f)}
        
    for feat in card_features:
        assert feat in rows, f"Missing Card feature {feat}"
        assert rows[feat]["trainable"] == "IEEE_TRAINABLE"
        assert "card1" in rows[feat]["ieee_source"]
        assert rows[feat]["mapping_quality"] == "High"


def test_customer_features_marked_ambiguous():
    """Verify Customer behavioral features are marked AMBIGUOUS due to synthetic pseudo-UID."""
    customer_features = [
        "customer_tx_count_5m",
        "customer_tx_count_1h",
        "customer_tx_count_24h",
        "customer_amount_mean",
        "customer_amount_std",
        "customer_amount_zscore",
        "customer_amount_ratio",
        "customer_is_new",
    ]
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        rows = {r["feature_name"]: r for r in csv.DictReader(f)}
        
    for feat in customer_features:
        assert feat in rows, f"Missing Customer feature {feat}"
        assert rows[feat]["trainable"] == "IEEE_AMBIGUOUS"
        assert "pseudo-uid" in rows[feat]["reason"].lower()
        assert rows[feat]["mapping_quality"] == "Low"


def test_time_and_amount_features_marked_trainable():
    """Verify Time/Calendar and Amount transform features are marked TRAINABLE."""
    time_features = [
        "hour",
        "weekday_index",
        "is_weekend",
        "is_night",
        "day_of_year",
        "time_since_midnight_sec",
    ]
    amount_features = [
        "amount_log",
        "amount_cents",
        "is_round_amount",
        "is_zero_cents",
    ]
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        rows = {r["feature_name"]: r for r in csv.DictReader(f)}
        
    for feat in time_features + amount_features:
        assert feat in rows, f"Missing feature {feat}"
        assert rows[feat]["trainable"] == "IEEE_TRAINABLE"
        assert rows[feat]["mapping_quality"] == "High"


def test_markdown_report_structure_and_completeness():
    """Verify phase2_ieee_training_compatibility.md contains all required sections and content."""
    assert MD_PATH.exists(), f"Markdown report not found at {MD_PATH}"
    content = MD_PATH.read_text(encoding="utf-8")
    
    assert "Training-Data Compatibility Audit Report" in content
    assert "Executive Summary & Tally" in content
    assert "Complete Audit of All 62 Features" in content
    assert "In-Depth Entity Mapping Analysis" in content
    assert "Temporal Feasibility Analysis" in content
    assert "Strategic Retraining Recommendation" in content
    assert "Option A: Train Production Model Strictly on the 22 IEEE-Trainable Features" in content
    assert "Option B: Switch to an Alternative Labeled Fraud Dataset" in content
    assert "Option C: Combined / Hybrid Strategy (RECOMMENDED)" in content
    
    # Confirm numerical metrics are present in report
    assert "22 (35.48%)" in content
    assert "18 (29.03%)" in content
