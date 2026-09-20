"""
tests/test_phase2_feature_analysis.py

Automated test suite verifying the integrity and correctness of the
Phase 2 Live Feature Analysis deliverables:
- docs/phase2_feature_mapping.csv
- docs/phase2_live_feature_schema.json
- docs/phase2_live_feature_analysis.md
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "docs" / "phase2_feature_mapping.csv"
JSON_PATH = BASE_DIR / "docs" / "phase2_live_feature_schema.json"
DOC_PATH = BASE_DIR / "docs" / "phase2_live_feature_analysis.md"


def test_feature_mapping_csv_existence_and_structure():
    """Verify CSV exists, has exactly 459 rows, and strictly required headers."""
    assert CSV_PATH.exists(), f"Feature mapping CSV not found at {CSV_PATH}"
    df = pd.read_csv(CSV_PATH)

    # 459 features must be audited
    assert len(df) == 459, f"Expected 459 features, got {len(df)}"

    required_cols = [
        "feature_name",
        "importance",
        "importance_percent",
        "category",
        "source",
        "live_field",
        "derivation",
        "production_candidate",
        "reason",
    ]
    assert list(df.columns) == required_cols, f"Columns mismatch: {list(df.columns)} vs {required_cols}"

    # No empty or null values in critical columns
    assert not df["feature_name"].isna().any()
    assert not df["category"].isna().any()
    assert not df["source"].isna().any()
    assert not df["derivation"].isna().any()


def test_feature_classification_counts():
    """Verify classification into categories A, B, C, D matches the audit."""
    df = pd.read_csv(CSV_PATH)
    counts = df["category"].value_counts().to_dict()

    assert counts.get("A", 0) == 22, f"Expected 22 Category A features, got {counts.get('A')}"
    assert counts.get("B", 0) == 15, f"Expected 15 Category B features, got {counts.get('B')}"
    assert counts.get("C", 0) == 31, f"Expected 31 Category C features, got {counts.get('C')}"
    assert counts.get("D", 0) == 391, f"Expected 391 Category D features, got {counts.get('D')}"

    # Sum must be exactly 459
    assert sum(counts.values()) == 459


def test_v_features_strictly_category_d():
    """All 339 V-features must be Category D and excluded from production candidates."""
    df = pd.read_csv(CSV_PATH)
    v_rows = df[df["feature_name"].str.match(r"^V\d+$")]

    assert len(v_rows) == 339, f"Expected 339 V-features, got {len(v_rows)}"
    assert (v_rows["category"] == "D").all(), "All V-features must be classified as Category D"
    assert (v_rows["production_candidate"] == False).all(), "V-features must NOT be production candidates"


def test_feature_importance_distribution():
    """Verify tree importance sums to ~100% and Category D accounts for majority."""
    df = pd.read_csv(CSV_PATH)
    total_pct = df["importance_percent"].sum()
    assert pytest.approx(total_pct, rel=1e-3) == 100.0

    # Category D accounts for >90% of model tree importance
    cat_d_pct = df[df["category"] == "D"]["importance_percent"].sum()
    assert cat_d_pct > 90.0, f"Expected Category D importance > 90%, got {cat_d_pct:.2f}%"


def test_production_feature_schema_json():
    """Verify JSON schema validity, metadata, feature groups, and risk signals."""
    assert JSON_PATH.exists(), f"Schema JSON not found at {JSON_PATH}"

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # Validate top-level keys
    for key in ["title", "version", "metadata", "raw_input_fields", "production_feature_groups", "user_facing_risk_signals"]:
        assert key in schema, f"Missing top-level key '{key}' in schema JSON"

    # Validate 14 raw input fields
    raw_fields = schema["raw_input_fields"]
    assert len(raw_fields) == 14, f"Expected 14 raw input fields, got {len(raw_fields)}"
    raw_names = {rf["name"] for rf in raw_fields}
    expected_inputs = {
        "transaction_id", "amount", "transaction_timestamp", "customer_id",
        "card_id", "device_id", "merchant_id", "product_code", "card_network",
        "card_type", "email_domain", "address_id", "ip_address", "country"
    }
    assert raw_names == expected_inputs

    # Validate 10 production feature groups
    groups = schema["production_feature_groups"]
    assert len(groups) == 10, f"Expected 10 production feature groups, got {len(groups)}"

    # Count total production candidate features across all 10 groups
    total_prod_feats = sum(len(g["features"]) for g in groups.values())
    assert total_prod_feats == 66, f"Expected 66 recommended production features, got {total_prod_feats}"

    # Validate risk signals
    signals = schema["user_facing_risk_signals"]
    assert len(signals) >= 8, f"Expected at least 8 user-facing risk signals, got {len(signals)}"


def test_documentation_markdown_exists():
    """Verify documentation file exists and covers all key sections."""
    assert DOC_PATH.exists(), f"Documentation markdown not found at {DOC_PATH}"
    content = DOC_PATH.read_text(encoding="utf-8")

    # Verify coverage of core sections
    assert "Category A" in content
    assert "Category B" in content
    assert "Category C" in content
    assert "Category D" in content
    assert "V1" in content
    assert "66" in content
    assert "Strict Anti-Leakage" in content
