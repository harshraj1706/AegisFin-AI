from __future__ import annotations

import re
from pathlib import Path
import pytest

MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "supabase"
    / "migrations"
    / "20260921000001_phase2_fraud_schema_update.sql"
)


def test_migration_file_exists():
    """Verify that the migration SQL file exists in the canonical migrations directory."""
    assert MIGRATION_PATH.exists(), f"Migration file missing at {MIGRATION_PATH}"
    assert MIGRATION_PATH.stat().st_size > 1000, "Migration file is unexpectedly small"


def test_migration_strictly_non_destructive():
    """Verify that the migration contains zero DROP TABLE, DROP COLUMN, or DELETE statements."""
    sql = MIGRATION_PATH.read_text(encoding="utf-8")

    # Disallow DROP TABLE
    assert not re.findall(r"\bDROP\s+TABLE\b", sql, re.IGNORECASE), "DROP TABLE is forbidden"

    # Disallow DROP COLUMN
    assert not re.findall(r"\bDROP\s+COLUMN\b", sql, re.IGNORECASE), "DROP COLUMN is forbidden"

    # Disallow TRUNCATE
    assert not re.findall(r"\bTRUNCATE\b", sql, re.IGNORECASE), "TRUNCATE is forbidden"

    # Disallow DELETE FROM
    for line in sql.splitlines():
        clean_line = re.sub(r"--.*$", "", line).strip()
        assert not re.search(r"^\s*DELETE\s+FROM\b", clean_line, re.IGNORECASE), (
            f"DELETE FROM forbidden: {clean_line}"
        )


def test_migration_idempotent_clauses():
    """Verify that all CREATE TABLE, ADD COLUMN, and CREATE INDEX statements use IF NOT EXISTS."""
    sql = MIGRATION_PATH.read_text(encoding="utf-8")

    # Every CREATE TABLE must use IF NOT EXISTS
    for m in re.finditer(r"CREATE\s+TABLE\s+(?!IF\s+NOT\s+EXISTS)(\S+)", sql, re.IGNORECASE):
        pytest.fail(f"CREATE TABLE without IF NOT EXISTS: {m.group(0)}")

    # Every ADD COLUMN must use IF NOT EXISTS
    for m in re.finditer(r"ADD\s+COLUMN\s+(?!IF\s+NOT\s+EXISTS)(\S+)", sql, re.IGNORECASE):
        pytest.fail(f"ADD COLUMN without IF NOT EXISTS: {m.group(0)}")

    # Every CREATE INDEX must use IF NOT EXISTS
    for m in re.finditer(r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?!IF\s+NOT\s+EXISTS)(\S+)", sql, re.IGNORECASE):
        pytest.fail(f"CREATE INDEX without IF NOT EXISTS: {m.group(0)}")


def test_required_phase2_columns_present():
    """Verify that all 7 canonical Phase 2 additive columns are added to fraud_predictions."""
    sql = MIGRATION_PATH.read_text(encoding="utf-8")
    required_cols = [
        "raw_fraud_probability",
        "calibrated_fraud_probability",
        "risk_band",
        "recommended_action",
        "feature_contract_version",
        "calibrator_type",
        "scored_at",
    ]
    for col in required_cols:
        pattern = rf"\bADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+{col}\b"
        assert re.search(pattern, sql, re.IGNORECASE), f"Column {col} missing from migration"


def test_domain_check_constraints_present():
    """Verify domain constraints for risk bands and operational actions."""
    sql = MIGRATION_PATH.read_text(encoding="utf-8")

    # Risk bands: LOW, MEDIUM, HIGH, CRITICAL
    for band in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        assert f"'{band}'" in sql, f"Risk band '{band}' not declared in SQL"

    # Operational actions: AUTO_APPROVE, STEP_UP_AUTH, MANUAL_REVIEW, HARD_DECLINE
    for action in ["AUTO_APPROVE", "STEP_UP_AUTH", "MANUAL_REVIEW", "HARD_DECLINE"]:
        assert f"'{action}'" in sql, f"Action '{action}' not declared in SQL"


def test_feature_snapshots_table_present():
    """Verify dedicated public.fraud_feature_snapshots table definition and columns."""
    sql = MIGRATION_PATH.read_text(encoding="utf-8")
    assert "public.fraud_feature_snapshots" in sql

    for col in ["transaction_id", "feature_contract_version", "model_version", "features", "created_at"]:
        assert col in sql, f"Snapshot column {col} missing"


def test_trigger_bidirectional_mapping_logic():
    """
    Verify the bidirectional synchronization trigger function exists and maps correctly.
    Note: The mapping from canonical (4 tiers) to legacy (3 tiers) is an INTENTIONALLY
    LOSSY compatibility projection (MEDIUM & HIGH -> REVIEW, CRITICAL -> HIGH).
    """
    sql = MIGRATION_PATH.read_text(encoding="utf-8")
    assert "fn_sync_fraud_predictions_compatibility" in sql
    assert "trg_sync_fraud_predictions_compatibility" in sql

    # Check key mappings in trigger
    assert "'AUTO_APPROVE'" in sql and "'ALLOW'" in sql
    assert "'STEP_UP_AUTH'" in sql and "'MANUAL_REVIEW'" in sql
    assert "'HARD_DECLINE'" in sql and "'BLOCK'" in sql
    assert "'CRITICAL'" in sql and "'HIGH'" in sql
    assert "'MEDIUM'" in sql and "'REVIEW'" in sql
