"""
tests/test_behavioral_splits.py

Automated regression and integrity test suite for the chronological ML splits:
- train.csv (60%)
- validation.csv (10%)
- calibration.csv (10%)
- policy.csv (10%)
- test.csv (10%)
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from app.production_feature_definitions import (
    VALID_FEATURE_NAMES,
    parse_utc_timestamp,
)

BASE_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = BASE_DIR / "data" / "behavioral" / "splits"
RAW_100K_PATH = BASE_DIR / "data" / "behavioral" / "raw_transactions_100k.csv"


@pytest.fixture(scope="module")
def split_data():
    """Loads all 5 split files and maps timestamps."""
    split_names = ["train", "validation", "calibration", "policy", "test"]
    data = {}
    for name in split_names:
        csv_path = SPLITS_DIR / f"{name}.csv"
        assert csv_path.exists(), f"Missing split file: {csv_path}"
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data[name] = {
                "fieldnames": reader.fieldnames,
                "rows": list(reader),
            }

    # Load raw timestamps
    timestamp_map = {}
    with open(RAW_100K_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp_map[row["transaction_id"]] = parse_utc_timestamp(row["transaction_timestamp"])

    # Attach timestamps
    for name in split_names:
        for r in data[name]["rows"]:
            r["_ts"] = timestamp_map[r["transaction_id"]]

    meta_path = SPLITS_DIR / "split_metadata.json"
    assert meta_path.exists(), f"Missing split metadata: {meta_path}"
    with open(meta_path, mode="r", encoding="utf-8") as f:
        metadata = json.load(f)

    return data, metadata


def test_split_row_counts_and_proportions(split_data):
    data, meta = split_data
    assert len(data["train"]["rows"]) == 60000
    assert len(data["validation"]["rows"]) == 10000
    assert len(data["calibration"]["rows"]) == 10000
    assert len(data["policy"]["rows"]) == 10000
    assert len(data["test"]["rows"]) == 10000
    assert meta["total_rows"] == 100000


def test_disjoint_transaction_ids(split_data):
    data, _ = split_data
    split_names = ["train", "validation", "calibration", "policy", "test"]
    seen_ids = set()
    for name in split_names:
        ids = {r["transaction_id"] for r in data[name]["rows"]}
        assert seen_ids.isdisjoint(ids), f"Overlapping transaction IDs detected in {name}!"
        seen_ids.update(ids)
    assert len(seen_ids) == 100000


def test_strict_chronological_ordering(split_data):
    data, _ = split_data
    split_names = ["train", "validation", "calibration", "policy", "test"]

    for i in range(len(split_names) - 1):
        curr_name = split_names[i]
        next_name = split_names[i + 1]

        max_curr_ts = max(r["_ts"] for r in data[curr_name]["rows"])
        min_next_ts = min(r["_ts"] for r in data[next_name]["rows"])

        assert max_curr_ts < min_next_ts, (
            f"Chronological ordering violated: max({curr_name})={max_curr_ts} >= min({next_name})={min_next_ts}"
        )


def test_no_nan_or_inf(split_data):
    data, _ = split_data
    for name, split in data.items():
        for r in split["rows"]:
            for feat in VALID_FEATURE_NAMES:
                val = float(r[feat])
                assert not math.isnan(val), f"NaN in {name} feature {feat}"
                assert not math.isinf(val), f"Inf in {name} feature {feat}"


def test_exact_62_features_schema(split_data):
    data, _ = split_data
    for name, split in data.items():
        fieldnames = split["fieldnames"]
        assert fieldnames[0] == "transaction_id"
        assert fieldnames[1] == "fraud_label"
        feat_cols = fieldnames[2:]
        assert len(feat_cols) == 62
        assert feat_cols == VALID_FEATURE_NAMES


def test_binary_fraud_labels(split_data):
    data, _ = split_data
    for name, split in data.items():
        labels = {int(r["fraud_label"]) for r in split["rows"]}
        assert labels.issubset({0, 1})
        # Each split should contain both legitimate and fraud cases
        assert 0 in labels and 1 in labels


def test_exclusion_of_id_and_scenarios(split_data):
    data, _ = split_data
    for name, split in data.items():
        fieldnames = split["fieldnames"]
        assert "transaction_id" not in VALID_FEATURE_NAMES
        for col in fieldnames:
            assert "scenario" not in col.lower()
