"""
tests/test_behavioral_splits_v2.py

Automated regression and integrity test suite for the chronological ML splits (v2):
- data/behavioral/splits_v2/train.csv (60%)
- data/behavioral/splits_v2/validation.csv (10%)
- data/behavioral/splits_v2/calibration.csv (10%)
- data/behavioral/splits_v2/policy.csv (10%)
- data/behavioral/splits_v2/test.csv (10%)

Verifies all 16 mandatory Step 8 criteria:
1. Exactly 100,000 total rows.
2. Exactly 60,000 train rows.
3. Exactly 10,000 validation rows.
4. Exactly 10,000 calibration rows.
5. Exactly 10,000 policy rows.
6. Exactly 10,000 final test rows.
7. No transaction_id overlap.
8. Strict chronological ordering between every split.
9. Zero NaN.
10. Zero Inf.
11. Exactly 62 production features in every split.
12. Binary fraud_label.
13. Both classes present in every split.
14. No scenario columns.
15. No modification to v1 split files.
16. Final test remains isolated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from app.production_feature_definitions import (
    VALID_FEATURE_NAMES,
    parse_utc_timestamp,
)

BASE_DIR = Path(__file__).resolve().parent.parent
SPLITS_V2_DIR = BASE_DIR / "data" / "behavioral" / "splits_v2"
SPLITS_V1_DIR = BASE_DIR / "data" / "behavioral" / "splits"
RAW_100K_V2_PATH = BASE_DIR / "data" / "behavioral" / "raw_transactions_100k_v2.csv"

# Known SHA256 checksums of the original v1 splits
EXPECTED_V1_SHA256 = {
    "calibration.csv": "009ab87a9d38e8e6676fce95f22cc565af5f853a1ccea5b4af29565de1f50bf0",
    "policy.csv": "35a2001f55078affd91856eda29e215a4a84387573ea6afcacda3858aa28ec74",
    "split_metadata.json": "77ff37b98bd95c0541f1bac9e56e2694484de00757a8f0de3f251a1fab49d7dd",
    "test.csv": "a4e370ffbfd77443ddd86a2334f42a2ad714b73ab0ecc4e847898c6e9019022e",
    "train.csv": "61cca17da3a85f636c12e0a4e00df7f7ddf029397a9d591d8dddab8bec988773",
    "validation.csv": "da3bf6f73b928c7e43f15cf3d9aca7ce0f47b11289f7d1777781a5f0f3dcde0f",
}


@pytest.fixture(scope="module")
def split_v2_data():
    """Loads all 5 split files from splits_v2 and maps timestamps from raw CSV."""
    split_names = ["train", "validation", "calibration", "policy", "test"]
    data = {}
    for name in split_names:
        csv_path = SPLITS_V2_DIR / f"{name}.csv"
        assert csv_path.exists(), f"Missing v2 split file: {csv_path}"
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data[name] = {
                "fieldnames": reader.fieldnames,
                "rows": list(reader),
            }

    # Load raw timestamps
    timestamp_map = {}
    with open(RAW_100K_V2_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp_map[row["transaction_id"]] = parse_utc_timestamp(row["transaction_timestamp"])

    # Attach timestamps
    for name in split_names:
        for r in data[name]["rows"]:
            r["_ts"] = timestamp_map[r["transaction_id"]]

    meta_path = SPLITS_V2_DIR / "split_metadata.json"
    assert meta_path.exists(), f"Missing split metadata: {meta_path}"
    with open(meta_path, mode="r", encoding="utf-8") as f:
        metadata = json.load(f)

    return data, metadata


def test_check_1_to_6_row_counts_and_proportions(split_v2_data):
    """Checks 1-6: Exactly 100k total rows, 60k train, 10k val, 10k calib, 10k policy, 10k test."""
    data, meta = split_v2_data
    assert len(data["train"]["rows"]) == 60000
    assert len(data["validation"]["rows"]) == 10000
    assert len(data["calibration"]["rows"]) == 10000
    assert len(data["policy"]["rows"]) == 10000
    assert len(data["test"]["rows"]) == 10000
    assert meta["total_rows"] == 100000


def test_check_7_disjoint_transaction_ids(split_v2_data):
    """Check 7: No transaction_id appears in more than one split."""
    data, _ = split_v2_data
    split_names = ["train", "validation", "calibration", "policy", "test"]
    seen_ids = set()
    for name in split_names:
        ids = {r["transaction_id"] for r in data[name]["rows"]}
        assert seen_ids.isdisjoint(ids), f"Overlapping transaction IDs detected in {name}!"
        seen_ids.update(ids)
    assert len(seen_ids) == 100000


def test_check_8_strict_chronological_ordering(split_v2_data):
    """Check 8: Strict chronological ordering between every split."""
    data, _ = split_v2_data
    split_names = ["train", "validation", "calibration", "policy", "test"]

    for i in range(len(split_names) - 1):
        curr_name = split_names[i]
        next_name = split_names[i + 1]

        max_curr_ts = max(r["_ts"] for r in data[curr_name]["rows"])
        min_next_ts = min(r["_ts"] for r in data[next_name]["rows"])

        assert max_curr_ts < min_next_ts, (
            f"Chronological ordering violated: max({curr_name})={max_curr_ts} >= min({next_name})={min_next_ts}"
        )


def test_check_9_and_10_no_nan_or_inf(split_v2_data):
    """Checks 9 & 10: Zero NaN and Zero Inf in all features across all splits."""
    data, _ = split_v2_data
    for name, split in data.items():
        for r in split["rows"]:
            for feat in VALID_FEATURE_NAMES:
                val = float(r[feat])
                assert not math.isnan(val), f"NaN in {name} feature {feat}"
                assert not math.isinf(val), f"Inf in {name} feature {feat}"


def test_check_11_exact_62_features_schema(split_v2_data):
    """Check 11: Exactly 62 production features in every split."""
    data, _ = split_v2_data
    for name, split in data.items():
        fieldnames = split["fieldnames"]
        assert fieldnames[0] == "transaction_id"
        assert fieldnames[1] == "fraud_label"
        feat_cols = fieldnames[2:]
        assert len(feat_cols) == 62
        assert feat_cols == VALID_FEATURE_NAMES


def test_check_12_and_13_binary_labels_and_both_classes(split_v2_data):
    """Checks 12 & 13: Binary fraud_label, and both classes present in every split."""
    data, _ = split_v2_data
    for name, split in data.items():
        labels = {int(r["fraud_label"]) for r in split["rows"]}
        assert labels.issubset({0, 1})
        assert 0 in labels and 1 in labels


def test_check_14_exclusion_of_id_and_scenarios(split_v2_data):
    """Check 14: No scenario columns, transaction_id excluded from feature names."""
    data, _ = split_v2_data
    for name, split in data.items():
        fieldnames = split["fieldnames"]
        assert "transaction_id" not in VALID_FEATURE_NAMES
        for col in fieldnames:
            assert "scenario" not in col.lower()


def test_check_15_v1_splits_unmodified():
    """Check 15: v1 split files are completely untouched (SHA256 checksums match)."""
    assert SPLITS_V1_DIR.exists(), f"Missing v1 splits directory: {SPLITS_V1_DIR}"
    for filename, expected_hash in EXPECTED_V1_SHA256.items():
        file_path = SPLITS_V1_DIR / filename
        assert file_path.exists(), f"Missing v1 file: {file_path}"
        with open(file_path, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        assert actual_hash == expected_hash, f"v1 file modified: {filename} ({actual_hash} != {expected_hash})"


def test_check_16_final_test_isolated(split_v2_data):
    """Check 16: Final test partition is strictly after policy and isolated."""
    data, _ = split_v2_data
    max_policy_ts = max(r["_ts"] for r in data["policy"]["rows"])
    min_test_ts = min(r["_ts"] for r in data["test"]["rows"])
    assert max_policy_ts < min_test_ts
    assert len(data["test"]["rows"]) == 10000
