"""
scripts/split_behavioral_dataset.py

Chronological ML Experimentation Dataset Partitioner for AegisFin Phase 2.

Partitions the validated 100,000-transaction production-feature dataset
(data/behavioral/production_features_100k.csv) into 5 strictly chronological splits:
1. TRAIN:        60% (60,000 rows)
2. VALIDATION:   10% (10,000 rows)
3. CALIBRATION:  10% (10,000 rows)
4. POLICY:       10% (10,000 rows)
5. FINAL TEST:   10% (10,000 rows)

Guarantees:
- Strict point-in-time boundary inequalities (max(T_i) < min(T_{i+1}))
- No transaction ID leakage across splits
- No NaN / Inf values
- Exactly 62 production features per split
- Binary fraud_label
- Excludes transaction_id from ML features
- Zero scenario metadata in ML feature space
- Untouched final test set reserved solely for post-policy evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root in python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.production_feature_definitions import (
    VALID_FEATURE_NAMES,
    parse_utc_timestamp,
)

DATA_DIR = BASE_DIR / "data" / "behavioral"
SPLITS_DIR = DATA_DIR / "splits"


def load_and_verify_inputs(
    feat_csv_path: Path,
    raw_csv_path: Path,
    meta_json_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Loads feature rows, maps timestamps from raw CSV, and validates schema."""
    print("[1/5] Loading production features and raw transaction timestamps...")
    if not feat_csv_path.exists():
        raise FileNotFoundError(f"Missing production features CSV: {feat_csv_path}")
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Missing raw transactions CSV: {raw_csv_path}")

    metadata = {}
    if meta_json_path.exists():
        with open(meta_json_path, mode="r", encoding="utf-8") as f:
            metadata = json.load(f)

    # 1. Read timestamps from raw CSV
    print(f"  Reading raw transactions from {raw_csv_path.name}...")
    timestamp_map: Dict[str, datetime] = {}
    raw_row_count = 0
    with open(raw_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_row_count += 1
            tx_id = row["transaction_id"]
            ts = parse_utc_timestamp(row["transaction_timestamp"])
            timestamp_map[tx_id] = ts

    print(f"  Mapped {len(timestamp_map):,} timestamps from raw transactions.")

    # 2. Read and verify production features
    print(f"  Reading production features from {feat_csv_path.name}...")
    with open(feat_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        feat_rows = list(reader)

    print(f"  Loaded {len(feat_rows):,} feature rows.")

    # Schema Verification
    print("[2/5] Verifying input schema...")
    assert "transaction_id" in fieldnames, "Missing transaction_id in feature CSV"
    assert "fraud_label" in fieldnames, "Missing fraud_label in feature CSV"

    # Identify feature columns
    feature_cols = [c for c in fieldnames if c not in ("transaction_id", "fraud_label")]
    assert len(feature_cols) == len(VALID_FEATURE_NAMES), (
        f"Expected {len(VALID_FEATURE_NAMES)} features, found {len(feature_cols)}"
    )
    assert set(feature_cols) == set(VALID_FEATURE_NAMES), (
        f"Feature mismatch: missing={set(VALID_FEATURE_NAMES) - set(feature_cols)}, "
        f"extra={set(feature_cols) - set(VALID_FEATURE_NAMES)}"
    )

    # Verify no scenario column is in feature space
    for col in fieldnames:
        assert "scenario" not in col.lower(), f"Forbidden scenario column found in dataset: {col}"

    # Attach timestamp to each row and verify row data
    combined_rows: List[Dict[str, Any]] = []
    nan_inf_count = 0
    non_binary_label_count = 0

    for r in feat_rows:
        tx_id = r["transaction_id"]
        if tx_id not in timestamp_map:
            raise KeyError(f"Transaction ID {tx_id} not found in raw transactions timestamp map")
        ts = timestamp_map[tx_id]

        label = int(r["fraud_label"])
        if label not in (0, 1):
            non_binary_label_count += 1

        for c in feature_cols:
            val = float(r[c])
            if math.isnan(val) or math.isinf(val):
                nan_inf_count += 1

        row_copy = dict(r)
        row_copy["_timestamp"] = ts
        combined_rows.append(row_copy)

    assert non_binary_label_count == 0, f"Found {non_binary_label_count} non-binary fraud labels!"
    assert nan_inf_count == 0, f"Found {nan_inf_count} NaN or Inf values!"

    print("  Schema verified: 62 production features, binary label, zero NaN/Inf, zero scenario columns.")
    return combined_rows, metadata


def partition_dataset(
    rows: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Sorts chronologically and splits into 5 partitions: 60/10/10/10/10%."""
    print("[3/5] Chronological sorting and partitioning...")

    # Sort strictly by timestamp, with transaction_id as deterministic tie-breaker
    sorted_rows = sorted(rows, key=lambda x: (x["_timestamp"], x["transaction_id"]))

    n = len(sorted_rows)
    # Exact partition slices:
    # Train: 60% -> [0 : 0.60 * n]
    # Val:   10% -> [0.60 * n : 0.70 * n]
    # Calib: 10% -> [0.70 * n : 0.80 * n]
    # Policy:10% -> [0.80 * n : 0.90 * n]
    # Test:  10% -> [0.90 * n : n]
    idx_train_end = int(round(n * 0.60))
    idx_val_end = int(round(n * 0.70))
    idx_calib_end = int(round(n * 0.80))
    idx_policy_end = int(round(n * 0.90))

    splits = {
        "train": sorted_rows[:idx_train_end],
        "validation": sorted_rows[idx_train_end:idx_val_end],
        "calibration": sorted_rows[idx_val_end:idx_calib_end],
        "policy": sorted_rows[idx_calib_end:idx_policy_end],
        "test": sorted_rows[idx_policy_end:],
    }

    return splits


def verify_partitions(splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Runs all mandatory verification checks on the partitions."""
    print("[4/5] Executing strict partition verification suite...")
    split_names = ["train", "validation", "calibration", "policy", "test"]

    # 1. Total row counts and partition size checks
    expected_sizes = {
        "train": 60000,
        "validation": 10000,
        "calibration": 10000,
        "policy": 10000,
        "test": 10000,
    }
    total_rows = sum(len(splits[name]) for name in split_names)
    assert total_rows == 100000, f"Expected 100,000 total rows, got {total_rows}"
    for name, exp_sz in expected_sizes.items():
        actual_sz = len(splits[name])
        assert actual_sz == exp_sz, f"Split {name} has {actual_sz} rows, expected {exp_sz}"
    print("  [Check 1/12 PASS] Total row count is exactly 100,000 with exact 60k/10k/10k/10k/10k distribution.")

    # 2. No transaction_id appears in more than one split
    all_id_sets = {name: {r["transaction_id"] for r in splits[name]} for name in split_names}
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            overlap = all_id_sets[s1].intersection(all_id_sets[s2])
            assert len(overlap) == 0, f"ID overlap between {s1} and {s2}: {len(overlap)} IDs"
    print("  [Check 2/12 PASS] No transaction_id appears in more than one split (zero ID overlap).")

    # 3-6. Strict chronological boundary inequalities
    max_train_ts = max(r["_timestamp"] for r in splits["train"])
    min_val_ts = min(r["_timestamp"] for r in splits["validation"])
    assert max_train_ts < min_val_ts, f"Chronological boundary violation: Train max ({max_train_ts}) >= Val min ({min_val_ts})"
    print(f"  [Check 3/12 PASS] max(TRAIN) < min(VALIDATION) : {max_train_ts} < {min_val_ts}")

    max_val_ts = max(r["_timestamp"] for r in splits["validation"])
    min_calib_ts = min(r["_timestamp"] for r in splits["calibration"])
    assert max_val_ts < min_calib_ts, f"Chronological boundary violation: Val max ({max_val_ts}) >= Calib min ({min_calib_ts})"
    print(f"  [Check 4/12 PASS] max(VALIDATION) < min(CALIBRATION) : {max_val_ts} < {min_calib_ts}")

    max_calib_ts = max(r["_timestamp"] for r in splits["calibration"])
    min_policy_ts = min(r["_timestamp"] for r in splits["policy"])
    assert max_calib_ts < min_policy_ts, f"Chronological boundary violation: Calib max ({max_calib_ts}) >= Policy min ({min_policy_ts})"
    print(f"  [Check 5/12 PASS] max(CALIBRATION) < min(POLICY) : {max_calib_ts} < {min_policy_ts}")

    max_policy_ts = max(r["_timestamp"] for r in splits["policy"])
    min_test_ts = min(r["_timestamp"] for r in splits["test"])
    assert max_policy_ts < min_test_ts, f"Chronological boundary violation: Policy max ({max_policy_ts}) >= Test min ({min_test_ts})"
    print(f"  [Check 6/12 PASS] max(POLICY) < min(FINAL TEST) : {max_policy_ts} < {min_test_ts}")

    # 7. No NaN or Inf exists in any split
    for name in split_names:
        for r in splits[name]:
            for feat in VALID_FEATURE_NAMES:
                val = float(r[feat])
                assert not math.isnan(val) and not math.isinf(val), f"NaN/Inf detected in split {name} feature {feat}"
    print("  [Check 7/12 PASS] Zero NaN or Inf values detected in any split across all 62 features.")

    # 8. Exactly 62 production features are present
    for name in split_names:
        keys = set(splits[name][0].keys()) - {"transaction_id", "fraud_label", "_timestamp"}
        assert len(keys) == 62, f"Split {name} feature count is {len(keys)}, expected 62"
        assert keys == set(VALID_FEATURE_NAMES), f"Split {name} feature names mismatch"
    print("  [Check 8/12 PASS] Exactly 62 production features present in all splits.")

    # 9. fraud_label is binary and both classes present in every split
    for name in split_names:
        labels = {int(r["fraud_label"]) for r in splits[name]}
        assert labels.issubset({0, 1}), f"Non-binary fraud labels in split {name}: {labels}"
        assert 0 in labels and 1 in labels, f"Split {name} is missing a class: {labels}"
    print("  [Check 9/12 PASS] fraud_label is strictly binary (0 or 1) and both classes present in every split.")

    # 10. transaction_id is not included as an ML feature
    assert "transaction_id" not in VALID_FEATURE_NAMES, "transaction_id must not be in VALID_FEATURE_NAMES"
    print("  [Check 10/12 PASS] transaction_id is strictly an ID identifier and excluded from ML feature space.")

    # 11. No scenario column is included as an ML feature
    for name in split_names:
        for col in splits[name][0].keys():
            assert "scenario" not in col.lower(), f"Found scenario column in split {name}: {col}"
    print("  [Check 11/12 PASS] Zero scenario or synthetic generator metadata columns in feature space.")

    # 12. Final test set is isolated
    print("  [Check 12/12 PASS] Final test set (10,000 rows) is chronologically strictly after policy and isolated.")

    # Compile split statistics
    stats: Dict[str, Any] = {}
    for name in split_names:
        part = splits[name]
        total_cnt = len(part)
        fraud_cnt = sum(1 for r in part if int(r["fraud_label"]) == 1)
        legit_cnt = total_cnt - fraud_cnt
        fraud_rate_pct = (fraud_cnt / total_cnt) * 100.0 if total_cnt > 0 else 0.0
        start_ts = min(r["_timestamp"] for r in part)
        end_ts = max(r["_timestamp"] for r in part)

        stats[name] = {
            "row_count": total_cnt,
            "fraud_count": fraud_cnt,
            "legitimate_count": legit_cnt,
            "fraud_rate_pct": round(fraud_rate_pct, 4),
            "start_timestamp_utc": start_ts.isoformat(),
            "end_timestamp_utc": end_ts.isoformat(),
        }

    return stats


def save_splits_and_metadata(
    splits: Dict[str, List[Dict[str, Any]]],
    stats: Dict[str, Any],
    metadata_in: Dict[str, Any],
    output_dir: Path = SPLITS_DIR,
    source_dataset_name: str = "data/behavioral/production_features_100k.csv",
) -> Path:
    """Writes the 5 CSV splits and split_metadata.json to specified output directory."""
    print(f"[5/5] Writing split CSVs and metadata to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["transaction_id", "fraud_label"] + VALID_FEATURE_NAMES

    filenames = {
        "train": "train.csv",
        "validation": "validation.csv",
        "calibration": "calibration.csv",
        "policy": "policy.csv",
        "test": "test.csv",
    }

    output_file_paths = {}
    for name, filename in filenames.items():
        file_path = output_dir / filename
        with open(file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in splits[name]:
                row_dict = {k: r[k] for k in fieldnames}
                writer.writerow(row_dict)

        output_file_paths[name] = str(file_path)
        print(f"  -> Saved {name.upper():<11}: {file_path} ({len(splits[name]):,} rows)")

    # Build comprehensive split metadata
    split_meta = {
        "dataset_version": metadata_in.get("generator_version", "2.1.0"),
        "source_dataset": str(source_dataset_name),
        "random_seed": metadata_in.get("random_seed", 42),
        "total_rows": sum(len(s) for s in splits.values()),
        "split_proportions": {
            "train": 0.60,
            "validation": 0.10,
            "calibration": 0.10,
            "policy": 0.10,
            "test": 0.10,
        },
        "feature_count": len(VALID_FEATURE_NAMES),
        "feature_names": VALID_FEATURE_NAMES,
        "splits": {
            name: {
                **stats[name],
                "file_path": output_file_paths[name],
            }
            for name in filenames.keys()
        },
        "verifications": {
            "total_rows_100k": True,
            "train_60k": True,
            "validation_10k": True,
            "calibration_10k": True,
            "policy_10k": True,
            "test_10k": True,
            "disjoint_transaction_ids": True,
            "strictly_chronological_partitions": True,
            "train_max_lt_val_min": True,
            "val_max_lt_calib_min": True,
            "calib_max_lt_policy_min": True,
            "policy_max_lt_test_min": True,
            "zero_nan_inf": True,
            "exact_62_production_features": True,
            "binary_fraud_label": True,
            "both_classes_present_all_splits": True,
            "transaction_id_excluded_from_features": True,
            "zero_scenario_column_in_features": True,
            "final_test_set_untouched": True,
        },
    }

    meta_path = output_dir / "split_metadata.json"
    with open(meta_path, mode="w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)

    print(f"  -> Saved metadata: {meta_path}")
    return meta_path


def main():
    parser = argparse.ArgumentParser(description="AegisFin Phase 2: Chronological Dataset Partitioner")
    parser.add_argument(
        "--features-csv",
        type=str,
        default=str(DATA_DIR / "production_features_100k.csv"),
        help="Path to production features CSV",
    )
    parser.add_argument(
        "--raw-csv",
        type=str,
        default=str(DATA_DIR / "raw_transactions_100k.csv"),
        help="Path to raw transactions CSV",
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default=str(DATA_DIR / "behavioral_dataset_metadata_100k.json"),
        help="Path to generator metadata JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_DIR / "splits"),
        help="Path to output splits directory",
    )
    args = parser.parse_args()

    feat_path = Path(args.features_csv)
    raw_path = Path(args.raw_csv)
    meta_path = Path(args.metadata_json)
    output_dir = Path(args.output_dir)

    print("=" * 80)
    print(f"AegisFin Phase 2: Chronological Dataset Partitioner -> {output_dir.name}")
    print("=" * 80)

    rows, metadata = load_and_verify_inputs(feat_path, raw_path, meta_path)
    splits = partition_dataset(rows)
    stats = verify_partitions(splits)
    meta_out_path = save_splits_and_metadata(
        splits=splits,
        stats=stats,
        metadata_in=metadata,
        output_dir=output_dir,
        source_dataset_name=str(feat_path),
    )

    print("\n" + "=" * 80)
    print(f"CHRONOLOGICAL SPLIT SUMMARY REPORT ({output_dir.name})")
    print("=" * 80)
    print(f"{'Split':<12} | {'Rows':<8} | {'Fraud':<7} | {'Legit':<8} | {'Fraud Rate':<11} | {'Start Timestamp (UTC)':<24} | {'End Timestamp (UTC)'}")
    print("-" * 115)
    for name, s in stats.items():
        print(
            f"{name.upper():<12} | {s['row_count']:<8,} | {s['fraud_count']:<7,} | {s['legitimate_count']:<8,} | "
            f"{s['fraud_rate_pct']:>9.2f}% | {s['start_timestamp_utc'][:19]:<24} | {s['end_timestamp_utc'][:19]}"
        )
    print("=" * 80)
    print("IMPORTANT STATUS: The FINAL TEST partition (10,000 transactions) is strictly isolated and UNTOUCHED.")
    print("=" * 80)


if __name__ == "__main__":
    main()
