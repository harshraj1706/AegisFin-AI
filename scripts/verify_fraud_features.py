"""
verify_fraud_features.py

Demonstration and acceptance script for AegisFin Phase 2 Step 5 Fraud Feature Engine:
- Transaction A: customer=C100, amount=500
- Transaction B: customer=C100, amount=1000 (must see Transaction A in history)
- Transaction C: timestamp earlier than future Transaction D (D must NOT affect C)
- Feature schema verification: exactly 459 columns and exact ordering match.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.fraud_feature_service import get_fraud_feature_service, build_fraud_features


def run_acceptance_demonstration():
    print("=" * 70)
    print(" AegisFin Phase 2: Fraud Feature Engine Acceptance Demonstration")
    print("=" * 70)

    service = get_fraud_feature_service()

    # -------------------------------------------------------------------------
    # Scenario 1: Transaction A (Customer C100, $500 at 10:00:00Z)
    # -------------------------------------------------------------------------
    print("\n--- [Step 1] Processing Transaction A (Initial Transaction) ---")
    txn_a = {
        "customer_id": "C100",
        "amount": 500.0,
        "transaction_timestamp": "2026-09-20T10:00:00Z",
        "card_id": "9633",
        "address_id": "299.0",
        "product_code": "W",
        "card_network": "visa",
        "card_type": "debit",
    }
    history_for_a = []  # No prior transactions
    df_a = build_fraud_features(txn_a, history=history_for_a)

    print(f"Transaction A features generated: shape={df_a.shape}")
    print(f"  - customer_id: {txn_a['customer_id']}")
    print(f"  - amount: {txn_a['amount']}")
    print(f"  - uid_past_count: {df_a['uid_past_count'].iloc[0]} (Expected: 0.0)")
    print(f"  - uid_is_new: {df_a['uid_is_new'].iloc[0]} (Expected: 1.0)")
    print(f"  - uid_past_mean_amt: {df_a['uid_past_mean_amt'].iloc[0]} (Expected: 500.0)")
    print(f"  - uid_amount_ratio: {df_a['uid_amount_ratio'].iloc[0]} (Expected: 1.0)")

    assert df_a["uid_past_count"].iloc[0] == 0.0, "Transaction A must have past_count = 0"
    assert df_a["uid_is_new"].iloc[0] == 1.0, "Transaction A must have uid_is_new = 1.0"

    # -------------------------------------------------------------------------
    # Scenario 2: Transaction B (Customer C100, $1000 at 11:00:00Z)
    # Must see Transaction A because A happened earlier.
    # -------------------------------------------------------------------------
    print("\n--- [Step 2] Processing Transaction B (Follow-up Transaction) ---")
    txn_b = {
        "customer_id": "C100",
        "amount": 1000.0,
        "transaction_timestamp": "2026-09-20T11:00:00Z",
        "card_id": "9633",
        "address_id": "299.0",
        "product_code": "W",
        "card_network": "visa",
        "card_type": "debit",
    }
    # History now contains Transaction A
    history_for_b = [txn_a]
    df_b = build_fraud_features(txn_b, history=history_for_b)

    print(f"Transaction B features generated: shape={df_b.shape}")
    print(f"  - customer_id: {txn_b['customer_id']}")
    print(f"  - amount: {txn_b['amount']}")
    print(f"  - uid_past_count: {df_b['uid_past_count'].iloc[0]} (Expected: 1.0)")
    print(f"  - uid_is_new: {df_b['uid_is_new'].iloc[0]} (Expected: 0.0)")
    print(f"  - uid_past_mean_amt: {df_b['uid_past_mean_amt'].iloc[0]} (Expected: 500.0)")
    print(f"  - uid_amount_ratio: {df_b['uid_amount_ratio'].iloc[0]:.4f} (Expected: ~2.0)")

    assert df_b["uid_past_count"].iloc[0] == 1.0, "Transaction B must see Transaction A (past_count=1)"
    assert df_b["uid_is_new"].iloc[0] == 0.0, "Transaction B must recognize returning entity (is_new=0)"
    assert df_b["uid_past_mean_amt"].iloc[0] == 500.0, "Historical mean must strictly equal Transaction A amount"
    assert abs(df_b["uid_amount_ratio"].iloc[0] - 2.0) < 0.01, "Amount ratio 1000/500 should be ~2.0"

    # -------------------------------------------------------------------------
    # Scenario 3: Transaction C (12:00:00Z) vs Future Transaction D (13:00:00Z)
    # Anti-leakage: Transaction D must NOT affect Transaction C.
    # -------------------------------------------------------------------------
    print("\n--- [Step 3] Anti-Leakage Demonstration: Transaction C vs Future Transaction D ---")
    txn_c = {
        "customer_id": "C200",
        "amount": 350.0,
        "transaction_timestamp": "2026-09-20T12:00:00Z",
    }
    txn_d = {
        "customer_id": "C200",
        "amount": 99999.0,
        "transaction_timestamp": "2026-09-20T13:00:00Z",
    }

    # Pass history containing future transaction D
    mixed_history = [txn_d]
    df_c = build_fraud_features(txn_c, history=mixed_history)

    print(f"Transaction C timestamp: {txn_c['transaction_timestamp']}")
    print(f"Transaction D timestamp: {txn_d['transaction_timestamp']} (Future: $99,999.0)")
    print(f"Transaction C features:")
    print(f"  - uid_past_count: {df_c['uid_past_count'].iloc[0]} (Expected: 0.0 - D is ignored)")
    print(f"  - uid_is_new: {df_c['uid_is_new'].iloc[0]} (Expected: 1.0)")
    print(f"  - uid_past_mean_amt: {df_c['uid_past_mean_amt'].iloc[0]} (Expected: 350.0 - no influence from $99,999)")

    assert df_c["uid_past_count"].iloc[0] == 0.0, "Transaction C must NOT see future transaction D!"
    assert df_c["uid_is_new"].iloc[0] == 1.0, "Transaction C must be classified as new entity"
    assert df_c["uid_past_mean_amt"].iloc[0] == 350.0, "Transaction D amount leaked into Transaction C!"

    # -------------------------------------------------------------------------
    # Scenario 4: Exact Feature Count & Column Ordering Verification
    # -------------------------------------------------------------------------
    print("\n--- [Step 4] Validating Feature Schema & Ordering ---")
    expected_count = 459
    actual_count = df_b.shape[1]
    order_matches = list(df_b.columns) == service.feature_names
    nan_count = int(df_b.isna().sum().sum())
    inf_count = int(float(np.isinf(df_b.values).sum()))

    print(f"  - Feature count: {actual_count} / {expected_count}")
    print(f"  - Column ordering matches champion schema: {order_matches}")
    print(f"  - NaN values count: {nan_count}")
    print(f"  - Infinite values count: {inf_count}")

    assert actual_count == expected_count, f"Feature count mismatch: {actual_count} != {expected_count}"
    assert order_matches, "Feature ordering deviates from saved champion model schema!"
    assert nan_count == 0, "NaN values detected in feature matrix!"
    assert inf_count == 0, "Infinite values detected in feature matrix!"

    print("\n" + "=" * 70)
    print(" ACCEPTANCE TEST SUCCESSFUL: All constraints verified!")
    print("=" * 70)


if __name__ == "__main__":
    run_acceptance_demonstration()
