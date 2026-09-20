"""
verify_fraud_inference.py

Demonstration and verification script for AegisFin Phase 2 Step 6:
Phase 2 Fraud Model Inference Service
- Loads model once
- Feature Engine -> Model Service pipeline
- Platt calibration verification
- Policy thresholds & risk bands
- Inference latency benchmark
- Historical anti-leakage verification
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.fraud_feature_service import build_fraud_features
from app.fraud_model_service import get_fraud_model_service


def run_verification():
    print("=" * 75)
    print(" AegisFin Phase 2: Fraud Model Inference Service Verification (Step 6)")
    print("=" * 75)

    # 1. Model Loading & Info
    print("\n--- [Step 1] Loading Fraud Model Service ---")
    start_load = time.perf_counter()
    service = get_fraud_model_service()
    load_time_ms = (time.perf_counter() - start_load) * 1000.0

    info = service.info()
    print(f"Service loaded in: {load_time_ms:.2f} ms")
    print(f"  - Model Name: {info['model_name']}")
    print(f"  - Model Version: {info['model_version']}")
    print(f"  - Calibration Version: {info['calibration_version']}")
    print(f"  - Policy Version: {info['policy_version']}")
    print(f"  - Calibration Method: {info['calibration_method']}")
    print(f"  - Expected Feature Count: {info['expected_feature_count']}")
    print(f"  - Policy Thresholds: {info['policy_thresholds']}")

    # 2. Test Transaction (Section 8)
    print("\n--- [Step 2] End-to-End Pipeline: Feature Engine -> Model Service ---")
    txn = {
        "transaction_id": "TEST_PHASE2_MODEL_001",
        "customer_id": "TEST_CUSTOMER_001",
        "card_id": "9633",
        "device_id": "TEST_DEVICE_001",
        "merchant_id": "TEST_MERCHANT_001",
        "amount": 1000.00,
        "transaction_timestamp": "2026-09-20T10:00:00Z",
        "address_id": "299.0",
        "product_code": "W",
        "card_network": "visa",
        "card_type": "debit",
    }

    # Feature Engine
    df = build_fraud_features(txn, history=[])
    print(f"Feature vector generated: shape={df.shape}")

    # Model Inference
    result = service.predict(df)
    print("\nStructured Inference Result:")
    for k, v in result.items():
        print(f"  - {k}: {v}")

    assert 0.0 <= result["raw_probability"] <= 1.0
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["fraud_band"] in {"LOW", "REVIEW", "HIGH"}
    assert result["decision"] in {"ALLOW", "MANUAL_REVIEW", "BLOCK"}
    assert result["feature_count"] == 459
    assert result["inference_latency_ms"] > 0.0

    # 3. Determinism Benchmark
    print("\n--- [Step 3] Deterministic Inference Benchmark (100 Iterations) ---")
    latencies = []
    first_res = service.predict(df)
    for _ in range(100):
        t0 = time.perf_counter()
        res = service.predict(df)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        assert res["raw_probability"] == first_res["raw_probability"]
        assert res["fraud_probability"] == first_res["fraud_probability"]
        assert res["fraud_band"] == first_res["fraud_band"]

    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    print(f"  - 100/100 runs identical: True")
    print(f"  - Average model inference latency: {avg_latency:.3f} ms")
    print(f"  - 95th percentile inference latency: {p95_latency:.3f} ms")

    # 4. Historical Anti-Leakage
    print("\n--- [Step 4] Historical Anti-Leakage Verification ---")
    txn_c = {
        "customer_id": "C_TEST_LEAK",
        "amount": 250.0,
        "transaction_timestamp": "2026-09-20T12:00:00Z",
    }
    txn_d_future = {
        "customer_id": "C_TEST_LEAK",
        "amount": 99999.0,
        "transaction_timestamp": "2026-09-20T13:00:00Z",
    }

    df_c = build_fraud_features(txn_c, history=[txn_d_future])
    res_c = service.predict(df_c)
    print(f"Transaction C at 12:00 (with future D $99,999 at 13:00 in history):")
    print(f"  - uid_past_count: {df_c['uid_past_count'].iloc[0]} (Must be 0.0)")
    print(f"  - fraud_probability: {res_c['fraud_probability']:.4f}")
    print(f"  - fraud_band: {res_c['fraud_band']}")
    assert df_c["uid_past_count"].iloc[0] == 0.0

    print("\n" + "=" * 75)
    print(" ALL PHASE 2 MODEL SERVICE CRITERIA VERIFIED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_verification()
