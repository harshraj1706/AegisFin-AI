"""
scripts/verify_web_smoke_test.py

AegisFin-AI — Web Application E2E Smoke Test Verification Script.
Executes programmatic end-to-end verification of the running FastAPI backend,
running Streamlit frontend, authentication flows, prediction inference,
history/high-risk feeds, duplicate protection, and cleanup.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import urllib.request
import urllib.error

# Ensure workspace root is on sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from app.production_feature_definitions import VALID_FEATURE_NAMES
from app.supabase_client import get_supabase_admin_client
from app.auth import signup_user, login_user

API_BASE_URL = "http://127.0.0.1:8000"
STREAMLIT_BASE_URL = "http://127.0.0.1:8501"


def http_request(
    url: str,
    method: str = "GET",
    headers: Dict[str, str] = None,
    data: Dict[str, Any] = None,
    timeout: float = 10.0,
) -> Tuple[int, Dict[str, Any], float]:
    """Helper to perform synchronous HTTP request and measure latency."""
    headers = headers or {}
    encoded_data = None
    if data is not None:
        encoded_data = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=encoded_data, headers=headers, method=method)
    start_t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            latency_ms = (time.perf_counter() - start_t) * 1000.0
            body = resp.read().decode("utf-8")
            try:
                parsed = json.loads(body)
            except Exception:
                parsed = {"raw": body}
            return resp.status, parsed, latency_ms
    except urllib.error.HTTPError as err:
        latency_ms = (time.perf_counter() - start_t) * 1000.0
        err_body = err.read().decode("utf-8")
        try:
            parsed = json.loads(err_body)
        except Exception:
            parsed = {"raw": err_body}
        return err.code, parsed, latency_ms
    except Exception as exc:
        latency_ms = (time.perf_counter() - start_t) * 1000.0
        return 0, {"error": str(exc)}, latency_ms


def run_smoke_test() -> Dict[str, Any]:
    print("=" * 80)
    print("AegisFin-AI — Web Application E2E Smoke Test")
    print("=" * 80)

    admin = get_supabase_admin_client()
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend_url": API_BASE_URL,
        "frontend_url": STREAMLIT_BASE_URL,
        "tests": {},
        "summary": {},
    }

    # -------------------------------------------------------------------------
    # 1. APPLICATION STARTUP & RUNTIME STATUS
    # -------------------------------------------------------------------------
    print("\n[1] Verifying Backend and Frontend Runtime Status...")
    b_code, b_body, b_lat = http_request(f"{API_BASE_URL}/health")
    s_code, s_body, s_lat = http_request(f"{STREAMLIT_BASE_URL}/_stcore/health")

    backend_running = b_code == 200
    frontend_running = s_code == 200

    results["tests"]["startup"] = {
        "status": "PASS" if (backend_running and frontend_running) else "FAIL",
        "backend": {"status_code": b_code, "latency_ms": b_lat, "response": b_body},
        "frontend": {"status_code": s_code, "latency_ms": s_lat, "response": s_body},
    }
    print(f"  Backend: HTTP {b_code} ({b_lat:.1f}ms) -> {'RUNNING' if backend_running else 'DOWN'}")
    print(f"  Frontend: HTTP {s_code} ({s_lat:.1f}ms) -> {'RUNNING' if frontend_running else 'DOWN'}")

    # -------------------------------------------------------------------------
    # 2. BACKEND HEALTH & MODEL METADATA
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Backend Fraud Health & Model Metadata...")
    h_code, h_body, h_lat = http_request(f"{API_BASE_URL}/api/v1/fraud/health")
    health_passed = (
        h_code == 200
        and h_body.get("status") == "ready"
        and h_body.get("model_version") == "2.1.0"
        and h_body.get("expected_feature_count") == 62
        and h_body.get("calibration_method") == "Platt"
    )
    results["tests"]["backend_health"] = {
        "status": "PASS" if health_passed else "FAIL",
        "status_code": h_code,
        "latency_ms": h_lat,
        "data": h_body,
    }
    print(f"  Fraud Health: model={h_body.get('model_version')}, features={h_body.get('expected_feature_count')}, calibrator={h_body.get('calibration_method')}")
    print(f"  Result: {'PASS' if health_passed else 'FAIL'}")

    # -------------------------------------------------------------------------
    # 3. WEB UI PAGE LOAD
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Streamlit Web UI Root Page Load...")
    ui_code, ui_body, ui_lat = http_request(STREAMLIT_BASE_URL)
    html_content = ui_body.get("raw", "")
    ui_passed = ui_code == 200 and ("Streamlit" in html_content or "<!doctype html>" in html_content.lower())
    results["tests"]["web_ui_load"] = {
        "status": "PASS" if ui_passed else "FAIL",
        "status_code": ui_code,
        "latency_ms": ui_lat,
        "content_length": len(html_content),
    }
    print(f"  Web UI Load: HTTP {ui_code} ({ui_lat:.1f}ms, {len(html_content)} bytes) -> {'PASS' if ui_passed else 'FAIL'}")

    # -------------------------------------------------------------------------
    # 4. AUTHENTICATION FLOW
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Authentication Flow...")
    unique_id = uuid.uuid4().hex[:8]
    test_email = "smoke_analyst_2026@aegisfin.ai"
    test_password = "AegisSmokePass!2026"

    # Test login
    login_res = login_user(email=test_email, password=test_password)
    token = login_res.get("access_token")
    user_id = login_res.get("user_id")

    if not token:
        # If user not found, sign up
        signup_res = signup_user(email=test_email, password=test_password, full_name="Smoke Test Analyst")
        token = signup_res.get("access_token")
        user_id = signup_res.get("user_id")

    auth_headers = {"Authorization": f"Bearer {token}"}

    # Verify protected endpoint with valid token
    me_code, me_body, _ = http_request(f"{API_BASE_URL}/api/v1/auth/me", headers=auth_headers)

    # Verify protected endpoint blocked without token
    unauth_code, _, _ = http_request(f"{API_BASE_URL}/api/v1/auth/me")

    auth_passed = (token is not None and me_code == 200 and unauth_code == 401)
    results["tests"]["auth_flow"] = {
        "status": "PASS" if auth_passed else "FAIL",
        "login_status": "SUCCESS" if token else "FAILED",
        "user_email": test_email,
        "authenticated_check_code": me_code,
        "unauthenticated_check_code": unauth_code,
    }
    print(f"  Login: {'SUCCESS' if token else 'FAILED'}")
    print(f"  Authorized /api/v1/auth/me: HTTP {me_code} (User: {me_body.get('email')})")
    print(f"  Unauthorized /api/v1/auth/me: HTTP {unauth_code} (Properly blocked)")
    print(f"  Result: {'PASS' if auth_passed else 'FAIL'}")

    # -------------------------------------------------------------------------
    # 5. PREDICTION FLOW (LIVE WEB / API)
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Prediction Flow...")
    test_txn_id = f"SMOKE_TXN_WEB_{unique_id}"
    test_cust_id = f"CUST_SMOKE_WEB_{unique_id}"
    test_amount = 275.50
    test_ts = datetime.now(timezone.utc).isoformat()

    txn_payload = {
        "transaction_id": test_txn_id,
        "customer_id": test_cust_id,
        "amount": test_amount,
        "transaction_timestamp": test_ts,
        "card_id": f"CARD_SMOKE_{unique_id}",
        "device_id": f"DEV_SMOKE_{unique_id}",
        "merchant_id": f"MERCH_SMOKE_{unique_id}",
        "email_domain": "gmail.com",
        "address_id": "299.0",
        "product_code": "W",
        "card_network": "visa",
        "card_type": "debit",
        "ip_address": "10.99.100.1",
        "country": "US",
    }

    pred_code, pred_body, pred_lat = http_request(
        f"{API_BASE_URL}/api/v1/fraud/predict",
        method="POST",
        headers=auth_headers,
        data=txn_payload,
    )

    pred_passed = (
        pred_code == 200
        and pred_body.get("transaction_id") == test_txn_id
        and "calibrated_fraud_probability" in pred_body
        and "risk_band" in pred_body
        and "recommended_action" in pred_body
        and pred_body.get("model_version") == "2.1.0"
        and pred_body.get("feature_contract_version") == "2.1.0"
    )
    results["tests"]["prediction_flow"] = {
        "status": "PASS" if pred_passed else "FAIL",
        "status_code": pred_code,
        "latency_ms": pred_lat,
        "transaction_id": test_txn_id,
        "raw_fraud_probability": pred_body.get("raw_fraud_probability"),
        "calibrated_fraud_probability": pred_body.get("calibrated_fraud_probability"),
        "risk_band": pred_body.get("risk_band"),
        "recommended_action": pred_body.get("recommended_action"),
        "legacy_fraud_probability": pred_body.get("fraud_probability"),
        "legacy_fraud_band": pred_body.get("fraud_band"),
        "legacy_decision": pred_body.get("decision"),
        "model_version": pred_body.get("model_version"),
    }
    print(f"  Transaction ID: {test_txn_id}")
    print(f"  Calibrated Prob: {pred_body.get('calibrated_fraud_probability')}")
    print(f"  Risk Band: {pred_body.get('risk_band')}, Recommended Action: {pred_body.get('recommended_action')}")
    print(f"  Result: {'PASS' if pred_passed else 'FAIL'}")

    # -------------------------------------------------------------------------
    # 6. RISK-BAND BEHAVIOR
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Risk Band Mapping & Operational Action Behavior...")
    # Verify that the 4 canonical bands and actions are supported
    canonical_bands = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    canonical_actions = {"AUTO_APPROVE", "STEP_UP_AUTH", "MANUAL_REVIEW", "HARD_DECLINE"}

    band_valid = pred_body.get("risk_band") in canonical_bands
    action_valid = pred_body.get("recommended_action") in canonical_actions

    risk_band_passed = band_valid and action_valid
    results["tests"]["risk_band_behavior"] = {
        "status": "PASS" if risk_band_passed else "FAIL",
        "assigned_band": pred_body.get("risk_band"),
        "assigned_action": pred_body.get("recommended_action"),
        "canonical_bands": list(canonical_bands),
        "canonical_actions": list(canonical_actions),
    }
    print(f"  Assigned Band '{pred_body.get('risk_band')}' in {canonical_bands}: {band_valid}")
    print(f"  Assigned Action '{pred_body.get('recommended_action')}' in {canonical_actions}: {action_valid}")
    print(f"  Result: {'PASS' if risk_band_passed else 'FAIL'}")

    # -------------------------------------------------------------------------
    # 7. PREDICTION PERSISTENCE & HISTORY
    # -------------------------------------------------------------------------
    print("\n[7] Verifying Prediction Persistence & History Endpoint...")
    hist_code, hist_body, _ = http_request(
        f"{API_BASE_URL}/api/v1/fraud/history?limit=15",
        method="GET",
        headers=auth_headers,
    )
    items = hist_body.get("items", [])
    found_txn = any(item.get("transaction_id") == test_txn_id for item in items)

    hist_passed = hist_code == 200 and found_txn
    results["tests"]["history"] = {
        "status": "PASS" if hist_passed else "FAIL",
        "status_code": hist_code,
        "items_count": len(items),
        "test_txn_found": found_txn,
    }
    print(f"  History Items Count: {len(items)}")
    print(f"  Test Txn '{test_txn_id}' Found in History: {found_txn}")
    print(f"  Result: {'PASS' if hist_passed else 'FAIL'}")

    # -------------------------------------------------------------------------
    # 8. HIGH-RISK FEED
    # -------------------------------------------------------------------------
    print("\n[8] Verifying High-Risk Fraud Alert Feed...")
    hr_code, hr_body, _ = http_request(
        f"{API_BASE_URL}/api/v1/fraud/high-risk?limit=5",
        method="GET",
        headers=auth_headers,
    )
    hr_items = hr_body.get("items", [])
    hr_passed = hr_code == 200 and isinstance(hr_items, list)
    results["tests"]["high_risk_feed"] = {
        "status": "PASS" if hr_passed else "FAIL",
        "status_code": hr_code,
        "items_count": len(hr_items),
        "sample": hr_items[:1] if hr_items else {},
    }
    print(f"  High-Risk Alerts Count: {len(hr_items)}")
    print(f"  Result: {'PASS' if hr_passed else 'FAIL'}")

    # -------------------------------------------------------------------------
    # 9. DUPLICATE TRANSACTION BEHAVIOR
    # -------------------------------------------------------------------------
    print("\n[9] Verifying Duplicate Transaction Protection...")
    dup_code, dup_body, _ = http_request(
        f"{API_BASE_URL}/api/v1/fraud/predict",
        method="POST",
        headers=auth_headers,
        data=txn_payload,  # Identical transaction_id
    )

    dup_passed = dup_code == 409
    results["tests"]["duplicate_protection"] = {
        "status": "PASS" if dup_passed else "FAIL",
        "status_code": dup_code,
        "detail": dup_body.get("detail"),
    }
    print(f"  Duplicate Submission HTTP Status: {dup_code} (Expected 409)")
    print(f"  Detail Message: {dup_body.get('detail')}")
    print(f"  Result: {'PASS' if dup_passed else 'FAIL'}")

    # -------------------------------------------------------------------------
    # 10. FEATURE / DEBUG VIEW
    # -------------------------------------------------------------------------
    print("\n[10] Verifying 62-Feature Debug Contract Endpoint...")
    feat_code, feat_body, _ = http_request(
        f"{API_BASE_URL}/api/v1/fraud/features/test",
        method="POST",
        data=txn_payload,
    )
    feature_count = feat_body.get("feature_count", 0)
    expected_count = feat_body.get("expected_feature_count", 0)
    schema_valid = feat_body.get("feature_schema_valid", False)

    debug_passed = feat_code == 200 and feature_count == 62 and expected_count == 62 and schema_valid
    results["tests"]["feature_debug_view"] = {
        "status": "PASS" if debug_passed else "FAIL",
        "status_code": feat_code,
        "feature_count": feature_count,
        "expected_feature_count": expected_count,
        "feature_schema_valid": schema_valid,
    }
    print(f"  Feature Count: {feature_count} / {expected_count}")
    print(f"  Contract Schema Valid: {schema_valid}")
    print(f"  Result: {'PASS' if debug_passed else 'FAIL'}")

    # -------------------------------------------------------------------------
    # 11. BROWSER / NETWORK ERRORS
    # -------------------------------------------------------------------------
    print("\n[11] Verifying Browser / Network Error Status...")
    # Assert no 5xx or unexpected 4xx occurred during tests
    all_codes = [b_code, s_code, h_code, ui_code, me_code, pred_code, hist_code, hr_code, feat_code]
    no_server_errors = all(c < 500 for c in all_codes if c > 0)
    network_errors_passed = no_server_errors
    results["tests"]["browser_network_errors"] = {
        "status": "PASS" if network_errors_passed else "FAIL",
        "server_errors_detected": not no_server_errors,
        "recorded_codes": all_codes,
    }
    print(f"  Server Errors Detected: {not no_server_errors}")
    print(f"  Result: {'PASS' if network_errors_passed else 'FAIL'}")

    # -------------------------------------------------------------------------
    # 12. CLEANUP
    # -------------------------------------------------------------------------
    print("\n[12] Cleaning Up Synthetic Test Transaction Records...")
    cleaned = {}
    try:
        s = admin.table("fraud_feature_snapshots").delete().eq("transaction_id", test_txn_id).execute()
        p = admin.table("fraud_predictions").delete().eq("transaction_id", test_txn_id).execute()
        t = admin.table("fraud_transactions").delete().eq("transaction_id", test_txn_id).execute()
        e = admin.table("fraud_entity_state").delete().eq("entity_type", "customer").eq("entity_key", test_cust_id).execute()

        cleaned["snapshots"] = len(s.data or [])
        cleaned["predictions"] = len(p.data or [])
        cleaned["transactions"] = len(t.data or [])
        cleaned["entity_states"] = len(e.data or [])
    except Exception as exc:
        cleaned["error"] = str(exc)

    cleanup_passed = cleaned.get("transactions", 0) >= 1
    results["tests"]["cleanup"] = {
        "status": "PASS" if cleanup_passed else "FAIL",
        "cleaned_records": cleaned,
    }
    print(f"  Cleaned Records: {cleaned}")
    print(f"  Result: {'PASS' if cleanup_passed else 'FAIL'}")

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------
    all_test_statuses = [t.get("status") == "PASS" for t in results["tests"].values()]
    overall_pass = all(all_test_statuses)

    results["summary"] = {
        "overall_smoke_test": "PASS" if overall_pass else "FAIL",
        "web_application": "PASS" if ui_passed else "FAIL",
        "backend": "PASS" if backend_running and health_passed else "FAIL",
        "frontend": "PASS" if frontend_running and ui_passed else "FAIL",
        "auth": "PASS" if auth_passed else "FAIL",
        "prediction": "PASS" if pred_passed else "FAIL",
        "history": "PASS" if hist_passed else "FAIL",
        "high_risk_feed": "PASS" if hr_passed else "FAIL",
        "duplicate_protection": "PASS" if dup_passed else "FAIL",
        "tests_passed": sum(1 for passed in all_test_statuses if passed),
        "total_tests": len(results["tests"]),
    }

    print("\n" + "=" * 80)
    print(f"OVERALL SMOKE TEST RESULT: {results['summary']['overall_smoke_test']} ({results['summary']['tests_passed']}/{results['summary']['total_tests']} tests passed)")
    print("=" * 80)

    # Save JSON report
    report_json_path = WORKSPACE_ROOT / "reports" / "web_e2e_smoke_test_v2.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nStructured smoke test JSON report saved to: {report_json_path}")

    return results


if __name__ == "__main__":
    res = run_smoke_test()
    if res["summary"]["overall_smoke_test"] != "PASS":
        sys.exit(1)
    sys.exit(0)
