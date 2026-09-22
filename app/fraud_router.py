from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from .auth import get_current_user
from .fraud_feature_service import (
    fetch_historical_transactions,
    parse_timestamp,
    update_entity_state,
)
from .fraud_model_service import (
    FraudModelArtifactError,
    FraudModelCalibrationError,
    FraudModelInferenceError,
    FraudModelSchemaError,
    FraudPolicyError,
    get_fraud_model_service,
)
from .production_feature_definitions import (
    VALID_FEATURE_NAMES,
    generate_production_features,
)
from .schemas import FraudPredictionRequest, FraudPredictionResponse
from .supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/fraud", tags=["Phase 2 Fraud Detection"])


# -----------------------------------------------------------------------------
# Health / Model Status Endpoint
# -----------------------------------------------------------------------------
@router.get("/health")
def fraud_health():
    """Returns Phase 2 Fraud Model status and policy metadata."""
    try:
        model_service = get_fraud_model_service()
        return {
            "status": "ok",
            "fraud_model_loaded": True,
            **model_service.info(),
        }
    except Exception as exc:
        logger.error(f"Fraud model health check failed: {exc}")
        return {
            "status": "degraded",
            "fraud_model_loaded": False,
            "error": str(exc),
        }


# -----------------------------------------------------------------------------
# Development Feature Test Endpoint
# -----------------------------------------------------------------------------
@router.post("/features/test")
def fraud_features_test(transaction: Dict[str, Any]):
    """
    Development endpoint verifying Phase 2 62-feature production contract.
    Returns feature count (62), contract validity, and production feature values.
    """
    try:
        features = generate_production_features(current_transaction=transaction, history=[])
        return {
            "feature_count": len(features),
            "expected_feature_count": 62,
            "feature_schema_valid": bool(
                len(features) == 62 and list(features.keys()) == VALID_FEATURE_NAMES
            ),
            "sample_features": {
                k: features[k] for k in list(features.keys())[:15]
            },
            "features": features,
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {exc}") from exc


# -----------------------------------------------------------------------------
# Final Phase 2 Fraud Prediction Endpoint (Cutover to 62-Feature Pipeline)
# -----------------------------------------------------------------------------
@router.post(
    "/predict",
    response_model=FraudPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate live transaction for fraud risk",
    description=(
        "Executes end-to-end Phase 2 fraud risk evaluation: queries historical point-in-time state, "
        "constructs the frozen 62 production features without leakage, performs calibrated XGBoost v2 "
        "inference, applies the 4-tier risk policy, persists the transaction, prediction, and 62-feature "
        "audit snapshot in Supabase, and updates entity history."
    ),
)
def predict_fraud(
    request: FraudPredictionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Full Phase 2 Production Fraud Prediction Lifecycle:
    1. Authenticate user via Supabase Bearer token
    2. Validate transaction request
    3. Check transaction_id for duplicate submission (409 Conflict if duplicate)
    4. Query historical state strictly prior to current transaction timestamp (anti-leakage)
    5. Generate 62-dimensional production feature vector (Single Source of Truth)
    6. Validate model feature schema (count == 62, ordering == VALID_FEATURE_NAMES)
    7. Execute frozen XGBoost v2 base model inference
    8. Apply frozen Platt sigmoid probability calibration
    9. Apply frozen 4-tier policy thresholds (LOW, MEDIUM, HIGH, CRITICAL)
    10. Map operational decision (AUTO_APPROVE, STEP_UP_AUTH, MANUAL_REVIEW, HARD_DECLINE)
        plus legacy compatibility projections (ALLOW, MANUAL_REVIEW, BLOCK)
    11. Persist transaction in public.fraud_transactions
    12. Persist prediction in public.fraud_predictions (canonical + legacy compatibility)
    13. Persist feature snapshot in public.fraud_feature_snapshots (62 features JSONB)
    14. Update public.fraud_entity_state aggregates
    15. Return structured response (canonical + legacy compatibility fields)
    """
    user_id = current_user.get("user_id", "unknown_user")
    txn_id = request.transaction_id

    # 1. Normalize timestamp
    parsed_dt = parse_timestamp(request.transaction_timestamp)
    ts_iso = parsed_dt.isoformat()

    # 2. Check for duplicate transaction
    admin = get_supabase_admin_client()
    try:
        dup_check = (
            admin.table("fraud_transactions")
            .select("transaction_id")
            .eq("transaction_id", txn_id)
            .execute()
        )
        if dup_check.data and len(dup_check.data) > 0:
            logger.warning(
                f"Duplicate transaction rejected: transaction_id='{txn_id}', user_id='{user_id}'"
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Transaction '{txn_id}' has already been processed.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed during duplicate check for transaction '{txn_id}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database verification failure during fraud evaluation.",
        ) from exc

    # 3. Assemble transaction dict for feature engineering
    txn_dict: Dict[str, Any] = {
        "transaction_id": txn_id,
        "amount": float(request.amount),
        "transaction_timestamp": ts_iso,
        "customer_id": request.customer_id,
        "card_id": request.card_id,
        "device_id": request.device_id,
        "merchant_id": request.merchant_id,
        "email_domain": request.email_domain,
        "address_id": request.address_id,
        "product_code": request.product_code or "W",
        "card_network": request.card_network,
        "card_type": request.card_type,
        "ip_address": request.ip_address,
        "country": request.country,
    }

    # 4. Query Prior History (Strict Anti-Leakage: strictly timestamp < current timestamp)
    # The current transaction is NOT yet persisted in database, ensuring strict anti-leakage.
    try:
        history = fetch_historical_transactions(txn_dict, client=admin)
    except Exception as exc:
        logger.warning(f"Failed to retrieve history for '{txn_id}': {exc}. Defaulting to empty history.")
        history = []

    # 5. Generate 62 Production Features using frozen contract generator
    try:
        features_dict = generate_production_features(current_transaction=txn_dict, history=history)
    except Exception as exc:
        logger.error(f"Production feature generation failed for transaction '{txn_id}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Fraud feature engineering failed: {exc}",
        ) from exc

    # 6 & 7. Frozen Model Inference, Platt Sigmoid Calibration & Risk Policy Assignment
    try:
        model_service = get_fraud_model_service()
        pred_result = model_service.predict(features_dict)
    except FraudModelSchemaError as exc:
        logger.error(f"Feature schema error for transaction '{txn_id}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except (FraudModelCalibrationError, FraudModelInferenceError, FraudPolicyError) as exc:
        logger.error(f"Model prediction error for transaction '{txn_id}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fraud model inference failed: {exc}",
        ) from exc
    except Exception as exc:
        logger.error(f"Unexpected prediction failure for transaction '{txn_id}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal fraud prediction failure.",
        ) from exc

    # Safe log of decision (NEVER log tokens or secrets)
    logger.info(
        f"Fraud evaluated (Phase 2): transaction_id='{txn_id}', user_id='{user_id}', "
        f"risk_band='{pred_result['risk_band']}', action='{pred_result['recommended_action']}', "
        f"calibrated_prob={pred_result['calibrated_fraud_probability']:.4f}, "
        f"latency={pred_result['prediction_latency_ms']:.2f}ms"
    )

    # 8. Save transaction in public.fraud_transactions (AFTER prediction computation)
    raw_txn_record = {
        "transaction_id": txn_id,
        "customer_id": request.customer_id,
        "card_id": request.card_id,
        "device_id": request.device_id,
        "merchant_id": request.merchant_id,
        "amount": float(request.amount),
        "transaction_timestamp": ts_iso,
        "email_domain": request.email_domain,
        "address_id": request.address_id,
        "product_code": request.product_code or "W",
        "ip_address": request.ip_address,
        "country": request.country,
    }

    try:
        admin.table("fraud_transactions").insert(raw_txn_record).execute()
    except Exception as exc:
        logger.error(f"Failed to persist fraud transaction '{txn_id}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist fraud transaction record.",
        ) from exc

    # 9. Save prediction in public.fraud_predictions
    raw_pred_record = {
        "transaction_id": txn_id,
        "model_name": pred_result["model_name"],
        "model_version": pred_result["model_version"],
        "calibration_version": pred_result["calibration_version"],
        "policy_version": pred_result["policy_version"],
        # Legacy compatibility columns
        "fraud_probability": float(pred_result["fraud_probability"]),
        "fraud_band": pred_result["fraud_band"],
        "decision": pred_result["decision"],
        "prediction_latency_ms": float(pred_result["prediction_latency_ms"]),
        # Canonical Phase 2 columns
        "raw_fraud_probability": float(pred_result["raw_fraud_probability"]),
        "calibrated_fraud_probability": float(pred_result["calibrated_fraud_probability"]),
        "risk_band": pred_result["risk_band"],
        "recommended_action": pred_result["recommended_action"],
        "feature_contract_version": pred_result["feature_contract_version"],
        "calibrator_type": pred_result["calibrator_type"],
        "scored_at": pred_result["scored_at"],
    }

    try:
        admin.table("fraud_predictions").insert(raw_pred_record).execute()
    except Exception as exc:
        logger.error(f"Failed to persist fraud prediction for '{txn_id}': {exc}")
        # Clean up transaction to avoid orphaned transaction record
        try:
            admin.table("fraud_transactions").delete().eq("transaction_id", txn_id).execute()
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist fraud prediction record.",
        ) from exc

    # 10. Save Feature Snapshot in public.fraud_feature_snapshots
    snapshot_record = {
        "transaction_id": txn_id,
        "feature_contract_version": pred_result["feature_contract_version"],
        "model_version": pred_result["model_version"],
        "feature_count": 62,
        "features": features_dict,
        "created_at": pred_result["scored_at"],
    }

    try:
        admin.table("fraud_feature_snapshots").insert(snapshot_record).execute()
    except Exception as exc:
        logger.warning(f"Failed to persist feature snapshot for '{txn_id}': {exc}")

    # 11. Update Entity State in public.fraud_entity_state (AFTER feature generation and prediction)
    try:
        update_entity_state(txn_dict)
    except Exception as exc:
        logger.warning(f"Entity state update encountered an issue for '{txn_id}': {exc}")

    # 12. Return structured response with Canonical and Legacy fields
    return FraudPredictionResponse(
        transaction_id=txn_id,
        # Legacy compatibility fields
        fraud_probability=pred_result["fraud_probability"],
        fraud_band=pred_result["fraud_band"],
        decision=pred_result["decision"],
        model_name=pred_result["model_name"],
        model_version=pred_result["model_version"],
        calibration_version=pred_result["calibration_version"],
        policy_version=pred_result["policy_version"],
        prediction_latency_ms=pred_result["prediction_latency_ms"],
        raw_probability=pred_result["raw_probability"],
        calibration_method=pred_result["calibration_method"],
        feature_count=pred_result["feature_count"],
        # Canonical Phase 2 fields
        raw_fraud_probability=pred_result["raw_fraud_probability"],
        calibrated_fraud_probability=pred_result["calibrated_fraud_probability"],
        risk_band=pred_result["risk_band"],
        recommended_action=pred_result["recommended_action"],
        feature_contract_version=pred_result["feature_contract_version"],
        calibrator_type=pred_result["calibrator_type"],
        scored_at=pred_result["scored_at"],
    )


# -----------------------------------------------------------------------------
# Transaction History & High-Risk Endpoints (For Streamlit & Dashboard)
# -----------------------------------------------------------------------------
@router.get("/history", summary="Get recent fraud transactions and predictions")
def get_fraud_history(
    limit: int = 15,
    offset: int = 0,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Returns paginated recent fraud predictions joined with transaction details.
    Restricted to authenticated analysts.
    """
    admin = get_supabase_admin_client()
    try:
        pred_res = (
            admin.table("fraud_predictions")
            .select(
                "transaction_id, model_version, fraud_probability, fraud_band, decision, "
                "prediction_latency_ms, created_at, raw_fraud_probability, "
                "calibrated_fraud_probability, risk_band, recommended_action"
            )
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        predictions = pred_res.data or []
        if not predictions:
            return {"items": [], "count": 0}

        txn_ids = [p["transaction_id"] for p in predictions]

        tx_res = (
            admin.table("fraud_transactions")
            .select("transaction_id, amount, customer_id, transaction_timestamp")
            .in_("transaction_id", txn_ids)
            .execute()
        )
        tx_map = {t["transaction_id"]: t for t in (tx_res.data or [])}

        items = []
        for p in predictions:
            tid = p["transaction_id"]
            tx_info = tx_map.get(tid, {})
            items.append({
                "transaction_id": tid,
                "amount": float(tx_info.get("amount", 0.0)),
                "customer_id": tx_info.get("customer_id", "N/A"),
                "timestamp": tx_info.get("transaction_timestamp") or p.get("created_at"),
                "fraud_probability": float(p.get("fraud_probability", 0.0)),
                "fraud_band": p.get("fraud_band", "LOW"),
                "decision": p.get("decision", "ALLOW"),
                "model_version": p.get("model_version", "2.1.0"),
                "prediction_latency_ms": float(p.get("prediction_latency_ms", 0.0)),
                # Canonical fields
                "raw_fraud_probability": float(p.get("raw_fraud_probability", 0.0)) if p.get("raw_fraud_probability") is not None else None,
                "calibrated_fraud_probability": float(p.get("calibrated_fraud_probability", 0.0)) if p.get("calibrated_fraud_probability") is not None else None,
                "risk_band": p.get("risk_band"),
                "recommended_action": p.get("recommended_action"),
            })

        return {"items": items, "count": len(items)}
    except Exception as exc:
        logger.error(f"Failed to fetch fraud history: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve transaction history.",
        ) from exc


@router.get("/high-risk", summary="Get recent high-risk fraud alerts")
def get_fraud_high_risk(
    limit: int = 10,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Returns recent high-risk or blocked transactions.
    Restricted to authenticated analysts.
    """
    admin = get_supabase_admin_client()
    try:
        pred_res = (
            admin.table("fraud_predictions")
            .select(
                "transaction_id, model_version, fraud_probability, fraud_band, decision, created_at, "
                "raw_fraud_probability, calibrated_fraud_probability, risk_band, recommended_action"
            )
            .in_("fraud_band", ["HIGH", "REVIEW"])
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        predictions = pred_res.data or []
        if not predictions:
            return {"items": [], "count": 0}

        txn_ids = [p["transaction_id"] for p in predictions]
        tx_res = (
            admin.table("fraud_transactions")
            .select("transaction_id, amount, customer_id, transaction_timestamp")
            .in_("transaction_id", txn_ids)
            .execute()
        )
        tx_map = {t["transaction_id"]: t for t in (tx_res.data or [])}

        items = []
        for p in predictions:
            tid = p["transaction_id"]
            tx_info = tx_map.get(tid, {})
            items.append({
                "transaction_id": tid,
                "amount": float(tx_info.get("amount", 0.0)),
                "customer_id": tx_info.get("customer_id", "N/A"),
                "timestamp": tx_info.get("transaction_timestamp") or p.get("created_at"),
                "fraud_probability": float(p.get("fraud_probability", 0.0)),
                "fraud_band": p.get("fraud_band", "HIGH"),
                "decision": p.get("decision", "BLOCK"),
                "model_version": p.get("model_version", "2.1.0"),
                "risk_band": p.get("risk_band"),
                "recommended_action": p.get("recommended_action"),
            })

        return {"items": items, "count": len(items)}
    except Exception as exc:
        logger.error(f"Failed to fetch high-risk fraud alerts: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve high-risk alerts.",
        ) from exc
