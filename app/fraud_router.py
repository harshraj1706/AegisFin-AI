from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from .auth import get_current_user
from .fraud_feature_service import build_fraud_features, get_fraud_feature_service, parse_timestamp, update_entity_state
from .fraud_model_service import (
    FraudModelCalibrationError,
    FraudModelInferenceError,
    FraudModelSchemaError,
    get_fraud_model_service,
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
    Development-only endpoint verifying Phase 2 fraud feature engineering.
    Returns feature count, schema validity, and non-sensitive sample features.
    """
    try:
        service = get_fraud_feature_service()
        df = build_fraud_features(transaction)

        sample_keys = [
            "TransactionAmt",
            "hour",
            "weekday_index",
            "is_weekend",
            "is_night",
            "TransactionAmt_log",
            "TransactionAmt_cents",
            "missing_count",
            "uid_past_count",
            "uid_past_mean_amt",
            "uid_amount_ratio",
            "uid_is_new",
            "card1_past_count",
            "card1_past_mean_amt",
            "card1_is_new",
            "ProductCD__freq",
            "card4__freq",
            "P_emaildomain__freq",
        ]
        sample_features = {k: float(df[k].values[0]) for k in sample_keys if k in df.columns}

        return {
            "feature_count": int(df.shape[1]),
            "expected_feature_count": 459,
            "feature_schema_valid": bool(df.shape[1] == 459 and list(df.columns) == service.feature_names),
            "sample_features": sample_features,
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {exc}") from exc


# -----------------------------------------------------------------------------
# Final Phase 2 Fraud Prediction Endpoint
# -----------------------------------------------------------------------------
@router.post(
    "/predict",
    response_model=FraudPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate live transaction for fraud risk",
    description=(
        "Executes end-to-end fraud risk evaluation: reads historical point-in-time state, "
        "constructs the 459 model features without leakage, performs calibrated XGBoost inference, "
        "persists the transaction and prediction in Supabase, and updates entity history."
    ),
)
def predict_fraud(
    request: FraudPredictionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Full Phase 2 Fraud Prediction Lifecycle:
    1. Authenticate user via Supabase Bearer token
    2. Validate transaction request (amount >= 0, non-empty identifiers)
    3. Check transaction_id for duplicate submission (409 Conflict if duplicate)
    4. Read historical state strictly prior to current transaction timestamp (anti-leakage)
    5. Construct 459-dimensional feature vector
    6. Validate model feature schema
    7. Execute XGBoost champion model inference
    8. Apply Platt probability calibration
    9. Apply dynamic policy thresholds (LOW, REVIEW, HIGH)
    10. Map operational decision (ALLOW, MANUAL_REVIEW, BLOCK)
    11. Persist transaction in public.fraud_transactions
    12. Persist prediction in public.fraud_predictions
    13. Update public.fraud_entity_state aggregates
    14. Return structured response (without exposing internal features or secrets)
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

    # 4 & 5. Generate Leakage-Safe Features (Current transaction NOT yet in database)
    try:
        features_df = build_fraud_features(txn_dict)
    except Exception as exc:
        logger.error(f"Feature engineering failed for transaction '{txn_id}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Fraud feature engineering failed: {exc}",
        ) from exc

    # 6, 7, 8, 9, 10. Model Inference, Platt Calibration & Policy Assignment
    try:
        model_service = get_fraud_model_service()
        pred_result = model_service.predict(features_df)
    except FraudModelSchemaError as exc:
        logger.error(f"Feature schema error for transaction '{txn_id}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except (FraudModelCalibrationError, FraudModelInferenceError) as exc:
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
        f"Fraud evaluated: transaction_id='{txn_id}', user_id='{user_id}', "
        f"band='{pred_result['fraud_band']}', prob={pred_result['fraud_probability']:.4f}, "
        f"latency={pred_result['inference_latency_ms']:.2f}ms"
    )

    # 11. Save transaction in public.fraud_transactions (AFTER prediction)
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
        # Continue or raise; per requirements, saving transaction is required
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist fraud transaction record.",
        ) from exc

    # 12. Save prediction in public.fraud_predictions
    raw_pred_record = {
        "transaction_id": txn_id,
        "model_name": pred_result["model_name"],
        "model_version": pred_result["model_version"],
        "calibration_version": pred_result["calibration_version"],
        "policy_version": pred_result["policy_version"],
        "fraud_probability": float(pred_result["fraud_probability"]),
        "fraud_band": pred_result["fraud_band"],
        "decision": pred_result["decision"],
        "prediction_latency_ms": float(pred_result["inference_latency_ms"]),
    }

    try:
        admin.table("fraud_predictions").insert(raw_pred_record).execute()
    except Exception as exc:
        logger.error(f"Failed to persist fraud prediction for '{txn_id}': {exc}")
        # Note: transaction already saved, clean up or raise
        admin.table("fraud_transactions").delete().eq("transaction_id", txn_id).execute()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist fraud prediction record.",
        ) from exc

    # 13. Update Entity State (AFTER features generated & prediction stored)
    try:
        update_entity_state(txn_dict)
    except Exception as exc:
        logger.warning(f"Entity state update encountered an issue for '{txn_id}': {exc}")

    # 14. Return structured response
    return FraudPredictionResponse(
        transaction_id=txn_id,
        fraud_probability=pred_result["fraud_probability"],
        fraud_band=pred_result["fraud_band"],
        decision=pred_result["decision"],
        model_name=pred_result["model_name"],
        model_version=pred_result["model_version"],
        calibration_version=pred_result["calibration_version"],
        policy_version=pred_result["policy_version"],
        prediction_latency_ms=pred_result["inference_latency_ms"],
        raw_probability=pred_result["raw_probability"],
        calibration_method=pred_result["calibration_method"],
        feature_count=pred_result["feature_count"],
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
            .select("transaction_id, model_version, fraud_probability, fraud_band, decision, prediction_latency_ms, created_at")
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
                "model_version": p.get("model_version", "phase2-xgb-v1"),
                "prediction_latency_ms": float(p.get("prediction_latency_ms", 0.0)),
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
            .select("transaction_id, model_version, fraud_probability, fraud_band, decision, created_at")
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
                "model_version": p.get("model_version", "phase2-xgb-v1"),
            })

        return {"items": items, "count": len(items)}
    except Exception as exc:
        logger.error(f"Failed to fetch high-risk fraud alerts: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve high-risk alerts.",
        ) from exc
