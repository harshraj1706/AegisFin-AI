from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from .config import (
    FRAUD_CALIBRATOR_V2_PATH,
    FRAUD_FEATURE_LIST_PATH,
    FRAUD_MODEL_PATH,
    FRAUD_MODEL_V2_PATH,
    FRAUD_POLICY_V2_PATH,
    FRAUD_TRAINING_METADATA_PATH,
)
from .production_feature_definitions import VALID_FEATURE_NAMES

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Custom Exceptions for Clear Error Diagnostics
# -----------------------------------------------------------------------------
class FraudModelArtifactError(Exception):
    """Raised when a model or calibrator artifact is missing, unreadable, or corrupted."""
    pass


class FraudModelSchemaError(ValueError):
    """Raised when the input feature schema does not match the 62-feature production contract."""
    pass


class FraudModelCalibrationError(ValueError):
    """Raised when the calibrator is missing or produces an invalid probability."""
    pass


class FraudModelInferenceError(RuntimeError):
    """Raised when XGBoost predict_proba fails during execution."""
    pass


class FraudPolicyError(RuntimeError):
    """Raised when the fraud risk policy configuration is missing or corrupted."""
    pass


# -----------------------------------------------------------------------------
# Standard Decision & Compatibility Maps
# -----------------------------------------------------------------------------
LEGACY_FRAUD_BAND_MAP: Dict[str, str] = {
    "LOW": "LOW",
    "MEDIUM": "REVIEW",
    "HIGH": "REVIEW",
    "CRITICAL": "HIGH",
}

LEGACY_DECISION_MAP: Dict[str, str] = {
    "AUTO_APPROVE": "ALLOW",
    "STEP_UP_AUTH": "MANUAL_REVIEW",
    "MANUAL_REVIEW": "MANUAL_REVIEW",
    "HARD_DECLINE": "BLOCK",
}


class FraudModelService:
    """
    Singleton service managing the frozen AegisFin Phase 2 production runtime:
    - 62-feature schema validation against production_feature_definitions
    - Frozen XGBoost v2 base model inference
    - Frozen Platt sigmoid probability calibration
    - Frozen 4-tier risk policy evaluation (LOW, MEDIUM, HIGH, CRITICAL)
    - Legacy 3-tier compatibility projection (LOW/REVIEW/HIGH, ALLOW/MANUAL_REVIEW/BLOCK)
    - Inference-only mode: Never calls fit(), retrains, or recalibrates.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        calibrator_path: Optional[Path] = None,
        policy_path: Optional[Path] = None,
    ):
        self.model_path = model_path or FRAUD_MODEL_V2_PATH
        self.calibrator_path = calibrator_path or FRAUD_CALIBRATOR_V2_PATH
        self.policy_path = policy_path or FRAUD_POLICY_V2_PATH

        # 1. Verify and Load Base XGBoost v2 Artifact
        if not self.model_path.exists():
            raise FraudModelArtifactError(
                f"Phase 2 XGBoost baseline model artifact not found at {self.model_path}"
            )

        try:
            logger.info(f"Loading Phase 2 XGBoost v2 Artifact from {self.model_path}")
            self.model_artifact: Dict[str, Any] = joblib.load(self.model_path)
        except Exception as exc:
            raise FraudModelArtifactError(
                f"Failed to load Phase 2 XGBoost v2 artifact at {self.model_path}: {exc}"
            ) from exc

        if not isinstance(self.model_artifact, dict) or "model" not in self.model_artifact:
            raise FraudModelArtifactError(
                "Invalid model artifact structure: expected dictionary containing 'model' key."
            )

        self.model = self.model_artifact["model"]
        if not hasattr(self.model, "predict_proba"):
            raise FraudModelArtifactError("Model object does not expose 'predict_proba'.")

        # 2. Extract Feature Schema (Contract: Exactly 62 Production Features)
        artifact_features = self.model_artifact.get("feature_names")
        if artifact_features:
            self.feature_names: List[str] = list(artifact_features)
        else:
            self.feature_names: List[str] = list(VALID_FEATURE_NAMES)

        self.expected_feature_count: int = len(self.feature_names)
        if self.expected_feature_count != 62:
            raise FraudModelSchemaError(
                f"Phase 2 model contract violation: expected 62 features, got {self.expected_feature_count}."
            )

        if self.feature_names != VALID_FEATURE_NAMES:
            raise FraudModelSchemaError(
                "Phase 2 model feature names/order deviate from VALID_FEATURE_NAMES contract."
            )

        # 3. Verify and Load Frozen Calibrator Artifact
        if not self.calibrator_path.exists():
            raise FraudModelArtifactError(
                f"Phase 2 probability calibrator artifact not found at {self.calibrator_path}"
            )

        try:
            logger.info(f"Loading Phase 2 Probability Calibrator from {self.calibrator_path}")
            self.calibrator_artifact: Dict[str, Any] = joblib.load(self.calibrator_path)
        except Exception as exc:
            raise FraudModelArtifactError(
                f"Failed to load Phase 2 probability calibrator at {self.calibrator_path}: {exc}"
            ) from exc

        if not isinstance(self.calibrator_artifact, dict) or "calibrator" not in self.calibrator_artifact:
            raise FraudModelCalibrationError(
                "Invalid calibrator artifact structure: expected dictionary containing 'calibrator' key."
            )

        self.calibrator = self.calibrator_artifact["calibrator"]
        if not hasattr(self.calibrator, "predict_proba") and not hasattr(self.calibrator, "predict"):
            raise FraudModelCalibrationError("Calibrator object lacks predict or predict_proba.")

        self.calibration_method: str = "Platt"
        self.calibrator_type: str = str(self.calibrator_artifact.get("selected_method", "sigmoid"))

        # 4. Verify and Load Frozen Risk Policy Config
        if not self.policy_path.exists():
            raise FraudPolicyError(
                f"Phase 2 fraud risk policy config not found at {self.policy_path}"
            )

        try:
            logger.info(f"Loading Phase 2 Fraud Risk Policy from {self.policy_path}")
            with open(self.policy_path, "r", encoding="utf-8") as pf:
                self.policy_config: Dict[str, Any] = json.load(pf)
        except Exception as exc:
            raise FraudPolicyError(
                f"Failed to load Phase 2 risk policy JSON at {self.policy_path}: {exc}"
            ) from exc

        self.policy_risk_bands: Dict[str, Any] = self.policy_config.get("risk_bands", {})
        if not self.policy_risk_bands:
            raise FraudPolicyError("Policy configuration missing required 'risk_bands' section.")

        # 5. Metadata Versions
        self.model_name: str = "XGBoost"
        self.model_version: str = str(self.policy_config.get("model_version", "2.1.0"))
        self.calibration_version: str = str(self.policy_config.get("calibrator_version", "2.1.0"))
        self.policy_version: str = str(self.policy_config.get("policy_version", "2.0.0"))
        self.feature_contract_version: str = str(self.policy_config.get("feature_contract_version", "2.1.0"))

        # Policy Thresholds
        self.decision_threshold: float = float(self.policy_config.get("provisional_action_threshold", 0.50))
        self.review_threshold: float = 0.10
        self.high_threshold: float = 0.80

        logger.info(
            f"FraudModelService (Phase 2 Cutover) initialized: model={self.model_name} "
            f"v{self.model_version}, features={self.expected_feature_count}, "
            f"calibrator={self.calibration_method} ({self.calibrator_type}), policy=v{self.policy_version}"
        )

    def validate_features(self, df_or_dict: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """
        Strictly validates the input against the 62-feature production contract.
        Accepts either a pandas DataFrame or a dictionary of features.
        Raises FraudModelSchemaError on any count, ordering, type, NaN, or Inf mismatch.
        Returns a validated 1-row or multi-row pandas DataFrame.
        """
        if isinstance(df_or_dict, dict):
            # Check for missing contract features
            missing = [f for f in self.feature_names if f not in df_or_dict]
            if missing:
                raise FraudModelSchemaError(
                    f"Input feature dictionary missing {len(missing)} required features: {missing[:5]}"
                )
            df = pd.DataFrame([[df_or_dict[f] for f in self.feature_names]], columns=self.feature_names)
        elif isinstance(df_or_dict, pd.DataFrame):
            df = df_or_dict
        else:
            raise FraudModelSchemaError(
                f"Invalid feature input type: expected DataFrame or dict, received {type(df_or_dict)}."
            )

        # 1. Feature Count Validation
        actual_count = df.shape[1]
        if actual_count != self.expected_feature_count:
            raise FraudModelSchemaError(
                f"Fraud model feature schema mismatch. Model expects {self.expected_feature_count} features, "
                f"received {actual_count} features."
            )

        # 2. Feature Names and Ordering Validation
        actual_cols = list(df.columns)
        if actual_cols != self.feature_names:
            missing = [c for c in self.feature_names if c not in actual_cols]
            unexpected = [c for c in actual_cols if c not in self.feature_names]
            raise FraudModelSchemaError(
                f"Fraud model feature schema mismatch. Missing features: {missing[:5]}, "
                f"Unexpected features: {unexpected[:5]}, exact ordering match: False."
            )

        # 3. Numeric Compatibility
        non_numeric = [c for c in df.columns if not np.issubdtype(df[c].dtype, np.number)]
        if non_numeric:
            raise FraudModelSchemaError(
                f"Fraud model received non-numeric feature columns: {non_numeric[:5]}"
            )

        # 4. NaN / Null Check
        if df.isna().any().any():
            nan_cols = df.columns[df.isna().any()].tolist()
            raise FraudModelSchemaError(
                f"Fraud model feature matrix contains unexpected NaN values in columns: {nan_cols[:5]}"
            )

        # 5. Infinity Check
        vals = df.to_numpy(dtype=float)
        if np.isinf(vals).any():
            inf_cols = df.columns[np.isinf(vals).any(axis=0)].tolist()
            raise FraudModelSchemaError(
                f"Fraud model feature matrix contains infinite values in columns: {inf_cols[:5]}"
            )

        return df

    def calibrate_probability(self, raw_prob: float) -> float:
        """
        Applies the frozen Platt sigmoid calibrator to the raw positive-class probability.
        Guarantees result is strictly bounded within [0.0, 1.0].
        Formula:
        1. Clip raw_prob to [1e-15, 1 - 1e-15]
        2. Compute logit: log(p / (1 - p))
        3. Predict calibrated probability using frozen logistic regression
        """
        if not (0.0 <= raw_prob <= 1.0):
            raise FraudModelCalibrationError(
                f"Raw probability out of bounds [0, 1]: {raw_prob}"
            )

        try:
            eps = 1e-15
            p_clipped = float(np.clip(raw_prob, eps, 1.0 - eps))
            logit_val = float(np.log(p_clipped / (1.0 - p_clipped)))
            input_arr = np.array([[logit_val]])

            if hasattr(self.calibrator, "predict_proba"):
                cal_probs = self.calibrator.predict_proba(input_arr)
                cal_prob = float(cal_probs[0, 1])
            elif hasattr(self.calibrator, "predict"):
                cal_prob = float(self.calibrator.predict(input_arr)[0])
            else:
                raise FraudModelCalibrationError(
                    f"Calibrator {type(self.calibrator)} lacks predict_proba or predict."
                )
        except Exception as exc:
            raise FraudModelCalibrationError(f"Calibration application failed: {exc}") from exc

        return max(0.0, min(1.0, cal_prob))

    def evaluate_risk_policy(self, calibrated_prob: float) -> Tuple[str, str]:
        """
        Evaluates calibrated fraud probability against the frozen Phase 2 4-tier risk policy.
        Returns:
            (canonical_risk_band, canonical_recommended_action)
            Bands: LOW, MEDIUM, HIGH, CRITICAL
            Actions: AUTO_APPROVE, STEP_UP_AUTH, MANUAL_REVIEW, HARD_DECLINE
        """
        if not (0.0 <= calibrated_prob <= 1.0):
            raise ValueError(f"Calibrated probability must be between 0 and 1, got {calibrated_prob}")

        # Evaluate against loaded policy_risk_bands
        for band_name in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            band_info = self.policy_risk_bands.get(band_name)
            if not band_info:
                continue
            prange = band_info.get("probability_range", [0.0, 1.0])
            lower, upper = float(prange[0]), float(prange[1])
            action = str(band_info.get("operational_action", "MANUAL_REVIEW"))

            if band_name == "CRITICAL":
                # [0.80, 1.00] inclusive of upper boundary
                if lower <= calibrated_prob <= upper:
                    return band_name, action
            else:
                # [lower, upper) half-open
                if lower <= calibrated_prob < upper:
                    return band_name, action

        # Fallback to standard canonical boundaries
        if calibrated_prob < 0.10:
            return "LOW", "AUTO_APPROVE"
        elif calibrated_prob < 0.40:
            return "MEDIUM", "STEP_UP_AUTH"
        elif calibrated_prob < 0.80:
            return "HIGH", "MANUAL_REVIEW"
        else:
            return "CRITICAL", "HARD_DECLINE"

    def predict(self, features: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes complete frozen Phase 2 scoring:
        1. Validates 62-feature schema
        2. Executes frozen XGBoost v2 predict_proba
        3. Applies frozen Platt sigmoid probability calibration
        4. Evaluates frozen 4-tier risk policy
        5. Computes legacy 3-tier compatibility projections
        6. Measures inference latency
        """
        df = self.validate_features(features)

        start_time = time.perf_counter()

        try:
            raw_probs = self.model.predict_proba(df)
        except Exception as exc:
            raise FraudModelInferenceError(f"XGBoost v2 predict_proba failed: {exc}") from exc

        raw_prob = float(raw_probs[0, 1])
        calibrated_prob = self.calibrate_probability(raw_prob)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        risk_band, recommended_action = self.evaluate_risk_policy(calibrated_prob)
        legacy_band = LEGACY_FRAUD_BAND_MAP.get(risk_band, "REVIEW")
        legacy_decision = LEGACY_DECISION_MAP.get(recommended_action, "MANUAL_REVIEW")
        now_iso = datetime.now(timezone.utc).isoformat()

        return {
            # Canonical Phase 2 Fields (Source of Truth)
            "raw_fraud_probability": float(raw_prob),
            "calibrated_fraud_probability": float(calibrated_prob),
            "risk_band": risk_band,
            "recommended_action": recommended_action,
            "feature_contract_version": self.feature_contract_version,
            "calibrator_type": self.calibrator_type,
            "scored_at": now_iso,

            # Legacy Compatibility Projections
            "fraud_probability": float(calibrated_prob),
            "fraud_band": legacy_band,
            "decision": legacy_decision,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "calibration_version": self.calibration_version,
            "policy_version": self.policy_version,
            "prediction_latency_ms": round(latency_ms, 4),
            "raw_probability": float(raw_prob),
            "calibration_method": self.calibration_method,
            "feature_count": self.expected_feature_count,
            "inference_latency_ms": round(latency_ms, 4),
            "model_inference_latency_ms": round(latency_ms, 4),
        }

    def get_fraud_band(self, probability: float) -> str:
        """Assigns a legacy fraud band ('LOW', 'REVIEW', 'HIGH') using policy thresholds."""
        band, _ = self.evaluate_risk_policy(probability)
        return LEGACY_FRAUD_BAND_MAP.get(band, "REVIEW")

    def get_fraud_decision(self, band_or_action: str) -> str:
        """Maps risk band or action to legacy operational decision ('ALLOW', 'MANUAL_REVIEW', 'BLOCK')."""
        if band_or_action in LEGACY_DECISION_MAP:
            return LEGACY_DECISION_MAP[band_or_action]
        if band_or_action in ("LOW", "ALLOW"):
            return "ALLOW"
        if band_or_action in ("REVIEW", "MANUAL_REVIEW", "MEDIUM", "HIGH"):
            return "MANUAL_REVIEW"
        if band_or_action in ("CRITICAL", "BLOCK", "HARD_DECLINE"):
            return "BLOCK"
        return "MANUAL_REVIEW"

    def info(self) -> Dict[str, Any]:
        """Returns model service metadata, contract versions, and policy thresholds."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "calibration_version": self.calibration_version,
            "policy_version": self.policy_version,
            "feature_contract_version": self.feature_contract_version,
            "calibration_method": self.calibration_method,
            "calibrator_type": self.calibrator_type,
            "expected_feature_count": self.expected_feature_count,
            "risk_bands": list(self.policy_risk_bands.keys()),
            "status": "ready",
        }


# -----------------------------------------------------------------------------
# Legacy IEEE-CIS 459-Feature Benchmark Model Service (Offline / Reference Only)
# -----------------------------------------------------------------------------
class LegacyFraudModelService:
    """
    Offline reference benchmark service for the legacy 459-feature IEEE-CIS model.
    NOT used by live Phase 2 prediction endpoint.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        feature_list_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ):
        self.model_path = model_path or FRAUD_MODEL_PATH
        self.feature_list_path = feature_list_path or FRAUD_FEATURE_LIST_PATH
        self.metadata_path = metadata_path or FRAUD_TRAINING_METADATA_PATH

        if not self.model_path.exists():
            raise FraudModelArtifactError(f"Legacy model artifact not found at {self.model_path}")

        self.artifact = joblib.load(self.model_path)
        self.model = self.artifact["model"]
        self.feature_names = list(self.artifact["feature_names"])
        self.expected_feature_count = len(self.feature_names)
        self.calibration = self.artifact.get("calibration", {})
        self.calibrator = self.calibration.get("selected_model")
        self.policy = self.artifact.get("policy", {})
        self.model_name = str(self.artifact.get("model_name", "XGBoost"))
        self.model_version = "phase2-xgb-legacy-459"


# -----------------------------------------------------------------------------
# Singleton Management & Functional Helpers
# -----------------------------------------------------------------------------
_fraud_model_service: Optional[FraudModelService] = None
_legacy_model_service: Optional[LegacyFraudModelService] = None


def get_fraud_model_service() -> FraudModelService:
    """Returns cached singleton FraudModelService instance (frozen Phase 2 62-feature pipeline)."""
    global _fraud_model_service
    if _fraud_model_service is None:
        _fraud_model_service = FraudModelService()
    return _fraud_model_service


def get_legacy_fraud_model_service() -> LegacyFraudModelService:
    """Returns legacy 459-feature model service for offline benchmark reference."""
    global _legacy_model_service
    if _legacy_model_service is None:
        _legacy_model_service = LegacyFraudModelService()
    return _legacy_model_service


def predict_fraud_risk(features: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
    """Public helper running Phase 2 model inference on 62 features."""
    service = get_fraud_model_service()
    return service.predict(features)


def get_fraud_band(probability: float) -> str:
    """Public helper mapping calibrated probability to legacy risk band."""
    service = get_fraud_model_service()
    return service.get_fraud_band(probability)


def get_fraud_decision(band_or_action: str) -> str:
    """Public helper mapping risk band or action to legacy operational decision."""
    service = get_fraud_model_service()
    return service.get_fraud_decision(band_or_action)
