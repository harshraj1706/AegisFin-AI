from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .config import (
    FRAUD_FEATURE_LIST_PATH,
    FRAUD_MODEL_PATH,
    FRAUD_TRAINING_METADATA_PATH,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Custom Exceptions for Clear Error Diagnostics
# -----------------------------------------------------------------------------
class FraudModelArtifactError(Exception):
    """Raised when the champion model artifact is missing, unreadable, or corrupted."""
    pass


class FraudModelSchemaError(ValueError):
    """Raised when the input feature schema does not match the model's exact requirements."""
    pass


class FraudModelCalibrationError(ValueError):
    """Raised when the calibrator is missing or produces an invalid probability."""
    pass


class FraudModelInferenceError(RuntimeError):
    """Raised when XGBoost predict_proba fails during execution."""
    pass


# -----------------------------------------------------------------------------
# Standard Decision Map
# -----------------------------------------------------------------------------
FRAUD_DECISION_MAP: Dict[str, str] = {
    "LOW": "ALLOW",
    "REVIEW": "MANUAL_REVIEW",
    "HIGH": "BLOCK",
}


class FraudModelService:
    """
    Singleton service managing model loading, schema validation, raw inference,
    probability calibration, and policy threshold evaluation for Phase 2 Fraud Detection.
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

        # 1. Verify artifact exists
        if not self.model_path.exists():
            raise FraudModelArtifactError(
                f"Phase 2 fraud champion model artifact not found at {self.model_path}"
            )

        # 2. Load artifact once
        try:
            logger.info(f"Loading Phase 2 Fraud Champion Artifact from {self.model_path}")
            self.artifact: Dict[str, Any] = joblib.load(self.model_path)
        except Exception as exc:
            raise FraudModelArtifactError(
                f"Failed to load Phase 2 fraud model artifact at {self.model_path}: {exc}"
            ) from exc

        if not isinstance(self.artifact, dict):
            raise FraudModelArtifactError(
                f"Invalid model artifact structure: expected dict, received {type(self.artifact)}"
            )

        # 3. Validate required artifact keys
        required_keys = {"model", "preprocessor", "feature_names", "calibration", "policy"}
        missing_keys = required_keys - set(self.artifact.keys())
        if missing_keys:
            raise FraudModelArtifactError(
                f"Corrupted model artifact: missing required keys: {sorted(missing_keys)}"
            )

        # 4. Model Object
        self.model = self.artifact["model"]
        if not hasattr(self.model, "predict_proba"):
            raise FraudModelArtifactError(
                "Model object in artifact does not expose 'predict_proba' method."
            )

        # 5. Dynamic Schema & Feature Names
        self.feature_names: List[str] = list(self.artifact["feature_names"])
        self.expected_feature_count: int = len(self.feature_names)

        # Cross-verify with feature_list_path if available
        if self.feature_list_path.exists():
            try:
                feat_df = pd.read_csv(self.feature_list_path)
                col_name = "feature" if "feature" in feat_df.columns else feat_df.columns[0]
                csv_features = feat_df[col_name].astype(str).tolist()
                if len(csv_features) != self.expected_feature_count:
                    raise FraudModelSchemaError(
                        f"Feature count mismatch: CSV has {len(csv_features)}, "
                        f"artifact has {self.expected_feature_count}."
                    )
                if csv_features != self.feature_names:
                    raise FraudModelSchemaError(
                        "Feature ordering in feature_list.csv deviates from champion artifact."
                    )
            except Exception as exc:
                if isinstance(exc, FraudModelSchemaError):
                    raise
                logger.warning(f"Feature list cross-validation skipped: {exc}")

        # 6. Calibration Object & Method
        cal_dict = self.artifact.get("calibration", {})
        if not isinstance(cal_dict, dict) or "selected_model" not in cal_dict:
            raise FraudModelCalibrationError(
                "Model artifact is missing calibration configuration or selected_model."
            )

        self.calibration_method: str = str(cal_dict.get("selected", "Platt"))
        self.calibrator = cal_dict["selected_model"]
        if self.calibrator is None:
            raise FraudModelCalibrationError("Calibrator model object is None in artifact.")

        # 7. Policy Thresholds & Bands
        self.policy: Dict[str, Any] = self.artifact.get("policy", {})
        self.bands: List[str] = self.policy.get("bands", ["LOW", "REVIEW", "HIGH"])
        self.decision_threshold: float = float(self.policy.get("decision_threshold", 0.11))
        self.review_threshold: float = float(self.policy.get("review_threshold", 0.24520720672829832))
        self.high_threshold: float = float(self.policy.get("high_threshold", 0.3544080190457745))

        # 8. Preprocessor Metadata
        self.preprocessor: Dict[str, Any] = self.artifact.get("preprocessor", {})
        self.numeric_fill: Dict[str, float] = self.preprocessor.get("numeric_fill", {})
        self.frequency_maps: Dict[str, Dict[str, float]] = self.preprocessor.get("frequency_maps", {})

        # 9. Model Metadata & Versions
        self.model_name: str = str(self.artifact.get("model_name", "XGBoost"))
        self.model_version: str = "phase2-xgb-v1"
        self.calibration_version: str = f"phase2-{self.calibration_method.lower()}-v1"
        self.policy_version: str = "phase2-policy-v1"

        logger.info(
            f"FraudModelService initialized successfully: model={self.model_name}, "
            f"features={self.expected_feature_count}, calibrator={self.calibration_method}, "
            f"review_thresh={self.review_threshold:.4f}, high_thresh={self.high_threshold:.4f}"
        )

    def validate_features(self, df: pd.DataFrame) -> None:
        """
        Strictly validates the input DataFrame against the champion model feature schema.
        Raises FraudModelSchemaError on any count, ordering, type, NaN, or Inf mismatch.
        """
        if not isinstance(df, pd.DataFrame):
            raise FraudModelSchemaError(
                f"Invalid feature input type: expected pandas.DataFrame, received {type(df)}."
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
                f"Fraud model feature schema mismatch. Missing features: {missing[:10]} (total {len(missing)}), "
                f"Unexpected features: {unexpected[:10]} (total {len(unexpected)}), "
                f"Exact ordering matches: False."
            )

        # 3. Numeric Compatibility
        non_numeric = [c for c in df.columns if not np.issubdtype(df[c].dtype, np.number)]
        if non_numeric:
            raise FraudModelSchemaError(
                f"Fraud model received non-numeric feature columns: {non_numeric[:10]}"
            )

        # 4. NaN / Null Check
        if df.isna().any().any():
            nan_cols = df.columns[df.isna().any()].tolist()
            raise FraudModelSchemaError(
                f"Fraud model feature matrix contains unexpected NaN values in columns: {nan_cols[:10]}"
            )

        # 5. Infinity Check
        vals = df.to_numpy()
        if np.isinf(vals).any():
            inf_cols = df.columns[np.isinf(vals).any(axis=0)].tolist()
            raise FraudModelSchemaError(
                f"Fraud model feature matrix contains infinite values in columns: {inf_cols[:10]}"
            )

    def calibrate_probability(self, raw_prob: float) -> float:
        """
        Applies the stored calibrator (Platt or Isotonic) to the raw positive-class probability.
        Guarantees result is strictly bounded within [0.0, 1.0].

        For Platt scaling:
        1. Clip raw probability to (0, 1) strictly to avoid log(0) or division by zero.
        2. Compute logit(raw_prob) = log(p / (1 - p)).
        3. Reshape to (-1, 1).
        4. Call saved Platt LogisticRegression.predict_proba().
        5. Return positive-class probability.
        """
        if not (0.0 <= raw_prob <= 1.0):
            raise FraudModelCalibrationError(
                f"Raw probability out of bounds [0, 1]: {raw_prob}"
            )

        try:
            # 1. Clip raw probability strictly to (0, 1) for numerical stability
            eps = 1e-12
            p_clipped = float(np.clip(raw_prob, eps, 1.0 - eps))

            if self.calibration_method == "Platt" or hasattr(self.calibrator, "predict_proba"):
                # 2. Compute logit(raw_probability) = log(p / (1 - p))
                logit_val = float(np.log(p_clipped / (1.0 - p_clipped)))
                # 3. Reshape to (-1, 1)
                input_arr = np.array([[logit_val]])
                # 4. Call saved Platt LogisticRegression.predict_proba()
                cal_probs = self.calibrator.predict_proba(input_arr)
                # 5. Extract positive-class probability
                cal_prob = float(cal_probs[0, 1])
            elif hasattr(self.calibrator, "predict"):
                input_arr = np.array([p_clipped])
                cal_prob = float(self.calibrator.predict(input_arr)[0])
            else:
                raise FraudModelCalibrationError(
                    f"Calibrator object of type {type(self.calibrator)} lacks predict or predict_proba."
                )
        except Exception as exc:
            raise FraudModelCalibrationError(f"Calibration application failed: {exc}") from exc

        # Numerical stability clamp
        cal_prob = max(0.0, min(1.0, cal_prob))
        return cal_prob

    def get_fraud_band(self, probability: float) -> str:
        """
        Assigns a risk band ('LOW', 'REVIEW', 'HIGH') dynamically using saved policy thresholds.
        """
        if not (0.0 <= probability <= 1.0):
            raise ValueError(f"Probability must be between 0 and 1, got {probability}")

        if probability < self.review_threshold:
            return "LOW"
        elif probability < self.high_threshold:
            return "REVIEW"
        else:
            return "HIGH"

    def get_fraud_decision(self, band: str) -> str:
        """
        Maps a risk band to an automated operational decision ('ALLOW', 'MANUAL_REVIEW', 'BLOCK').
        """
        if band not in FRAUD_DECISION_MAP:
            raise ValueError(
                f"Invalid fraud band '{band}'. Expected one of {list(FRAUD_DECISION_MAP.keys())}"
            )
        return FRAUD_DECISION_MAP[band]

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes complete model inference for a single or batch feature DataFrame.
        Validates feature schema, runs XGBoost predict_proba, applies Platt calibration,
        determines risk band and policy decision, and measures model inference latency.
        """
        # Validate schema
        self.validate_features(df)

        # Measure model inference latency only (excludes startup/feature engineering)
        start_time = time.perf_counter()

        try:
            raw_probs = self.model.predict_proba(df)
        except Exception as exc:
            raise FraudModelInferenceError(f"XGBoost predict_proba failed: {exc}") from exc

        # Extract positive class probability
        raw_prob = float(raw_probs[0, 1])

        # Apply calibration
        calibrated_prob = self.calibrate_probability(raw_prob)

        # Measure elapsed time in milliseconds
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Apply policy thresholds
        band = self.get_fraud_band(calibrated_prob)
        decision = self.get_fraud_decision(band)

        return {
            "raw_probability": float(raw_prob),
            "fraud_probability": float(calibrated_prob),
            "fraud_band": band,
            "decision": decision,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "calibration_version": self.calibration_version,
            "policy_version": self.policy_version,
            "calibration_method": self.calibration_method,
            "feature_count": self.expected_feature_count,
            "inference_latency_ms": round(latency_ms, 4),
            "model_inference_latency_ms": round(latency_ms, 4),
            "policy_thresholds": {
                "decision_threshold": self.decision_threshold,
                "review_threshold": self.review_threshold,
                "high_threshold": self.high_threshold,
            },
        }

    def info(self) -> Dict[str, Any]:
        """Returns model service metadata and status."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "calibration_version": self.calibration_version,
            "policy_version": self.policy_version,
            "calibration_method": self.calibration_method,
            "expected_feature_count": self.expected_feature_count,
            "policy_thresholds": {
                "decision_threshold": self.decision_threshold,
                "review_threshold": self.review_threshold,
                "high_threshold": self.high_threshold,
            },
            "status": "ready",
        }


# -----------------------------------------------------------------------------
# Singleton Management & Functional Helpers
# -----------------------------------------------------------------------------
_fraud_model_service: Optional[FraudModelService] = None


def get_fraud_model_service() -> FraudModelService:
    """Returns cached singleton FraudModelService instance."""
    global _fraud_model_service
    if _fraud_model_service is None:
        _fraud_model_service = FraudModelService()
    return _fraud_model_service


def predict_fraud_risk(df: pd.DataFrame) -> Dict[str, Any]:
    """Public helper running model inference on a model-ready feature DataFrame."""
    service = get_fraud_model_service()
    return service.predict(df)


def get_fraud_band(probability: float) -> str:
    """Public helper mapping a calibrated probability to a fraud risk band."""
    service = get_fraud_model_service()
    return service.get_fraud_band(probability)


def get_fraud_decision(band: str) -> str:
    """Public helper mapping a fraud risk band to an operational decision."""
    service = get_fraud_model_service()
    return service.get_fraud_decision(band)
