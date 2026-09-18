from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .feature_engineering import engineer_features


# Deterministic risk-band thresholds (provisional technical / demo thresholds)
DEFAULT_RISK_THRESHOLDS = {
    "LOW_MAX": 0.15,
    "MODERATE_MAX": 0.40,
    "ELEVATED_MAX": 0.70,
}


def safe_logit(probabilities: np.ndarray | List[float] | float) -> np.ndarray:
    """Compute numerical log-odds (logit) clamped to [1e-6, 1 - 1e-6]."""
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return np.log(probs / (1.0 - probs))


def classify_risk_band(
    calibrated_probability: float,
    thresholds: Optional[Dict[str, float]] = None,
) -> str:
    """
    Deterministic risk-band classification based on calibrated default probability.
    Thresholds are provisional technical/demo thresholds (not official banking lending policy).
    """
    p = float(calibrated_probability)
    if not (0.0 <= p <= 1.0):
        raise ValueError("calibrated_probability must be between 0 and 1.")

    t = thresholds or DEFAULT_RISK_THRESHOLDS
    low_max = t.get("LOW_MAX", 0.15)
    mod_max = t.get("MODERATE_MAX", 0.40)
    elev_max = t.get("ELEVATED_MAX", 0.70)

    if p < low_max:
        return "LOW"
    if p < mod_max:
        return "MODERATE"
    if p < elev_max:
        return "ELEVATED"
    return "HIGH"


class Phase1RiskService:
    """
    Canonical credit-risk inference service for Phase 1B.
    Loads the frozen Phase 1B bundle (containing frozen XGBoost, preprocessing state,
    and Platt scaling calibrator) and executes deterministic inference.
    No retraining or model fitting happens during inference.
    """

    def __init__(self, bundle_path: str | Path):
        self.bundle_path = Path(bundle_path)

        if not self.bundle_path.exists():
            raise FileNotFoundError(f"Phase 1B model bundle not found: {self.bundle_path}")

        with self.bundle_path.open("rb") as f:
            self.bundle: Dict[str, Any] = pickle.load(f)

        required_keys = {
            "model",
            "preprocessing_state",
            "selected_features",
            "categorical_features",
            "calibrator",
            "calibration_method",
            "risk_policy",
            "metadata",
        }
        missing = required_keys - set(self.bundle)
        if missing:
            raise ValueError(
                f"Invalid Phase 1B calibrated model bundle. Missing keys: {sorted(missing)}"
            )

        self.model = self.bundle["model"]
        self.preprocessing_state = self.bundle["preprocessing_state"]
        self.selected_features = list(self.bundle["selected_features"])
        self.categorical_features = list(self.bundle["categorical_features"])
        self.calibrator = self.bundle["calibrator"]
        self.calibration_method = self.bundle["calibration_method"]
        self.risk_policy = self.bundle["risk_policy"]
        self.metadata = self.bundle["metadata"]

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Applies Phase 1A feature engineering and saved preprocessing state."""
        X = engineer_features(raw_df.copy())
        state = self.preprocessing_state

        X = X.drop(columns=state["drop_cols"], errors="ignore")

        for col in state["indicator_cols"]:
            if col in X.columns:
                X[f"{col}__MISSING"] = X[col].isna().astype("int8")

        for col in state["numeric_cols"]:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")

        for col in state["categorical_cols"]:
            if col in X.columns:
                values = X[col].astype("string").fillna("__MISSING__")
                X[col] = pd.Categorical(
                    values,
                    categories=state["category_vocab"][col],
                )

        X = X.reindex(columns=self.selected_features)

        # Reapply categorical dtype to preserve training vocabularies after reindex
        for col in state["categorical_cols"]:
            if col in X.columns:
                X[col] = pd.Categorical(
                    X[col].astype("string").fillna("__MISSING__"),
                    categories=state["category_vocab"][col],
                )

        return X

    def predict_raw_probability(self, raw_df: pd.DataFrame) -> float:
        """Inference step: Frozen XGBoost predict_proba."""
        X_ready = self.transform(raw_df)
        raw_prob = float(self.model.predict_proba(X_ready)[0, 1])
        return raw_prob

    def calibrate_probability(self, raw_prob: float) -> float:
        """Calibration step: Platt scaling logit transformation."""
        logit_val = safe_logit([raw_prob]).reshape(-1, 1)
        calibrated_prob = float(self.calibrator.predict_proba(logit_val)[0, 1])
        return calibrated_prob

    def predict_risk(self, raw_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Full Phase 1B inference pipeline:
        Raw application -> feature engineering -> preprocessing state -> frozen XGBoost
        -> raw probability -> calibrator -> calibrated probability -> risk band.
        """
        raw_prob = self.predict_raw_probability(raw_df)
        calibrated_prob = self.calibrate_probability(raw_prob)
        band = classify_risk_band(calibrated_prob, self.risk_policy)

        return {
            "default_probability": float(calibrated_prob),
            "raw_probability": float(raw_prob),
            "risk_band": band,
            "model_name": self.metadata.get("model_name", "XGBoost"),
            "model_version": self.metadata.get("model_version", "credit-xgb-v1.0.0"),
            "calibration_version": self.metadata.get("calibration_version", "credit-calibration-v1.0.0"),
            "policy_version": self.metadata.get("policy_version", "credit-risk-policy-v1.0.0"),
            "calibration_method": self.calibration_method,
            "feature_count": len(self.selected_features),
        }

    def predict_probability(self, raw_df: pd.DataFrame) -> float:
        """Returns the calibrated default probability (primary metric for Phase 1B)."""
        return self.predict_risk(raw_df)["default_probability"]

    def info(self) -> Dict[str, Any]:
        return {
            "model_name": self.metadata.get("model_name", "XGBoost"),
            "model_version": self.metadata.get("model_version", "credit-xgb-v1.0.0"),
            "calibration_version": self.metadata.get("calibration_version", "credit-calibration-v1.0.0"),
            "policy_version": self.metadata.get("policy_version", "credit-risk-policy-v1.0.0"),
            "feature_count": len(self.selected_features),
            "calibration_method": self.calibration_method,
            "risk_policy": self.risk_policy,
        }
