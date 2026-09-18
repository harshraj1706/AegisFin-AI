from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict
import pandas as pd

from .feature_engineering import engineer_features


class Phase1ModelService:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        with self.model_path.open("rb") as f:
            self.bundle: Dict[str, Any] = pickle.load(f)

        required = {
            "model",
            "preprocessing_state",
            "selected_features",
            "metadata",
        }
        missing = required - set(self.bundle)
        if missing:
            raise ValueError(
                f"Invalid Phase 1 model bundle. Missing keys: {sorted(missing)}"
            )

        self.model = self.bundle["model"]
        self.preprocessing_state = self.bundle["preprocessing_state"]
        self.selected_features = list(self.bundle["selected_features"])
        self.metadata = self.bundle["metadata"]

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
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

        # Recreate categorical dtype after reindex so missing columns preserve
        # the training representation.
        for col in state["categorical_cols"]:
            if col in X.columns:
                X[col] = pd.Categorical(
                    X[col].astype("string").fillna("__MISSING__"),
                    categories=state["category_vocab"][col],
                )

        return X

    def predict_probability(self, raw_df: pd.DataFrame) -> float:
        X = self.transform(raw_df)
        return float(self.model.predict_proba(X)[0, 1])

    def info(self) -> Dict[str, Any]:
        return {
            "model_name": self.metadata.get("model_name", "XGBoost"),
            "model_version": self.metadata.get("model_version", "phase1"),
            "feature_count": len(self.selected_features),
        }
