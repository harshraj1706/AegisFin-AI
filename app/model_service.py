from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import pandas as pd

from .risk_service import Phase1RiskService, classify_risk_band, safe_logit


class Phase1ModelService(Phase1RiskService):
    """
    Backwards-compatible alias for Phase1RiskService.
    Preserves existing API signatures while using the Phase 1B canonical pipeline.
    """

    def predict_probability(self, raw_df: pd.DataFrame) -> float:
        """Returns the calibrated default probability from Phase 1B."""
        return self.predict_risk(raw_df)["default_probability"]
