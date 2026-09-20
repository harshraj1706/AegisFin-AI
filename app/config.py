from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from dotenv import find_dotenv, load_dotenv

# Base directory of this backend package (AegisFin-AI-Phase1-Rich-Integration)
BASE_DIR = Path(__file__).resolve().parents[1]

# Discover and load .env file from current directory, parent dirs, or workspace root
_env_file: Optional[str] = find_dotenv(usecwd=True)
if not _env_file:
    for candidate in [
        BASE_DIR / ".env",
        BASE_DIR.parent / ".env",
        BASE_DIR.parent.parent / ".env",
    ]:
        if candidate.is_file():
            _env_file = str(candidate)
            break

if _env_file:
    load_dotenv(_env_file)

# Supabase Credentials (read from .env)
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_PUBLISHABLE_KEY: str = os.getenv("SUPABASE_PUBLISHABLE_KEY", "")
SUPABASE_SECRET_KEY: str = os.getenv("SUPABASE_SECRET_KEY", "")

# Backend Service Configuration
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Phase 1 Model Path (maintaining existing config pattern)
MODEL_PATH: Path = Path(
    os.getenv(
        "AEGISFIN_MODEL_PATH",
        BASE_DIR / "models" / "aegisfin_phase1b_calibrated_model.pkl",
    )
)

# Phase 2 Fraud Model Path
FRAUD_MODEL_PATH: Path = Path(
    os.getenv(
        "FRAUD_MODEL_PATH",
        BASE_DIR / "models" / "aegisfin_phase2_fraud_champion.pkl",
    )
)
if not FRAUD_MODEL_PATH.exists():
    for candidate in [
        BASE_DIR / "models" / "aegisfin_phase2_fraud_champion.pkl",
        BASE_DIR / "fraud_detection_file" / "aegisfin_phase2_fraud_champion.pkl",
    ]:
        if candidate.exists():
            FRAUD_MODEL_PATH = candidate
            break

# Phase 2 Feature List and Metadata Paths
FRAUD_FEATURE_LIST_PATH: Path = Path(
    os.getenv(
        "FRAUD_FEATURE_LIST_PATH",
        BASE_DIR / "fraud_detection_file" / "phase2_feature_list.csv",
    )
)
if not FRAUD_FEATURE_LIST_PATH.exists():
    for candidate in [
        BASE_DIR / "models" / "phase2_feature_list.csv",
        BASE_DIR / "backend" / "models" / "phase2_feature_list.csv",
    ]:
        if candidate.exists():
            FRAUD_FEATURE_LIST_PATH = candidate
            break

FRAUD_TRAINING_METADATA_PATH: Path = Path(
    os.getenv(
        "FRAUD_TRAINING_METADATA_PATH",
        BASE_DIR / "fraud_detection_file" / "phase2_training_metadata.json",
    )
)
if not FRAUD_TRAINING_METADATA_PATH.exists():
    for candidate in [
        BASE_DIR / "models" / "phase2_training_metadata.json",
        BASE_DIR / "backend" / "models" / "phase2_training_metadata.json",
    ]:
        if candidate.exists():
            FRAUD_TRAINING_METADATA_PATH = candidate
            break

