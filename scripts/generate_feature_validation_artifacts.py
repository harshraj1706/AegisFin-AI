"""
scripts/generate_feature_validation_artifacts.py

Generates:
- docs/phase2_production_feature_validation.csv
- docs/phase2_production_feature_schema.json

directly from app.production_feature_definitions (Single Source of Truth).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.production_feature_definitions import (
    ALL_AUDITED_FEATURES,
    VALID_PRODUCTION_FEATURES,
    NEEDS_DATA_FEATURES,
    INVALID_FEATURES,
)

DOCS_DIR = BASE_DIR / "docs"
CSV_PATH = DOCS_DIR / "phase2_production_feature_validation.csv"
JSON_PATH = DOCS_DIR / "phase2_production_feature_schema.json"


def generate_validation_csv() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature_name",
        "group",
        "training_source",
        "live_source",
        "training_formula",
        "live_formula",
        "lookback_window",
        "leakage_rule",
        "required_raw_fields",
        "required_supabase_fields",
        "status",
        "reason",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for feat in ALL_AUDITED_FEATURES:
            writer.writerow({
                "feature_name": feat.feature_name,
                "group": feat.group,
                "training_source": feat.training_source,
                "live_source": feat.live_source,
                "training_formula": feat.training_formula,
                "live_formula": feat.live_formula,
                "lookback_window": feat.lookback_window or "None",
                "leakage_rule": feat.leakage_rule,
                "required_raw_fields": ", ".join(feat.required_raw_fields),
                "required_supabase_fields": ", ".join(feat.required_supabase_fields),
                "status": feat.production_status,
                "reason": feat.reason,
            })
    print(f"Generated validation CSV at: {CSV_PATH} with {len(ALL_AUDITED_FEATURES)} rows.")


def generate_production_schema_json() -> None:
    # Group the valid features by functional group
    groups_dict: Dict[str, Any] = {}
    for feat in VALID_PRODUCTION_FEATURES:
        grp = feat.group
        if grp not in groups_dict:
            groups_dict[grp] = []
        groups_dict[grp].append({
            "name": feat.feature_name,
            "data_type": feat.data_type,
            "training_source": feat.training_source,
            "live_source": feat.live_source,
            "lookback_window": feat.lookback_window,
            "leakage_rule": feat.leakage_rule,
            "formula": feat.live_formula,
            "nullable_behavior": feat.nullable_behavior,
            "required_raw_fields": feat.required_raw_fields,
            "required_supabase_fields": feat.required_supabase_fields,
        })

    schema_data = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "AegisFin Phase 2 Validated Production Fraud Feature Schema",
        "version": "2.1.0",
        "description": "Validated, production-compatible feature engineering contract for AegisFin Phase 2 Fraud Detection. Eliminates legacy anonymous IEEE-CIS V-features and unverified signals, establishing an authentic, reproducible schema combining live transaction attributes and Supabase point-in-time entity history.",
        "metadata": {
            "total_candidate_features_audited": len(ALL_AUDITED_FEATURES),
            "audit_breakdown": {
                "VALID": {
                    "count": len(VALID_PRODUCTION_FEATURES),
                    "description": "Fully reproducible identically during both historical training and live inference.",
                    "status": "Production Candidate Feature Suite",
                },
                "NEEDS_DATA": {
                    "count": len(NEEDS_DATA_FEATURES),
                    "description": "Valuable fraud risk features that currently lack raw input fields, Supabase schema support, or third-party APIs.",
                    "status": "Deferred to Phase 2.5 Enrichment",
                },
                "INVALID": {
                    "count": len(INVALID_FEATURES),
                    "description": "Features rejected due to mathematical instability, alphanumeric casting fragility, or ambiguous multi-entity collinearity.",
                    "status": "Permanently Excluded",
                },
            },
            "final_production_feature_count": len(VALID_PRODUCTION_FEATURES),
            "anti_leakage_guarantee": "Strict point-in-time temporal filtering: historical_timestamp < current_transaction_timestamp enforced on all historical queries.",
        },
        "raw_input_fields": [
            {"name": "transaction_id", "type": "string", "required": True, "description": "Globally unique payment transaction reference identifier."},
            {"name": "amount", "type": "number", "required": True, "minimum": 0.0, "description": "Transaction monetary value in USD."},
            {"name": "transaction_timestamp", "type": "string", "format": "date-time", "required": False, "default": "current_utc_time", "description": "ISO-8601 UTC timestamp of the transaction event."},
            {"name": "customer_id", "type": "string", "required": True, "description": "Unique identifier of the account holder or customer."},
            {"name": "card_id", "type": "string", "required": True, "description": "Payment card reference / card1 token identifier."},
            {"name": "device_id", "type": "string", "required": False, "description": "Client hardware fingerprint or device ID string."},
            {"name": "merchant_id", "type": "string", "required": False, "description": "Merchant or POS terminal reference identifier."},
            {"name": "product_code", "type": "string", "enum": ["W", "H", "C", "S", "R"], "default": "W", "description": "Transaction category product code (W=Web, H=Home, C=Commercial, S=Specialty, R=Retail)."},
            {"name": "card_network", "type": "string", "enum": ["visa", "mastercard", "discover", "american express"], "default": "visa", "description": "Card payment network scheme."},
            {"name": "card_type", "type": "string", "enum": ["debit", "credit"], "default": "debit", "description": "Card funding type."},
            {"name": "email_domain", "type": "string", "required": False, "description": "Purchaser email domain (e.g. gmail.com, yahoo.com)."},
            {"name": "address_id", "type": "string", "required": False, "description": "Billing zip code or location identifier."},
            {"name": "ip_address", "type": "string", "required": False, "description": "Client IPv4 or IPv6 address initiating transaction."},
            {"name": "country", "type": "string", "required": False, "description": "Two-letter ISO country code of transaction origin."},
        ],
        "valid_production_feature_groups": groups_dict,
        "needs_data_features": [
            {
                "feature_name": f.feature_name,
                "group": f.group,
                "missing_dependency": f.reason,
                "required_raw_fields": f.required_raw_fields,
                "required_supabase_fields": f.required_supabase_fields,
            }
            for f in NEEDS_DATA_FEATURES
        ],
        "invalid_features": [
            {
                "feature_name": f.feature_name,
                "group": f.group,
                "rejection_reason": f.reason,
            }
            for f in INVALID_FEATURES
        ],
    }

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(schema_data, f, indent=2)
    print(f"Generated production schema JSON at: {JSON_PATH}")


if __name__ == "__main__":
    generate_validation_csv()
    generate_production_schema_json()
