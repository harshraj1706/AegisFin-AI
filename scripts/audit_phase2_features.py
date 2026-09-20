"""
scripts/audit_phase2_features.py

Audits all 459 features of the AegisFin Phase 2 Fraud champion model artifact:
- Extracts feature names, tree importance weights, and relative importance %.
- Classifies each feature into Category A, B, C, or D.
- Determines live field mappings, derivations, production viability, and rationale.
- Emits docs/phase2_feature_mapping.csv.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "aegisfin_phase2_fraud_champion.pkl"
FEATURE_LIST_PATH = BASE_DIR / "fraud_detection_file" / "phase2_feature_list.csv"
OUTPUT_CSV_PATH = BASE_DIR / "docs" / "phase2_feature_mapping.csv"


def classify_feature(feature_name: str) -> Tuple[str, str, str, str, bool, str]:
    """
    Classifies a feature into:
      category: 'A' | 'B' | 'C' | 'D'
      source: Data origin
      live_field: Input parameter or database column
      derivation: Exact mathematical or lookup derivation
      production_candidate: Boolean indicating if it belongs in future production schema
      reason: Detailed rationale
    """
    f = feature_name

    # 1. Category B: Derivable from Supabase Transaction & Entity History (15 current features)
    if any(k in f for k in ["uid_past_", "uid_amount_ratio", "uid_is_new"]):
        derivation_map = {
            "uid_past_count": "COUNT(transactions) where customer_id = current AND ts < current_ts",
            "uid_past_mean_amt": "AVG(amount) where customer_id = current AND ts < current_ts",
            "uid_past_std_amt": "STDDEV(amount) where customer_id = current AND ts < current_ts",
            "uid_amount_ratio": "amount / (uid_past_mean_amt + 1e-5)",
            "uid_is_new": "1.0 if uid_past_count == 0 else 0.0",
        }
        return (
            "B",
            "Supabase History (fraud_transactions / fraud_entity_state)",
            "customer_id, amount, transaction_timestamp",
            derivation_map.get(f, "Anti-leakage customer aggregate"),
            True,
            "Directly derivable from point-in-time Supabase history with strict anti-leakage temporal filter (ts < current_ts).",
        )

    if any(k in f for k in ["card1_past_", "card1_amount_ratio", "card1_is_new"]):
        derivation_map = {
            "card1_past_count": "COUNT(transactions) where card_id = current AND ts < current_ts",
            "card1_past_mean_amt": "AVG(amount) where card_id = current AND ts < current_ts",
            "card1_past_std_amt": "STDDEV(amount) where card_id = current AND ts < current_ts",
            "card1_amount_ratio": "amount / (card1_past_mean_amt + 1e-5)",
            "card1_is_new": "1.0 if card1_past_count == 0 else 0.0",
        }
        return (
            "B",
            "Supabase History (fraud_transactions / fraud_entity_state)",
            "card_id, amount, transaction_timestamp",
            derivation_map.get(f, "Anti-leakage card aggregate"),
            True,
            "Directly derivable from point-in-time Supabase card history with strict anti-leakage temporal filter.",
        )

    if any(k in f for k in ["card1_addr1_past_", "card1_addr1_amount_ratio", "card1_addr1_is_new"]):
        derivation_map = {
            "card1_addr1_past_count": "COUNT(transactions) where card_id = current AND address_id = current AND ts < current_ts",
            "card1_addr1_past_mean_amt": "AVG(amount) where card_id = current AND address_id = current AND ts < current_ts",
            "card1_addr1_past_std_amt": "STDDEV(amount) where card_id = current AND address_id = current AND ts < current_ts",
            "card1_addr1_amount_ratio": "amount / (card1_addr1_past_mean_amt + 1e-5)",
            "card1_addr1_is_new": "1.0 if card1_addr1_past_count == 0 else 0.0",
        }
        return (
            "B",
            "Supabase History (fraud_transactions / fraud_entity_state)",
            "card_id, address_id, amount, transaction_timestamp",
            derivation_map.get(f, "Anti-leakage card-address aggregate"),
            True,
            "Directly derivable from point-in-time Supabase card+address history with strict anti-leakage filter.",
        )

    # 2. Category A: Directly Available from Current AegisFin Transaction Input
    if f in [
        "TransactionAmt",
        "TransactionAmt_log",
        "TransactionAmt_cents",
        "TransactionDT",
        "hour",
        "weekday_index",
        "is_weekend",
        "is_night",
        "day_index",
        "card1",
        "addr1",
        "missing_count",
    ]:
        derivation_map = {
            "TransactionAmt": "amount (float)",
            "TransactionAmt_log": "log1p(amount)",
            "TransactionAmt_cents": "round(amount % 1, 2)",
            "TransactionDT": "int(dt.timestamp())",
            "hour": "float(dt.hour)",
            "weekday_index": "float(dt.weekday())",
            "is_weekend": "1.0 if weekday in (5, 6) else 0.0",
            "is_night": "1.0 if hour < 6 else 0.0",
            "day_index": "float(dt.timetuple().tm_yday)",
            "card1": "float(card_id) with training median fallback",
            "addr1": "float(address_id) with training median fallback",
            "missing_count": "Total count of unprovided/missing input fields",
        }
        live_field_map = {
            "TransactionAmt": "amount",
            "TransactionAmt_log": "amount",
            "TransactionAmt_cents": "amount",
            "TransactionDT": "transaction_timestamp",
            "hour": "transaction_timestamp",
            "weekday_index": "transaction_timestamp",
            "is_weekend": "transaction_timestamp",
            "is_night": "transaction_timestamp",
            "day_index": "transaction_timestamp",
            "card1": "card_id",
            "addr1": "address_id",
            "missing_count": "transaction payload completeness",
        }
        return (
            "A",
            "Live Transaction Input",
            live_field_map[f],
            derivation_map[f],
            True,
            "Directly provided in live transaction API request or deterministically transformed from payload.",
        )

    # Frequency-encoded features mapping directly to current live inputs
    if f in [
        "ProductCD__freq",
        "card4__freq",
        "card6__freq",
        "P_emaildomain__freq",
        "DeviceInfo__freq",
        "DeviceType__freq",
        "card1_ProductCD__freq",
        "card1_email__freq",
        "card1_addr1__freq",
        "uid__freq",
    ]:
        freq_field_map = {
            "ProductCD__freq": ("product_code", "Frequency lookup of product_code in training frequency map"),
            "card4__freq": ("card_network", "Frequency lookup of card_network (visa, mastercard, etc.)"),
            "card6__freq": ("card_type", "Frequency lookup of card_type (debit, credit)"),
            "P_emaildomain__freq": ("email_domain", "Frequency lookup of purchaser email domain"),
            "DeviceInfo__freq": ("device_id", "Frequency lookup of device_id string"),
            "DeviceType__freq": ("device_id / device_type", "Frequency lookup of device type category"),
            "card1_ProductCD__freq": ("card_id, product_code", "Frequency lookup of tuple key f'{card_id}_{product_code}'"),
            "card1_email__freq": ("card_id, email_domain", "Frequency lookup of tuple key f'{card_id}_{email_domain}'"),
            "card1_addr1__freq": ("card_id, address_id", "Frequency lookup of tuple key f'{card_id}_{address_id}'"),
            "uid__freq": ("product_code, card_id, address_id", "Frequency lookup of composite entity UID key"),
        }
        live_f, deriv = freq_field_map[f]
        return (
            "A",
            "Live Transaction Input + Precomputed Map",
            live_f,
            deriv,
            True,
            "Mapped directly from current live input and encoded via training artifact frequency map.",
        )

    # 3. Category C: Potentially Available if AegisFin Adds New Input / Upstream Fields
    # D-timedeltas (D1-D15): days since events (registration, card issue, etc.)
    if f.startswith("D") and f[1:].isdigit():
        return (
            "C",
            "Customer / Account Lifecycle (Future Feed)",
            "None (Requires account_created_at, card_issued_at, or bank authorization timedelta)",
            f"Calendar days elapsed between reference event and transaction timestamp for {f}",
            False,
            "IEEE-CIS timedelta days. Can be computed in future if AegisFin captures account opening date and card issue timestamp.",
        )

    # Secondary card & address attributes
    if f in ["card2", "card3", "card5", "addr2", "dist1", "dist2", "R_emaildomain__freq"]:
        c_map = {
            "card2": ("card_id (BIN lookup)", "Bank identification sub-code from card BIN table"),
            "card3": ("card_id (BIN lookup)", "Card issuing bank country code from BIN lookup"),
            "card5": ("card_id (BIN lookup)", "Card category / product code from BIN database"),
            "addr2": ("billing_country / shipping_country", "Country or state code of billing address"),
            "dist1": ("billing_zip, shipping_zip", "Geographic distance between billing and delivery address"),
            "dist2": ("ip_address, billing_zip", "Geographic distance between client IP geo-coordinates and billing zip"),
            "R_emaildomain__freq": ("recipient_email_domain", "Frequency lookup of recipient email domain in P2P/transfer flows"),
        }
        live_f, deriv = c_map[f]
        return (
            "C",
            "External Enrichment / Future Input Field",
            live_f,
            deriv,
            False,
            "Feasible if AegisFin adds third-party BIN lookup service, shipping address, or recipient domain input.",
        )

    # Browser & Device telemetry fields (id_30 through id_38)
    if f in [
        "id_30__freq",
        "id_31__freq",
        "id_32",
        "id_33__freq",
        "id_34__freq",
        "id_35__freq",
        "id_36__freq",
        "id_37__freq",
        "id_38__freq",
    ]:
        return (
            "C",
            "Client Telemetry / Browser Fingerprinting (Future SDK)",
            "None (Requires Web/Mobile SDK telemetry: User-Agent, screen resolution, browser headers)",
            f"Telemetry extraction for client property {f.replace('__freq', '')}",
            False,
            "Can be captured if AegisFin integrates frontend device fingerprinting (e.g. FingerprintJS) to collect browser & OS attributes.",
        )

    # 4. Category D: Impossible / Unreliable to Reproduce from Current System
    # V-features: V1 through V339 (339 features)
    if f.startswith("V") and f[1:].isdigit():
        return (
            "D",
            "Proprietary Vesta Internal Consortium (Obfuscated)",
            "None (No live upstream feed exists)",
            "Anonymous multi-entity consortium aggregations computed by Vesta internally",
            False,
            "Anonymous V-features with undisclosed proprietary formulas. Cannot be reproduced reliably in production; must NOT be fabricated or median-imputed.",
        )

    # C-features: C1 through C14 (14 features)
    if f.startswith("C") and f[1:].isdigit():
        return (
            "D",
            "Proprietary IEEE-CIS Consortium Counts (Obfuscated)",
            "None (No live upstream feed exists)",
            "Consortium-level entity count features across undisclosed payment networks",
            False,
            "Anonymous consortium counts with undocumented definitions. Unreliable for live production; should be replaced with explicit Supabase entity metrics.",
        )

    # M-features: M1__freq through M9__freq (9 features)
    if f.startswith("M") and f.endswith("__freq"):
        return (
            "D",
            "Payment Gateway Match Flags (AVS / 3DS Verification)",
            "None (Requires raw AVS / 3DS gateway response codes)",
            f"Match indicator {f.split('__')[0]} (e.g. cardholder name match, billing address match)",
            False,
            "Obfuscated match indicators from card processor. Unavailable without dedicated merchant processor integration.",
        )

    # id_01 through id_29 (numeric & categorical identity flags)
    if f.startswith("id_"):
        return (
            "D",
            "Proprietary IEEE-CIS Identity Risk Signals (Obfuscated)",
            "None (No live upstream feed exists)",
            f"Obfuscated fraud/risk score or flag {f}",
            False,
            "Anonymous identity scoring features from IEEE-CIS identity table with no real-world schema documentation.",
        )

    # Default fallback if any unclassified
    return (
        "D",
        "Unknown Legacy Benchmark Feature",
        "None",
        "None",
        False,
        "Legacy IEEE-CIS artifact feature not supported in production deployment.",
    )


def audit_features():
    print(f"Loading champion model artifact from {MODEL_PATH}...")
    art = joblib.load(MODEL_PATH)
    model = art["model"]
    feature_names = art["feature_names"]
    expected_count = len(feature_names)
    print(f"Total features in artifact: {expected_count}")

    # Extract feature importances
    importances = model.feature_importances_
    total_imp = float(np.sum(importances))
    imp_percentages = (importances / (total_imp if total_imp > 0 else 1.0)) * 100.0

    rows = []
    category_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    category_importances = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}

    for idx, f in enumerate(feature_names):
        imp = float(importances[idx])
        imp_pct = float(imp_percentages[idx])
        cat, source, live_field, deriv, is_prod, reason = classify_feature(f)

        category_counts[cat] += 1
        category_importances[cat] += imp_pct

        rows.append({
            "feature_name": f,
            "importance": round(imp, 8),
            "importance_percent": round(imp_pct, 6),
            "category": cat,
            "source": source,
            "live_field": live_field,
            "derivation": deriv,
            "production_candidate": is_prod,
            "reason": reason,
        })

    df = pd.DataFrame(rows)
    # Sort by importance descending
    df_sorted = df.sort_values(by="importance", ascending=False)

    # Save CSV
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved feature mapping to {OUTPUT_CSV_PATH} ({len(df)} rows)")

    print("\n" + "=" * 60)
    print("AUDIT SUMMARY:")
    print("=" * 60)
    for cat in ["A", "B", "C", "D"]:
        cnt = category_counts[cat]
        pct = (cnt / expected_count) * 100.0
        imp_sum = category_importances[cat]
        print(f"Category {cat}: {cnt:3d} features ({pct:5.2f}%) | Cumulative Tree Importance: {imp_sum:6.2f}%")

    prod_candidates = df[df["production_candidate"] == True]
    print(f"\nTotal Current Features Recommended as Production Candidates: {len(prod_candidates)}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    audit_features()
