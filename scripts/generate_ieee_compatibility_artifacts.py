"""
scripts/generate_ieee_compatibility_artifacts.py

Generates docs/phase2_ieee_training_compatibility.csv
auditing all 62 VALID production features against IEEE-CIS columns.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.production_feature_definitions import VALID_PRODUCTION_FEATURES

DOCS_DIR = BASE_DIR / "docs"
CSV_PATH = DOCS_DIR / "phase2_ieee_training_compatibility.csv"

# Detailed mapping specifications for each of the 62 features
IEEE_AUDIT_MAPPINGS = {
    # ------------------------------------------------------------------------
    # Group 1: Direct Transaction Features (6)
    # ------------------------------------------------------------------------
    "amount": {
        "ieee_source": "TransactionAmt",
        "mapping_type": "Direct 1:1 Match",
        "trainable": "IEEE_TRAINABLE",
        "mapping_quality": "High",
        "reason": "Direct monetary transaction amount available in train_transaction.csv.",
    },
    "product_code_freq": {
        "ieee_source": "ProductCD",
        "mapping_type": "Direct Frequency Map",
        "trainable": "IEEE_TRAINABLE",
        "mapping_quality": "High",
        "reason": "Direct 1:1 match; ProductCD contains exact same categories ('W', 'H', 'C', 'S', 'R').",
    },
    "card_network_freq": {
        "ieee_source": "card4",
        "mapping_type": "Direct Frequency Map",
        "trainable": "IEEE_TRAINABLE",
        "mapping_quality": "High",
        "reason": "Direct 1:1 match; card4 contains payment schemes ('visa', 'mastercard', 'discover', 'american express').",
    },
    "card_type_freq": {
        "ieee_source": "card6",
        "mapping_type": "Direct Frequency Map",
        "trainable": "IEEE_TRAINABLE",
        "mapping_quality": "High",
        "reason": "Direct 1:1 match; card6 contains card funding types ('debit', 'credit').",
    },
    "email_domain_freq": {
        "ieee_source": "P_emaildomain",
        "mapping_type": "Direct Frequency Map",
        "trainable": "IEEE_TRAINABLE",
        "mapping_quality": "High",
        "reason": "Direct 1:1 match; P_emaildomain contains purchaser email domains ('gmail.com', 'yahoo.com', etc.).",
    },
    "missing_fields_count": {
        "ieee_source": "DeviceInfo, addr1, P_emaildomain, card4, card6",
        "mapping_type": "Partial Null Proxy",
        "trainable": "IEEE_AMBIGUOUS",
        "mapping_quality": "Medium",
        "reason": "Partial proxy; IEEE-CIS lacks merchant_id, true ip_address, and ISO country, so missing field counts reflect a different null-distribution.",
    },

    # ------------------------------------------------------------------------
    # Group 2: Time & Calendar Features (6)
    # ------------------------------------------------------------------------
    "hour": {
        "ieee_source": "TransactionDT",
        "mapping_type": "Modulo Derivation ((TransactionDT // 3600) % 24)",
        "trainable": "IEEE_TRAINABLE",
        "mapping_quality": "High",
        "reason": "TransactionDT is elapsed integer seconds; diurnal cycle modulo 3600 / 24 is exact and semantically sound.",
    },
    "weekday_index": {
        "ieee_source": "TransactionDT",
        "mapping_type": "Modulo Derivation (((TransactionDT // 86400) + offset) % 7)",
        "trainable": "IEEE_TRAINABLE",
        "mapping_quality": "High",
        "reason": "Day of week cycle modulo 86400 / 7 is exact and consistent.",
    },
    "is_weekend": {
        "ieee_source": "TransactionDT",
        "mapping_type": "Modulo Derivation (weekday in (5, 6))",
        "trainable": "IEEE_TRAINABLE",
        "mapping_quality": "High",
        "reason": "Binary weekend indicator derived deterministically from TransactionDT.",
    },
    "is_night": {
        "ieee_source": "TransactionDT",
        "mapping_type": "Modulo Derivation (hour < 6)",
        "trainable": "IEEE_TRAINABLE",
        "mapping_quality": "High",
        "reason": "Binary off-peak nocturnal indicator derived directly from TransactionDT hour.",
    },
    "day_of_year": {
        "ieee_source": "TransactionDT",
        "mapping_type": "Modulo Derivation (((TransactionDT // 86400) + offset) % 365 + 1)",
        "trainable": "IEEE_TRAINABLE",
        "mapping_quality": "High",
        "reason": "Ordinal day of year capturing macro-calendar progression in TransactionDT.",
    },
    "time_since_midnight_sec": {
        "ieee_source": "TransactionDT",
        "mapping_type": "Modulo Derivation (TransactionDT % 86400)",
        "trainable": "IEEE_TRAINABLE",
        "mapping_quality": "High",
        "reason": "Continuous diurnal seconds variable derived directly from TransactionDT.",
    },

    # ------------------------------------------------------------------------
    # Group 3: Amount Transformation Features (4)
    # ------------------------------------------------------------------------
    "amount_log": {
        "ieee_source": "TransactionAmt",
        "mapping_type": "Mathematical Transform (log1p(TransactionAmt))",
        "trainable": "IEEE_TRAINABLE",
        "mapping_quality": "High",
        "reason": "Deterministic mathematical transformation of TransactionAmt.",
    },
    "amount_cents": {
        "ieee_source": "TransactionAmt",
        "mapping_type": "Mathematical Transform (round(TransactionAmt % 1.0, 2))",
        "trainable": "IEEE_TRAINABLE",
        "mapping_quality": "High",
        "reason": "Deterministic fractional cents calculation from TransactionAmt.",
    },
    "is_round_amount": {
        "ieee_source": "TransactionAmt",
        "mapping_type": "Mathematical Transform (TransactionAmt >= 10 and % 10 == 0)",
        "trainable": "IEEE_TRAINABLE",
        "mapping_quality": "High",
        "reason": "Round denomination flag computed directly from TransactionAmt.",
    },
    "is_zero_cents": {
        "ieee_source": "TransactionAmt",
        "mapping_type": "Mathematical Transform (TransactionAmt % 1.0 == 0)",
        "trainable": "IEEE_TRAINABLE",
        "mapping_quality": "High",
        "reason": "Zero fractional cents flag computed directly from TransactionAmt.",
    },

    # ------------------------------------------------------------------------
    # Group 4: Customer Behavioral History (10)
    # ------------------------------------------------------------------------
    "customer_tx_count_5m": {
        "ieee_source": "card1 + addr1 + TransactionDT (pseudo-UID)",
        "mapping_type": "Point-in-time Pseudo-UID Heuristic (t_curr - 300s)",
        "trainable": "IEEE_AMBIGUOUS",
        "mapping_quality": "Low",
        "reason": "IEEE-CIS contains no customer_id; relies on synthetic pseudo-UID (card1 + addr1), conflating cardholders with cards.",
    },
    "customer_tx_count_1h": {
        "ieee_source": "card1 + addr1 + TransactionDT (pseudo-UID)",
        "mapping_type": "Point-in-time Pseudo-UID Heuristic (t_curr - 3600s)",
        "trainable": "IEEE_AMBIGUOUS",
        "mapping_quality": "Low",
        "reason": "IEEE-CIS contains no customer_id; relies on synthetic pseudo-UID heuristic.",
    },
    "customer_tx_count_24h": {
        "ieee_source": "card1 + addr1 + TransactionDT (pseudo-UID)",
        "mapping_type": "Point-in-time Pseudo-UID Heuristic (t_curr - 86400s)",
        "trainable": "IEEE_AMBIGUOUS",
        "mapping_quality": "Low",
        "reason": "IEEE-CIS contains no customer_id; relies on synthetic pseudo-UID heuristic.",
    },
    "customer_amount_mean": {
        "ieee_source": "card1 + addr1 + TransactionAmt (pseudo-UID)",
        "mapping_type": "Point-in-time Pseudo-UID Heuristic",
        "trainable": "IEEE_AMBIGUOUS",
        "mapping_quality": "Low",
        "reason": "IEEE-CIS contains no customer_id; historical average is computed on pseudo-UID.",
    },
    "customer_amount_std": {
        "ieee_source": "card1 + addr1 + TransactionAmt (pseudo-UID)",
        "mapping_type": "Point-in-time Pseudo-UID Heuristic",
        "trainable": "IEEE_AMBIGUOUS",
        "mapping_quality": "Low",
        "reason": "IEEE-CIS contains no customer_id; historical variance is computed on pseudo-UID.",
    },
    "customer_amount_zscore": {
        "ieee_source": "card1 + addr1 + TransactionAmt (pseudo-UID)",
        "mapping_type": "Point-in-time Pseudo-UID Heuristic",
        "trainable": "IEEE_AMBIGUOUS",
        "mapping_quality": "Low",
        "reason": "IEEE-CIS contains no customer_id; z-score reflects synthetic pseudo-UID history.",
    },
    "customer_amount_ratio": {
        "ieee_source": "card1 + addr1 + TransactionAmt (pseudo-UID)",
        "mapping_type": "Point-in-time Pseudo-UID Heuristic",
        "trainable": "IEEE_AMBIGUOUS",
        "mapping_quality": "Low",
        "reason": "IEEE-CIS contains no customer_id; ratio reflects synthetic pseudo-UID history.",
    },
    "customer_is_new": {
        "ieee_source": "card1 + addr1 + TransactionDT (pseudo-UID)",
        "mapping_type": "Point-in-time Pseudo-UID Heuristic",
        "trainable": "IEEE_AMBIGUOUS",
        "mapping_quality": "Low",
        "reason": "IEEE-CIS contains no customer_id; indicates first occurrence of synthetic pseudo-UID.",
    },
    "customer_unique_merchants_24h": {
        "ieee_source": "None (no customer_id, no merchant_id)",
        "mapping_type": "None",
        "trainable": "IEEE_UNAVAILABLE",
        "mapping_quality": "None",
        "reason": "IEEE-CIS contains neither a true customer identifier nor a merchant identifier; impossible to compute.",
    },
    "customer_unique_devices_24h": {
        "ieee_source": "pseudo-UID + DeviceInfo (train_identity.csv)",
        "mapping_type": "Point-in-time Low-Coverage Heuristic",
        "trainable": "IEEE_AMBIGUOUS",
        "mapping_quality": "Low",
        "reason": "DeviceInfo is only present for 24.4% of transactions and represents generic OS/browser family strings rather than unique hardware fingerprints.",
    },

    # ------------------------------------------------------------------------
    # Group 5: Card Behavioral History (7)
    # ------------------------------------------------------------------------
    "card_tx_count_5m": {
        "ieee_source": "card1 + TransactionDT",
        "mapping_type": "Point-in-time Entity Aggregation (t_curr - 300s)",
        "trainable": "IEEE_TRAINABLE",
        "mapping_quality": "High",
        "reason": "card1 is an authentic card issuer token; point-in-time 5m window is fully computable via TransactionDT.",
    },
    "card_tx_count_1h": {
        "ieee_source": "card1 + TransactionDT",
        "mapping_type": "Point-in-time Entity Aggregation (t_curr - 3600s)",
        "trainable": "IEEE_TRAINABLE",
        "mapping_quality": "High",
        "reason": "card1 is an authentic card issuer token; point-in-time 1h window is fully computable via TransactionDT.",
    },
    "card_tx_count_24h": {
        "ieee_source": "card1 + TransactionDT",
        "mapping_type": "Point-in-time Entity Aggregation (t_curr - 86400s)",
        "trainable": "IEEE_TRAINABLE",
        "mapping_quality": "High",
        "reason": "card1 is an authentic card issuer token; point-in-time 24h window is fully computable via TransactionDT.",
    },
    "card_amount_mean": {
        "ieee_source": "card1 + TransactionAmt + TransactionDT",
        "mapping_type": "Point-in-time Entity Aggregation",
        "trainable": "IEEE_TRAINABLE",
        "mapping_quality": "High",
        "reason": "Historical average spending for card1 strictly prior to TransactionDT.",
    },
    "card_amount_std": {
        "ieee_source": "card1 + TransactionAmt + TransactionDT",
        "mapping_type": "Point-in-time Entity Aggregation",
        "trainable": "IEEE_TRAINABLE",
        "mapping_quality": "High",
        "reason": "Historical standard deviation of spend on card1 strictly prior to TransactionDT.",
    },
    "card_amount_ratio": {
        "ieee_source": "card1 + TransactionAmt + TransactionDT",
        "mapping_type": "Point-in-time Entity Aggregation",
        "trainable": "IEEE_TRAINABLE",
        "mapping_quality": "High",
        "reason": "Ratio of current TransactionAmt over historical mean of card1.",
    },
    "card_is_new": {
        "ieee_source": "card1 + TransactionDT",
        "mapping_type": "Point-in-time Entity Aggregation",
        "trainable": "IEEE_TRAINABLE",
        "mapping_quality": "High",
        "reason": "Binary flag indicating first observation of card1 in dataset history.",
    },

    # ------------------------------------------------------------------------
    # Group 6: Device Fingerprint History (6)
    # ------------------------------------------------------------------------
    "device_tx_count_5m": {
        "ieee_source": "DeviceInfo + TransactionDT (train_identity.csv)",
        "mapping_type": "Low-Coverage Coarse Aggregate (t_curr - 300s)",
        "trainable": "IEEE_AMBIGUOUS",
        "mapping_quality": "Low",
        "reason": "DeviceInfo only present in 24.4% of rows; generic strings ('Windows') create massive spurious velocity spikes.",
    },
    "device_tx_count_1h": {
        "ieee_source": "DeviceInfo + TransactionDT (train_identity.csv)",
        "mapping_type": "Low-Coverage Coarse Aggregate (t_curr - 3600s)",
        "trainable": "IEEE_AMBIGUOUS",
        "mapping_quality": "Low",
        "reason": "DeviceInfo only present in 24.4% of rows; non-unique device families distort hourly counts.",
    },
    "device_tx_count_24h": {
        "ieee_source": "DeviceInfo + TransactionDT (train_identity.csv)",
        "mapping_type": "Low-Coverage Coarse Aggregate (t_curr - 86400s)",
        "trainable": "IEEE_AMBIGUOUS",
        "mapping_quality": "Low",
        "reason": "DeviceInfo only present in 24.4% of rows; non-unique device families distort 24h counts.",
    },
    "device_unique_customers_24h": {
        "ieee_source": "None (no true customer_id, non-unique DeviceInfo)",
        "mapping_type": "None",
        "trainable": "IEEE_UNAVAILABLE",
        "mapping_quality": "None",
        "reason": "IEEE-CIS contains neither true customer identifiers nor unique hardware IDs; impossible to compute device-sharing accurately.",
    },
    "device_unique_cards_24h": {
        "ieee_source": "DeviceInfo + card1 + TransactionDT",
        "mapping_type": "Low-Coverage Coarse Aggregate",
        "trainable": "IEEE_AMBIGUOUS",
        "mapping_quality": "Low",
        "reason": "Coarse DeviceInfo ('Windows') falsely aggregates thousands of unrelated cards under the same device bucket.",
    },
    "device_is_new": {
        "ieee_source": "DeviceInfo + TransactionDT",
        "mapping_type": "Low-Coverage String First-Seen",
        "trainable": "IEEE_AMBIGUOUS",
        "mapping_quality": "Low",
        "reason": "Coarse device strings ('iOS Device') become permanently 'old' after the first few thousand rows, zeroing out utility.",
    },

    # ------------------------------------------------------------------------
    # Group 7: Merchant Risk & History (6)
    # ------------------------------------------------------------------------
    "merchant_tx_count_1h": {
        "ieee_source": "None (no merchant_id)",
        "mapping_type": "None",
        "trainable": "IEEE_UNAVAILABLE",
        "mapping_quality": "None",
        "reason": "IEEE-CIS contains no merchant identifier or store ID column; cannot use ProductCD as a merchant proxy.",
    },
    "merchant_tx_count_24h": {
        "ieee_source": "None (no merchant_id)",
        "mapping_type": "None",
        "trainable": "IEEE_UNAVAILABLE",
        "mapping_quality": "None",
        "reason": "IEEE-CIS contains no merchant identifier or store ID column.",
    },
    "merchant_amount_mean": {
        "ieee_source": "None (no merchant_id)",
        "mapping_type": "None",
        "trainable": "IEEE_UNAVAILABLE",
        "mapping_quality": "None",
        "reason": "IEEE-CIS contains no merchant identifier or store ID column.",
    },
    "merchant_amount_std": {
        "ieee_source": "None (no merchant_id)",
        "mapping_type": "None",
        "trainable": "IEEE_UNAVAILABLE",
        "mapping_quality": "None",
        "reason": "IEEE-CIS contains no merchant identifier or store ID column.",
    },
    "customer_merchant_tx_count": {
        "ieee_source": "None (no customer_id, no merchant_id)",
        "mapping_type": "None",
        "trainable": "IEEE_UNAVAILABLE",
        "mapping_quality": "None",
        "reason": "Neither customer nor merchant exists in IEEE-CIS dataset.",
    },
    "customer_merchant_is_new": {
        "ieee_source": "None (no customer_id, no merchant_id)",
        "mapping_type": "None",
        "trainable": "IEEE_UNAVAILABLE",
        "mapping_quality": "None",
        "reason": "Neither customer nor merchant exists in IEEE-CIS dataset.",
    },

    # ------------------------------------------------------------------------
    # Group 8: Client IP Network History (7)
    # ------------------------------------------------------------------------
    "ip_tx_count_5m": {
        "ieee_source": "None (no IP address column)",
        "mapping_type": "None",
        "trainable": "IEEE_UNAVAILABLE",
        "mapping_quality": "None",
        "reason": "IEEE-CIS completely omitted IP addresses for privacy; impossible to calculate IP network velocity.",
    },
    "ip_tx_count_1h": {
        "ieee_source": "None (no IP address column)",
        "mapping_type": "None",
        "trainable": "IEEE_UNAVAILABLE",
        "mapping_quality": "None",
        "reason": "IEEE-CIS completely omitted IP addresses for privacy.",
    },
    "ip_tx_count_24h": {
        "ieee_source": "None (no IP address column)",
        "mapping_type": "None",
        "trainable": "IEEE_UNAVAILABLE",
        "mapping_quality": "None",
        "reason": "IEEE-CIS completely omitted IP addresses for privacy.",
    },
    "ip_unique_customers_24h": {
        "ieee_source": "None (no IP address, no customer_id)",
        "mapping_type": "None",
        "trainable": "IEEE_UNAVAILABLE",
        "mapping_quality": "None",
        "reason": "Neither IP address nor customer_id exists in IEEE-CIS.",
    },
    "ip_unique_cards_24h": {
        "ieee_source": "None (no IP address column)",
        "mapping_type": "None",
        "trainable": "IEEE_UNAVAILABLE",
        "mapping_quality": "None",
        "reason": "No IP address column available to calculate distinct payment cards tested.",
    },
    "ip_unique_devices_24h": {
        "ieee_source": "None (no IP address column)",
        "mapping_type": "None",
        "trainable": "IEEE_UNAVAILABLE",
        "mapping_quality": "None",
        "reason": "No IP address column available in IEEE-CIS dataset.",
    },
    "ip_is_new": {
        "ieee_source": "None (no IP address column)",
        "mapping_type": "None",
        "trainable": "IEEE_UNAVAILABLE",
        "mapping_quality": "None",
        "reason": "No IP address column available in IEEE-CIS dataset.",
    },

    # ------------------------------------------------------------------------
    # Group 9: Cross-Entity Relationship Features (6)
    # ------------------------------------------------------------------------
    "customer_device_seen_before": {
        "ieee_source": "pseudo-UID + DeviceInfo (24.4% coverage)",
        "mapping_type": "Low-Coverage Heuristic Co-occurrence",
        "trainable": "IEEE_AMBIGUOUS",
        "mapping_quality": "Low",
        "reason": "Relies on pseudo-UID and coarse DeviceInfo; missing on 75.6% of transactions.",
    },
    "customer_card_seen_before": {
        "ieee_source": "pseudo-UID + card1",
        "mapping_type": "Tautological Heuristic",
        "trainable": "IEEE_AMBIGUOUS",
        "mapping_quality": "Low",
        "reason": "Because pseudo-UID is partially defined by card1, customer-card co-occurrence becomes largely tautological.",
    },
    "customer_merchant_seen_before": {
        "ieee_source": "None (no merchant_id)",
        "mapping_type": "None",
        "trainable": "IEEE_UNAVAILABLE",
        "mapping_quality": "None",
        "reason": "No merchant_id column exists in IEEE-CIS.",
    },
    "customer_ip_seen_before": {
        "ieee_source": "None (no ip_address)",
        "mapping_type": "None",
        "trainable": "IEEE_UNAVAILABLE",
        "mapping_quality": "None",
        "reason": "No ip_address column exists in IEEE-CIS.",
    },
    "card_device_seen_before": {
        "ieee_source": "card1 + DeviceInfo (24.4% coverage)",
        "mapping_type": "Low-Coverage Co-occurrence",
        "trainable": "IEEE_AMBIGUOUS",
        "mapping_quality": "Low",
        "reason": "DeviceInfo only present in 24.4% of transactions; non-unique device strings create false co-occurrence.",
    },
    "card_ip_seen_before": {
        "ieee_source": "None (no ip_address)",
        "mapping_type": "None",
        "trainable": "IEEE_UNAVAILABLE",
        "mapping_quality": "None",
        "reason": "No ip_address column exists in IEEE-CIS.",
    },

    # ------------------------------------------------------------------------
    # Group 10: Velocity & Burst Features (4)
    # ------------------------------------------------------------------------
    "rapid_transaction_flag": {
        "ieee_source": "pseudo-UID + TransactionDT",
        "mapping_type": "Pseudo-UID Temporal Delta (delta_sec <= 60s)",
        "trainable": "IEEE_AMBIGUOUS",
        "mapping_quality": "Low",
        "reason": "Calculable using TransactionDT seconds delta, but completely reliant on accuracy of synthetic pseudo-UID grouping.",
    },
    "transactions_last_15m": {
        "ieee_source": "pseudo-UID + TransactionDT",
        "mapping_type": "Pseudo-UID Temporal Window (t_curr - 900s)",
        "trainable": "IEEE_AMBIGUOUS",
        "mapping_quality": "Low",
        "reason": "Calculable using TransactionDT window, but reliant on synthetic pseudo-UID grouping.",
    },
    "amount_sum_last_1h": {
        "ieee_source": "pseudo-UID + TransactionAmt + TransactionDT",
        "mapping_type": "Pseudo-UID Temporal Sum (t_curr - 3600s)",
        "trainable": "IEEE_AMBIGUOUS",
        "mapping_quality": "Low",
        "reason": "Calculable using TransactionDT window, but reliant on synthetic pseudo-UID grouping.",
    },
    "amount_sum_last_24h": {
        "ieee_source": "pseudo-UID + TransactionAmt + TransactionDT",
        "mapping_type": "Pseudo-UID Temporal Sum (t_curr - 86400s)",
        "trainable": "IEEE_AMBIGUOUS",
        "mapping_quality": "Low",
        "reason": "Calculable using TransactionDT window, but reliant on synthetic pseudo-UID grouping.",
    },
}


def generate_ieee_compatibility_csv() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature_name",
        "production_group",
        "ieee_source",
        "mapping_type",
        "trainable",
        "mapping_quality",
        "reason",
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for feat in VALID_PRODUCTION_FEATURES:
            fname = feat.feature_name
            meta = IEEE_AUDIT_MAPPINGS.get(fname, {
                "ieee_source": "Unknown",
                "mapping_type": "Unknown",
                "trainable": "IEEE_UNAVAILABLE",
                "mapping_quality": "None",
                "reason": "Not mapped",
            })
            writer.writerow({
                "feature_name": fname,
                "production_group": feat.group,
                "ieee_source": meta["ieee_source"],
                "mapping_type": meta["mapping_type"],
                "trainable": meta["trainable"],
                "mapping_quality": meta["mapping_quality"],
                "reason": meta["reason"],
            })

    print(f"Generated IEEE compatibility CSV at: {CSV_PATH} with {len(VALID_PRODUCTION_FEATURES)} rows.")


if __name__ == "__main__":
    generate_ieee_compatibility_csv()
