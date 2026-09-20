"""
app/production_feature_definitions.py

AegisFin Phase 2 Fraud Detection: Single Source of Truth for Production Features.

Defines the formal feature contract for all 78 audited candidate features,
classifying each into:
- VALID_PRODUCTION_FEATURES (58 features)
- NEEDS_DATA_FEATURES (16 features)
- INVALID_FEATURES (4 features)

Provides unified, deterministic feature generators for both:
- training_feature_generator(history, current_transaction)
- live_feature_generator(history, current_transaction)

Strict anti-leakage guarantee:
All historical aggregates strictly enforce: historical_timestamp < current_transaction_timestamp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ============================================================================
# 1. FORMAL FEATURE CONTRACT SPECIFICATION
# ============================================================================

@dataclass(frozen=True)
class FeatureContract:
    """
    Formal specification contract for an individual feature candidate.
    """
    feature_name: str
    group: str
    data_type: str  # "float", "int", "bool"
    training_source: str
    live_source: str
    training_formula: str
    live_formula: str
    lookback_window: Optional[str]  # e.g., None, "5m", "15m", "1h", "24h", "lifetime"
    leakage_rule: str
    nullable_behavior: str
    required_raw_fields: List[str]
    required_supabase_fields: List[str]
    production_status: str  # Exactly "VALID", "NEEDS_DATA", or "INVALID"
    reason: str


# ============================================================================
# 2. DETERMINISTIC PRECOMPUTED FREQUENCY LOOKUPS
# ============================================================================

# Calibrated category frequencies derived from historical distribution
DEFAULT_PRODUCT_FREQ = {
    "W": 0.7432,
    "H": 0.0558,
    "C": 0.1165,
    "S": 0.0197,
    "R": 0.0648,
}

DEFAULT_CARD_NETWORK_FREQ = {
    "visa": 0.6512,
    "mastercard": 0.3205,
    "discover": 0.0115,
    "american express": 0.0168,
}

DEFAULT_CARD_TYPE_FREQ = {
    "debit": 0.7425,
    "credit": 0.2575,
}

DEFAULT_EMAIL_DOMAIN_FREQ = {
    "gmail.com": 0.3850,
    "yahoo.com": 0.1690,
    "hotmail.com": 0.0765,
    "anonymous.com": 0.0620,
    "aol.com": 0.0475,
    "outlook.com": 0.0160,
    "comcast.net": 0.0135,
    "icloud.com": 0.0105,
}


def parse_utc_timestamp(ts_val: Any) -> datetime:
    """Safely parses timestamps to timezone-aware UTC datetime."""
    if isinstance(ts_val, datetime):
        if ts_val.tzinfo is None:
            return ts_val.replace(tzinfo=timezone.utc)
        return ts_val.astimezone(timezone.utc)
    if isinstance(ts_val, (int, float)):
        return datetime.fromtimestamp(float(ts_val), tz=timezone.utc)
    if isinstance(ts_val, str):
        cleaned = ts_val.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(cleaned)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)


# ============================================================================
# 3. CATALOG OF ALL 78 AUDITED CANDIDATE FEATURES
# ============================================================================

ALL_AUDITED_FEATURES: List[FeatureContract] = [
    # ------------------------------------------------------------------------
    # Group 1: Direct Transaction Features (8 candidates)
    # ------------------------------------------------------------------------
    FeatureContract(
        feature_name="amount",
        group="Direct Transaction Features",
        data_type="float",
        training_source="raw_transaction.amount",
        live_source="FastAPI request.amount",
        training_formula="float(amount)",
        live_formula="float(amount)",
        lookback_window=None,
        leakage_rule="Intrinsic current transaction attribute; no leakage possible",
        nullable_behavior="Required field; minimum 0.0",
        required_raw_fields=["amount"],
        required_supabase_fields=[],
        production_status="VALID",
        reason="Direct monetary value present in both training data and live API.",
    ),
    FeatureContract(
        feature_name="product_code_freq",
        group="Direct Transaction Features",
        data_type="float",
        training_source="raw_transaction.product_code",
        live_source="FastAPI request.product_code",
        training_formula="frequency_map.get(product_code, 0.01)",
        live_formula="frequency_map.get(product_code, 0.01)",
        lookback_window=None,
        leakage_rule="Intrinsic current transaction attribute mapped via frozen frequency dictionary",
        nullable_behavior="Defaults to 'W' frequency (0.7432) if missing",
        required_raw_fields=["product_code"],
        required_supabase_fields=[],
        production_status="VALID",
        reason="Deterministic frequency mapping using precomputed category distribution.",
    ),
    FeatureContract(
        feature_name="card_network_freq",
        group="Direct Transaction Features",
        data_type="float",
        training_source="raw_transaction.card_network",
        live_source="FastAPI request.card_network",
        training_formula="frequency_map.get(card_network.lower(), 0.01)",
        live_formula="frequency_map.get(card_network.lower(), 0.01)",
        lookback_window=None,
        leakage_rule="Intrinsic current transaction attribute mapped via frozen frequency dictionary",
        nullable_behavior="Defaults to 'visa' frequency (0.6512) if missing",
        required_raw_fields=["card_network"],
        required_supabase_fields=[],
        production_status="VALID",
        reason="Deterministic frequency mapping for payment schemes.",
    ),
    FeatureContract(
        feature_name="card_type_freq",
        group="Direct Transaction Features",
        data_type="float",
        training_source="raw_transaction.card_type",
        live_source="FastAPI request.card_type",
        training_formula="frequency_map.get(card_type.lower(), 0.01)",
        live_formula="frequency_map.get(card_type.lower(), 0.01)",
        lookback_window=None,
        leakage_rule="Intrinsic current transaction attribute mapped via frozen frequency dictionary",
        nullable_behavior="Defaults to 'debit' frequency (0.7425) if missing",
        required_raw_fields=["card_type"],
        required_supabase_fields=[],
        production_status="VALID",
        reason="Deterministic frequency mapping for card funding types (debit/credit).",
    ),
    FeatureContract(
        feature_name="email_domain_freq",
        group="Direct Transaction Features",
        data_type="float",
        training_source="raw_transaction.email_domain",
        live_source="FastAPI request.email_domain",
        training_formula="frequency_map.get(email_domain.lower(), 0.01)",
        live_formula="frequency_map.get(email_domain.lower(), 0.01)",
        lookback_window=None,
        leakage_rule="Intrinsic current transaction attribute mapped via frozen frequency dictionary",
        nullable_behavior="Defaults to fallback frequency (0.01) if missing",
        required_raw_fields=["email_domain"],
        required_supabase_fields=[],
        production_status="VALID",
        reason="Purchaser email domain frequency encoding.",
    ),
    FeatureContract(
        feature_name="address_id_numeric",
        group="Direct Transaction Features",
        data_type="float",
        training_source="raw_transaction.address_id",
        live_source="FastAPI request.address_id",
        training_formula="float(address_id) if is_digit else 0.0",
        live_formula="float(address_id) if is_digit else 0.0",
        lookback_window=None,
        leakage_rule="None",
        nullable_behavior="0.0 on non-numeric strings",
        required_raw_fields=["address_id"],
        required_supabase_fields=[],
        production_status="INVALID",
        reason="address_id is an arbitrary alphanumeric identifier in production (e.g., 'ZIP_94103', 'NY'); direct float cast crashes or produces spurious ordinal relationships in tree models.",
    ),
    FeatureContract(
        feature_name="country_freq",
        group="Direct Transaction Features",
        data_type="float",
        training_source="Historical country distribution",
        live_source="FastAPI request.country",
        training_formula="country_distribution.get(country, 0.01)",
        live_formula="country_distribution.get(country, 0.01)",
        lookback_window=None,
        leakage_rule="None",
        nullable_behavior="0.01",
        required_raw_fields=["country"],
        required_supabase_fields=[],
        production_status="NEEDS_DATA",
        reason="Historical IEEE-CIS benchmark dataset did not contain an ISO 2-letter country column; requires establishing an authoritative cross-border country frequency table.",
    ),
    FeatureContract(
        feature_name="missing_fields_count",
        group="Direct Transaction Features",
        data_type="float",
        training_source="raw_transaction fields null count",
        live_source="FastAPI request fields null count",
        training_formula="sum(1.0 for f in optional_fields if raw[f] is None)",
        live_formula="sum(1.0 for f in optional_fields if request[f] is None)",
        lookback_window=None,
        leakage_rule="Intrinsic current transaction attribute",
        nullable_behavior="0.0 to 8.0 integer count represented as float",
        required_raw_fields=["device_id", "merchant_id", "email_domain", "address_id", "ip_address", "country", "card_network", "card_type"],
        required_supabase_fields=[],
        production_status="VALID",
        reason="Count of unpopulated optional parameters is an effective proxy for anonymous checkout fraud.",
    ),

    # ------------------------------------------------------------------------
    # Group 2: Time & Calendar Features (6 candidates)
    # ------------------------------------------------------------------------
    FeatureContract(
        feature_name="hour",
        group="Time & Calendar Features",
        data_type="float",
        training_source="raw_transaction.transaction_timestamp",
        live_source="FastAPI request.transaction_timestamp",
        training_formula="float(dt.hour)",
        live_formula="float(dt.hour)",
        lookback_window=None,
        leakage_rule="Computed directly from event timestamp",
        nullable_behavior="Defaults to current UTC hour if timestamp omitted",
        required_raw_fields=["transaction_timestamp"],
        required_supabase_fields=[],
        production_status="VALID",
        reason="Hour of day (0-23) captures nocturnal fraud spikes.",
    ),
    FeatureContract(
        feature_name="weekday_index",
        group="Time & Calendar Features",
        data_type="float",
        training_source="raw_transaction.transaction_timestamp",
        live_source="FastAPI request.transaction_timestamp",
        training_formula="float(dt.weekday())",
        live_formula="float(dt.weekday())",
        lookback_window=None,
        leakage_rule="Computed directly from event timestamp",
        nullable_behavior="Defaults to current UTC weekday",
        required_raw_fields=["transaction_timestamp"],
        required_supabase_fields=[],
        production_status="VALID",
        reason="Day of week index (0=Monday, 6=Sunday).",
    ),
    FeatureContract(
        feature_name="is_weekend",
        group="Time & Calendar Features",
        data_type="float",
        training_source="raw_transaction.transaction_timestamp",
        live_source="FastAPI request.transaction_timestamp",
        training_formula="1.0 if dt.weekday() in (5, 6) else 0.0",
        live_formula="1.0 if dt.weekday() in (5, 6) else 0.0",
        lookback_window=None,
        leakage_rule="Computed directly from event timestamp",
        nullable_behavior="0.0 or 1.0",
        required_raw_fields=["transaction_timestamp"],
        required_supabase_fields=[],
        production_status="VALID",
        reason="Binary weekend flag.",
    ),
    FeatureContract(
        feature_name="is_night",
        group="Time & Calendar Features",
        data_type="float",
        training_source="raw_transaction.transaction_timestamp",
        live_source="FastAPI request.transaction_timestamp",
        training_formula="1.0 if dt.hour < 6 else 0.0",
        live_formula="1.0 if dt.hour < 6 else 0.0",
        lookback_window=None,
        leakage_rule="Computed directly from event timestamp",
        nullable_behavior="0.0 or 1.0",
        required_raw_fields=["transaction_timestamp"],
        required_supabase_fields=[],
        production_status="VALID",
        reason="Binary flag for late-night / off-peak hours (00:00 to 05:59 UTC).",
    ),
    FeatureContract(
        feature_name="day_of_year",
        group="Time & Calendar Features",
        data_type="float",
        training_source="raw_transaction.transaction_timestamp",
        live_source="FastAPI request.transaction_timestamp",
        training_formula="float(dt.timetuple().tm_yday)",
        live_formula="float(dt.timetuple().tm_yday)",
        lookback_window=None,
        leakage_rule="Computed directly from event timestamp",
        nullable_behavior="Defaults to current day of year",
        required_raw_fields=["transaction_timestamp"],
        required_supabase_fields=[],
        production_status="VALID",
        reason="Ordinal day of year (1-366) capturing seasonality.",
    ),
    FeatureContract(
        feature_name="time_since_midnight_sec",
        group="Time & Calendar Features",
        data_type="float",
        training_source="raw_transaction.transaction_timestamp",
        live_source="FastAPI request.transaction_timestamp",
        training_formula="float(dt.hour * 3600 + dt.minute * 60 + dt.second)",
        live_formula="float(dt.hour * 3600 + dt.minute * 60 + dt.second)",
        lookback_window=None,
        leakage_rule="Computed directly from event timestamp",
        nullable_behavior="Defaults to seconds elapsed since 00:00:00 UTC",
        required_raw_fields=["transaction_timestamp"],
        required_supabase_fields=[],
        production_status="VALID",
        reason="Continuous diurnal seconds variable (0.0 to 86399.0).",
    ),

    # ------------------------------------------------------------------------
    # Group 3: Amount Transformation Features (4 candidates)
    # ------------------------------------------------------------------------
    FeatureContract(
        feature_name="amount_log",
        group="Amount Transformation Features",
        data_type="float",
        training_source="raw_transaction.amount",
        live_source="FastAPI request.amount",
        training_formula="math.log1p(amount)",
        live_formula="math.log1p(amount)",
        lookback_window=None,
        leakage_rule="Intrinsic current transaction attribute",
        nullable_behavior="0.0 if amount <= 0",
        required_raw_fields=["amount"],
        required_supabase_fields=[],
        production_status="VALID",
        reason="Logarithmic transformation handling extreme right-skewed amount distributions.",
    ),
    FeatureContract(
        feature_name="amount_cents",
        group="Amount Transformation Features",
        data_type="float",
        training_source="raw_transaction.amount",
        live_source="FastAPI request.amount",
        training_formula="round(amount % 1.0, 2)",
        live_formula="round(amount % 1.0, 2)",
        lookback_window=None,
        leakage_rule="Intrinsic current transaction attribute",
        nullable_behavior="0.0",
        required_raw_fields=["amount"],
        required_supabase_fields=[],
        production_status="VALID",
        reason="Fractional component of transaction amount; fraud transactions frequently exhibit irregular cents fractions.",
    ),
    FeatureContract(
        feature_name="is_round_amount",
        group="Amount Transformation Features",
        data_type="float",
        training_source="raw_transaction.amount",
        live_source="FastAPI request.amount",
        training_formula="1.0 if amount >= 10.0 and (amount % 10.0 == 0.0) else 0.0",
        live_formula="1.0 if amount >= 10.0 and (amount % 10.0 == 0.0) else 0.0",
        lookback_window=None,
        leakage_rule="Intrinsic current transaction attribute",
        nullable_behavior="0.0",
        required_raw_fields=["amount"],
        required_supabase_fields=[],
        production_status="VALID",
        reason="Flag for round denomination amounts ($50, $100, $500), common in prepaid card testing.",
    ),
    FeatureContract(
        feature_name="is_zero_cents",
        group="Amount Transformation Features",
        data_type="float",
        training_source="raw_transaction.amount",
        live_source="FastAPI request.amount",
        training_formula="1.0 if (amount % 1.0 == 0.0) else 0.0",
        live_formula="1.0 if (amount % 1.0 == 0.0) else 0.0",
        lookback_window=None,
        leakage_rule="Intrinsic current transaction attribute",
        nullable_behavior="1.0",
        required_raw_fields=["amount"],
        required_supabase_fields=[],
        production_status="VALID",
        reason="Binary flag indicating whether amount has zero fractional cents ($100.00 vs $100.49).",
    ),

    # ------------------------------------------------------------------------
    # Group 4: Customer Behavioral History (10 candidates)
    # ------------------------------------------------------------------------
    FeatureContract(
        feature_name="customer_tx_count_5m",
        group="Customer Behavioral History",
        data_type="float",
        training_source="history where customer_id=curr and ts in [t_curr - 5m, t_curr)",
        live_source="Supabase fraud_transactions query where customer_id=curr and ts in [t_curr - 5m, t_curr)",
        training_formula="float(count(tx in history if tx.customer == curr and t_curr - 300 <= tx.ts < t_curr))",
        live_formula="float(count(tx in history if tx.customer == curr and t_curr - 300 <= tx.ts < t_curr))",
        lookback_window="5m",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if no prior history",
        required_raw_fields=["customer_id", "transaction_timestamp"],
        required_supabase_fields=["customer_id", "transaction_timestamp"],
        production_status="VALID",
        reason="High-frequency 5-minute customer velocity burst signal.",
    ),
    FeatureContract(
        feature_name="customer_tx_count_1h",
        group="Customer Behavioral History",
        data_type="float",
        training_source="history where customer_id=curr and ts in [t_curr - 1h, t_curr)",
        live_source="Supabase fraud_transactions query where customer_id=curr and ts in [t_curr - 1h, t_curr)",
        training_formula="float(count(tx in history if tx.customer == curr and t_curr - 3600 <= tx.ts < t_curr))",
        live_formula="float(count(tx in history if tx.customer == curr and t_curr - 3600 <= tx.ts < t_curr))",
        lookback_window="1h",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if no prior history",
        required_raw_fields=["customer_id", "transaction_timestamp"],
        required_supabase_fields=["customer_id", "transaction_timestamp"],
        production_status="VALID",
        reason="Intraday 1-hour customer transaction velocity.",
    ),
    FeatureContract(
        feature_name="customer_tx_count_24h",
        group="Customer Behavioral History",
        data_type="float",
        training_source="history where customer_id=curr and ts in [t_curr - 24h, t_curr)",
        live_source="Supabase fraud_transactions query where customer_id=curr and ts in [t_curr - 24h, t_curr)",
        training_formula="float(count(tx in history if tx.customer == curr and t_curr - 86400 <= tx.ts < t_curr))",
        live_formula="float(count(tx in history if tx.customer == curr and t_curr - 86400 <= tx.ts < t_curr))",
        lookback_window="24h",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if no prior history",
        required_raw_fields=["customer_id", "transaction_timestamp"],
        required_supabase_fields=["customer_id", "transaction_timestamp"],
        production_status="VALID",
        reason="Daily 24-hour customer transaction volume.",
    ),
    FeatureContract(
        feature_name="customer_amount_mean",
        group="Customer Behavioral History",
        data_type="float",
        training_source="history where customer_id=curr and ts < t_curr",
        live_source="Supabase fraud_entity_state or fraud_transactions where customer_id=curr and ts < t_curr",
        training_formula="mean([tx.amount for tx in history if tx.customer == curr and tx.ts < t_curr])",
        live_formula="mean([tx.amount for tx in history if tx.customer == curr and tx.ts < t_curr])",
        lookback_window="lifetime",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if no prior history",
        required_raw_fields=["customer_id", "amount", "transaction_timestamp"],
        required_supabase_fields=["customer_id", "amount", "transaction_timestamp"],
        production_status="VALID",
        reason="Historical average spending baseline for account.",
    ),
    FeatureContract(
        feature_name="customer_amount_std",
        group="Customer Behavioral History",
        data_type="float",
        training_source="history where customer_id=curr and ts < t_curr",
        live_source="Supabase fraud_transactions where customer_id=curr and ts < t_curr",
        training_formula="std([tx.amount for tx in history if tx.customer == curr and tx.ts < t_curr])",
        live_formula="std([tx.amount for tx in history if tx.customer == curr and tx.ts < t_curr])",
        lookback_window="lifetime",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if count < 2",
        required_raw_fields=["customer_id", "amount", "transaction_timestamp"],
        required_supabase_fields=["customer_id", "amount", "transaction_timestamp"],
        production_status="VALID",
        reason="Standard deviation of past customer spend; handles count < 2 deterministically.",
    ),
    FeatureContract(
        feature_name="customer_amount_zscore",
        group="Customer Behavioral History",
        data_type="float",
        training_source="Derived from amount, customer_amount_mean, customer_amount_std",
        live_source="Derived from amount, customer_amount_mean, customer_amount_std",
        training_formula="(amount - mean) / (std + 1e-5) if count >= 2 and std > 0 else 0.0",
        live_formula="(amount - mean) / (std + 1e-5) if count >= 2 and std > 0 else 0.0",
        lookback_window="lifetime",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if count < 2 or std == 0",
        required_raw_fields=["customer_id", "amount", "transaction_timestamp"],
        required_supabase_fields=["customer_id", "amount", "transaction_timestamp"],
        production_status="VALID",
        reason="Z-score standardization of transaction amount relative to account history.",
    ),
    FeatureContract(
        feature_name="customer_amount_ratio",
        group="Customer Behavioral History",
        data_type="float",
        training_source="Derived from amount and customer_amount_mean",
        live_source="Derived from amount and customer_amount_mean",
        training_formula="amount / (mean + 1e-5) if count > 0 else 1.0",
        live_formula="amount / (mean + 1e-5) if count > 0 else 1.0",
        lookback_window="lifetime",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="1.0 if new customer",
        required_raw_fields=["customer_id", "amount", "transaction_timestamp"],
        required_supabase_fields=["customer_id", "amount", "transaction_timestamp"],
        production_status="VALID",
        reason="Multiplier of current transaction amount over customer's historical average spend.",
    ),
    FeatureContract(
        feature_name="customer_is_new",
        group="Customer Behavioral History",
        data_type="float",
        training_source="history where customer_id=curr and ts < t_curr",
        live_source="Supabase fraud_transactions where customer_id=curr and ts < t_curr",
        training_formula="1.0 if past_customer_count == 0 else 0.0",
        live_formula="1.0 if past_customer_count == 0 else 0.0",
        lookback_window="lifetime",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="1.0 for first transaction",
        required_raw_fields=["customer_id", "transaction_timestamp"],
        required_supabase_fields=["customer_id", "transaction_timestamp"],
        production_status="VALID",
        reason="Indicator flag for new customer account with no prior transactions.",
    ),
    FeatureContract(
        feature_name="customer_unique_merchants_24h",
        group="Customer Behavioral History",
        data_type="float",
        training_source="history where customer_id=curr and ts in [t_curr - 24h, t_curr)",
        live_source="Supabase fraud_transactions where customer_id=curr and ts in [t_curr - 24h, t_curr)",
        training_formula="float(len(set(tx.merchant for tx in history if tx.customer == curr and tx.merchant and t_curr - 86400 <= tx.ts < t_curr)))",
        live_formula="float(len(set(tx.merchant for tx in history if tx.customer == curr and tx.merchant and t_curr - 86400 <= tx.ts < t_curr)))",
        lookback_window="24h",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if no prior merchants",
        required_raw_fields=["customer_id", "merchant_id", "transaction_timestamp"],
        required_supabase_fields=["customer_id", "merchant_id", "transaction_timestamp"],
        production_status="VALID",
        reason="Number of distinct merchants visited by customer in 24 hours (card-hopping detection).",
    ),
    FeatureContract(
        feature_name="customer_unique_devices_24h",
        group="Customer Behavioral History",
        data_type="float",
        training_source="history where customer_id=curr and ts in [t_curr - 24h, t_curr)",
        live_source="Supabase fraud_transactions where customer_id=curr and ts in [t_curr - 24h, t_curr)",
        training_formula="float(len(set(tx.device for tx in history if tx.customer == curr and tx.device and t_curr - 86400 <= tx.ts < t_curr)))",
        live_formula="float(len(set(tx.device for tx in history if tx.customer == curr and tx.device and t_curr - 86400 <= tx.ts < t_curr)))",
        lookback_window="24h",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if no prior devices",
        required_raw_fields=["customer_id", "device_id", "transaction_timestamp"],
        required_supabase_fields=["customer_id", "device_id", "transaction_timestamp"],
        production_status="VALID",
        reason="Number of distinct hardware devices utilized by customer in 24 hours.",
    ),

    # ------------------------------------------------------------------------
    # Group 5: Card Behavioral History (7 candidates)
    # ------------------------------------------------------------------------
    FeatureContract(
        feature_name="card_tx_count_5m",
        group="Card Behavioral History",
        data_type="float",
        training_source="history where card_id=curr and ts in [t_curr - 5m, t_curr)",
        live_source="Supabase fraud_transactions where card_id=curr and ts in [t_curr - 5m, t_curr)",
        training_formula="float(count(tx in history if tx.card == curr and t_curr - 300 <= tx.ts < t_curr))",
        live_formula="float(count(tx in history if tx.card == curr and t_curr - 300 <= tx.ts < t_curr))",
        lookback_window="5m",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0",
        required_raw_fields=["card_id", "transaction_timestamp"],
        required_supabase_fields=["card_id", "transaction_timestamp"],
        production_status="VALID",
        reason="Card-level 5-minute velocity burst.",
    ),
    FeatureContract(
        feature_name="card_tx_count_1h",
        group="Card Behavioral History",
        data_type="float",
        training_source="history where card_id=curr and ts in [t_curr - 1h, t_curr)",
        live_source="Supabase fraud_transactions where card_id=curr and ts in [t_curr - 1h, t_curr)",
        training_formula="float(count(tx in history if tx.card == curr and t_curr - 3600 <= tx.ts < t_curr))",
        live_formula="float(count(tx in history if tx.card == curr and t_curr - 3600 <= tx.ts < t_curr))",
        lookback_window="1h",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0",
        required_raw_fields=["card_id", "transaction_timestamp"],
        required_supabase_fields=["card_id", "transaction_timestamp"],
        production_status="VALID",
        reason="Card-level 1-hour velocity volume.",
    ),
    FeatureContract(
        feature_name="card_tx_count_24h",
        group="Card Behavioral History",
        data_type="float",
        training_source="history where card_id=curr and ts in [t_curr - 24h, t_curr)",
        live_source="Supabase fraud_transactions where card_id=curr and ts in [t_curr - 24h, t_curr)",
        training_formula="float(count(tx in history if tx.card == curr and t_curr - 86400 <= tx.ts < t_curr))",
        live_formula="float(count(tx in history if tx.card == curr and t_curr - 86400 <= tx.ts < t_curr))",
        lookback_window="24h",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0",
        required_raw_fields=["card_id", "transaction_timestamp"],
        required_supabase_fields=["card_id", "transaction_timestamp"],
        production_status="VALID",
        reason="Card-level 24-hour transaction volume.",
    ),
    FeatureContract(
        feature_name="card_amount_mean",
        group="Card Behavioral History",
        data_type="float",
        training_source="history where card_id=curr and ts < t_curr",
        live_source="Supabase fraud_transactions where card_id=curr and ts < t_curr",
        training_formula="mean([tx.amount for tx in history if tx.card == curr and tx.ts < t_curr])",
        live_formula="mean([tx.amount for tx in history if tx.card == curr and tx.ts < t_curr])",
        lookback_window="lifetime",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if no prior card transactions",
        required_raw_fields=["card_id", "amount", "transaction_timestamp"],
        required_supabase_fields=["card_id", "amount", "transaction_timestamp"],
        production_status="VALID",
        reason="Historical mean spend across all prior transactions on this card.",
    ),
    FeatureContract(
        feature_name="card_amount_std",
        group="Card Behavioral History",
        data_type="float",
        training_source="history where card_id=curr and ts < t_curr",
        live_source="Supabase fraud_transactions where card_id=curr and ts < t_curr",
        training_formula="std([tx.amount for tx in history if tx.card == curr and tx.ts < t_curr])",
        live_formula="std([tx.amount for tx in history if tx.card == curr and tx.ts < t_curr])",
        lookback_window="lifetime",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if count < 2",
        required_raw_fields=["card_id", "amount", "transaction_timestamp"],
        required_supabase_fields=["card_id", "amount", "transaction_timestamp"],
        production_status="VALID",
        reason="Standard deviation of spend for this card token.",
    ),
    FeatureContract(
        feature_name="card_amount_ratio",
        group="Card Behavioral History",
        data_type="float",
        training_source="Derived from amount and card_amount_mean",
        live_source="Derived from amount and card_amount_mean",
        training_formula="amount / (card_mean + 1e-5) if count > 0 else 1.0",
        live_formula="amount / (card_mean + 1e-5) if count > 0 else 1.0",
        lookback_window="lifetime",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="1.0 if new card",
        required_raw_fields=["card_id", "amount", "transaction_timestamp"],
        required_supabase_fields=["card_id", "amount", "transaction_timestamp"],
        production_status="VALID",
        reason="Multiplier of current transaction amount over card historical mean.",
    ),
    FeatureContract(
        feature_name="card_is_new",
        group="Card Behavioral History",
        data_type="float",
        training_source="history where card_id=curr and ts < t_curr",
        live_source="Supabase fraud_transactions where card_id=curr and ts < t_curr",
        training_formula="1.0 if card_past_count == 0 else 0.0",
        live_formula="1.0 if card_past_count == 0 else 0.0",
        lookback_window="lifetime",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="1.0 for first transaction on card",
        required_raw_fields=["card_id", "transaction_timestamp"],
        required_supabase_fields=["card_id", "transaction_timestamp"],
        production_status="VALID",
        reason="Binary flag indicating first observation of card on platform.",
    ),

    # ------------------------------------------------------------------------
    # Group 6: Device Fingerprint History (6 candidates)
    # ------------------------------------------------------------------------
    FeatureContract(
        feature_name="device_tx_count_5m",
        group="Device Fingerprint History",
        data_type="float",
        training_source="history where device_id=curr and ts in [t_curr - 5m, t_curr)",
        live_source="Supabase fraud_transactions where device_id=curr and ts in [t_curr - 5m, t_curr)",
        training_formula="float(count(tx in history if tx.device == curr and t_curr - 300 <= tx.ts < t_curr)) if curr else 0.0",
        live_formula="float(count(tx in history if tx.device == curr and t_curr - 300 <= tx.ts < t_curr)) if curr else 0.0",
        lookback_window="5m",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if device_id is null or no prior events",
        required_raw_fields=["device_id", "transaction_timestamp"],
        required_supabase_fields=["device_id", "transaction_timestamp"],
        production_status="VALID",
        reason="Hardware bot flood velocity indicator.",
    ),
    FeatureContract(
        feature_name="device_tx_count_1h",
        group="Device Fingerprint History",
        data_type="float",
        training_source="history where device_id=curr and ts in [t_curr - 1h, t_curr)",
        live_source="Supabase fraud_transactions where device_id=curr and ts in [t_curr - 1h, t_curr)",
        training_formula="float(count(tx in history if tx.device == curr and t_curr - 3600 <= tx.ts < t_curr)) if curr else 0.0",
        live_formula="float(count(tx in history if tx.device == curr and t_curr - 3600 <= tx.ts < t_curr)) if curr else 0.0",
        lookback_window="1h",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if device_id is null or no prior events",
        required_raw_fields=["device_id", "transaction_timestamp"],
        required_supabase_fields=["device_id", "transaction_timestamp"],
        production_status="VALID",
        reason="1-hour hardware fingerprint velocity.",
    ),
    FeatureContract(
        feature_name="device_tx_count_24h",
        group="Device Fingerprint History",
        data_type="float",
        training_source="history where device_id=curr and ts in [t_curr - 24h, t_curr)",
        live_source="Supabase fraud_transactions where device_id=curr and ts in [t_curr - 24h, t_curr)",
        training_formula="float(count(tx in history if tx.device == curr and t_curr - 86400 <= tx.ts < t_curr)) if curr else 0.0",
        live_formula="float(count(tx in history if tx.device == curr and t_curr - 86400 <= tx.ts < t_curr)) if curr else 0.0",
        lookback_window="24h",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if device_id is null or no prior events",
        required_raw_fields=["device_id", "transaction_timestamp"],
        required_supabase_fields=["device_id", "transaction_timestamp"],
        production_status="VALID",
        reason="24-hour hardware fingerprint activity count.",
    ),
    FeatureContract(
        feature_name="device_unique_customers_24h",
        group="Device Fingerprint History",
        data_type="float",
        training_source="history where device_id=curr and ts in [t_curr - 24h, t_curr)",
        live_source="Supabase fraud_transactions where device_id=curr and ts in [t_curr - 24h, t_curr)",
        training_formula="float(len(set(tx.customer for tx in history if tx.device == curr and t_curr - 86400 <= tx.ts < t_curr))) if curr else 0.0",
        live_formula="float(len(set(tx.customer for tx in history if tx.device == curr and t_curr - 86400 <= tx.ts < t_curr))) if curr else 0.0",
        lookback_window="24h",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if device_id is null",
        required_raw_fields=["device_id", "customer_id", "transaction_timestamp"],
        required_supabase_fields=["device_id", "customer_id", "transaction_timestamp"],
        production_status="VALID",
        reason="High-risk indicator for single device used by multiple distinct customer accounts.",
    ),
    FeatureContract(
        feature_name="device_unique_cards_24h",
        group="Device Fingerprint History",
        data_type="float",
        training_source="history where device_id=curr and ts in [t_curr - 24h, t_curr)",
        live_source="Supabase fraud_transactions where device_id=curr and ts in [t_curr - 24h, t_curr)",
        training_formula="float(len(set(tx.card for tx in history if tx.device == curr and t_curr - 86400 <= tx.ts < t_curr))) if curr else 0.0",
        live_formula="float(len(set(tx.card for tx in history if tx.device == curr and t_curr - 86400 <= tx.ts < t_curr))) if curr else 0.0",
        lookback_window="24h",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if device_id is null",
        required_raw_fields=["device_id", "card_id", "transaction_timestamp"],
        required_supabase_fields=["device_id", "card_id", "transaction_timestamp"],
        production_status="VALID",
        reason="Card-stuffing indicator: single hardware fingerprint attempting multiple cards.",
    ),
    FeatureContract(
        feature_name="device_is_new",
        group="Device Fingerprint History",
        data_type="float",
        training_source="history where device_id=curr and ts < t_curr",
        live_source="Supabase fraud_transactions where device_id=curr and ts < t_curr",
        training_formula="1.0 if curr and device_past_count == 0 else 0.0",
        live_formula="1.0 if curr and device_past_count == 0 else 0.0",
        lookback_window="lifetime",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if device_id is omitted",
        required_raw_fields=["device_id", "transaction_timestamp"],
        required_supabase_fields=["device_id", "transaction_timestamp"],
        production_status="VALID",
        reason="First observation of hardware fingerprint across platform history.",
    ),

    # ------------------------------------------------------------------------
    # Group 7: Merchant Risk & Pairing History (6 candidates)
    # ------------------------------------------------------------------------
    FeatureContract(
        feature_name="merchant_tx_count_1h",
        group="Merchant Risk & History",
        data_type="float",
        training_source="history where merchant_id=curr and ts in [t_curr - 1h, t_curr)",
        live_source="Supabase fraud_transactions where merchant_id=curr and ts in [t_curr - 1h, t_curr)",
        training_formula="float(count(tx in history if tx.merchant == curr and t_curr - 3600 <= tx.ts < t_curr)) if curr else 0.0",
        live_formula="float(count(tx in history if tx.merchant == curr and t_curr - 3600 <= tx.ts < t_curr)) if curr else 0.0",
        lookback_window="1h",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if merchant_id is null",
        required_raw_fields=["merchant_id", "transaction_timestamp"],
        required_supabase_fields=["merchant_id", "transaction_timestamp"],
        production_status="VALID",
        reason="1-hour merchant transaction volume.",
    ),
    FeatureContract(
        feature_name="merchant_tx_count_24h",
        group="Merchant Risk & History",
        data_type="float",
        training_source="history where merchant_id=curr and ts in [t_curr - 24h, t_curr)",
        live_source="Supabase fraud_transactions where merchant_id=curr and ts in [t_curr - 24h, t_curr)",
        training_formula="float(count(tx in history if tx.merchant == curr and t_curr - 86400 <= tx.ts < t_curr)) if curr else 0.0",
        live_formula="float(count(tx in history if tx.merchant == curr and t_curr - 86400 <= tx.ts < t_curr)) if curr else 0.0",
        lookback_window="24h",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if merchant_id is null",
        required_raw_fields=["merchant_id", "transaction_timestamp"],
        required_supabase_fields=["merchant_id", "transaction_timestamp"],
        production_status="VALID",
        reason="24-hour merchant daily transaction traffic.",
    ),
    FeatureContract(
        feature_name="merchant_amount_mean",
        group="Merchant Risk & History",
        data_type="float",
        training_source="history where merchant_id=curr and ts < t_curr",
        live_source="Supabase fraud_transactions where merchant_id=curr and ts < t_curr",
        training_formula="mean([tx.amount for tx in history if tx.merchant == curr and tx.ts < t_curr]) if curr else 0.0",
        live_formula="mean([tx.amount for tx in history if tx.merchant == curr and tx.ts < t_curr]) if curr else 0.0",
        lookback_window="lifetime",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if merchant_id is null or count == 0",
        required_raw_fields=["merchant_id", "amount", "transaction_timestamp"],
        required_supabase_fields=["merchant_id", "amount", "transaction_timestamp"],
        production_status="VALID",
        reason="Merchant average ticket purchase size.",
    ),
    FeatureContract(
        feature_name="merchant_amount_std",
        group="Merchant Risk & History",
        data_type="float",
        training_source="history where merchant_id=curr and ts < t_curr",
        live_source="Supabase fraud_transactions where merchant_id=curr and ts < t_curr",
        training_formula="std([tx.amount for tx in history if tx.merchant == curr and tx.ts < t_curr]) if curr else 0.0",
        live_formula="std([tx.amount for tx in history if tx.merchant == curr and tx.ts < t_curr]) if curr else 0.0",
        lookback_window="lifetime",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if count < 2",
        required_raw_fields=["merchant_id", "amount", "transaction_timestamp"],
        required_supabase_fields=["merchant_id", "amount", "transaction_timestamp"],
        production_status="VALID",
        reason="Standard deviation of merchant purchase size.",
    ),
    FeatureContract(
        feature_name="customer_merchant_tx_count",
        group="Merchant Risk & History",
        data_type="float",
        training_source="history where customer_id=curr and merchant_id=curr and ts < t_curr",
        live_source="Supabase fraud_transactions where customer_id=curr and merchant_id=curr and ts < t_curr",
        training_formula="float(count(tx in history if tx.customer == curr and tx.merchant == curr and tx.ts < t_curr))",
        live_formula="float(count(tx in history if tx.customer == curr and tx.merchant == curr and tx.ts < t_curr))",
        lookback_window="lifetime",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if merchant_id is null or no prior interaction",
        required_raw_fields=["customer_id", "merchant_id", "transaction_timestamp"],
        required_supabase_fields=["customer_id", "merchant_id", "transaction_timestamp"],
        production_status="VALID",
        reason="Repeat customer shopping frequency at specific merchant.",
    ),
    FeatureContract(
        feature_name="customer_merchant_is_new",
        group="Merchant Risk & History",
        data_type="float",
        training_source="history where customer_id=curr and merchant_id=curr and ts < t_curr",
        live_source="Supabase fraud_transactions where customer_id=curr and merchant_id=curr and ts < t_curr",
        training_formula="1.0 if curr_merchant and customer_merchant_tx_count == 0 else 0.0",
        live_formula="1.0 if curr_merchant and customer_merchant_tx_count == 0 else 0.0",
        lookback_window="lifetime",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if merchant_id is null",
        required_raw_fields=["customer_id", "merchant_id", "transaction_timestamp"],
        required_supabase_fields=["customer_id", "merchant_id", "transaction_timestamp"],
        production_status="VALID",
        reason="First-time merchant interaction flag.",
    ),

    # ------------------------------------------------------------------------
    # Group 8: Client IP Network History (7 candidates)
    # ------------------------------------------------------------------------
    FeatureContract(
        feature_name="ip_tx_count_5m",
        group="Client IP Network History",
        data_type="float",
        training_source="history where ip_address=curr and ts in [t_curr - 5m, t_curr)",
        live_source="Supabase fraud_transactions where ip_address=curr and ts in [t_curr - 5m, t_curr)",
        training_formula="float(count(tx in history if tx.ip == curr and t_curr - 300 <= tx.ts < t_curr)) if curr else 0.0",
        live_formula="float(count(tx in history if tx.ip == curr and t_curr - 300 <= tx.ts < t_curr)) if curr else 0.0",
        lookback_window="5m",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if ip is null",
        required_raw_fields=["ip_address", "transaction_timestamp"],
        required_supabase_fields=["ip_address", "transaction_timestamp"],
        production_status="VALID",
        reason="IP-level 5-minute transaction burst velocity (botnet indicator).",
    ),
    FeatureContract(
        feature_name="ip_tx_count_1h",
        group="Client IP Network History",
        data_type="float",
        training_source="history where ip_address=curr and ts in [t_curr - 1h, t_curr)",
        live_source="Supabase fraud_transactions where ip_address=curr and ts in [t_curr - 1h, t_curr)",
        training_formula="float(count(tx in history if tx.ip == curr and t_curr - 3600 <= tx.ts < t_curr)) if curr else 0.0",
        live_formula="float(count(tx in history if tx.ip == curr and t_curr - 3600 <= tx.ts < t_curr)) if curr else 0.0",
        lookback_window="1h",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if ip is null",
        required_raw_fields=["ip_address", "transaction_timestamp"],
        required_supabase_fields=["ip_address", "transaction_timestamp"],
        production_status="VALID",
        reason="IP-level 1-hour transaction volume.",
    ),
    FeatureContract(
        feature_name="ip_tx_count_24h",
        group="Client IP Network History",
        data_type="float",
        training_source="history where ip_address=curr and ts in [t_curr - 24h, t_curr)",
        live_source="Supabase fraud_transactions where ip_address=curr and ts in [t_curr - 24h, t_curr)",
        training_formula="float(count(tx in history if tx.ip == curr and t_curr - 86400 <= tx.ts < t_curr)) if curr else 0.0",
        live_formula="float(count(tx in history if tx.ip == curr and t_curr - 86400 <= tx.ts < t_curr)) if curr else 0.0",
        lookback_window="24h",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if ip is null",
        required_raw_fields=["ip_address", "transaction_timestamp"],
        required_supabase_fields=["ip_address", "transaction_timestamp"],
        production_status="VALID",
        reason="IP-level 24-hour transaction volume.",
    ),
    FeatureContract(
        feature_name="ip_unique_customers_24h",
        group="Client IP Network History",
        data_type="float",
        training_source="history where ip_address=curr and ts in [t_curr - 24h, t_curr)",
        live_source="Supabase fraud_transactions where ip_address=curr and ts in [t_curr - 24h, t_curr)",
        training_formula="float(len(set(tx.customer for tx in history if tx.ip == curr and t_curr - 86400 <= tx.ts < t_curr))) if curr else 0.0",
        live_formula="float(len(set(tx.customer for tx in history if tx.ip == curr and t_curr - 86400 <= tx.ts < t_curr))) if curr else 0.0",
        lookback_window="24h",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if ip is null",
        required_raw_fields=["ip_address", "customer_id", "transaction_timestamp"],
        required_supabase_fields=["ip_address", "customer_id", "transaction_timestamp"],
        production_status="VALID",
        reason="Distinct customer accounts originating from same IP address in 24 hours (credential stuffing indicator).",
    ),
    FeatureContract(
        feature_name="ip_unique_cards_24h",
        group="Client IP Network History",
        data_type="float",
        training_source="history where ip_address=curr and ts in [t_curr - 24h, t_curr)",
        live_source="Supabase fraud_transactions where ip_address=curr and ts in [t_curr - 24h, t_curr)",
        training_formula="float(len(set(tx.card for tx in history if tx.ip == curr and t_curr - 86400 <= tx.ts < t_curr))) if curr else 0.0",
        live_formula="float(len(set(tx.card for tx in history if tx.ip == curr and t_curr - 86400 <= tx.ts < t_curr))) if curr else 0.0",
        lookback_window="24h",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if ip is null",
        required_raw_fields=["ip_address", "card_id", "transaction_timestamp"],
        required_supabase_fields=["ip_address", "card_id", "transaction_timestamp"],
        production_status="VALID",
        reason="Distinct payment cards tested from single IP address within 24 hours.",
    ),
    FeatureContract(
        feature_name="ip_unique_devices_24h",
        group="Client IP Network History",
        data_type="float",
        training_source="history where ip_address=curr and ts in [t_curr - 24h, t_curr)",
        live_source="Supabase fraud_transactions where ip_address=curr and ts in [t_curr - 24h, t_curr)",
        training_formula="float(len(set(tx.device for tx in history if tx.ip == curr and tx.device and t_curr - 86400 <= tx.ts < t_curr))) if curr else 0.0",
        live_formula="float(len(set(tx.device for tx in history if tx.ip == curr and tx.device and t_curr - 86400 <= tx.ts < t_curr))) if curr else 0.0",
        lookback_window="24h",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if ip is null",
        required_raw_fields=["ip_address", "device_id", "transaction_timestamp"],
        required_supabase_fields=["ip_address", "device_id", "transaction_timestamp"],
        production_status="VALID",
        reason="Distinct devices associated with same IP address.",
    ),
    FeatureContract(
        feature_name="ip_is_new",
        group="Client IP Network History",
        data_type="float",
        training_source="history where ip_address=curr and ts < t_curr",
        live_source="Supabase fraud_transactions where ip_address=curr and ts < t_curr",
        training_formula="1.0 if curr_ip and ip_past_count == 0 else 0.0",
        live_formula="1.0 if curr_ip and ip_past_count == 0 else 0.0",
        lookback_window="lifetime",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if ip is omitted",
        required_raw_fields=["ip_address", "transaction_timestamp"],
        required_supabase_fields=["ip_address", "transaction_timestamp"],
        production_status="VALID",
        reason="First observation of IP address across historical platform transactions.",
    ),

    # ------------------------------------------------------------------------
    # Group 9: Cross-Entity Relationship Features (6 candidates)
    # ------------------------------------------------------------------------
    FeatureContract(
        feature_name="customer_device_seen_before",
        group="Cross-Entity Relationship Features",
        data_type="float",
        training_source="history where customer_id=curr and device_id=curr and ts < t_curr",
        live_source="Supabase fraud_transactions where customer_id=curr and device_id=curr and ts < t_curr",
        training_formula="1.0 if curr_device and any(tx.customer == curr and tx.device == curr and tx.ts < t_curr for tx in history) else 0.0",
        live_formula="1.0 if curr_device and any(tx.customer == curr and tx.device == curr and tx.ts < t_curr for tx in history) else 0.0",
        lookback_window="lifetime",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if device_id is null",
        required_raw_fields=["customer_id", "device_id", "transaction_timestamp"],
        required_supabase_fields=["customer_id", "device_id", "transaction_timestamp"],
        production_status="VALID",
        reason="Recognized device for customer account.",
    ),
    FeatureContract(
        feature_name="customer_card_seen_before",
        group="Cross-Entity Relationship Features",
        data_type="float",
        training_source="history where customer_id=curr and card_id=curr and ts < t_curr",
        live_source="Supabase fraud_transactions where customer_id=curr and card_id=curr and ts < t_curr",
        training_formula="1.0 if any(tx.customer == curr and tx.card == curr and tx.ts < t_curr for tx in history) else 0.0",
        live_formula="1.0 if any(tx.customer == curr and tx.card == curr and tx.ts < t_curr for tx in history) else 0.0",
        lookback_window="lifetime",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 for first time card-customer pairing",
        required_raw_fields=["customer_id", "card_id", "transaction_timestamp"],
        required_supabase_fields=["customer_id", "card_id", "transaction_timestamp"],
        production_status="VALID",
        reason="Recognized card for customer account.",
    ),
    FeatureContract(
        feature_name="customer_merchant_seen_before",
        group="Cross-Entity Relationship Features",
        data_type="float",
        training_source="history where customer_id=curr and merchant_id=curr and ts < t_curr",
        live_source="Supabase fraud_transactions where customer_id=curr and merchant_id=curr and ts < t_curr",
        training_formula="1.0 if curr_merchant and any(tx.customer == curr and tx.merchant == curr and tx.ts < t_curr for tx in history) else 0.0",
        live_formula="1.0 if curr_merchant and any(tx.customer == curr and tx.merchant == curr and tx.ts < t_curr for tx in history) else 0.0",
        lookback_window="lifetime",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if merchant is null",
        required_raw_fields=["customer_id", "merchant_id", "transaction_timestamp"],
        required_supabase_fields=["customer_id", "merchant_id", "transaction_timestamp"],
        production_status="VALID",
        reason="Repeat relationship between customer and merchant.",
    ),
    FeatureContract(
        feature_name="customer_ip_seen_before",
        group="Cross-Entity Relationship Features",
        data_type="float",
        training_source="history where customer_id=curr and ip_address=curr and ts < t_curr",
        live_source="Supabase fraud_transactions where customer_id=curr and ip_address=curr and ts < t_curr",
        training_formula="1.0 if curr_ip and any(tx.customer == curr and tx.ip == curr and tx.ts < t_curr for tx in history) else 0.0",
        live_formula="1.0 if curr_ip and any(tx.customer == curr and tx.ip == curr and tx.ts < t_curr for tx in history) else 0.0",
        lookback_window="lifetime",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if ip is null",
        required_raw_fields=["customer_id", "ip_address", "transaction_timestamp"],
        required_supabase_fields=["customer_id", "ip_address", "transaction_timestamp"],
        production_status="VALID",
        reason="Familiar network origin for customer.",
    ),
    FeatureContract(
        feature_name="card_device_seen_before",
        group="Cross-Entity Relationship Features",
        data_type="float",
        training_source="history where card_id=curr and device_id=curr and ts < t_curr",
        live_source="Supabase fraud_transactions where card_id=curr and device_id=curr and ts < t_curr",
        training_formula="1.0 if curr_device and any(tx.card == curr and tx.device == curr and tx.ts < t_curr for tx in history) else 0.0",
        live_formula="1.0 if curr_device and any(tx.card == curr and tx.device == curr and tx.ts < t_curr for tx in history) else 0.0",
        lookback_window="lifetime",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if device is null",
        required_raw_fields=["card_id", "device_id", "transaction_timestamp"],
        required_supabase_fields=["card_id", "device_id", "transaction_timestamp"],
        production_status="VALID",
        reason="Card previously authorized on this specific hardware fingerprint.",
    ),
    FeatureContract(
        feature_name="card_ip_seen_before",
        group="Cross-Entity Relationship Features",
        data_type="float",
        training_source="history where card_id=curr and ip_address=curr and ts < t_curr",
        live_source="Supabase fraud_transactions where card_id=curr and ip_address=curr and ts < t_curr",
        training_formula="1.0 if curr_ip and any(tx.card == curr and tx.ip == curr and tx.ts < t_curr for tx in history) else 0.0",
        live_formula="1.0 if curr_ip and any(tx.card == curr and tx.ip == curr and tx.ts < t_curr for tx in history) else 0.0",
        lookback_window="lifetime",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if ip is null",
        required_raw_fields=["card_id", "ip_address", "transaction_timestamp"],
        required_supabase_fields=["card_id", "ip_address", "transaction_timestamp"],
        production_status="VALID",
        reason="Card previously authorized from this specific client IP address.",
    ),

    # ------------------------------------------------------------------------
    # Group 10: Velocity & Burst Features (6 candidates)
    # ------------------------------------------------------------------------
    FeatureContract(
        feature_name="rapid_transaction_flag",
        group="Velocity & Burst Features",
        data_type="float",
        training_source="history where customer_id=curr and ts < t_curr",
        live_source="Supabase fraud_transactions where customer_id=curr and ts < t_curr",
        training_formula="1.0 if last_tx_delta_sec <= 60.0 else 0.0",
        live_formula="1.0 if last_tx_delta_sec <= 60.0 else 0.0",
        lookback_window="60s",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0 if first customer transaction",
        required_raw_fields=["customer_id", "transaction_timestamp"],
        required_supabase_fields=["customer_id", "transaction_timestamp"],
        production_status="VALID",
        reason="High-risk flag for consecutive transactions within 60 seconds (carding bot blast signal).",
    ),
    FeatureContract(
        feature_name="transactions_last_5m",
        group="Velocity & Burst Features",
        data_type="float",
        training_source="history across customer/card",
        live_source="history across customer/card",
        training_formula="COUNT(*) across customer/card in 5m",
        live_formula="COUNT(*) across customer/card in 5m",
        lookback_window="5m",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0",
        required_raw_fields=["customer_id", "card_id", "transaction_timestamp"],
        required_supabase_fields=["customer_id", "card_id", "transaction_timestamp"],
        production_status="INVALID",
        reason="Ambiguous formulation ('across customer/card') introduces double-counting and collinearity with customer_tx_count_5m and card_tx_count_5m.",
    ),
    FeatureContract(
        feature_name="transactions_last_15m",
        group="Velocity & Burst Features",
        data_type="float",
        training_source="history where customer_id=curr and ts in [t_curr - 15m, t_curr)",
        live_source="Supabase fraud_transactions where customer_id=curr and ts in [t_curr - 15m, t_curr)",
        training_formula="float(count(tx in history if tx.customer == curr and t_curr - 900 <= tx.ts < t_curr))",
        live_formula="float(count(tx in history if tx.customer == curr and t_curr - 900 <= tx.ts < t_curr))",
        lookback_window="15m",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0",
        required_raw_fields=["customer_id", "transaction_timestamp"],
        required_supabase_fields=["customer_id", "transaction_timestamp"],
        production_status="VALID",
        reason="15-minute intermediate burst transaction volume window.",
    ),
    FeatureContract(
        feature_name="transactions_last_1h",
        group="Velocity & Burst Features",
        data_type="float",
        training_source="history across customer/card",
        live_source="history across customer/card",
        training_formula="COUNT(*) across customer/card in 1h",
        live_formula="COUNT(*) across customer/card in 1h",
        lookback_window="1h",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0",
        required_raw_fields=["customer_id", "card_id", "transaction_timestamp"],
        required_supabase_fields=["customer_id", "card_id", "transaction_timestamp"],
        production_status="INVALID",
        reason="Ambiguous entity definition and direct collinear duplicate of customer_tx_count_1h.",
    ),
    FeatureContract(
        feature_name="amount_sum_last_1h",
        group="Velocity & Burst Features",
        data_type="float",
        training_source="history where customer_id=curr and ts in [t_curr - 1h, t_curr)",
        live_source="Supabase fraud_transactions where customer_id=curr and ts in [t_curr - 1h, t_curr)",
        training_formula="float(sum(tx.amount for tx in history if tx.customer == curr and t_curr - 3600 <= tx.ts < t_curr))",
        live_formula="float(sum(tx.amount for tx in history if tx.customer == curr and t_curr - 3600 <= tx.ts < t_curr))",
        lookback_window="1h",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0",
        required_raw_fields=["customer_id", "amount", "transaction_timestamp"],
        required_supabase_fields=["customer_id", "amount", "transaction_timestamp"],
        production_status="VALID",
        reason="1-hour cumulative dollar outflow for customer account (balance draining detector).",
    ),
    FeatureContract(
        feature_name="amount_sum_last_24h",
        group="Velocity & Burst Features",
        data_type="float",
        training_source="history where customer_id=curr and ts in [t_curr - 24h, t_curr)",
        live_source="Supabase fraud_transactions where customer_id=curr and ts in [t_curr - 24h, t_curr)",
        training_formula="float(sum(tx.amount for tx in history if tx.customer == curr and t_curr - 86400 <= tx.ts < t_curr))",
        live_formula="float(sum(tx.amount for tx in history if tx.customer == curr and t_curr - 86400 <= tx.ts < t_curr))",
        lookback_window="24h",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="0.0",
        required_raw_fields=["customer_id", "amount", "transaction_timestamp"],
        required_supabase_fields=["customer_id", "amount", "transaction_timestamp"],
        production_status="VALID",
        reason="24-hour total dollar outflow for customer account.",
    ),

    # ------------------------------------------------------------------------
    # Additional Candidate Features Under Scrutiny (12 candidates)
    # ------------------------------------------------------------------------
    FeatureContract(
        feature_name="device_risk_score",
        group="Device Intelligence",
        data_type="float",
        training_source="Third-party vendor risk lookup",
        live_source="Third-party vendor risk lookup (e.g. FingerprintJS Pro / Sift)",
        training_formula="external_vendor_api(device_id).risk_score",
        live_formula="external_vendor_api(device_id).risk_score",
        lookback_window=None,
        leakage_rule="Point-in-time vendor score",
        nullable_behavior="0.0",
        required_raw_fields=["device_id"],
        required_supabase_fields=[],
        production_status="NEEDS_DATA",
        reason="Requires contract and integration with commercial device intelligence vendor (FingerprintJS / Sift); not present in current AegisFin system.",
    ),
    FeatureContract(
        feature_name="merchant_fraud_rate_historical",
        group="Merchant Risk & History",
        data_type="float",
        training_source="Supabase historical confirmed fraud disputes by merchant",
        live_source="Supabase historical confirmed fraud disputes by merchant",
        training_formula="sum(is_fraud) / total_tx for merchant with ts < t_curr",
        live_formula="sum(is_fraud) / total_tx for merchant with ts < t_curr",
        lookback_window="lifetime",
        leakage_rule="Strict: fraud label confirmation ts < current_timestamp",
        nullable_behavior="0.0",
        required_raw_fields=["merchant_id"],
        required_supabase_fields=["merchant_id", "is_fraud_confirmed"],
        production_status="NEEDS_DATA",
        reason="Requires confirmed chargeback/fraud outcome feedback labels in Supabase; current fraud_transactions table does not track dispute labels.",
    ),
    FeatureContract(
        feature_name="ip_is_vpn_or_proxy",
        group="Client IP Network History",
        data_type="float",
        training_source="Third-party IP intelligence lookup",
        live_source="Third-party IP intelligence lookup (e.g. IPQualityScore, MaxMind)",
        training_formula="1.0 if external_ip_api(ip).is_vpn else 0.0",
        live_formula="1.0 if external_ip_api(ip).is_vpn else 0.0",
        lookback_window=None,
        leakage_rule="Point-in-time external proxy intelligence",
        nullable_behavior="0.0",
        required_raw_fields=["ip_address"],
        required_supabase_fields=[],
        production_status="NEEDS_DATA",
        reason="Requires external IP proxy/VPN database integration (MaxMind GeoIP2 Precision or IPQualityScore); not available in offline AegisFin environment.",
    ),
    FeatureContract(
        feature_name="ip_country_mismatch",
        group="Client IP Network History",
        data_type="float",
        training_source="Comparison of IP origin country vs billing country",
        live_source="Comparison of IP origin country vs billing country",
        training_formula="1.0 if ip_geo_country != country else 0.0",
        live_formula="1.0 if ip_geo_country != country else 0.0",
        lookback_window=None,
        leakage_rule="Intrinsic current transaction comparison",
        nullable_behavior="0.0",
        required_raw_fields=["ip_address", "country"],
        required_supabase_fields=[],
        production_status="NEEDS_DATA",
        reason="Requires MaxMind GeoIP2 city/country database to resolve client IP to ISO country for comparison against transaction country.",
    ),
    FeatureContract(
        feature_name="consecutive_declines_24h",
        group="Velocity & Burst Features",
        data_type="float",
        training_source="Payment gateway authorization logs in last 24h",
        live_source="Payment gateway authorization logs in last 24h",
        training_formula="COUNT(*) gateway declines on card/customer in [t_curr - 24h, t_curr)",
        live_formula="COUNT(*) gateway declines on card/customer in [t_curr - 24h, t_curr)",
        lookback_window="24h",
        leakage_rule="Strict: decline_ts < current_timestamp",
        nullable_behavior="0.0",
        required_raw_fields=["card_id", "customer_id"],
        required_supabase_fields=["gateway_status", "decline_code"],
        production_status="NEEDS_DATA",
        reason="Requires payment gateway decline webhook ingestion; public.fraud_transactions only records transactions submitted for fraud scoring, not upstream bank declines.",
    ),
    FeatureContract(
        feature_name="card_days_active",
        group="Card Behavioral History",
        data_type="float",
        training_source="Payment card issuance/tokenization timestamp",
        live_source="Payment card issuance/tokenization timestamp",
        training_formula="(t_curr - card_issuance_date).days",
        live_formula="(t_curr - card_issuance_date).days",
        lookback_window="lifetime",
        leakage_rule="card_issuance_date < current_timestamp",
        nullable_behavior="0.0",
        required_raw_fields=["card_id"],
        required_supabase_fields=["card_issued_at"],
        production_status="NEEDS_DATA",
        reason="Requires card issuance date from issuing bank API or payment gateway tokenization metadata.",
    ),
    FeatureContract(
        feature_name="account_age_days",
        group="Customer Behavioral History",
        data_type="float",
        training_source="User registration timestamp (profiles.created_at)",
        live_source="User registration timestamp (profiles.created_at)",
        training_formula="(t_curr - user_created_at).days",
        live_formula="(t_curr - user_created_at).days",
        lookback_window="lifetime",
        leakage_rule="user_created_at < current_timestamp",
        nullable_behavior="0.0",
        required_raw_fields=["customer_id"],
        required_supabase_fields=["profiles.created_at"],
        production_status="NEEDS_DATA",
        reason="Requires linking customer_id to Supabase auth/profiles.created_at timestamp during fraud feature extraction.",
    ),
    FeatureContract(
        feature_name="transaction_channel",
        group="Direct Transaction Features",
        data_type="float",
        training_source="raw_transaction.channel",
        live_source="FastAPI request.channel",
        training_formula="frequency_map.get(channel, 0.01)",
        live_formula="frequency_map.get(channel, 0.01)",
        lookback_window=None,
        leakage_rule="Intrinsic current transaction attribute",
        nullable_behavior="0.01",
        required_raw_fields=["channel"],
        required_supabase_fields=[],
        production_status="NEEDS_DATA",
        reason="Channel ('WEB', 'MOBILE_APP', 'POS', 'API') is not currently present in Streamlit form or FastAPI FraudPredictionRequest schema.",
    ),
    FeatureContract(
        feature_name="shipping_country",
        group="Direct Transaction Features",
        data_type="float",
        training_source="raw_transaction.shipping_country",
        live_source="FastAPI request.shipping_country",
        training_formula="1.0 if shipping_country == billing_country else 0.0",
        live_formula="1.0 if shipping_country == billing_country else 0.0",
        lookback_window=None,
        leakage_rule="Intrinsic current transaction attribute",
        nullable_behavior="0.0",
        required_raw_fields=["shipping_country", "country"],
        required_supabase_fields=[],
        production_status="NEEDS_DATA",
        reason="Shipping destination country is not currently collected in Streamlit or accepted in FastAPI FraudPredictionRequest schema.",
    ),
    FeatureContract(
        feature_name="browser_user_agent",
        group="Device Intelligence",
        data_type="float",
        training_source="HTTP Request User-Agent header",
        live_source="HTTP Request User-Agent header",
        training_formula="parsed_browser_risk(user_agent)",
        live_formula="parsed_browser_risk(user_agent)",
        lookback_window=None,
        leakage_rule="Intrinsic current transaction attribute",
        nullable_behavior="0.0",
        required_raw_fields=["user_agent"],
        required_supabase_fields=[],
        production_status="NEEDS_DATA",
        reason="Browser client telemetry / User-Agent string is not passed in current JSON payload or extracted from FastAPI request headers.",
    ),
    FeatureContract(
        feature_name="card_velocity_surge",
        group="Velocity & Burst Features",
        data_type="float",
        training_source="Card 1h velocity vs 30d baseline",
        live_source="Card 1h velocity vs 30d baseline",
        training_formula="card_tx_count_1h / (card_tx_count_30d / 720.0 + 1e-5)",
        live_formula="card_tx_count_1h / (card_tx_count_30d / 720.0 + 1e-5)",
        lookback_window="30d",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="1.0",
        required_raw_fields=["card_id"],
        required_supabase_fields=["card_id", "transaction_timestamp"],
        production_status="NEEDS_DATA",
        reason="Requires retaining a rolling 30-day baseline hourly velocity; current Supabase state only tracks all-time aggregates.",
    ),
    FeatureContract(
        feature_name="device_velocity_surge",
        group="Velocity & Burst Features",
        data_type="float",
        training_source="Device 1h velocity vs 30d baseline",
        live_source="Device 1h velocity vs 30d baseline",
        training_formula="device_tx_count_1h / (device_tx_count_30d / 720.0 + 1e-5)",
        live_formula="device_tx_count_1h / (device_tx_count_30d / 720.0 + 1e-5)",
        lookback_window="30d",
        leakage_rule="Strict: tx.timestamp < current_timestamp",
        nullable_behavior="1.0",
        required_raw_fields=["device_id"],
        required_supabase_fields=["device_id", "transaction_timestamp"],
        production_status="NEEDS_DATA",
        reason="Requires 30-day rolling hardware baseline velocity; current Supabase state does not retain device-level rolling windows.",
    ),
]


# ============================================================================
# 4. PARTITIONED FEATURE SUITES
# ============================================================================

VALID_PRODUCTION_FEATURES: List[FeatureContract] = [
    f for f in ALL_AUDITED_FEATURES if f.production_status == "VALID"
]

NEEDS_DATA_FEATURES: List[FeatureContract] = [
    f for f in ALL_AUDITED_FEATURES if f.production_status == "NEEDS_DATA"
]

INVALID_FEATURES: List[FeatureContract] = [
    f for f in ALL_AUDITED_FEATURES if f.production_status == "INVALID"
]

VALID_FEATURE_NAMES: List[str] = [f.feature_name for f in VALID_PRODUCTION_FEATURES]


# ============================================================================
# 5. UNIFIED PRODUCTION FEATURE GENERATOR (SINGLE SOURCE OF TRUTH)
# ============================================================================

def generate_production_features(
    current_transaction: Dict[str, Any],
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, float]:
    """
    Unified, deterministic production feature generator.
    
    Used identically during:
    - Model Training Feature Engineering (training_feature_generator)
    - Live Inference Feature Engineering (live_feature_generator)
    
    Strict Anti-Leakage Guarantee:
    Only historical transactions with timestamp STRICTLY LESS than
    current_transaction['transaction_timestamp'] participate in feature calculations.
    """
    # 1. Parse Current Transaction Attributes
    curr_amount = float(current_transaction.get("amount", 0.0))
    curr_ts = parse_utc_timestamp(current_transaction.get("transaction_timestamp"))
    curr_cust = str(current_transaction.get("customer_id") or "")
    curr_card = str(current_transaction.get("card_id") or current_transaction.get("card1") or "")
    curr_device = current_transaction.get("device_id")
    curr_device_str = str(curr_device) if curr_device is not None and str(curr_device).strip() else None
    curr_merchant = current_transaction.get("merchant_id")
    curr_merchant_str = str(curr_merchant) if curr_merchant is not None and str(curr_merchant).strip() else None
    curr_ip = current_transaction.get("ip_address")
    curr_ip_str = str(curr_ip) if curr_ip is not None and str(curr_ip).strip() else None
    curr_product = str(current_transaction.get("product_code") or "W").upper()
    curr_card_net = str(current_transaction.get("card_network") or "visa").lower()
    curr_card_type = str(current_transaction.get("card_type") or "debit").lower()
    curr_email = str(current_transaction.get("email_domain") or "").lower()

    # 2. Strict Point-in-Time Anti-Leakage History Filtering
    # Exclude current transaction and any concurrent/future transactions
    clean_history: List[Dict[str, Any]] = []
    if history:
        for tx in history:
            tx_ts = parse_utc_timestamp(tx.get("transaction_timestamp"))
            if tx_ts < curr_ts:
                clean_history.append(
                    {
                        "amount": float(tx.get("amount", 0.0)),
                        "timestamp": tx_ts,
                        "customer_id": str(tx.get("customer_id") or ""),
                        "card_id": str(tx.get("card_id") or tx.get("card1") or ""),
                        "device_id": str(tx.get("device_id")) if tx.get("device_id") is not None else None,
                        "merchant_id": str(tx.get("merchant_id")) if tx.get("merchant_id") is not None else None,
                        "ip_address": str(tx.get("ip_address")) if tx.get("ip_address") is not None else None,
                    }
                )

    curr_epoch = curr_ts.timestamp()

    # Pre-slice history for customer, card, device, merchant, ip
    cust_history = [tx for tx in clean_history if tx["customer_id"] == curr_cust]
    card_history = [tx for tx in clean_history if tx["card_id"] == curr_card]
    dev_history = [tx for tx in clean_history if curr_device_str and tx["device_id"] == curr_device_str]
    merch_history = [tx for tx in clean_history if curr_merchant_str and tx["merchant_id"] == curr_merchant_str]
    ip_history = [tx for tx in clean_history if curr_ip_str and tx["ip_address"] == curr_ip_str]

    features: Dict[str, float] = {}

    # ------------------------------------------------------------------------
    # Group 1: Direct Transaction Features
    # ------------------------------------------------------------------------
    features["amount"] = curr_amount
    features["product_code_freq"] = DEFAULT_PRODUCT_FREQ.get(curr_product, 0.01)
    features["card_network_freq"] = DEFAULT_CARD_NETWORK_FREQ.get(curr_card_net, 0.01)
    features["card_type_freq"] = DEFAULT_CARD_TYPE_FREQ.get(curr_card_type, 0.01)
    features["email_domain_freq"] = DEFAULT_EMAIL_DOMAIN_FREQ.get(curr_email, 0.01)

    # Missing fields count
    opt_fields = [
        current_transaction.get("device_id"),
        current_transaction.get("merchant_id"),
        current_transaction.get("email_domain"),
        current_transaction.get("address_id"),
        current_transaction.get("ip_address"),
        current_transaction.get("country"),
        current_transaction.get("card_network"),
        current_transaction.get("card_type"),
    ]
    features["missing_fields_count"] = float(sum(1 for f in opt_fields if f is None or str(f).strip() == ""))

    # ------------------------------------------------------------------------
    # Group 2: Time & Calendar Features
    # ------------------------------------------------------------------------
    features["hour"] = float(curr_ts.hour)
    features["weekday_index"] = float(curr_ts.weekday())
    features["is_weekend"] = 1.0 if curr_ts.weekday() in (5, 6) else 0.0
    features["is_night"] = 1.0 if curr_ts.hour < 6 else 0.0
    features["day_of_year"] = float(curr_ts.timetuple().tm_yday)
    features["time_since_midnight_sec"] = float(curr_ts.hour * 3600 + curr_ts.minute * 60 + curr_ts.second)

    # ------------------------------------------------------------------------
    # Group 3: Amount Transformation Features
    # ------------------------------------------------------------------------
    features["amount_log"] = float(math.log1p(max(0.0, curr_amount)))
    features["amount_cents"] = float(round(curr_amount % 1.0, 2))
    features["is_round_amount"] = 1.0 if (curr_amount >= 10.0 and (curr_amount % 10.0 == 0.0)) else 0.0
    features["is_zero_cents"] = 1.0 if ((curr_amount % 1.0) == 0.0) else 0.0

    # ------------------------------------------------------------------------
    # Group 4: Customer Behavioral History
    # ------------------------------------------------------------------------
    cust_5m = [tx for tx in cust_history if (curr_epoch - tx["timestamp"].timestamp()) <= 300.0]
    cust_1h = [tx for tx in cust_history if (curr_epoch - tx["timestamp"].timestamp()) <= 3600.0]
    cust_24h = [tx for tx in cust_history if (curr_epoch - tx["timestamp"].timestamp()) <= 86400.0]

    features["customer_tx_count_5m"] = float(len(cust_5m))
    features["customer_tx_count_1h"] = float(len(cust_1h))
    features["customer_tx_count_24h"] = float(len(cust_24h))

    cust_amounts = [tx["amount"] for tx in cust_history]
    cust_cnt = len(cust_amounts)
    if cust_cnt > 0:
        cust_mean = float(sum(cust_amounts) / cust_cnt)
        features["customer_amount_mean"] = cust_mean
        features["customer_amount_ratio"] = float(curr_amount / (cust_mean + 1e-5))
        features["customer_is_new"] = 0.0
        if cust_cnt >= 2:
            var_val = sum((a - cust_mean) ** 2 for a in cust_amounts) / cust_cnt
            cust_std = float(math.sqrt(max(0.0, var_val)))
            features["customer_amount_std"] = cust_std
            features["customer_amount_zscore"] = float((curr_amount - cust_mean) / (cust_std + 1e-5)) if cust_std > 0 else 0.0
        else:
            features["customer_amount_std"] = 0.0
            features["customer_amount_zscore"] = 0.0
    else:
        features["customer_amount_mean"] = 0.0
        features["customer_amount_std"] = 0.0
        features["customer_amount_zscore"] = 0.0
        features["customer_amount_ratio"] = 1.0
        features["customer_is_new"] = 1.0

    cust_merchants_24h = {tx["merchant_id"] for tx in cust_24h if tx["merchant_id"]}
    features["customer_unique_merchants_24h"] = float(len(cust_merchants_24h))

    cust_devices_24h = {tx["device_id"] for tx in cust_24h if tx["device_id"]}
    features["customer_unique_devices_24h"] = float(len(cust_devices_24h))

    # ------------------------------------------------------------------------
    # Group 5: Card Behavioral History
    # ------------------------------------------------------------------------
    card_5m = [tx for tx in card_history if (curr_epoch - tx["timestamp"].timestamp()) <= 300.0]
    card_1h = [tx for tx in card_history if (curr_epoch - tx["timestamp"].timestamp()) <= 3600.0]
    card_24h = [tx for tx in card_history if (curr_epoch - tx["timestamp"].timestamp()) <= 86400.0]

    features["card_tx_count_5m"] = float(len(card_5m))
    features["card_tx_count_1h"] = float(len(card_1h))
    features["card_tx_count_24h"] = float(len(card_24h))

    card_amounts = [tx["amount"] for tx in card_history]
    card_cnt = len(card_amounts)
    if card_cnt > 0:
        card_mean = float(sum(card_amounts) / card_cnt)
        features["card_amount_mean"] = card_mean
        features["card_amount_ratio"] = float(curr_amount / (card_mean + 1e-5))
        features["card_is_new"] = 0.0
        if card_cnt >= 2:
            var_card = sum((a - card_mean) ** 2 for a in card_amounts) / card_cnt
            card_std = float(math.sqrt(max(0.0, var_card)))
            features["card_amount_std"] = card_std
        else:
            features["card_amount_std"] = 0.0
    else:
        features["card_amount_mean"] = 0.0
        features["card_amount_std"] = 0.0
        features["card_amount_ratio"] = 1.0
        features["card_is_new"] = 1.0

    # ------------------------------------------------------------------------
    # Group 6: Device Fingerprint History
    # ------------------------------------------------------------------------
    if curr_device_str:
        dev_5m = [tx for tx in dev_history if (curr_epoch - tx["timestamp"].timestamp()) <= 300.0]
        dev_1h = [tx for tx in dev_history if (curr_epoch - tx["timestamp"].timestamp()) <= 3600.0]
        dev_24h = [tx for tx in dev_history if (curr_epoch - tx["timestamp"].timestamp()) <= 86400.0]

        features["device_tx_count_5m"] = float(len(dev_5m))
        features["device_tx_count_1h"] = float(len(dev_1h))
        features["device_tx_count_24h"] = float(len(dev_24h))
        features["device_unique_customers_24h"] = float(len({tx["customer_id"] for tx in dev_24h if tx["customer_id"]}))
        features["device_unique_cards_24h"] = float(len({tx["card_id"] for tx in dev_24h if tx["card_id"]}))
        features["device_is_new"] = 1.0 if len(dev_history) == 0 else 0.0
    else:
        features["device_tx_count_5m"] = 0.0
        features["device_tx_count_1h"] = 0.0
        features["device_tx_count_24h"] = 0.0
        features["device_unique_customers_24h"] = 0.0
        features["device_unique_cards_24h"] = 0.0
        features["device_is_new"] = 0.0

    # ------------------------------------------------------------------------
    # Group 7: Merchant Risk & History
    # ------------------------------------------------------------------------
    if curr_merchant_str:
        merch_1h = [tx for tx in merch_history if (curr_epoch - tx["timestamp"].timestamp()) <= 3600.0]
        merch_24h = [tx for tx in merch_history if (curr_epoch - tx["timestamp"].timestamp()) <= 86400.0]

        features["merchant_tx_count_1h"] = float(len(merch_1h))
        features["merchant_tx_count_24h"] = float(len(merch_24h))

        merch_amounts = [tx["amount"] for tx in merch_history]
        m_cnt = len(merch_amounts)
        if m_cnt > 0:
            m_mean = float(sum(merch_amounts) / m_cnt)
            features["merchant_amount_mean"] = m_mean
            if m_cnt >= 2:
                m_var = sum((a - m_mean) ** 2 for a in merch_amounts) / m_cnt
                features["merchant_amount_std"] = float(math.sqrt(max(0.0, m_var)))
            else:
                features["merchant_amount_std"] = 0.0
        else:
            features["merchant_amount_mean"] = 0.0
            features["merchant_amount_std"] = 0.0

        cust_merch_txs = [tx for tx in cust_history if tx["merchant_id"] == curr_merchant_str]
        features["customer_merchant_tx_count"] = float(len(cust_merch_txs))
        features["customer_merchant_is_new"] = 1.0 if len(cust_merch_txs) == 0 else 0.0
    else:
        features["merchant_tx_count_1h"] = 0.0
        features["merchant_tx_count_24h"] = 0.0
        features["merchant_amount_mean"] = 0.0
        features["merchant_amount_std"] = 0.0
        features["customer_merchant_tx_count"] = 0.0
        features["customer_merchant_is_new"] = 0.0

    # ------------------------------------------------------------------------
    # Group 8: Client IP Network History
    # ------------------------------------------------------------------------
    if curr_ip_str:
        ip_5m = [tx for tx in ip_history if (curr_epoch - tx["timestamp"].timestamp()) <= 300.0]
        ip_1h = [tx for tx in ip_history if (curr_epoch - tx["timestamp"].timestamp()) <= 3600.0]
        ip_24h = [tx for tx in ip_history if (curr_epoch - tx["timestamp"].timestamp()) <= 86400.0]

        features["ip_tx_count_5m"] = float(len(ip_5m))
        features["ip_tx_count_1h"] = float(len(ip_1h))
        features["ip_tx_count_24h"] = float(len(ip_24h))
        features["ip_unique_customers_24h"] = float(len({tx["customer_id"] for tx in ip_24h if tx["customer_id"]}))
        features["ip_unique_cards_24h"] = float(len({tx["card_id"] for tx in ip_24h if tx["card_id"]}))
        features["ip_unique_devices_24h"] = float(len({tx["device_id"] for tx in ip_24h if tx["device_id"]}))
        features["ip_is_new"] = 1.0 if len(ip_history) == 0 else 0.0
    else:
        features["ip_tx_count_5m"] = 0.0
        features["ip_tx_count_1h"] = 0.0
        features["ip_tx_count_24h"] = 0.0
        features["ip_unique_customers_24h"] = 0.0
        features["ip_unique_cards_24h"] = 0.0
        features["ip_unique_devices_24h"] = 0.0
        features["ip_is_new"] = 0.0

    # ------------------------------------------------------------------------
    # Group 9: Cross-Entity Relationship Features
    # ------------------------------------------------------------------------
    features["customer_device_seen_before"] = (
        1.0 if curr_device_str and any(tx["customer_id"] == curr_cust and tx["device_id"] == curr_device_str for tx in clean_history) else 0.0
    )
    features["customer_card_seen_before"] = (
        1.0 if any(tx["customer_id"] == curr_cust and tx["card_id"] == curr_card for tx in clean_history) else 0.0
    )
    features["customer_merchant_seen_before"] = (
        1.0 if curr_merchant_str and any(tx["customer_id"] == curr_cust and tx["merchant_id"] == curr_merchant_str for tx in clean_history) else 0.0
    )
    features["customer_ip_seen_before"] = (
        1.0 if curr_ip_str and any(tx["customer_id"] == curr_cust and tx["ip_address"] == curr_ip_str for tx in clean_history) else 0.0
    )
    features["card_device_seen_before"] = (
        1.0 if curr_device_str and any(tx["card_id"] == curr_card and tx["device_id"] == curr_device_str for tx in clean_history) else 0.0
    )
    features["card_ip_seen_before"] = (
        1.0 if curr_ip_str and any(tx["card_id"] == curr_card and tx["ip_address"] == curr_ip_str for tx in clean_history) else 0.0
    )

    # ------------------------------------------------------------------------
    # Group 10: Velocity & Burst Features
    # ------------------------------------------------------------------------
    if cust_history:
        # Most recent past transaction for this customer
        last_tx_time = max(tx["timestamp"] for tx in cust_history)
        delta_sec = max(0.0, curr_epoch - last_tx_time.timestamp())
        features["rapid_transaction_flag"] = 1.0 if delta_sec <= 60.0 else 0.0
    else:
        features["rapid_transaction_flag"] = 0.0

    cust_15m = [tx for tx in cust_history if (curr_epoch - tx["timestamp"].timestamp()) <= 900.0]
    features["transactions_last_15m"] = float(len(cust_15m))

    features["amount_sum_last_1h"] = float(sum(tx["amount"] for tx in cust_1h))
    features["amount_sum_last_24h"] = float(sum(tx["amount"] for tx in cust_24h))

    # Sanity check: Ensure every VALID feature name is produced and is non-NaN float
    for name in VALID_FEATURE_NAMES:
        if name not in features:
            features[name] = 0.0
        val = features[name]
        if math.isnan(val) or math.isinf(val):
            features[name] = 0.0

    return features


def training_feature_generator(
    history: List[Dict[str, Any]],
    current_transaction: Dict[str, Any],
) -> Dict[str, float]:
    """
    Feature generator entrypoint for model training workflows.
    Delegates to the single source of truth generator.
    """
    return generate_production_features(current_transaction, history)


def live_feature_generator(
    history: List[Dict[str, Any]],
    current_transaction: Dict[str, Any],
) -> Dict[str, float]:
    """
    Feature generator entrypoint for live inference workflows.
    Delegates to the single source of truth generator.
    """
    return generate_production_features(current_transaction, history)
