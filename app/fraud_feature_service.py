from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .config import FRAUD_MODEL_PATH
from .supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)


def parse_timestamp(ts: Any) -> datetime:
    """Parses various timestamp inputs into a timezone-aware UTC datetime."""
    if isinstance(ts, datetime):
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
    if isinstance(ts, str):
        clean_ts = ts.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(clean_ts)
        except Exception:
            return pd.to_datetime(ts, utc=True).to_pydatetime()
    return datetime.now(timezone.utc)


class FraudFeatureService:
    """
    Singleton service managing feature engineering for the Phase 2 Fraud XGBoost Model.
    Reproduces the exact 459-feature schema derived from IEEE-CIS training artifacts.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or FRAUD_MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(f"Phase 2 fraud champion model not found at {self.model_path}")

        logger.info(f"Loading Phase 2 Fraud champion artifact from {self.model_path}")
        artifact = joblib.load(self.model_path)

        if not isinstance(artifact, dict) or "feature_names" not in artifact or "preprocessor" not in artifact:
            raise ValueError("Corrupted Phase 2 model artifact: missing 'feature_names' or 'preprocessor'.")

        self.feature_names: List[str] = artifact["feature_names"]
        self.preprocessor: Dict[str, Any] = artifact["preprocessor"]

        self.categorical_cols: List[str] = self.preprocessor.get("categorical_cols", [])
        self.numeric_cols: List[str] = self.preprocessor.get("numeric_cols", [])
        self.frequency_maps: Dict[str, Dict[str, float]] = self.preprocessor.get("frequency_maps", {})
        self.numeric_fill: Dict[str, float] = self.preprocessor.get("numeric_fill", {})

        if len(self.feature_names) != 459:
            raise ValueError(
                f"Feature schema mismatch! Expected 459 features, but artifact contains {len(self.feature_names)}."
            )

    def get_frequency_value(self, col_name: str, raw_value: Any) -> float:
        """Looks up frequency encoding for a categorical column, defaulting to __MISSING__ weight."""
        fm = self.frequency_maps.get(col_name, {})
        if raw_value is None or raw_value == "":
            return float(fm.get("__MISSING__", 0.0))

        val_str = str(raw_value)
        if val_str in fm:
            return float(fm[val_str])

        val_lower = val_str.lower()
        for k, v in fm.items():
            if str(k).lower() == val_lower:
                return float(v)

        return float(fm.get("__MISSING__", 0.0))

    def compute_entity_history(
        self,
        entity_type: str,
        entity_key: str,
        current_timestamp: datetime,
        current_amount: float,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """
        Calculates anti-leakage historical statistics for an entity strictly before current_timestamp.
        Uses in-memory history slice if provided; otherwise inspects fraud_entity_state or fraud_transactions.
        """
        past_amounts: List[float] = []

        if history is not None:
            # Strictly filter passed historical records where timestamp < current_timestamp
            for txn in history:
                txn_ts = parse_timestamp(txn.get("transaction_timestamp"))
                if txn_ts >= current_timestamp:
                    continue  # Anti-leakage: ignore concurrent or future transactions

                match = False
                if entity_type == "customer":
                    match = str(txn.get("customer_id", "")) == str(entity_key)
                elif entity_type == "card":
                    card_val = str(txn.get("card_id") or txn.get("card1") or "")
                    match = card_val == str(entity_key)
                elif entity_type == "card_addr":
                    c_val = str(txn.get("card_id") or txn.get("card1") or "")
                    a_val = str(txn.get("address_id") or txn.get("addr1") or "")
                    match = f"{c_val}_{a_val}" == str(entity_key)

                if match:
                    past_amounts.append(float(txn.get("amount", 0.0)))

        else:
            # Query Supabase via admin client
            try:
                admin = get_supabase_admin_client()

                # 1. Check fraud_entity_state if valid point-in-time
                res_state = (
                    admin.table("fraud_entity_state")
                    .select("*")
                    .eq("entity_type", entity_type)
                    .eq("entity_key", entity_key)
                    .execute()
                )
                if res_state.data:
                    row = res_state.data[0]
                    last_seen = parse_timestamp(row.get("last_seen"))
                    # If entity state is strictly prior to current transaction, we can use $O(1)$ aggregates
                    if last_seen < current_timestamp:
                        cnt = float(row.get("transaction_count", 0))
                        s = float(row.get("amount_sum", 0.0))
                        s2 = float(row.get("amount_sq_sum", 0.0))
                        if cnt > 0:
                            mean_val = s / cnt
                            var_val = max(0.0, (s2 / cnt) - (mean_val**2))
                            std_val = float(np.sqrt(var_val))
                            return {
                                "past_count": cnt,
                                "past_mean_amt": float(mean_val),
                                "past_std_amt": std_val,
                                "amount_ratio": float(current_amount / (mean_val + 1e-5)),
                                "is_new": 0.0,
                            }

                # 2. Fallback to point-in-time query on fraud_transactions with strict timestamp filter
                field_col = "customer_id" if entity_type == "customer" else ("card_id" if entity_type == "card" else "address_id")
                q = (
                    admin.table("fraud_transactions")
                    .select("amount, transaction_timestamp")
                    .lt("transaction_timestamp", current_timestamp.isoformat())
                )
                if entity_type in ("customer", "card"):
                    q = q.eq(field_col, entity_key)
                
                rows = q.execute().data
                if entity_type == "card_addr":
                    parts = str(entity_key).split("_")
                    c_target = parts[0] if len(parts) > 0 else ""
                    a_target = parts[1] if len(parts) > 1 else ""
                    rows = [r for r in rows if str(r.get("card_id", "")) == c_target and str(r.get("address_id", "")) == a_target]

                past_amounts = [float(r.get("amount", 0.0)) for r in rows]

            except Exception as exc:
                logger.warning(f"Historical lookup fallback triggered: {exc}")
                past_amounts = []

        N = len(past_amounts)
        if N == 0:
            return {
                "past_count": 0.0,
                "past_mean_amt": float(current_amount),
                "past_std_amt": 0.0,
                "amount_ratio": 1.0,
                "is_new": 1.0,
            }

        mean_amt = float(np.mean(past_amounts))
        std_amt = float(np.std(past_amounts)) if N > 1 else 0.0
        return {
            "past_count": float(N),
            "past_mean_amt": mean_amt,
            "past_std_amt": std_amt,
            "amount_ratio": float(current_amount / (mean_amt + 1e-5)),
            "is_new": 0.0,
        }

    def build_fraud_features(
        self,
        transaction: Dict[str, Any],
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """
        Converts raw transaction attributes + anti-leakage historical context into the
        exact 459 model-ready features.
        """
        dt = parse_timestamp(transaction.get("transaction_timestamp"))
        amount = float(transaction.get("amount", 0.0))

        # Basic identity attributes
        customer_id = str(transaction.get("customer_id") or "UNKNOWN_CUST")
        card_id = str(transaction.get("card_id") or transaction.get("card1") or "9633")
        address_id = str(transaction.get("address_id") or transaction.get("addr1") or "299.0")
        product_code = str(transaction.get("product_code") or transaction.get("ProductCD") or "W")
        email_domain = str(transaction.get("email_domain") or transaction.get("P_emaildomain") or "gmail.com")
        card_network = str(transaction.get("card_network") or transaction.get("card4") or "visa")
        card_type = str(transaction.get("card_type") or transaction.get("card6") or "debit")
        device_id = str(transaction.get("device_id") or transaction.get("DeviceInfo") or "__MISSING__")
        device_type = str(transaction.get("device_type") or transaction.get("DeviceType") or "__MISSING__")

        # Numeric conversions with training medians fallback
        try:
            card1_num = float(card_id)
        except Exception:
            card1_num = float(self.numeric_fill.get("card1", 9633.0))

        try:
            addr1_num = float(address_id)
        except Exception:
            addr1_num = float(self.numeric_fill.get("addr1", 299.0))

        card1_addr1_key = f"{card_id}_{address_id}"

        # 1. Historical Anti-Leakage Feature Extraction
        uid_stats = self.compute_entity_history("customer", customer_id, dt, amount, history)
        card1_stats = self.compute_entity_history("card", card_id, dt, amount, history)
        card1_addr1_stats = self.compute_entity_history("card_addr", card1_addr1_key, dt, amount, history)

        # 2. Derived Date/Time & Amount Features
        hour = float(dt.hour)
        weekday_index = float(dt.weekday())
        is_weekend = 1.0 if weekday_index in (5, 6) else 0.0
        is_night = 1.0 if hour < 6 else 0.0
        day_index = float(dt.timetuple().tm_yday)
        amt_log = float(np.log1p(amount))
        amt_cents = float(round(amount % 1, 2))
        
        # TransactionDT in seconds
        explicit_dt = transaction.get("TransactionDT")
        txn_dt_seconds = float(explicit_dt) if explicit_dt is not None else float(int(dt.timestamp()))

        # Interaction keys
        card1_prod_key = f"{int(card1_num)}_{product_code}"
        card1_email_key = f"{int(card1_num)}_{email_domain}"
        card1_addr_key = f"{int(card1_num)}_{addr1_num}"
        uid_key = f"{product_code}_{int(card1_num)}_{addr1_num}_0.0"

        # 3. Assemble Feature Vector
        row: Dict[str, float] = {}

        for col in self.feature_names:
            # A. Categorical / Frequency-Encoded Features
            if col.endswith("__freq"):
                base_col = col[:-6]
                if base_col == "ProductCD":
                    row[col] = self.get_frequency_value("ProductCD", product_code)
                elif base_col == "card4":
                    row[col] = self.get_frequency_value("card4", card_network)
                elif base_col == "card6":
                    row[col] = self.get_frequency_value("card6", card_type)
                elif base_col == "P_emaildomain":
                    row[col] = self.get_frequency_value("P_emaildomain", email_domain)
                elif base_col == "R_emaildomain":
                    row[col] = self.get_frequency_value("R_emaildomain", transaction.get("R_emaildomain"))
                elif base_col == "DeviceType":
                    row[col] = self.get_frequency_value("DeviceType", device_type)
                elif base_col == "DeviceInfo":
                    row[col] = self.get_frequency_value("DeviceInfo", device_id)
                elif base_col == "card1_ProductCD":
                    row[col] = self.get_frequency_value("card1_ProductCD", card1_prod_key)
                elif base_col == "card1_email":
                    row[col] = self.get_frequency_value("card1_email", card1_email_key)
                elif base_col == "card1_addr1":
                    row[col] = self.get_frequency_value("card1_addr1", card1_addr_key)
                elif base_col == "uid":
                    row[col] = self.get_frequency_value("uid", uid_key)
                else:
                    # Match / identity columns like M1..M9, id_12..id_38
                    val = transaction.get(base_col)
                    row[col] = self.get_frequency_value(base_col, val)

            # B. Historical Features
            elif col == "uid_past_count":
                row[col] = uid_stats["past_count"]
            elif col == "uid_past_mean_amt":
                row[col] = uid_stats["past_mean_amt"]
            elif col == "uid_past_std_amt":
                row[col] = uid_stats["past_std_amt"]
            elif col == "uid_amount_ratio":
                row[col] = uid_stats["amount_ratio"]
            elif col == "uid_is_new":
                row[col] = uid_stats["is_new"]
            elif col == "card1_past_count":
                row[col] = card1_stats["past_count"]
            elif col == "card1_past_mean_amt":
                row[col] = card1_stats["past_mean_amt"]
            elif col == "card1_past_std_amt":
                row[col] = card1_stats["past_std_amt"]
            elif col == "card1_amount_ratio":
                row[col] = card1_stats["amount_ratio"]
            elif col == "card1_is_new":
                row[col] = card1_stats["is_new"]
            elif col == "card1_addr1_past_count":
                row[col] = card1_addr1_stats["past_count"]
            elif col == "card1_addr1_past_mean_amt":
                row[col] = card1_addr1_stats["past_mean_amt"]
            elif col == "card1_addr1_past_std_amt":
                row[col] = card1_addr1_stats["past_std_amt"]
            elif col == "card1_addr1_amount_ratio":
                row[col] = card1_addr1_stats["amount_ratio"]
            elif col == "card1_addr1_is_new":
                row[col] = card1_addr1_stats["is_new"]

            # C. Derived Date/Time & Amount Features
            elif col == "TransactionAmt":
                row[col] = amount
            elif col == "TransactionDT":
                row[col] = txn_dt_seconds
            elif col == "TransactionAmt_log":
                row[col] = amt_log
            elif col == "TransactionAmt_cents":
                row[col] = amt_cents
            elif col == "hour":
                row[col] = hour
            elif col == "weekday_index":
                row[col] = weekday_index
            elif col == "is_weekend":
                row[col] = is_weekend
            elif col == "is_night":
                row[col] = is_night
            elif col == "day_index":
                row[col] = day_index
            elif col == "card1":
                row[col] = card1_num
            elif col == "addr1":
                row[col] = addr1_num
            elif col == "missing_count":
                # Number of unavailable features imputed by default
                provided_count = sum(1 for v in transaction.values() if v is not None and v != "")
                row[col] = float(max(0, len(self.feature_names) - provided_count))

            # D. Explicitly Passed Numeric Columns
            elif col in transaction and transaction[col] is not None:
                try:
                    row[col] = float(transaction[col])
                except Exception:
                    row[col] = float(self.numeric_fill.get(col, 0.0))

            # E. Unavailable IEEE-CIS Numeric Features (V-cols, C-cols, D-cols, id-cols)
            else:
                row[col] = float(self.numeric_fill.get(col, 0.0))

        # Build DataFrame with strict column order
        df = pd.DataFrame([row], columns=self.feature_names)

        # Validation assertions
        if df.shape[1] != 459:
            raise ValueError(f"Feature engineering failed: produced {df.shape[1]} columns, expected 459.")
        if list(df.columns) != self.feature_names:
            raise ValueError("Feature columns order or composition deviates from champion feature schema!")
        if df.isna().any().any():
            nan_cols = df.columns[df.isna().any()].tolist()
            raise ValueError(f"Feature matrix contains unexpected NaN values in columns: {nan_cols}")
        if np.isinf(df.values).any():
            inf_cols = df.columns[np.isinf(df.values).any(axis=0)].tolist()
            raise ValueError(f"Feature matrix contains infinite values in columns: {inf_cols}")

        return df


# Singleton instance
_fraud_feature_service: Optional[FraudFeatureService] = None


def get_fraud_feature_service() -> FraudFeatureService:
    """Returns cached singleton FraudFeatureService instance."""
    global _fraud_feature_service
    if _fraud_feature_service is None:
        _fraud_feature_service = FraudFeatureService()
    return _fraud_feature_service


def build_fraud_features(
    transaction: Dict[str, Any],
    history: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Public functional API to generate 459-feature vector from transaction input."""
    service = get_fraud_feature_service()
    return service.build_fraud_features(transaction, history=history)


def update_entity_state(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-prediction service updating historical aggregates in public.fraud_entity_state.
    Maintains transaction_count, amount_sum, amount_sq_sum, and timestamps.
    Kept strictly separate from feature generation.
    """
    admin = get_supabase_admin_client()
    ts = parse_timestamp(transaction.get("transaction_timestamp")).isoformat()
    amount = float(transaction.get("amount", 0.0))

    customer_id = str(transaction.get("customer_id") or "")
    card_id = str(transaction.get("card_id") or transaction.get("card1") or "")
    addr_id = str(transaction.get("address_id") or transaction.get("addr1") or "")
    merchant_id = transaction.get("merchant_id")
    device_id = transaction.get("device_id")

    entities = []
    if customer_id:
        entities.append(("customer", customer_id))
    if card_id:
        entities.append(("card", card_id))
    if card_id and addr_id:
        entities.append(("card_addr", f"{card_id}_{addr_id}"))

    updated_count = 0
    now_iso = datetime.now(timezone.utc).isoformat()

    for etype, ekey in entities:
        res = (
            admin.table("fraud_entity_state")
            .select("*")
            .eq("entity_type", etype)
            .eq("entity_key", ekey)
            .execute()
        )
        if res.data:
            rec = res.data[0]
            new_cnt = int(rec["transaction_count"]) + 1
            new_sum = float(rec["amount_sum"]) + amount
            new_sq_sum = float(rec["amount_sq_sum"]) + (amount**2)
            admin.table("fraud_entity_state").update(
                {
                    "transaction_count": new_cnt,
                    "amount_sum": new_sum,
                    "amount_sq_sum": new_sq_sum,
                    "last_seen": ts,
                    "last_amount": amount,
                    "updated_at": now_iso,
                }
            ).eq("id", rec["id"]).execute()
            updated_count += 1
        else:
            admin.table("fraud_entity_state").insert(
                {
                    "entity_type": etype,
                    "entity_key": ekey,
                    "transaction_count": 1,
                    "amount_sum": amount,
                    "amount_sq_sum": amount**2,
                    "first_seen": ts,
                    "last_seen": ts,
                    "last_amount": amount,
                    "unique_merchant_count": 1 if merchant_id else 0,
                    "unique_device_count": 1 if device_id else 0,
                    "updated_at": now_iso,
                }
            ).execute()
            updated_count += 1

    return {"status": "success", "entities_updated": updated_count}


def fetch_historical_transactions(
    current_transaction: Dict[str, Any],
    client: Optional[Any] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """
    Retrieves historical transactions strictly prior to the current transaction timestamp
    from Supabase public.fraud_transactions for Phase 2 62-feature production calculation.

    Anti-Leakage Invariant:
    Enforces transaction_timestamp < current_transaction_timestamp at the database level.
    Never returns the current transaction or future/concurrent events.

    Returns an empty list [] on zero prior history (cold start) or on database lookup failure.
    """
    curr_ts = parse_timestamp(current_transaction.get("transaction_timestamp"))
    curr_ts_iso = curr_ts.isoformat()

    cust_id = str(current_transaction.get("customer_id") or "").strip()
    card_id = str(current_transaction.get("card_id") or current_transaction.get("card1") or "").strip()
    device_id = current_transaction.get("device_id")
    device_str = str(device_id).strip() if device_id is not None and str(device_id).strip() else None
    ip_addr = current_transaction.get("ip_address")
    ip_str = str(ip_addr).strip() if ip_addr is not None and str(ip_addr).strip() else None

    # If no identifiers provided, no entity history to look up
    if not cust_id and not card_id and not device_str and not ip_str:
        return []

    admin = client or get_supabase_admin_client()
    merged_history: Dict[str, Dict[str, Any]] = {}

    try:
        # 1. Query by customer_id
        if cust_id:
            res_cust = (
                admin.table("fraud_transactions")
                .select("*")
                .eq("customer_id", cust_id)
                .lt("transaction_timestamp", curr_ts_iso)
                .order("transaction_timestamp", desc=True)
                .limit(limit)
                .execute()
            )
            for row in (res_cust.data or []):
                tid = row.get("transaction_id")
                if tid:
                    merged_history[tid] = row

        # 2. Query by card_id
        if card_id:
            res_card = (
                admin.table("fraud_transactions")
                .select("*")
                .eq("card_id", card_id)
                .lt("transaction_timestamp", curr_ts_iso)
                .order("transaction_timestamp", desc=True)
                .limit(limit)
                .execute()
            )
            for row in (res_card.data or []):
                tid = row.get("transaction_id")
                if tid and tid not in merged_history:
                    merged_history[tid] = row

        # 3. Query by device_id
        if device_str:
            res_dev = (
                admin.table("fraud_transactions")
                .select("*")
                .eq("device_id", device_str)
                .lt("transaction_timestamp", curr_ts_iso)
                .order("transaction_timestamp", desc=True)
                .limit(limit)
                .execute()
            )
            for row in (res_dev.data or []):
                tid = row.get("transaction_id")
                if tid and tid not in merged_history:
                    merged_history[tid] = row

        # 4. Query by ip_address
        if ip_str:
            res_ip = (
                admin.table("fraud_transactions")
                .select("*")
                .eq("ip_address", ip_str)
                .lt("transaction_timestamp", curr_ts_iso)
                .order("transaction_timestamp", desc=True)
                .limit(limit)
                .execute()
            )
            for row in (res_ip.data or []):
                tid = row.get("transaction_id")
                if tid and tid not in merged_history:
                    merged_history[tid] = row

    except Exception as exc:
        logger.warning(f"fetch_historical_transactions encountered error: {exc}")
        return []

    # Sort chronological ascending
    results = list(merged_history.values())
    results.sort(key=lambda x: str(x.get("transaction_timestamp") or ""))
    return results
