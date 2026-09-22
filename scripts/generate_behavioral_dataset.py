"""
scripts/generate_behavioral_dataset.py

AegisFin Phase 2: Scalable Behavioral Transaction Dataset Generator (v2.1.0).

Generates raw chronological transactions simulating realistic payment behavior
(Legitimate Stable, Legitimate High-Value, Shared Device/IP Ring, Account Takeover,
Card Testing, Stolen Card, and Burst Fraud), then transforms each transaction
through the frozen single-source-of-truth production feature engine:
`from app.production_feature_definitions import generate_production_features`

Features & Realism Enhancements (v2):
- Configurable dataset size (--n-transactions, e.g. 10000, 100000)
- Configurable random seed (--seed, default: 42)
- Configurable generator version (--version v2 [realistic] or v1 [legacy])
- Natural overlap between legitimate and fraud distributions:
  * Legitimate high-velocity behavior (household shared devices, corporate NAT/Wi-Fi IPs, shopping bursts)
  * Legitimate micro-purchases ($0.99 - $4.99 digital subscriptions/snacks)
  * Seamless continuous amounts (eliminating the artificial $250-$800 gap)
  * Stealth Account Takeover (single isolated transactions, moderate amounts, normal cadence)
  * Stealth Card Testing (isolated low-velocity tests with rotating IPs)
  * Legitimate unfamiliar entity relationships (travel IPs, new phones, cookie clearing)
  * Realistic telemetry missingness (privacy browsing, ad-blockers)
- Strictly chronological point-in-time entity history (t_hist < t_curr)
- Automated saving of raw CSV, production features CSV, and metadata JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Ensure project root is in python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.production_feature_definitions import (
    VALID_FEATURE_NAMES,
    generate_production_features,
)

DATA_DIR = BASE_DIR / "data" / "behavioral"
GENERATOR_VERSION = "2.1.0"


@dataclass
class GeneratorConfig:
    """Configuration specification for behavioral transaction generation."""
    n_transactions: int = 10000
    random_seed: int = 42
    # Target proportions for the 7 scenarios (must sum to 1.0)
    weight_legit_stable: float = 0.880
    weight_legit_high_value: float = 0.035
    weight_shared_device_ring: float = 0.020
    weight_account_takeover: float = 0.018
    weight_card_testing: float = 0.017
    weight_stolen_card: float = 0.015
    weight_burst_fraud: float = 0.015
    # Entity scaling ratios relative to n_transactions:
    customer_ratio: float = 0.080   # 800 for 10k, 8,000 for 100k
    merchant_ratio: float = 0.012   # 120 for 10k, 1,200 for 100k
    time_span_days: float = 45.0
    output_suffix: Optional[str] = None
    version: str = "v2"  # "v2" for realistic overlapping behavior, "v1" for legacy

    def target_fraud_rate(self) -> float:
        return (
            self.weight_shared_device_ring
            + self.weight_account_takeover
            + self.weight_card_testing
            + self.weight_stolen_card
            + self.weight_burst_fraud
        )

    def get_suffix(self) -> str:
        if self.output_suffix:
            return self.output_suffix.strip()
        ver_tag = "_v2" if self.version == "v2" else ""
        if self.n_transactions >= 1_000_000 and self.n_transactions % 1_000_000 == 0:
            return f"{self.n_transactions // 1_000_000}m{ver_tag}"
        if self.n_transactions >= 1_000 and self.n_transactions % 1_000 == 0:
            return f"{self.n_transactions // 1_000}k{ver_tag}"
        return f"{self.n_transactions}{ver_tag}"


def setup_simulation_entities(rng: random.Random, config: GeneratorConfig) -> Dict[str, Any]:
    """Pre-configures stable pools of customers, merchants, devices, corporate IPs, and fraudsters."""
    email_domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com", "aol.com"]
    card_networks = ["visa", "mastercard", "discover", "american express"]
    product_codes = ["W", "H", "R", "C", "S"]

    num_customers = max(800, int(round(config.n_transactions * config.customer_ratio)))
    num_merchants = max(120, int(round(config.n_transactions * config.merchant_ratio)))

    # Realism pools for v2:
    # 1. Household shared devices (group ~8% of customers into multi-user households)
    num_households = max(10, int(round(num_customers * 0.04)))
    household_devices = [f"DEV_HOUSEHOLD_{h:04d}" for h in range(1, num_households + 1)]

    # 2. Corporate shared egress IPs (office NAT, campus Wi-Fi, public transit)
    num_corp_ips = max(15, int(round(config.n_transactions / 2000)))
    corp_ips = [f"198.51.100.{(k % 250) + 1}" for k in range(1, num_corp_ips + 1)]
    public_wifi_ips = [f"172.24.{(k // 250) + 1}.{(k % 250) + 1}" for k in range(1, max(20, num_corp_ips + 1))]

    # 1. Stable Customer Profiles
    customers = {}
    for i in range(1, num_customers + 1):
        cid = f"C_{i:05d}" if num_customers >= 10000 else f"C_{i:04d}"
        card_a = f"CARD_{i:05d}_A" if num_customers >= 10000 else f"CARD_{i:04d}_A"
        card_b = (f"CARD_{i:05d}_B" if num_customers >= 10000 else f"CARD_{i:04d}_B") if rng.random() < 0.22 else None

        # Assign shared household device if customer is in household pool
        is_household = (config.version == "v2" and i <= num_households * 2)
        if is_household:
            h_idx = (i - 1) % num_households
            dev_a = household_devices[h_idx]
            dev_b = f"DEV_{i:05d}_B" if num_customers >= 10000 else f"DEV_{i:04d}_B"
        else:
            dev_a = f"DEV_{i:05d}_A" if num_customers >= 10000 else f"DEV_{i:04d}_A"
            dev_b = (f"DEV_{i:05d}_B" if num_customers >= 10000 else f"DEV_{i:04d}_B") if rng.random() < 0.22 else None

        ip_addr = f"192.168.{(i // 250) + 10}.{(i % 250) + 1}"
        email = f"user_{i}@{rng.choice(email_domains)}"
        domain = email.split("@")[1]
        zip_code = f"ZIP_{10000 + (i % 600)}"
        country = "US" if rng.random() < 0.94 else rng.choice(["CA", "GB"])

        # Base mean spending: $35 to $130
        base_mean = rng.uniform(35.0, 130.0)
        base_std = base_mean * rng.uniform(0.30, 0.45)

        customers[cid] = {
            "customer_id": cid,
            "primary_card": card_a,
            "secondary_card": card_b,
            "primary_device": dev_a,
            "secondary_device": dev_b,
            "ip_address": ip_addr,
            "email_domain": domain,
            "address_id": zip_code,
            "country": country,
            "base_mean": base_mean,
            "base_std": base_std,
            "card_network": rng.choice(card_networks),
            "card_type": "debit" if rng.random() < 0.74 else "credit",
            "preferred_products": rng.sample(product_codes, k=2),
            "is_household": is_household,
        }

    # 2. Merchants
    merchants = {}
    for j in range(1, num_merchants + 1):
        mid = f"M_{j:04d}" if num_merchants >= 1000 else f"M_{j:03d}"
        pcode = rng.choice(product_codes)
        merchants[mid] = {
            "merchant_id": mid,
            "product_code": pcode,
            "price_level": rng.choice(["low", "medium", "high"]),
        }

    # 3. Fraud Syndicates & Attack Infrastructure
    num_syndicates = max(5, int(round(config.n_transactions * config.weight_shared_device_ring / 40)))
    num_ato_attacks = max(45, int(round(config.n_transactions * config.weight_account_takeover / 4)))
    num_card_tests = max(17, int(round(config.n_transactions * config.weight_card_testing / 10)))

    fraud_infra = {
        "bot_devices": [f"DEV_BOT_RING_{k}" for k in range(1, max(11, num_syndicates * 2 + 1))],
        "proxy_ips": [f"10.77.{(k // 250) + 1}.{(k % 250) + 1}" for k in range(1, max(21, num_card_tests + 1))],
        "residential_proxies": [f"172.16.{(k // 250) + 1}.{(k % 250) + 1}" for k in range(1, max(31, num_ato_attacks + 1))],
        "ato_devices": [f"DEV_ATO_ATTACK_{k}" for k in range(1, max(31, num_ato_attacks + 1))],
        "ato_ips": [f"172.16.{(k // 250) + 5}.{(k % 250) + 1}" for k in range(1, max(31, num_ato_attacks + 1))],
    }

    return {
        "customers": customers,
        "merchants": merchants,
        "fraud_infra": fraud_infra,
        "household_devices": household_devices,
        "corp_ips": corp_ips,
        "public_wifi_ips": public_wifi_ips,
    }


def generate_raw_transactions(
    rng: random.Random,
    config: Optional[GeneratorConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Generates chronological raw transactions adhering strictly to the 7 behavioral scenarios,
    with version-controlled realism upgrades in v2.
    """
    if config is None:
        config = GeneratorConfig()

    entities = setup_simulation_entities(rng, config)
    customers = entities["customers"]
    merchants = entities["merchants"]
    fraud_infra = entities["fraud_infra"]
    corp_ips = entities.get("corp_ips", [])
    public_wifi_ips = entities.get("public_wifi_ips", [])

    cust_ids = sorted(list(customers.keys()))
    merch_ids = sorted(list(merchants.keys()))

    start_dt = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    total_seconds = config.time_span_days * 86400

    raw_txs: List[Dict[str, Any]] = []

    num_legit = int(round(config.n_transactions * config.weight_legit_stable))
    num_high_val = int(round(config.n_transactions * config.weight_legit_high_value))
    num_ring = int(round(config.n_transactions * config.weight_shared_device_ring))
    num_ato = int(round(config.n_transactions * config.weight_account_takeover))
    num_card_test = int(round(config.n_transactions * config.weight_card_testing))
    num_stolen = int(round(config.n_transactions * config.weight_stolen_card))
    num_burst = int(round(config.n_transactions * config.weight_burst_fraud))

    is_v2 = (config.version == "v2")

    # -------------------------------------------------------------------------
    # Scenario 1: Legitimate Stable Customers (fraud = 0)
    # -------------------------------------------------------------------------
    legit_stable_txs = []
    for _ in range(num_legit):
        cid = rng.choice(cust_ids)
        c_prof = customers[cid]

        sec_offset = rng.uniform(0, total_seconds)
        dt = start_dt + timedelta(seconds=sec_offset)

        # Card selection
        card = c_prof["secondary_card"] if (c_prof["secondary_card"] and rng.random() < 0.20) else c_prof["primary_card"]

        # Device selection with realism enhancements:
        if is_v2:
            dev_roll = rng.random()
            if dev_roll < 0.05:
                # 5% telemetry missingness (adblocker, incognito, Safari privacy)
                dev = None
            elif dev_roll < 0.09:
                # 4% new device / cookie reset (customer_device_seen_before == 0)
                dev = f"DEV_NEW_{cid}_{rng.randint(1, 999):03d}"
            elif dev_roll < 0.20 and c_prof["is_household"]:
                # 11% uses shared household device
                dev = c_prof["primary_device"]
            elif dev_roll < 0.35 and c_prof["secondary_device"]:
                dev = c_prof["secondary_device"]
            else:
                dev = c_prof["primary_device"]
        else:
            dev = c_prof["secondary_device"] if (c_prof["secondary_device"] and rng.random() < 0.25) else c_prof["primary_device"]
            if rng.random() < 0.03:
                dev = None

        # IP selection with realism enhancements:
        if is_v2:
            ip_roll = rng.random()
            if ip_roll < 0.10 and corp_ips:
                # 10% corporate office NAT / campus Wi-Fi (legitimate shared IP velocity)
                ip = rng.choice(corp_ips)
            elif ip_roll < 0.15 and public_wifi_ips:
                # 5% public Wi-Fi / coffee shop / hotel (customer_ip_seen_before == 0)
                ip = rng.choice(public_wifi_ips)
            elif ip_roll < 0.19:
                # 4% travel IP
                ip = f"172.28.{rng.randint(1, 25)}.{rng.randint(1, 254)}"
            else:
                ip = c_prof["ip_address"]
        else:
            ip = c_prof["ip_address"] if rng.random() < 0.94 else f"172.20.1.{rng.randint(1, 254)}"

        mid = rng.choice(merch_ids)
        m_prof = merchants[mid]

        # Continuous amount distribution with micro-purchases and medium-value:
        if is_v2:
            amt_roll = rng.random()
            if amt_roll < 0.045:
                # 4.5% Micro-purchases ($0.99 - $4.99 digital goods, snacks, subscriptions)
                amt = round(rng.choice([0.99, 1.49, 1.99, 2.49, 2.99, 3.49, 3.99, 4.49, 4.99, 9.99]), 2)
            elif amt_roll < 0.86:
                # 81.5% Standard everyday transactions ($10 - $249)
                amt = max(5.00, min(249.00, round(float(np.random.normal(c_prof["base_mean"], c_prof["base_std"])), 2)))
            elif amt_roll < 0.96:
                # 10.0% Medium-value transactions ($250 - $799) - eliminates artificial gap!
                amt = round(rng.uniform(250.0, 799.0), 2)
            else:
                # 4.0% High-value transactions ($800 - $3,200)
                amt = round(rng.uniform(800.0, 3200.0), 2)
        else:
            amt = max(3.50, round(float(np.random.normal(c_prof["base_mean"], c_prof["base_std"])), 2))

        tx = {
            "transaction_timestamp": dt,
            "amount": amt,
            "customer_id": cid,
            "card_id": card,
            "device_id": dev,
            "merchant_id": mid,
            "ip_address": ip,
            "product_code": m_prof["product_code"],
            "card_network": c_prof["card_network"],
            "card_type": c_prof["card_type"],
            "email_domain": c_prof["email_domain"],
            "address_id": c_prof["address_id"],
            "country": c_prof["country"],
            "fraud_label": 0,
            "scenario": "Legitimate Stable",
        }
        legit_stable_txs.append(tx)

    # In v2: Add legitimate shopping bursts (~3.5% of legitimate transactions)
    if is_v2:
        num_burst_seeds = int(round(len(legit_stable_txs) * 0.035))
        burst_seeds = rng.sample(legit_stable_txs, k=num_burst_seeds)
        for seed_tx in burst_seeds:
            cid = seed_tx["customer_id"]
            c_prof = customers[cid]
            # Follow-up transaction 2 to 7 minutes later
            followup_dt = seed_tx["transaction_timestamp"] + timedelta(seconds=rng.randint(120, 420))
            if followup_dt.timestamp() <= (start_dt.timestamp() + total_seconds):
                followup_mid = rng.choice(merch_ids)
                followup_amt = round(rng.uniform(8.0, 85.0), 2)
                legit_stable_txs.append({
                    "transaction_timestamp": followup_dt,
                    "amount": followup_amt,
                    "customer_id": cid,
                    "card_id": seed_tx["card_id"],
                    "device_id": seed_tx["device_id"],
                    "merchant_id": followup_mid,
                    "ip_address": seed_tx["ip_address"],
                    "product_code": merchants[followup_mid]["product_code"],
                    "card_network": c_prof["card_network"],
                    "card_type": c_prof["card_type"],
                    "email_domain": c_prof["email_domain"],
                    "address_id": c_prof["address_id"],
                    "country": c_prof["country"],
                    "fraud_label": 0,
                    "scenario": "Legitimate Stable",
                })

    raw_txs.extend(legit_stable_txs)

    # -------------------------------------------------------------------------
    # Scenario 2: Legitimate High-Value Purchases (fraud = 0)
    # -------------------------------------------------------------------------
    for _ in range(num_high_val):
        cid = rng.choice(cust_ids)
        c_prof = customers[cid]

        sec_offset = rng.uniform(86400 * 3, total_seconds)
        dt = start_dt + timedelta(seconds=sec_offset)

        card = c_prof["primary_card"]

        if is_v2:
            # Add realistic device/IP noise:
            dev = None if rng.random() < 0.04 else (f"DEV_NEW_{cid}_{rng.randint(1, 999):03d}" if rng.random() < 0.05 else c_prof["primary_device"])
            ip = rng.choice(corp_ips) if (corp_ips and rng.random() < 0.08) else (c_prof["ip_address"] if rng.random() < 0.88 else f"172.28.{rng.randint(1, 20)}.{rng.randint(1, 254)}")
            amt = round(rng.uniform(500.0, 3500.0), 2)
        else:
            dev = c_prof["primary_device"]
            ip = c_prof["ip_address"]
            amt = round(rng.uniform(800.0, 3500.0), 2)

        mid = rng.choice(merch_ids)
        m_prof = merchants[mid]

        raw_txs.append({
            "transaction_timestamp": dt,
            "amount": amt,
            "customer_id": cid,
            "card_id": card,
            "device_id": dev,
            "merchant_id": mid,
            "ip_address": ip,
            "product_code": m_prof["product_code"],
            "card_network": c_prof["card_network"],
            "card_type": c_prof["card_type"],
            "email_domain": c_prof["email_domain"],
            "address_id": c_prof["address_id"],
            "country": c_prof["country"],
            "fraud_label": 0,
            "scenario": "Legitimate High-Value",
        })

    # -------------------------------------------------------------------------
    # Scenario 3: Account Takeover (ATO) (fraud = 1)
    # -------------------------------------------------------------------------
    # In v2: 35% Stealth ATO (isolated single tx, moderate amounts, normal cadence)
    #        65% Aggressive ATO (rapid bursts, higher amounts)
    num_stealth_ato = int(round(num_ato * 0.35)) if is_v2 else 0
    num_burst_ato = num_ato - num_stealth_ato

    # A. Stealth ATO
    if num_stealth_ato > 0:
        stealth_custs = rng.sample(cust_ids, k=min(len(cust_ids), num_stealth_ato))
        for k, cid in enumerate(stealth_custs):
            c_prof = customers[cid]
            dt = start_dt + timedelta(seconds=rng.uniform(86400 * 5, total_seconds - 86400))
            stealth_dev = f"DEV_RESIDENTIAL_{k % 500:03d}" if rng.random() < 0.90 else None
            stealth_ip = f"172.16.{rng.randint(1, 40)}.{rng.randint(1, 254)}"
            target_merch = rng.choice(merch_ids)
            # Moderate amounts overlapping legitimate medium-values:
            stealth_amt = round(rng.choice([145.0, 185.0, 240.0, 310.0, 450.0, 580.0, 720.0]) + rng.uniform(0.1, 9.9), 2)

            raw_txs.append({
                "transaction_timestamp": dt,
                "amount": stealth_amt,
                "customer_id": cid,
                "card_id": c_prof["primary_card"],
                "device_id": stealth_dev,
                "merchant_id": target_merch,
                "ip_address": stealth_ip,
                "product_code": "W",
                "card_network": c_prof["card_network"],
                "card_type": c_prof["card_type"],
                "email_domain": c_prof["email_domain"],
                "address_id": c_prof["address_id"],
                "country": "US" if rng.random() < 0.85 else "RU",
                "fraud_label": 1,
                "scenario": "Account Takeover",
            })

    # B. Aggressive / Multi-Attempt ATO
    num_ato_sessions = max(1, num_burst_ato // 4)
    ato_target_custs = rng.sample(cust_ids, k=min(len(cust_ids), num_ato_sessions))
    for k, cid in enumerate(ato_target_custs):
        c_prof = customers[cid]
        attack_base_sec = rng.uniform(86400 * 10, total_seconds - 86400 * 2)
        attacker_dev = fraud_infra["ato_devices"][k % len(fraud_infra["ato_devices"])]
        attacker_ip = fraud_infra["ato_ips"][k % len(fraud_infra["ato_ips"])]
        target_merch = rng.choice(merch_ids)

        tx_count_in_session = 4 if not is_v2 else rng.choice([2, 3, 4])
        for tx_idx in range(tx_count_in_session):
            dt = start_dt + timedelta(seconds=attack_base_sec + tx_idx * rng.randint(45, 240))
            amt = round(rng.uniform(450.0, 2400.0), 2)

            raw_txs.append({
                "transaction_timestamp": dt,
                "amount": amt,
                "customer_id": cid,
                "card_id": c_prof["primary_card"],
                "device_id": attacker_dev,
                "merchant_id": target_merch,
                "ip_address": attacker_ip,
                "product_code": "W",
                "card_network": c_prof["card_network"],
                "card_type": c_prof["card_type"],
                "email_domain": c_prof["email_domain"],
                "address_id": c_prof["address_id"],
                "country": "RU" if rng.random() < 0.5 else "US",
                "fraud_label": 1,
                "scenario": "Account Takeover",
            })

    # -------------------------------------------------------------------------
    # Scenario 4: Card Testing (fraud = 1)
    # -------------------------------------------------------------------------
    # In v2: 35% Stealth / Low-and-Slow Card Testing (isolated transactions with rotating IPs)
    #        65% Bot Burst Card Testing (rapid sequence)
    num_stealth_card_test = int(round(num_card_test * 0.35)) if is_v2 else 0
    num_burst_card_test = num_card_test - num_stealth_card_test

    # A. Stealth Card Testing
    if num_stealth_card_test > 0:
        stealth_test_custs = rng.sample(cust_ids, k=min(len(cust_ids), num_stealth_card_test))
        for k, cid in enumerate(stealth_test_custs):
            c_prof = customers[cid]
            dt = start_dt + timedelta(seconds=rng.uniform(86400 * 2, total_seconds - 86400))
            rotated_ip = f"10.88.{rng.randint(1, 40)}.{rng.randint(1, 254)}"
            test_merch = rng.choice(merch_ids)
            amt = round(rng.choice([0.99, 1.49, 2.00, 2.99, 4.99]), 2)
            dev = f"DEV_STEALTH_TEST_{k % 300:03d}" if rng.random() < 0.92 else None

            raw_txs.append({
                "transaction_timestamp": dt,
                "amount": amt,
                "customer_id": cid,
                "card_id": c_prof["primary_card"],
                "device_id": dev,
                "merchant_id": test_merch,
                "ip_address": rotated_ip,
                "product_code": "W",
                "card_network": c_prof["card_network"],
                "card_type": c_prof["card_type"],
                "email_domain": c_prof["email_domain"],
                "address_id": c_prof["address_id"],
                "country": c_prof["country"],
                "fraud_label": 1,
                "scenario": "Card Testing",
            })

    # B. Bot Burst Card Testing
    num_card_test_sessions = max(1, num_burst_card_test // 8)
    for session_idx in range(num_card_test_sessions):
        test_base_sec = rng.uniform(86400 * 2, total_seconds - 86400)
        bot_ip = fraud_infra["proxy_ips"][session_idx % len(fraud_infra["proxy_ips"])]
        bot_dev = fraud_infra["bot_devices"][session_idx % len(fraud_infra["bot_devices"])]
        test_merch = rng.choice(merch_ids)

        session_steps = 8 if is_v2 else 10
        for step in range(session_steps):
            cid = rng.choice(cust_ids)
            c_prof = customers[cid]
            dt = start_dt + timedelta(seconds=test_base_sec + step * rng.randint(15, 50))
            amt = round(rng.choice([0.99, 1.00, 1.50, 2.00, 3.49, 4.99, 8.50]), 2)

            raw_txs.append({
                "transaction_timestamp": dt,
                "amount": amt,
                "customer_id": cid,
                "card_id": c_prof["primary_card"],
                "device_id": bot_dev,
                "merchant_id": test_merch,
                "ip_address": bot_ip,
                "product_code": "W",
                "card_network": c_prof["card_network"],
                "card_type": c_prof["card_type"],
                "email_domain": "anonymous.com" if rng.random() < 0.5 else c_prof["email_domain"],
                "address_id": c_prof["address_id"],
                "country": c_prof["country"],
                "fraud_label": 1,
                "scenario": "Card Testing",
            })

    # -------------------------------------------------------------------------
    # Scenario 5: Stolen Card Behavior (fraud = 1)
    # -------------------------------------------------------------------------
    num_stolen_cards = max(1, num_stolen // 4)
    stolen_target_custs = rng.sample(cust_ids, k=min(len(cust_ids), num_stolen_cards))
    for k, cid in enumerate(stolen_target_custs):
        c_prof = customers[cid]
        sec_offset = rng.uniform(86400 * 6, total_seconds - 86400)
        thief_dev = None if (is_v2 and rng.random() < 0.06) else f"DEV_THIEF_{k % 500:03d}"
        thief_ip = f"198.51.100.{(k % 250) + 1}"
        retail_merch = rng.choice(merch_ids)

        tx_count_stolen = 4 if is_v2 else 5
        for tx_idx in range(tx_count_stolen):
            dt = start_dt + timedelta(seconds=sec_offset + tx_idx * rng.randint(120, 600))
            amt = round(rng.uniform(180.0, 1800.0), 2)

            raw_txs.append({
                "transaction_timestamp": dt,
                "amount": amt,
                "customer_id": cid,
                "card_id": c_prof["primary_card"],
                "device_id": thief_dev,
                "merchant_id": retail_merch,
                "ip_address": thief_ip,
                "product_code": "R",
                "card_network": c_prof["card_network"],
                "card_type": c_prof["card_type"],
                "email_domain": c_prof["email_domain"],
                "address_id": f"ZIP_{90000 + (k % 9000)}",
                "country": "US",
                "fraud_label": 1,
                "scenario": "Stolen Card",
            })

    # -------------------------------------------------------------------------
    # Scenario 6: Shared Device / Shared IP Abuse (fraud = 1)
    # -------------------------------------------------------------------------
    num_syndicates = max(1, num_ring // 35)
    for s_idx in range(num_syndicates):
        syndicate_dev = fraud_infra["bot_devices"][s_idx % len(fraud_infra["bot_devices"])]
        syndicate_ip = fraud_infra["proxy_ips"][s_idx % len(fraud_infra["proxy_ips"])]
        syndicate_merch = rng.choice(merch_ids)
        syndicate_base_sec = rng.uniform(86400 * 12, total_seconds - 86400 * 2)

        target_pool = rng.sample(cust_ids, k=min(8, len(cust_ids)))
        step_count = 35 if is_v2 else 40
        for tx_idx in range(step_count):
            target_cid = target_pool[tx_idx % len(target_pool)]
            c_prof = customers[target_cid]
            dt = start_dt + timedelta(seconds=syndicate_base_sec + tx_idx * rng.randint(180, 700))
            amt = round(rng.uniform(120.0, 950.0), 2)

            raw_txs.append({
                "transaction_timestamp": dt,
                "amount": amt,
                "customer_id": target_cid,
                "card_id": c_prof["primary_card"],
                "device_id": syndicate_dev,
                "merchant_id": syndicate_merch,
                "ip_address": syndicate_ip,
                "product_code": "W",
                "card_network": c_prof["card_network"],
                "card_type": c_prof["card_type"],
                "email_domain": "anonymous.com",
                "address_id": c_prof["address_id"],
                "country": "US",
                "fraud_label": 1,
                "scenario": "Shared Device / IP Ring",
            })

    # -------------------------------------------------------------------------
    # Scenario 7: Burst Velocity Fraud (fraud = 1)
    # -------------------------------------------------------------------------
    num_bursts = max(1, num_burst // 4)
    burst_custs = rng.sample(cust_ids, k=min(len(cust_ids), num_bursts))
    for k, cid in enumerate(burst_custs):
        c_prof = customers[cid]
        base_sec = rng.uniform(86400 * 10, total_seconds - 86400)
        b_dev = None if (is_v2 and rng.random() < 0.05) else f"DEV_BURST_{k % 500:03d}"
        b_ip = f"203.0.113.{(k % 250) + 1}"
        b_merch = rng.choice(merch_ids)

        burst_steps = 4 if is_v2 else 5
        for step in range(burst_steps):
            dt = start_dt + timedelta(seconds=base_sec + step * rng.randint(20, 90))
            amt = round(rng.uniform(180.0, 1200.0), 2)

            raw_txs.append({
                "transaction_timestamp": dt,
                "amount": amt,
                "customer_id": cid,
                "card_id": c_prof["primary_card"],
                "device_id": b_dev,
                "merchant_id": b_merch,
                "ip_address": b_ip,
                "product_code": "W",
                "card_network": c_prof["card_network"],
                "card_type": c_prof["card_type"],
                "email_domain": c_prof["email_domain"],
                "address_id": c_prof["address_id"],
                "country": "US",
                "fraud_label": 1,
                "scenario": "Burst Fraud",
            })

    # Adjust to exactly config.n_transactions
    current_count = len(raw_txs)
    if current_count > config.n_transactions:
        legit_indices = [idx for idx, tx in enumerate(raw_txs) if tx["scenario"] == "Legitimate Stable"]
        surplus = current_count - config.n_transactions
        to_remove = set(rng.sample(legit_indices, k=surplus))
        raw_txs = [tx for idx, tx in enumerate(raw_txs) if idx not in to_remove]
    elif current_count < config.n_transactions:
        deficit = config.n_transactions - current_count
        for _ in range(deficit):
            cid = rng.choice(cust_ids)
            c_prof = customers[cid]
            sec_offset = rng.uniform(0, total_seconds)
            dt = start_dt + timedelta(seconds=sec_offset)
            mid = rng.choice(merch_ids)
            amt = max(5.00, round(float(np.random.normal(c_prof["base_mean"], c_prof["base_std"])), 2))
            raw_txs.append({
                "transaction_timestamp": dt,
                "amount": amt,
                "customer_id": cid,
                "card_id": c_prof["primary_card"],
                "device_id": c_prof["primary_device"],
                "merchant_id": mid,
                "ip_address": c_prof["ip_address"],
                "product_code": merchants[mid]["product_code"],
                "card_network": c_prof["card_network"],
                "card_type": c_prof["card_type"],
                "email_domain": c_prof["email_domain"],
                "address_id": c_prof["address_id"],
                "country": c_prof["country"],
                "fraud_label": 0,
                "scenario": "Legitimate Stable",
            })

    # Sort strictly chronologically
    raw_txs.sort(key=lambda x: x["transaction_timestamp"])

    # Re-assign sequential transaction_ids
    id_padding = 7 if config.n_transactions >= 100000 else 6
    for i, tx in enumerate(raw_txs, start=1):
        tx["transaction_id"] = f"TX_{i:0{id_padding}d}"

    return raw_txs


def generate_behavioral_dataset(config: Optional[GeneratorConfig] = None) -> Dict[str, Any]:
    """
    Main pipeline:
    1. Generates raw chronological transactions according to GeneratorConfig.
    2. Passes each through generate_production_features() using entity-indexed history.
    3. Saves raw CSV, features CSV, and metadata JSON.
    """
    if config is None:
        config = GeneratorConfig()

    t_start = time.time()
    suffix = config.get_suffix()

    print("=" * 76)
    print("AegisFin Phase 2: Behavioral Dataset Generator")
    print(f"Generator Version   : {GENERATOR_VERSION} (Engine: {config.version.upper()})")
    print(f"Target Transactions : {config.n_transactions:,}")
    print(f"Random Seed         : {config.random_seed}")
    print(f"Dataset Suffix      : {suffix}")
    print(f"Target Fraud Rate   : {config.target_fraud_rate() * 100:.2f}%")
    print("=" * 76)

    rng = random.Random(config.random_seed)
    np.random.seed(config.random_seed)

    # 1. Generate Raw Transactions
    print("\n[Step 1/4] Generating raw transactions across 7 behavioral scenarios...")
    raw_txs = generate_raw_transactions(rng, config)
    assert len(raw_txs) == config.n_transactions, f"Expected {config.n_transactions}, got {len(raw_txs)}"
    print(f"-> Generated {len(raw_txs):,} raw transactions.")

    # 2. Check Chronological Integrity
    print("\n[Step 2/4] Verifying strict chronological ordering...")
    is_chronological = True
    for i in range(len(raw_txs) - 1):
        if raw_txs[i]["transaction_timestamp"] > raw_txs[i + 1]["transaction_timestamp"]:
            is_chronological = False
            break

    print(f"-> Chronological ordering verified: {is_chronological}")
    if not is_chronological:
        raise ValueError("FATAL: Raw transactions are not strictly chronological!")

    # 3. Transform Each Transaction through Production Feature Engine
    print("\n[Step 3/4] Transforming transactions via generate_production_features()...")
    feature_rows: List[Dict[str, Any]] = []
    fraud_count = 0
    scenario_counts: Dict[str, int] = {}
    has_nan_inf = False

    by_cust: Dict[str, List[Dict[str, Any]]] = {}
    by_card: Dict[str, List[Dict[str, Any]]] = {}
    by_dev: Dict[str, List[Dict[str, Any]]] = {}
    by_merch: Dict[str, List[Dict[str, Any]]] = {}
    by_ip: Dict[str, List[Dict[str, Any]]] = {}

    log_interval = max(1000, config.n_transactions // 10)

    for i, curr_tx in enumerate(raw_txs):
        curr_cust = curr_tx["customer_id"]
        curr_card = curr_tx["card_id"]
        curr_dev = curr_tx["device_id"]
        curr_merch = curr_tx["merchant_id"]
        curr_ip = curr_tx["ip_address"]
        sc = curr_tx["scenario"]

        scenario_counts[sc] = scenario_counts.get(sc, 0) + 1
        if curr_tx["fraud_label"] == 1:
            fraud_count += 1

        # Construct previous transactions relevant to this transaction's entities
        c_list = by_cust.get(curr_cust, [])
        cd_list = by_card.get(curr_card, [])
        d_list = by_dev.get(curr_dev, []) if curr_dev else []
        m_list = by_merch.get(curr_merch, []) if curr_merch else []
        ip_list = by_ip.get(curr_ip, []) if curr_ip else []

        seen_tids: Set[str] = set()
        entity_history: List[Dict[str, Any]] = []
        for entity_txs in (c_list, cd_list, d_list, m_list, ip_list):
            for h_tx in entity_txs:
                tid = h_tx["transaction_id"]
                if tid not in seen_tids:
                    seen_tids.add(tid)
                    entity_history.append(h_tx)

        # Call frozen production feature generator
        feat_dict = generate_production_features(
            current_transaction=curr_tx,
            history=entity_history,
        )

        if len(feat_dict) != len(VALID_FEATURE_NAMES):
            raise ValueError(f"Feature count mismatch at {curr_tx['transaction_id']}: expected {len(VALID_FEATURE_NAMES)}, got {len(feat_dict)}")

        for fval in feat_dict.values():
            if math.isnan(fval) or math.isinf(fval):
                has_nan_inf = True

        row = {
            "transaction_id": curr_tx["transaction_id"],
            "fraud_label": curr_tx["fraud_label"],
        }
        for name in VALID_FEATURE_NAMES:
            row[name] = feat_dict[name]

        feature_rows.append(row)

        by_cust.setdefault(curr_cust, []).append(curr_tx)
        by_card.setdefault(curr_card, []).append(curr_tx)
        if curr_dev:
            by_dev.setdefault(curr_dev, []).append(curr_tx)
        if curr_merch:
            by_merch.setdefault(curr_merch, []).append(curr_tx)
        if curr_ip:
            by_ip.setdefault(curr_ip, []).append(curr_tx)

        if (i + 1) % log_interval == 0 or (i + 1) == config.n_transactions:
            print(f"   Processed {i + 1:,} / {config.n_transactions:,} transactions...")

    print("-> Production feature transformation complete.")

    # 4. Save Artifacts to data/behavioral/
    print("\n[Step 4/4] Writing dataset artifacts to disk...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_csv_path = DATA_DIR / f"raw_transactions_{suffix}.csv"
    feat_csv_path = DATA_DIR / f"production_features_{suffix}.csv"
    meta_json_path = DATA_DIR / f"behavioral_dataset_metadata_{suffix}.json"

    # Also support default 10k metadata path compatibility and v2 metadata path
    default_meta_path = DATA_DIR / "behavioral_dataset_metadata.json"
    v2_meta_path = DATA_DIR / "behavioral_dataset_metadata_v2.json"

    raw_fieldnames = [
        "transaction_id",
        "transaction_timestamp",
        "amount",
        "customer_id",
        "card_id",
        "device_id",
        "merchant_id",
        "ip_address",
        "product_code",
        "card_network",
        "card_type",
        "email_domain",
        "address_id",
        "country",
        "fraud_label",
    ]

    with open(raw_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fieldnames)
        writer.writeheader()
        for tx in raw_txs:
            row_dict = {k: tx[k] for k in raw_fieldnames}
            row_dict["transaction_timestamp"] = tx["transaction_timestamp"].isoformat()
            writer.writerow(row_dict)

    print(f"-> Saved raw transactions to: {raw_csv_path}")

    feat_fieldnames = ["transaction_id", "fraud_label"] + VALID_FEATURE_NAMES
    with open(feat_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=feat_fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)

    print(f"-> Saved production features to: {feat_csv_path}")

    fraud_rate = fraud_count / config.n_transactions
    runtime_sec = round(time.time() - t_start, 2)

    meta = {
        "generator_version": GENERATOR_VERSION,
        "engine_version": config.version,
        "configuration": asdict(config),
        "random_seed": config.random_seed,
        "total_transactions": config.n_transactions,
        "fraud_count": fraud_count,
        "legitimate_count": config.n_transactions - fraud_count,
        "fraud_rate_pct": round(fraud_rate * 100, 3),
        "production_feature_count": len(VALID_FEATURE_NAMES),
        "timestamp_range": {
            "start_utc": raw_txs[0]["transaction_timestamp"].isoformat(),
            "end_utc": raw_txs[-1]["transaction_timestamp"].isoformat(),
            "duration_days": config.time_span_days,
        },
        "has_nan_inf": has_nan_inf,
        "is_chronological": is_chronological,
        "scenario_breakdown": scenario_counts,
        "raw_schema_fields": raw_fieldnames,
        "production_feature_columns": feat_fieldnames,
        "runtime_seconds": runtime_sec,
        "output_files": {
            "raw_transactions_csv": str(raw_csv_path),
            "production_features_csv": str(feat_csv_path),
            "metadata_json": str(meta_json_path),
        },
    }

    with open(meta_json_path, mode="w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if suffix == "10k":
        with open(default_meta_path, mode="w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    if suffix in ("100k_v2", "v2"):
        with open(v2_meta_path, mode="w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    if "seed123" in suffix:
        seed123_meta_path = DATA_DIR / "behavioral_dataset_metadata_seed123.json"
        with open(seed123_meta_path, mode="w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print(f"-> Saved dataset metadata to: {meta_json_path}")
    if suffix in ("100k_v2", "v2"):
        print(f"-> Saved v2 dataset metadata to: {v2_meta_path}")
    if "seed123" in suffix:
        print(f"-> Saved seed123 dataset metadata to: {seed123_meta_path}")

    print("\n" + "=" * 76)
    print("DATASET GENERATION SUMMARY")
    print("=" * 76)
    print(f"Total Transactions:        {config.n_transactions:,}")
    print(f"Legitimate Transactions:   {config.n_transactions - fraud_count:,} ({100 - fraud_rate*100:.2f}%)")
    print(f"Fraud Transactions:        {fraud_count:,} ({fraud_rate*100:.2f}%)")
    print(f"Production Features:       {len(VALID_FEATURE_NAMES)} features generated")
    print(f"NaN / Inf Values:          {'None (Clean)' if not has_nan_inf else 'DETECTED'}")
    print(f"Chronological Ordering:    {'Strictly Monotonic' if is_chronological else 'Violation'}")
    print(f"Timestamp Range:           {raw_txs[0]['transaction_timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')} to {raw_txs[-1]['transaction_timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Total Runtime:             {runtime_sec:.2f} seconds")
    print("\nScenario Breakdown:")
    for sc, count in sorted(scenario_counts.items(), key=lambda x: -x[1]):
        print(f"  - {sc:<26}: {count:>7,} transactions")
    print("=" * 76)

    return meta


def main():
    parser = argparse.ArgumentParser(description="AegisFin Phase 2: Behavioral Transaction Dataset Generator")
    parser.add_argument("--n-transactions", type=int, default=100000, help="Number of transactions to generate (default: 100000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--fraud-rate", type=float, default=None, help="Target fraud rate override between 0.01 and 0.50")
    parser.add_argument("--suffix", type=str, default="100k_v2", help="Custom output file suffix (default: '100k_v2')")
    parser.add_argument("--days", type=float, default=45.0, help="Simulation duration in days (default: 45.0)")
    parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2"], help="Generator behavioral engine version ('v1' legacy or 'v2' realistic)")
    args = parser.parse_args()

    config = GeneratorConfig(
        n_transactions=args.n_transactions,
        random_seed=args.seed,
        output_suffix=args.suffix,
        time_span_days=args.days,
        version=args.version,
    )

    if args.fraud_rate is not None:
        if not (0.01 <= args.fraud_rate <= 0.50):
            raise ValueError(f"Fraud rate {args.fraud_rate} must be between 0.01 and 0.50")
        current_fraud_total = config.target_fraud_rate()
        scale = args.fraud_rate / current_fraud_total
        config.weight_shared_device_ring *= scale
        config.weight_account_takeover *= scale
        config.weight_card_testing *= scale
        config.weight_stolen_card *= scale
        config.weight_burst_fraud *= scale
        remaining_legit = 1.0 - args.fraud_rate
        legit_scale = remaining_legit / (config.weight_legit_stable + config.weight_legit_high_value)
        config.weight_legit_stable *= legit_scale
        config.weight_legit_high_value *= legit_scale

    generate_behavioral_dataset(config)


if __name__ == "__main__":
    main()
