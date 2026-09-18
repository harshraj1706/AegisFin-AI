from __future__ import annotations
import numpy as np
import pandas as pd


def safe_divide(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    out = pd.Series(np.nan, index=a.index, dtype="float64")
    valid = a.notna() & b.notna() & (b != 0)
    out.loc[valid] = a.loc[valid] / b.loc[valid]
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED_ANOMALY"] = (
            df["DAYS_EMPLOYED"] == 365243
        ).astype("int8")
        df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

    if "DAYS_EMPLOYED" in df.columns:
        df["EMPLOYMENT_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["CREDIT_INCOME_RATIO"] = safe_divide(
            df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"]
        )

    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["ANNUITY_INCOME_RATIO"] = safe_divide(
            df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"]
        )

    if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(df.columns):
        df["CREDIT_ANNUITY_RATIO"] = safe_divide(
            df["AMT_CREDIT"], df["AMT_ANNUITY"]
        )

    if {"AMT_GOODS_PRICE", "AMT_CREDIT"}.issubset(df.columns):
        df["GOODS_CREDIT_RATIO"] = safe_divide(
            df["AMT_GOODS_PRICE"], df["AMT_CREDIT"]
        )

    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(df.columns):
        df["INCOME_PER_FAMILY_MEMBER"] = safe_divide(
            df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"] + 1
        )

    if {"AMT_INCOME_TOTAL", "CNT_CHILDREN"}.issubset(df.columns):
        df["INCOME_PER_CHILD"] = safe_divide(
            df["AMT_INCOME_TOTAL"], df["CNT_CHILDREN"] + 1
        )

    ext_cols = [
        c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        if c in df.columns
    ]

    if ext_cols:
        df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
        df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)
        df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)
        df["EXT_SOURCE_MAX"] = df[ext_cols].max(axis=1)

    return df
