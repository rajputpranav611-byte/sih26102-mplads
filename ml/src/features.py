from __future__ import annotations

from typing import Iterable

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered feature columns while preserving the original dataframe."""
    out = df.copy()

    # Numeric conversions
    for col in ["sanctioned_amount", "expenditure_amount", "completion_percent"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Date parsing with graceful NaN handling
    for col in ["recommended_date", "sanction_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    today = pd.Timestamp.today().normalize()

    # Ratios and time-based features
    if {"sanctioned_amount", "expenditure_amount"}.issubset(out.columns):
        out["expenditure_ratio"] = (
            out["expenditure_amount"] / out["sanctioned_amount"]
        ).replace([float("inf"), -float("inf")], pd.NA)

    if "sanction_date" in out.columns:
        out["days_since_sanction"] = (
            (today - out["sanction_date"]).dt.days
        ).where(out["sanction_date"].notna(), pd.NA)

    if "recommended_date" in out.columns:
        out["days_since_recommendation"] = (
            (today - out["recommended_date"]).dt.days
        ).where(out["recommended_date"].notna(), pd.NA)

    if {"recommended_date", "sanction_date"}.issubset(out.columns):
        out["sanction_gap_days"] = (
            (out["sanction_date"] - out["recommended_date"]).dt.days
        ).where(out["recommended_date"].notna() & out["sanction_date"].notna(), pd.NA)

    if {"days_since_sanction", "completion_percent"}.issubset(out.columns):
        out["is_overdue"] = (
            (out["days_since_sanction"].gt(365)) & (out["completion_percent"].lt(100))
        ).fillna(False)

    # Vendor frequency feature
    if "vendor_id" in out.columns:
        vendor_counts = out["vendor_id"].value_counts(dropna=False)
        out["vendor_work_count"] = out["vendor_id"].map(vendor_counts).fillna(0)

    # Beneficiary frequency feature
    if "beneficiary_ids" in out.columns:
        beneficiary_rows = out["beneficiary_ids"].dropna().astype(str)
        beneficiary_counts = pd.Series(
            [
                str(val).split(",")
                for val in beneficiary_rows
            ],
            index=beneficiary_rows.index,
        ).explode().str.strip()
        beneficiary_counts = beneficiary_counts[beneficiary_counts != ""]
        beneficiary_counts = beneficiary_counts.value_counts()

        def count_for_entry(value: object) -> int:
            if pd.isna(value):
                return 0
            items = [str(x).strip() for x in str(value).split(",") if str(x).strip()]
            if not items:
                return 0
            return sum(beneficiary_counts.get(item, 0) for item in items)

        out["beneficiary_work_count"] = out["beneficiary_ids"].map(count_for_entry).fillna(0)

    # One-hot encode categorical columns while retaining original columns
    if "house" in out.columns:
        house_dummies = pd.get_dummies(out["house"], prefix="house")
        out = pd.concat([out, house_dummies], axis=1)

    if "work_category" in out.columns:
        category_dummies = pd.get_dummies(out["work_category"], prefix="work_category")
        out = pd.concat([out, category_dummies], axis=1)

    return out


if __name__ == "__main__":
    import os
    from pathlib import Path

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data" / "processed"
    input_path = data_dir / "works.csv"
    output_path = data_dir / "works_features.csv"

    df = pd.read_csv(input_path)
    print(f"Before: {df.shape}")

    df_features = engineer_features(df)
    print(f"After: {df_features.shape}")

    new_cols = [
        "expenditure_ratio",
        "days_since_sanction",
        "days_since_recommendation",
        "sanction_gap_days",
        "is_overdue",
        "vendor_work_count",
        "beneficiary_work_count",
    ]

    print(df_features[["expenditure_ratio", "days_since_sanction", "sanction_gap_days", "is_overdue", "vendor_work_count"]].describe())
    print("\nNull counts for new columns:")
    print(df_features[new_cols].isnull().sum())

    zero_or_missing = df_features[
        (df_features["sanctioned_amount"].isna()) |
        (df_features["sanctioned_amount"].eq(0))
    ].copy()
    if not zero_or_missing.empty:
        print("\nRows with sanctioned_amount == 0 or null:")
        print(zero_or_missing.to_string(index=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)
    print(f"Saved feature dataframe to: {output_path}")
