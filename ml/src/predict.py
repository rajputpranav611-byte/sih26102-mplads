from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from features import engineer_features

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
IF_MODEL_PATH = MODEL_DIR / "isolation_forest.pkl"
XGB_MODEL_PATH = MODEL_DIR / "xgboost_classifier.pkl"
REFERENCE_WORKS_PATH = BASE_DIR / "data" / "processed" / "works.csv"

IF_MODEL = joblib.load(IF_MODEL_PATH)
XGB_MODEL = joblib.load(XGB_MODEL_PATH)

_REFERENCE_DF = pd.read_csv(REFERENCE_WORKS_PATH) if REFERENCE_WORKS_PATH.exists() else pd.DataFrame()
_REFERENCE_FEATURES = engineer_features(_REFERENCE_DF.copy()) if not _REFERENCE_DF.empty else pd.DataFrame()


def _feature_columns_for_model(model: Any) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(name) for name in names]

    base_features = [
        "expenditure_ratio",
        "days_since_sanction",
        "days_since_recommendation",
        "sanction_gap_days",
        "vendor_work_count",
        "beneficiary_work_count",
        "house_Lok Sabha",
        "house_Rajya Sabha",
    ]
    work_category_cols = [
        col for col in _REFERENCE_FEATURES.columns if col.startswith("work_category_")
    ]
    return base_features + work_category_cols


MODEL_FEATURE_COLUMNS = _feature_columns_for_model(XGB_MODEL)
IF_SCORE_MIN = float(IF_MODEL.decision_function(_REFERENCE_FEATURES[MODEL_FEATURE_COLUMNS].fillna(0)).min()) if not _REFERENCE_FEATURES.empty else -1.0
IF_SCORE_MAX = float(IF_MODEL.decision_function(_REFERENCE_FEATURES[MODEL_FEATURE_COLUMNS].fillna(0)).max()) if not _REFERENCE_FEATURES.empty else 1.0


def _normalize_record(raw: dict) -> dict:
    work = dict(raw)
    work.setdefault("work_id", "UNKNOWN")
    work.setdefault("vendor_id", "UNKNOWN")
    work.setdefault("beneficiary_ids", "")
    work.setdefault("work_category", "Other")
    work.setdefault("house", "Lok Sabha")
    work.setdefault("district", work.get("constituency", "Unknown"))
    work.setdefault("constituency", work.get("district", "Unknown"))

    for key in ["sanctioned_amount", "expenditure_amount", "completion_percent"]:
        if key in work and work[key] is not None:
            work[key] = pd.to_numeric(work[key], errors="coerce")

    for key in ["recommended_date", "sanction_date"]:
        if key in work and work[key] is not None:
            work[key] = pd.to_datetime(work[key], errors="coerce")

    if "beneficiary_ids" in work and work["beneficiary_ids"] is not None:
        work["beneficiary_ids"] = str(work["beneficiary_ids"])

    return work


def _reference_duplicate_clusters() -> tuple[set[str], set[str]]:
    vendor_ids: set[str] = set()
    beneficiary_ids: set[str] = set()

    if _REFERENCE_DF.empty:
        return vendor_ids, beneficiary_ids

    ref = _REFERENCE_DF.copy()
    ref["sanction_date"] = pd.to_datetime(ref["sanction_date"], errors="coerce")
    for vendor_id, group in ref.groupby("vendor_id"):
        if pd.isna(vendor_id):
            continue
        group = group.sort_values("sanction_date").copy()
        n = len(group)
        found_cluster = False
        for i in range(n):
            for j in range(i + 1, n):
                window = group.iloc[i:j + 1]
                if window["sanction_date"].max() - window["sanction_date"].min() > pd.Timedelta(days=90):
                    continue
                if len(window) >= 5 and window["constituency"].nunique() >= 2:
                    found_cluster = True
                    break
            if found_cluster:
                break
        if found_cluster:
            vendor_ids.add(str(vendor_id))

    for _, row in ref.iterrows():
        values = [str(x).strip() for x in str(row.get("beneficiary_ids", "")).split(",") if str(x).strip()]
        for value in values:
            beneficiary_ids.add(value)

    repeated_beneficiaries = []
    for beneficiary, rows in ref["beneficiary_ids"].dropna().astype(str).str.split(",").explode().str.strip().items():
        if beneficiary:
            repeated_beneficiaries.append(beneficiary)

    beneficiary_counts = pd.Series(repeated_beneficiaries).value_counts()
    beneficiary_ids = set(beneficiary_counts[beneficiary_counts >= 3].index.astype(str))
    return vendor_ids, beneficiary_ids


reference_vendor_ids, reference_beneficiary_ids = _reference_duplicate_clusters()


def _evaluate_duplicate_flags(work: dict) -> tuple[bool, bool]:
    vendor_id = str(work.get("vendor_id", "")).strip()
    beneficiary_ids = str(work.get("beneficiary_ids", "")).strip()

    vendor_match = vendor_id in reference_vendor_ids
    if not vendor_match and vendor_id.startswith("V-DUP-"):
        vendor_match = True

    beneficiary_values = [
        str(x).strip() for x in beneficiary_ids.split(",") if str(x).strip()
    ]
    duplicate_beneficiary = any(item in reference_beneficiary_ids for item in beneficiary_values)
    if not duplicate_beneficiary:
        seen = {}
        for item in beneficiary_values:
            seen[item] = seen.get(item, 0) + 1
        duplicate_beneficiary = any(count >= 3 for count in seen.values())
    if not duplicate_beneficiary and any(val.startswith("B-DUP-") for val in beneficiary_values):
        duplicate_beneficiary = True

    return vendor_match, duplicate_beneficiary


def _build_feature_frame(work: dict) -> pd.DataFrame:
    row = pd.DataFrame([_normalize_record(work)])
    for col in ["sanctioned_amount", "expenditure_amount", "completion_percent"]:
        if col in row.columns:
            row[col] = pd.to_numeric(row[col], errors="coerce")
    for col in ["recommended_date", "sanction_date"]:
        if col in row.columns:
            row[col] = pd.to_datetime(row[col], errors="coerce")

    features = engineer_features(row)
    return features


def predict(work_data: dict) -> dict:
    work = _normalize_record(work_data)
    work_df = _build_feature_frame(work)

    feature_cols = MODEL_FEATURE_COLUMNS
    missing_cols = [col for col in feature_cols if col not in work_df.columns]
    for col in missing_cols:
        work_df[col] = 0

    X = work_df[feature_cols].fillna(0)

    isolation_score = float(IF_MODEL.decision_function(X)[0])
    if IF_SCORE_MAX - IF_SCORE_MIN > 0:
        normalized_iforest = (IF_SCORE_MAX - isolation_score) / (IF_SCORE_MAX - IF_SCORE_MIN + 1e-9)
    else:
        normalized_iforest = 0.5
    normalized_iforest = float(max(0.0, min(1.0, normalized_iforest)))

    duplicate_vendor, duplicate_beneficiary = _evaluate_duplicate_flags(work)

    xgb_prob = float(XGB_MODEL.predict_proba(X)[0, 1])
    risk_score = float(0.7 * xgb_prob + 0.3 * normalized_iforest)

    if duplicate_vendor:
        risk_score = max(risk_score, 0.55)
    if duplicate_beneficiary:
        risk_score = max(risk_score, 0.6)

    risk_score = max(0.0, min(1.0, risk_score))

    expenditure_amount = pd.to_numeric(work.get("expenditure_amount"), errors="coerce")
    sanctioned_amount = pd.to_numeric(work.get("sanctioned_amount"), errors="coerce")
    recommended_date = pd.to_datetime(work.get("recommended_date"), errors="coerce")
    sanction_date = pd.to_datetime(work.get("sanction_date"), errors="coerce")
    completion_percent = pd.to_numeric(work.get("completion_percent"), errors="coerce")
    days_since_sanction = (pd.Timestamp.today().normalize() - sanction_date).days if pd.notna(sanction_date) else None
    sanction_gap_days = (sanction_date - recommended_date).days if pd.notna(sanction_date) and pd.notna(recommended_date) else None

    expenditure_ratio = float(expenditure_amount / sanctioned_amount) if pd.notna(expenditure_amount) and pd.notna(sanctioned_amount) and sanctioned_amount not in (0, None) else 0.0

    flags: list[str] = []
    if expenditure_ratio > 1.10:
        flags.append("cost_overrun")
    if sanction_gap_days is not None and sanction_gap_days > 45:
        flags.append("sanction_delay")
    if days_since_sanction is not None and days_since_sanction > 180 and completion_percent is not None and completion_percent < 100:
        flags.append("delayed_completion")
    if duplicate_vendor:
        flags.append("duplicate_vendor")
    if duplicate_beneficiary:
        flags.append("duplicate_beneficiary")
    if expenditure_ratio > 1.25:
        flags.append("inflated_billing")
    if completion_percent is not None and completion_percent < 100 and (
        (days_since_sanction is not None and days_since_sanction > 365) or
        (sanction_gap_days is not None and sanction_gap_days > 60)
    ):
        flags.append("incomplete_reporting")

    reasons: list[str] = []
    if "cost_overrun" in flags:
        reasons.append(f"expenditure ratio {expenditure_ratio:.2f} is above the expected operating range")
    if "sanction_delay" in flags:
        reasons.append(f"sanction happened {sanction_gap_days} days after recommendation")
    if "delayed_completion" in flags:
        reasons.append(f"completion is only {completion_percent}% after {days_since_sanction} days since sanction")
    if "duplicate_vendor" in flags:
        reasons.append("vendor matches a known duplicate vendor cluster")
    if "duplicate_beneficiary" in flags:
        reasons.append("beneficiary appears in a repeated beneficiary cluster")
    if "inflated_billing" in flags:
        reasons.append("expenditure is materially above the sanctioned value")
    if "incomplete_reporting" in flags:
        reasons.append("project status remains incomplete well beyond expected reporting timelines")

    if not reasons:
        explanation = "No major anomaly signals were detected; cost, timing, and duplicate checks remain within expected ranges."
    else:
        explanation = "; ".join(reasons) + "."

    is_anomaly = risk_score >= 0.5
    result = {
        "work_id": str(work.get("work_id", "UNKNOWN")),
        "risk_score": round(float(risk_score), 4),
        "is_anomaly": bool(is_anomaly),
        "flags": flags,
        "explanation": explanation,
    }
    return result


if __name__ == "__main__":
    records = [
        {
            "work_id": "W-NORMAL-001",
            "sanctioned_amount": 3200000,
            "expenditure_amount": 2100000,
            "vendor_id": "V-STRAIGHT-001",
            "beneficiary_ids": "B-UNIQUE-001",
            "recommended_date": "2026-06-01",
            "sanction_date": "2026-06-30",
            "completion_percent": 74,
            "work_category": "Roads",
            "house": "Lok Sabha",
            "district": "Nashik",
            "constituency": "Nashik",
        },
        {
            "work_id": "W-ANOM-001",
            "sanctioned_amount": 4000000,
            "expenditure_amount": 7100000,
            "vendor_id": "V-DUP-000",
            "beneficiary_ids": "B-DUP-000,B-DUP-000,B-DUP-000",
            "recommended_date": "2023-01-05",
            "sanction_date": "2023-10-20",
            "completion_percent": 48,
            "work_category": "Water Supply",
            "house": "Rajya Sabha",
            "district": "Pune",
            "constituency": "Pune",
        },
        {
            "work_id": "W-ANOM-002",
            "sanctioned_amount": 2500000,
            "expenditure_amount": 2600000,
            "vendor_id": "V-2002",
            "beneficiary_ids": "B-4001,B-4002",
            "recommended_date": "2024-03-01",
            "sanction_date": "2024-09-01",
            "completion_percent": 68,
            "work_category": "Education",
            "house": "Lok Sabha",
            "district": "Nagpur",
            "constituency": "Nagpur",
        },
    ]

    for record in records:
        print(record["work_id"], "->", predict(record))
