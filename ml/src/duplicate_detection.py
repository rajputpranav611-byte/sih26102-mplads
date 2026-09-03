from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "works_features.csv"


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Flag duplicate vendor or beneficiary patterns.

    Rules:
    - vendor_id is flagged if the same vendor appears in 5+ works within a 90-day window
      across different districts.
    - beneficiary_ids is flagged if the same beneficiary appears across 3+ different works.
    """
    out = df.copy()
    if "sanction_date" not in out.columns:
        raise ValueError("Column 'sanction_date' is required for duplicate detection.")
    if "vendor_id" not in out.columns:
        raise ValueError("Column 'vendor_id' is required for duplicate detection.")
    if "constituency" not in out.columns:
        raise ValueError("Column 'constituency' is required for duplicate detection.")
    if "beneficiary_ids" not in out.columns:
        raise ValueError("Column 'beneficiary_ids' is required for duplicate detection.")

    out["sanction_date"] = pd.to_datetime(out["sanction_date"], errors="coerce")
    out["is_duplicate_vendor"] = False
    out["is_duplicate_beneficiary"] = False

    if out.empty:
        return out

    vendor_flag = pd.Series(False, index=out.index)
    for vendor_id, group in out.groupby("vendor_id"):
        if pd.isna(vendor_id):
            continue
        group = group.sort_values("sanction_date").copy()
        group_indices = group.index.tolist()
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
            for idx in group_indices:
                vendor_flag.loc[idx] = True

    out["is_duplicate_vendor"] = vendor_flag.fillna(False).astype(bool)

    beneficiary_flag = pd.Series(False, index=out.index)
    beneficiary_work_map: dict[str, set[int]] = {}
    for idx, row in out.iterrows():
        entries = [
            str(x).strip()
            for x in str(row["beneficiary_ids"]).split(",")
            if str(x).strip()
        ]
        for beneficiary in entries:
            beneficiary_work_map.setdefault(beneficiary, set()).add(int(idx))

    for beneficiary, idxs in beneficiary_work_map.items():
        if len(idxs) >= 3:
            for idx in idxs:
                beneficiary_flag.loc[idx] = True

    out["is_duplicate_beneficiary"] = beneficiary_flag.fillna(False).astype(bool)
    return out


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    flagged = check_duplicates(df)

    duplicate_vendor_true = int((df["anomaly_type"] == "duplicate_vendor").sum())
    duplicate_vendor_detected = int(flagged["is_duplicate_vendor"].sum())
    duplicate_beneficiary_detected = int(flagged["is_duplicate_beneficiary"].sum())

    print(f"True duplicate_vendor anomaly rows: {duplicate_vendor_true}")
    print(f"Flagged vendor duplicates (90-day / 5+ works / multi-district rule): {duplicate_vendor_detected}")
    print(f"Flagged beneficiary duplicates (3+ works): {duplicate_beneficiary_detected}")

    vendor_overlap = set(flagged.index[flagged["is_duplicate_vendor"]]) & set(df.index[df["anomaly_type"] == "duplicate_vendor"])
    print(f"Overlap between duplicate_vendor rule and true duplicate_vendor labels: {len(vendor_overlap)}")

    beneficiary_overlap = set(flagged.index[flagged["is_duplicate_beneficiary"]]) & set(df.index[df["anomaly_type"] == "duplicate_beneficiary"])
    print(f"Overlap between duplicate_beneficiary rule and true duplicate_beneficiary labels: {len(beneficiary_overlap)}")


if __name__ == "__main__":
    main()
