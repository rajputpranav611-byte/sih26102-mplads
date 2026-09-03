from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def normalize_mp_df(df: pd.DataFrame, house: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    # Drop non-data summary rows using the Sr. No. column; this removes the raw CSV "Grand Total" row.
    df = df[pd.to_numeric(df["Sr. No."], errors="coerce").notna()].copy()

    mp_col = None
    for col in df.columns:
        if "Member" in col or "member" in col:
            mp_col = col
            break
    if mp_col is None:
        raise ValueError(f"Could not find MP name column in {house} data")

    amount_col = None
    for col in df.columns:
        if "Allocated" in col or "allocated" in col or "AMOUNT" in str(col).upper():
            amount_col = col
            break
    if amount_col is None:
        raise ValueError(f"Could not find allocated amount column in {house} data")

    state_col = "State" if "State" in df.columns else next(
        (col for col in df.columns if col.lower() == "state"), None
    )
    constituency_col = "Constituency" if "Constituency" in df.columns else next(
        (col for col in df.columns if col.lower() == "constituency"), None
    )

    df = df.rename(
        columns={
            mp_col: "mp_name",
            amount_col: "allocated_amount",
            state_col: "state",
            constituency_col: "constituency",
        }
    )

    df["house"] = house
    df["allocated_amount"] = pd.to_numeric(
        df["allocated_amount"].astype(str).str.replace(",", "").str.replace("₹", "").str.replace(" ", ""),
        errors="coerce",
    ).fillna(0.0)
    df["mp_name"] = df["mp_name"].astype(str).str.strip()
    df["state"] = df["state"].astype(str).str.strip()
    df["constituency"] = df["constituency"].astype(str).str.strip()

    if house == "Rajya Sabha":
        df["tenure_start"] = np.nan
        df["tenure_end"] = np.nan
        for idx, name in df["mp_name"].items():
            match = re.search(r"\((\d{4})\s*-\s*(\d{2,4})\)", name)
            if match:
                start_year = int(match.group(1))
                end_year_raw = match.group(2)
                end_year = int(end_year_raw) if len(end_year_raw) == 4 else 2000 + int(end_year_raw)
                df.at[idx, "tenure_start"] = start_year
                df.at[idx, "tenure_end"] = end_year
                df.at[idx, "mp_name"] = re.sub(r"\s*\(\d{4}\s*-\s*\d{2,4}\)\s*$", "", name).strip()
    else:
        df["tenure_start"] = np.nan
        df["tenure_end"] = np.nan

    return df[["mp_name", "house", "state", "constituency", "allocated_amount", "tenure_start", "tenure_end"]]


def generate_work_amounts(total_amount: float, count: int, rng: np.random.Generator) -> list[float]:
    if count <= 0:
        return []
    if total_amount <= 0:
        return [0.0 for _ in range(count)]

    total_amount = float(total_amount)
    min_per_work = 50_000.0
    safe_total = total_amount * 0.995
    max_single_cap = min(safe_total, 200_000_000.0)

    # Reduce work count so every work can meet the minimum amount floor.
    max_works = max(1, int(safe_total // min_per_work))
    count = min(count, max_works)

    if count <= 0:
        return [0.0]

    weights = np.abs(rng.gamma(shape=2.0, scale=1.0, size=count)) + 0.05
    weights = np.clip(weights, 0.05, None)
    shares = weights / weights.sum()

    amounts = np.full(count, min_per_work, dtype=float)
    extra_budget = max(0.0, safe_total - amounts.sum())
    amounts += np.round(shares * extra_budget, 2)

    # If any portion exceeds the single-work cap, trim it and redistribute the surplus
    # to remaining works that still have room below the cap.
    while np.any(amounts > max_single_cap):
        over = np.where(amounts > max_single_cap)[0]
        i = over[0]
        excess = float(amounts[i] - max_single_cap)
        amounts[i] = max_single_cap
        remaining = np.where(amounts < max_single_cap)[0]
        if len(remaining) == 0:
            break
        for j in remaining:
            if amounts[j] < max_single_cap:
                room = max_single_cap - amounts[j]
                add = min(excess, room)
                amounts[j] += add
                excess -= add
                if excess <= 1e-9:
                    break

    # Keep total within safe ceiling and preserve the minimum per-work floor.
    current_sum = float(amounts.sum())
    if current_sum > safe_total:
        excess = current_sum - safe_total
        amounts[np.argmax(amounts)] = round(float(max(amounts[np.argmax(amounts)] - excess, min_per_work)), 2)

    current_sum = float(amounts.sum())
    if current_sum > safe_total:
        amounts[-1] = round(float(max(safe_total - float(amounts[:-1].sum()), min_per_work)), 2)

    # Final no-zero safety.
    amounts = np.maximum(amounts, min_per_work)
    total_after = float(amounts.sum())
    if total_after > safe_total:
        amounts[-1] = round(float(max(safe_total - float(amounts[:-1].sum()), min_per_work)), 2)

    return [max(0.0, float(v)) for v in amounts]


def generate_beneficiary_ids(fake: Faker, count: int) -> str:
    ids = [f"B-{fake.random_int(1000, 999999)}" for _ in range(count)]
    return ",".join(ids)


def build_work_records(mp_df: pd.DataFrame, fake: Faker, rng: np.random.Generator) -> pd.DataFrame:
    categories = ["Road", "Water", "Education", "Health", "Sanitation", "Other"]
    records = []

    for _, row in mp_df.iterrows():
        allocated_amount = float(row["allocated_amount"])
        num_works = int(rng.integers(3, 9))
        allocations = generate_work_amounts(allocated_amount, num_works, rng)

        # IMPORTANT: each work is one portion of this MP's own allocation; never the house max, never another MP.
        for sanctioned_amount in allocations:
            work_id = f"W-{fake.random_int(100000, 999999)}"
            work_category = categories[int(rng.integers(0, len(categories)))]
            recommended_date = fake.date_between(start_date="-5y", end_date="today")
            recommended_date = pd.Timestamp(recommended_date)
            sanction_delay_days = int(rng.integers(0, 35))
            sanction_date = recommended_date + pd.Timedelta(days=sanction_delay_days)

            expenditure_multiplier = rng.uniform(0.65, 0.98)
            expenditure_amount = round(float(sanctioned_amount * expenditure_multiplier), 2)
            completion_percent = int(rng.integers(35, 96))
            beneficiary_count = int(rng.integers(2, 6))
            beneficiary_ids = generate_beneficiary_ids(fake, beneficiary_count)

            record = {
                "work_id": work_id,
                "mp_name": row["mp_name"],
                "house": row["house"],
                "state": row["state"],
                "constituency": row["constituency"],
                "vendor_id": f"V-{fake.random_int(1000, 999999)}",
                "work_category": work_category,
                "recommended_date": recommended_date.strftime("%Y-%m-%d"),
                "sanction_date": sanction_date.strftime("%Y-%m-%d"),
                "sanctioned_amount": round(float(sanctioned_amount), 2),
                "expenditure_amount": round(float(expenditure_amount), 2),
                "completion_percent": completion_percent,
                "beneficiary_ids": beneficiary_ids,
                "tenure_end": row["tenure_end"] if row["house"] == "Rajya Sabha" else None,
                "is_anomaly": False,
                "anomaly_type": None,
            }
            records.append(record)

    works_df = pd.DataFrame(records)
    works_df = works_df.reset_index(drop=True)
    return works_df


def inject_anomalies(works_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if works_df.empty:
        return works_df

    df = works_df.copy()
    df["is_anomaly"] = False
    df["anomaly_type"] = None

    total_anomalies = max(1, int(round(len(df) * 0.10)))
    anomaly_types = [
        "cost_overrun",
        "delayed_completion",
        "sanction_delay",
        "duplicate_vendor",
        "duplicate_beneficiary",
    ]
    type_counts = {t: total_anomalies // len(anomaly_types) for t in anomaly_types}
    remainder = total_anomalies % len(anomaly_types)
    for t in anomaly_types[:remainder]:
        type_counts[t] += 1

    # 1) duplicate_vendor anomalies: a small cluster of vendors, each used in 5-10 works.
    dup_target = type_counts["duplicate_vendor"]
    if dup_target > 0:
        duplicate_vendor_count = min(20, max(15, int(np.ceil(dup_target / 7))))
        vendor_ids = [f"V-DUP-{i:03d}" for i in range(duplicate_vendor_count)]
        candidate_rows = df.sort_values("sanction_date").index.tolist()
        used = set()
        vendor_allocations = {vendor_id: 0 for vendor_id in vendor_ids}
        vendor_id_order = list(vendor_ids)
        v_idx = 0

        for row_idx in candidate_rows:
            if len(used) >= dup_target:
                break
            if row_idx in used:
                continue

            vendor_id = vendor_id_order[v_idx % len(vendor_id_order)]
            if vendor_allocations[vendor_id] >= 10:
                v_idx += 1
                vendor_id = vendor_id_order[v_idx % len(vendor_id_order)]

            df.at[row_idx, "vendor_id"] = vendor_id
            df.at[row_idx, "is_anomaly"] = True
            df.at[row_idx, "anomaly_type"] = "duplicate_vendor"
            used.add(row_idx)
            vendor_allocations[vendor_id] += 1
            v_idx += 1

    # 2) duplicate_beneficiary anomalies: a small set of beneficiaries reused across 3-6 works,
    # across different vendors and districts, within a tight 90-day window.
    dup_beneficiary_target = type_counts["duplicate_beneficiary"]
    if dup_beneficiary_target > 0:
        beneficiary_pool = [f"B-DUP-{i:03d}" for i in range(12, 12 + 15)]
        target_rows = max(1, int(np.ceil(dup_beneficiary_target / 3)))
        candidate_rows = df.index[df["is_anomaly"] == False].tolist()
        rng.shuffle(candidate_rows)
        used_indices = set()

        for beneficiary_id in beneficiary_pool:
            if len(used_indices) >= target_rows:
                break
            possible_rows = [idx for idx in candidate_rows if idx not in used_indices]
            if len(possible_rows) < 3:
                break

            chosen_rows = []
            # Choose a tight cluster in time, then ensure different vendors/districts.
            cluster_candidates = sorted(possible_rows, key=lambda idx: pd.to_datetime(df.at[idx, "sanction_date"]))
            if len(cluster_candidates) < 3:
                break

            for row_idx in cluster_candidates:
                if len(chosen_rows) >= min(6, max(3, dup_beneficiary_target // 3)):
                    break
                if row_idx in used_indices:
                    continue
                if len(chosen_rows) > 0:
                    last_dt = pd.to_datetime(df.at[chosen_rows[-1], "sanction_date"])
                    current_dt = pd.to_datetime(df.at[row_idx, "sanction_date"])
                    if (current_dt - last_dt).days > 90:
                        continue
                chosen_rows.append(row_idx)
                used_indices.add(row_idx)

            if len(chosen_rows) < 3:
                # fallback: select a few rows spread across different vendors/districts
                fallback = []
                for row_idx in cluster_candidates:
                    if row_idx in used_indices:
                        continue
                    fallback.append(row_idx)
                    used_indices.add(row_idx)
                    if len(fallback) >= 3:
                        break
                chosen_rows = fallback

            if len(chosen_rows) < 3:
                continue

            for row_idx in chosen_rows:
                existing = df.at[row_idx, "beneficiary_ids"]
                if pd.isna(existing):
                    df.at[row_idx, "beneficiary_ids"] = beneficiary_id
                else:
                    current_ids = [x.strip() for x in str(existing).split(",") if x.strip()]
                    if beneficiary_id not in current_ids:
                        current_ids.append(beneficiary_id)
                        df.at[row_idx, "beneficiary_ids"] = ",".join(current_ids)
                df.at[row_idx, "is_anomaly"] = True
                df.at[row_idx, "anomaly_type"] = "duplicate_beneficiary"

    remaining_idx = df.index[(df["is_anomaly"] == False)].tolist()
    rng.shuffle(remaining_idx)
    remaining_types = [t for t in anomaly_types if t != "duplicate_vendor" and t != "duplicate_beneficiary"]
    for anomaly_type in remaining_types:
        needed = type_counts[anomaly_type]
        if needed <= 0:
            continue
        selected = remaining_idx[:needed]
        remaining_idx = remaining_idx[needed:]

        for idx in selected:
            df.at[idx, "is_anomaly"] = True
            df.at[idx, "anomaly_type"] = anomaly_type

            if anomaly_type == "cost_overrun":
                sanctioned = float(df.at[idx, "sanctioned_amount"])
                ratio = float(rng.uniform(1.02, 1.5))
                partial = rng.random() < 0.2
                if partial:
                    ratio = float(rng.uniform(1.02, 1.08))
                df.at[idx, "expenditure_amount"] = round(sanctioned * ratio, 2)
                df.at[idx, "completion_percent"] = int(rng.integers(70, 100))

            elif anomaly_type == "delayed_completion":
                sanction_dt = pd.to_datetime(df.at[idx, "sanction_date"])
                delay_days = int(rng.integers(120, 1500))
                partial = rng.random() < 0.25
                if partial:
                    delay_days = int(rng.integers(45, 180))
                df.at[idx, "sanction_date"] = (sanction_dt - pd.Timedelta(days=delay_days)).strftime("%Y-%m-%d")
                df.at[idx, "completion_percent"] = int(rng.integers(10, 70))
                if pd.notna(df.at[idx, "tenure_end"]):
                    tenure_end_dt = pd.Timestamp(f"{int(df.at[idx, 'tenure_end'])}-12-31")
                    if pd.to_datetime(df.at[idx, "sanction_date"]) + pd.Timedelta(days=548) > tenure_end_dt:
                        df.at[idx, "completion_percent"] = int(rng.integers(5, 45))

            elif anomaly_type == "sanction_delay":
                recommended_dt = pd.to_datetime(df.at[idx, "recommended_date"])
                delay_days = int(rng.integers(30, 220))
                partial = rng.random() < 0.25
                if partial:
                    delay_days = int(rng.integers(15, 45))
                sanction_dt = recommended_dt + pd.Timedelta(days=delay_days)
                df.at[idx, "sanction_date"] = sanction_dt.strftime("%Y-%m-%d")
                df.at[idx, "completion_percent"] = int(rng.integers(35, 90))

    # Add moderate overlap into the normal population so some normal rows sit near anomaly thresholds.
    normal_idx = df.index[df["is_anomaly"] == False].tolist()
    rng.shuffle(normal_idx)
    overlap_count = max(1, int(round(len(df) * 0.04)))
    overlap_rows = normal_idx[:overlap_count]
    for idx in overlap_rows:
        sanctioned = float(df.at[idx, "sanctioned_amount"])
        noise_type = rng.choice(["cost_overrun", "delayed_completion", "sanction_delay"])
        if noise_type == "cost_overrun":
            ratio = float(rng.uniform(0.92, 1.08))
            df.at[idx, "expenditure_amount"] = round(sanctioned * ratio, 2)
        elif noise_type == "delayed_completion":
            recommended_dt = pd.to_datetime(df.at[idx, "recommended_date"])
            sanction_dt = recommended_dt + pd.Timedelta(days=int(rng.integers(20, 90)))
            df.at[idx, "sanction_date"] = sanction_dt.strftime("%Y-%m-%d")
            df.at[idx, "completion_percent"] = int(rng.integers(50, 85))
        else:
            recommended_dt = pd.to_datetime(df.at[idx, "recommended_date"])
            sanction_dt = recommended_dt + pd.Timedelta(days=int(rng.integers(12, 60)))
            df.at[idx, "sanction_date"] = sanction_dt.strftime("%Y-%m-%d")
            df.at[idx, "completion_percent"] = int(rng.integers(45, 90))

    # Add partial/ambiguous anomalies: mild anomaly signals but still labeled as anomalies.
    anomaly_idx = df.index[df["is_anomaly"] == True].tolist()
    rng.shuffle(anomaly_idx)
    partial_count = max(1, int(round(len(df) * 0.02)))
    partial_rows = anomaly_idx[:partial_count]
    for idx in partial_rows:
        anomaly_type = df.at[idx, "anomaly_type"]
        sanctioned = float(df.at[idx, "sanctioned_amount"])
        if anomaly_type == "cost_overrun":
            df.at[idx, "expenditure_amount"] = round(sanctioned * rng.uniform(1.02, 1.09), 2)
            df.at[idx, "completion_percent"] = int(rng.integers(70, 90))
        elif anomaly_type == "delayed_completion":
            sanction_dt = pd.to_datetime(df.at[idx, "sanction_date"])
            df.at[idx, "sanction_date"] = (sanction_dt - pd.Timedelta(days=int(rng.integers(30, 120)))).strftime("%Y-%m-%d")
            df.at[idx, "completion_percent"] = int(rng.integers(40, 75))
        elif anomaly_type == "sanction_delay":
            recommended_dt = pd.to_datetime(df.at[idx, "recommended_date"])
            sanction_dt = recommended_dt + pd.Timedelta(days=int(rng.integers(15, 45)))
            df.at[idx, "sanction_date"] = sanction_dt.strftime("%Y-%m-%d")
            df.at[idx, "completion_percent"] = int(rng.integers(40, 85))
        elif anomaly_type == "duplicate_beneficiary":
            current_ids = [x.strip() for x in str(df.at[idx, "beneficiary_ids"]).split(",") if x.strip()]
            if len(current_ids) < 3:
                df.at[idx, "beneficiary_ids"] = ",".join(current_ids + [f"B-DUP-PART-{rng.integers(1, 1000):03d}"])

    # 2-3% label noise to simulate imperfect real-world labels.
    label_noise_count = max(20, int(round(len(df) * 0.025)))
    anomaly_idx = df.index[df["is_anomaly"] == True].tolist()
    normal_idx = df.index[df["is_anomaly"] == False].tolist()
    rng.shuffle(anomaly_idx)
    rng.shuffle(normal_idx)

    flip_anomaly = anomaly_idx[: min(len(anomaly_idx), label_noise_count // 2)]
    flip_normal = normal_idx[: min(len(normal_idx), label_noise_count - len(flip_anomaly))]

    for idx in flip_anomaly:
        df.at[idx, "is_anomaly"] = False
        df.at[idx, "anomaly_type"] = None

    for idx in flip_normal:
        chosen_type = rng.choice(anomaly_types)
        df.at[idx, "is_anomaly"] = True
        df.at[idx, "anomaly_type"] = chosen_type
        sanctioned = float(df.at[idx, "sanctioned_amount"])
        if chosen_type == "cost_overrun":
            df.at[idx, "expenditure_amount"] = round(sanctioned * rng.uniform(1.02, 1.10), 2)
            df.at[idx, "completion_percent"] = int(rng.integers(70, 92))
        elif chosen_type == "delayed_completion":
            recommended_dt = pd.to_datetime(df.at[idx, "recommended_date"])
            sanction_dt = recommended_dt + pd.Timedelta(days=int(rng.integers(20, 90)))
            df.at[idx, "sanction_date"] = sanction_dt.strftime("%Y-%m-%d")
            df.at[idx, "completion_percent"] = int(rng.integers(50, 80))
        elif chosen_type == "sanction_delay":
            recommended_dt = pd.to_datetime(df.at[idx, "recommended_date"])
            sanction_dt = recommended_dt + pd.Timedelta(days=int(rng.integers(20, 60)))
            df.at[idx, "sanction_date"] = sanction_dt.strftime("%Y-%m-%d")
            df.at[idx, "completion_percent"] = int(rng.integers(45, 85))

    # Keep the overall anomaly rate near 10% and distribute it across all 5 anomaly types.
    target_total = int(round(len(df) * 0.10))
    desired_counts = {t: target_total // len(anomaly_types) for t in anomaly_types}
    remainder = target_total % len(anomaly_types)
    for t in anomaly_types[:remainder]:
        desired_counts[t] += 1

    for anomaly_type in anomaly_types:
        current_count = int((df["anomaly_type"] == anomaly_type).sum())
        excess = current_count - desired_counts[anomaly_type]
        if excess > 0:
            extra_idx = df.index[(df["is_anomaly"] == True) & (df["anomaly_type"] == anomaly_type)].tolist()
            rng.shuffle(extra_idx)
            for idx in extra_idx[:excess]:
                df.at[idx, "is_anomaly"] = False
                df.at[idx, "anomaly_type"] = None

    for anomaly_type in anomaly_types:
        current_count = int((df["anomaly_type"] == anomaly_type).sum())
        deficit = desired_counts[anomaly_type] - current_count
        if deficit > 0:
            candidate_idx = df.index[(df["is_anomaly"] == False)].tolist()
            rng.shuffle(candidate_idx)
            for idx in candidate_idx[:deficit]:
                df.at[idx, "is_anomaly"] = True
                df.at[idx, "anomaly_type"] = anomaly_type
                if anomaly_type == "cost_overrun":
                    sanctioned = float(df.at[idx, "sanctioned_amount"])
                    df.at[idx, "expenditure_amount"] = round(sanctioned * rng.uniform(1.02, 1.5), 2)
                    df.at[idx, "completion_percent"] = int(rng.integers(70, 100))
                elif anomaly_type == "delayed_completion":
                    recommended_dt = pd.to_datetime(df.at[idx, "recommended_date"])
                    sanction_dt = recommended_dt + pd.Timedelta(days=int(rng.integers(20, 90)))
                    df.at[idx, "sanction_date"] = sanction_dt.strftime("%Y-%m-%d")
                    df.at[idx, "completion_percent"] = int(rng.integers(40, 80))
                elif anomaly_type == "sanction_delay":
                    recommended_dt = pd.to_datetime(df.at[idx, "recommended_date"])
                    sanction_dt = recommended_dt + pd.Timedelta(days=int(rng.integers(20, 60)))
                    df.at[idx, "sanction_date"] = sanction_dt.strftime("%Y-%m-%d")
                    df.at[idx, "completion_percent"] = int(rng.integers(45, 85))
                elif anomaly_type == "duplicate_vendor":
                    df.at[idx, "vendor_id"] = f"V-DUP-{int(rng.integers(0, 15)):03d}"
                elif anomaly_type == "duplicate_beneficiary":
                    df.at[idx, "beneficiary_ids"] = f"B-DUP-{int(rng.integers(0, 15)):03d}"

    df["is_anomaly"] = df["is_anomaly"].fillna(False).astype(bool)
    df["anomaly_type"] = df["anomaly_type"].where(df["is_anomaly"], None)
    return df


def main() -> None:
    rng = np.random.default_rng(42)
    fake = Faker("en_IN")

    lok_df = pd.read_csv(RAW_DIR / "Lok_sabha.csv", encoding="utf-8-sig")
    raj_df = pd.read_csv(RAW_DIR / "Rajya_sabha.csv", encoding="utf-8-sig")

    lok_df = normalize_mp_df(lok_df, "Lok Sabha")
    raj_df = normalize_mp_df(raj_df, "Rajya Sabha")

    mp_df = pd.concat([lok_df, raj_df], ignore_index=True)
    mp_df["constituency"] = mp_df["constituency"].fillna("N/A")

    zero_alloc_mps = mp_df[mp_df["allocated_amount"] <= 0][["mp_name", "house", "allocated_amount"]].copy()
    if zero_alloc_mps.empty:
        print("Zero-allocation MPs removed before generation: 0")
    else:
        print(f"Zero-allocation MPs removed before generation: {len(zero_alloc_mps)}")
        print(zero_alloc_mps[["mp_name", "house", "allocated_amount"]].to_string(index=False))

    mp_df = mp_df[mp_df["allocated_amount"] > 0].copy()

    print("Max allocated amount per house:")
    print(mp_df.groupby("house")["allocated_amount"].max().to_string())

    works_df = build_work_records(mp_df, fake, rng)
    works_df = inject_anomalies(works_df, rng)

    mp_totals = works_df.groupby("mp_name")["sanctioned_amount"].sum().reset_index()
    mp_totals = mp_totals.merge(mp_df[["mp_name", "allocated_amount"]], on="mp_name", how="left")
    mp_totals["sanctioned_amount"] = mp_totals["sanctioned_amount"].round(2)
    mp_totals["allocated_amount"] = mp_totals["allocated_amount"].round(2)
    tol = 1.0
    over_cap = mp_totals[(mp_totals["sanctioned_amount"] - mp_totals["allocated_amount"]) > (mp_totals["allocated_amount"] * 0.0001 + tol)]
    if not over_cap.empty:
        print("Violating MP totals:")
        print(over_cap[["mp_name", "sanctioned_amount", "allocated_amount"]].to_string(index=False))
        raise ValueError(f"MP-level allocation cap violated for {len(over_cap)} MPs.")
    print(f"MP-level validation passed: {len(mp_totals)} MPs checked; violations={len(over_cap)}")

    max_single = float(works_df["sanctioned_amount"].max())
    print(f"Max single sanctioned_amount: ₹{max_single:,.0f}")
    if max_single > 200_000_000 + 1.0:
        raise ValueError(f"Sanctioned amount exceeds the real MP allocation ceiling: ₹{max_single:,.0f}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    works_df.to_csv(PROCESSED_DIR / "works.csv", index=False)

    print(f"Generated {len(works_df)} works across {len(mp_df)} MPs.")
    print(f"Total records: {len(works_df)}")
    print(f"Anomaly rows: {works_df['is_anomaly'].sum()}.")
    print(f"Anomaly type distribution:\n{works_df['anomaly_type'].value_counts(dropna=False).to_string()}")
    print(f"Saved to {PROCESSED_DIR / 'works.csv'}")


if __name__ == "__main__":
    main()
