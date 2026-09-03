from __future__ import annotations

from predict import predict


def print_case_result(name: str, expected: bool, result: dict) -> None:
    print(f"\n=== {name} ===")
    print(result)
    ok = result.get("is_anomaly") is expected
    status = "PASS" if ok else "FAIL"
    print(f"{status}: expected is_anomaly={expected}, got {result.get('is_anomaly')}")
    assert result.get("is_anomaly") is expected, (
        f"{name}: expected is_anomaly={expected}, got {result.get('is_anomaly')}"
    )


def build_normal_work() -> dict:
    return {
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
    }


def build_cost_overrun_case() -> dict:
    return {
        "work_id": "W-COST-001",
        "sanctioned_amount": 2500000,
        "expenditure_amount": 3300000,
        "vendor_id": "V-REG-001",
        "beneficiary_ids": "B-REG-001",
        "recommended_date": "2026-05-10",
        "sanction_date": "2026-06-15",
        "completion_percent": 82,
        "work_category": "Roads",
        "house": "Lok Sabha",
        "district": "Jaipur",
        "constituency": "Jaipur",
    }


def build_delayed_completion_case() -> dict:
    return {
        "work_id": "W-DELAY-001",
        "sanctioned_amount": 2000000,
        "expenditure_amount": 1800000,
        "vendor_id": "V-OK-NEW",
        "beneficiary_ids": "B-OK-NEW",
        "recommended_date": "2025-01-15",
        "sanction_date": "2025-07-15",
        "completion_percent": 10,
        "work_category": "Sanitation",
        "house": "Rajya Sabha",
        "district": "Bhopal",
        "constituency": "Bhopal",
    }


def build_sanction_delay_case() -> dict:
    return {
        "work_id": "W-SANCTION-DELAY-001",
        "sanctioned_amount": 2100000,
        "expenditure_amount": 1700000,
        "vendor_id": "V-REG-003",
        "beneficiary_ids": "B-REG-003",
        "recommended_date": "2026-01-10",
        "sanction_date": "2026-09-10",
        "completion_percent": 84,
        "work_category": "Education",
        "house": "Lok Sabha",
        "district": "Patna",
        "constituency": "Patna",
    }


def build_duplicate_vendor_case() -> dict:
    return {
        "work_id": "W-DUP-VENDOR-001",
        "sanctioned_amount": 3000000,
        "expenditure_amount": 2600000,
        "vendor_id": "V-DUP-000",
        "beneficiary_ids": "B-UNIQUE-101",
        "recommended_date": "2026-02-01",
        "sanction_date": "2026-03-10",
        "completion_percent": 80,
        "work_category": "Water Supply",
        "house": "Lok Sabha",
        "district": "Kolkata",
        "constituency": "Kolkata",
    }


def build_duplicate_beneficiary_case() -> dict:
    return {
        "work_id": "W-DUP-BENEF-001",
        "sanctioned_amount": 2900000,
        "expenditure_amount": 2400000,
        "vendor_id": "V-REG-004",
        "beneficiary_ids": "B-DUP-000,B-DUP-000,B-DUP-000",
        "recommended_date": "2026-04-15",
        "sanction_date": "2026-05-05",
        "completion_percent": 79,
        "work_category": "Education",
        "house": "Rajya Sabha",
        "district": "Lucknow",
        "constituency": "Lucknow",
    }


def main() -> None:
    cases = [
        ("normal", False, build_normal_work()),
        ("cost_overrun", True, build_cost_overrun_case()),
        ("delayed_completion", True, build_delayed_completion_case()),
        ("sanction_delay", True, build_sanction_delay_case()),
        ("duplicate_vendor", True, build_duplicate_vendor_case()),
        ("duplicate_beneficiary", True, build_duplicate_beneficiary_case()),
    ]

    for name, expected, record in cases:
        result = predict(record)
        print_case_result(name, expected, result)

    print("\nAll cases completed.")


if __name__ == "__main__":
    main()
