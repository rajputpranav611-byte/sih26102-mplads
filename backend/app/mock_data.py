"""CSV-backed work repository used by the API while a database is unavailable."""

import csv
import time
from datetime import date
from pathlib import Path
from typing import List, Optional

from app.schemas import DashboardSummary, GeoLocation, RiskScore, Work, WorkCategory, WorkStatus

DATA_PATH = Path(__file__).resolve().parents[2] / "ml" / "data" / "processed" / "works.csv"


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _category(value: str) -> WorkCategory:
    try:
        return WorkCategory(value.title())
    except ValueError:
        return WorkCategory.other


def _status(completion_percent: float) -> WorkStatus:
    if completion_percent <= 0:
        return WorkStatus.recommended
    if completion_percent >= 100:
        return WorkStatus.completed
    return WorkStatus.ongoing


def _load_works() -> tuple[List[Work], dict[str, str]]:
    works: List[Work] = []
    houses: dict[str, str] = {}
    with DATA_PATH.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            sanctioned_amount = float(row.get("sanctioned_amount") or 0)
            expenditure_amount = float(row.get("expenditure_amount") or 0)
            completion_percent = float(row.get("completion_percent") or 0)
            work_id = row["work_id"]
            constituency = row.get("constituency", "Unknown").strip()
            beneficiary_ids = [item.strip() for item in row.get("beneficiary_ids", "").split(",") if item.strip()]
            works.append(Work(
                work_id=work_id,
                mp_name=row.get("mp_name", "Unknown").strip(),
                constituency=constituency,
                state=row.get("state", "Unknown").strip(),
                district=constituency,
                implementing_agency="MPLADS implementing agency",
                vendor_id=row.get("vendor_id", "UNKNOWN").strip(),
                work_category=_category(row.get("work_category", "Other")),
                recommended_date=_parse_date(row.get("recommended_date")) or date.today(),
                recommended_amount=sanctioned_amount,
                sanction_date=_parse_date(row.get("sanction_date")),
                sanctioned_amount=sanctioned_amount,
                expenditure_amount=expenditure_amount,
                status=_status(completion_percent),
                completion_percent=completion_percent,
                completion_date=None,
                beneficiary_ids=beneficiary_ids,
                geo_location=GeoLocation(lat=0.0, lng=0.0),
                photo_urls=[],
            ))
            houses[work_id] = row.get("house", "Lok Sabha").strip()
    return works, houses


_WORKS, _HOUSES = _load_works()


def get_all_works(house: Optional[str] = None) -> List[Work]:
    if not house:
        return _WORKS
    return [work for work in _WORKS if _HOUSES.get(work.work_id, "").lower() == house.lower()]


def get_work_by_id(work_id: str, house: Optional[str] = None) -> Work | None:
    work = next((item for item in _WORKS if item.work_id == work_id), None)
    if work is None or (house and _HOUSES.get(work_id, "").lower() != house.lower()):
        return None
    return work


def get_house(work_id: str) -> str | None:
    return _HOUSES.get(work_id)


def _prediction_input(work: Work) -> dict:
    return {
        "work_id": work.work_id,
        "sanctioned_amount": work.sanctioned_amount,
        "expenditure_amount": work.expenditure_amount,
        "vendor_id": work.vendor_id,
        "beneficiary_ids": ",".join(work.beneficiary_ids),
        "recommended_date": work.recommended_date,
        "sanction_date": work.sanction_date,
        "completion_percent": work.completion_percent,
        "work_category": work.work_category.value,
        "house": get_house(work.work_id) or "Lok Sabha",
        "district": work.district,
        "constituency": work.constituency,
    }


def mock_predict(work: Work) -> RiskScore:
    """Run one prediction for detail/test callers."""
    from app.services import ml_client
    return RiskScore.model_validate(ml_client.get_risk_prediction(_prediction_input(work)))


# Predict once at startup with vectorized feature engineering and batch models.
from app.services import ml_client

_cache_started = time.perf_counter()
print(f"Computing risk scores for all works... ({len(_WORKS)} records)", flush=True)
_BATCH_PREDICTIONS = ml_client.get_risk_predictions([_prediction_input(work) for work in _WORKS])
_RISK_SCORES = {item["work_id"]: RiskScore.model_validate(item) for item in _BATCH_PREDICTIONS}
print(f"Done in {time.perf_counter() - _cache_started:.3f} seconds", flush=True)


def get_risk_score(work_id: str, house: Optional[str] = None) -> RiskScore | None:
    if work_id not in _RISK_SCORES:
        return None
    if house and _HOUSES.get(work_id, "").lower() != house.lower():
        return None
    return _RISK_SCORES[work_id]


def get_all_risk_scores(house: Optional[str] = None) -> List[RiskScore]:
    return [_RISK_SCORES[work.work_id] for work in get_all_works(house)]


def get_dashboard_summary(house: Optional[str] = None) -> DashboardSummary:
    works = get_all_works(house)
    scores = get_all_risk_scores(house)
    works_by_status: dict[str, int] = {}
    works_by_category: dict[str, int] = {}
    for work in works:
        works_by_status[work.status.value] = works_by_status.get(work.status.value, 0) + 1
        works_by_category[work.work_category.value] = works_by_category.get(work.work_category.value, 0) + 1
    return DashboardSummary(
        total_works=len(works),
        total_sanctioned_amount=round(sum(work.sanctioned_amount or 0 for work in works), 2),
        total_expenditure_amount=round(sum(work.expenditure_amount for work in works), 2),
        total_alerts=sum(1 for score in scores if score.is_anomaly or score.flags),
        high_risk_works=sum(1 for score in scores if score.risk_score >= 0.5),
        works_by_status=works_by_status,
        works_by_category=works_by_category,
        avg_risk_score=round(sum(score.risk_score for score in scores) / len(scores), 3) if scores else 0,
    )
