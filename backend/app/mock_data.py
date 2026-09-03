"""
Synthetic data generator.

Used by routers until the real PostgreSQL-backed repository layer
and the teammate's ML predict() function are wired in. Swap this out
without changing router signatures or response shapes.
"""
import hashlib
import random
from datetime import date, timedelta
from typing import List

from app.schemas import (
    Alert,
    DashboardSummary,
    GeoLocation,
    RiskFlag,
    RiskScore,
    Work,
    WorkCategory,
    WorkStatus,
)

random.seed(42)

STATES_DISTRICTS = {
    "Kerala": ["Ernakulam", "Thrissur", "Kozhikode"],
    "Maharashtra": ["Pune", "Nagpur", "Nashik"],
    "Uttar Pradesh": ["Lucknow", "Varanasi", "Kanpur"],
    "Gujarat": ["Ahmedabad", "Surat", "Bhavnagar"],
}
MP_NAMES = [

    "A. Sharma", "B. Nair", "C. Reddy", "D. Singh", "E. Patel", "F. Iyer",
]
AGENCIES = ["PWD", "Zilla Parishad", "Municipal Corporation", "Rural Dev Dept"]
VENDORS = [f"VEND-{i:04d}" for i in range(1, 21)]


def _rand_date(start_year=2024, end_year=2026) -> date:
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def _work_id(i: int, state: str) -> str:
    code = state[:2].upper()
    return f"MPLAD-2026-{code}-{i:05d}"


def generate_work(i: int) -> Work:
    state = random.choice(list(STATES_DISTRICTS.keys()))
    district = random.choice(STATES_DISTRICTS[state])
    recommended_date = _rand_date(2024, 2025)
    status = random.choices(
        list(WorkStatus), weights=[0.1, 0.2, 0.4, 0.3]
    )[0]

    sanction_date = None
    sanctioned_amount = None
    expenditure_amount = 0.0
    completion_date = None
    completion_percent = 0.0


    recommended_amount = round(random.uniform(500000, 5000000), 2)

    if status != WorkStatus.recommended:
        sanction_date = recommended_date + timedelta(days=random.randint(5, 60))
        sanctioned_amount = round(
            recommended_amount * random.uniform(0.9, 1.05), 2
        )

    if status in (WorkStatus.ongoing, WorkStatus.completed):
        completion_percent = round(random.uniform(10, 99), 1)
        expenditure_amount = round(
            sanctioned_amount * (completion_percent / 100) * random.uniform(0.9, 1.3),
            2,
        )

    if status == WorkStatus.completed:
        completion_percent = 100.0
        completion_date = sanction_date + timedelta(days=random.randint(180, 600))
        expenditure_amount = round(sanctioned_amount * random.uniform(0.95, 1.4), 2)

    return Work(
        work_id=_work_id(i, state),
        mp_name=random.choice(MP_NAMES),
        constituency=f"{district} Constituency",
        state=state,
        district=district,
        implementing_agency=random.choice(AGENCIES),
        vendor_id=random.choice(VENDORS),
        work_category=random.choice(list(WorkCategory)),
        recommended_date=recommended_date,
        recommended_amount=recommended_amount,
        sanction_date=sanction_date,

        sanctioned_amount=sanctioned_amount,
        expenditure_amount=expenditure_amount,
        status=status,
        completion_percent=completion_percent,
        completion_date=completion_date,
        beneficiary_ids=[f"BEN-{i}-{j}" for j in range(random.randint(0, 5))],
        geo_location=GeoLocation(
            lat=round(random.uniform(8.0, 30.0), 6),
            lng=round(random.uniform(70.0, 88.0), 6),
        ),
        photo_urls=[],
    )


_WORKS: List[Work] = [generate_work(i) for i in range(1, 121)]


def get_all_works() -> List[Work]:
    return _WORKS


def get_work_by_id(work_id: str) -> Work | None:
    return next((w for w in _WORKS if w.work_id == work_id), None)


def mock_predict(work: Work) -> RiskScore:
    """
    Stand-in for the teammate's ML predict(work_data) -> RiskScore.
    Deterministic (hash-based) so results are stable across calls/tests
    until the real model is wired in via app/ml_integration.py.
    """
    seed = int(hashlib.md5(work.work_id.encode()).hexdigest(), 16)

    rng = random.Random(seed)

    flags: List[RiskFlag] = []
    score = rng.uniform(0.0, 0.35)

    if work.sanctioned_amount and work.expenditure_amount > work.sanctioned_amount * 1.15:
        flags.append(RiskFlag.cost_overrun)
        score += 0.25

    if work.sanction_date and (work.sanction_date - work.recommended_date).days > 45:
        flags.append(RiskFlag.sanction_delay)
        score += 0.15

    if (
        work.status == WorkStatus.ongoing
        and work.sanction_date
        and (date.today() - work.sanction_date).days > 365
    ):
        flags.append(RiskFlag.delayed_completion)
        score += 0.2

    if rng.random() < 0.08:
        flags.append(RiskFlag.duplicate_vendor)
        score += 0.2

    if rng.random() < 0.06:
        flags.append(RiskFlag.duplicate_beneficiary)
        score += 0.2

    if rng.random() < 0.1:
        flags.append(RiskFlag.incomplete_reporting)
        score += 0.1


    score = min(round(score, 3), 1.0)
    is_anomaly = score >= 0.5

    if flags:
        explanation = "Flagged for: " + ", ".join(f.value.replace("_", " ") for f in flags)
    else:
        explanation = "No significant anomalies detected."

    return RiskScore(
        work_id=work.work_id,
        risk_score=score,
        is_anomaly=is_anomaly,
        flags=flags,
        explanation=explanation,
    )


def get_all_risk_scores() -> List[RiskScore]:
    return [mock_predict(w) for w in _WORKS]


def get_alerts() -> List[Alert]:
    alerts = []
    for work in _WORKS:
        rs = mock_predict(work)
        if rs.is_anomaly or rs.flags:
            alerts.append(
                Alert(
                    work_id=work.work_id,
                    mp_name=work.mp_name,
                    constituency=work.constituency,

                    state=work.state,
                    district=work.district,
                    risk_score=rs.risk_score,
                    flags=rs.flags,
                    explanation=rs.explanation,
                    raised_at=date.today(),
                )
            )
    alerts.sort(key=lambda a: a.risk_score, reverse=True)
    return alerts


def get_dashboard_summary() -> DashboardSummary:
    works = _WORKS
    scores = get_all_risk_scores()

    works_by_status: dict = {}
    for w in works:
        works_by_status[w.status.value] = works_by_status.get(w.status.value, 0) + 1

    works_by_category: dict = {}
    for w in works:
        works_by_category[w.work_category.value] = (
            works_by_category.get(w.work_category.value, 0) + 1
        )

    return DashboardSummary(
        total_works=len(works),
        total_sanctioned_amount=round(
            sum(w.sanctioned_amount or 0 for w in works), 2
        ),
        total_expenditure_amount=round(

            sum(w.expenditure_amount for w in works), 2
        ),
        total_alerts=sum(1 for s in scores if s.is_anomaly or s.flags),
        high_risk_works=sum(1 for s in scores if s.risk_score >= 0.5),
        works_by_status=works_by_status,
        works_by_category=works_by_category,
        avg_risk_score=round(sum(s.risk_score for s in scores) / len(scores), 3),
    )