"""
Pydantic models mirroring the team's shared API contract.

IMPORTANT: These shapes are locked by team agreement (api-contracts.md).
Do not add/rename/remove fields without updating the contract doc and
notifying the ML + frontend teammates — they build against this shape.
"""
from datetime import date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class WorkCategory(str, Enum):
    road = "Road"
    water = "Water"
    education = "Education"
    health = "Health"
    sanitation = "Sanitation"
    other = "Other"


class WorkStatus(str, Enum):
    recommended = "recommended"
    sanctioned = "sanctioned"
    ongoing = "ongoing"
    completed = "completed"


class GeoLocation(BaseModel):
    lat: float
    lng: float


class Work(BaseModel):
    """Core Work entity — the shared shape everyone builds against."""

    work_id: str
    mp_name: str
    constituency: str
    state: str
    district: str
    implementing_agency: str
    vendor_id: str
    work_category: WorkCategory
    recommended_date: date
    recommended_amount: float
    sanction_date: Optional[date] = None
    sanctioned_amount: Optional[float] = None
    expenditure_amount: float = 0
    status: WorkStatus
    completion_percent: float = Field(ge=0, le=100, default=0)
    completion_date: Optional[date] = None
    beneficiary_ids: List[str] = []
    geo_location: GeoLocation
    photo_urls: List[str] = []


class RiskFlag(str, Enum):
    cost_overrun = "cost_overrun"
    delayed_completion = "delayed_completion"
    sanction_delay = "sanction_delay"
    duplicate_vendor = "duplicate_vendor"
    duplicate_beneficiary = "duplicate_beneficiary"
    inflated_billing = "inflated_billing"
    incomplete_reporting = "incomplete_reporting"


class RiskScore(BaseModel):
    """Fixed-vocabulary output contract for the ML predict() function."""

    work_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    is_anomaly: bool
    flags: List[RiskFlag] = []
    explanation: str


class Alert(BaseModel):
    """An alert is a RiskScore for a flagged work, plus display metadata."""

    work_id: str
    mp_name: str
    constituency: str
    state: str
    district: str
    risk_score: float = Field(ge=0.0, le=1.0)
    flags: List[RiskFlag] = []
    explanation: str
    raised_at: date


class DashboardSummary(BaseModel):
    total_works: int
    total_sanctioned_amount: float
    total_expenditure_amount: float
    total_alerts: int
    high_risk_works: int
    works_by_status: dict
    works_by_category: dict
    avg_risk_score: float