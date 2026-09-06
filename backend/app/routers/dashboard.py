from fastapi import APIRouter, Depends, Query

from app import mock_data
from app.auth import Role, get_current_role
from app.schemas import DashboardSummary

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummary)
def dashboard_summary(house: str = Query(default="Lok Sabha"), role: Role = Depends(get_current_role)):
    """Aggregate stats for the dashboard landing view."""
    return mock_data.get_dashboard_summary(house=house)