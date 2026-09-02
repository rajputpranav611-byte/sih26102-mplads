from typing import List, Optional

from fastapi import APIRouter, Depends, Query

from app import mock_data
from app.auth import Role, get_current_role
from app.schemas import Alert, RiskFlag

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("", response_model=List[Alert])
def list_alerts(
    min_risk_score: float = Query(default=0.0, ge=0.0, le=1.0),
    flag: Optional[RiskFlag] = None,
    state: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=500),
    role: Role = Depends(get_current_role),
):
    """Sorted (highest risk first) list of flagged works, filterable by
    minimum risk score, a specific flag, and/or state."""
    alerts = mock_data.get_alerts()

    alerts = [a for a in alerts if a.risk_score >= min_risk_score]
    if flag:
        alerts = [a for a in alerts if flag in a.flags]
    if state:
        alerts = [a for a in alerts if a.state.lower() == state.lower()]

    return alerts[:limit]