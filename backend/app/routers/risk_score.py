from fastapi import APIRouter, Depends, HTTPException

from app import mock_data
from app.auth import Role, get_current_role
from app.schemas import RiskScore

router = APIRouter(prefix="/api/risk-score", tags=["risk-score"])


@router.get("/{work_id}", response_model=RiskScore)
def get_risk_score(work_id: str, role: Role = Depends(get_current_role)):
    """
    Returns the ML risk assessment for one work.

    Currently backed by mock_data.mock_predict(). Once the teammate's
    real predict(work_data) -> dict is ready, swap the call in
    app/ml_integration.py — this router's contract stays the same.
    """
    work = mock_data.get_work_by_id(work_id)
    if not work:
        raise HTTPException(status_code=404, detail=f"Work '{work_id}' not found")
    return mock_data.mock_predict(work)