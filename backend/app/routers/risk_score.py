from fastapi import APIRouter, Depends, HTTPException, Query

from app import mock_data
from app.auth import Role, get_current_role
from app.schemas import RiskScore

router = APIRouter(prefix="/api/risk-score", tags=["risk-score"])


@router.get("/{work_id}", response_model=RiskScore)
def get_risk_score(work_id: str, house: str | None = Query(default=None), role: Role = Depends(get_current_role)):
    work = mock_data.get_work_by_id(work_id, house=house)
    if not work:
        raise HTTPException(status_code=404, detail=f"Work '{work_id}' not found")

    prediction = mock_data.get_risk_score(work_id, house=house)
    if prediction is None:
        raise HTTPException(status_code=404, detail=f"Work '{work_id}' not found")
    return prediction