from fastapi import APIRouter, Depends, HTTPException, Query

from app import mock_data
from app.auth import Role, get_current_role
from app.schemas import RiskScore

router = APIRouter(prefix="/api/risk-score", tags=["risk-score"])

import sys
import os
import datetime
from pydantic import BaseModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../ml/src")))
from predict import predict as ml_predict

class TestPredictRequest(BaseModel):
    sanctioned_amount: float
    expenditure_amount: float
    days_since_sanction: int
    completion_percent: float
    vendor_id: str
    work_category: str
    house: str

@router.post("/test-predict", response_model=RiskScore)
def test_predict(request: TestPredictRequest):
    sanction_date = datetime.date.today() - datetime.timedelta(days=request.days_since_sanction)
    work_data = {
        "work_id": "W-TEST-001",
        "sanctioned_amount": request.sanctioned_amount,
        "expenditure_amount": request.expenditure_amount,
        "sanction_date": sanction_date.isoformat(),
        "completion_percent": request.completion_percent,
        "vendor_id": request.vendor_id,
        "work_category": request.work_category,
        "house": request.house,
    }
    result = ml_predict(work_data)
    return result

@router.get("/{work_id}", response_model=RiskScore)
def get_risk_score(work_id: str, house: str | None = Query(default=None), role: Role = Depends(get_current_role)):
    work = mock_data.get_work_by_id(work_id, house=house)
    if not work:
        raise HTTPException(status_code=404, detail=f"Work '{work_id}' not found")

    prediction = mock_data.get_risk_score(work_id, house=house)
    if prediction is None:
        raise HTTPException(status_code=404, detail=f"Work '{work_id}' not found")
    return prediction