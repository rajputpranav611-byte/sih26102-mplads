from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app import mock_data
from app.auth import Role, get_current_role
from app.schemas import Work, WorkCategory, WorkStatus

router = APIRouter(prefix="/api/works", tags=["works"])


@router.get("", response_model=List[Work])
def list_works(
    state: Optional[str] = None,
    district: Optional[str] = None,
    status_: Optional[WorkStatus] = Query(default=None, alias="status"),
    category: Optional[WorkCategory] = None,
    mp_name: Optional[str] = None,
    role: Role = Depends(get_current_role),
):
    """
    List works, optionally filtered by state/district/status/category/mp_name.

    Role scoping (mock-data stage — replace with real ownership lookups
    once DB is wired in):
    - MP: intended to see only their own constituency's works.
    - DistrictAuthority: intended to see only their district's works.
    - Ministry: sees everything.
    Filtering is currently by query params rather than enforced by
    identity, since there's no auth/user table yet.
    """
    works = mock_data.get_all_works()


    if state:
        works = [w for w in works if w.state.lower() == state.lower()]
    if district:
        works = [w for w in works if w.district.lower() == district.lower()]
    if status_:
        works = [w for w in works if w.status == status_]
    if category:
        works = [w for w in works if w.work_category == category]
    if mp_name:
        works = [w for w in works if w.mp_name.lower() == mp_name.lower()]

    return works


@router.get("/{work_id}", response_model=Work)
def get_work(work_id: str, role: Role = Depends(get_current_role)):
    work = mock_data.get_work_by_id(work_id)
    if not work:
        raise HTTPException(status_code=404, detail=f"Work '{work_id}' not found")
    return work