"""
Minimal role-based access control.

STUB: reads role from an `X-User-Role` header so frontend/ML teammates
can start integrating immediately. Swap `get_current_role` internals
for real JWT/session auth later without changing router code, since
routers only depend on this function via FastAPI's Depends().
"""
from enum import Enum

from fastapi import Header, HTTPException, status


class Role(str, Enum):
    mp = "MP"
    district_authority = "DistrictAuthority"
    ministry = "Ministry"


def get_current_role(x_user_role: str = Header(default=Role.ministry.value)) -> Role:
    try:
        return Role(x_user_role)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid X-User-Role header. Must be one of: {[r.value for r in Role]}",
        )


def require_roles(*allowed: Role):
    """Dependency factory: restrict an endpoint to specific roles.

    Usage: `role: Role = Depends(require_roles(Role.ministry, Role.district_authority))`
    """

    def _check(role: Role = Header(default=Role.ministry.value, alias="X-User-Role")):
        resolved = get_current_role(role)
        if resolved not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{resolved.value}' not permitted. Allowed: {[r.value for r in allowed]}",
            )
        return resolved

    return _check