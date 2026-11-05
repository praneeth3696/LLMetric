"""LLMetrics Budget Tracking Endpoints."""

from fastapi import APIRouter, HTTPException, status
from backend.schemas.api_models import BudgetUpdateRequest, BudgetResponse
from backend.services.analytics_service import update_and_project_budget

router = APIRouter(prefix="", tags=["Budget & Alerting"])

@router.post(
    "/budget",
    response_model=BudgetResponse,
    summary="Update monthly budget limit",
)
def set_budget(req: BudgetUpdateRequest) -> BudgetResponse:
    """Updates the target monthly budget and re-evaluates breach probability."""
    try:
        res = update_and_project_budget(new_budget=req.monthly_budget)
        return BudgetResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get(
    "/budget",
    response_model=BudgetResponse,
    summary="Retrieve current budget tracking & breach risk",
)
def get_budget() -> BudgetResponse:
    """Returns current month-end projected spend, prediction intervals, and breach probability."""
    try:
        res = update_and_project_budget()
        return BudgetResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
