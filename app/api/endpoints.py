"""
API Endpoint Definitions.

Contains health check and diagnostic route handlers for system verification.
"""

from fastapi import APIRouter
from app.schemas.management import HealthCheckResponse
from app.core.health import get_system_health

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse, summary="Health Check Endpoint")
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint to verify backend operational readiness and dependencies.
    
    Returns:
        JSON response with complete service diagnostics health status.
    """
    health_data = get_system_health()
    return HealthCheckResponse(**health_data)
