"""
System & Telemetry API Route Handlers.

Provides system health checks and live Groq LLM telemetry status endpoints.
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


@router.get("/telemetry", summary="Groq LLM Live Telemetry")
async def get_telemetry() -> dict:
    """
    Returns recent live Groq API calls with exact token counts and rate limit headers.
    """
    from app.core.telemetry import groq_telemetry
    return {
        "status": "healthy",
        "recent_calls": groq_telemetry.get_recent(20)
    }
