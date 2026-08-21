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


@router.get("/token-window-debug", summary="Groq Token Window Live Diagnostic")
async def get_token_window_debug() -> dict:
    """
    Returns live in-memory state of GroqTokenWindow: active reservations, load ratio, provider telemetry.
    """
    import time
    from app.core.rate_limit import groq_token_window
    from app.core.config import settings

    now = time.time()
    with groq_token_window.lock:
        cutoff = now - 60
        active_reservations = []
        for res_id, (ts, tok) in list(groq_token_window.reservations.items()):
            active_reservations.append({
                "res_id": res_id,
                "tokens": tok,
                "timestamp": ts,
                "age_seconds": round(now - ts, 2),
                "is_expired": (ts <= cutoff)
            })

        used_tokens, effective_limit, ratio = groq_token_window.get_window_load(limit=settings.GROQ_TPM_LIMIT)

        return {
            "current_time": now,
            "used_tokens": used_tokens,
            "effective_limit": effective_limit,
            "soft_cap_ratio": getattr(settings, "GROQ_SOFT_CAP_RATIO", 0.85),
            "current_load_ratio": round(ratio, 4),
            "soft_cap_triggered": (ratio >= getattr(settings, "GROQ_SOFT_CAP_RATIO", 0.85)),
            "provider_remaining": groq_token_window.provider_remaining,
            "provider_limit": groq_token_window.provider_limit,
            "provider_reset_ts": groq_token_window.provider_reset_ts,
            "provider_reset_seconds_left": round(groq_token_window.provider_reset_ts - now, 2) if groq_token_window.provider_reset_ts else None,
            "total_reservations_count": len(groq_token_window.reservations),
            "reservations": active_reservations,
        }

