"""
PDF Maintenance API Route Handlers.

Provides endpoints to trigger system-wide storage maintenance operations.
"""

from fastapi import APIRouter, status

from app.schemas.management import CleanupResponse
from app.services.pdf.maintenance_service import MaintenanceService

router = APIRouter()
maintenance_service = MaintenanceService()


@router.post(
    "/maintenance/cleanup",
    response_model=CleanupResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger Storage Maintenance Cleanup",
    description="Sweeps the storage folder removing all temporary *.tmp files, cleaning empty folders, and removing orphan statistics definitions."
)
async def trigger_cleanup() -> CleanupResponse:
    """
    HTTP POST endpoint handler to trigger system cleanup.
    """
    res = maintenance_service.cleanup()
    return CleanupResponse(**res)
