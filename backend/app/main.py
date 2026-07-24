"""
DocMind AI Backend Application Entrypoint.

Configures the FastAPI application instance, CORS middleware, API routes,
and foundational system endpoints (Root and Health check).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict

from app.core.config import settings
from app.api.router import api_router

# Initialize FastAPI Application with metadata
app = FastAPI(
    title="DocMind AI API",
    version="1.0.0",
    description="Production-grade backend API foundation for DocMind AI.",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure Cross-Origin Resource Sharing (CORS) Middleware for Local Frontend Development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(api_router)


@app.get("/", response_model=Dict[str, str], summary="Root API Information Endpoint")
async def root() -> Dict[str, str]:
    """
    Root API endpoint providing core application metadata.
    
    Returns:
        JSON object containing project name, status, and application version.
    """
    return {
        "project": "DocMind AI",
        "status": "running",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)
