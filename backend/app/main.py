"""
DocMind AI Backend Application Entrypoint.

Configures the FastAPI application instance, CORS middleware, API routes,
and foundational system endpoints (Root and Health check).
"""

import logging
import sys
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict

from app.core.config import settings
from app.api.router import api_router
from app.services.embeddings.sentence_transformer_provider import SentenceTransformerProvider
from app.core.rate_limit import RateLimitMiddleware

# Configure basic logging level on startup
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Lifespan context manager for application startup and shutdown events
@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    # Startup: Load the embedding model singleton
    logger.info("FastAPI application starting up...")
    start_time = time.perf_counter()
    try:
        logger.info("Loading embedding model '%s'...", settings.EMBEDDING_MODEL)
        provider = SentenceTransformerProvider()
        provider.initialize_model()
        elapsed = time.perf_counter() - start_time
        logger.info("Embedding model loaded successfully. Initialization completed in %.2f seconds.", elapsed)
    except Exception as exc:
        logger.critical("CRITICAL: Failed to load embedding model on startup: %s", str(exc), exc_info=True)
        # Fail startup cleanly by exiting the process
        sys.exit(1)
    yield
    logger.info("FastAPI application shutting down...")

# Initialize FastAPI Application with metadata and lifespan
app = FastAPI(
    title="DocMind AI API",
    version="1.0.0",
    description="Production-grade backend API foundation for DocMind AI.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure Cross-Origin Resource Sharing (CORS) Middleware for Local Frontend Development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Rate Limiting Middleware
app.add_middleware(RateLimitMiddleware)

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
