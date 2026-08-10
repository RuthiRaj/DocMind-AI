"""
DocMind AI Backend Application Entrypoint.

Configures the FastAPI application instance, CORS middleware, API routes,
and foundational system endpoints (Root and Health check).
"""

import logging
import sys
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
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

# Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# Standardized Error Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    message = exc.detail
    error_code = "BAD_REQUEST"
    hint = "Please check your input parameters and try again."

    if exc.status_code == 404:
        error_code = "RESOURCE_NOT_FOUND"
        hint = "The requested resource does not exist on the server."
    elif exc.status_code == 409:
        error_code = "CONCURRENT_REQUEST_CONFLICT"
        hint = "Wait for the active stage execution to complete."
    elif exc.status_code == 413:
        error_code = "PAYLOAD_TOO_LARGE"
        path = request.url.path.rstrip("/")
        if path.startswith("/chat") or "/chat/" in path:
            hint = "Try a shorter question or fewer chat turns — the AI model's input limit was reached."
        elif path.startswith("/upload") or path.endswith("/upload"):
            hint = "Limit upload size to match application settings parameters."
        else:
            hint = "The request payload exceeds the allowed size limit."
    elif exc.status_code == 400:
        if "uuid" in str(message).lower() or "document id" in str(message).lower():
            error_code = "INVALID_DOCUMENT_ID"
            hint = "Ensure document ID is a valid UUID v4 format."
        elif "chunk" in str(message).lower():
            error_code = "CHUNK_VALIDATION_FAILED"
            hint = "Validate file structure content bounds."

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": error_code,
            "message": message,
            "hint": hint,
            "detail": message
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "message": "Request validation failed.",
            "hint": str(exc.errors()),
            "detail": "Request validation failed."
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled application error: %s", str(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred.",
            "hint": "Please try again later.",
            "detail": "An internal server error occurred."
        }
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
