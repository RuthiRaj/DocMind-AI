"""
System Diagnostic Health Check Helper.

Provides diagnostic check operations verifying library availability,
uploads folder write permissions, and core configurations without making network queries.
Tracks active system uptime.
"""

import os
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Any

from app.core.config import settings

logger = logging.getLogger(__name__)

# Start time tracking for uptime calculation
START_TIME = time.time()


def check_upload_directory() -> Dict[str, Any]:
    """
    Verifies that the upload folder directory exists.
    """
    uploads_dir = Path(__file__).resolve().parent.parent.parent.parent / settings.UPLOAD_DIRECTORY
    status = "healthy"
    details = ""

    if not uploads_dir.exists():
        try:
            uploads_dir.mkdir(parents=True, exist_ok=True)
            details = "Uploads directory did not exist; created successfully."
        except Exception as exc:
            status = "unhealthy"
            details = f"Failed to create uploads directory: {str(exc)}"
    else:
        details = "Uploads directory exists."

    return {
        "status": status,
        "path": str(uploads_dir),
        "details": details
    }


def check_disk_permissions() -> Dict[str, Any]:
    """
    Checks write permissions inside the upload folder directory.
    """
    uploads_dir = Path(__file__).resolve().parent.parent.parent.parent / settings.UPLOAD_DIRECTORY
    status = "healthy"
    details = ""

    test_file = uploads_dir / ".health_check.tmp"
    try:
        # Verify folder is writable
        uploads_dir.mkdir(parents=True, exist_ok=True)
        with open(test_file, "w") as tf:
            tf.write("health check write test")
        test_file.unlink()
        details = "Write permissions verified successfully."
    except Exception as exc:
        status = "unhealthy"
        details = f"Uploads directory is not writable: {str(exc)}"

    return {
        "status": status,
        "details": details
    }


def check_disk_space() -> Dict[str, Any]:
    """
    Fetches disk accessibility metrics for storage directory location.
    """
    uploads_dir = Path(__file__).resolve().parent.parent.parent.parent / settings.UPLOAD_DIRECTORY
    try:
        usage = shutil.disk_usage(str(uploads_dir.parent))
        status = "healthy"
        details = f"Free: {usage.free / (1024**3):.2f} GB, Total: {usage.total / (1024**3):.2f} GB"
    except Exception as exc:
        status = "unhealthy"
        details = f"Failed to inspect storage disk usage: {str(exc)}"
        usage = None

    return {
        "status": status,
        "total_bytes": usage.total if usage else 0,
        "used_bytes": usage.used if usage else 0,
        "free_bytes": usage.free if usage else 0,
        "details": details
    }


def check_sentence_transformer() -> Dict[str, Any]:
    """
    Checks SentenceTransformer library availability.
    """
    try:
        import sentence_transformers
        status = "healthy"
        details = f"SentenceTransformer library imported successfully. Configured model: {settings.EMBEDDING_MODEL}"
    except ImportError as err:
        status = "unhealthy"
        details = f"SentenceTransformer library could not be imported: {str(err)}"

    return {
        "status": status,
        "details": details
    }


def check_faiss() -> Dict[str, Any]:
    """
    Checks FAISS library availability.
    """
    try:
        import faiss
        status = "healthy"
        details = "FAISS library imported successfully."
    except ImportError as err:
        status = "unhealthy"
        details = f"FAISS library could not be imported: {str(err)}"

    return {
        "status": status,
        "details": details
    }


def check_groq_configuration() -> Dict[str, Any]:
    """
    Checks Groq API key configuration state without calling network endpoints.
    """
    api_key = settings.GROQ_API_KEY
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")

    # Since Settings validator checks placeholder, we check it here too
    if api_key and api_key.strip() and api_key.strip() != "your_groq_api_key_here":
        status = "healthy"
        details = "Groq API key configured."
    else:
        status = "unhealthy"
        details = "GROQ_API_KEY is missing."

    return {
        "status": status,
        "details": details
    }


def check_backend_configuration() -> Dict[str, Any]:
    """
    Verifies that standard env configuration parameters are loaded.
    """
    # Simple check on project settings parameters
    if settings.PROJECT_NAME and settings.UPLOAD_DIRECTORY:
        status = "healthy"
        details = f"Core configurations validated. Env mode: {settings.ENV}"
    else:
        status = "unhealthy"
        details = "System configurations are incomplete."

    return {
        "status": status,
        "details": details
    }


def get_system_health() -> Dict[str, Any]:
    """
    Aggregates diagnostic checkpoints to retrieve overall health status dictionary.
    """
    uploads_dir = Path(__file__).resolve().parent.parent.parent.parent / settings.UPLOAD_DIRECTORY
    
    upload_dir_info = check_upload_directory()
    disk_permissions = check_disk_permissions()
    disk_space = check_disk_space()
    model_info = check_sentence_transformer()
    faiss_info = check_faiss()
    groq_info = check_groq_configuration()
    config_info = check_backend_configuration()

    # Determine overall status
    is_healthy = (
        upload_dir_info["status"] == "healthy" and
        disk_permissions["status"] == "healthy" and
        disk_space["status"] == "healthy" and
        model_info["status"] == "healthy" and
        faiss_info["status"] == "healthy" and
        groq_info["status"] == "healthy" and
        config_info["status"] == "healthy"
    )

    # Count total documents
    total_docs = 0
    if uploads_dir.exists():
        try:
            for child in uploads_dir.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    if (child / "original.pdf").exists():
                        total_docs += 1
        except Exception:
            pass

    uptime = time.time() - START_TIME

    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "uploads_directory": upload_dir_info,
        "write_permission": disk_permissions,
        "disk_usage": disk_space,
        "embedding_model": model_info,
        "faiss_library": faiss_info,
        "groq_service": groq_info,
        "backend_version": settings.VERSION,
        "uptime_seconds": round(uptime, 2),
        "total_documents": total_docs
    }
