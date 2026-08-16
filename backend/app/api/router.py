"""
API Main Router Aggregator.

Combines all sub-routers and endpoints into a unified API router instance.
"""

from fastapi import APIRouter
from app.api.routes.system import router as system_router
from app.api.routes.upload import router as upload_router
from app.api.routes.processing import router as processing_router
from app.api.routes.chunking import router as chunking_router
from app.api.routes.embedding import router as embedding_router
from app.api.routes.indexing import router as indexing_router
from app.api.routes.retrieval import router as retrieval_router
from app.api.routes.chat import router as chat_router
from app.api.routes.management import router as management_router
from app.api.routes.maintenance import router as maintenance_router

api_router = APIRouter()

# Register health and system telemetry routes
api_router.include_router(system_router, tags=["System Status"])

# Register PDF upload routes
api_router.include_router(upload_router, tags=["PDF Ingestion"])

# Register PDF processing engine routes
api_router.include_router(processing_router, tags=["PDF Processing Engine"])

# Register Smart Text Chunking engine routes
api_router.include_router(chunking_router, tags=["Text Chunking Engine"])

# Register Vector Embedding engine routes
api_router.include_router(embedding_router, tags=["Vector Embedding Engine"])

# Register Vector Indexing engine routes
api_router.include_router(indexing_router, tags=["Vector Indexing Engine"])

# Register Semantic Retrieval engine routes
api_router.include_router(retrieval_router, tags=["Semantic Retrieval Engine"])

# Register AI Chat (RAG) Engine routes
api_router.include_router(chat_router, tags=["AI Chat (RAG) Engine"])

# Register Document Management routes
api_router.include_router(management_router, tags=["Document Management Engine"])

# Register System Maintenance routes
api_router.include_router(maintenance_router, tags=["System Maintenance Engine"])

