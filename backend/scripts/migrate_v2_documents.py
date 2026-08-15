"""
Batch Admin Migration Script for DocMind AI RAG V2.

Scans all uploaded document directories in backend/uploads and upgrades pre-fix V1
documents (pipeline_version < 2 or missing) to V2 by executing forcing re-extraction,
chunking, embedding generation, and indexing.
"""

import sys
import json
import asyncio
import logging
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.pdf.processing_service import PDFProcessingService
from app.services.pdf.chunking_service import ChunkingService
from app.services.pdf.embedding_service import EmbeddingService
from app.services.pdf.indexing_service import IndexingService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v2_migration")


async def migrate_all_documents():
    uploads_dir = backend_dir / "uploads"
    if not uploads_dir.exists():
        logger.warning("No uploads directory found at '%s'. Nothing to migrate.", uploads_dir)
        return

    doc_dirs = [d for d in uploads_dir.iterdir() if d.is_dir()]
    logger.info("Found %d document directories in '%s'. Scanning for V1 metadata...", len(doc_dirs), uploads_dir)

    migrated_count = 0
    skipped_count = 0
    failed_count = 0

    proc_svc = PDFProcessingService()
    chunk_svc = ChunkingService()
    emb_svc = EmbeddingService()
    idx_svc = IndexingService()

    for doc_dir in doc_dirs:
        doc_id = doc_dir.name
        meta_path = doc_dir / "metadata.json"

        if not meta_path.exists():
            logger.warning("Document '%s' missing metadata.json. Skipping.", doc_id)
            skipped_count += 1
            continue

        try:
            with open(meta_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
            version = meta.get("pipeline_version", 1)

            if version >= 2:
                logger.info("Document '%s' is already V2 (version=%s). Skipping.", doc_id, version)
                skipped_count += 1
                continue

            logger.info("Upgrading document '%s' from V%s to V2...", doc_id, version)
            
            # Execute force re-ingestion
            await proc_svc.process_pdf(doc_id, force=True)
            await chunk_svc.chunk_document(doc_id, force=True)
            await emb_svc.generate_document_embeddings(doc_id, force=True)
            await idx_svc.generate_document_index(doc_id, force=True)

            logger.info("Successfully migrated document '%s' to V2!", doc_id)
            migrated_count += 1

        except Exception as exc:
            logger.error("Failed to migrate document '%s': %s", doc_id, str(exc))
            failed_count += 1

    logger.info("==========================================")
    logger.info("MIGRATION COMPLETE SUMMARY:")
    logger.info("  Migrated: %d", migrated_count)
    logger.info("  Skipped:  %d", skipped_count)
    logger.info("  Failed:   %d", failed_count)
    logger.info("==========================================")


if __name__ == "__main__":
    asyncio.run(migrate_all_documents())
