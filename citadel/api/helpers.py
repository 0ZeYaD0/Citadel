"""Shared application helpers for pipeline lifecycle and ingestion jobs."""

from __future__ import annotations

import logging
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from citadel.core.config import settings
from citadel.core.concurrency import file_operations
from citadel.infrastructure.database.sqlite_repository import SQLiteRepository
from citadel.infrastructure.llm.groq_generator import GroqAnswerGenerator
from citadel.infrastructure.llm.local_embedder import LocalEmbedder
from citadel.infrastructure.loaders.pdf_chunker import TextbookDocumentProcessor
from citadel.infrastructure.vectorstore.chroma_repository import ChromaRepository
from citadel.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)


@dataclass
class RagComponents:
    """Runtime dependencies used by the API routes."""

    query_embedder: LocalEmbedder | None = None
    vector_repository: ChromaRepository | None = None
    answer_generator: GroqAnswerGenerator | None = None
    ingestion_service: IngestionService | None = None
    document_processor: TextbookDocumentProcessor | None = None
    db: SQLiteRepository | None = None

    def is_ready(self) -> bool:
        return all(
            (
                self.query_embedder,
                self.vector_repository,
                self.answer_generator,
                self.ingestion_service,
                self.document_processor,
                self.db,
            )
        )

    def pipeline_is_ready(self) -> bool:
        return all(
            (self.query_embedder, self.vector_repository, self.answer_generator)
        )


def load_rag_pipeline(components: RagComponents) -> RagComponents:
    """Initialize the embedding, retrieval, generation, and database services."""
    if components.is_ready():
        return components

    try:
        if components.db is None:
            components.db = SQLiteRepository(db_path=settings.database_path)
        components.query_embedder = LocalEmbedder()
        components.vector_repository = ChromaRepository(
            embedder=components.query_embedder,
            persist_directory=str(settings.chroma_persist_dir),
            collection_name=settings.chroma_collection_name,
        )
        components.answer_generator = GroqAnswerGenerator()
        components.ingestion_service = IngestionService(
            vector_repo=components.vector_repository,
            settings=settings,
        )
        components.document_processor = TextbookDocumentProcessor()
        logger.info(
            "RAG pipeline ready | model=%s | chunks=%d",
            settings.groq_model,
            components.vector_repository.document_count(),
        )
    except Exception:
        logger.exception("Failed to initialise the RAG pipeline")
        components.query_embedder = None
        components.vector_repository = None
        components.answer_generator = None
        components.ingestion_service = None
        components.document_processor = None

    return components


def health_status(components: RagComponents) -> dict[str, Any]:
    """Build the liveness response for the configured pipeline."""
    pipeline_ready = components.pipeline_is_ready()
    db_ready = components.db is not None
    chunk_count = (
        components.vector_repository.document_count()
        if components.vector_repository
        else 0
    )
    return {
        "status": "ok" if (pipeline_ready and db_ready) else "not_ready",
        "pipeline": "ready" if pipeline_ready else "not_initialized",
        "database": "ready" if db_ready else "not_initialized",
        "embedding_model": settings.local_embedding_model,
        "generation_model": settings.groq_model,
        "total_chunks": chunk_count,
    }


def run_ingestion_job(
    job_id: str,
    files_data: list[dict],
    components: RagComponents,
) -> None:
    """Process and embed uploaded files, updating job and document status."""
    db = components.db
    document_processor = components.document_processor
    ingestion_service = components.ingestion_service
    if db is None or document_processor is None or ingestion_service is None:
        return

    results: list[dict] = []
    db.update_job(
        job_id,
        status="running",
        stage="extracting",
        detail=f"Processing {len(files_data)} file(s)",
    )

    for file_info in files_data:
        original_name: str = file_info["name"]
        suffix: str = file_info["suffix"]
        content_bytes: bytes = file_info["content_bytes"]
        doc_id: str = file_info["doc_id"]
        db.update_job(
            job_id,
            status="running",
            stage="chunking",
            detail=f"Chunking {original_name}",
        )

        tmp_path: Path | None = None
        try:
            extraction_started = time.perf_counter()
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, dir=tempfile.gettempdir()
            ) as tmp:
                tmp_path = Path(tmp.name)
            file_operations.write_bytes(tmp_path, content_bytes)

            chunks = document_processor.process_single_file(
                tmp_path,
                source_name=original_name,
                document_id=doc_id,
            )
            extraction_ms = round((time.perf_counter() - extraction_started) * 1000, 1)
            logger.info("Extracted %s in %.1f ms.", original_name, extraction_ms)
            if not chunks:
                preview_text = "(No extractable text found in this document.)"
                db.update_document_status(
                    doc_id, status="ready", chunk_count=0, preview_text=preview_text
                )
                results.append(
                    _document_result(
                        doc_id, original_name, content_bytes, 0, preview_text
                    )
                )
                continue

            db.update_job(
                job_id,
                status="running",
                stage="embedding",
                detail=f"Embedding {len(chunks)} chunks from {original_name}",
            )
            embedding_started = time.perf_counter()
            stored = ingestion_service.ingest(chunks)
            logger.info(
                "Embedded %s chunks from %s in %.1f ms.",
                len(chunks),
                original_name,
                round((time.perf_counter() - embedding_started) * 1000, 1),
            )
            preview_text = "\n\n".join(chunk.page_content for chunk in chunks[:3])
            if len(chunks) > 3:
                preview_text += f"\n\n... and {len(chunks) - 3} more passages."

            db.update_document_status(
                doc_id,
                status="ready",
                chunk_count=stored,
                preview_text=preview_text,
            )
            results.append(
                _document_result(
                    doc_id, original_name, content_bytes, stored, preview_text
                )
            )
            logger.info(
                "Ingested %s → %d chunks (doc_id=%s)", original_name, stored, doc_id
            )
        except Exception as exc:
            logger.exception("Failed to ingest %s", original_name)
            error_msg = str(exc)
            db.update_document_status(doc_id, status="error", error_message=error_msg)
            results.append(
                {
                    "id": doc_id,
                    "name": original_name,
                    "size": format_size(len(content_bytes)),
                    "size_bytes": len(content_bytes),
                    "status": "error",
                    "error": error_msg,
                    "content": f"Processing failed: {exc}",
                    "active": False,
                }
            )
        finally:
            if tmp_path:
                file_operations.delete(tmp_path)

    db.update_job(
        job_id,
        status="done",
        stage="complete",
        detail=f"Processed {len(files_data)} file(s)",
        result={"sources": results},
    )


def extract_chapter_number(question: str) -> int | None:
    """Extract an explicit chapter number from a student question."""
    match = re.search(r"\bchapter\s+(\d+)\b", question, re.IGNORECASE)
    return int(match.group(1)) if match else None


def format_size(size_bytes: int) -> str:
    """Format a byte count for API responses."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1024**2:.1f} MB"


def _document_result(
    doc_id: str,
    name: str,
    content_bytes: bytes,
    chunk_count: int,
    preview_text: str,
) -> dict:
    return {
        "id": doc_id,
        "name": name,
        "size": format_size(len(content_bytes)),
        "size_bytes": len(content_bytes),
        "status": "ready",
        "chunk_count": chunk_count,
        "content": preview_text,
        "active": True,
    }
