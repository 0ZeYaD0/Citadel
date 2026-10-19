"""Service that orchestrates document ingestion into the vector store."""

import logging
import time

from langchain_core.documents import Document

from citadel.domain.interfaces import VectorRepository

logger = logging.getLogger(__name__)


class IngestionService:
    """Accepts pre-chunked LangChain Documents and stores them.

    Parameters
    ----------
    vector_repo : VectorRepository
        The vector store backend (e.g. ChromaRepository).
    batch_size : int
        Number of documents to embed and store per batch call.
    """

    def __init__(
        self,
        vector_repo: VectorRepository,
        batch_size: int = 50,
    ) -> None:
        self._repo = vector_repo
        self._batch_size = batch_size

    def ingest(self, documents: list[Document]) -> int:
        """Embed and store *documents* in batches.

        Returns the total number of documents stored.
        """
        if not documents:
            logger.warning("ingest() called with no documents.")
            return 0

        total_stored = 0
        for i in range(0, len(documents), self._batch_size):
            batch = documents[i : i + self._batch_size]
            ids = self._store_batch_with_retry(batch)
            total_stored += len(ids)
            logger.info(
                "Batch %d–%d stored (%d / %d total).",
                i,
                i + len(batch),
                total_stored,
                len(documents),
            )

        logger.info(
            "Ingestion complete: %d documents stored.",
            total_stored,
        )
        return total_stored

    def _store_batch_with_retry(self, batch: list[Document]) -> list[str]:
        """Retry transient provider throttling without duplicating batches."""
        for attempt in range(4):
            try:
                return self._repo.add_documents(batch)
            except Exception as exc:
                message = str(exc).upper()
                is_rate_limited = "429" in message or "RESOURCE_EXHAUSTED" in message
                if not is_rate_limited or attempt == 3:
                    raise
                delay = 5 * (attempt + 1)
                logger.warning(
                    "Embedding provider rate-limited the batch; retrying in %d seconds.",
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError("Unreachable retry state.")

    def clear_store(self) -> None:
        """Delete all documents from the vector store."""
        self._repo.delete_collection()
        logger.info("Vector store cleared.")

    @property
    def document_count(self) -> int:
        """Return the current number of stored documents."""
        return self._repo.document_count()
