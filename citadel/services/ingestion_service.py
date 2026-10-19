"""Service that orchestrates document ingestion into the vector store."""

import logging

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
            ids = self._repo.add_documents(batch)
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

    def clear_store(self) -> None:
        """Delete all documents from the vector store."""
        self._repo.delete_collection()
        logger.info("Vector store cleared.")

    @property
    def document_count(self) -> int:
        """Return the current number of stored documents."""
        return self._repo.document_count()
