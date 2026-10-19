"""Service that orchestrates document ingestion into the vector store."""

import logging
import random
import re
import time

from langchain_core.documents import Document

from citadel.domain.interfaces import VectorRepository

logger = logging.getLogger(__name__)

_MAX_RATE_LIMIT_RETRIES = 8
_MIN_RATE_LIMIT_WAIT_SECONDS = 60.0
_RETRY_DELAY_PATTERN = re.compile(r"retry(?:ing)? in\s+([0-9]+(?:\.[0-9]+)?)s")


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
        """Retry rate-limited batches across the provider quota window.

        Gemini's free embedding quota is measured over a rolling minute. A
        short fixed retry sequence can exhaust all retries before that window
        resets, so the provider's suggested delay is honored and never reduced
        below one minute for quota exhaustion.
        """
        for attempt in range(_MAX_RATE_LIMIT_RETRIES):
            try:
                return self._repo.add_documents(batch)
            except Exception as exc:
                message = str(exc).upper()
                is_rate_limited = "429" in message or "RESOURCE_EXHAUSTED" in message
                if not is_rate_limited or attempt == _MAX_RATE_LIMIT_RETRIES - 1:
                    raise
                delay = self._retry_delay_seconds(str(exc), attempt)
                logger.warning(
                    "Embedding provider rate-limited the batch; retrying in %.1f "
                    "seconds (attempt %d/%d).",
                    delay,
                    attempt + 1,
                    _MAX_RATE_LIMIT_RETRIES - 1,
                )
                time.sleep(delay)

        raise RuntimeError("Unreachable retry state.")

    @staticmethod
    def _retry_delay_seconds(error_message: str, attempt: int) -> float:
        """Calculate a quota-aware retry delay with small jitter."""
        match = _RETRY_DELAY_PATTERN.search(error_message)
        provider_delay = float(match.group(1)) if match else 0.0
        exponential_delay = 2**attempt * 5.0
        return max(
            _MIN_RATE_LIMIT_WAIT_SECONDS,
            provider_delay,
            exponential_delay,
        ) + random.uniform(0.0, 2.0)

    def clear_store(self) -> None:
        """Delete all documents from the vector store."""
        self._repo.delete_collection()
        logger.info("Vector store cleared.")

    @property
    def document_count(self) -> int:
        """Return the current number of stored documents."""
        return self._repo.document_count()
