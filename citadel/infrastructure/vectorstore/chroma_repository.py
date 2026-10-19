"""ChromaDB-backed vector repository using langchain-chroma."""

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document

from citadel.core.config import settings
from citadel.domain.interfaces import VectorRepository
from citadel.domain.models import DocumentChunk
from citadel.infrastructure.llm.gemini_embedder import GeminiEmbedder

logger = logging.getLogger(__name__)


class ChromaRepository(VectorRepository):
    """Persists and queries document embeddings in a local ChromaDB store.

    The repository owns its ``GeminiEmbedder`` instance so that embedding
    model and vector store always stay in sync.
    """

    def __init__(
        self,
        embedder: GeminiEmbedder | None = None,
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._embedder = embedder or GeminiEmbedder()
        self._persist_dir = persist_directory or settings.chroma_persist_dir
        self._collection_name = collection_name or settings.chroma_collection_name

        self._vectorstore = Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embedder.langchain_embeddings,
            persist_directory=self._persist_dir,
        )
        logger.info(
            "ChromaRepository initialised  "
            "(collection=%s, dir=%s, docs=%d)",
            self._collection_name,
            self._persist_dir,
            self.document_count(),
        )

    # ── VectorRepository interface ─────────────────────────────────────

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Embed and persist *documents*. Returns the generated IDs."""
        if not documents:
            logger.warning("add_documents called with an empty list.")
            return []

        ids = self._vectorstore.add_documents(documents)
        logger.info(
            "Stored %d documents (total now: %d).",
            len(ids),
            self.document_count(),
        )
        return ids

    def similarity_search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[DocumentChunk]:
        """Query ChromaDB using a pre-computed embedding vector and return top-k chunks."""
        if not query_embedding:
            raise ValueError("query_embedding cannot be empty.")

        results = self._vectorstore._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved_chunks: list[DocumentChunk] = []

        # Chroma wraps single-query results in an outer list
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, doc_text, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            # Convert cosine distance to similarity score
            # (1.0 = identical, 0.0 = orthogonal)
            similarity_score = 1.0 - distance if distance is not None else None
            retrieved_chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    content=doc_text,
                    metadata=metadata or {},
                    score=similarity_score,
                )
            )

        return retrieved_chunks

    def document_count(self) -> int:
        """Return the total number of stored documents."""
        return self._vectorstore._collection.count()

    def delete_collection(self) -> None:
        """Delete the entire Chroma collection."""
        self._vectorstore.delete_collection()
        logger.info("Deleted collection '%s'.", self._collection_name)

    # ── Convenience ────────────────────────────────────────────────────

    def as_retriever(self, **kwargs):
        """Return a LangChain retriever backed by this store.

        Useful for plugging into LangChain retrieval chains downstream.
        """
        return self._vectorstore.as_retriever(**kwargs)
