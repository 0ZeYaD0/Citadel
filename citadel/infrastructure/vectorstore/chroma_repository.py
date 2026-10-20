"""ChromaDB-backed vector repository using langchain-chroma."""

import logging
from typing import Any
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document

from citadel.core.config import settings
from citadel.domain.interfaces import VectorRepository
from citadel.domain.models import DocumentChunk
from citadel.infrastructure.llm.local_embedder import LocalEmbedder

logger = logging.getLogger(__name__)


class ChromaRepository(VectorRepository):
    """Persists and queries document embeddings in a local ChromaDB store.

    The repository owns its ``LocalEmbedder`` instance so that embedding
    model and vector store always stay in sync.
    """

    def __init__(
        self,
        embedder: LocalEmbedder | None = None,
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._embedder = embedder or LocalEmbedder()
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

        ids = [
            str(document.metadata.get("chunk_id") or uuid4())
            for document in documents
        ]
        existing = self._vectorstore._collection.get(include=["metadatas"])
        existing_ids = set(existing.get("ids", []))
        existing_chunk_ids = {
            metadata.get("chunk_id")
            for metadata in existing.get("metadatas", [])
            if metadata and metadata.get("chunk_id")
        }
        pending_documents = [
            document for document, document_id in zip(documents, ids)
            if document_id not in existing_ids
            and document.metadata.get("chunk_id") not in existing_chunk_ids
        ]
        pending_ids = [
            document_id
            for document, document_id in zip(documents, ids)
            if document_id not in existing_ids
            and document.metadata.get("chunk_id") not in existing_chunk_ids
        ]
        if pending_documents:
            self._vectorstore.add_documents(pending_documents, ids=pending_ids)
        logger.info(
            "Stored %d documents (total now: %d).",
            len(pending_documents),
            self.document_count(),
        )
        return ids

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> list[str]:
        """Store domain chunks with caller-provided embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length.")
        if not chunks:
            return []

        ids = [chunk.id for chunk in chunks]
        self._vectorstore._collection.add(
            ids=ids,
            documents=[chunk.content for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            embeddings=embeddings,
        )
        return ids

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Query ChromaDB using an embedding and optional metadata filter."""
        if not query_embedding:
            raise ValueError("query_embedding cannot be empty.")

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if metadata_filter:
            query_kwargs["where"] = metadata_filter

        results = self._vectorstore._collection.query(**query_kwargs)

        retrieved_chunks: list[DocumentChunk] = []

        # Chroma wraps single-query results in an outer list
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, doc_text, metadata, distance in zip(ids, documents, metadatas, distances):
            # Convert cosine distance to similarity score (1.0 = identical, 0.0 = orthogonal)
            similarity_score = 1.0 - distance if distance is not None else None
            retrieved_chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    content=doc_text,
                    metadata=metadata or {},
                    score=similarity_score
                )
            )

        return retrieved_chunks

    def document_count(self) -> int:
        """Return the number of documents in the collection."""
        return self._vectorstore._collection.count()

    def delete_collection(self) -> None:
        """Delete all documents while keeping the repository usable."""
        self._vectorstore.delete_collection()
        self._vectorstore = Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embedder.langchain_embeddings,
            persist_directory=self._persist_dir,
        )
