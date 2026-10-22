"""Abstractions owned by the document-ingestion domain."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List

from langchain_core.documents import Document


class BaseDocumentProcessor(ABC):
	"""Contract for loading source files into domain document chunks."""

	@abstractmethod
	def load_and_split(self, directory_path: Path) -> List[Document]:
		"""Load supported files into LangChain document chunks."""
		raise NotImplementedError

	@abstractmethod
	def process_single_file(
		self,
		file_path: Path,
		source_name: str | None = None,
		document_id: str | None = None,
	) -> List[Document]:
		"""Load one supported file into LangChain document chunks."""
		raise NotImplementedError
"""Abstract interfaces (ports) for the Citadel RAG system."""

from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.documents import Document

from citadel.domain.models import DocumentChunk, RAGResponse


class EmbeddingProvider(ABC):
    """Port for converting text into embedding vectors."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of text strings."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Return the embedding vector for a single query string."""


class VectorRepository(ABC):
    """Port for persisting and querying document embeddings."""

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> list[str]:
        """Embed and store documents. Return the generated IDs."""

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Query the store with an embedding and optional metadata filter."""

    @abstractmethod
    def document_count(self) -> int:
        """Return the total number of documents in the store."""

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the entire collection (useful for re-ingestion)."""

    @abstractmethod
    def delete_documents(self, document_id: str) -> int:
        """Delete all chunks belonging to a document and return their count."""
        raise NotImplementedError


# Compatibility names retained for callers from the earlier search branch.
IEmbedder = EmbeddingProvider
IVectorRepository = VectorRepository


class BaseAnswerGenerator(ABC):
    """Port for generating answers grounded in retrieved textbook chunks."""

    @abstractmethod
    def generate_grounded_answer(
        self,
        query: str,
        context_chunks: List[DocumentChunk],
    ) -> RAGResponse:
        """Generate a strictly grounded answer from retrieved chunks."""
        raise NotImplementedError
