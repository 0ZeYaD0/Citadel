from abc import ABC, abstractmethod
from typing import List
from citadel.domain.models import DocumentChunk


class IEmbedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Used by Ingestion (your friend) to embed PDF chunks."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Used by Retriever (you) to embed the user's question."""
        pass


class IVectorRepository(ABC):
    @abstractmethod
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """Used by Ingestion (your friend) to save chunks into ChromaDB."""
        pass

    @abstractmethod
    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[DocumentChunk]:
        """Used by Retriever (you) to fetch the top-k closest chunks."""
        pass