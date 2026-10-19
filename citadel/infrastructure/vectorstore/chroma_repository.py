from typing import List
import chromadb
from citadel.domain.interfaces import IVectorRepository
from citadel.domain.models import DocumentChunk


class ChromaRepository(IVectorRepository):
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "citadel_docs"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        # Using cosine similarity space so distances are normalized
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """Your friend's ingestion service will call this to store PDF chunks."""
        if not chunks:
            return
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            embeddings=embeddings,
            metadatas=[chunk.metadata for chunk in chunks]
        )

    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[DocumentChunk]:
        """Your part: Queries ChromaDB using the embedded question vector and returns top-k chunks."""
        if not query_embedding:
            raise ValueError("query_embedding cannot be empty.")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks: List[DocumentChunk] = []

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