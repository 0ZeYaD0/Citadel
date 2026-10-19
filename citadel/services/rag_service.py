from typing import List
from citadel.domain.interfaces import IEmbedder, IVectorRepository
from citadel.domain.models import DocumentChunk


class RAGService:
    def __init__(self, embedder: IEmbedder, vector_repo: IVectorRepository):
        self.embedder = embedder
        self.vector_repo = vector_repo

    def retrieve(self, question: str, top_k: int = 5) -> List[DocumentChunk]:
        """
        Takes a user's question, embeds it into a vector, and fetches
        the top-k most relevant chunks from the vector database.
        """
        cleaned_question = question.strip() if question else ""
        if not cleaned_question:
            raise ValueError("User question cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        # 1. Embed the user's question
        query_vector = self.embedder.embed_query(cleaned_question)

        # 2. Fetch the top-k chunks from ChromaDB
        top_chunks = self.vector_repo.similarity_search(
            query_embedding=query_vector,
            top_k=top_k
        )

        return top_chunks