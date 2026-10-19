from typing import List
from google import genai
from google.genai import types
from citadel.domain.interfaces import IEmbedder


class GeminiEmbedder(IEmbedder):
    def __init__(self, api_key: str, model_name: str = "text-embedding-004"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def embed_query(self, query: str) -> List[float]:
        """Takes the user's question and converts it into an embedding vector."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        response = self.client.models.embed_content(
            model=self.model_name,
            contents=query.strip(),
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        return response.embeddings[0].values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Takes a list of PDF text chunks and embeds them (for the ingestion phase)."""
        if not texts:
            return []

        response = self.client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        return [emb.values for emb in response.embeddings]