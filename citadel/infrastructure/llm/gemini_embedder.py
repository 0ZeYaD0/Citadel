"""Gemini embedding provider using langchain-google-genai."""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from citadel.core.config import settings
from citadel.domain.interfaces import EmbeddingProvider


class GeminiEmbedder(EmbeddingProvider):
    """Generates embedding vectors via Google's Gemini API.

    Uses the ``gemini-embedding-001`` model by default (configurable in Settings).
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model or settings.embedding_model
        self._api_key = api_key or settings.gemini_api_key

        if not self._api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file or pass it explicitly."
            )

        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=self._model,
            google_api_key=self._api_key,
        )

    # ── EmbeddingProvider interface ────────────────────────────────────

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts."""
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._embeddings.embed_query(text)

    @property
    def langchain_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Expose the underlying LangChain embeddings object.

        Needed by ChromaRepository which passes it to ``langchain_chroma.Chroma``.
        """
        return self._embeddings
