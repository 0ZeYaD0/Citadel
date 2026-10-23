"""Local sentence-transformer embedding provider for Citadel."""

import os

from langchain_huggingface import HuggingFaceEmbeddings

from citadel.core.config import Settings, get_settings
from citadel.domain.interfaces import EmbeddingProvider


class LocalEmbedder(EmbeddingProvider):
    """Generate textbook and query embeddings locally with MiniLM."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the local embedding model on CPU."""
        self._settings = settings or get_settings()
        self._model_name = self._settings.local_embedding_model
        if self._settings.hf_home is not None:
            os.environ.setdefault("HF_HOME", str(self._settings.hf_home))
        if self._settings.hf_token:
            os.environ.setdefault("HF_TOKEN", self._settings.hf_token)

        model_kwargs = {"device": "cpu"}
        if self._settings.hf_token:
            model_kwargs["token"] = self._settings.hf_token
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self._model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts locally."""
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed one query locally."""
        return self._embeddings.embed_query(text)

    @property
    def langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """Expose the LangChain embedding object for Chroma."""
        return self._embeddings