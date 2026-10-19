"""Centralised application settings, loaded from .env or environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide configuration."""

    # Gemini
    gemini_api_key: str = ""

    # Embedding model
    embedding_model: str = "models/text-embedding-004"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "citadel_docs"

    # Chunking defaults (consumed by the chunker, defined here for central control)
    chunk_size: int = 1000
    chunk_overlap: int = 200

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton instance importable across the app
settings = Settings()
