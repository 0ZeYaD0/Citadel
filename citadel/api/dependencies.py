"""Application dependency contracts exposed to API modules."""

from citadel.core.config import settings
from citadel.infrastructure.database.sqlite_repository import SQLiteRepository
from citadel.infrastructure.llm.groq_generator import GroqAnswerGenerator
from citadel.infrastructure.llm.local_embedder import LocalEmbedder
from citadel.infrastructure.loaders.pdf_chunker import TextbookDocumentProcessor
from citadel.infrastructure.vectorstore.chroma_repository import ChromaRepository
from citadel.services.ingestion_service import IngestionService

AnswerGenerator = GroqAnswerGenerator
Database = SQLiteRepository
DocumentProcessor = TextbookDocumentProcessor
Embedder = LocalEmbedder
Ingestion = IngestionService
VectorRepository = ChromaRepository

__all__ = [
    "AnswerGenerator",
    "Database",
    "DocumentProcessor",
    "Embedder",
    "Ingestion",
    "VectorRepository",
    "settings",
]