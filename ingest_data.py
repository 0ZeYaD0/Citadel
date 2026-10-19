"""CLI entry-point: load a PDF, chunk it, embed it, and store it in ChromaDB.

Usage:
    python ingest_data.py path/to/textbook.pdf
"""

import sys
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from citadel.core.config import settings
from citadel.infrastructure.vectorstore.chroma_repository import ChromaRepository
from citadel.services.ingestion_service import IngestionService

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main(pdf_path: str) -> None:
    # 1. Load the PDF into LangChain Documents (one per page)
    logger.info("Loading PDF: %s", pdf_path)
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    logger.info("Loaded %d pages.", len(pages))

    # 2. Split pages into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(pages)
    logger.info("Split into %d chunks.", len(chunks))

    # 3. Embed and store via the clean-architecture service
    repo = ChromaRepository()
    service = IngestionService(vector_repo=repo)
    stored = service.ingest(chunks)

    logger.info(
        "Done — %d chunks embedded and stored in '%s'.",
        stored,
        settings.chroma_persist_dir,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_data.py <path-to-pdf>")
        sys.exit(1)
    main(sys.argv[1])
