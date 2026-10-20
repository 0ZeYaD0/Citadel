"""CLI entry-point: load a PDF, chunk it, embed it, and store it in ChromaDB.

Usage:
    python ingest_data.py path/to/textbook.pdf
"""

import sys
import logging
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from citadel.core.config import settings
from citadel.infrastructure.vectorstore.chroma_repository import ChromaRepository
from citadel.services.ingestion_service import IngestionService

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

_CHAPTER_HEADING = re.compile(
    r"^\s*(?:\d+\s+)?CHAPTER\s+(\d+)\.\s*([^\r\n]+)",
    re.IGNORECASE | re.MULTILINE,
)


def _add_chapter_metadata(pages: list[Document]) -> list[Document]:
    """Propagate each PDF chapter heading to its following pages."""
    annotated_pages: list[Document] = []
    chapter_number: int | None = None
    chapter_title: str | None = None

    for page in pages:
        match = _CHAPTER_HEADING.search(page.page_content)
        if match:
            chapter_number = int(match.group(1))
            chapter_title = match.group(2).strip().title()

        metadata = dict(page.metadata)
        if chapter_number is not None and chapter_title is not None:
            metadata.update(
                {
                    "chapter_number": chapter_number,
                    "chapter_title": chapter_title,
                }
            )
        annotated_pages.append(
            Document(page_content=page.page_content, metadata=metadata)
        )

    return annotated_pages


def main(pdf_path: str, reset: bool = False) -> None:
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
    chunks = splitter.split_documents(_add_chapter_metadata(pages))
    logger.info("Split into %d chunks.", len(chunks))

    # 3. Embed and store via the clean-architecture service
    repo = ChromaRepository()
    service = IngestionService(vector_repo=repo)
    if reset:
        logger.info("Resetting the configured Chroma collection before ingestion.")
        service.clear_store()
    stored = service.ingest(chunks)

    logger.info(
        "Done — %d chunks embedded and stored in '%s'.",
        stored,
        settings.chroma_persist_dir,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_data.py <path-to-pdf> [--reset]")
        sys.exit(1)
    main(sys.argv[1], reset="--reset" in sys.argv[2:])
