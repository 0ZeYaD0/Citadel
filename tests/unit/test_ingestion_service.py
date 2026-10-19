"""Unit tests for IngestionService."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from citadel.services.ingestion_service import IngestionService


@pytest.fixture
def mock_repo():
    """Return a mock VectorRepository."""
    repo = MagicMock()
    repo.document_count.return_value = 0
    return repo


@pytest.fixture
def service(mock_repo):
    """Return an IngestionService wired to the mock repo."""
    return IngestionService(vector_repo=mock_repo, batch_size=3)


# ── Tests ──────────────────────────────────────────────────────────────


def test_ingest_empty_list_returns_zero(service, mock_repo):
    """Calling ingest with no documents should return 0 and never call the repo."""
    result = service.ingest([])
    assert result == 0
    mock_repo.add_documents.assert_not_called()


def test_ingest_single_batch(service, mock_repo):
    """When documents fit in one batch, add_documents is called exactly once."""
    docs = [Document(page_content=f"chunk {i}") for i in range(3)]
    mock_repo.add_documents.return_value = ["id0", "id1", "id2"]

    result = service.ingest(docs)

    assert result == 3
    mock_repo.add_documents.assert_called_once_with(docs)


def test_ingest_multiple_batches(service, mock_repo):
    """Documents exceeding batch_size should be split across calls."""
    docs = [Document(page_content=f"chunk {i}") for i in range(7)]
    # batch_size=3 → batches of 3, 3, 1
    mock_repo.add_documents.side_effect = [
        ["id0", "id1", "id2"],
        ["id3", "id4", "id5"],
        ["id6"],
    ]

    result = service.ingest(docs)

    assert result == 7
    assert mock_repo.add_documents.call_count == 3


def test_clear_store_delegates_to_repo(service, mock_repo):
    """clear_store should call delete_collection on the repo."""
    service.clear_store()
    mock_repo.delete_collection.assert_called_once()


def test_document_count_delegates_to_repo(service, mock_repo):
    """document_count property should delegate to the repo."""
    mock_repo.document_count.return_value = 42
    assert service.document_count == 42
