import pytest
from unittest.mock import MagicMock
from citadel.domain.models import DocumentChunk
from citadel.services.rag_service import RAGService
from citadel.infrastructure.vectorstore.chroma_repository import ChromaRepository


def test_retrieve_fetches_top_k_chunks():
    """Unit test: verifies RAGService embeds the question and queries the vector repo."""
    mock_embedder = MagicMock()
    mock_vector_repo = MagicMock()

    fake_vector = [0.1, 0.2, 0.3]
    expected_chunks = [
        DocumentChunk(id="chunk_1", content="Relevant PDF text 1", metadata={"page": 1}, score=0.92),
        DocumentChunk(id="chunk_2", content="Relevant PDF text 2", metadata={"page": 2}, score=0.85),
    ]

    mock_embedder.embed_query.return_value = fake_vector
    mock_vector_repo.similarity_search.return_value = expected_chunks

    service = RAGService(embedder=mock_embedder, vector_repo=mock_vector_repo)

    results = service.retrieve(question="What is the main topic?", top_k=2)

    mock_embedder.embed_query.assert_called_once_with("What is the main topic?")
    mock_vector_repo.similarity_search.assert_called_once_with(query_embedding=fake_vector, top_k=2)
    assert len(results) == 2
    assert results[0].id == "chunk_1"


def test_retrieve_raises_on_empty_question():
    """Unit test: verifies empty or whitespace-only questions raise a ValueError."""
    service = RAGService(embedder=MagicMock(), vector_repo=MagicMock())
    with pytest.raises(ValueError, match="cannot be empty"):
        service.retrieve("   ")


def test_chroma_similarity_search_real_db(tmp_path):
    """Integration test: verifies ChromaRepository stores vectors and returns top-k matches."""
    repo = ChromaRepository(persist_directory=str(tmp_path), collection_name="test_collection")

    chunks = [
        DocumentChunk(id="c1", content="Python is great for backend and AI.", metadata={"page": 1}),
        DocumentChunk(id="c2", content="JavaScript is used for frontend UI.", metadata={"page": 2}),
        DocumentChunk(id="c3", content="ChromaDB stores vector embeddings.", metadata={"page": 3}),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    repo.add_chunks(chunks, embeddings)

    results = repo.similarity_search(query_embedding=[0.0, 0.1, 0.99], top_k=2)

    assert len(results) == 2
    assert results[0].id == "c3"
    assert results[0].content == "ChromaDB stores vector embeddings."
    assert results[0].score > results[1].score