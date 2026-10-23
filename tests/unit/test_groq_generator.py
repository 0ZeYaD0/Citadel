from citadel.domain.models import DocumentChunk
from citadel.infrastructure.llm.groq_generator import GroqAnswerGenerator


def test_similarity_filter_keeps_best_chunks_for_broad_queries():
    generator = GroqAnswerGenerator.__new__(GroqAnswerGenerator)
    generator._min_similarity = 0.8
    chunks = [
        DocumentChunk(id="one", content="one", score=0.4),
        DocumentChunk(id="two", content="two", score=0.7),
        DocumentChunk(id="three", content="three", score=0.6),
        DocumentChunk(id="four", content="four", score=0.5),
    ]

    filtered = generator._filter_by_similarity(chunks)

    assert [chunk.id for chunk in filtered] == ["two", "three", "four"]
