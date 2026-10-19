"""Domain models for the Citadel RAG system."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class TextChunk:
    """A piece of text extracted from a source document.

    Attributes:
        content:  The textual content of the chunk.
        metadata: Arbitrary key-value metadata (source file, page number, etc.).
    """

    content: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    """A retrieved chunk returned from a similarity search.

    Attributes:
        id:       The unique identifier assigned by the vector store.
        content:  The textual content of the chunk.
        metadata: Arbitrary key-value metadata (source file, page number, etc.).
        score:    Similarity score (1.0 = identical, 0.0 = orthogonal).
    """

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
