"""Domain models for textbook ingestion."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ChunkMetadata(BaseModel):
	"""Metadata associated with one semantic document chunk."""

	model_config = ConfigDict(frozen=True, extra="forbid")

	source_file: str = Field(min_length=1)
	page_number: Optional[int] = Field(default=None, ge=1)
	chunk_index: int = Field(ge=0)
	total_char_length: int = Field(ge=1)


class DocumentChunk(BaseModel):
	"""A chunk of source content ready for downstream retrieval."""

	model_config = ConfigDict(frozen=True, extra="forbid")

	chunk_id: str = Field(min_length=1)
	content: str = Field(min_length=1)
	metadata: ChunkMetadata
"""Domain models for the Citadel RAG system."""

from dataclasses import dataclass, field
from typing import Optional


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
    metadata: dict = field(default_factory=dict)
    score: Optional[float] = None
