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
