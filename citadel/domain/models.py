"""Domain models for the Citadel RAG system."""

from dataclasses import dataclass, field
from typing import Any, List, Optional


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


@dataclass(frozen=True)
class SourceCitation:
    """A source reference supporting a generated answer."""

    source_file: str
    page_number: Optional[int]
    chunk_id: str
    excerpt: str


@dataclass(frozen=True)
class RAGResponse:
    """A grounded answer and the source passages used to produce it."""

    question: str
    answer: str
    citations: List[SourceCitation]
    is_grounded: bool
    model_used: str
    latency_ms: Optional[float] = None
    retrieved_chunk_count: int = 0
    evidence_found: bool = True


@dataclass
class DocumentRecord:
    """A document entry in the Citadel registry (from SQLite)."""

    id: str
    filename: str
    file_type: str
    upload_date: str
    file_size_bytes: int
    chunk_count: int
    status: str  # 'processing' | 'ready' | 'error'
    error_message: Optional[str] = None
    preview_text: Optional[str] = None

    def to_api_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable dict for the frontend."""
        return {
            "id": self.id,
            "name": self.filename,
            "file_type": self.file_type,
            "upload_date": self.upload_date,
            "size": _format_size(self.file_size_bytes),
            "size_bytes": self.file_size_bytes,
            "chunk_count": self.chunk_count,
            "status": self.status,
            "error": self.error_message,
            "content": self.preview_text or "",
            "active": self.status == "ready",
        }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1024 ** 2:.1f} MB"
