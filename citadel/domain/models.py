from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a PDF stored/retrieved from the vector DB."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None