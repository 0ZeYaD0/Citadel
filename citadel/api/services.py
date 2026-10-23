"""Runtime service container for API dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field

from citadel.api.helpers import RagComponents
from citadel.core.config import settings
from citadel.core.query_cache import QueryCache
from citadel.api.dependencies import (
    AnswerGenerator,
    Database,
    DocumentProcessor,
    Embedder,
    Ingestion,
    VectorRepository,
)


@dataclass
class ApplicationServices:
    """Own the runtime services shared by API routes."""

    query_embedder: Embedder | None = None
    vector_repository: VectorRepository | None = None
    answer_generator: AnswerGenerator | None = None
    ingestion_service: Ingestion | None = None
    document_processor: DocumentProcessor | None = None
    db: Database | None = None
    query_cache: QueryCache = field(
        default_factory=lambda: QueryCache(settings.query_cache_size)
    )

    def as_rag_components(self) -> RagComponents:
        return RagComponents(
            query_embedder=self.query_embedder,
            vector_repository=self.vector_repository,
            answer_generator=self.answer_generator,
            ingestion_service=self.ingestion_service,
            document_processor=self.document_processor,
            db=self.db,
        )

    def update_from(self, components: RagComponents) -> None:
        self.query_embedder = components.query_embedder
        self.vector_repository = components.vector_repository
        self.answer_generator = components.answer_generator
        self.ingestion_service = components.ingestion_service
        self.document_processor = components.document_processor
        self.db = components.db

    def pipeline_ready(self) -> bool:
        return all(
            (self.query_embedder, self.vector_repository, self.answer_generator)
        )

    def ready(self) -> bool:
        return all(
            (
                self.pipeline_ready(),
                self.ingestion_service,
                self.document_processor,
                self.db,
            )
        )
