"""Centralized, parameterized API errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from citadel.api.abstract_api import HTTPException
from citadel.core.exceptions import EmptyQuestionError, GenerationError


@dataclass(frozen=True)
class ErrorDefinition:
    """HTTP status and message template for one API error."""

    status_code: int
    detail: str


class ApiError(HTTPException):
    """HTTP error carrying the stable internal name of an API failure."""

    def __init__(self, *, name: str, status_code: int, detail: str) -> None:
        super().__init__(status_code=status_code, detail=detail)
        self.name = name


class ApiErrors:
    """Dynamic catalog for all errors exposed by the API."""

    _definitions = {
        "bad_request": ErrorDefinition(400, "{detail}"),
        "bad_gateway": ErrorDefinition(502, "{detail}"),
        "database_not_ready": ErrorDefinition(503, "Database not ready."),
        "pipeline_not_ready": ErrorDefinition(503, "RAG pipeline is not ready."),
        "pipeline_not_ready_retry": ErrorDefinition(
            503, "RAG pipeline is not ready. Please try again shortly."
        ),
        "pipeline_not_ready_ingestion": ErrorDefinition(
            503,
            "RAG pipeline is not ready. Check that ingestion completed successfully.",
        ),
        "notebook_not_found": ErrorDefinition(404, "Notebook not found."),
        "session_not_found": ErrorDefinition(404, "Session not found."),
        "note_not_found": ErrorDefinition(404, "Note not found."),
        "job_not_found": ErrorDefinition(404, "Job {job_id} not found."),
        "document_delete_failed": ErrorDefinition(
            502, "The document could not be deleted."
        ),
        "delete_all_failed": ErrorDefinition(502, "Could not delete all documents."),
        "rag_request_failed": ErrorDefinition(
            502, "The RAG request could not be completed."
        ),
        "study_tool_failed": ErrorDefinition(
            502, "Study tool generation failed: {detail}"
        ),
        "prompt_configuration_failed": ErrorDefinition(
            500, "Study tool prompts are not configured correctly."
        ),
    }

    @classmethod
    def create(cls, name: str, **values: Any) -> ApiError:
        """Create an HTTP exception from a named template."""
        try:
            definition = cls._definitions[name]
        except KeyError as exc:
            raise ValueError(f"Unknown API error: {name}") from exc
        return ApiError(
            name=name,
            status_code=definition.status_code,
            detail=definition.detail.format(**values),
        )

    @classmethod
    def from_exception(cls, exc: Exception, fallback: str) -> ApiError:
        """Translate expected application exceptions into API errors."""
        if isinstance(exc, ValueError):
            return cls.create("bad_request", detail=str(exc))
        if isinstance(exc, EmptyQuestionError):
            return cls.create("bad_request", detail=str(exc))
        if isinstance(exc, GenerationError):
            return cls.create("bad_gateway", detail=str(exc))
        return cls.create(fallback)

    @classmethod
    def bad_request(cls, detail: str) -> ApiError:
        return cls.create("bad_request", detail=detail)

    @classmethod
    def job_not_found(cls, job_id: str) -> ApiError:
        return cls.create("job_not_found", job_id=job_id)

    @classmethod
    def unknown_tool(cls, tool: str, choices: str) -> ApiError:
        return cls.create(
            "bad_request",
            detail=f"Unknown tool '{tool}'. Choose from: {choices}",
        )

    @classmethod
    def study_tool_failed(cls, detail: str) -> ApiError:
        return cls.create("study_tool_failed", detail=detail)

    @classmethod
    def bad_gateway(cls, detail: str) -> ApiError:
        return cls.create("bad_gateway", detail=detail)
