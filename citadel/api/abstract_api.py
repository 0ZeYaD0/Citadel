"""Application API abstraction and FastAPI implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from typing import Any

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


class AbstractAPI(ABC):
    """Contract for the web application used by the route module."""

    @property
    @abstractmethod
    def application(self) -> FastAPI:
        """Return the configured web application instance."""


class FastAPIApplication(AbstractAPI):
    """Create and configure the framework application in one place."""

    def __init__(
        self,
        *,
        title: str,
        description: str,
        version: str,
        allow_origins: list[str] | None = None,
        lifespan: AbstractAsyncContextManager[Any] | None = None,
    ) -> None:
        self._application = FastAPI(
            title=title,
            description=description,
            version=version,
            lifespan=lifespan,
        )
        self._application.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins or ["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @property
    def application(self) -> FastAPI:
        """Return the configured FastAPI application."""
        return self._application


__all__ = [
    "AbstractAPI",
    "BackgroundTasks",
    "FastAPIApplication",
    "File",
    "FileResponse",
    "Form",
    "HTTPException",
    "Query",
    "StaticFiles",
    "UploadFile",
]
