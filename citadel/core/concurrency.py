"""Shared concurrency and blocking file-operation abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, TypeVar

from citadel.core.config import get_settings

T = TypeVar("T")


class FileOperations(ABC):
    """Contract for file operations that may block the caller."""

    @abstractmethod
    def read_text(self, path: Path, *, encoding: str = "utf-8") -> str:
        """Read text from a file."""

    @abstractmethod
    def write_bytes(self, path: Path, content: bytes) -> None:
        """Write bytes to a file."""

    @abstractmethod
    def delete(self, path: Path) -> None:
        """Delete a file when it exists."""


class DirectFileOperations(FileOperations):
    """Perform small file operations without thread-pool handoff overhead."""

    def read_text(self, path: Path, *, encoding: str = "utf-8") -> str:
        return path.read_text(encoding=encoding)

    def write_bytes(self, path: Path, content: bytes) -> None:
        path.write_bytes(content)

    def delete(self, path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


class ThreadedFileOperations(FileOperations):
    """Run blocking file operations on a shared bounded thread pool."""

    def __init__(self, max_workers: int = 4) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="citadel-io",
        )

    def _run(self, operation: Callable[[], T]) -> T:
        return self._executor.submit(operation).result()

    def read_text(self, path: Path, *, encoding: str = "utf-8") -> str:
        return self._run(lambda: path.read_text(encoding=encoding))

    def write_bytes(self, path: Path, content: bytes) -> None:
        self._run(lambda: path.write_bytes(content))

    def delete(self, path: Path) -> None:
        def remove() -> None:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

        self._run(remove)

    def close(self) -> None:
        """Stop the shared file-operation workers."""
        self._executor.shutdown(wait=True)


def create_file_operations(max_workers: int) -> FileOperations:
    """Create the configured shared file-operation service."""
    return ThreadedFileOperations(max_workers=max_workers)


file_operations = create_file_operations(get_settings().file_io_workers)
direct_file_operations = DirectFileOperations()
