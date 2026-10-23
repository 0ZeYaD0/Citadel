"""Bounded in-memory caches for deterministic query pipeline stages."""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any


class QueryCache:
    """Cache embeddings and retrieval results until documents change."""

    def __init__(self, max_entries: int = 128) -> None:
        self._max_entries = max_entries
        self._embeddings: OrderedDict[str, list[float]] = OrderedDict()
        self._contexts: OrderedDict[str, list[Any]] = OrderedDict()
        self._lock = Lock()

    def _get(self, cache: OrderedDict, key: str) -> Any:
        with self._lock:
            value = cache.get(key)
            if value is None:
                return None
            cache.move_to_end(key)
            return value

    def _set(self, cache: OrderedDict, key: str, value: Any) -> None:
        with self._lock:
            cache[key] = value
            cache.move_to_end(key)
            while len(cache) > self._max_entries:
                cache.popitem(last=False)

    def embedding(self, key: str) -> list[float] | None:
        return self._get(self._embeddings, key)

    def store_embedding(self, key: str, value: list[float]) -> None:
        self._set(self._embeddings, key, value)

    def context(self, key: str) -> list[Any] | None:
        value = self._get(self._contexts, key)
        return list(value) if value is not None else None

    def store_context(self, key: str, value: list[Any]) -> None:
        self._set(self._contexts, key, list(value))

    def clear(self) -> None:
        with self._lock:
            self._embeddings.clear()
            self._contexts.clear()
