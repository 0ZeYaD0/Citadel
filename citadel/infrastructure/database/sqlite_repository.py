"""SQLite-backed persistence layer for Citadel.

Manages three concerns that ChromaDB cannot handle well:
  - Document registry (filename, size, status, chunk count …)
  - Study sessions and per-session chat messages
  - Pinned notes linked to a source document and page
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from citadel.core.config import get_settings

logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    file_type       TEXT NOT NULL,
    upload_date     TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL DEFAULT 0,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'pending',
    error_message   TEXT,
    preview_text    TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'Untitled Notebook',
    emoji       TEXT NOT NULL DEFAULT '📓',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user','ai')),
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_documents (
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (session_id, document_id)
);

CREATE TABLE IF NOT EXISTS notes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    document_id     TEXT REFERENCES documents(id) ON DELETE SET NULL,
    text            TEXT NOT NULL,
    page_number     INTEGER,
    tags            TEXT DEFAULT '[]',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL DEFAULT 'queued',
    stage       TEXT NOT NULL DEFAULT 'queued',
    detail      TEXT,
    result      TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_notes_session_id ON notes(session_id, id);
CREATE INDEX IF NOT EXISTS idx_session_documents_document_id
    ON session_documents(document_id);
CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at);
"""


class SQLiteRepository:
    """Thread-safe SQLite repository for Citadel metadata.

    Parameters
    ----------
    db_path:
        File path for the SQLite database.  Will be created on first use.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path or get_settings().database_path)
        self._settings = get_settings()
        self._local = threading.local()
        self._connections: set[sqlite3.Connection] = set()
        self._connections_lock = threading.Lock()
        self._init_db()
        logger.info("SQLiteRepository initialised at %s", self._db_path)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Return a per-thread connection, creating one if needed."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
                timeout=self._settings.sqlite_busy_timeout_ms / 1000,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(
                f"PRAGMA busy_timeout = {self._settings.sqlite_busy_timeout_ms}"
            )
            self._local.conn = conn
            with self._connections_lock:
                self._connections.add(conn)
        return self._local.conn

    @contextmanager
    def _cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            yield cur
            self._commit_with_retry(conn)
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def _init_db(self) -> None:
        """Create tables if they do not already exist."""
        conn = self._get_conn()
        conn.executescript(_DDL)
        self._commit_with_retry(conn)

    def _commit_with_retry(self, conn: sqlite3.Connection) -> None:
        for attempt in range(self._settings.sqlite_lock_retries + 1):
            try:
                conn.commit()
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt >= self._settings.sqlite_lock_retries:
                    raise
                time.sleep(self._settings.sqlite_retry_delay_seconds * (attempt + 1))

    def close(self) -> None:
        """Close all connections opened by this repository."""
        with self._connections_lock:
            connections = tuple(self._connections)
            self._connections.clear()
        for conn in connections:
            conn.close()
        self._local.conn = None


    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Document registry ─────────────────────────────────────────────────

    def upsert_document(
        self,
        *,
        doc_id: str,
        filename: str,
        file_type: str,
        file_size_bytes: int,
        chunk_count: int = 0,
        status: str = "processing",
        error_message: str | None = None,
        preview_text: str | None = None,
    ) -> None:
        """Insert or update a document record."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents
                    (id, filename, file_type, upload_date, file_size_bytes,
                     chunk_count, status, error_message, preview_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    filename        = excluded.filename,
                    file_type       = excluded.file_type,
                    file_size_bytes = excluded.file_size_bytes,
                    chunk_count     = excluded.chunk_count,
                    status          = excluded.status,
                    error_message   = excluded.error_message,
                    preview_text    = excluded.preview_text
                """,
                (
                    doc_id,
                    filename,
                    file_type,
                    self._now(),
                    file_size_bytes,
                    chunk_count,
                    status,
                    error_message,
                    preview_text,
                ),
            )

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Return one document record as a dict, or None."""
        with self._cursor() as cur:
            cur.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cur.fetchone()
        return dict(row) if row else None

    def list_documents(self, session_id: str | None = None) -> list[dict[str, Any]]:
        """Return documents, optionally scoped to one notebook."""
        with self._cursor() as cur:
            if session_id:
                cur.execute(
                    """
                    SELECT d.*
                    FROM documents d
                    JOIN session_documents sd ON sd.document_id = d.id
                    WHERE sd.session_id = ?
                    ORDER BY d.upload_date DESC
                    """,
                    (session_id,),
                )
            else:
                cur.execute("SELECT * FROM documents ORDER BY upload_date DESC")
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def link_document_to_session(self, session_id: str, document_id: str) -> None:
        """Associate a document with a notebook idempotently."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR IGNORE INTO session_documents
                    (session_id, document_id, created_at)
                VALUES (?, ?, ?)
                """,
                (session_id, document_id, self._now()),
            )

    def document_ids_for_session(self, session_id: str) -> list[str]:
        """Return document IDs owned by a notebook."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT document_id FROM session_documents WHERE session_id = ?",
                (session_id,),
            )
            return [row["document_id"] for row in cur.fetchall()]

    def session_count_for_document(self, document_id: str) -> int:
        """Return how many notebooks currently reference a document."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS count FROM session_documents WHERE document_id = ?",
                (document_id,),
            )
            return int(cur.fetchone()["count"])

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document record. Returns True if a row was deleted."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            return cur.rowcount > 0

    def update_document_status(
        self,
        doc_id: str,
        *,
        status: str,
        chunk_count: int | None = None,
        error_message: str | None = None,
        preview_text: str | None = None,
    ) -> None:
        """Patch status (and optionally chunk_count / error) for one document."""
        with self._cursor() as cur:
            if chunk_count is not None and preview_text is not None:
                cur.execute(
                    """UPDATE documents
                       SET status=?, chunk_count=?, error_message=?, preview_text=?
                       WHERE id=?""",
                    (status, chunk_count, error_message, preview_text, doc_id),
                )
            elif chunk_count is not None:
                cur.execute(
                    "UPDATE documents SET status=?, chunk_count=?, error_message=? WHERE id=?",
                    (status, chunk_count, error_message, doc_id),
                )
            else:
                cur.execute(
                    "UPDATE documents SET status=?, error_message=? WHERE id=?",
                    (status, error_message, doc_id),
                )

    # ── Sessions ──────────────────────────────────────────────────────────

    def upsert_session(
        self,
        *,
        session_id: str,
        title: str = "Untitled Notebook",
        emoji: str = "📓",
    ) -> None:
        """Insert or update a session."""
        now = self._now()
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions (id, title, emoji, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title      = excluded.title,
                    emoji      = excluded.emoji,
                    updated_at = excluded.updated_at
                """,
                (session_id, title, emoji, now, now),
            )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Return one notebook record, or None when it does not exist."""
        with self._cursor() as cur:
            cur.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cur.fetchone()
        return dict(row) if row else None

    def touch_session(self, session_id: str) -> None:
        """Update the updated_at timestamp for a session."""
        with self._cursor() as cur:
            cur.execute(
                "UPDATE sessions SET updated_at=? WHERE id=?",
                (self._now(), session_id),
            )

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return all sessions ordered by most recently updated."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC"
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and cascade-delete its messages and links."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cur.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cur.rowcount > 0

    # ── Messages ──────────────────────────────────────────────────────────

    def add_message(
        self,
        *,
        session_id: str,
        role: str,
        text: str,
    ) -> int:
        """Append one message to a session. Returns the new row id."""
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO messages (session_id, role, text, created_at)
                   VALUES (?, ?, ?, ?)""",
                (session_id, role, text, self._now()),
            )
            new_id = cur.lastrowid
        self.touch_session(session_id)
        return new_id

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Return all messages for a session in chronological order."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM messages WHERE session_id=? ORDER BY id ASC",
                (session_id,),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    # ── Notes ─────────────────────────────────────────────────────────────

    def add_note(
        self,
        *,
        session_id: str | None,
        document_id: str | None,
        text: str,
        page_number: int | None = None,
        tags: list[str] | None = None,
    ) -> int:
        """Save a pinned note. Returns the new note id."""
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO notes
                   (session_id, document_id, text, page_number, tags, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    document_id,
                    text,
                    page_number,
                    json.dumps(tags or []),
                    self._now(),
                ),
            )
            return cur.lastrowid

    def list_notes(self, session_id: str | None = None) -> list[dict[str, Any]]:
        """Return notes, optionally filtered by session."""
        with self._cursor() as cur:
            if session_id:
                cur.execute(
                    "SELECT * FROM notes WHERE session_id=? ORDER BY id DESC",
                    (session_id,),
                )
            else:
                cur.execute("SELECT * FROM notes ORDER BY id DESC")
            rows = cur.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["tags"] = json.loads(d.get("tags") or "[]")
            result.append(d)
        return result

    def delete_note(self, note_id: int) -> bool:
        """Delete one note by id."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            return cur.rowcount > 0

    # ── Background jobs ───────────────────────────────────────────────────

    def create_job(self, job_id: str) -> None:
        """Register a new background job in 'queued' state."""
        now = self._now()
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO jobs (id, status, stage, created_at, updated_at)
                   VALUES (?, 'queued', 'queued', ?, ?)""",
                (job_id, now, now),
            )

    def update_job(
        self,
        job_id: str,
        *,
        status: str,
        stage: str,
        detail: str | None = None,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        """Update a job's progress."""
        with self._cursor() as cur:
            cur.execute(
                """UPDATE jobs
                   SET status=?, stage=?, detail=?, result=?, error=?, updated_at=?
                   WHERE id=?""",
                (
                    status,
                    stage,
                    detail,
                    json.dumps(result) if result is not None else None,
                    error,
                    self._now(),
                    job_id,
                ),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Return one job record, or None."""
        with self._cursor() as cur:
            cur.execute("SELECT * FROM jobs WHERE id=?", (job_id,))
            row = cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("result"):
            try:
                d["result"] = json.loads(d["result"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d
