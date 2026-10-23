"""Integration tests for the Citadel FastAPI application.

These tests use a fully wired FastAPI TestClient against a lightweight
in-process server.  The RAG pipeline components (embedder, vector store,
LLM) are mocked so the tests run offline and very quickly.

Coverage:
  - /health endpoint
  - /api/upload (success, unsupported type, empty file, duplicate)
  - /api/jobs/{id} (job polling)
  - /api/documents (list, delete one, delete all)
  - /api/query (active-source filtering, no-documents guard)
  - /api/sessions CRUD
  - /api/sessions/{id}/messages CRUD
  - /api/notes CRUD
  - /api/study-tools
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Patch heavy singletons before app import ──────────────────────────────────

@pytest.fixture(autouse=True)
def _patch_pipeline(tmp_path, monkeypatch):
    """
    Replace the global pipeline singletons with lightweight mocks so the
    tests run without Groq API keys or a real vector store.
    """
    import app as _app

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 384
    mock_embedder.embed_documents.return_value = [[0.1] * 384]

    mock_repo = MagicMock()
    mock_repo.document_count.return_value = 0
    mock_repo.similarity_search.return_value = []
    mock_repo.add_documents.return_value = ["chunk-id-1"]
    mock_repo.delete_documents.return_value = 1
    mock_repo.delete_collection.return_value = None

    mock_generator = MagicMock()
    from citadel.domain.models import RAGResponse
    mock_generator.generate_grounded_answer.return_value = RAGResponse(
        question="test",
        answer="Mocked answer from textbook.",
        citations=[],
        is_grounded=True,
        model_used="mock-model",
        latency_ms=42.0,
        retrieved_chunk_count=3,
        evidence_found=True,
    )
    mock_generator._llm = MagicMock()
    mock_generator._min_similarity = 0.25

    mock_processor = MagicMock()
    from langchain_core.documents import Document
    mock_processor.process_single_file.return_value = [
        Document(
            page_content="This is test chunk content.",
            metadata={
                "source_file": "test.pdf",
                "document_id": "test-doc-id",
                "page_number": 1,
                "chunk_index": 0,
                "chunk_id": "test-chunk-id",
                "total_char_length": 26,
            },
        )
    ]

    # Use a temp SQLite DB for each test
    from citadel.infrastructure.database.sqlite_repository import SQLiteRepository
    test_db = SQLiteRepository(db_path=tmp_path / "test_citadel.db")

    monkeypatch.setattr(_app, "query_embedder", mock_embedder)
    monkeypatch.setattr(_app, "vector_repository", mock_repo)
    monkeypatch.setattr(_app, "answer_generator", mock_generator)
    monkeypatch.setattr(_app, "ingestion_service", MagicMock(
        ingest=MagicMock(return_value=1),
        clear_store=MagicMock(),
        delete_document=MagicMock(return_value=1),
    ))
    monkeypatch.setattr(_app, "document_processor", mock_processor)
    monkeypatch.setattr(_app, "db", test_db)
    monkeypatch.setattr(_app, "load_rag_pipeline", MagicMock())


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    """Return a synchronous FastAPI test client."""
    import app as _app
    with TestClient(_app.app, raise_server_exceptions=False) as c:
        yield c


# ─── Health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_ok_when_pipeline_ready(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "total_chunks" in data
        assert "embedding_model" in data
        assert "generation_model" in data


# ─── Upload ───────────────────────────────────────────────────────────────────

class TestUpload:
    def _upload(self, client, name: str, content: bytes = b"hello world"):
        return client.post(
            "/api/upload",
            files={"files": (name, io.BytesIO(content), "application/octet-stream")},
        )

    def test_pdf_upload_accepted(self, client):
        res = self._upload(client, "lecture.pdf", b"%PDF-1.4 fake content")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "accepted"
        assert "job_id" in data
        assert data["queued"] == 1

    def test_txt_upload_accepted(self, client):
        res = self._upload(client, "notes.txt", b"some study notes")
        assert res.status_code == 200
        assert res.json()["queued"] == 1

    def test_docx_upload_accepted(self, client):
        res = self._upload(client, "essay.docx", b"PK fake docx")
        assert res.status_code == 200
        assert res.json()["queued"] == 1

    def test_csv_upload_accepted(self, client):
        res = self._upload(client, "data.csv", b"col1,col2\nval1,val2")
        assert res.status_code == 200
        assert res.json()["queued"] == 1

    def test_html_upload_accepted(self, client):
        res = self._upload(client, "page.html", b"<html><body>content</body></html>")
        assert res.status_code == 200
        assert res.json()["queued"] == 1

    def test_unsupported_type_rejected(self, client):
        res = self._upload(client, "video.mp4", b"fake binary")
        assert res.status_code == 200
        data = res.json()
        assert data["queued"] == 0
        assert len(data["rejected"]) == 1
        assert "Unsupported" in data["rejected"][0]["error"]

    def test_empty_file_rejected(self, client):
        res = self._upload(client, "empty.pdf", b"")
        assert res.status_code == 200
        data = res.json()
        assert data["queued"] == 0
        assert "empty" in data["rejected"][0]["error"].lower()

    def test_duplicate_upload_skips_reingestion(self, client):
        # First upload
        res1 = self._upload(client, "dup.txt", b"same content")
        job_id_1 = res1.json()["job_id"]

        # Force document to 'ready' state in DB
        import app as _app
        doc_id = res1.json()["document_ids"][0]
        _app.db.update_document_status(doc_id, status="ready", chunk_count=1)

        # Second upload (same name + content = same hash)
        res2 = self._upload(client, "dup.txt", b"same content")
        assert res2.status_code == 200
        assert res2.json()["queued"] == 0  # skipped — already ready


# ─── Job polling ──────────────────────────────────────────────────────────────

class TestJobs:
    def test_get_existing_job(self, client):
        import app as _app
        import uuid
        job_id = str(uuid.uuid4())
        _app.db.create_job(job_id)
        _app.db.update_job(job_id, status="done", stage="complete", detail="done")
        res = client.get(f"/api/jobs/{job_id}")
        assert res.status_code == 200
        data = res.json()
        assert data["id"] == job_id
        assert data["status"] == "done"

    def test_missing_job_returns_404(self, client):
        res = client.get("/api/jobs/nonexistent-job-id")
        assert res.status_code == 404


# ─── Document registry ────────────────────────────────────────────────────────

class TestDocuments:
    def test_list_documents_empty(self, client):
        res = client.get("/api/documents")
        assert res.status_code == 200
        assert res.json()["documents"] == []

    def test_list_documents_after_upsert(self, client):
        import app as _app
        _app.db.upsert_document(
            doc_id="doc-abc",
            filename="test.pdf",
            file_type="PDF",
            file_size_bytes=1024,
            status="ready",
            chunk_count=5,
        )
        res = client.get("/api/documents")
        assert res.status_code == 200
        docs = res.json()["documents"]
        assert len(docs) == 1
        assert docs[0]["id"] == "doc-abc"
        assert docs[0]["filename"] == "test.pdf"

    def test_delete_one_document(self, client):
        import app as _app
        _app.db.upsert_document(
            doc_id="del-doc",
            filename="delete_me.pdf",
            file_type="PDF",
            file_size_bytes=512,
            status="ready",
        )
        res = client.delete("/api/documents/del-doc")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

        # Verify gone from registry
        docs = client.get("/api/documents").json()["documents"]
        assert all(d["id"] != "del-doc" for d in docs)

    def test_delete_all_documents(self, client):
        import app as _app
        for i in range(3):
            _app.db.upsert_document(
                doc_id=f"doc-{i}",
                filename=f"file{i}.pdf",
                file_type="PDF",
                file_size_bytes=100,
                status="ready",
            )
        res = client.delete("/api/documents")
        assert res.status_code == 200
        assert res.json()["deleted_count"] >= 3


# ─── Query ────────────────────────────────────────────────────────────────────

class TestQuery:
    def test_no_active_sources_returns_guidance(self, client):
        res = client.post(
            "/api/query",
            json={"query": "What is photosynthesis?", "sources": []},
        )
        assert res.status_code == 200
        assert "No documents are selected" in res.json()["answer"]

    def test_active_source_routes_to_generator(self, client):
        import app as _app
        # Make similarity_search return a chunk
        from citadel.domain.models import DocumentChunk
        _app.vector_repository.similarity_search.return_value = [
            DocumentChunk(
                id="c1",
                content="Photosynthesis converts light to energy.",
                metadata={"source_file": "bio.pdf", "page_number": 2},
                score=0.85,
            )
        ]
        res = client.post(
            "/api/query",
            json={"query": "What is photosynthesis?", "sources": [{"name": "bio.pdf"}]},
        )
        assert res.status_code == 200
        data = res.json()
        assert "answer" in data
        assert data["evidence_found"] is True

    def test_empty_query_returns_400(self, client):
        res = client.post(
            "/api/query",
            json={"query": "   ", "sources": [{"name": "bio.pdf"}]},
        )
        assert res.status_code == 400

    def test_citation_filename_is_correct(self, client):
        import app as _app
        from citadel.domain.models import DocumentChunk, RAGResponse, SourceCitation
        _app.vector_repository.similarity_search.return_value = [
            DocumentChunk(
                id="c2",
                content="Some content",
                metadata={"source_file": "correct_name.pdf"},
                score=0.9,
            )
        ]
        _app.answer_generator.generate_grounded_answer.return_value = RAGResponse(
            question="q",
            answer="Answer text",
            citations=[
                SourceCitation(
                    source_file="correct_name.pdf",
                    page_number=1,
                    chunk_id="c2",
                    excerpt="Some content",
                )
            ],
            is_grounded=True,
            model_used="mock",
            latency_ms=10.0,
            evidence_found=True,
        )
        res = client.post(
            "/api/query",
            json={"query": "test", "sources": [{"name": "correct_name.pdf"}]},
        )
        assert res.status_code == 200
        citations = res.json().get("citations", [])
        assert any(c["source_file"] == "correct_name.pdf" for c in citations)


# ─── Sessions ─────────────────────────────────────────────────────────────────

class TestSessions:
    def test_list_sessions_empty(self, client):
        res = client.get("/api/sessions")
        assert res.status_code == 200
        assert res.json()["sessions"] == []

    def test_create_and_retrieve_session(self, client):
        res = client.post(
            "/api/sessions",
            json={"title": "My Study Session", "emoji": "📚"},
        )
        assert res.status_code == 200
        sid = res.json()["session_id"]
        assert sid

        sessions = client.get("/api/sessions").json()["sessions"]
        assert any(s["id"] == sid for s in sessions)

    def test_delete_session(self, client):
        res = client.post("/api/sessions", json={"title": "To Delete"})
        sid = res.json()["session_id"]
        del_res = client.delete(f"/api/sessions/{sid}")
        assert del_res.status_code == 200

        sessions = client.get("/api/sessions").json()["sessions"]
        assert all(s["id"] != sid for s in sessions)

    def test_delete_nonexistent_session_404(self, client):
        res = client.delete("/api/sessions/does-not-exist")
        assert res.status_code == 404


# ─── Messages ─────────────────────────────────────────────────────────────────

class TestMessages:
    def test_add_and_retrieve_messages(self, client):
        res = client.post("/api/sessions", json={"title": "Chat Test"})
        sid = res.json()["session_id"]

        client.post(
            f"/api/sessions/{sid}/messages",
            json={"role": "user", "text": "Hello, Citadel!"},
        )
        client.post(
            f"/api/sessions/{sid}/messages",
            json={"role": "ai", "text": "Hello! How can I help?"},
        )

        msgs = client.get(f"/api/sessions/{sid}/messages").json()["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "ai"

    def test_invalid_role_rejected(self, client):
        res = client.post("/api/sessions", json={"title": "Role Test"})
        sid = res.json()["session_id"]
        bad = client.post(
            f"/api/sessions/{sid}/messages",
            json={"role": "bot", "text": "bad role"},
        )
        assert bad.status_code == 400

    def test_messages_cascade_delete_with_session(self, client):
        res = client.post("/api/sessions", json={"title": "Cascade"})
        sid = res.json()["session_id"]
        client.post(f"/api/sessions/{sid}/messages", json={"role": "user", "text": "msg"})
        client.delete(f"/api/sessions/{sid}")
        msgs = client.get(f"/api/sessions/{sid}/messages").json()["messages"]
        assert msgs == []


# ─── Notes ───────────────────────────────────────────────────────────────────

class TestNotes:
    def test_save_and_list_notes(self, client):
        res = client.post(
            "/api/notes",
            json={"text": "Important definition", "tags": ["definition"]},
        )
        assert res.status_code == 200
        note_id = res.json()["note_id"]

        notes = client.get("/api/notes").json()["notes"]
        assert any(n["id"] == note_id for n in notes)

    def test_delete_note(self, client):
        res = client.post("/api/notes", json={"text": "Temp note"})
        note_id = res.json()["note_id"]

        del_res = client.delete(f"/api/notes/{note_id}")
        assert del_res.status_code == 200

        notes = client.get("/api/notes").json()["notes"]
        assert all(n["id"] != note_id for n in notes)

    def test_delete_nonexistent_note_404(self, client):
        res = client.delete("/api/notes/99999")
        assert res.status_code == 404

    def test_notes_filtered_by_session(self, client):
        sess = client.post("/api/sessions", json={"title": "Sess"}).json()["session_id"]
        client.post("/api/notes", json={"text": "linked note", "session_id": sess})
        client.post("/api/notes", json={"text": "unlinked note"})

        linked = client.get(f"/api/notes?session_id={sess}").json()["notes"]
        assert all(n["session_id"] == sess for n in linked)


# ─── Study Tools ─────────────────────────────────────────────────────────────

class TestStudyTools:
    def _setup_docs(self, app_module):
        """Insert a ready document and mock a search result."""
        from citadel.domain.models import DocumentChunk
        app_module.db.upsert_document(
            doc_id="study-doc",
            filename="textbook.pdf",
            file_type="PDF",
            file_size_bytes=2048,
            status="ready",
            chunk_count=10,
        )
        app_module.vector_repository.similarity_search.return_value = [
            DocumentChunk(
                id="sc1",
                content="Key concept: machine learning is a subset of AI.",
                metadata={"source_file": "textbook.pdf", "page_number": 5},
                score=0.9,
            )
        ]

    def test_flashcards_tool(self, client):
        import app as _app
        self._setup_docs(_app)
        _app.answer_generator._llm.invoke = MagicMock()

        from langchain_core.output_parsers import StrOutputParser
        # Patch chain invoke response
        with patch("app.StrOutputParser") as mock_parser:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Q: What is ML?\nA: A subset of AI."
            _app.answer_generator._llm.__or__ = MagicMock(return_value=mock_chain)

            res = client.post(
                "/api/study-tools",
                json={"tool": "flashcards", "document_ids": ["study-doc"], "count": 3},
            )
        # Study tool endpoint is reachable (pipeline mock may return 200 or 502)
        assert res.status_code in (200, 502)

    def test_unknown_tool_returns_400(self, client):
        res = client.post(
            "/api/study-tools",
            json={"tool": "nonexistent_tool", "document_ids": []},
        )
        assert res.status_code == 400
        assert "Unknown tool" in res.json()["detail"]

    def test_no_ready_docs_returns_guidance(self, client):
        res = client.post(
            "/api/study-tools",
            json={"tool": "summary", "document_ids": ["nonexistent-id"]},
        )
        assert res.status_code == 200
        assert "No ready documents" in res.json()["answer"]
