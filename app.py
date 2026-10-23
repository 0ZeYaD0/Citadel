"""Citadel FastAPI application.

Endpoints
─────────
GET  /health                  — Pipeline + DB liveness check
POST /api/upload              — Accept files, start background ingestion job
GET  /api/jobs/{job_id}       — Poll background job status
GET  /api/documents           — List all persisted documents (registry)
DELETE /api/documents/{id}    — Delete one document from vector store + registry
DELETE /api/documents         — Delete ALL documents
POST /api/query               — RAG query with active-source filtering
GET  /api/sessions            — List persisted study sessions
POST /api/sessions            — Create / update a session
DELETE /api/sessions/{id}     — Delete a session + its messages
GET  /api/sessions/{id}/messages — All messages for a session
POST /api/sessions/{id}/messages — Append a message
GET  /api/notes               — List notes (optionally ?session_id=…)
POST /api/notes               — Save a note
DELETE /api/notes/{id}        — Delete a note
POST /api/study-tools         — Flashcards / quiz / summary / key-terms
GET  /                        — Serve SPA
"""

from __future__ import annotations

import hashlib
import logging
import re
import tempfile
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from citadel.core.config import settings
from citadel.core.exceptions import EmptyQuestionError, GenerationError
from citadel.infrastructure.database.sqlite_repository import SQLiteRepository
from citadel.infrastructure.llm.groq_generator import GroqAnswerGenerator
from citadel.infrastructure.llm.local_embedder import LocalEmbedder
from citadel.infrastructure.loaders.pdf_chunker import TextbookDocumentProcessor
from citadel.infrastructure.vectorstore.chroma_repository import ChromaRepository
from citadel.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)

# ── Application ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Citadel RAG API",
    description="Study assistant backed by a local RAG pipeline.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response Schemas ─────────────────────────────────────────────────

class UIQueryRequest(BaseModel):
    """Query payload sent by api_client.js → queryKnowledgeBase()."""
    query: str
    sources: list[dict] = []
    min_similarity: Optional[float] = None


class SessionUpsertRequest(BaseModel):
    title: str = "Untitled Notebook"
    emoji: str = "📓"


class MessageRequest(BaseModel):
    role: str
    text: str


class NoteRequest(BaseModel):
    session_id: Optional[str] = None
    document_id: Optional[str] = None
    text: str
    page_number: Optional[int] = None
    tags: list[str] = []


class StudyToolRequest(BaseModel):
    """Study tool generation request."""
    tool: str           # 'flashcards' | 'quiz' | 'summary' | 'key_terms' | 'explain_simple'
    document_ids: list[str] = []
    chapter: Optional[int] = None
    count: int = 5      # number of items to generate


# ── Singletons (initialised at startup) ───────────────────────────────────────

query_embedder: LocalEmbedder | None = None
vector_repository: ChromaRepository | None = None
answer_generator: GroqAnswerGenerator | None = None
ingestion_service: IngestionService | None = None
document_processor: TextbookDocumentProcessor | None = None
db: SQLiteRepository | None = None


def _extract_chapter_number(question: str) -> int | None:
    """Extract an explicit chapter number from a student question."""
    match = re.search(r"\bchapter\s+(\d+)\b", question, re.IGNORECASE)
    return int(match.group(1)) if match else None


# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
def load_rag_pipeline():
    """Load the embedding, retrieval, grounded-generation, and DB components."""
    global answer_generator, query_embedder, vector_repository
    global ingestion_service, document_processor, db

    # Do not overwrite singletons if already injected (e.g. in test suites)
    if all((query_embedder, vector_repository, answer_generator, ingestion_service, document_processor, db)):
        return

    try:
        db = SQLiteRepository(db_path=Path(__file__).resolve().parent / "citadel.db")

        persist_directory = str(settings.chroma_persist_dir)
        query_embedder = LocalEmbedder()
        vector_repository = ChromaRepository(
            embedder=query_embedder,
            persist_directory=persist_directory,
            collection_name=settings.chroma_collection_name,
        )
        answer_generator = GroqAnswerGenerator()
        ingestion_service = IngestionService(vector_repo=vector_repository)
        document_processor = TextbookDocumentProcessor()

        logger.info(
            "RAG pipeline ready | model=%s | chunks=%d",
            settings.groq_model,
            vector_repository.document_count(),
        )
    except Exception:
        logger.exception("Failed to initialise the RAG pipeline")
        query_embedder = None
        vector_repository = None
        answer_generator = None
        ingestion_service = None
        document_processor = None


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check() -> dict:
    """Report whether the retrieval and generation pipeline is ready."""
    pipeline_ready = all((query_embedder, vector_repository, answer_generator))
    db_ready = db is not None
    chunk_count = vector_repository.document_count() if vector_repository else 0
    return {
        "status": "ok" if (pipeline_ready and db_ready) else "not_ready",
        "pipeline": "ready" if pipeline_ready else "not_initialized",
        "database": "ready" if db_ready else "not_initialized",
        "embedding_model": settings.local_embedding_model,
        "generation_model": settings.groq_model,
        "total_chunks": chunk_count,
    }


# ── Background ingestion job ───────────────────────────────────────────────────

def _run_ingestion_job(
    job_id: str,
    files_data: list[dict],  # [{name, suffix, content_bytes}]
) -> None:
    """Background task: process and embed uploaded files, update job status."""
    if db is None or document_processor is None or ingestion_service is None:
        return

    results: list[dict] = []

    db.update_job(
        job_id,
        status="running",
        stage="extracting",
        detail=f"Processing {len(files_data)} file(s)",
    )

    for file_info in files_data:
        original_name: str = file_info["name"]
        suffix: str = file_info["suffix"]
        content_bytes: bytes = file_info["content_bytes"]
        doc_id: str = file_info["doc_id"]

        db.update_job(
            job_id,
            status="running",
            stage="chunking",
            detail=f"Chunking {original_name}",
        )

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, dir=tempfile.gettempdir()
            ) as tmp:
                tmp.write(content_bytes)
                tmp_path = Path(tmp.name)

            chunks = document_processor.process_single_file(
                tmp_path,
                source_name=original_name,
                document_id=doc_id,
            )

            if not chunks:
                db.update_document_status(
                    doc_id,
                    status="ready",
                    chunk_count=0,
                    preview_text="(No extractable text found in this document.)",
                )
                results.append({
                    "id": doc_id,
                    "name": original_name,
                    "size": _fmt_size(len(content_bytes)),
                    "size_bytes": len(content_bytes),
                    "status": "ready",
                    "chunk_count": 0,
                    "content": "(No extractable text found in this document.)",
                    "active": True,
                })
                continue

            db.update_job(
                job_id,
                status="running",
                stage="embedding",
                detail=f"Embedding {len(chunks)} chunks from {original_name}",
            )

            stored = ingestion_service.ingest(chunks)
            preview_text = "\n\n".join(chunk.page_content for chunk in chunks[:3])
            if len(chunks) > 3:
                preview_text += f"\n\n... and {len(chunks) - 3} more passages."

            db.update_document_status(
                doc_id,
                status="ready",
                chunk_count=stored,
                preview_text=preview_text,
            )

            results.append({
                "id": doc_id,
                "name": original_name,
                "size": _fmt_size(len(content_bytes)),
                "size_bytes": len(content_bytes),
                "status": "ready",
                "chunk_count": stored,
                "content": preview_text,
                "active": True,
            })

            logger.info(
                "Ingested %s → %d chunks (doc_id=%s)", original_name, stored, doc_id
            )

        except Exception as exc:
            logger.exception("Failed to ingest %s", original_name)
            error_msg = str(exc)
            db.update_document_status(doc_id, status="error", error_message=error_msg)
            results.append({
                "id": doc_id,
                "name": original_name,
                "size": _fmt_size(len(content_bytes)),
                "size_bytes": len(content_bytes),
                "status": "error",
                "error": error_msg,
                "content": f"Processing failed: {exc}",
                "active": False,
            })
        finally:
            if tmp_path:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    db.update_job(
        job_id,
        status="done",
        stage="complete",
        detail=f"Processed {len(files_data)} file(s)",
        result={"sources": results},
    )


# ── Upload endpoint ────────────────────────────────────────────────────────────

_SUPPORTED_TYPES = {".pdf", ".txt", ".md", ".docx", ".csv", ".html", ".htm"}


@app.post("/api/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
):
    """Accept file uploads, register them, and start a background ingestion job.

    Returns immediately with a ``job_id`` and the list of registered document IDs.
    Poll ``GET /api/jobs/{job_id}`` to follow progress.
    """
    if document_processor is None or ingestion_service is None or db is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not ready. Please try again shortly.",
        )
    if session_id and not db.get_session(session_id):
        raise HTTPException(status_code=404, detail="Notebook not found.")

    job_id = str(uuid.uuid4())
    db.create_job(job_id)

    files_data: list[dict] = []
    rejected: list[dict] = []

    for upload in files:
        original_name = Path(upload.filename or "file.bin").name
        suffix = Path(original_name).suffix.lower()

        if suffix not in _SUPPORTED_TYPES:
            rejected.append({
                "name": original_name,
                "error": (
                    f"Unsupported file type '{suffix}'. "
                    f"Supported: {', '.join(sorted(_SUPPORTED_TYPES))}"
                ),
            })
            continue

        content_bytes = await upload.read()

        if len(content_bytes) == 0:
            rejected.append({"name": original_name, "error": "File is empty."})
            continue

        # Deterministic document ID based on filename + content hash
        doc_id = hashlib.sha256(
            original_name.encode("utf-8") + b"\0" + content_bytes
        ).hexdigest()

        # Check if already ingested (idempotent re-upload)
        existing = db.get_document(doc_id)
        if existing and existing.get("status") == "ready":
            logger.info("Document %s already ingested; skipping.", original_name)
            if session_id:
                db.link_document_to_session(session_id, doc_id)
            files_data.append({
                "name": original_name,
                "suffix": suffix,
                "content_bytes": content_bytes,
                "doc_id": doc_id,
                "_skip": True,  # already done
            })
            continue

        # Register document in pending state
        db.upsert_document(
            doc_id=doc_id,
            filename=original_name,
            file_type=suffix.lstrip(".").upper(),
            file_size_bytes=len(content_bytes),
            status="processing",
        )
        if session_id:
            db.link_document_to_session(session_id, doc_id)

        files_data.append({
            "name": original_name,
            "suffix": suffix,
            "content_bytes": content_bytes,
            "doc_id": doc_id,
        })

    # Filter out already-done files before launching background task
    pending = [f for f in files_data if not f.get("_skip")]
    if pending:
        background_tasks.add_task(_run_ingestion_job, job_id, pending)
    else:
        # Nothing to process; mark job as instantly done
        db.update_job(
            job_id,
            status="done",
            stage="complete",
            detail="All files already ingested.",
            result={"sources": []},
        )

    return {
        "status": "accepted",
        "job_id": job_id,
        "queued": len(pending),
        "rejected": rejected,
        "document_ids": [f["doc_id"] for f in files_data],
    }


# ── Job status ─────────────────────────────────────────────────────────────────

@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    """Return the current status of a background ingestion job."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not ready.")
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return job


# ── Document registry ──────────────────────────────────────────────────────────

@app.get("/api/documents")
def list_documents(session_id: Optional[str] = Query(None)):
    """Return registered documents, optionally scoped to a notebook."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not ready.")
    docs = db.list_documents(session_id=session_id)
    return {"documents": docs}


@app.delete("/api/documents")
def delete_all_documents():
    """Delete ALL documents from both ChromaDB and the document registry."""
    if ingestion_service is None or db is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not ready.")
    try:
        ingestion_service.clear_store()
        docs = db.list_documents()
        for doc in docs:
            db.delete_document(doc["id"])
        logger.info("Deleted all %d documents.", len(docs))
        return {"status": "ok", "deleted_count": len(docs)}
    except Exception as exc:
        logger.exception("Failed to delete all documents")
        raise HTTPException(status_code=502, detail="Could not delete all documents.") from exc


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    """Delete one uploaded document from the vector store and the registry."""
    if ingestion_service is None or db is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not ready.")
    try:
        deleted_chunks = ingestion_service.delete_document(document_id)
        db.delete_document(document_id)
        return {
            "status": "ok",
            "document_id": document_id,
            "deleted_chunks": deleted_chunks,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to delete document %s", document_id)
        raise HTTPException(status_code=502, detail="The document could not be deleted.") from exc


# ── Query ──────────────────────────────────────────────────────────────────────

@app.post("/api/query")
def query_knowledge_base(request: UIQueryRequest):
    """Process a question from the UI and return a grounded answer with citations."""
    started = time.perf_counter()

    if query_embedder is None or vector_repository is None or answer_generator is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not ready. Check that ingestion completed successfully.",
        )

    question = request.query.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        active_source_ids = {
            str(source.get("id", "")).strip()
            for source in request.sources
            if str(source.get("id", "")).strip()
        }
        active_source_names = {
            str(source.get("name", "")).strip()
            for source in request.sources
            if str(source.get("name", "")).strip()
        }

        if not active_source_ids and not active_source_names:
            return {
                "answer": (
                    "No documents are selected. Check at least one document on "
                    "your desk before asking a question."
                ),
                "evidence_found": False,
            }

        query_embedding = query_embedder.embed_query(question)
        chapter_number = _extract_chapter_number(question)
        # Uploaded chunks are persisted with both identifiers. Use the
        # original filename when available because it is the same metadata
        # field used by the study-tools endpoint and remains compatible with
        # chunks created before stable document IDs were introduced.
        metadata_filter: dict = (
            {"source_file": {"$in": sorted(active_source_names)}}
            if active_source_names
            else {"document_id": {"$in": sorted(active_source_ids)}}
        )
        if chapter_number is not None:
            metadata_filter = {
                "$and": [
                    metadata_filter,
                    {"chapter_number": chapter_number},
                ]
            }

        context_chunks = vector_repository.similarity_search(
            query_embedding,
            top_k=settings.retrieval_top_k,
            metadata_filter=metadata_filter,
        )

        if not context_chunks:
            return {
                "answer": (
                    "I couldn't find any relevant passages in the documents on "
                    "your desk. Try rephrasing your question or uploading more "
                    "reading material."
                ),
                "evidence_found": False,
                "retrieved_chunk_count": 0,
            }

        # Optionally override the similarity threshold per-request
        if request.min_similarity is not None:
            answer_generator._min_similarity = request.min_similarity

        rag_response = answer_generator.generate_grounded_answer(
            question,
            context_chunks,
        )

        # Append [Source N] references for frontend citation badges
        source_labels = " ".join(
            f"[Source {i}]" for i in range(1, len(rag_response.citations) + 1)
        )
        answer_text = rag_response.answer
        if "[Source " not in answer_text and source_labels:
            answer_text = f"{answer_text}\n\n{source_labels}"

        total_latency = round((time.perf_counter() - started) * 1000, 1)

        return {
            "answer": answer_text,
            "citations": [asdict(c) for c in rag_response.citations],
            "is_grounded": rag_response.is_grounded,
            "evidence_found": rag_response.evidence_found,
            "model_used": rag_response.model_used,
            "latency_ms": rag_response.latency_ms,
            "total_latency_ms": total_latency,
            "retrieved_chunk_count": rag_response.retrieved_chunk_count,
        }

    except EmptyQuestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("RAG query from UI failed")
        raise HTTPException(
            status_code=502,
            detail="The RAG request could not be completed.",
        ) from exc


# ── Sessions ───────────────────────────────────────────────────────────────────

@app.get("/api/sessions")
def list_sessions():
    """Return all study sessions."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not ready.")
    return {"sessions": db.list_sessions()}


@app.post("/api/sessions")
def upsert_session(req: SessionUpsertRequest, session_id: Optional[str] = Query(None)):
    """Create a new session, or update an existing one."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not ready.")
    sid = session_id or f"nb-{uuid.uuid4().hex[:12]}"
    db.upsert_session(session_id=sid, title=req.title, emoji=req.emoji)
    return {"session_id": sid, "title": req.title, "emoji": req.emoji}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    """Delete a notebook, its messages, and its owned document vectors."""
    if db is None or ingestion_service is None:
        raise HTTPException(status_code=503, detail="Database not ready.")
    document_ids = db.document_ids_for_session(session_id)
    for document_id in document_ids:
        if db.session_count_for_document(document_id) == 1:
            ingestion_service.delete_document(document_id)
            db.delete_document(document_id)
    deleted = db.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"status": "ok", "session_id": session_id}


@app.get("/api/sessions/{session_id}/messages")
def get_messages(session_id: str):
    """Return all messages for a session in chronological order."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not ready.")
    return {"messages": db.get_messages(session_id)}


@app.post("/api/sessions/{session_id}/messages")
def add_message(session_id: str, req: MessageRequest):
    """Append one message to a session and return its id."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not ready.")
    if req.role not in ("user", "ai"):
        raise HTTPException(status_code=400, detail="role must be 'user' or 'ai'.")
    msg_id = db.add_message(session_id=session_id, role=req.role, text=req.text)
    return {"status": "ok", "message_id": msg_id}


# ── Notes ──────────────────────────────────────────────────────────────────────

@app.get("/api/notes")
def list_notes(session_id: Optional[str] = Query(None)):
    """Return notes, optionally filtered by session_id."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not ready.")
    return {"notes": db.list_notes(session_id=session_id)}


@app.post("/api/notes")
def save_note(req: NoteRequest):
    """Save a new note and return its id."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not ready.")
    note_id = db.add_note(
        session_id=req.session_id,
        document_id=req.document_id,
        text=req.text,
        page_number=req.page_number,
        tags=req.tags,
    )
    return {"status": "ok", "note_id": note_id}


@app.delete("/api/notes/{note_id}")
def delete_note(note_id: int):
    """Delete one note."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not ready.")
    deleted = db.delete_note(note_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Note not found.")
    return {"status": "ok", "note_id": note_id}


# ── Study Tools ────────────────────────────────────────────────────────────────

_STUDY_TOOL_PROMPTS: dict[str, str] = {
    "flashcards": (
        "Based ONLY on the textbook passages provided, generate exactly {count} "
        "flashcards as a numbered list. Each flashcard must follow this format:\n"
        "Q: <question>\nA: <answer>\n\n"
        "Focus on key concepts, definitions, and important facts. "
        "Do not include information not present in the passages."
    ),
    "quiz": (
        "Based ONLY on the textbook passages provided, create exactly {count} "
        "multiple-choice questions. Number them 1–{count}. For each question:\n"
        "- Write the question\n"
        "- Provide 4 options labeled A) B) C) D)\n"
        "- State the correct answer\n"
        "Use only facts present in the passages."
    ),
    "summary": (
        "Write a clear, structured summary of the textbook passages provided. "
        "Use bullet points and section headers where appropriate. "
        "Capture all key ideas, definitions, and relationships. "
        "Do not add information not present in the passages."
    ),
    "key_terms": (
        "Extract exactly {count} key terms or concepts from the textbook passages. "
        "For each term, provide: the term name and a one-sentence definition "
        "drawn directly from the text. Format as a numbered list."
    ),
    "explain_simple": (
        "Explain the main ideas of the textbook passages as if the student is "
        "encountering this topic for the very first time. Use simple language, "
        "real-world analogies, and short sentences. "
        "Do not include information not present in the passages."
    ),
}


@app.post("/api/study-tools")
def run_study_tool(req: StudyToolRequest):
    """Generate flashcards, quizzes, summaries, key terms, or simple explanations."""
    if (
        query_embedder is None
        or vector_repository is None
        or answer_generator is None
        or db is None
    ):
        raise HTTPException(status_code=503, detail="RAG pipeline is not ready.")

    tool_key = req.tool.lower().replace("-", "_")
    if tool_key not in _STUDY_TOOL_PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown tool '{req.tool}'. Choose from: {', '.join(_STUDY_TOOL_PROMPTS)}",
        )

    # Resolve which documents to use
    if req.document_ids:
        doc_names: set[str] = set()
        for did in req.document_ids:
            rec = db.get_document(did)
            if rec:
                doc_names.add(rec["filename"])
    else:
        # Use all ready documents
        doc_names = {
            d["filename"]
            for d in db.list_documents()
            if d.get("status") == "ready"
        }

    if not doc_names:
        return {
            "answer": "No ready documents found. Please upload and wait for processing to complete.",
            "tool": req.tool,
        }

    # Build a broad query embedding that will pull diverse content
    query_text = f"main concepts topics definitions {tool_key}"
    query_embedding = query_embedder.embed_query(query_text)

    metadata_filter: dict = {"source_file": {"$in": sorted(doc_names)}}
    if req.chapter is not None:
        metadata_filter = {
            "$and": [metadata_filter, {"chapter_number": req.chapter}]
        }

    context_chunks = vector_repository.similarity_search(
        query_embedding,
        top_k=min(settings.retrieval_top_k * 2, 20),
        metadata_filter=metadata_filter,
    )

    if not context_chunks:
        return {
            "answer": "No relevant passages found in the selected documents.",
            "tool": req.tool,
        }

    tool_instruction = _STUDY_TOOL_PROMPTS[tool_key].format(count=req.count)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"You are Citadel, an academic study assistant. {tool_instruction}\n"
            "Use ONLY the passages provided. Do not invent information.",
        ),
        (
            "human",
            "<textbook_context>\n{context}\n</textbook_context>\n\nGenerate the {tool} now.",
        ),
    ])
    chain = prompt | answer_generator._llm | StrOutputParser()

    context_text = "\n\n".join(
        f"[Source: {c.metadata.get('source_file','?')} | Page: {c.metadata.get('page_number','N/A')}]\n{c.content}"
        for c in context_chunks
    )

    try:
        result = chain.invoke({"context": context_text, "tool": req.tool})
        return {
            "answer": result.strip(),
            "tool": req.tool,
            "chunk_count": len(context_chunks),
        }
    except Exception as exc:
        logger.exception("Study tool '%s' failed", req.tool)
        raise HTTPException(
            status_code=502,
            detail=f"Study tool generation failed: {exc}",
        ) from exc


# ── Legacy /ask endpoint (backwards compatibility) ─────────────────────────────

class _UserQuery(BaseModel):
    question: str


@app.post("/ask")
def ask_question(query: _UserQuery):
    """Legacy endpoint kept for backwards compatibility."""
    if query_embedder is None or vector_repository is None or answer_generator is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not ready. Check that ingestion completed successfully.",
        )
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        query_embedding = query_embedder.embed_query(query.question)
        chapter_number = _extract_chapter_number(query.question)
        metadata_filter = (
            {"chapter_number": chapter_number} if chapter_number is not None else None
        )
        context_chunks = vector_repository.similarity_search(
            query_embedding,
            top_k=settings.retrieval_top_k,
            metadata_filter=metadata_filter,
        )
        response = answer_generator.generate_grounded_answer(query.question, context_chunks)
        return asdict(response)
    except EmptyQuestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("RAG request failed")
        raise HTTPException(status_code=502, detail="The RAG request could not be completed.") from exc


# ── Static UI ──────────────────────────────────────────────────────────────────

app.mount("/css", StaticFiles(directory="static/css"), name="css")
app.mount("/js", StaticFiles(directory="static/js"), name="js")


@app.get("/")
def serve_ui():
    """Serve the main Citadel study interface."""
    return FileResponse("static/index.html")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1024 ** 2:.1f} MB"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)