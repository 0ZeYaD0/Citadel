from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from citadel.api.abstract_api import BackgroundTasks, File, Form, Query, UploadFile
from citadel.api.errors import ApiErrors
from citadel.api.helpers import (
    extract_chapter_number,
    health_status,
    run_ingestion_job,
)
from citadel.api.prompt_loader import load_study_prompts
from citadel.api.schemas import (
    MessageRequest,
    NoteRequest,
    SessionUpsertRequest,
    StudyToolRequest,
    UIQueryRequest,
    UserQuery,
)
from citadel.api.dependencies import settings
from citadel.api.services import ApplicationServices
from fastapi import APIRouter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)
router = APIRouter()
services = ApplicationServices()
@router.get("/health")
def health_check() -> dict:
    # Report whether the pipeline is ready.
    return health_status(services.as_rag_components())


_SUPPORTED_TYPES = {".pdf", ".txt", ".md", ".docx", ".csv", ".html", ".htm"}


@router.post("/api/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
):
    # Accept uploads and start background ingestion.
    if services.document_processor is None or services.ingestion_service is None or services.db is None:
        raise ApiErrors.create("pipeline_not_ready_retry")
    if session_id and not services.db.get_session(session_id):
        raise ApiErrors.create("notebook_not_found")

    job_id = str(uuid.uuid4())
    services.db.create_job(job_id)

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
        existing = services.db.get_document(doc_id)
        if existing and existing.get("status") == "ready":
            logger.info("Document %s already ingested; skipping.", original_name)
            if session_id:
                services.db.link_document_to_session(session_id, doc_id)
            files_data.append({
                "name": original_name,
                "suffix": suffix,
                "content_bytes": content_bytes,
                "doc_id": doc_id,
                "_skip": True,  # already done
            })
            continue

        # Register document in pending state
        services.db.upsert_document(
            doc_id=doc_id,
            filename=original_name,
            file_type=suffix.lstrip(".").upper(),
            file_size_bytes=len(content_bytes),
            status="processing",
        )
        if session_id:
            services.db.link_document_to_session(session_id, doc_id)

        files_data.append({
            "name": original_name,
            "suffix": suffix,
            "content_bytes": content_bytes,
            "doc_id": doc_id,
        })

    # Filter out already-done files before launching background task
    pending = [f for f in files_data if not f.get("_skip")]
    if pending:
        services.query_cache.clear()
    if pending:
        background_tasks.add_task(
            run_ingestion_job,
            job_id,
            pending,
            services.as_rag_components(),
        )
    else:
        # Nothing to process; mark job as instantly done
        services.db.update_job(
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


@router.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    # Return the current ingestion job status.
    if services.db is None:
        raise ApiErrors.create("database_not_ready")
    job = services.db.get_job(job_id)
    if not job:
        raise ApiErrors.job_not_found(job_id)
    return job


@router.get("/api/documents")
def list_documents(session_id: Optional[str] = Query(None)):
    # Return registered documents.
    if services.db is None:
        raise ApiErrors.create("database_not_ready")
    docs = services.db.list_documents(session_id=session_id)
    return {"documents": docs}


@router.delete("/api/documents")
def delete_all_documents():
    # Delete all documents from storage and the registry.
    if services.ingestion_service is None or services.db is None:
        raise ApiErrors.create("pipeline_not_ready")
    try:
        services.ingestion_service.clear_store()
        docs = services.db.list_documents()
        for doc in docs:
            services.db.delete_document(doc["id"])
        services.query_cache.clear()
        logger.info("Deleted all %d documents.", len(docs))
        return {"status": "ok", "deleted_count": len(docs)}
    except Exception as exc:
        logger.exception("Failed to delete all documents")
        raise ApiErrors.create("delete_all_failed") from exc


@router.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    # Delete one document from storage and the registry.
    if services.ingestion_service is None or services.db is None:
        raise ApiErrors.create("pipeline_not_ready")
    try:
        deleted_chunks = services.ingestion_service.delete_document(document_id)
        services.db.delete_document(document_id)
        services.query_cache.clear()
        return {
            "status": "ok",
            "document_id": document_id,
            "deleted_chunks": deleted_chunks,
        }
    except Exception as exc:
        logger.exception("Failed to delete document %s", document_id)
        raise ApiErrors.from_exception(exc, "document_delete_failed") from exc


@router.post("/api/query")
def query_knowledge_base(request: UIQueryRequest):
    # Process a question and return a grounded answer.
    started = time.perf_counter()
    timings: dict[str, float] = {}

    if services.query_embedder is None or services.vector_repository is None or services.answer_generator is None:
        raise ApiErrors.create("pipeline_not_ready_ingestion")

    question = request.query.strip()
    if not question:
        raise ApiErrors.bad_request("Question cannot be empty.")

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

        embedding_started = time.perf_counter()
        embedding_key = question.casefold()
        query_embedding = services.query_cache.embedding(embedding_key)
        if query_embedding is None:
            query_embedding = services.query_embedder.embed_query(question)
            services.query_cache.store_embedding(embedding_key, query_embedding)
        timings["embedding_ms"] = round(
            (time.perf_counter() - embedding_started) * 1000, 1
        )
        chapter_number = extract_chapter_number(question)
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

        retrieval_started = time.perf_counter()
        retrieval_key = repr(
            (
                embedding_key,
                sorted(active_source_ids),
                sorted(active_source_names),
                chapter_number,
                settings.retrieval_top_k,
            )
        )
        context_chunks = services.query_cache.context(retrieval_key)
        if context_chunks is None:
            context_chunks = services.vector_repository.similarity_search(
                query_embedding,
                top_k=settings.retrieval_top_k,
                metadata_filter=metadata_filter,
            )
            services.query_cache.store_context(retrieval_key, context_chunks)
        timings["retrieval_ms"] = round(
            (time.perf_counter() - retrieval_started) * 1000, 1
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
            services.answer_generator._min_similarity = request.min_similarity

        generation_started = time.perf_counter()
        rag_response = services.answer_generator.generate_grounded_answer(
            question,
            context_chunks,
        )
        timings["generation_ms"] = round(
            (time.perf_counter() - generation_started) * 1000, 1
        )

        answer_text = rag_response.answer

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
            "timings_ms": timings,
        }

    except Exception as exc:
        logger.exception("RAG query from UI failed")
        raise ApiErrors.from_exception(exc, "rag_request_failed") from exc


@router.get("/api/sessions")
def list_sessions():
    # Return all study sessions.
    if services.db is None:
        raise ApiErrors.create("database_not_ready")
    return {"sessions": services.db.list_sessions()}


@router.post("/api/sessions")
def upsert_session(req: SessionUpsertRequest, session_id: Optional[str] = Query(None)):
    # Create or update a study session.
    if services.db is None:
        raise ApiErrors.create("database_not_ready")
    sid = session_id or f"nb-{uuid.uuid4().hex[:12]}"
    services.db.upsert_session(session_id=sid, title=req.title, emoji=req.emoji)
    return {"session_id": sid, "title": req.title, "emoji": req.emoji}


@router.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    # Delete a notebook and its related data.
    if services.db is None or services.ingestion_service is None:
        raise ApiErrors.create("database_not_ready")
    document_ids = services.db.document_ids_for_session(session_id)
    for document_id in document_ids:
        if services.db.session_count_for_document(document_id) == 1:
            services.ingestion_service.delete_document(document_id)
            services.db.delete_document(document_id)
    deleted = services.db.delete_session(session_id)
    if not deleted:
        raise ApiErrors.create("session_not_found")
    return {"status": "ok", "session_id": session_id}


@router.get("/api/sessions/{session_id}/messages")
def get_messages(session_id: str):
    # Return session messages in chronological order.
    if services.db is None:
        raise ApiErrors.create("database_not_ready")
    return {"messages": services.db.get_messages(session_id)}


@router.post("/api/sessions/{session_id}/messages")
def add_message(session_id: str, req: MessageRequest):
    # Append a message to a session.
    if services.db is None:
        raise ApiErrors.create("database_not_ready")
    if req.role not in ("user", "ai"):
        raise ApiErrors.bad_request("role must be 'user' or 'ai'.")
    msg_id = services.db.add_message(session_id=session_id, role=req.role, text=req.text)
    return {"status": "ok", "message_id": msg_id}


@router.get("/api/notes")
def list_notes(session_id: Optional[str] = Query(None)):
    # Return notes, optionally filtered by session.
    if services.db is None:
        raise ApiErrors.create("database_not_ready")
    return {"notes": services.db.list_notes(session_id=session_id)}


@router.post("/api/notes")
def save_note(req: NoteRequest):
    # Save a note.
    if services.db is None:
        raise ApiErrors.create("database_not_ready")
    note_id = services.db.add_note(
        session_id=req.session_id,
        document_id=req.document_id,
        text=req.text,
        page_number=req.page_number,
        tags=req.tags,
    )
    return {"status": "ok", "note_id": note_id}


@router.delete("/api/notes/{note_id}")
def delete_note(note_id: int):
    # Delete one note.
    if services.db is None:
        raise ApiErrors.create("database_not_ready")
    deleted = services.db.delete_note(note_id)
    if not deleted:
        raise ApiErrors.create("note_not_found")
    return {"status": "ok", "note_id": note_id}


@router.post("/api/study-tools")
def run_study_tool(req: StudyToolRequest):
    # Generate a study tool response.
    if (
        services.query_embedder is None
        or services.vector_repository is None
        or services.answer_generator is None
        or services.db is None
    ):
        raise ApiErrors.create("pipeline_not_ready")

    try:
        study_prompts = load_study_prompts(settings.study_prompts_path)
    except (OSError, ValueError) as exc:
        logger.exception("Study prompt catalog could not be loaded")
        raise ApiErrors.create("prompt_configuration_failed") from exc

    tool_key = req.tool.lower().replace("-", "_")
    if tool_key not in study_prompts:
        raise ApiErrors.unknown_tool(
            req.tool, ", ".join(study_prompts)
        )

    if req.document_ids:
        doc_names: set[str] = set()
        for did in req.document_ids:
            rec = services.db.get_document(did)
            if rec:
                doc_names.add(rec["filename"])
    else:
        doc_names = {
            d["filename"]
            for d in services.db.list_documents()
            if d.get("status") == "ready"
        }

    if not doc_names:
        return {
            "answer": "No ready documents found. Please upload and wait for processing to complete.",
            "tool": req.tool,
        }

    query_text = f"main concepts topics definitions {tool_key}"
    query_embedding = services.query_embedder.embed_query(query_text)

    metadata_filter: dict = {"source_file": {"$in": sorted(doc_names)}}
    if req.chapter is not None:
        metadata_filter = {
            "$and": [metadata_filter, {"chapter_number": req.chapter}]
        }

    context_chunks = services.vector_repository.similarity_search(
        query_embedding,
        top_k=min(settings.retrieval_top_k * 2, 20),
        metadata_filter=metadata_filter,
    )

    if not context_chunks:
        return {
            "answer": "No relevant passages found in the selected documents.",
            "tool": req.tool,
        }

    tool_instruction = study_prompts[tool_key].replace("{count}", str(req.count))

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
    chain = prompt | services.answer_generator._llm | StrOutputParser()

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
        raise ApiErrors.study_tool_failed(str(exc)) from exc


@router.post("/ask")
def ask_question(query: UserQuery):
    # Preserve the legacy endpoint.
    if services.query_embedder is None or services.vector_repository is None or services.answer_generator is None:
        raise ApiErrors.create("pipeline_not_ready_ingestion")
    if not query.question.strip():
        raise ApiErrors.bad_request("Question cannot be empty.")
    try:
        query_embedding = services.query_embedder.embed_query(query.question)
        chapter_number = extract_chapter_number(query.question)
        metadata_filter = (
            {"chapter_number": chapter_number} if chapter_number is not None else None
        )
        context_chunks = services.vector_repository.similarity_search(
            query_embedding,
            top_k=settings.retrieval_top_k,
            metadata_filter=metadata_filter,
        )
        response = services.answer_generator.generate_grounded_answer(query.question, context_chunks)
        return asdict(response)
    except Exception as exc:
        logger.exception("RAG request failed")
        raise ApiErrors.from_exception(exc, "rag_request_failed") from exc

