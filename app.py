import logging
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from citadel.core.config import settings
from citadel.core.exceptions import EmptyQuestionError, GenerationError
from citadel.infrastructure.llm.groq_generator import GroqAnswerGenerator
from citadel.infrastructure.llm.local_embedder import LocalEmbedder
from citadel.infrastructure.loaders.pdf_chunker import TextbookDocumentProcessor
from citadel.infrastructure.vectorstore.chroma_repository import ChromaRepository
from citadel.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG System API",
    description="API for querying the RAG vector database.",
    version="1.0.0"
)

# Define the expected JSON input structure
class UserQuery(BaseModel):
    question: str


class UIQueryRequest(BaseModel):
    """Matches the shape sent by api_client.js → queryKnowledgeBase()."""
    query: str
    sources: list[dict] = []


# Components are initialized during application startup after configuration and
# the persisted Chroma collection have been validated.
query_embedder: LocalEmbedder | None = None
vector_repository: ChromaRepository | None = None
answer_generator: GroqAnswerGenerator | None = None
ingestion_service: IngestionService | None = None
document_processor: TextbookDocumentProcessor | None = None


def _extract_chapter_number(question: str) -> int | None:
    """Extract an explicit chapter number from a student question."""
    match = re.search(r"\bchapter\s+(\d+)\b", question, re.IGNORECASE)
    return int(match.group(1)) if match else None


@app.on_event("startup")
def load_rag_pipeline():
    """Load the embedding, retrieval, and grounded-generation components."""
    global answer_generator, query_embedder, vector_repository
    global ingestion_service, document_processor

    persist_directory = str(settings.chroma_persist_dir)
    if not os.path.exists(persist_directory):
        print(f"WARNING: {persist_directory} not found. Please run ingestion first.")
        return

    try:
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
            "RAG pipeline loaded with chat model: %s and %d stored chunks",
            settings.groq_model,
            vector_repository.document_count(),
        )
    except Exception:
        logger.exception("Failed to initialize the RAG pipeline")
        query_embedder = None
        vector_repository = None
        answer_generator = None
        ingestion_service = None
        document_processor = None


@app.get("/health")
def health_check() -> dict[str, str]:
    """Report whether the retrieval and generation pipeline is ready."""
    ready = all((query_embedder, vector_repository, answer_generator))
    return {
        "status": "ok" if ready else "not_ready",
        "pipeline": "ready" if ready else "not_initialized",
    }


# ── Original /ask endpoint (kept for backwards compatibility) ──────────────

@app.post("/ask")
def ask_question(query: UserQuery):
    """Endpoint to submit a query to the RAG system."""
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
            {"chapter_number": chapter_number}
            if chapter_number is not None
            else None
        )
        context_chunks = vector_repository.similarity_search(
            query_embedding,
            top_k=settings.retrieval_top_k,
            metadata_filter=metadata_filter,
        )
        response = answer_generator.generate_grounded_answer(
            query.question,
            context_chunks,
        )
        return asdict(response)
    except EmptyQuestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("RAG request failed")
        raise HTTPException(
            status_code=502,
            detail="The RAG request could not be completed.",
        ) from exc


# ── UI API endpoints (/api/*) ──────────────────────────────────────────────

@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Accept file uploads from the UI, ingest them into Chroma, and return
    source metadata so the frontend can show them on the desk."""
    if document_processor is None or ingestion_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not ready. Please try again shortly.",
        )

    sources = []
    for idx, upload in enumerate(files):
        # Save to a temp location so the document processor can read it
        suffix = Path(upload.filename or "file.txt").suffix
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, dir=tempfile.gettempdir()
        ) as tmp:
            content_bytes = await upload.read()
            tmp.write(content_bytes)
            tmp_path = Path(tmp.name)

        try:
            # Process the single file into LangChain Document chunks
            chunks = document_processor.process_single_file(tmp_path)

            if not chunks:
                logger.warning("File %s produced no chunks.", upload.filename)
                sources.append({
                    "id": f"doc-{idx}-{upload.filename}",
                    "name": upload.filename,
                    "size": f"{len(content_bytes) / 1024:.1f} KB",
                    "active": True,
                    "content": "(No extractable text found in this document.)",
                })
                continue

            # Ingest into the vector store
            stored = ingestion_service.ingest(chunks)
            logger.info(
                "Uploaded %s: %d chunks ingested.",
                upload.filename, stored,
            )

            # Build a preview from the first few chunks
            preview_text = "\n\n".join(
                chunk.page_content for chunk in chunks[:3]
            )
            if len(chunks) > 3:
                preview_text += f"\n\n... and {len(chunks) - 3} more passages."

            sources.append({
                "id": f"doc-{idx}-{upload.filename}",
                "name": upload.filename,
                "size": f"{len(content_bytes) / 1024:.1f} KB",
                "active": True,
                "content": preview_text,
            })
        except Exception as exc:
            logger.exception("Failed to process uploaded file %s", upload.filename)
            sources.append({
                "id": f"doc-{idx}-{upload.filename}",
                "name": upload.filename,
                "size": f"{len(content_bytes) / 1024:.1f} KB",
                "active": True,
                "content": f"Could not process file: {exc}",
            })
        finally:
            # Clean up temp file
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return {"status": "ok", "sources": sources}


@app.post("/api/query")
def query_knowledge_base(request: UIQueryRequest):
    """Process a question from the UI and return a formatted answer with
    [Source N] references that the frontend can render as citation badges."""
    if query_embedder is None or vector_repository is None or answer_generator is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not ready. Check that ingestion completed successfully.",
        )

    question = request.query.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        query_embedding = query_embedder.embed_query(question)
        chapter_number = _extract_chapter_number(question)
        metadata_filter = (
            {"chapter_number": chapter_number}
            if chapter_number is not None
            else None
        )
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
            }

        rag_response = answer_generator.generate_grounded_answer(
            question,
            context_chunks,
        )

        # Build [Source N] references at the end so the UI can render
        # clickable citation badges.
        source_labels = " ".join(
            f"[Source {i}]" for i in range(1, len(rag_response.citations) + 1)
        )

        answer_text = rag_response.answer
        # Append source tags if the model didn't already include them
        if "[Source " not in answer_text and source_labels:
            answer_text = f"{answer_text}\n\n{source_labels}"

        return {
            "answer": answer_text,
            "citations": [asdict(c) for c in rag_response.citations],
            "is_grounded": rag_response.is_grounded,
            "model_used": rag_response.model_used,
            "latency_ms": rag_response.latency_ms,
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


# ── Serve the static UI ────────────────────────────────────────────────────
# Mount static assets (css, js) at /css and /js
app.mount("/css", StaticFiles(directory="static/css"), name="css")
app.mount("/js", StaticFiles(directory="static/js"), name="js")


@app.get("/")
def serve_ui():
    """Serve the main Citadel study interface."""
    return FileResponse("static/index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)