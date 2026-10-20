import logging
import os
import re
from dataclasses import asdict

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from citadel.core.config import settings
from citadel.core.exceptions import EmptyQuestionError, GenerationError
from citadel.infrastructure.llm.groq_generator import GroqAnswerGenerator
from citadel.infrastructure.llm.local_embedder import LocalEmbedder
from citadel.infrastructure.vectorstore.chroma_repository import ChromaRepository

logger = logging.getLogger(__name__)

# Load environment variables (GEMINI_API_KEY)
load_dotenv()

app = FastAPI(
    title="RAG System API",
    description="API for querying the RAG vector database.",
    version="1.0.0"
)

# Define the expected JSON input structure
class UserQuery(BaseModel):
    question: str

# Components are initialized during application startup after configuration and
# the persisted Chroma collection have been validated.
query_embedder: LocalEmbedder | None = None
vector_repository: ChromaRepository | None = None
answer_generator: GroqAnswerGenerator | None = None


def _extract_chapter_number(question: str) -> int | None:
    """Extract an explicit chapter number from a student question."""
    match = re.search(r"\bchapter\s+(\d+)\b", question, re.IGNORECASE)
    return int(match.group(1)) if match else None

@app.on_event("startup")
def load_rag_pipeline():
    """Load the embedding, retrieval, and grounded-generation components."""
    global answer_generator, query_embedder, vector_repository

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


@app.get("/health")
def health_check() -> dict[str, str]:
    """Report whether the retrieval and generation pipeline is ready."""
    ready = all((query_embedder, vector_repository, answer_generator))
    return {
        "status": "ok" if ready else "not_ready",
        "pipeline": "ready" if ready else "not_initialized",
    }
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model=settings.chat_model,
        temperature=0,
        google_api_key=settings.gemini_api_key
    )

    system_prompt = (
        "You are an expert assistant strictly relying on the provided context. "
        "Use the following pieces of retrieved context to answer the question. "
        "If the answer is not in the context, explicitly state that you do not know. "
        "Do not hallucinate external information."
        "\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    logger.info("RAG pipeline loaded with chat model: %s", settings.chat_model)

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
        logger.exception(
            "RAG request failed",
    
    try:
        response = rag_chain.invoke({"input": query.question})
    except Exception as exc:
        logger.exception(
            "Gemini response generation failed using model %s: %s",
            settings.chat_model,
            exc,
        )
        message = str(exc).upper()
        if "503" in message or "UNAVAILABLE" in message:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The LLM provider is temporarily unavailable. "
                    "Please retry shortly."
                ),
            ) from exc
        if "429" in message or "RESOURCE_EXHAUSTED" in message:
            raise HTTPException(
                status_code=429,
                detail="LLM quota is currently exhausted. Please retry later.",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail="The RAG request could not be completed.",
        ) from exc

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)