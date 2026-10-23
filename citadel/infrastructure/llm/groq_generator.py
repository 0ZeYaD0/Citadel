"""Groq implementation of Citadel's grounded answer generator.

Uses Groq's ultra-fast LPU inference with open-weight models (e.g. Llama 3.3)
for near-instant, free-tier-friendly chat completions.  Embeddings still go
through a local sentence-transformer — only the *generation* step is routed
to Groq.
"""

import logging
import time
from pathlib import Path
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from citadel.core.config import Settings, get_settings
from citadel.core.exceptions import EmptyQuestionError, GenerationError
from citadel.domain.interfaces import BaseAnswerGenerator
from citadel.domain.models import DocumentChunk, RAGResponse, SourceCitation


logger = logging.getLogger(__name__)


REFUSAL_MESSAGE = (
    "I cannot find the answer to this question in the provided textbook material."
)

# Minimum cosine-similarity score a chunk must have to be included in context.
# Chunks below this threshold are considered too dissimilar to be useful and
# are filtered out before generation to reduce hallucination risk.
DEFAULT_MIN_SIMILARITY = 0.25


class GroqAnswerGenerator(BaseAnswerGenerator):
    """Generate low-temperature, textbook-grounded answers with Groq."""

    def __init__(
        self,
        settings: Settings | None = None,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ) -> None:
        """Initialize Groq and the grounded LCEL generation chain.

        Args:
            settings: Optional settings object, useful for testing or custom
                deployments. The application settings are used by default.
            min_similarity: Minimum cosine similarity score for a retrieved
                chunk to be used as context. Chunks below this are discarded.
        """
        self._settings = settings or get_settings()
        self._model_name = self._settings.groq_model
        self._min_similarity = min_similarity
        self._llm = ChatGroq(
            model=self._model_name,
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_output_tokens,
            api_key=self._settings.groq_api_key,
            max_retries=2,
        )
        self._chain = self._build_chain()

    def _build_chain(self):
        """Build the prompt, Groq model, and string parser with LCEL."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are Citadel, an academic study assistant grounded strictly in the student's uploaded textbook material.

Use ONLY the passages inside <textbook_context> to answer the student's question. Do not use prior general knowledge, assumptions, or external facts.

Whenever you state a fact from a passage, cite it inline using [Source: filename, Page: number]. If the page is unavailable, use [Source: filename, Page: N/A].

If the passages do not contain enough information, reply exactly: "{refusal_message}"
Do not guess or fabricate an answer.

You may summarize chapters, explain concepts simply, compare ideas, or create revision questions, but every point must come strictly from the supplied passages.""",
                ),
                (
                    "human",
                    "<textbook_context>\n{context}\n</textbook_context>\n\nStudent question: {question}",
                ),
            ]
        )
        return prompt | self._llm | StrOutputParser()

    def generate_grounded_answer(
        self,
        query: str,
        context_chunks: List[DocumentChunk],
    ) -> RAGResponse:
        """Generate a grounded answer and attach source citations.

        Raises:
            EmptyQuestionError: If ``query`` is empty or whitespace-only.
            GenerationError: If Groq or the LCEL chain fails.
        """
        started_at = time.perf_counter()
        question = query.strip()
        if not question:
            raise EmptyQuestionError("Question cannot be empty.")

        # Apply similarity threshold filter
        filtered_chunks = self._filter_by_similarity(context_chunks)
        raw_count = len(context_chunks)
        filtered_count = len(filtered_chunks)

        if filtered_count < raw_count:
            logger.debug(
                "Similarity filter removed %d / %d chunks (threshold=%.2f).",
                raw_count - filtered_count,
                raw_count,
                self._min_similarity,
            )

        if not filtered_chunks:
            response = self._refusal_response(question, started_at)
            return RAGResponse(
                question=response.question,
                answer=response.answer,
                citations=response.citations,
                is_grounded=response.is_grounded,
                model_used=response.model_used,
                latency_ms=response.latency_ms,
                retrieved_chunk_count=raw_count,
                evidence_found=False,
            )

        # Deduplicate context chunks to avoid identical passages
        deduplicated = self._deduplicate_chunks(filtered_chunks)

        context = self._format_context(deduplicated)
        try:
            answer = self._chain.invoke(
                {
                    "context": context,
                    "question": question,
                    "refusal_message": REFUSAL_MESSAGE,
                }
            ).strip()
        except Exception as exc:
            logger.exception(
                "Groq grounded generation failed",
                extra={
                    "model": self._model_name,
                    "question_length": len(question),
                    "context_chunk_count": len(deduplicated),
                },
            )
            raise GenerationError(
                "Failed to generate answer from Groq API"
            ) from exc

        is_grounded = REFUSAL_MESSAGE.lower() not in answer.lower()
        evidence_found = is_grounded

        return RAGResponse(
            question=question,
            answer=answer,
            citations=self._build_citations(deduplicated),
            is_grounded=is_grounded,
            model_used=self._model_name,
            latency_ms=self._latency_ms(started_at),
            retrieved_chunk_count=raw_count,
            evidence_found=evidence_found,
        )

    def _filter_by_similarity(
        self, chunks: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """Remove chunks whose similarity score falls below the threshold."""
        result = []
        for chunk in chunks:
            # If no score available (e.g. from older ingestion), keep the chunk
            if chunk.score is None or chunk.score >= self._min_similarity:
                result.append(chunk)
        return result

    @staticmethod
    def _deduplicate_chunks(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Remove duplicate chunks based on their ID, preserving order."""
        seen: set[str] = set()
        deduped: List[DocumentChunk] = []
        for chunk in chunks:
            if chunk.id not in seen:
                seen.add(chunk.id)
                deduped.append(chunk)
        return deduped

    @staticmethod
    def _format_context(context_chunks: List[DocumentChunk]) -> str:
        """Format retrieved chunks as clearly separated textbook passages."""
        passages: List[str] = []
        for passage_number, chunk in enumerate(context_chunks, start=1):
            source_file = GroqAnswerGenerator._source_file(chunk)
            page_number = GroqAnswerGenerator._page_number(chunk) or "N/A"
            score_str = f"{chunk.score:.3f}" if chunk.score is not None else "N/A"
            passages.append(
                f"--- [Passage {passage_number} | Source: {source_file} | "
                f"Page: {page_number} | Relevance: {score_str} | ID: {chunk.id}] ---\n{chunk.content}"
            )
        return "\n\n".join(passages)

    @classmethod
    def _build_citations(
        cls,
        context_chunks: List[DocumentChunk],
    ) -> List[SourceCitation]:
        """Convert retrieved chunks to concise UI-ready citations."""
        return [
            SourceCitation(
                source_file=cls._source_file(chunk),
                page_number=cls._page_number(chunk),
                chunk_id=chunk.id,
                excerpt=cls._excerpt(chunk.content),
            )
            for chunk in context_chunks
        ]

    @staticmethod
    def _page_number(chunk: DocumentChunk) -> int | None:
        """Read a one-based page number from current or legacy metadata."""
        page_number = chunk.metadata.get("page_number")
        if isinstance(page_number, int):
            return page_number

        legacy_page = chunk.metadata.get("page")
        if isinstance(legacy_page, int):
            return legacy_page + 1
        return None

    @staticmethod
    def _source_file(chunk: DocumentChunk) -> str:
        """Read a clean filename from current or legacy metadata."""
        source = chunk.metadata.get("source_file") or chunk.metadata.get("source")
        if not source:
            return "Unknown"
        return Path(str(source)).name

    @staticmethod
    def _excerpt(content: str, max_length: int = 200) -> str:
        """Return a clean, word-boundary-limited citation excerpt."""
        excerpt = " ".join(content.split())
        if len(excerpt) <= max_length:
            return excerpt
        shortened = excerpt[: max_length - 3].rsplit(" ", 1)[0].rstrip()
        return f"{shortened}..."

    def _refusal_response(
        self,
        question: str,
        started_at: float,
    ) -> RAGResponse:
        """Return a refusal without making an unnecessary Groq call."""
        return RAGResponse(
            question=question,
            answer=REFUSAL_MESSAGE,
            citations=[],
            is_grounded=False,
            model_used=self._model_name,
            latency_ms=self._latency_ms(started_at),
            retrieved_chunk_count=0,
            evidence_found=False,
        )

    @staticmethod
    def _latency_ms(started_at: float) -> float:
        """Convert elapsed monotonic time to milliseconds."""
        return round((time.perf_counter() - started_at) * 1000, 3)
