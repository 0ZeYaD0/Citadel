"""Request models used by the API routes."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class UIQueryRequest(BaseModel):
    # Query payload sent by the frontend.
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
    # Study tool generation request.
    tool: str
    document_ids: list[str] = []
    chapter: Optional[int] = None
    count: int = 5


class UserQuery(BaseModel):
    question: str
