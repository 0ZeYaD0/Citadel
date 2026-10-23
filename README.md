# Citadel

Citadel is a Notebook-inspired AI study assistant. Upload your textbooks and documents, then ask questions, generate flashcards, get quizzes, and create summaries — all grounded in your own material.

## How It Works

1. **Upload** — Drop in PDF, DOCX, TXT, MD, CSV, or HTML files via the browser.
2. **Embed** — Documents are chunked and embedded locally with `all-MiniLM-L6-v2` (no API required for embeddings).
3. **Retrieve** — Student questions are embedded and matched against stored passages using vector similarity.
4. **Generate** — Groq produces a grounded answer using only retrieved passages as context.
5. **Persist** — Documents, sessions, messages, and notes are stored in a local SQLite database (`citadel.db`) and survive page refreshes.

## Stack

| Concern | Provider |
|---|---|
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (local CPU, no API key) |
| **Answer generation** | Groq (free tier, fast) |
| **Vector store** | ChromaDB (local, persistent) |
| **Document registry / sessions / notes** | SQLite (`citadel.db`) |
| **Supported file types** | `.pdf` `.docx` `.txt` `.md` `.csv` `.html` |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=qwen/qwen3-32b
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=chroma_db
CHROMA_COLLECTION_NAME=citadel_local_minilm
RETRIEVAL_TOP_K=5
CHUNK_SIZE=1500
CHUNK_OVERLAP=200
```

> **Note**: `GEMINI_API_KEY` is not required unless you switch to `gemini_generator.py`. The default stack uses Groq + local embeddings only.

## Running

```powershell
python app.py
```

Open [http://localhost:8000](http://localhost:8000).  
Upload documents, wait for the processing badge to turn green, then ask questions.

## Features

### Document Management
- Upload multiple files at once (PDF, DOCX, TXT, MD, CSV, HTML)
- Background processing with live progress stages (Uploading → Extracting → Chunking → Embedding → Ready)
- Documents persist in SQLite registry — survive page refreshes
- Delete individual documents or all documents at once
- Per-document chunk count display

### Active-Source Filtering
- Check/uncheck documents to include or exclude them from retrieval
- If no documents are selected, a clear message is shown instead of searching everything
- Filtering is enforced in the backend — not just cosmetic

### Study Tools
From the sidebar, select any combination of checked documents and click:
- 🃏 **Flashcards** — Q&A pairs from your material
- 📝 **Quiz me** — Multiple-choice questions
- 📋 **Summarize** — Structured bullet-point summary
- 🔑 **Key terms** — Definitions extracted from the text
- 💡 **Explain simply** — Beginner-friendly explanation with analogies

### Grounded Answers
- Every answer cites the source file and page number
- **Grounded** / **No evidence found** badges on each response
- Generation latency displayed per response
- Similarity threshold filtering removes irrelevant chunks before generation
- If no supporting evidence is found, the assistant says so rather than guessing

### Sessions & Notes
- Study sessions (notebooks) persist across page refreshes
- Chat messages are saved per session to SQLite
- Save any AI response as a pinned note
- Notes persist across sessions and can be deleted individually

### Observability
`GET /health` returns:
```json
{
  "status": "ok",
  "pipeline": "ready",
  "database": "ready",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "generation_model": "qwen/qwen3-32b",
  "total_chunks": 1847
}
```

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Pipeline + DB liveness |
| `POST` | `/api/upload` | Upload files → returns `job_id` |
| `GET` | `/api/jobs/{id}` | Poll background job status |
| `GET` | `/api/documents` | List document registry |
| `DELETE` | `/api/documents/{id}` | Delete one document |
| `DELETE` | `/api/documents` | Delete all documents |
| `POST` | `/api/query` | RAG query with source filter |
| `POST` | `/api/study-tools` | Flashcards, quiz, summary, key-terms |
| `GET` | `/api/sessions` | List sessions |
| `POST` | `/api/sessions` | Create / rename session |
| `DELETE` | `/api/sessions/{id}` | Delete session + messages |
| `GET` | `/api/sessions/{id}/messages` | Get session messages |
| `POST` | `/api/sessions/{id}/messages` | Append message |
| `GET` | `/api/notes` | List notes |
| `POST` | `/api/notes` | Save note |
| `DELETE` | `/api/notes/{id}` | Delete note |

## Running Tests

```powershell
pytest tests/ -v
```

Integration tests mock the embedding, vector, and LLM components so they run fully offline and quickly.

## CLI Ingestion (Optional)

Pre-load files before opening the browser:

```powershell
python ingest_data.py data\textbooks\your_textbook.pdf
```
