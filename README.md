# Citadel

Citadel is a Notebook-inspired study assistant. A student can upload or
add a textbook to the system and ask questions about its contents. The assistant
searches the textbook for relevant passages and uses them to produce a grounded
answer instead of relying only on general knowledge.

## Idea

The project is designed to help students study from their own learning material.
Possible questions include:

- "Summarise this chapter."
- "Explain this concept in simple terms."
- "What are the key differences between these two ideas?"
- "Create revision questions from this section."

If the answer cannot be found in the textbook, the assistant is instructed to
say so rather than inventing information.

## How it works

1. The textbook is split into smaller pieces of text.
2. The pieces are embedded locally with `all-MiniLM-L6-v2` and stored in Chroma.
3. A student's question is used to retrieve the most relevant textbook passages.
4. Groq generates an answer using those passages as context.

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a `.env` file and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

The local embedding model runs on CPU and does not use an API quota. Groq is
used only for answer generation. Keep the `.env` file private.

## Running the API

Because the embedding provider changed, rebuild the old Gemini Chroma database
once. This removes only the generated vector store, not the textbook:

```powershell
Remove-Item -Recurse -Force .\chroma_db -ErrorAction SilentlyContinue
python ingest_data.py data\textbooks\pythonlearn.pdf
```

After ingestion completes, start the server:

```powershell
python app.py
```

The API runs at `http://localhost:8000`. Ask a question by sending a `POST`
request to `/ask`:

```json
{
  "question": "What is the main idea of chapter one?"
}
```
