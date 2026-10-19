# Citadel

Citadel is a Gemini Notebook-inspired study assistant. A student can upload or
add a textbook to the system and ask questions about its contents. The assistant
answer instead of relying only on general knowledge.

## Idea

material. Possible questions include:

- "Summarise this chapter."
- "Explain this concept in simple terms."
- "What are the key differences between these two ideas?"
- "Create revision questions from this section."

If the answer cannot be found in the textbook, the assistant is instructed to
say so rather than inventing information.

## How it works

1. The textbook is split into smaller pieces of text.
2. The pieces are converted into embeddings and stored in a Chroma vector database.
3. A student's question is used to retrieve the most relevant textbook passages.
4. The language model generates an answer using those passages as context.

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a `.env` file and add your API key:

```env
OPENAI_API_KEY=your_api_key_here
```

The current prototype uses OpenAI embeddings and a chat model. The model
provider can be changed later if the project is migrated to Gemini.

## Running the API

After the textbook has been ingested into `chroma_db`, start the server:

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

>>>>>>> b0dc72b (README change)
