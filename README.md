# Citadel

Citadel is a Gemini Notebook-inspired study assistant. A student can upload or
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
GEMINI_API_KEY=your_gemini_api_key_here
```

The application uses Gemini for both embeddings and chat responses. Replace the
placeholder with your actual Gemini API key and keep the `.env` file private.

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
