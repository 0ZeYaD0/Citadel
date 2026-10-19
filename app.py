import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from citadel.core.config import settings

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

# Global variable to store the loaded RAG pipeline
rag_chain = None

@app.on_event("startup")
def load_rag_pipeline():
    """Loads the vector database and LLM chain when the server starts."""
    global rag_chain

    persist_directory = str(settings.chroma_persist_dir)
    if not os.path.exists(persist_directory):
        print(f"WARNING: {persist_directory} not found. Please run ingestion first.")
        return

    vectorstore = Chroma(
        collection_name=settings.chroma_collection_name,
        persist_directory=persist_directory,
        embedding_function=GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.gemini_api_key
        )
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
    if rag_chain is None:
        raise HTTPException(
            status_code=500, 
            detail="RAG pipeline not initialized. Ensure ingestion.py was run."
        )
    
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
                    "Gemini is temporarily unavailable. "
                    "Please retry shortly or choose another CHAT_MODEL."
                ),
            ) from exc
        if "429" in message or "RESOURCE_EXHAUSTED" in message:
            raise HTTPException(
                status_code=429,
                detail="Gemini quota is currently exhausted. Please retry later.",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail="The Gemini response could not be generated.",
        ) from exc
    
    return {
        "question": query.question, 
        "answer": response["answer"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)