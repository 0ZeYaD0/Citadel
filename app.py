import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

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

    if not os.path.exists("./chroma_db"):
        print("WARNING: ./chroma_db not found. Please run ingestion.py first.")
        return

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY")
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
    print("RAG Pipeline successfully loaded into memory.")

@app.post("/ask")
def ask_question(query: UserQuery):
    """Endpoint to submit a query to the RAG system."""
    if rag_chain is None:
        raise HTTPException(
            status_code=500, 
            detail="RAG pipeline not initialized. Ensure ingestion.py was run."
        )
    
    response = rag_chain.invoke({"input": query.question})
    
    return {
        "question": query.question, 
        "answer": response["answer"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)