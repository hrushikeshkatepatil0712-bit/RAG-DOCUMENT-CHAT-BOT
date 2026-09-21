
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# PATH CONFIGURATION
# ============================================================

# main.py location:
# E:\RAGCHATBOT\RAG\rag_chatbot\backend\main.py
#
# parent       -> backend
# parent.parent -> rag_chatbot

BASE_DIR = Path(__file__).resolve().parent.parent

# .env:
# E:\RAGCHATBOT\RAG\rag_chatbot\.env
ENV_PATH = BASE_DIR / ".env"

# FAISS:
# E:\RAGCHATBOT\RAG\rag_chatbot\faiss_index
FAISS_PATH = BASE_DIR / "faiss_index"


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(ENV_PATH)

# Optional check
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError(
        f"GEMINI_API_KEY not found. Check your .env file at: {ENV_PATH}"
    )


# ============================================================
# EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# LOAD FAISS VECTOR DATABASE
# ============================================================

print("Loading FAISS index...")

if not FAISS_PATH.exists():
    raise FileNotFoundError(
        f"FAISS index folder not found: {FAISS_PATH}"
    )

db = FAISS.load_local(
    FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS index loaded successfully!")


# ============================================================
# GEMINI LLM
# ============================================================

print("Loading Gemini model...")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)

print("Gemini model loaded successfully!")


# ============================================================
# RETRIEVER
# ============================================================

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a helpful assistant.

Use the provided context to answer the user's question.

Rules:
1. Answer using only the provided context.
2. Keep the answer within three sentences.
3. If the answer is not available in the context, say:
   "I don't know based on the provided documents."

Context:
{context}
"""


# ============================================================
# PROMPT TEMPLATE
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}")
    ]
)


# ============================================================
# RAG CHAIN
# ============================================================

qa_chain = create_stuff_documents_chain(
    llm,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    qa_chain
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Document RAG Chatbot API",
    description="RAG chatbot using FAISS and Gemini",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# REQUEST MODEL
# ============================================================

class Query(BaseModel):
    text: str


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():
    return {
        "message": "RAG API is running!"
    }


# ============================================================
# QUERY ENDPOINT
# ============================================================

@app.post("/query")
def query_rag(query: Query):

    try:
        response = rag_chain.invoke(
            {
                "input": query.text
            }
        )

        return {
            "answer": response.get(
                "answer",
                "No answer found"
            )
        }

    except Exception as e:

        return {
            "error": str(e)
        }