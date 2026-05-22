# app.py – Separate FastAPI backend for the RAG system

import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the RAG class and helpers from the existing demo module
from demo import RAG
# Import ingestion helpers from the refactored populate_db script
from populate_db import (
    ingest_wikipedia_topic,
    ingest_arxiv_query,
    ingest_pubmed_query,
    ingest_text_content,
)

# Global RAG instance – created once at startup for efficiency
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_PATH = os.getenv("MODEL_PATH", "BAAI/bge-m3")
DB_PATH = os.getenv("DB_PATH", "database")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "research_rag")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY missing – ensure .env is present")

rag = RAG(
    model_path=MODEL_PATH,
    google_api_key=GOOGLE_API_KEY,
    db_path=DB_PATH,
    collection_name=COLLECTION_NAME,
)

app = FastAPI(title="RAG API", version="0.1.0")

# Allow requests from the frontend (localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic request models ----------
class ChatRequest(BaseModel):
    question: str
    top_k: int = 3

class WikiIngestRequest(BaseModel):
    topic: str

class ArxivIngestRequest(BaseModel):
    query: str
    max_results: int = 2

class PubmedIngestRequest(BaseModel):
    query: str
    max_results: int = 2

# ---------- API Endpoints ----------
@app.get("/api/status")
def get_status():
    """Return basic status information about the vector store."""
    try:
        count = rag.collection.count()
        return {
            "collection": rag.collection.name,
            "document_count": count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    """Ask a question to the RAG system and receive answer & sources."""
    try:
        result = rag.ask(request.question, top_k=request.top_k)
        # Clean carriage returns for safe JSON transmission
        answer = str(result["answer"]).replace("\r", "")
        sources = [str(s).replace("\r", "") for s in result["sources"]]
        distances = result.get("distances", [])
        return {
            "answer": answer,
            "sources": sources,
            "distances": distances,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest/wiki")
def ingest_wiki(payload: WikiIngestRequest):
    """Ingest a Wikipedia topic into the collection."""
    try:
        ingest_wikipedia_topic(payload.topic)
        return {"status": "ok", "message": f"Wikipedia topic '{payload.topic}' ingested."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest/arxiv")
def ingest_arxiv(payload: ArxivIngestRequest):
    """Ingest arXiv results for a query."""
    try:
        ingest_arxiv_query(payload.query, max_results=payload.max_results)
        return {"status": "ok", "message": f"ArXiv query '{payload.query}' ingested."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest/pubmed")
def ingest_pubmed(payload: PubmedIngestRequest):
    """Ingest PubMed results for a query."""
    try:
        ingest_pubmed_query(payload.query, max_results=payload.max_results)
        return {"status": "ok", "message": f"PubMed query '{payload.query}' ingested."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """Upload a TXT/PDF/DOCX file and ingest its text content."""
    try:
        content = await file.read()
        # Determine simple handling based on extension – PDF/DOCX parsing is done by the helper
        filename = file.filename.lower()
        if filename.endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(content)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif filename.endswith(".docx"):
            from io import BytesIO
            from docx import Document
            doc = Document(BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs)
        else:
            # Assume plain text for other extensions
            text = content.decode("utf-8", errors="ignore")
        ingest_text_content(text, source_name=file.filename)
        return {"status": "ok", "message": f"File '{file.filename}' ingested."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Main entry point (optional) ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
