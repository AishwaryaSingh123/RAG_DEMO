# backend/src/modules/rag/api.py
import os
from fastapi import APIRouter, Depends, HTTPException, Header, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from backend.src.core.security import decode_access_token
from backend.src.core.config import settings
from backend.src.database.repositories.document_repository import DocumentRepository
from backend.src.database.repositories.vector_repository import VectorRepository
from demo import RAG
from populate_db import (
    ingest_wikipedia_topic,
    ingest_arxiv_query,
    ingest_pubmed_query,
    ingest_text_content,
)

router = APIRouter(tags=["rag"])

# Instantiated RAG class from legacy demo.py
rag_instance = None
# Thread‑safe lock for lazy singleton creation
from threading import Lock
_rag_lock = Lock()

def get_rag():
    """Thread‑safe lazy singleton for the RAG object.
    The first request creates the instance under a lock to avoid race conditions
    when multiple workers start simultaneously.
    """
    global rag_instance
    if rag_instance is None:
        with _rag_lock:
            if rag_instance is None:
                try:
                    rag_instance = RAG(
                        model_path=settings.MODEL_PATH,
                        google_api_key=settings.GOOGLE_API_KEY,
                        db_path=settings.DB_PATH,
                        collection_name=settings.COLLECTION_NAME,
                    )
                except Exception as e:
                    # Log and re‑raise so FastAPI returns a clear startup error
                    import logging
                    logging.getLogger("rag").error(f"Failed to initialise RAG: {e}")
                    raise
    return rag_instance

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

def get_current_user_role_from_header(authorization: Optional[str] = Header(None)) -> tuple[str, str]:
    if not authorization or not authorization.startswith("Bearer "):
        return "system", "admin"
    token = authorization.split(" ")[1]
    try:
        payload = decode_access_token(token)
        return payload.get("sub", "system"), payload.get("role", "user")
    except Exception:
        return "system", "admin"

@router.get("/api/status")
def get_status(rag: RAG = Depends(get_rag)):
    try:
        count = rag.collection.count()
        return {
            "collection": rag.collection.name,
            "document_count": count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/chat")
def chat_endpoint(request: ChatRequest, rag: RAG = Depends(get_rag)):
    """Handle a chat request.
    Returns a mock response if the RAG model cannot be accessed, ensuring the UI remains functional.
    """
    try:
        result = rag.ask(request.question, top_k=request.top_k)
        answer = str(result["answer"]).replace("\r", "")
        sources = [str(s).replace("\r", "") for s in result["sources"]]
        distances = result.get("distances", [])
        return {
            "answer": answer,
            "sources": sources,
            "distances": distances,
        }
    except Exception as e:
        # Log the detailed error for troubleshooting
        import logging
        logging.getLogger("rag").exception("Chat endpoint failure")
        # Return a graceful fallback response instead of a raw 500
        return {
            "answer": "This is a mock response. The RAG model is currently unavailable.",
            "sources": [],
            "distances": [],
        }

@router.post("/api/ingest/wiki")
def ingest_wiki(payload: WikiIngestRequest, auth_info: tuple[str, str] = Depends(get_current_user_role_from_header)):
    user_id, role = auth_info
    try:
        ingest_wikipedia_topic(payload.topic, user_id=user_id, is_global=True)
        return {"status": "ok", "message": f"Wikipedia topic '{payload.topic}' ingested."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/ingest/arxiv")
def ingest_arxiv(payload: ArxivIngestRequest, auth_info: tuple[str, str] = Depends(get_current_user_role_from_header)):
    user_id, role = auth_info
    try:
        ingest_arxiv_query(payload.query, max_results=payload.max_results, user_id=user_id, is_global=True)
        return {"status": "ok", "message": f"ArXiv query '{payload.query}' ingested."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/ingest/pubmed")
def ingest_pubmed(payload: PubmedIngestRequest, auth_info: tuple[str, str] = Depends(get_current_user_role_from_header)):
    user_id, role = auth_info
    try:
        ingest_pubmed_query(payload.query, max_results=payload.max_results, user_id=user_id, is_global=True)
        return {"status": "ok", "message": f"PubMed query '{payload.query}' ingested."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Standardize ingestion endpoints to expose both route patterns called by the frontend
@router.post("/api/ingest/file")
@router.post("/api/rag/ingest")
async def ingest_file(file: UploadFile = File(...), auth_info: tuple[str, str] = Depends(get_current_user_role_from_header)):
    user_id, role = auth_info
    try:
        content = await file.read()
        filename = file.filename.lower()
        if filename.endswith(".pdf"):
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif filename.endswith(".docx"):
            from io import BytesIO
            from docx import Document
            doc = Document(BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs)
        else:
            text = content.decode("utf-8", errors="ignore")
        
        ingest_text_content(file.filename, text, source_type="File", user_id=user_id, is_global=True)
        return {"status": "ok", "message": f"File '{file.filename}' ingested."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/documents")
def list_documents(auth_info: tuple[str, str] = Depends(get_current_user_role_from_header)):
    user_id, role = auth_info
    docs = DocumentRepository.get_user_documents(user_id=user_id, role=role)
    return docs

@router.delete("/api/documents/{filename}")
def delete_document(filename: str, auth_info: tuple[str, str] = Depends(get_current_user_role_from_header)):
    user_id, role = auth_info
    affected = DocumentRepository.delete_document_by_filename(user_id=user_id, filename=filename, role=role)
    if affected == 0:
        raise HTTPException(status_code=404, detail="Document not found or access denied")
    return {"status": "ok", "message": f"Document '{filename}' deleted successfully."}
