"""Merged RAG pipeline for web + local PDF ingestion with shared embeddings and conversation memory.

This script intentionally keeps the current web ingestion flow and adds:
- local PDF ingestion using process_all_pdfs()
- one shared EmbeddingManager
- one shared VectorStore collection
- conversation memory / chat helpers for multiple users and conversations
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import chromadb
import requests
from dotenv import load_dotenv
from pypdf import PdfReader
from urllib.parse import quote

try:
    from text2vec import SentenceModel
except Exception:  # pragma: no cover - fallback for environments without text2vec
    SentenceModel = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

try:
    from google import genai
except Exception:  # pragma: no cover
    genai = None

# Existing web ingestion helpers (kept intact for backward compatibility)
from populate_db import (
    ingest_arxiv_query,
    ingest_pubmed_query,
    ingest_wikipedia_topic,
)

load_dotenv()


class EmbeddingManager:
    """Lightweight wrapper around the embedding model used by both ingestion and retrieval."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.getenv("MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")
        print(f"[Embedding] Loading model: {self.model_path}")

        if SentenceModel is not None:
            self.model = SentenceModel(self.model_path)
        elif SentenceTransformer is not None:
            self.model = SentenceTransformer(self.model_path)
        else:
            raise RuntimeError("No supported embedding backend found. Install 'text2vec' or 'sentence-transformers'.")

    def embed(self, texts: List[str]) -> List[List[float]]:
        print(f"[Embedding] Encoding {len(texts)} text(s)...")
        embeddings = self.model.encode(texts)
        return [vec.tolist() for vec in embeddings]


class VectorStore:
    """Persistent ChromaDB wrapper with one shared collection for web + local documents."""

    def __init__(self, db_path: Optional[str] = None, collection_name: Optional[str] = None):
        self.db_path = db_path or os.getenv("DB_PATH", "database")
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "research_rag")
        print(f"[VectorStore] Opening collection '{self.collection_name}' at '{self.db_path}'")
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_documents(self, docs: List[str], metadatas: List[Dict[str, str]], ids: List[str]) -> int:
        if not docs:
            return 0
        print(f"[VectorStore] Adding {len(docs)} chunk(s) to '{self.collection_name}'")
        embeddings = embedding_manager.embed(docs)
        self.collection.add(documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids)
        return len(docs)

    def query(self, text: str, top_k: int = 5) -> Dict[str, List[object]]:
        print(f"[VectorStore] Retrieving top {top_k} chunks for query: {text[:80]!r}")
        query_embedding = embedding_manager.embed([text])
        return self.collection.query(query_embeddings=query_embedding, n_results=top_k)


@dataclass
class Document:
    text: str
    source_url: Optional[str] = None
    source_type: str = "web"
    title: Optional[str] = None


class ConversationMemory:
    """Simple in-memory conversation memory with optional summary text for multi-user support."""

    def __init__(self):
        self._messages: Dict[str, List[Dict[str, str]]] = {}
        self._summaries: Dict[str, str] = {}

    def add_message(self, conv_id: str, role: str, content: str) -> None:
        self._messages.setdefault(conv_id, []).append({"role": role, "content": content})

    def get_recent(self, conv_id: str, limit: int = 6) -> List[Dict[str, str]]:
        return self._messages.get(conv_id, [])[-limit:]

    def get_summary(self, conv_id: str) -> str:
        return self._summaries.get(conv_id, "")

    def update_summary(self, conv_id: str, summary: str) -> None:
        self._summaries[conv_id] = summary

    def list_conversations(self) -> List[Dict[str, object]]:
        return [
            {"conversation_id": conv_id, "message_count": len(messages), "summary": self._summaries.get(conv_id, "")}
            for conv_id, messages in self._messages.items()
        ]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Sentence-aware chunking for long PDF / web text."""
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_len + sentence_len > chunk_size and current_chunk:
            chunk = ' '.join(current_chunk).strip()
            if chunk:
                chunks.append(chunk)

            overlap_chunk = []
            overlap_len = 0
            for prev in reversed(current_chunk):
                if overlap_len + len(prev) < overlap:
                    overlap_chunk.insert(0, prev)
                    overlap_len += len(prev)
                else:
                    break
            current_chunk = overlap_chunk
            current_len = sum(len(x) for x in current_chunk)

        current_chunk.append(sentence)
        current_len += sentence_len + 1

    if current_chunk:
        chunk = ' '.join(current_chunk).strip()
        if chunk:
            chunks.append(chunk)

    return [c for c in chunks if len(c) > 25]


def safe_pdf_text(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(p for p in pages if p).strip()
    except Exception as exc:
        print(f"[PDF] Failed to read {path}: {exc}")
        return ""


def process_all_pdfs(pdf_dir: str = "pdfs") -> int:
    """Ingest every local PDF file under a directory into the same vector store."""
    root = Path(pdf_dir)
    if not root.exists():
        print(f"[PDF] No local PDF directory found at {pdf_dir}")
        return 0

    total = 0
    for path in sorted(root.rglob("*.pdf")):
        text = safe_pdf_text(path)
        if not text:
            continue
        chunks = chunk_text(text, chunk_size=700, overlap=80)
        total += add_chunks_to_store(chunks, source_file=str(path), source_type="local_pdf")
    return total


def add_chunks_to_store(chunks: List[str], source_file: Optional[str] = None, source_url: Optional[str] = None,
                        source_type: str = "web") -> int:
    """Create embeddings and store chunks in the shared Chroma collection."""
    if not chunks:
        return 0

    metadata = []
    docs = []
    ids = []
    base = (source_file or source_url or "source").replace("/", "_").replace("\\", "_")

    for idx, chunk in enumerate(chunks):
        docs.append(chunk)
        ids.append(f"{base}_{idx}_{os.urandom(2).hex()}")
        metadata_entry = {"source_type": source_type}
        if source_file:
            metadata_entry["source_file"] = source_file
        if source_url:
            metadata_entry["source_url"] = source_url
        metadata.append(metadata_entry)

    vector_store.add_documents(docs, metadata, ids)
    print(f"[Ingest] Stored {len(docs)} chunk(s) from {source_file or source_url}")
    return len(docs)


def fetch_wikipedia_summary(topic: str) -> Optional[Document]:
    """Fetch a Wikipedia summary from the official REST API and return clean text."""
    encoded_title = quote(topic.strip().replace(' ', '_'))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"
    print(f"[Web] Fetching Wikipedia summary for '{topic}' from {url}")

    headers = {"User-Agent": "RAGDemo/1.0 (contact: dev@example.com)"}
    try:
        response = requests.get(url, headers=headers, timeout=30)
    except requests.RequestException as exc:
        print(f"[Web] Wikipedia request failed for '{topic}': {exc}")
        return None

    if response.status_code != 200:
        print(f"[Web] Wikipedia request returned status {response.status_code} for '{topic}'; skipping.")
        return None

    try:
        payload = response.json()
    except ValueError as exc:
        print(f"[Web] Wikipedia response was not valid JSON for '{topic}': {exc}")
        return None

    if not isinstance(payload, dict):
        print(f"[Web] Wikipedia response payload was not an object for '{topic}'; skipping.")
        return None

    extract = (payload.get("extract") or "").strip()
    title = (payload.get("title") or topic).strip()
    content_urls = payload.get("content_urls") if isinstance(payload.get("content_urls"), dict) else {}
    source_url = (content_urls.get("desktop") or {}).get("page") if isinstance(content_urls.get("desktop"), dict) else None
    if not extract:
        print(f"[Web] Wikipedia summary missing 'extract' for '{topic}'; skipping.")
        return None

    cleaned = re.sub(r'\s+', ' ', extract).strip()
    return Document(text=f"[WIKIPEDIA: {title}] {cleaned}", source_url=source_url or url, source_type="web", title=title)


def ingest_web_documents(topics: List[str]) -> List[Document]:
    """Fetch web documents from the official Wikipedia REST API, chunk them, and store them."""
    documents: List[Document] = []
    for topic in topics:
        doc = fetch_wikipedia_summary(topic)
        if doc is None:
            print(f"[Web] Skipping topic '{topic}' due to invalid or empty response.")
            continue

        chunks = chunk_text(doc.text, chunk_size=700, overlap=80)
        if not chunks:
            print(f"[Web] No usable chunks generated for '{topic}'; skipping.")
            continue

        add_chunks_to_store(chunks, source_url=doc.source_url, source_type="web")
        documents.append(doc)
        print(f"[Web] Successfully ingested '{topic}' into the vector store ({len(chunks)} chunks).")

    return documents


# Lightweight wrappers around the existing web ingestion helpers so we can reuse the same chunker.
# These functions intentionally keep the current ingestion pipeline intact.

def ingest_text_from_wikipedia(topic: str) -> str:
    import wikipedia

    search_results = wikipedia.search(topic)
    if not search_results:
        raise ValueError(f"No Wikipedia pages found for '{topic}'")
    page = wikipedia.page(search_results[0], auto_suggest=False)
    content = page.content
    content = re.split(r'==\s*(See also|References|External links|Notes)\s*==', content)[0]
    return content


def ingest_text_from_arxiv(query: str) -> str:
    import arxiv

    search = arxiv.Search(query=query, max_results=2, sort_by=arxiv.SortCriterion.Relevance)
    results = list(arxiv.Client().results(search))
    pieces = []
    for paper in results:
        pieces.append(f"Title: {paper.title}. Authors: {', '.join(auth.name for auth in paper.authors)}. Summary: {paper.summary}")
    return "\n\n".join(pieces)


def ingest_text_from_pubmed(query: str) -> str:
    from Bio import Entrez

    Entrez.email = os.getenv("PUBMED_EMAIL", "masterblaster5132@gmail.com")
    handle = Entrez.esearch(db="pubmed", term=query, retmax=2)
    record = Entrez.read(handle)
    handle.close()
    ids = record.get("IdList", [])
    if not ids:
        raise ValueError(f"No PubMed results for '{query}'")

    handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="medline", retmode="xml")
    xml_data = handle.read()
    handle.close()

    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_data)
    text_parts = []
    for article in root.findall('.//PubmedArticle'):
        title = article.find('.//ArticleTitle')
        title_text = title.text if title is not None and title.text else 'Unknown title'
        abstracts = [a.text for a in article.findall('.//AbstractText') if a.text]
        if abstracts:
            text_parts.append(f"Title: {title_text}. Abstract: {' '.join(abstracts)}")
    return "\n\n".join(text_parts)


def build_prompt(context: str, summary: str, recent_messages: List[Dict[str, str]], user_query: str) -> str:
    recent = "\n".join(f"{m['role']}: {m['content']}" for m in recent_messages[-6:])
    return f"""You are a careful RAG assistant.
Use the provided context and the conversation summary to answer the user's question.
If the answer is not in the context, say so clearly.

Conversation summary:
{summary}

Recent messages:
{recent}

Relevant context:
{context}

User question:
{user_query}

Answer concisely and cite source metadata when helpful.
""".format(summary=summary or "(no summary yet)", recent=recent, context=context, user_query=user_query)


def chat(user_query: str, conv_id: str, vector_store: VectorStore, embedding_manager: EmbeddingManager,
         memory: ConversationMemory, llm_fn: Callable[[str], str], top_k: int = 5) -> str:
    """Retrieve context, build a prompt, call the LLM, and store the exchange in memory."""
    print(f"\n[Chat] conv={conv_id} query={user_query!r}")
    retrieval = vector_store.query(user_query, top_k=top_k)
    docs = retrieval.get("documents", [[]])[0]
    distances = retrieval.get("distances", [[]])[0]
    context = "\n\n---\n\n".join(docs)

    summary = memory.get_summary(conv_id)
    recent_messages = memory.get_recent(conv_id, limit=6)
    prompt = build_prompt(context=context, summary=summary, recent_messages=recent_messages, user_query=user_query)
    print("[Chat] Prompt built; calling model...")

    try:
        reply = llm_fn(prompt)
    except Exception as exc:
        print(f"[Chat] LLM call failed: {exc}")
        raise

    memory.add_message(conv_id, role="user", content=user_query)
    memory.add_message(conv_id, role="assistant", content=reply)

    # Very small summary update so multiple conversations remain coherent.
    new_summary = (summary + " " + user_query).strip()
    if len(new_summary) > 0:
        memory.update_summary(conv_id, new_summary[:500])

    print(f"[Chat] Reply generated ({len(docs)} retrieved chunks, nearest distance={distances[0] if distances else 'n/a'})")
    return reply


def make_llm_fn() -> Callable[[str], str]:
    """Use the modern google.genai client. Falls back to the older google.generativeai if needed."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing from .env")

    if genai is not None:
        client = genai.Client(api_key=api_key)

        def llm(prompt: str) -> str:
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return response.text

        return llm

    try:
        import google.generativeai as genai_legacy
        genai_legacy.configure(api_key=api_key)
        model = genai_legacy.GenerativeModel("gemini-2.5-flash")

        def llm(prompt: str) -> str:
            response = model.generate_content(prompt)
            return response.text

        return llm
    except Exception as exc:
        raise RuntimeError("Unable to initialize Gemini client. Install 'google-genai' or 'google-generativeai'.") from exc


# Global singletons used by the example below.
embedding_manager = EmbeddingManager()
vector_store = VectorStore()
memory = ConversationMemory()
llm = make_llm_fn()


if __name__ == "__main__":
    print("=" * 70)
    print("Merged RAG Pipeline - Web + Local PDF Ingestion")
    print("=" * 70)

    # 1) Seed from the existing web ingestion sources.
    web_docs = ingest_web_documents(["Thyroid cancer", "Retrieval-Augmented Generation"])
    print(f"[Main] Web ingestion added {len(web_docs)} document(s) from Wikipedia")

    # 2) Ingest local PDFs if they exist in ./pdfs.
    pdf_count = process_all_pdfs("pdfs")
    print(f"[Main] PDF ingestion added {pdf_count} chunks")

    # 3) Quick health check on the shared collection.
    print("[Main] Total documents in collection:", vector_store.collection.count())

    # 4) Example multi-turn conversation on the merged store.
    ram_conv1 = "ram_conv1"
    reply = chat(
        user_query="What is thyroid cancer?",
        conv_id=ram_conv1,
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        memory=memory,
        llm_fn=llm,
        top_k=5,
    )
    print("\nANSWER 1:\n", reply)

    reply = chat(
        user_query="What are probable causes?",
        conv_id=ram_conv1,
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        memory=memory,
        llm_fn=llm,
        top_k=5,
    )
    print("\nANSWER 2:\n", reply)

    print("\n[Main] Conversation memory entries:", memory.list_conversations())
