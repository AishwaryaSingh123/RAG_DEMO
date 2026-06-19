import os
import re
import chromadb
import wikipedia
import arxiv
from dotenv import load_dotenv
from text2vec import SentenceModel
from Bio import Entrez

# Load environment configurations
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "BAAI/bge-m3")
DB_PATH = os.getenv("DB_PATH", "database")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "research_rag")

# Configure PubMed Entrez email (NCBI requirement)
Entrez.email = os.getenv("PUBMED_EMAIL", "masterblaster5132@gmail.com")

# Global variables for models and clients (initialized lazily)
_embedding_model = None
_chroma_client = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model: {MODEL_PATH}...")
        _embedding_model = SentenceModel(MODEL_PATH)
    return _embedding_model

def get_chroma_collection(collection_name=COLLECTION_NAME):
    global _chroma_client
    if _chroma_client is None:
        print(f"Opening ChromaDB persistent store at: {DB_PATH}...")
        _chroma_client = chromadb.PersistentClient(path=DB_PATH)
    return _chroma_client.get_or_create_collection(name=collection_name)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Sentence-aware paragraph chunking targeting ~500 characters.
    Ensures that content boundaries are semantically meaningful.
    """
    # Clean up excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []

    # Split into rough sentences using simple regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_length + sentence_len > chunk_size and current_chunk:
            # Save the current chunk
            chunks.append(" ".join(current_chunk))
            
            # Keep some sentences for overlap
            overlap_count = 0
            overlap_chunk = []
            # Gather sentences from back to front for overlap
            for prev_sentence in reversed(current_chunk):
                if overlap_count + len(prev_sentence) < overlap:
                    overlap_chunk.insert(0, prev_sentence)
                    overlap_count += len(prev_sentence)
                else:
                    break
            
            current_chunk = overlap_chunk
            current_length = sum(len(s) for s in current_chunk) + len(current_chunk)
            
        current_chunk.append(sentence)
        current_length += sentence_len + 1 # +1 for join space
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return [c.strip() for c in chunks if len(c.strip()) > 15]


def add_chunks_to_db(chunks: list[str], source_name: str, source_type: str, user_id: str = "system", is_global: bool = True):
    """
    Computes vector embeddings for a list of text chunks and inserts them into ChromaDB.
    Prefixes documents with '[SourceType: SourceName]' to easily surface citation in the frontend.
    """
    if not chunks:
        print(f"No valid chunks for {source_name} to ingest.")
        return 0
        
    collection = get_chroma_collection()
    model = get_embedding_model()
    
    # Prefix chunks to inject metadata into the raw text (100% robust retrieval representation)
    prefixed_docs = []
    ids = []
    
    # Generate unique IDs based on hashes or sequence numbers
    base_id = re.sub(r'[^a-zA-Z0-9_]', '_', f"{source_type}_{source_name}").lower()
    
    for idx, chunk in enumerate(chunks):
        prefixed_docs.append(f"[{source_type.upper()}: {source_name}] {chunk}")
        ids.append(f"{base_id}_chunk_{idx}_{os.urandom(2).hex()}")
        
    print(f"Encoding {len(prefixed_docs)} chunks for '{source_name}'...")
    embeddings = model.encode(prefixed_docs)
    embeddings_list = [vec.tolist() for vec in embeddings]
    
    print(f"Storing chunks in ChromaDB...")
    collection.add(
        documents=prefixed_docs,
        embeddings=embeddings_list,
        ids=ids,
        metadatas=[{"source": source_name, "type": source_type, "user_id": user_id, "is_global": is_global}] * len(prefixed_docs)
    )
    
    # Store chunks and keywords in SQLite FTS5
    import sqlite3
    db_file = os.path.join(DB_PATH, "db.sqlite3")
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Ensure documents table exists (run init if needed)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source_type TEXT NOT NULL,
                kb_id TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL
            )
        """)
        cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(chunk_id UNINDEXED, content)")
        
        # Get or create document reference
        cursor.execute("SELECT id FROM documents WHERE filename = ?", (source_name,))
        row = cursor.fetchone()
        if row:
            doc_id = row[0]
        else:
            import uuid
            doc_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO documents (id, user_id, filename, source_type) VALUES (?, ?, ?, ?)",
                (doc_id, user_id, source_name, source_type)
            )
        
        # Write chunks to SQLite and FTS5
        for chunk_id, chunk_text in zip(ids, prefixed_docs):
            cursor.execute(
                "INSERT OR REPLACE INTO document_chunks (id, document_id, user_id, content) VALUES (?, ?, ?, ?)",
                (chunk_id, doc_id, user_id, chunk_text)
            )
            cursor.execute(
                "INSERT OR REPLACE INTO document_chunks_fts (chunk_id, content) VALUES (?, ?)",
                (chunk_id, chunk_text)
            )
        conn.commit()
        conn.close()
        print(f"Indexed {len(prefixed_docs)} chunks in SQLite FTS5.")
    except Exception as sqle:
        print(f"Warning: Failed to index in SQLite FTS5: {sqle}")

    print(f"Ingested '{source_name}' successfully ({len(prefixed_docs)} chunks).")
    return len(prefixed_docs)


def ingest_wikipedia_topic(topic: str, user_id: str = "system", is_global: bool = True) -> int:
    """
    Scrapes Wikipedia for a topic, chunks its content, and populates ChromaDB.
    """
    print(f"\n[WIKIPEDIA INGESTION] Fetching topic: '{topic}'...")
    try:
        # Search page first to get exact title
        search_results = wikipedia.search(topic)
        if not search_results:
            raise ValueError(f"No Wikipedia pages matched '{topic}'")
            
        page_title = search_results[0]
        page = wikipedia.page(page_title, auto_suggest=False)
        content = page.content
        
        # Remove markdown/formatting sections like 'References' or 'External links'
        content = re.split(r'==\s*(See also|References|External links|Notes)\s*==', content)[0]
        
        chunks = chunk_text(content)
        return add_chunks_to_db(chunks, page_title, "Wikipedia", user_id=user_id, is_global=is_global)
    except Exception as e:
        print(f"Wikipedia Ingestion Error for '{topic}': {e}")
        raise e


def ingest_arxiv_query(query: str, max_results: int = 5, user_id: str = "system", is_global: bool = True) -> int:
    """
    Queries arXiv for research abstracts, chunks summaries, and populates ChromaDB.
    """
    print(f"\n[ARXIV INGESTION] Searching query: '{query}' (Max results: {max_results})...")
    total_ingested = 0
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        client = arxiv.Client()
        results = list(client.results(search))
        
        if not results:
            print(f"No arXiv articles found for: '{query}'")
            return 0
            
        for paper in results:
            # Combine paper metadata and abstract text
            full_text = f"Title: {paper.title}. Authors: {', '.join(auth.name for auth in paper.authors)}. Published: {paper.published.strftime('%Y-%m')}. Summary: {paper.summary}"
            chunks = chunk_text(full_text, chunk_size=600)
            total_ingested += add_chunks_to_db(chunks, paper.title, "arXiv", user_id=user_id, is_global=is_global)
            
        return total_ingested
    except Exception as e:
        print(f"arXiv Ingestion Error for query '{query}': {e}")
        raise e


def ingest_pubmed_query(query: str, max_results: int = 5, user_id: str = "system", is_global: bool = True) -> int:
    """
    Queries PubMed for medical papers, chunks abstracts, and populates ChromaDB.
    """
    print(f"\n[PUBMED INGESTION] Searching query: '{query}' (Max results: {max_results})...")
    total_ingested = 0
    try:
        # Step 1: Search PubMed for IDs matching query
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record.get("IdList", [])
        if not id_list:
            print(f"No PubMed documents found for: '{query}'")
            return 0
            
        print(f"Found {len(id_list)} PubMed IDs. Fetching details...")
        
        # Step 2: Fetch details for these IDs
        handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="medline", retmode="xml")
        xml_data = handle.read()
        handle.close()
        
        # Parse XML to extract titles and abstracts
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_data)
        
        for article in root.findall(".//PubmedArticle"):
            title_node = article.find(".//ArticleTitle")
            title = title_node.text if title_node is not None else "Unknown PubMed Title"
            
            abstract_texts = []
            for abstract_node in article.findall(".//AbstractText"):
                if abstract_node.text:
                    abstract_texts.append(abstract_node.text)
                    
            abstract = " ".join(abstract_texts) if abstract_texts else ""
            if not abstract:
                continue
                
            full_text = f"Title: {title}. Abstract: {abstract}"
            chunks = chunk_text(full_text, chunk_size=600)
            total_ingested += add_chunks_to_db(chunks, title[:80], "PubMed", user_id=user_id, is_global=is_global)
            
        return total_ingested
    except Exception as e:
        print(f"PubMed Ingestion Error for query '{query}': {e}")
        raise e


def ingest_text_content(name: str, content: str, source_type: str = "File", user_id: str = "system", is_global: bool = True) -> int:
    """
    Ingests arbitrary text content (e.g. from uploaded files).
    """
    print(f"\n[FILE INGESTION] Ingesting '{name}' as source {source_type}...")
    try:
        chunks = chunk_text(content)
        return add_chunks_to_db(chunks, name, source_type, user_id=user_id, is_global=is_global)
    except Exception as e:
        print(f"File Ingestion Error for '{name}': {e}")
        raise e


def populate_default():
    """
    Standard run to quickly bootstrap the vector database with interesting multi-domain contexts.
    """
    print("=" * 70)
    print("RUNNING DEFAULT VECTOR DATABASE POPULATION")
    print("=" * 70)
    
    # Let's seed Wikipedia topics
    wiki_topics = ["Thyroid cancer", "Retrieval-Augmented Generation"]
    for topic in wiki_topics:
        try:
            ingest_wikipedia_topic(topic)
        except Exception:
            pass
            
    # Seed arXiv research papers
    arxiv_queries = ["artificial intelligence in healthcare"]
    for q in arxiv_queries:
        try:
            ingest_arxiv_query(q, max_results=2)
        except Exception:
            pass
            
    # Seed PubMed medical reports
    pubmed_queries = ["thyroid nodule classification"]
    for q in pubmed_queries:
        try:
            ingest_pubmed_query(q, max_results=2)
        except Exception:
            pass
            
    print("\n" + "=" * 70)
    print("DEFAULT DATABASE POPULATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    populate_default()
