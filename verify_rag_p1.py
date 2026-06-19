# verify_rag_p1.py
import os
import sys
import uuid
import sqlite3

# Ensure backend/src is in path so internal repositories can be imported
sys.path.append(os.path.abspath("backend/src"))

from demo import RAG
from backend.src.database.connection import init_db, get_db_connection
from backend.src.database.repositories.document_repository import DocumentRepository
from backend.src.database.repositories.vector_repository import search as vector_search
from populate_db import ingest_text_content

def run_tests():
    print("Initializing Database...")
    init_db()

    print("Instantiating RAG...")
    # Initialize RAG class
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    MODEL_PATH = os.getenv("MODEL_PATH", "BAAI/bge-m3")
    DB_PATH = os.getenv("DB_PATH", "database")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "research_rag")

    rag = RAG(
        model_path=MODEL_PATH,
        google_api_key=GOOGLE_API_KEY,
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME,
    )

    # Mock the Gemini API call to isolate database/retrieval verification from remote service spikes
    class MockResponse:
        text = "Verification: Optimization requires moving along the steepest descent of the loss landscape."
    rag.client.models.generate_content = lambda *args, **kwargs: MockResponse()

    print("\n" + "="*50)
    print("TEST V-1: Dense Retrieval")
    print("="*50)
    # Seed context about optimization, specifically matching gradient descent semantically
    ingest_text_content(
        name="opt_guide.txt",
        content="Optimization algorithms find parameter values that minimize cost functions. In machine learning, parameter optimization is done by iteratively taking steps in the direction of the steepest descent of the loss landscape, scaling the step size by a learning rate parameter.",
        source_type="File",
        user_id="test_user",
        is_global=True
    )
    # Query without using exact wording: "gradient descent" is not in the text, but steepest descent is.
    res_v1 = rag.ask("What is gradient descent?", top_k=1)
    print(f"Query: 'What is gradient descent?'")
    print(f"Top Chunk: {res_v1['sources']}")
    print(f"L2 Distance: {res_v1['distances']}")

    print("\n" + "="*50)
    print("TEST V-2: Sparse Retrieval")
    print("="*50)
    ingest_text_content(
        name="secrets.txt",
        content="The configuration file requires the token JWT_SECRET_KEY to sign security JSON web tokens.",
        source_type="File",
        user_id="test_user",
        is_global=True
    )
    # Query using exact keyword matching
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT chunk_id, content FROM document_chunks_fts WHERE content MATCH ? LIMIT 1",
        ("JWT_SECRET_KEY",)
    )
    fts_res = cursor.fetchone()
    conn.close()
    print(f"FTS5 Query: 'JWT_SECRET_KEY'")
    if fts_res:
        print(f"Match Chunk ID: {fts_res[0]}")
        print(f"Match Content: {fts_res[1]}")
    else:
        print("No matches found in FTS5!")

    print("\n" + "="*50)
    print("TEST V-3: Hybrid Retrieval")
    print("="*50)
    # Target chunk containing specific keyword "audits" and semantic meaning of required policy
    ingest_text_content(
        name="policy.txt",
        content="The company policy requires annual security audits to ensure compliance and identify potential system vulnerabilities.",
        source_type="File",
        user_id="test_user",
        is_global=True
    )
    res_v3 = rag.ask("What audits are required?", top_k=2)
    print(f"Query: 'What audits are required?'")
    print(f"Answer: {res_v3['answer']}")
    print(f"Sources: {res_v3['sources']}")

    print("\n" + "="*50)
    print("TEST V-4: Multi-Tenant Isolation")
    print("="*50)
    # Ingest document for User A
    ingest_text_content(
        name="alpha.txt",
        content="Secret Project Alpha focuses on building the neural retriever model.",
        source_type="File",
        user_id="user_a",
        is_global=False
    )
    # Ingest document for User B
    ingest_text_content(
        name="beta.txt",
        content="Secret Project Beta focuses on building the front-end layout.",
        source_type="File",
        user_id="user_b",
        is_global=False
    )
    # Vector Search query matching Project Beta from User A's perspective
    query_vector = rag.embedding_model.encode(["What is Project Beta?"])
    query_vector_list = [vec.tolist() for vec in query_vector]

    print("Querying User B content as User A (role=user):")
    res_a = vector_search(query_vector_list, user_id="user_a", role="user", top_k=5)
    print(f"User A retrieved docs: {res_a['documents'][0]}")
    print(f"User A retrieved metadata: {res_a['metadatas'][0]}")

    print("\nQuerying User B content as User B (role=user):")
    res_b = vector_search(query_vector_list, user_id="user_b", role="user", top_k=5)
    print(f"User B retrieved docs: {res_b['documents'][0]}")
    print(f"User B retrieved metadata: {res_b['metadatas'][0]}")

    print("\n" + "="*50)
    print("TEST V-5: Global Dataset Visibility")
    print("="*50)
    # Ingest global document
    ingest_text_content(
        name="global_lang.txt",
        content="Python is a programming language.",
        source_type="File",
        user_id="system",
        is_global=True
    )
    # Query as user_c (who has no uploads)
    query_vector_v5 = rag.embedding_model.encode(["What is Python?"])
    query_vector_v5_list = [vec.tolist() for vec in query_vector_v5]
    res_c = vector_search(query_vector_v5_list, user_id="user_c", role="user", top_k=5)
    print(f"User C retrieved global doc: {res_c['documents'][0]}")
    print(f"User C retrieved global metadata: {res_c['metadatas'][0]}")

    print("\n" + "="*50)
    print("TEST V-6: Delete Cascade Integrity")
    print("="*50)
    # Ingest document
    doc_name = "secret_archive.txt"
    ingest_text_content(
        name=doc_name,
        content="My secret document contains classified design parameters.",
        source_type="File",
        user_id="user_d",
        is_global=False
    )
    
    # 1. Inspect before state
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM documents WHERE filename = ?", (doc_name,))
    doc_row = cursor.fetchone()
    print(f"Before Delete - Documents table entry: {doc_row}")
    if doc_row:
        doc_id = doc_row[0]
        cursor.execute("SELECT id FROM document_chunks WHERE document_id = ?", (doc_id,))
        chunks_before = cursor.fetchall()
        print(f"Before Delete - document_chunks table: {chunks_before}")
        cursor.execute("SELECT chunk_id FROM document_chunks_fts")
        fts_before = cursor.fetchall()
        print(f"Before Delete - document_chunks_fts index counts: {len(fts_before)}")
        
        # 2. Execute deletion
        print("Executing Delete Cascade...")
        affected = DocumentRepository.delete_document_by_filename("user_d", doc_name, role="user")
        print(f"Affected rows in SQLite: {affected}")
        
        # 3. Inspect after state
        cursor.execute("SELECT id FROM documents WHERE filename = ?", (doc_name,))
        doc_after = cursor.fetchone()
        print(f"After Delete - Documents table entry: {doc_after}")
        cursor.execute("SELECT id FROM document_chunks WHERE document_id = ?", (doc_id,))
        chunks_after = cursor.fetchall()
        print(f"After Delete - document_chunks table: {chunks_after}")
        cursor.execute("SELECT chunk_id FROM document_chunks_fts WHERE chunk_id = ?", (chunks_before[0][0],))
        fts_after = cursor.fetchone()
        print(f"After Delete - document_chunks_fts match: {fts_after}")
        
    conn.close()

if __name__ == "__main__":
    run_tests()
