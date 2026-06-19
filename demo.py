import chromadb
from text2vec import SentenceModel
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

class RAG:
    def __init__(self, model_path: str, google_api_key: str,db_path,collection_name):
        print("Loading embedding model...")
        self.embedding_model = SentenceModel(model_path)
        print("Connecting to Gemini API...")
        self.client = genai.Client(api_key=google_api_key)
        print("Opening vector database...")
        chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = chroma_client.get_or_create_collection(name=collection_name)
        print("RAG system ready!\n")
        
    def ask(self, question: str, top_k: int = 3):
        import sqlite3
        import numpy as np

        print(f"\nSearching database for: '{question}'")

        # 1. Dense Vector Search (Retrieve Top 10)
        query_vector = self.embedding_model.encode([question])
        query_vector_list = [vec.tolist() for vec in query_vector]

        results = self.collection.query(
            query_embeddings=query_vector_list,
            n_results=10
        )
        dense_documents = results["documents"][0] if results.get("documents") else []
        dense_ids = results["ids"][0] if results.get("ids") else []

        # 2. SQLite FTS5 Keyword Search (Retrieve Top 10)
        keyword_results = []
        try:
            # Use same db_path
            db_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database", "db.sqlite3")
            if not os.path.exists(db_file):
                # Fallback to current directory database
                db_file = os.path.join("database", "db.sqlite3")
            
            if os.path.exists(db_file):
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                # Clean special characters to avoid FTS syntax errors
                cleaned_query = "".join(c if c.isalnum() or c.isspace() else " " for c in question).strip()
                if cleaned_query:
                    # Parse into words and join with OR or AND
                    fts_query = " OR ".join(cleaned_query.split())
                    cursor.execute(
                        "SELECT chunk_id, content FROM document_chunks_fts WHERE content MATCH ? LIMIT 10",
                        (fts_query,)
                    )
                    keyword_results = cursor.fetchall()
                conn.close()
        except Exception as e:
            print(f"Warning: Keyword search failed: {e}")

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_k = 60
        scores = {}
        content_map = {}

        # Rank dense results
        for rank, (chunk_id, doc_text) in enumerate(zip(dense_ids, dense_documents), 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))
            content_map[chunk_id] = doc_text

        # Rank keyword results
        for rank, (chunk_id, doc_text) in enumerate(keyword_results, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))
            content_map[chunk_id] = doc_text

        # 4. Reranking / Selecting Top Candidates
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        candidate_ids = [chunk_id for chunk_id, score in sorted_chunks[:top_k * 2]]
        
        # Calculate final cosine similarity reranking for top candidates
        reranked_results = []
        if candidate_ids:
            candidate_texts = [content_map[cid] for cid in candidate_ids]
            candidate_embeddings = self.embedding_model.encode(candidate_texts)
            
            q_vec = query_vector[0]
            q_norm = np.linalg.norm(q_vec)
            
            for cid, text, emb in zip(candidate_ids, candidate_texts, candidate_embeddings):
                emb_norm = np.linalg.norm(emb)
                sim = float(np.dot(q_vec, emb) / (q_norm * emb_norm)) if q_norm > 0 and emb_norm > 0 else 0.0
                reranked_results.append((text, sim))
                
            # Sort by cosine similarity descending
            reranked_results.sort(key=lambda x: x[1], reverse=True)

        # Final top_k chunks
        final_candidates = reranked_results[:top_k]
        documents = [item[0] for item in final_candidates]
        distances = [1.0 - item[1] for item in final_candidates] # Convert cosine similarity to distance

        print(f"Found {len(documents)} relevant documents (fused BM25 + Vector)")

        # Sort documents in prompt context to place most relevant at the end
        # (reduces "lost in the middle" LLM bias)
        context_docs = list(reversed(documents))

        context = "\n\n".join(
            [f"Document {i+1}: {doc}" for i, doc in enumerate(context_docs)]
        )

        prompt = f"""Answer the question based on the provided context. 
If the answer is not in the context, say so clearly.
The context can be in any language but answer in same language as the question only.
Context from database:
{context}

Question: {question}

Answer:
"""

        print("Generating answer with Gemini AI...")

        import time
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=prompt
                )
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    print(f"Rate limited. Waiting 20 seconds before retrying... (Attempt {attempt+1}/3)")
                    time.sleep(20)
                else:
                    raise e

        return {
            "answer": response.text,
            "sources": documents,
            "distances": distances
        }


def main():
    print("=" * 70)
    print("RAG SYSTEM - Question Answering with Your Vector Database")
    print("=" * 70)

    #setx GOOGLE_API_KEY API_KEY 
    MODEL_PATH = "BAAI/bge-m3"
    DB_PATH = "database"
    COLLECTION_NAME = "20230915"

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    MODEL_PATH = os.getenv("MODEL_PATH", "BAAI/bge-m3")
    DB_PATH = os.getenv("DB_PATH", "database")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "20230915")

    rag = RAG(
        model_path=MODEL_PATH,
        google_api_key=GOOGLE_API_KEY,
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME
    )
    print("\nYou can now ask questions! (Type 'quit' or 'exit' to stop)\n")
    while True:
        question = input("Your Question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        if not question:
            print("Please enter a question!\n")
            continue
        try:
            result = rag.ask(question=question, top_k=3)
            print("\n" + "=" * 70)
            print("ANSWER:")
            print("-" * 70)
            # Remove \r which causes PowerShell to overwrite lines
            clean_answer = str(result['answer']).replace('\r', '')
            print(clean_answer)
            print("=" * 70)
            print("\nSOURCES (from your database):")
            print("-" * 70)
            for i, (doc, dist) in enumerate(zip(result['sources'], result['distances']), 1):
                clean_doc = str(doc).replace('\n', ' ').replace('\r', '')
                print(f"{i}. [L2 Distance: {dist:.2f}] {clean_doc[:100]}...")
            print("=" * 70 + "\n")
        except Exception as e:
            print(f"\nError: {str(e)}\n")


if __name__ == "__main__":
    main()
