import chromadb
from text2vec import SentenceModel
from google import genai

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

        print(f"\nSearching database for: '{question}'")

        query_vector = self.embedding_model.encode([question])
        query_vector = [vec.tolist() for vec in query_vector]

        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=top_k
        )

        documents = results["documents"][0]
        distances = results["distances"][0]

        print(f"Found {len(documents)} relevant documents")

        context = "\n\n".join(
            [f"Document {i+1}: {doc}" for i, doc in enumerate(documents)]
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

        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

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
            print(result['answer'])
            print("=" * 70)
            print("\nSOURCES (from your database):")
            print("-" * 70)
            for i, (doc, dist) in enumerate(zip(result['sources'], result['distances']), 1):
                print(f"{i}. [Similarity: {1-dist:.2%}] {doc[:100]}...")
            print("=" * 70 + "\n")
        except Exception as e:
            print(f"\nError: {str(e)}\n")


if __name__ == "__main__":
    main()
