import chromadb
from text2vec import SentenceModel
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "BAAI/bge-m3")
DB_PATH = os.getenv("DB_PATH", "database")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "20230915")

print("Loading embedding model...")
embedding_model = SentenceModel(MODEL_PATH)

print("Opening vector database...")
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

documents = [
    "Thyroid cancer is a disease that you get when abnormal cells begin to grow in your thyroid gland. The thyroid gland is shaped like a butterfly and is located in the front of your neck.",
    "RAG (Retrieval-Augmented Generation) is a technique that enhances large language models by retrieving relevant information from a knowledge base to ground the model's responses.",
    "Paris is the capital and most populous city of France.",
    "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation."
]

ids = ["doc1", "doc2", "doc3", "doc4"]

print("Generating embeddings...")
embeddings = embedding_model.encode(documents)
embeddings_list = [vec.tolist() for vec in embeddings]

print("Adding documents to ChromaDB...")
collection.add(
    documents=documents,
    embeddings=embeddings_list,
    ids=ids
)

print(f"Added {len(documents)} documents to the database.")
