import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "database")

client = chromadb.PersistentClient(path=DB_PATH)
collections = client.list_collections()
print("Collections:", [c.name for c in collections])

if collections:
    for c in collections:
        collection = client.get_collection(name=c.name)
        count = collection.count()
        print(f"Collection '{c.name}' has {count} documents.")
