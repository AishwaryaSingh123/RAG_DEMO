

# RAG Question Answering System

A Retrieval-Augmented Generation (RAG) system that answers questions by searching a local vector database and generating responses using Google's Gemini AI.

## How It Works

```
Your Question
     │
     ▼
Convert to vector embedding (BAAI/bge-m3)
     │
     ▼
Search ChromaDB for similar documents
     │
     ▼
Inject top-K results into prompt
     │
     ▼
Gemini AI generates a grounded answer
     │
     ▼
Answer + Sources returned
```

1. **Embed the query** — the question is converted into a vector using a local sentence embedding model (`BAAI/bge-m3` via `text2vec`)
2. **Retrieve context** — ChromaDB is searched for the most semantically similar documents
3. **Generate an answer** — the retrieved documents are injected into a prompt sent to Gemini 2.0 Flash, which produces a grounded answer
4. If the answer isn't found in the retrieved context, the model says so explicitly

## Components

| Component | Role |
|---|---|
| `text2vec` + `BAAI/bge-m3` | Local multilingual sentence embeddings |
| `ChromaDB` | Persistent local vector database |
| `Google Gemini 2.0 Flash` | LLM for answer generation |

## Setup

### 1. Install dependencies

```bash
pip install chromadb text2vec google-genai
```

### 2. Set your Google API key

**Windows:**
```cmd
setx GOOGLE_API_KEY your_api_key_here
```

**macOS/Linux:**
```bash
export GOOGLE_API_KEY=your_api_key_here
```

### 3. Prepare your vector database

Populate a ChromaDB collection at the path and collection name defined in `main()` before running. The system queries an **existing** collection — it does not ingest documents itself.

### 4. Run

```bash
python rag.py
```

## Configuration

Edit these constants in `main()` to match your setup:

```python
MODEL_PATH      = "BAAI/bge-m3"    # Embedding model (auto-downloaded on first run)
DB_PATH         = "database"        # Folder containing your ChromaDB data
COLLECTION_NAME = "20230915"        # Name of the ChromaDB collection to query
```

## Usage

Once running, type any question at the prompt:

```
Your Question: What is the refund policy?

======================================================================
ANSWER:
----------------------------------------------------------------------
According to the documents, refunds are processed within 5–7 business days...
======================================================================

SOURCES (from your database):
----------------------------------------------------------------------
1. [Similarity: 92.34%] Refunds are issued to the original payment method within 5...
2. [Similarity: 87.11%] Customers may request a refund within 30 days of purchase...
3. [Similarity: 81.05%] For damaged goods, contact support@example.com with your ord...
======================================================================
```

Type `quit`, `exit`, or `q` to stop.

## Multilingual Support

The system supports questions and documents in any language. Answers are always returned **in the same language as the question**.

## Notes

- The embedding model is downloaded automatically on first use
- The vector database must be pre-populated separately — this script is query-only
- Similarity scores are displayed as `1 - distance`, so higher is better
=======
# Research RAG System

This project is a Retrieval-Augmented Generation (RAG) system built in Python. It allows you to build a local vector database with research data from Wikipedia, ArXiv, and PubMed, and then uses the Google Gemini AI to answer questions based strictly on that local context.

## Features

- **Automated Data Scraping:** `populate_db.py` pulls recent and relevant research from Wikipedia, ArXiv, and PubMed.
- **Local Vector Database:** Uses `ChromaDB` to store the document chunks and `text2vec` (BAAI/bge-m3) for dense vector embeddings.
- **AI-Powered Q&A:** `demo.py` provides an interactive terminal to chat with the Google Gemini AI (`gemini-3.5-flash`), completely grounded in your local database.

## Prerequisites

- Python 3.12 (or higher)
- A Google AI Studio API Key (Free tier works!)

## Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <repository-folder>
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv rag_env
   .\rag_env\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install chromadb text2vec google-genai python-dotenv wikipedia arxiv biopython
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory (do not commit this file to GitHub) and add the following variables:
   ```env
   GOOGLE_API_KEY=your_actual_api_key_here
   MODEL_PATH=BAAI/bge-m3
   DB_PATH=database
   COLLECTION_NAME=research_rag
   ```

## Usage

**1. Populate the Database:**
Before asking questions, you need to populate your vector database with the research documents.
```bash
python populate_db.py
```

**2. Chat with the AI:**
Once the database is populated, launch the interactive query script.
```bash
python demo.py
```

Type your questions into the prompt. The AI will retrieve the most relevant chunks from your database and generate a comprehensive answer!

