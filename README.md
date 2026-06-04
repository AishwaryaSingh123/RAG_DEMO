# RAG Research Assistant

A polished, local-first Retrieval-Augmented Generation (RAG) demo for research Q&A, built with FastAPI, ChromaDB, Gemini, and a modern Vite + TypeScript frontend. The project combines web ingestion, PDF ingestion, citation-aware retrieval, and a conversational interface into one cohesive experience.

---

## What this project does

- Lets you ask questions about local research content and retrieved web sources.
- Stores embeddings in a persistent local ChromaDB collection.
- Uses Gemini for grounded answer generation.
- Shows the top relevant source passages with similarity scores in the frontend.
- Supports ingestion from Wikipedia, arXiv, PubMed, and uploaded PDF/TXT/DOCX files.

---

## Highlights

- Local vector store with persistent embeddings
- Shared retrieval path for web + PDF content
- Conversation-aware memory support for follow-up questions
- Clean UI with multi-session chat history and source cards
- Easy setup for demos, workshops, and internal prototypes

---

## Architecture overview

```text
Frontend (Vite + TypeScript)
      │
      ▼
FastAPI backend (app.py)
      │
      ├── /api/chat        → answers questions using the merged RAG pipeline
      ├── /api/status      → reports collection size and DB status
      └── /api/ingest/*    → adds Wikipedia / arXiv / PubMed / file content
      │
      ▼
Merged RAG pipeline (merged_pipeline.py)
      ├── shared embedding manager
      ├── shared Chroma collection
      ├── chunking + retrieval
      └── Gemini answer generation
      │
      ▼
ChromaDB (local persistent store)
```

> In the current setup, the live answer path is driven by the merged retrieval pipeline in `merged_pipeline.py`, while `app.py` exposes the HTTP API consumed by the frontend.

---

## Tech stack

| Layer           | Technology                       |
| --------------- | -------------------------------- |
| LLM             | Google Gemini 2.5 Flash          |
| Embeddings      | sentence-transformers / text2vec |
| Vector Database | ChromaDB                         |
| Backend         | FastAPI + Uvicorn                |
| Frontend        | Vite + TypeScript + CSS/TS UI    |
| Environment     | python-dotenv                    |

---

## Project structure

```text
RAG_DEMO-main/
├── app.py                 # FastAPI API entry point used by the frontend
├── demo.py                # Legacy RAG flow reference
├── merged_pipeline.py     # Current merged ingestion + retrieval + chat flow
├── populate_db.py         # Data ingestion utilities
├── check_db.py            # DB inspection helper
├── database/              # Persistent ChromaDB data
├── frontend/              # Vite frontend
│   ├── index.html
│   ├── package.json
│   └── src/
│       ├── main.ts        # Chat UI, sessions, and fetch logic
│       └── style.css      # Interface styling
└── rag_env_312/           # Python environment (Windows)
```

---

## Prerequisites

- Python 3.12+
- Node.js 18+
- A Google AI Studio API key
- Internet access for Wikipedia / arXiv / PubMed ingestion and model download

---

## Quick start

### 1. Create and activate the Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 2. Install Python dependencies

```powershell
pip install -U pip
pip install fastapi uvicorn python-dotenv chromadb text2vec google-genai
pip install wikipedia arxiv biopython pypdf python-docx python-multipart
```

### 3. Configure the environment

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
MODEL_PATH=sentence-transformers/all-MiniLM-L6-v2
DB_PATH=database
COLLECTION_NAME=research_rag
```

### 4. Install frontend dependencies

```powershell
cd frontend
npm install
cd ..
```

### 5. Populate the vector store

```powershell
python populate_db.py
```

This seeds the local ChromaDB collection with example research content.

---

## Run the application

You need two terminals:

### Terminal 1 — backend

```powershell
.\.venv\Scripts\python.exe app.py
```

The backend will start on:

```text
http://127.0.0.1:8000
```

### Terminal 2 — frontend

```powershell
cd frontend
npm run dev
```

Open the Vite URL shown in the terminal, usually:

```text
http://127.0.0.1:5173/
```

---

## API overview

| Endpoint             | Method | Purpose                                                            |
| -------------------- | ------ | ------------------------------------------------------------------ |
| `/api/status`        | GET    | Returns collection name and number of stored chunks                |
| `/api/chat`          | POST   | Sends a question and receives an answer, sources, and similarities |
| `/api/ingest/wiki`   | POST   | Ingests a Wikipedia topic                                          |
| `/api/ingest/arxiv`  | POST   | Ingests arXiv results for a query                                  |
| `/api/ingest/pubmed` | POST   | Ingests PubMed results for a query                                 |
| `/api/ingest/file`   | POST   | Ingests a text/PDF/DOCX upload                                     |

Example:

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/api/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What is CRISPR?\",\"top_k\":3}"
```

---

## Ingestion options

The project supports multiple ingestion pathways:

1. Wikipedia summaries via the official REST API
2. arXiv paper summaries by query
3. PubMed abstracts by query
4. Local PDFs and text uploads

These are all designed to feed the same ChromaDB collection used by retrieval.

---

## Troubleshooting

### Backend cannot start

- Make sure your `.env` file contains `GOOGLE_API_KEY`.
- Use the project virtual environment rather than the global Python install.
- Check whether another process is using port 8000.

### Frontend says “Failed to reach the server”

- Ensure the backend is running on port 8000.
- Verify the status endpoint responds:

```powershell
curl.exe -s http://127.0.0.1:8000/api/status
```

### No answers or weak results

- Re-run `python populate_db.py` to seed the vector store.
- Confirm that the collection contains documents in `database/`.
- Try a more specific question or a different ingestion topic.

---

## Demo notes

This repository is intended as a lightweight, presentable RAG application for demos, research assistants, and prototype deployments. It is not yet a production-grade multi-user SaaS backend, but it is structured well for extension and presentation.

---

## Next steps

- Improve prompt quality and grounding rules
- Add model switching and streaming responses
- Add user authentication and persistent server-side sessions
- Expand ingestion to more document formats and external sources

The frontend uses a custom dark theme inspired by ChatGPT, Gemini, and Claude:

| Token              | Value     | Usage                               |
| ------------------ | --------- | ----------------------------------- |
| `--bg-app`         | `#0f1117` | Main background                     |
| `--bg-sidebar`     | `#171923` | Sidebar panel                       |
| `--bg-card`        | `#1e2230` | Message cards, inputs               |
| `--border`         | `#2a2f3d` | All borders and dividers            |
| `--accent`         | `#7c3aed` | Buttons, active states, focus rings |
| `--text-primary`   | `#f3f4f6` | Primary text                        |
| `--text-secondary` | `#9ca3af` | Secondary/muted text                |

---

## Notes

- The embedding model (`BAAI/bge-m3`) is downloaded automatically on first run (~1.5 GB)
- Similarity scores are displayed as `1 - distance` — higher is better
- The backend requires `python-multipart` for file upload endpoints
- Chat history is stored entirely in the browser via `localStorage`
- The frontend uses Tailwind CSS v4 with `@import "tailwindcss"` syntax

---

## License

This project is for educational and research purposes.
