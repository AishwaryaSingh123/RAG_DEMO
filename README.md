# RAG Research Assistant

A production-grade Retrieval-Augmented Generation (RAG) system with a modern ChatGPT-style web interface. Ask questions about your research documents — the system retrieves the most relevant context from a local vector database and generates grounded answers using Google's Gemini AI.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Vite + TS)                  │
│         Modern dark-theme AI chat interface              │
│         http://localhost:5173                            │
└────────────────────────┬────────────────────────────────┘
                         │  fetch (JSON)
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  Backend (FastAPI)                       │
│              REST API on port 8000                      │
│         ┌──────────┬──────────┬──────────┐              │
│         │ /api/chat│/api/ingest│/api/status│             │
│         └────┬─────┴─────┬────┴─────┬────┘              │
│              │           │          │                    │
│              ▼           ▼          ▼                    │
│         ┌─────────┐ ┌────────┐ ┌────────┐              │
│         │ RAG Core│ │Ingest  │ │ChromaDB│              │
│         │(demo.py)│ │Pipeline│ │ Status │              │
│         └────┬────┘ └────┬───┘ └────────┘              │
│              │           │                              │
│         ┌────▼───────────▼────┐                         │
│         │   ChromaDB (Local)  │                         │
│         │   Vector Database   │                         │
│         └─────────────────────┘                         │
└─────────────────────────────────────────────────────────┘
```

---

## Features

### Phase 1 — RAG Core
- **Automated Data Scraping** — `populate_db.py` pulls research from Wikipedia, arXiv, and PubMed
- **Local Vector Database** — ChromaDB with `text2vec` (BAAI/bge-m3) for dense multilingual embeddings
- **AI-Powered Q&A** — Google Gemini (`gemini-3.5-flash`) generates answers grounded in your local data
- **Sentence-aware Chunking** — Intelligent text splitting with overlap for semantic consistency
- **Multilingual Support** — Questions and documents in any language; answers match the query language

### Phase 2 — API Layer & Web UI

#### FastAPI Backend (`app.py`)
| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | Returns collection name and document count |
| `/api/chat` | POST | Sends a question, returns answer + sources + similarity scores |
| `/api/ingest/wiki` | POST | Ingest a Wikipedia topic into the vector database |
| `/api/ingest/arxiv` | POST | Ingest arXiv papers by search query |
| `/api/ingest/pubmed` | POST | Ingest PubMed abstracts by search query |
| `/api/ingest/file` | POST | Upload and ingest TXT, PDF, or DOCX files |

#### Modern Chat Frontend (`frontend/`)
- **ChatGPT/Gemini-inspired UI** — Dark theme, glassmorphism, smooth animations
- **Multi-session chat history** — Create, switch, delete, and search conversations
- **Persistent storage** — All chat sessions saved to `localStorage`
- **Source citations** — Collapsible cards with similarity score badges per response
- **Empty state with suggestions** — Clickable hint chips to get started quickly
- **Typing indicator** — Animated dots while waiting for AI response
- **Auto-resizing input** — Textarea grows with content, supports Shift+Enter for newlines
- **Responsive layout** — Collapsible sidebar drawer on mobile
- **SVG icon system** — No emoji dependencies; clean vector icons throughout
- **Keyboard shortcuts** — `Enter` to send, `Shift+Enter` for new line

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Google Gemini 3.5 Flash |
| **Embeddings** | BAAI/bge-m3 (via text2vec) |
| **Vector DB** | ChromaDB (persistent, local) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Vite + TypeScript + Tailwind CSS v4 |
| **Env Config** | python-dotenv |

---

## Project Structure

```
demo/
├── app.py              # FastAPI REST API server
├── demo.py             # RAG core engine (embedding + retrieval + generation)
├── populate_db.py      # Data ingestion pipeline (Wikipedia, arXiv, PubMed)
├── check_db.py         # Utility to inspect ChromaDB document counts
├── .env                # Environment variables (not committed)
├── database/           # ChromaDB persistent storage
├── frontend/           # Vite + TypeScript web application
│   ├── index.html      # Entry HTML with Inter font
│   ├── package.json    # Node dependencies (Tailwind v4, Vite)
│   └── src/
│       ├── main.ts     # Full chat UI with multi-session logic
│       └── style.css   # Custom dark theme with CSS variables
└── rag_env_312/        # Python virtual environment
```

---

## Setup

### Prerequisites
- Python 3.12+
- Node.js 18+
- A [Google AI Studio](https://aistudio.google.com/) API key (free tier works)

### 1. Clone & create virtual environment

```bash
git clone <your-repository-url>
cd demo
python -m venv rag_env_312
.\rag_env_312\Scripts\activate       # Windows
# source rag_env_312/bin/activate    # macOS/Linux
```

### 2. Install Python dependencies

```bash
pip install chromadb text2vec google-genai python-dotenv fastapi uvicorn python-multipart
pip install wikipedia arxiv biopython pypdf python-docx
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_actual_api_key_here
MODEL_PATH=BAAI/bge-m3
DB_PATH=database
COLLECTION_NAME=research_rag
```

### 4. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 5. Populate the vector database

```bash
python populate_db.py
```

This scrapes Wikipedia, arXiv, and PubMed for research data, chunks the text, and stores embeddings in ChromaDB.

---

## Running the Application

You need **two terminals** — one for the backend, one for the frontend.

### Terminal 1 — Start the backend

```bash
.\rag_env_312\Scripts\activate
python app.py
```

The API server starts at `http://127.0.0.1:8000`.

### Terminal 2 — Start the frontend

```bash
cd frontend
npm run dev
```

The web UI opens at `http://localhost:5173` (or the next available port).

---

## Usage

### Web Interface
1. Open the frontend URL in your browser
2. Type a research question in the input box (or click a suggestion chip)
3. The AI retrieves relevant documents and generates a grounded answer
4. Expand **Source** cards below each answer to see the retrieved passages and similarity scores
5. Use the **sidebar** to create new chats, switch between sessions, search history, or delete old chats

### CLI (Terminal)
You can still use the original terminal interface:

```bash
python demo.py
```

### API (Direct)
```bash
# Ask a question
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is CRISPR?", "top_k": 3}'

# Check database status
curl http://127.0.0.1:8000/api/status

# Ingest a Wikipedia topic
curl -X POST http://127.0.0.1:8000/api/ingest/wiki \
  -H "Content-Type: application/json" \
  -d '{"topic": "CRISPR gene editing"}'
```

---

## UI Design

The frontend uses a custom dark theme inspired by ChatGPT, Gemini, and Claude:

| Token | Value | Usage |
|---|---|---|
| `--bg-app` | `#0f1117` | Main background |
| `--bg-sidebar` | `#171923` | Sidebar panel |
| `--bg-card` | `#1e2230` | Message cards, inputs |
| `--border` | `#2a2f3d` | All borders and dividers |
| `--accent` | `#7c3aed` | Buttons, active states, focus rings |
| `--text-primary` | `#f3f4f6` | Primary text |
| `--text-secondary` | `#9ca3af` | Secondary/muted text |

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
