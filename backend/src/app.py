# backend/src/app.py
import os
import sys

# Ensure project root is in path so imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.src.database.connection import init_db
from backend.src.database.rbac_init import init_rbac_tables

# Initialize FastAPI App
app = FastAPI(title="RAG API", version="0.1.0")

# Setup CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Database tables
try:
    init_db()
    init_rbac_tables()
except Exception as e:
    import logging
    logging.getLogger("app").warning(f"Database initialization deferred or failed: {e}")

# Register modular API routers
from backend.src.modules.auth.api import router as auth_router
from backend.src.modules.kb.api import router as kb_router
from backend.src.modules.users.api import router as users_router
from backend.src.modules.rag.api import router as rag_router

app.include_router(auth_router)
app.include_router(kb_router)
app.include_router(users_router)
app.include_router(rag_router)


# ── Health & Info Endpoints ──────────────────────────────────────────
@app.get("/", tags=["system"])
async def root():
    """API landing page with version and available module summary."""
    return {
        "name": "RAG API",
        "version": "0.1.0",
        "status": "running",
        "modules": ["auth", "kb", "users", "rag"],
        "docs": "/docs",
    }


@app.get("/health", tags=["system"])
async def health_check():
    """Lightweight readiness probe."""
    return {"status": "healthy"}


# Self-executing development entrypoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.src.app:app", host="0.0.0.0", port=8000, reload=True)
