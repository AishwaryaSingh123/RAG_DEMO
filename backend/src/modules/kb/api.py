# backend/src/modules/kb/api.py
import uuid
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from backend.src.core.security import decode_access_token
from backend.src.database.connection import get_db_connection

router = APIRouter(prefix="/api/kb", tags=["kb"])

class KbCreateRequest(BaseModel):
    name: str
    description: Optional[str] = ""

def get_current_user_role_from_header(authorization: Optional[str] = Header(None)) -> tuple[str, str]:
    if not authorization or not authorization.startswith("Bearer "):
        return "system", "admin"
    token = authorization.split(" ")[1]
    try:
        payload = decode_access_token(token)
        return payload.get("sub", "system"), payload.get("role", "user")
    except Exception:
        return "system", "admin"

@router.get("")
def list_kbs(auth_info: tuple[str, str] = Depends(get_current_user_role_from_header)):
    user_id, role = auth_info
    conn = get_db_connection()
    cursor = conn.cursor()
    # Simple logic: Admin sees all, standard users see everything or owned
    if role == "admin":
        cursor.execute("SELECT id, name, description, owner_id, status, visibility, created_at FROM knowledge_bases")
    else:
        cursor.execute(
            "SELECT id, name, description, owner_id, status, visibility, created_at FROM knowledge_bases WHERE owner_id = ? OR visibility = 'OPEN'",
            (user_id,)
        )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

@router.post("")
def create_kb(payload: KbCreateRequest, auth_info: tuple[str, str] = Depends(get_current_user_role_from_header)):
    user_id, role = auth_info
    kb_id = str(uuid.uuid4())
    conn = get_db_connection()
    cursor = conn.cursor()
    # Ensure a project exists to satisfy foreign key (or insert a default one)
    cursor.execute("SELECT id FROM projects LIMIT 1")
    proj = cursor.fetchone()
    if proj:
        project_id = proj[0]
    else:
        project_id = "default_project"
        cursor.execute(
            "INSERT OR IGNORE INTO projects (id, name, description, owner_id) VALUES (?, ?, ?, ?)",
            (project_id, "Default Project", "Default workspace project", user_id)
        )
    
    cursor.execute(
        "INSERT INTO knowledge_bases (id, project_id, name, description, owner_id, status, visibility) VALUES (?, ?, ?, ?, ?, 'OPEN', 'OPEN')",
        (kb_id, project_id, payload.name, payload.description, user_id)
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "message": "Knowledge base created successfully", "id": kb_id}
