# backend/src/modules/auth/api.py
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional, Dict, Any
from backend.src.core.security import get_password_hash, verify_password, create_access_token
from backend.src.database.repositories.user_repository import UserRepository

router = APIRouter(prefix="/api/auth", tags=["auth"])

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    role: Optional[str] = "user"

class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/register")
def register(payload: RegisterRequest):
    existing = UserRepository.get_by_email(payload.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    existing_username = UserRepository.get_by_username(payload.username)
    if existing_username:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed = get_password_hash(payload.password)
    user = UserRepository.create_user(
        username=payload.username,
        email=payload.email,
        hashed_password=hashed,
        role=payload.role or "user"
    )
    return {"status": "ok", "message": "User registered successfully", "user_id": user.id}

@router.post("/login")
def login(payload: LoginRequest):
    user = UserRepository.get_by_email(payload.email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    UserRepository.update_last_login(user.id)
    token = create_access_token(subject=user.id, extra={"role": user.role, "email": user.email})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role
        }
    }
