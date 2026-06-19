# backend/src/core/security.py
"""Security utilities for the RAG backend.

Provides password hashing, verification, and JWT token handling.
All helpers rely on the centralized Settings from config.py.
"""

import datetime
from typing import Any, Dict, Optional

import jwt
from passlib.context import CryptContext

from .config import settings

# ---------------------------------------------------------------------------
# Password hashing utilities
# ---------------------------------------------------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    """Return a bcrypt hash for the given plaintext password."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a stored hash."""
    return pwd_context.verify(plain_password, hashed_password)

# ---------------------------------------------------------------------------
# JWT token utilities
# ---------------------------------------------------------------------------

def _create_token(data: Dict[str, Any], expires_delta: Optional[datetime.timedelta] = None) -> str:
    """Encode a JWT token.

    Args:
        data: Payload to encode. Must be JSON serializable.
        expires_delta: Optional expiry timedelta. If omitted, defaults to
            SETTINGS.ACCESS_TOKEN_EXPIRE_MINUTES.
    """
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + (
        expires_delta
        or datetime.timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def create_access_token(subject: str, extra: Optional[Dict[str, Any]] = None) -> str:
    """Create an access token for a given user identifier.

    Args:
        subject: Usually the user ID.
        extra: Additional claims to embed.
    """
    payload = {"sub": subject}
    if extra:
        payload.update(extra)
    return _create_token(payload)


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode a JWT token and return its payload.

    Raises:
        jwt.ExpiredSignatureError: If the token has expired.
        jwt.InvalidTokenError: For any other validation problem.
    """
    return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.ALGORITHM])

# ---------------------------------------------------------------------------
# Helper for extracting user ID from token (used in dependencies)
# ---------------------------------------------------------------------------

def get_current_user_id(token: str) -> str:
    """Utility to extract the ``sub`` claim from a verified token.
    """
    payload = decode_access_token(token)
    return payload.get("sub")

