# backend/src/core/telemetry.py
"""Simple telemetry layer built on top of the JSON logger.

Provides three high‑level helpers:
* record_event   – log a structured business event (e.g. user_login)
* record_metric  – log a numeric metric (e.g. latency)
* record_error   – log an exception with optional context

All telemetry is emitted through the shared logger as JSON lines, making it
cost‑free and consumable by log‑aggregation tools.
"""

from .logger import get_logger
from .exceptions import CoreError

logger = get_logger("telemetry")


def _sanitize_payload(payload: dict) -> dict:
    """Remove any sensitive fields from the payload.

    The board forbids logging passwords, JWT tokens, API keys, raw file
    contents, and document chunk text. The sanitizer drops keys that match
    these names (case‑insensitive) and replaces their values with the string
    "[REDACTED]".
    """
    redacted_keys = {
        "password",
        "pwd",
        "jwt",
        "token",
        "access_token",
        "api_key",
        "apiKey",
        "secret",
        "file_content",
        "content",
        "chunk_text",
    }
    sanitized = {}
    for k, v in payload.items():
        if isinstance(k, str) and k.lower() in redacted_keys:
            sanitized[k] = "[REDACTED]"
        else:
            sanitized[k] = v
    return sanitized


def record_event(event_name: str, payload: dict) -> None:
    """Record a business‑level event.

    Example payload for a login event::

        {"user_id": "123", "ip_address": "1.2.3.4"}
    """
    safe_payload = _sanitize_payload(payload)
    logger.info({"event": event_name, "payload": safe_payload})


def record_metric(metric_name: str, value: float) -> None:
    """Record a numeric metric (e.g. latency, request size)."""
    logger.info({"metric": metric_name, "value": value})


def record_error(error: Exception, context: dict | None = None) -> None:
    """Log an exception with optional contextual information.

    The error message itself is never stripped, but any provided context is
    passed through the same sanitizer to avoid leaking secrets.
    """
    payload = {"error": str(error)}
    if context:
        payload["context"] = _sanitize_payload(context)
    logger.error(payload)

# Export symbols for import *
__all__ = ["record_event", "record_metric", "record_error"]
