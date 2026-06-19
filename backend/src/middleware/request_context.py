# backend/src/middleware/request_context.py
"""Middleware to generate a unique request ID for each incoming HTTP request.

The ID is stored on ``request.state.request_id`` and added to the response as
the ``X-Request-ID`` header. Helper ``get_request_id()`` retrieves the current ID
via a ``contextvar``. The middleware also measures latency and emits a
``request_completed`` telemetry event via ``core.telemetry``.
"""

import uuid
import time
import contextvars
from typing import Callable

from fastapi import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from ..core.telemetry import record_event
from ..core.logger import get_logger

logger = get_logger("request_context")

# Context variable to hold the request ID for the current async task
_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")

def get_request_id() -> str | None:
    """Return the current request ID from the context variable.

    Returns ``None`` if called outside of a request context.
    """
    try:
        return _request_id_ctx.get()
    except LookupError:
        return None


class RequestContextMiddleware:
    """FastAPI ASGI middleware that injects a UUID4 request ID.

    It records request duration and emits a telemetry event ``request_completed``
    with request_id, endpoint, method, status_code, latency_ms, and optional user_id.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive=receive)
        request_id = str(uuid.uuid4())
        # Store ID in both request.state and contextvar
        request.state.request_id = request_id
        _request_id_ctx.set(request_id)
        start_time = time.time()
        status_code: int | None = None

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 200)
                headers = dict(message.get("headers", []))
                headers[b"x-request-id"] = request_id.encode()
                message["headers"] = list(headers.items())
            await send(message)

        await self.app(scope, receive, send_wrapper)
        latency_ms = int((time.time() - start_time) * 1000)
        user_id = getattr(request.state, "user_id", None)
        record_event(
            "request_completed",
            {
                "request_id": request_id,
                "endpoint": request.url.path,
                "method": request.method,
                "status_code": status_code,
                "latency_ms": latency_ms,
                "user_id": user_id,
            },
        )
        logger.info({"event": "request_started", "request_id": request_id, "endpoint": request.url.path, "method": request.method})


def add_request_context_middleware(app):
    app.add_middleware(RequestContextMiddleware)
