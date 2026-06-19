# backend/src/core/exceptions.py
"""Custom exception classes for the RAG backend.

These exceptions are used throughout the core and modules to provide
consistent error handling and enable the API layer to return appropriate
HTTP status codes.
"""

class CoreError(Exception):
    """Base class for core-related errors."""
    pass

class UnauthorizedError(CoreError):
    """Raised when authentication fails or token is invalid/expired."""
    pass

class ForbiddenError(CoreError):
    """Raised when a user attempts an action they are not permitted to perform."""
    pass

class ValidationError(CoreError):
    """Raised when input data fails validation checks."""
    pass

class NotFoundError(CoreError):
    """Raised when a requested resource cannot be found."""
    pass

