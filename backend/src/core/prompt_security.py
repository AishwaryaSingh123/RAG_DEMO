# backend/src/core/prompt_security.py
"""Prompt security utilities for the RAG backend.

This module provides lightweight defenses against prompt injection attacks and a
simple trust‑scoring mechanism. The implementation follows the board’s
requirement that each core file has a single responsibility.

* ``detect_prompt_injection`` returns ``True`` when the prompt contains patterns
  that are commonly used in prompt‑injection attempts.
* ``compute_trust_score`` returns a float between 0.0 (untrusted) and 1.0 (fully
  trusted) based on heuristic checks.

These helpers are deliberately conservative – they aim to block clearly malicious
inputs while avoiding false positives for normal user queries.
"""

import re
from typing import List

# Common suspicious phrases (case‑insensitive) that indicate an attempt to
# override system instructions or extract hidden context.
_SUSPICIOUS_PATTERNS: List[re.Pattern] = [
    re.compile(r"ignore\s+your\s+instructions", re.IGNORECASE),
    re.compile(r"pretend\s+you\s+are\s+.*", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+.*", re.IGNORECASE),
    re.compile(r"disregard\s+the\s+previous\s+rules", re.IGNORECASE),
    re.compile(r"as\s+an\s+assistant", re.IGNORECASE),
    re.compile(r"act\s+as\s+.*\bassistant\b", re.IGNORECASE),
    re.compile(r"\b\$\{.*\}\b", re.IGNORECASE),  # template injection
    re.compile(r"<\s*script\b", re.IGNORECASE),   # script tags
]

def detect_prompt_injection(prompt: str) -> bool:
    """Return ``True`` if the prompt appears to contain an injection attempt.

    The function scans the prompt for any of the patterns defined in
    ``_SUSPICIOUS_PATTERNS``. It also checks for a high ratio of special
    characters (e.g., ``{}``, ``<>``) which are atypical for normal user queries.
    """
    if not prompt:
        return False
    # Direct pattern match
    for pat in _SUSPICIOUS_PATTERNS:
        if pat.search(prompt):
            return True
    # Heuristic: if >30% of characters are non‑alphanumeric, flag it
    special_ratio = len([c for c in prompt if not c.isalnum() and not c.isspace()]) / len(prompt)
    if special_ratio > 0.3:
        return True
    return False

def compute_trust_score(prompt: str) -> float:
    """Compute a heuristic trust score for a prompt.

    The score is in the range ``0.0`` (low trust) to ``1.0`` (high trust). A lower
    score is given when suspicious patterns are present or when the prompt is
    unusually short/long. The algorithm is intentionally simple and can be tuned
    later without breaking the module contract.
    """
    if not prompt:
        return 0.0
    length = len(prompt)
    # Base score: longer prompts (up to 200 chars) are considered more
    # trustworthy because they are less likely to be a terse injection.
    base = min(1.0, length / 200)
    # Penalize if injection patterns are detected
    if detect_prompt_injection(prompt):
        base *= 0.2
    # Additional penalty for excessive special characters
    special_ratio = len([c for c in prompt if not c.isalnum() and not c.isspace()]) / length
    if special_ratio > 0.25:
        base *= 0.5
    return round(base, 3)

__all__ = ["detect_prompt_injection", "compute_trust_score"]
