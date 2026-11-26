"""
input_validator.py — Input validation and prompt-injection protection.

All public functions raise FastAPI HTTPExceptions on invalid input so
they can be called directly from route handlers.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from fastapi import HTTPException, status
from loguru import logger

from backend.config import settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_QUESTION_LENGTH: int = 2000

# Patterns that indicate prompt-injection or jailbreak attempts.
# Each entry is a compiled regex; matching is case-insensitive.
FORBIDDEN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+(instructions?|prompts?)", re.IGNORECASE),
    re.compile(r"system\s*:", re.IGNORECASE),
    re.compile(r"</s>", re.IGNORECASE),
    re.compile(r"###\s*(system|instruction|prompt)", re.IGNORECASE),
    re.compile(r"\bact\s+as\b", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"IGNORE\s+ALL\s+PREVIOUS", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all)\s+(you|above)", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"<\|im_end\|>", re.IGNORECASE),
    re.compile(r"\[INST\]", re.IGNORECASE),
    re.compile(r"\[/INST\]", re.IGNORECASE),
    re.compile(r"prompt\s+injection", re.IGNORECASE),
    re.compile(r"disregard\s+(the\s+)?(above|previous|prior)", re.IGNORECASE),
]

# Allowed characters in equipment IDs
_EQUIPMENT_ID_RE = re.compile(r"^[a-zA-Z0-9\-]{1,100}$")


# ---------------------------------------------------------------------------
# HTML stripping helper
# ---------------------------------------------------------------------------

class _HTMLStripper(HTMLParser):
    """Minimal HTML parser that collects only text nodes."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:  # noqa: D401
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


# ---------------------------------------------------------------------------
# Public validators
# ---------------------------------------------------------------------------

def validate_question(question: str) -> str:
    """
    Validate and sanitise a user question before passing it to the LLM.

    - Strips leading/trailing whitespace.
    - Enforces MAX_QUESTION_LENGTH.
    - Rejects known prompt-injection patterns.

    Args:
        question: Raw question string from the API request.

    Returns:
        Cleaned question string.

    Raises:
        HTTPException 400: on empty, too-long, or injected input.
    """
    cleaned = question.strip()

    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question must not be empty.",
        )

    if len(cleaned) > MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Question exceeds maximum length of {MAX_QUESTION_LENGTH} characters.",
        )

    for pattern in FORBIDDEN_PATTERNS:
        if pattern.search(cleaned):
            logger.warning(f"Prompt-injection attempt detected: pattern={pattern.pattern!r}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question contains forbidden content and cannot be processed.",
            )

    return cleaned


def validate_file_size(file_size: int) -> None:
    """
    Raise HTTP 413 if *file_size* exceeds the configured maximum.

    Args:
        file_size: File size in bytes.

    Raises:
        HTTPException 413: if the file is too large.
    """
    max_bytes: int = settings.max_file_size_mb * 1024 * 1024
    if file_size > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size {file_size / (1024 * 1024):.1f} MB exceeds the "
                f"maximum allowed size of {settings.max_file_size_mb} MB."
            ),
        )


def sanitize_text_output(text: str) -> str:
    """
    Strip any HTML/script tags from LLM-generated output.

    Prevents stored-XSS if the response is rendered in a browser without
    proper escaping.

    Args:
        text: Raw LLM response string.

    Returns:
        Plain-text string with HTML tags removed.
    """
    stripper = _HTMLStripper()
    stripper.feed(text)
    return stripper.get_text()


def validate_equipment_id(equipment_id: str) -> str:
    """
    Validate that *equipment_id* contains only safe characters.

    Allowed: alphanumeric characters and hyphens, 1–100 chars.

    Args:
        equipment_id: Raw equipment ID from the request.

    Returns:
        Validated equipment ID string.

    Raises:
        HTTPException 400: if the format is invalid.
    """
    if not _EQUIPMENT_ID_RE.match(equipment_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Invalid equipment_id format. "
                "Only alphanumeric characters and hyphens are allowed (max 100 chars)."
            ),
        )
    return equipment_id
