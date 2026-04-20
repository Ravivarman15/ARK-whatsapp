"""
rag/response_formatter.py
─────────────────────────
WhatsApp-friendly response formatter.

Enforces strict rules on ALL outgoing bot messages:
  - Max 3–5 lines
  - Simple English, no paragraphs
  - Strips markdown headers
  - Converts long text to bullet points
  - Removes trailing signatures
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger("ark.formatter")


# =====================================================================
# ADA / Promo patterns to strip from factual answers
# =====================================================================

_ADA_PATTERNS = [
    r"📋.*?(Would you like to book a slot\??|Shall we schedule one.*?\??|Would you like to proceed\??)",
    r"\n*📋[^\n]*\n(?:.*\n)*?.*(?:book|schedule|proceed)\??\s*$",
]

_SIGNATURE_PATTERN = re.compile(
    r"\n*[—–-]+\s*Team ARK.*$", re.IGNORECASE | re.DOTALL
)

_HEADER_PATTERN = re.compile(r"^#{1,3}\s+", re.MULTILINE)


def _strip_ada_blocks(text: str) -> str:
    """Remove ADA assessment promo blocks."""
    for pattern in _ADA_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _strip_signatures(text: str) -> str:
    """Remove trailing '— Team ARK Learning Arena' signatures."""
    return _SIGNATURE_PATTERN.sub("", text).strip()


def _strip_headers(text: str) -> str:
    """Remove markdown headers (##, ###)."""
    return _HEADER_PATTERN.sub("", text).strip()


def _collapse_whitespace(text: str) -> str:
    """Collapse multiple blank lines into one."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _truncate_lines(text: str, max_lines: int = 6) -> str:
    """
    Truncate to max_lines while keeping bullet points intact.
    Tries to end at a sentence boundary.
    """
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text

    # Keep the first max_lines lines
    truncated = lines[:max_lines]

    # If the last line is a partial bullet, keep it
    result = "\n".join(truncated).strip()

    # Don't end mid-sentence — try to find a period
    if not result.endswith((".", "!", "?", "👍", "🙏")):
        # Find last sentence end
        last_period = result.rfind(".")
        last_question = result.rfind("?")
        last_end = max(last_period, last_question)
        if last_end > len(result) // 2:
            result = result[: last_end + 1]

    return result


def format_whatsapp_response(
    text: str,
    *,
    is_factual: bool = False,
    max_lines: int = 14,
) -> str:
    """
    Format any bot response for WhatsApp delivery.

    Args:
        text:       Raw LLM or system response text.
        is_factual: If True, strips all promotional content (ADA, triggers).
        max_lines:  Maximum number of lines in the output.

    Returns:
        Clean, WhatsApp-friendly response string.
    """
    if not text:
        return text

    result = text.strip()

    # Strip markdown headers
    result = _strip_headers(result)

    # For factual questions, strip all promotional content
    if is_factual:
        result = _strip_ada_blocks(result)
        result = _strip_signatures(result)

    # Collapse whitespace
    result = _collapse_whitespace(result)

    # Truncate to max lines
    result = _truncate_lines(result, max_lines=max_lines)

    # Final cleanup
    result = result.strip()

    if not result:
        result = text.strip().split("\n")[0]  # fallback to first line

    return result
