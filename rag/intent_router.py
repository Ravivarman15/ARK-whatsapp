"""
rag/intent_router.py
────────────────────
Unified intent classification with strict priority ordering.

Priority (highest first):
  1. COMPLAINT        — dissatisfaction / escalation
  2. HUMAN_ESCALATION — wants to speak to a person
  3. FACTUAL_QUESTION — direct knowledge question (RAG only, NO follow-up)
  4. ADMISSION_INTENT — wants to join / fees / demo / course
  5. QUALIFICATION    — already in qualification flow (state machine)
  6. GENERAL          — fallback

Key rule:
  If user asks a factual question → answer ONLY via RAG.
  Do NOT trigger follow-up, ADA, or qualification flow.
"""

from __future__ import annotations

import logging
import re
from enum import Enum

logger = logging.getLogger("ark.router")


# =====================================================================
# Route Types
# =====================================================================

class Route(str, Enum):
    COMPLAINT = "complaint"
    HUMAN_ESCALATION = "human_escalation"
    FACTUAL_QUESTION = "factual_question"
    ADMISSION_INTENT = "admission_intent"
    QUALIFICATION = "qualification"
    GENERAL = "general"


# =====================================================================
# Keyword Sets
# =====================================================================

# Admission intent keywords — ONLY these trigger qualification
ADMISSION_KEYWORDS = [
    "join", "admission", "admit", "fees", "fee", "demo",
    "course", "enroll", "enrol", "register", "registration",
    "apply", "application", "seat", "book seat",
]

# Explicit admission phrases (stronger signal)
ADMISSION_PHRASES = [
    "i want to join",
    "want to join",
    "how to join",
    "i want admission",
    "want admission",
    "want to enroll",
    "want to enrol",
    "ready to join",
    "book a seat",
    "reserve seat",
    "confirm admission",
    "sign me up",
]

# Question indicators
QUESTION_STARTERS = {
    "what", "who", "whom", "whose", "how", "why", "when", "where",
    "which", "is", "are", "can", "could", "do", "does", "did",
    "will", "would", "should", "shall", "may", "might",
    "tell", "explain", "describe",
}

# Additional question patterns (Hindi/Tamil)
QUESTION_PATTERNS = [
    r"\?",
    r"^(tell\s+me|explain|describe|show\s+me)\b",
    r"^(kya|kaise|kab|kahan|kaun|kitna|konsa|bata)\b",
]

# Words that indicate a factual question even without question marks
FACTUAL_SIGNAL_WORDS = {
    "founder", "founded", "established", "location", "address",
    "branch", "timing", "schedule", "result", "results",
    "syllabus", "method", "approach", "system", "pillar",
    "teacher", "faculty", "batch", "size", "duration",
    "history", "about", "started",
}


# =====================================================================
# Classification Logic
# =====================================================================

def _is_question(msg: str) -> bool:
    """Check if message is a question (syntactically)."""
    words = msg.split()
    if not words:
        return False

    # Question mark anywhere
    if "?" in msg:
        return True

    # Starts with a question word
    first_word = words[0].lower()
    if first_word in QUESTION_STARTERS:
        return True

    # First two words check (e.g., "tell me")
    if len(words) >= 2:
        two_words = f"{words[0]} {words[1]}".lower()
        if two_words in {"tell me", "show me", "explain the", "describe the"}:
            return True

    # Regex patterns
    for pattern in QUESTION_PATTERNS:
        if re.search(pattern, msg, re.IGNORECASE):
            return True

    return False


def _has_admission_intent(msg: str) -> bool:
    """Check if message contains explicit admission/join intent."""
    msg_lower = msg.lower()

    # Check phrases first (stronger signal)
    for phrase in ADMISSION_PHRASES:
        if phrase in msg_lower:
            return True

    # Check individual keywords
    for keyword in ADMISSION_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", msg_lower):
            return True

    return False


def _has_factual_signals(msg: str) -> bool:
    """Check if message contains factual-knowledge-seeking signals."""
    msg_lower = msg.lower()
    words = set(msg_lower.split())
    return bool(words & FACTUAL_SIGNAL_WORDS)


def classify_message(
    message: str,
    *,
    is_in_qualification: bool = False,
    has_complaint: bool = False,
    has_human_request: bool = False,
) -> Route:
    """
    Classify a user message into a route with strict priority.

    Args:
        message:              Raw user message text.
        is_in_qualification:  Whether the user is currently in a qual flow.
        has_complaint:        Pre-computed complaint detection result.
        has_human_request:    Pre-computed human escalation detection result.

    Returns:
        The Route enum value indicating how to handle this message.
    """
    msg = message.strip()
    if not msg:
        return Route.GENERAL

    # ── Priority 1: Complaint ────────────────────────────────────
    if has_complaint:
        return Route.COMPLAINT

    # ── Priority 2: Human escalation ─────────────────────────────
    if has_human_request:
        return Route.HUMAN_ESCALATION

    is_q = _is_question(msg)
    has_admission = _has_admission_intent(msg)

    # ── Priority 3: Factual question (no admission intent) ───────
    # If it's a question AND does NOT contain admission keywords,
    # treat as a pure factual question → RAG only, no follow-up.
    if is_q and not has_admission:
        return Route.FACTUAL_QUESTION

    # ── Priority 4: Admission intent ─────────────────────────────
    if has_admission:
        return Route.ADMISSION_INTENT

    # ── Priority 5: Active qualification flow ────────────────────
    if is_in_qualification:
        return Route.QUALIFICATION

    # ── Priority 6: General fallback ─────────────────────────────
    return Route.GENERAL
