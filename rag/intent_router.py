"""
rag/intent_router.py
────────────────────
Unified intent classification with strict priority ordering.

Priority (highest first):
  1. COMPLAINT        — dissatisfaction / escalation
  2. HUMAN_ESCALATION — wants to speak to a person
  3. MULTI_INTENT     — question + explicit admission verb in same msg
                        ("I want to join NEET, what are fees?")
  4. FACTUAL_QUESTION — direct knowledge question, even if it touches a
                        topic word like "fees" / "course" (RAG only)
  5. ADMISSION_INTENT — explicit admission verb (join/enroll/admit/
                        register/apply) OR a topic word with no qual flow
                        active yet
  6. QUALIFICATION    — already in qualification flow (state machine)
  7. GENERAL          — fallback

Key rules:
  - "Topic" words (fees, course, demo, batch, timing) are NOT enough to
    trigger admission flow when the message is phrased as a question.
    They route to RAG so the user gets an answer instead of being
    re-prompted for their name.
  - "Verb" words (join, enroll, admit, apply, register) ARE strong
    enough to trigger admission flow even mixed into a question — but
    in that case we answer the question first (MULTI_INTENT), then
    start qualification.
  - Mid-qualification users asking a topic-only question get RAG too,
    so the bot acts like a counsellor (answer, then politely re-prompt
    the pending field) instead of robotically repeating the same prompt.
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
    SMALL_TALK = "small_talk"
    MULTI_INTENT = "multi_intent"
    FACTUAL_QUESTION = "factual_question"
    ADMISSION_INTENT = "admission_intent"
    QUALIFICATION = "qualification"
    GENERAL = "general"


# =====================================================================
# Keyword Sets
# =====================================================================

# Admission VERBS — strong commitment signal. A message containing one
# of these expresses intent to enroll, even if phrased as a question.
ADMISSION_VERBS = [
    "join", "admit", "admission",
    "enroll", "enrol", "register", "registration",
    "apply", "application",
]

# Admission TOPICS — words that name something the institute offers.
# These overlap with factual-question topics; "what are the fees?" is
# a topic question, not an admission signal. Topic words alone trigger
# admission flow only when the user is NOT mid-qualification.
ADMISSION_TOPICS = [
    "fees", "fee", "demo", "course", "courses",
    "batch", "batches", "timing", "timings", "schedule",
    "seat", "book seat",
]

# Explicit admission phrases (stronger signal — bypass question heuristic)
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

# Additional question patterns (Hindi/Tamil + declarative-form questions)
QUESTION_PATTERNS = [
    r"\?",
    r"^(tell\s+me|explain|describe|show\s+me)\b",
    r"^(kya|kaise|kab|kahan|kaun|kitna|konsa|bata)\b",
    # Declarative-form questions ("I'm asking about fees", "want to know fees")
    r"\b(i'?m|i\s+am)\s+asking\b",
    r"\b(want|need|wanted)\s+to\s+know\b",
    r"^asking\b",
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


def _has_admission_verb(msg: str) -> bool:
    """
    True if the message contains an explicit admission verb (join,
    enroll, admit, register, apply) or a multi-word admission phrase.
    These are strong enough to trigger admission flow even when the
    sentence is also a question.
    """
    msg_lower = msg.lower()

    for phrase in ADMISSION_PHRASES:
        if phrase in msg_lower:
            return True

    for verb in ADMISSION_VERBS:
        if re.search(rf"\b{re.escape(verb)}\b", msg_lower):
            return True

    return False


def _has_admission_topic(msg: str) -> bool:
    """
    True if the message names something the institute offers (fees,
    course, demo, batch, timing). These are NOT enough on their own to
    classify as admission intent when the message is a question — a
    user asking "what are the fees?" wants an answer, not a name prompt.
    """
    msg_lower = msg.lower()
    for topic in ADMISSION_TOPICS:
        if re.search(rf"\b{re.escape(topic)}\b", msg_lower):
            return True
    return False


# Backwards-compatible alias — older callers (scoring, segmentation,
# tests) may still import _has_admission_intent. Treat any verb OR
# topic as "intent" for those legacy uses; the router itself uses the
# split helpers above.
def _has_admission_intent(msg: str) -> bool:
    return _has_admission_verb(msg) or _has_admission_topic(msg)


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

    # ── Priority 3: Small talk (greeting / ack / thanks / bye) ────
    # Only when NOT mid-qualification — otherwise an "ok" should be
    # handled by the qualification state machine, not short-circuited
    # to a greeting. Local import avoids a circular dependency.
    if not is_in_qualification:
        from rag.greeting_handler import detect_small_talk
        if detect_small_talk(msg):
            return Route.SMALL_TALK

    is_q = _is_question(msg)
    has_verb = _has_admission_verb(msg)
    has_topic = _has_admission_topic(msg)

    # ── Priority 3: Multi-intent (question + admission verb) ─────
    # "I want to join NEET, what are fees?" — the user is committing
    # AND asking. Caller answers via RAG, then starts qualification.
    if is_q and has_verb:
        return Route.MULTI_INTENT

    # ── Priority 4: Factual question ─────────────────────────────
    # Any question — including topic questions like "what are the
    # fees?" — gets answered. Topic words alone are NOT admission
    # intent when phrased as a question. This is the fix for the
    # mid-qualification "What's the fees?" → "What's the student's
    # name?" loop.
    if is_q:
        return Route.FACTUAL_QUESTION

    # ── Priority 5: Mid-qualification + topic word (no question) ─
    # User mid-flow types just "fees" or "course details". Treat as
    # a topic question so they get an answer + soft re-prompt,
    # rather than a robotic re-ask of the pending field.
    if is_in_qualification and has_topic and not has_verb:
        return Route.FACTUAL_QUESTION

    # ── Priority 6: Admission intent (verb, or topic when fresh) ─
    if has_verb or has_topic:
        return Route.ADMISSION_INTENT

    # ── Priority 7: Active qualification flow ────────────────────
    if is_in_qualification:
        return Route.QUALIFICATION

    # ── Priority 8: General fallback ─────────────────────────────
    return Route.GENERAL
