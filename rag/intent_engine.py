"""
rag/intent_engine.py
────────────────────
Multi-intent handling + high-intent interruption engine for the
ARK Learning Arena WhatsApp bot.

Capabilities:
    - Detect multiple intents in a single message
    - Prioritize intents: Admission > Fees > Course Info > General
    - Detect high-intent "I want to join NOW" interruptions
    - Provide structured response guidance for multi-intent messages
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Optional

logger = logging.getLogger("ark.intent")


# =====================================================================
# Intent Types (ordered by priority — higher index = higher priority)
# =====================================================================

class Intent(str, Enum):
    GENERAL = "general"
    LOCATION = "location"
    DEMO = "demo"
    TEACHING_METHOD = "teaching_method"
    COURSE_INFO = "course_info"
    BATCH_INFO = "batch_info"
    FEES = "fees"
    ADMISSION = "admission"


# Priority order: highest priority first
INTENT_PRIORITY = [
    Intent.ADMISSION,
    Intent.FEES,
    Intent.COURSE_INFO,
    Intent.BATCH_INFO,
    Intent.TEACHING_METHOD,
    Intent.DEMO,
    Intent.LOCATION,
    Intent.GENERAL,
]


# =====================================================================
# Intent Detection Patterns
# =====================================================================

_INTENT_PATTERNS: dict[Intent, list[str]] = {
    Intent.ADMISSION: [
        r"\b(admission|admit|enrol|enroll|join|register|registration)\b",
        r"\b(apply|application|seat|book\s+seat)\b",
        r"\b(how\s+to\s+join|want\s+to\s+join|when\s+can\s+(i|we)\s+start)\b",
    ],
    Intent.FEES: [
        r"\b(fee|fees|cost|price|charges|amount|expensive|cheap)\b",
        r"\b(discount|concession|scholarship|negotiate)\b",
        r"\b(how\s+much|kitna|evlo)\b",
    ],
    Intent.COURSE_INFO: [
        r"\b(neet|jee|foundation|coaching|program|course|batch)\b",
        r"\b(class\s+\d+|10th|11th|12th|repeater)\b",
        r"\b(tuition|olympiad|school)\b",
    ],
    Intent.BATCH_INFO: [
        r"\b(batch\s+size|how\s+many\s+students|small\s+batch)\b",
        r"\b(timing|schedule|hours|duration)\b",
    ],
    Intent.TEACHING_METHOD: [
        r"\b(teaching\s+method|how\s+do\s+you\s+teach|system|approach)\b",
        r"\b(5.?pillar|concept|mentoring|testing\s+system)\b",
        r"\b(track\s+progress|performance|analytics|report)\b",
    ],
    Intent.DEMO: [
        r"\b(demo|trial\s+class|free\s+class|sample\s+class)\b",
    ],
    Intent.LOCATION: [
        r"\b(location|where|address|branch|centre|center)\b",
        r"\b(how\s+to\s+reach|direction|map)\b",
    ],
}


# =====================================================================
# High-Intent Interruption Patterns
# =====================================================================

HIGH_INTENT_PHRASES = [
    "i want to join",
    "i want admission",
    "admission now",
    "want to enroll",
    "want to enrol",
    "when can we start",
    "when can i start",
    "book seat",
    "book a seat",
    "reserve seat",
    "confirm admission",
    "ready to join",
    "let's start",
    "lets start",
    "i'm ready",
    "im ready",
    "sign me up",
    "start immediately",
]

HIGH_INTENT_PATTERNS = [
    r"\bwant\s+to\s+(join|enrol|enroll|admit|start)\b",
    r"\badmission\s+(now|immediately|today)\b",
    r"\b(book|reserve|confirm)\s+(a\s+)?(seat|admission)\b",
    r"\b(ready|want)\s+to\s+(join|start)\b",
    r"\b(i'?m|i\s+am)\s+ready\b",
]


# =====================================================================
# Public API
# =====================================================================

def detect_intents(message: str) -> list[Intent]:
    """
    Detect ALL intents present in a message.

    Returns a list of Intent enums, sorted by priority (highest first).
    """
    msg = message.lower().strip()
    found: set[Intent] = set()

    for intent, patterns in _INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, msg):
                found.add(intent)
                break

    if not found:
        return [Intent.GENERAL]

    # Sort by priority
    return sorted(found, key=lambda i: INTENT_PRIORITY.index(i))


def get_primary_intent(message: str) -> Intent:
    """Get the highest-priority intent from a message."""
    intents = detect_intents(message)
    return intents[0] if intents else Intent.GENERAL


def is_multi_intent(message: str) -> bool:
    """Check if a message contains multiple distinct intents."""
    intents = detect_intents(message)
    return len(intents) > 1 and Intent.GENERAL not in intents


def detect_high_intent(message: str) -> bool:
    """
    Detect whether the message signals an immediate join/admission intent
    that should interrupt the current conversation flow.

    Returns True if the user wants to join NOW.
    """
    msg = message.lower().strip()

    for phrase in HIGH_INTENT_PHRASES:
        if phrase in msg:
            return True

    for pattern in HIGH_INTENT_PATTERNS:
        if re.search(pattern, msg):
            return True

    return False


# High-intent interruption response
HIGH_INTENT_RESPONSE = (
    "Great! Let's get this started. 🎯\n\n"
    "Please share your *contact number* so our academic counsellor "
    "can assist you immediately with the admission process.\n\n"
    "— Team ARK Learning Arena"
)

# Multi-intent acknowledgement
MULTI_INTENT_PREFIX = (
    "Great questions! Let me address them:\n\n"
)
