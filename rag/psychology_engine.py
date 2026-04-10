"""
rag/psychology_engine.py
────────────────────────
Psychological trigger rotation and trust reinforcement engine
for the ARK Learning Arena WhatsApp bot.

Rotates 4 psychological triggers across responses to maximize
conversion without being repetitive.

Triggers:
    AUTHORITY       — structured system, weekly testing, analytics
    SCARCITY        — limited batch sizes, filling fast
    RISK_REVERSAL   — free diagnostic assessment first
    OUTCOME         — consistent improvement, target scores

Rules:
    - Inject a trigger every 2–3 responses
    - Never repeat the same trigger consecutively
    - Inject trust builders periodically
"""

from __future__ import annotations

import logging
import random
import re
import time
from enum import Enum
from typing import Optional

logger = logging.getLogger("ark.psychology")


# =====================================================================
# Trigger Types
# =====================================================================

class PsychTrigger(str, Enum):
    AUTHORITY = "authority"
    SCARCITY = "scarcity"
    RISK_REVERSAL = "risk_reversal"
    OUTCOME = "outcome"


# =====================================================================
# Trigger Lines (multiple variants per trigger for natural rotation)
# =====================================================================

TRIGGER_LINES: dict[PsychTrigger, list[str]] = {
    PsychTrigger.AUTHORITY: [
        "Our program is structured with weekly testing and performance analytics to ensure consistent improvement.",
        "ARK follows a systematic 5-pillar approach: diagnostic assessment, structured planning, weekly testing, performance analytics, and parent reporting.",
        "With 500+ students trained since 2015, our structured academic system has consistently delivered results.",
    ],
    PsychTrigger.SCARCITY: [
        "We maintain limited batch sizes of 15–20 students to ensure individual attention and quality.",
        "Our batches fill up quickly — we keep them small to maintain our teaching standards.",
        "Seats are limited as we prioritize quality over quantity with our small batch model.",
    ],
    PsychTrigger.RISK_REVERSAL: [
        "That's why we start with a free Diagnostic Assessment before recommending any program — so you can make an informed decision.",
        "We offer a free Academic Diagnostic Assessment (ADA) first, so there's no commitment until you see the analysis.",
        "There's no risk — our free diagnostic test helps us understand the student's level before suggesting anything.",
    ],
    PsychTrigger.OUTCOME: [
        "This structured approach helps students steadily improve towards their target NEET scores.",
        "Our system is designed for consistent improvement — students see measurable progress through regular testing.",
        "With our data-driven approach, students consistently build towards strong academic performance.",
    ],
}


# =====================================================================
# Trust Builder Lines
# =====================================================================

TRUST_LINES = [
    "ARK has guided 500+ students with a structured, discipline-driven system since 2015.",
    "Our approach focuses on consistent improvement rather than shortcuts.",
    "We are a system-based academic performance institute, not regular tuition.",
]


# =====================================================================
# ADA Conversion Lines (injected after most responses)
# =====================================================================

ADA_LINES = [
    (
        "\n\n📋 *We recommend starting with our free Academic Diagnostic Assessment (ADA).*\n"
        "This helps us:\n"
        "• Identify current academic level\n"
        "• Analyse strengths & weaknesses\n"
        "• Create a personalized improvement plan\n\n"
        "Would you like to book a slot?"
    ),
    (
        "\n\n📋 *Have you taken our free Academic Diagnostic Assessment yet?*\n"
        "It's the first step — we evaluate the student's current level "
        "and provide a personalized roadmap.\n\n"
        "Shall we schedule one for you?"
    ),
    (
        "\n\n📋 *To guide you accurately, we recommend a free Diagnostic Assessment (ADA).*\n"
        "The assessment helps us understand the student's "
        "strengths and areas for improvement.\n\n"
        "Would you like to proceed?"
    ),
]


# =====================================================================
# Per-User State
# =====================================================================

class _UserTriggerState:
    """Track trigger rotation for a single user."""

    def __init__(self):
        self.response_count: int = 0
        self.last_trigger: Optional[PsychTrigger] = None
        self.last_trigger_index: dict[PsychTrigger, int] = {}
        self.ada_count: int = 0  # how many times ADA has been shown
        self.updated_at: float = time.time()


# In-memory store
_user_states: dict[str, _UserTriggerState] = {}
_STATE_TIMEOUT = 86400  # 24 hours


def _get_state(user_id: str) -> _UserTriggerState:
    state = _user_states.get(user_id)
    if state is None or (time.time() - state.updated_at > _STATE_TIMEOUT):
        state = _UserTriggerState()
        _user_states[user_id] = state
    return state


# =====================================================================
# Public API
# =====================================================================

def should_inject_trigger(user_id: str, frequency: int = 2) -> bool:
    """
    Check if a psychological trigger should be injected for this response.

    Args:
        user_id: The user identifier.
        frequency: Inject every N responses (default: 2).
    """
    state = _get_state(user_id)
    state.response_count += 1
    state.updated_at = time.time()
    return state.response_count % frequency == 0


def get_next_trigger(user_id: str) -> str:
    """
    Get the next psychological trigger line, ensuring no consecutive repeats.

    Returns a trigger line string.
    """
    state = _get_state(user_id)

    # Pick a trigger type that's different from the last one
    available = [t for t in PsychTrigger if t != state.last_trigger]
    if not available:
        available = list(PsychTrigger)

    trigger = random.choice(available)
    state.last_trigger = trigger

    # Rotate through variants
    lines = TRIGGER_LINES[trigger]
    idx = state.last_trigger_index.get(trigger, -1)
    idx = (idx + 1) % len(lines)
    state.last_trigger_index[trigger] = idx

    line = lines[idx]
    logger.debug("TRIGGER | user=%s | type=%s", user_id, trigger.value)
    return line


def should_inject_ada(user_id: str, frequency: int = 3) -> bool:
    """
    Check if an ADA recommendation should be injected.

    Injects ADA less frequently than triggers, max 3 times per session.

    Args:
        user_id: The user identifier.
        frequency: Inject every N responses (default: 3).
    """
    state = _get_state(user_id)
    if state.ada_count >= 3:
        return False
    return state.response_count % frequency == 0 and state.response_count > 0


def get_ada_line(user_id: str) -> str:
    """
    Get the next ADA conversion line.

    Returns an ADA recommendation string.
    """
    state = _get_state(user_id)
    idx = state.ada_count % len(ADA_LINES)
    state.ada_count += 1
    return ADA_LINES[idx]


def get_trust_line() -> str:
    """Get a random trust-building line."""
    return random.choice(TRUST_LINES)


def reset_triggers(user_id: str) -> None:
    """Reset trigger state for a user."""
    _user_states.pop(user_id, None)


# =====================================================================
# Language Detection (Tamil)
# =====================================================================

TAMIL_WORDS = [
    "evlo", "enna", "epdi", "enga", "yaar", "edhu",
    "sollunga", "sollu", "theriyum", "theriyala",
    "nalla", "romba", "konjam", "illai", "irukku",
    "vendum", "mudiyum", "mudiyala", "vaanga",
    "ponga", "pannunga", "solra", "padikka",
    "fees", "class", "school",  # common Tamil-English mix
    "enakku", "enga", "paiyan", "ponnu",
    "eppadi", "ethuku", "yen",
]

# Tamil Unicode range (basic)
_TAMIL_UNICODE = re.compile(r"[\u0B80-\u0BFF]")


def detect_tamil(message: str) -> bool:
    """
    Detect if a message is likely in Tamil (words or script).

    Returns True if Tamil is detected.
    """
    msg = message.lower().strip()

    # Check Tamil Unicode characters
    if _TAMIL_UNICODE.search(msg):
        return True

    # Check Tamil transliteration words
    words = msg.split()
    tamil_count = sum(1 for w in words if w in TAMIL_WORDS)
    return tamil_count >= 2 or (len(words) <= 3 and tamil_count >= 1)
