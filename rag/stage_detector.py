"""
rag/stage_detector.py
─────────────────────
Decision-stage detection engine for the ARK Learning Arena WhatsApp bot.

Classifies each user message into one of four buyer-journey stages,
and provides per-stage system-prompt instructions so the bot adjusts
its behaviour accordingly.

Stages:
    EXPLORING  — user is gathering general info
    EVALUATING — user is assessing ARK's teaching system
    COMPARING  — user is comparing with competitors / asking fees
    READY      — user is ready to join / asking about admission
"""

from __future__ import annotations

import logging
import re
import time
from enum import Enum
from typing import Optional

logger = logging.getLogger("ark.stage")


# =====================================================================
# Stage Enum
# =====================================================================

class DecisionStage(str, Enum):
    EXPLORING = "EXPLORING"
    EVALUATING = "EVALUATING"
    COMPARING = "COMPARING"
    READY = "READY"


# =====================================================================
# Detection Patterns
# =====================================================================

_READY_PATTERNS = [
    r"\b(how\s+to\s+join|want\s+to\s+join|i\s+want\s+to\s+join)\b",
    r"\badmission\b",
    r"\b(enrol|enroll|register|registration)\b",
    r"\b(when\s+can\s+(we|i)\s+start)\b",
    r"\b(book\s+(a\s+)?seat)\b",
    r"\b(confirm\s+admission)\b",
    r"\b(start\s+classes)\b",
]

_COMPARING_PATTERNS = [
    r"\b(fee|fees|cost|price|charges|expensive|cheap|afford)\b",
    r"\b(other\s+institute|competitor|alternative|better\s+than)\b",
    r"\b(why\s+ark)\b",
    r"\b(discount|concession|scholarship|negotiate)\b",
    r"\b(compare|comparison|versus|vs)\b",
    r"\b(what\s+is\s+different|how\s+are\s+you\s+different)\b",
]

_EVALUATING_PATTERNS = [
    r"\b(how\s+do\s+you\s+teach|teaching\s+method|teaching\s+system)\b",
    r"\b(what\s+is\s+your\s+system|5.?pillar|five.?pillar)\b",
    r"\b(batch\s+size|how\s+many\s+students)\b",
    r"\b(track\s+progress|performance\s+analytics)\b",
    r"\b(weekly\s+test|monthly\s+test|assessment)\b",
    r"\b(how\s+is\s+your|what\s+makes)\b",
    r"\b(mentoring|mentor|guidance)\b",
    r"\b(parent\s+report|parent\s+update)\b",
    r"\b(result|track\s+record|success)\b",
]

# Everything else is EXPLORING (default)


# =====================================================================
# Per-User Stage State
# =====================================================================

# In-memory store: user_id → current stage
_user_stages: dict[str, DecisionStage] = {}
_stage_updated: dict[str, float] = {}
_STAGE_TIMEOUT = 86400  # 24 hours


def _stage_upgrade(current: DecisionStage, detected: DecisionStage) -> DecisionStage:
    """Stages only move forward (never regress)."""
    order = [DecisionStage.EXPLORING, DecisionStage.EVALUATING,
             DecisionStage.COMPARING, DecisionStage.READY]
    ci = order.index(current)
    di = order.index(detected)
    return order[max(ci, di)]


# =====================================================================
# Public API
# =====================================================================

def detect_stage(message: str) -> DecisionStage:
    """
    Classify a single message into a decision stage.

    Does NOT update per-user state — use detect_and_update_stage() for that.
    """
    msg = message.lower().strip()

    for pattern in _READY_PATTERNS:
        if re.search(pattern, msg):
            return DecisionStage.READY

    for pattern in _COMPARING_PATTERNS:
        if re.search(pattern, msg):
            return DecisionStage.COMPARING

    for pattern in _EVALUATING_PATTERNS:
        if re.search(pattern, msg):
            return DecisionStage.EVALUATING

    return DecisionStage.EXPLORING


def detect_and_update_stage(user_id: str, message: str) -> DecisionStage:
    """
    Detect stage from message and update per-user state.

    Stages only advance forward (never regress).

    Returns the user's current stage after update.
    """
    detected = detect_stage(message)

    # Check timeout
    last_update = _stage_updated.get(user_id, 0)
    if time.time() - last_update > _STAGE_TIMEOUT:
        _user_stages.pop(user_id, None)

    current = _user_stages.get(user_id, DecisionStage.EXPLORING)
    new_stage = _stage_upgrade(current, detected)

    if new_stage != current:
        logger.info(
            "STAGE_CHANGE | user=%s | %s → %s",
            user_id, current.value, new_stage.value,
        )

    _user_stages[user_id] = new_stage
    _stage_updated[user_id] = time.time()
    return new_stage


def get_stage(user_id: str) -> DecisionStage:
    """Get the current stage for a user."""
    return _user_stages.get(user_id, DecisionStage.EXPLORING)


def reset_stage(user_id: str) -> None:
    """Reset stage tracking for a user."""
    _user_stages.pop(user_id, None)
    _stage_updated.pop(user_id, None)


# =====================================================================
# Stage-Aware Prompt Instructions
# =====================================================================

_STAGE_INSTRUCTIONS: dict[DecisionStage, str] = {
    DecisionStage.EXPLORING: (
        "The user is in the EXPLORING stage — they are gathering general information. "
        "Educate them about ARK's programs, highlight the structured approach, "
        "and gently guide them toward specific courses."
    ),
    DecisionStage.EVALUATING: (
        "The user is in the EVALUATING stage — they are assessing our system. "
        "Build authority by highlighting the 5-Pillar Model: diagnostic assessment, "
        "structured planning, weekly testing, performance analytics, and parent reporting. "
        "Emphasize system and structure."
    ),
    DecisionStage.COMPARING: (
        "The user is in the COMPARING stage — they are comparing with other institutes or asking about fees. "
        "Differentiate ARK strongly: small batch sizes (15–20), weekly testing, data-driven tracking, "
        "personalized mentoring. Use authority and scarcity triggers. "
        "Position ARK as a system-based institute, not regular tuition."
    ),
    DecisionStage.READY: (
        "The user is in the READY stage — they want to join. "
        "Push the Academic Diagnostic Assessment (ADA) immediately. "
        "Capture their contact details. Trigger counsellor handover. "
        "Be direct and action-oriented."
    ),
}


def get_stage_instruction(stage: DecisionStage) -> str:
    """Get the system prompt modifier for a given stage."""
    return _STAGE_INSTRUCTIONS.get(stage, "")
