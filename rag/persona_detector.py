"""
rag/persona_detector.py
───────────────────────
Parent persona detection engine for the ARK Learning Arena WhatsApp bot.

Detects the parent's mindset from their message patterns and provides
tailored system-prompt instructions so the bot adjusts its emphasis.

Personas:
    MARKS_FOCUSED  — wants results, scores, ranks
    CONCERNED      — worried about weak child
    SKEPTICAL      — questioning what's different
    GENERAL        — no specific persona detected
"""

from __future__ import annotations

import logging
import re
import time
from enum import Enum
from typing import Optional

logger = logging.getLogger("ark.persona")


# =====================================================================
# Persona Enum
# =====================================================================

class ParentPersona(str, Enum):
    MARKS_FOCUSED = "MARKS_FOCUSED"
    CONCERNED = "CONCERNED"
    SKEPTICAL = "SKEPTICAL"
    GENERAL = "GENERAL"


# =====================================================================
# Detection Patterns
# =====================================================================

_CONCERNED_PATTERNS = [
    r"\b(weak|struggling|not\s+scoring|poor\s+marks|low\s+marks)\b",
    r"\b(basics?\s+not\s+clear|can'?t\s+understand|difficulty)\b",
    r"\b(worried|concern|afraid|scared|fear)\b",
    r"\b(help\s+my\s+(child|kid|son|daughter))\b",
    r"\b(not\s+good\s+(at|in)|falling\s+behind)\b",
    r"\b(need\s+support|need\s+help|extra\s+help)\b",
    r"\b(remedial|slow\s+learner|below\s+average)\b",
]

_MARKS_FOCUSED_PATTERNS = [
    r"\b(marks?|scores?|rank|percentage|results?|topp?er)\b",
    r"\b(how\s+much\s+score|what\s+score|target\s+score)\b",
    r"\b(guarantee|assure|promise|ensure\s+results?)\b",
    r"\b(600\+?|650\+?|700\+?|neet\s+score)\b",
    r"\b(improvement|improve\s+marks|increase\s+score)\b",
    r"\b(previous\s+results?|success\s+rate|track\s+record)\b",
    r"\b(expect|expected)\b",
]

_SKEPTICAL_PATTERNS = [
    r"\b(what'?s?\s+different|how\s+are\s+you\s+different)\b",
    r"\b(why\s+ark|why\s+should\s+(i|we))\b",
    r"\b(better\s+than|compared?\s+to|versus|vs)\b",
    r"\b(not\s+convinced|not\s+sure|doubt|skeptic)\b",
    r"\b(other\s+institute|other\s+coaching|already\s+in\s+coaching)\b",
    r"\b(prove|evidence|show\s+me)\b",
    r"\bwhat\s+makes\s+\w+",
    r"\bhow\s+is\s+(your|the|this|their)\b",
    r"\bdifferent\b",
]


# =====================================================================
# Per-User Persona State
# =====================================================================

_user_personas: dict[str, ParentPersona] = {}
_persona_updated: dict[str, float] = {}
_PERSONA_TIMEOUT = 86400  # 24 hours


# =====================================================================
# Public API
# =====================================================================

def detect_persona(message: str) -> ParentPersona:
    """
    Detect a parent persona from a single message.

    Does NOT update per-user state — use detect_and_update_persona() for that.
    """
    msg = message.lower().strip()

    # Concerned parent has highest priority (most conversion potential)
    for pattern in _CONCERNED_PATTERNS:
        if re.search(pattern, msg):
            return ParentPersona.CONCERNED

    for pattern in _SKEPTICAL_PATTERNS:
        if re.search(pattern, msg):
            return ParentPersona.SKEPTICAL

    for pattern in _MARKS_FOCUSED_PATTERNS:
        if re.search(pattern, msg):
            return ParentPersona.MARKS_FOCUSED

    return ParentPersona.GENERAL


def detect_and_update_persona(user_id: str, message: str) -> ParentPersona:
    """
    Detect persona from message and update per-user state.

    Once a specific persona is detected, it persists (doesn't revert to GENERAL).

    Returns the user's current persona after update.
    """
    detected = detect_persona(message)

    # Check timeout
    last_update = _persona_updated.get(user_id, 0)
    if time.time() - last_update > _PERSONA_TIMEOUT:
        _user_personas.pop(user_id, None)

    current = _user_personas.get(user_id, ParentPersona.GENERAL)

    # Only upgrade from GENERAL to a specific persona
    if current == ParentPersona.GENERAL and detected != ParentPersona.GENERAL:
        _user_personas[user_id] = detected
        _persona_updated[user_id] = time.time()
        logger.info(
            "PERSONA_DETECTED | user=%s | persona=%s",
            user_id, detected.value,
        )
        return detected

    return current


def get_persona(user_id: str) -> ParentPersona:
    """Get the current persona for a user."""
    return _user_personas.get(user_id, ParentPersona.GENERAL)


def reset_persona(user_id: str) -> None:
    """Reset persona for a user."""
    _user_personas.pop(user_id, None)
    _persona_updated.pop(user_id, None)


# =====================================================================
# Persona-Aware Prompt Instructions
# =====================================================================

_PERSONA_INSTRUCTIONS: dict[ParentPersona, str] = {
    ParentPersona.MARKS_FOCUSED: (
        "This parent is MARKS-FOCUSED. They care about scores, results, and rankings. "
        "Emphasize: weekly chapter tests, monthly full-length mock exams, "
        "performance analytics, data-driven improvement, target NEET scores (600+), "
        "and our track record of consistent academic improvement."
    ),
    ParentPersona.CONCERNED: (
        "This parent is CONCERNED about their child. They worry the child is weak or struggling. "
        "Emphasize: personalized mentoring, remedial sessions, step-by-step improvement, "
        "diagnostic assessment to identify gaps, small batch sizes for individual attention, "
        "and our structured system that helps weak students improve systematically. "
        "Be empathetic and reassuring. Normalize their concern."
    ),
    ParentPersona.SKEPTICAL: (
        "This parent is SKEPTICAL. They are questioning what makes ARK different. "
        "Emphasize: the 5-Pillar system (diagnostic → planning → testing → analytics → reporting), "
        "ARK is NOT regular tuition but a system-based academic performance institute, "
        "500+ students trained since 2015, structured discipline-driven approach, "
        "small batch sizes, and data-driven accountability."
    ),
    ParentPersona.GENERAL: "",
}


def get_persona_instruction(persona: ParentPersona) -> str:
    """Get the system prompt modifier for a given persona."""
    return _PERSONA_INSTRUCTIONS.get(persona, "")
