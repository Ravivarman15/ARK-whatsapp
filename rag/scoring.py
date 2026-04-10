"""
rag/scoring.py
──────────────
Numeric lead scoring system for the ARK Learning Arena WhatsApp bot.

Replaces the simple HOT/WARM/COLD classification with a point-based
model that accumulates score as the user interacts.

Score Bands:
    0–20  → COLD
    21–50 → WARM
    51–80 → HOT
    80+   → VERY_HOT
"""

from __future__ import annotations

import logging
import re
import time
from enum import Enum
from typing import Optional

logger = logging.getLogger("ark.scoring")


# =====================================================================
# Lead Score Categories
# =====================================================================

class LeadScoreType(str, Enum):
    COLD = "COLD"
    WARM = "WARM"
    HOT = "HOT"
    VERY_HOT = "VERY_HOT"


# =====================================================================
# Scoring Table
# =====================================================================

class ScoreAction(str, Enum):
    ASKED_COURSES = "asked_courses"
    ASKED_NEET = "asked_neet"
    SHARED_NAME = "shared_name"
    SHARED_CLASS = "shared_class"
    SHARED_SCHOOL = "shared_school"
    SHARED_PHONE = "shared_phone"
    ASKED_FEES = "asked_fees"
    ASKED_ADMISSION = "asked_admission"


SCORE_VALUES: dict[ScoreAction, int] = {
    ScoreAction.ASKED_COURSES: 5,
    ScoreAction.ASKED_NEET: 10,
    ScoreAction.SHARED_NAME: 10,
    ScoreAction.SHARED_CLASS: 10,
    ScoreAction.SHARED_SCHOOL: 10,
    ScoreAction.SHARED_PHONE: 20,
    ScoreAction.ASKED_FEES: 30,
    ScoreAction.ASKED_ADMISSION: 40,
}


# =====================================================================
# Per-User Score State
# =====================================================================

class _UserScore:
    """Internal score tracker for a single user."""

    def __init__(self):
        self.total: int = 0
        self.actions: set[str] = set()  # track which actions already counted
        self.updated_at: float = time.time()

    def add(self, action: ScoreAction) -> int:
        """Add score for an action (each action only counted once). Returns new total."""
        if action.value in self.actions:
            return self.total
        points = SCORE_VALUES.get(action, 0)
        self.total += points
        self.actions.add(action.value)
        self.updated_at = time.time()
        logger.info(
            "SCORE_UPDATE | action=%s | +%d | total=%d",
            action.value, points, self.total,
        )
        return self.total


# In-memory store: user_id → _UserScore
_scores: dict[str, _UserScore] = {}

# Timeout: clear scores after 24 hours of inactivity
_SCORE_TIMEOUT = 86400


def _get_or_create(user_id: str) -> _UserScore:
    score = _scores.get(user_id)
    if score is None or (time.time() - score.updated_at > _SCORE_TIMEOUT):
        score = _UserScore()
        _scores[user_id] = score
    return score


# =====================================================================
# Public API
# =====================================================================

def update_score(user_id: str, action: ScoreAction) -> int:
    """
    Add score for a specific action. Each action only counted once per user.

    Returns:
        The updated total score.
    """
    return _get_or_create(user_id).add(action)


def get_score(user_id: str) -> int:
    """Get the current total score for a user."""
    return _get_or_create(user_id).total


def get_lead_type(user_id: str) -> LeadScoreType:
    """Get the lead classification based on current score."""
    return classify_score(get_score(user_id))


def classify_score(score: int) -> LeadScoreType:
    """Classify a numeric score into a lead type."""
    if score > 80:
        return LeadScoreType.VERY_HOT
    elif score > 50:
        return LeadScoreType.HOT
    elif score > 20:
        return LeadScoreType.WARM
    return LeadScoreType.COLD


def get_score_actions(user_id: str) -> set[str]:
    """Get the set of actions already counted for a user."""
    return set(_get_or_create(user_id).actions)


def reset_score(user_id: str) -> None:
    """Reset score for a user."""
    _scores.pop(user_id, None)


# =====================================================================
# Message-Based Scoring (auto-detect from message text)
# =====================================================================

_COURSE_PATTERNS = [
    r"\b(course|program|coaching|class|batch|tuition)\b",
]

_NEET_PATTERNS = [
    r"\bneet\b", r"\bmedical\b", r"\bdoctor\b",
]

_FEE_PATTERNS = [
    r"\b(fee|fees|cost|price|charges|amount|expensive|cheap|afford)\b",
]

_ADMISSION_PATTERNS = [
    r"\b(admission|admit|enrol|enroll|join|registration|apply|seat|book)\b",
]


def score_from_message(user_id: str, message: str) -> int:
    """
    Scan a message for scoring signals and update the user's score.

    Returns the new total score.
    """
    msg = message.lower().strip()
    us = _get_or_create(user_id)

    for pattern in _NEET_PATTERNS:
        if re.search(pattern, msg):
            us.add(ScoreAction.ASKED_NEET)
            break

    for pattern in _ADMISSION_PATTERNS:
        if re.search(pattern, msg):
            us.add(ScoreAction.ASKED_ADMISSION)
            break

    for pattern in _FEE_PATTERNS:
        if re.search(pattern, msg):
            us.add(ScoreAction.ASKED_FEES)
            break

    for pattern in _COURSE_PATTERNS:
        if re.search(pattern, msg):
            us.add(ScoreAction.ASKED_COURSES)
            break

    return us.total
