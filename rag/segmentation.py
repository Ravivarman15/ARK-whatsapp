"""
rag/segmentation.py
───────────────────
Student segmentation engine for the ARK Learning Arena WhatsApp bot.

Classifies students by class into segments with tailored bot behaviour
and automatic program recommendations.

Segments:
    JUNIOR_FOUNDATION — Class 6–8
    FOUNDATION        — Class 9–10
    NEET_CORE         — Class 11–12
    REPEATER          — Drop year / repeater
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Optional

logger = logging.getLogger("ark.segment")


# =====================================================================
# Segment Enum
# =====================================================================

class StudentSegment(str, Enum):
    JUNIOR_FOUNDATION = "JUNIOR_FOUNDATION"
    FOUNDATION = "FOUNDATION"
    NEET_CORE = "NEET_CORE"
    REPEATER = "REPEATER"
    UNKNOWN = "UNKNOWN"


# =====================================================================
# Segment Detection
# =====================================================================

def detect_segment(class_str: str) -> StudentSegment:
    """
    Classify a student into a segment based on their class.

    Args:
        class_str: The student's class (e.g. "Class 11", "10", "Repeater").

    Returns:
        The detected StudentSegment.
    """
    text = class_str.lower().strip()

    # Check for repeater/drop year
    if any(kw in text for kw in ["repeat", "drop", "dropper", "gap year", "re-attempt"]):
        return StudentSegment.REPEATER

    # Extract numeric class
    match = re.search(r"(\d{1,2})", text)
    if match:
        class_num = int(match.group(1))
        if 6 <= class_num <= 8:
            return StudentSegment.JUNIOR_FOUNDATION
        elif 9 <= class_num <= 10:
            return StudentSegment.FOUNDATION
        elif 11 <= class_num <= 12:
            return StudentSegment.NEET_CORE

    return StudentSegment.UNKNOWN


# =====================================================================
# Segment Focus (Bot Behaviour)
# =====================================================================

_SEGMENT_FOCUS: dict[StudentSegment, str] = {
    StudentSegment.JUNIOR_FOUNDATION: (
        "Student is in Junior Foundation (Class 6–8). "
        "Focus on: building strong basics in Science & Maths, "
        "confidence building, early discipline, and foundational skill development. "
        "Highlight the ARK Nestlings and Foundation programs."
    ),
    StudentSegment.FOUNDATION: (
        "Student is in Foundation (Class 9–10). "
        "Focus on: NEET foundation preparation, concept clarity, "
        "early competitive advantage, and board exam excellence. "
        "Recommend the Foundation Program with early NEET prep."
    ),
    StudentSegment.NEET_CORE: (
        "Student is in NEET Core (Class 11–12). "
        "This is the MOST IMPORTANT segment. "
        "Focus on: NEET score targets (600+), structured 2-year program, "
        "weekly testing, performance analytics, and personalized mentoring. "
        "Recommend the 2-Year NEET Intensive Program."
    ),
    StudentSegment.REPEATER: (
        "Student is a NEET Repeater. "
        "Focus on: gap analysis, weak area correction, "
        "intensive test-based training, and strategy refinement. "
        "Recommend the NEET Repeaters Batch."
    ),
    StudentSegment.UNKNOWN: "",
}


def get_segment_focus(segment: StudentSegment) -> str:
    """Get the bot behaviour instruction for a given segment."""
    return _SEGMENT_FOCUS.get(segment, "")


# =====================================================================
# Program Recommendation
# =====================================================================

_PROGRAM_RECOMMENDATIONS: dict[StudentSegment, str] = {
    StudentSegment.JUNIOR_FOUNDATION: (
        "Based on the student's class, we recommend our **Foundation Program** "
        "which builds a strong base in Science and Mathematics, "
        "giving a 3-year head start for NEET preparation."
    ),
    StudentSegment.FOUNDATION: (
        "Based on the student's class, we recommend starting with our "
        "**Foundation + Early NEET Prep** program to build concept clarity "
        "and competitive exam readiness."
    ),
    StudentSegment.NEET_CORE: (
        "Based on the student's class, we recommend our structured "
        "**2-Year NEET Coaching Program** designed for consistent preparation "
        "and high scores (600+)."
    ),
    StudentSegment.REPEATER: (
        "We recommend our **NEET Repeaters Batch** with intensive "
        "test-based training, gap analysis, and focused strategy "
        "to improve performance."
    ),
    StudentSegment.UNKNOWN: "",
}


def get_program_recommendation(segment: StudentSegment) -> str:
    """Get the recommended program description for a given segment."""
    return _PROGRAM_RECOMMENDATIONS.get(segment, "")


# =====================================================================
# Detect Segment from Message Text
# =====================================================================

_REPEATER_KEYWORDS = [
    "repeat", "repeater", "drop year", "dropper", "gap year",
    "re-attempt", "reattempt", "failed neet", "retake",
]


def detect_segment_from_message(message: str) -> Optional[StudentSegment]:
    """
    Try to detect a student segment from a free-text message.

    Returns None if no segment can be determined.
    """
    msg = message.lower().strip()

    # Check repeater keywords first
    for kw in _REPEATER_KEYWORDS:
        if kw in msg:
            return StudentSegment.REPEATER

    # Check for class mentions
    match = re.search(r"(?:class\s*)?(\d{1,2})(?:th|st|nd|rd)?\s*(?:class|std)?", msg)
    if match:
        class_num = int(match.group(1))
        if 6 <= class_num <= 8:
            return StudentSegment.JUNIOR_FOUNDATION
        elif 9 <= class_num <= 10:
            return StudentSegment.FOUNDATION
        elif 11 <= class_num <= 12:
            return StudentSegment.NEET_CORE

    return None
