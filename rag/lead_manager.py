"""
rag/lead_manager.py
───────────────────
Full lead management pipeline for the ARK Learning Arena WhatsApp bot.

Features:
  - Lead type classification (Fee Negotiation, Callback, Demo, etc.)
  - Hot lead detection (high-intent signals)
  - Course interest detection
  - Multi-step lead qualification conversation flow
  - Smart input validation (intent detection + field validators)
  - Lead scoring integration (numeric 0–100+)
  - Student segmentation (Junior Foundation / Foundation / NEET Core / Repeater)
  - Decision stage tracking
  - Parent persona tracking
  - Counsellor summary generation
  - Lead storage in Supabase (ark_leads table)
  - Per-user state machine for qualification steps
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from config.settings import get_settings

logger = logging.getLogger("ark.leads")


# =====================================================================
# Lead Types
# =====================================================================

class LeadType(str, Enum):
    FEE_NEGOTIATION = "Fee Negotiation"
    CALLBACK_REQUEST = "Callback Request"
    DEMO_CLASS = "Demo Class"
    ADMISSION_ENQUIRY = "Admission Enquiry"
    GENERAL_ENQUIRY = "General Enquiry"


class LeadPriority(str, Enum):
    HIGH = "HIGH"
    NORMAL = "NORMAL"


# =====================================================================
# Lead Classification
# =====================================================================

def classify_lead(message: str) -> LeadType:
    """
    Classify the type of lead based on the user's message.

    Args:
        message: The raw user message.

    Returns:
        The detected LeadType enum value.
    """
    msg = message.lower().strip()

    # Fee negotiation
    if any(kw in msg for kw in ["fee", "fees", "negotiat", "discount", "price", "cost", "reduce", "concession", "scholarship"]):
        return LeadType.FEE_NEGOTIATION

    # Callback / contact request
    if any(kw in msg for kw in ["call me", "call back", "callback", "contact me", "talk to", "speak to", "speak with", "connect me"]):
        return LeadType.CALLBACK_REQUEST

    # Demo class
    if any(kw in msg for kw in ["demo", "trial class", "trial", "sample class", "free class"]):
        return LeadType.DEMO_CLASS

    # Admission enquiry
    if any(kw in msg for kw in ["admission", "enrol", "enroll", "join", "registration", "how to join", "seat", "apply"]):
        return LeadType.ADMISSION_ENQUIRY

    return LeadType.GENERAL_ENQUIRY


# =====================================================================
# Hot Lead Detection
# =====================================================================

HOT_LEAD_PHRASES = [
    # ── English ────────────────────────────────────────────────
    "fees negotiation", "fee negotiation",
    "negotiate fees", "reduce fees", "discount",
    "call me now", "call me back", "callback",
    "how to join", "when does admission start",
    "send fee structure", "fee structure",
    "i want to join", "i want admission",
    "want to enroll", "want to enrol",
    "book a seat", "reserve seat",
    "can you reduce", "any concession",
    "scholarship", "pay fees",
    "admission open", "last date",
    "confirm admission",
    # ── Thanglish (Tamil in English script) ────────────────────
    "join panna", "join pannanum", "join pannanumnu",
    "chernu", "chernanum", "serum",
    "admission venum", "admission aagumu",
    "padikka vendum", "padikka venum",
    "fees evlo", "fees enna", "fees yavalo",
    "enroll panna", "enroll pannanum",
]

HOT_LEAD_PATTERNS = [
    # ── English ────────────────────────────────────────────────
    r"\bcall\s+me\s+now\b",
    r"\bfees?\s+(negotiat|reduc|discount|concess)",
    r"\bhow\s+to\s+join\b",
    r"\bwant\s+to\s+(join|enrol|enroll|admit)",
    r"\bsend\s+(fee|fees|price)\b",
    r"\badmission\s+(open|start|last|confirm)",
    r"\bbook\s+(a\s+)?seat\b",
    # ── Thanglish ──────────────────────────────────────────────
    # "NEET-la join pannanum", "course-la chernanum", etc.
    r"\b(neet|jee|course|class|batch|coaching)\s*(la|le)?\s*(join|chern|enroll|padi)",
    # "join pannanum", "chernanum", "padikanum"
    r"\b(join|chern|padi|enroll)\s*(pann|an)",
    # "admission venum / aagum", "seat venum"
    r"\b(admission|seat|fees?)\s+(venum|aagum|iruku)",
    # "fees evlo / enna / yavalo"
    r"\bfees?\s+(evlo|enna|yavalo|eppadi)",
]


def detect_hot_lead(message: str) -> bool:
    """
    Detect whether the user is a high-intent (hot) lead.

    Hot leads show strong purchase / admission intent and should
    be notified to admin immediately with HIGH priority.

    Args:
        message: The raw user message.

    Returns:
        True if the message signals hot lead intent.
    """
    msg = message.lower().strip()

    for phrase in HOT_LEAD_PHRASES:
        if phrase in msg:
            return True

    for pattern in HOT_LEAD_PATTERNS:
        if re.search(pattern, msg):
            return True

    return False


# =====================================================================
# Course Interest Detection
# =====================================================================

COURSE_KEYWORDS = {
    "neet": "NEET",
    "medical": "NEET",
    "jee": "JEE",
    "engineering": "JEE",
    "foundation": "Foundation",
    "tuition": "Tuition",
    "school tuition": "Tuition",
    "olympiad": "Olympiad",
    "class 11": "Class 11-12",
    "class 12": "Class 11-12",
    "class 10": "Class 10",
    "class 9": "Class 9-10",
    "class 8": "Class 8",
}

# Admission-intent keywords that must be present alongside course keywords
# to trigger qualification (prevents false positives from factual questions)
_ADMISSION_CONTEXT_WORDS = {
    "join", "admission", "admit", "enroll", "enrol", "register",
    "fees", "fee", "demo", "apply", "seat", "interested",
    "want", "need", "looking", "start",
}


def detect_course_interest(message: str) -> Optional[str]:
    """
    Detect if the user is interested in a specific course
    AND has admission intent.

    Only returns a course if the message also contains an
    admission-context word — prevents triggering on pure
    factual questions like "which school is this?"

    Args:
        message: The raw user message.

    Returns:
        The course name if detected with intent, None otherwise.
    """
    msg = message.lower().strip()
    words = set(msg.split())

    # Must have at least one admission-context word
    has_admission_context = bool(words & _ADMISSION_CONTEXT_WORDS)
    if not has_admission_context:
        return None

    for keyword, course in COURSE_KEYWORDS.items():
        if keyword in msg:
            return course
    return None


# =====================================================================
# Lead Qualification State Machine
# =====================================================================

class QualStep(str, Enum):
    """Steps in the lead qualification conversation."""
    ASK_NAME = "ask_name"
    ASK_CLASS = "ask_class"
    ASK_SCHOOL = "ask_school"
    ASK_PARENT_PHONE = "ask_parent_phone"
    COMPLETE = "complete"


# Prompts for each qualification step — WhatsApp-friendly, 1-2 lines max
QUAL_PROMPTS = {
    QualStep.ASK_NAME: "Great! What's the student's name?",
    QualStep.ASK_CLASS: "Which class is the student in?",
    QualStep.ASK_SCHOOL: "Which school does the student study in?",
    QualStep.ASK_PARENT_PHONE: "Please share a contact number so our counsellor can call.",
    QualStep.COMPLETE: "Thanks! Our counsellor will call you shortly \U0001f44d",
}

# The order in which fields are collected
QUAL_FLOW = [
    QualStep.ASK_NAME,
    QualStep.ASK_CLASS,
    QualStep.ASK_SCHOOL,
    QualStep.ASK_PARENT_PHONE,
    QualStep.COMPLETE,
]


@dataclass
class LeadData:
    """Temporary storage for a lead being qualified."""
    phone: str = ""
    student_name: str = ""
    student_class: str = ""
    school: str = ""
    parent_phone: str = ""
    course: str = ""
    lead_type: str = ""
    priority: str = LeadPriority.NORMAL.value
    current_step: QualStep = QualStep.ASK_NAME
    started_at: float = 0.0
    # ── Intelligence fields ──────────────────────────────────────
    segment: str = ""          # StudentSegment value
    stage: str = ""            # DecisionStage value
    lead_score: int = 0        # Numeric score
    lead_score_type: str = ""  # LeadScoreType value (COLD/WARM/HOT/VERY_HOT)
    persona: str = ""          # ParentPersona value
    concern: str = ""          # Key concern (fees / weak student / urgent)
    internal_flags: list = field(default_factory=list)  # [CONFUSED_PARENT], [PRICE_SENSITIVE], etc.


# In-memory store: user_id -> LeadData
_active_leads: dict[str, LeadData] = {}

# Timeout: if qualification is not completed within 30 min, reset
QUAL_TIMEOUT = 1800


def is_in_qualification(user_id: str) -> bool:
    """Check if the user is currently in a qualification flow."""
    lead = _active_leads.get(user_id)
    if lead is None:
        return False
    # Check timeout
    if time.time() - lead.started_at > QUAL_TIMEOUT:
        _active_leads.pop(user_id, None)
        return False
    return lead.current_step != QualStep.COMPLETE


def start_lead_qualification(
    user_id: str,
    phone: str = "",
    course: str = "",
) -> str:
    """
    Start or resume a lead qualification flow for the user.

    If the user already has an active (non-timed-out) qualification,
    resumes from the current step instead of restarting.

    Returns the next question prompt.
    """
    from rag.scoring import update_score, ScoreAction, get_score, get_lead_type

    # ── Resume existing flow if active ───────────────────────────
    existing = _active_leads.get(user_id)
    if existing is not None:
        elapsed = time.time() - existing.started_at
        if elapsed <= QUAL_TIMEOUT and existing.current_step != QualStep.COMPLETE:
            # Update course if a more specific one was detected
            if course and existing.course in ("", "General"):
                existing.course = course
            logger.info(
                "LEAD_QUAL_RESUME | user=%s | step=%s | course=%s",
                user_id, existing.current_step.value, existing.course,
            )
            return QUAL_PROMPTS[existing.current_step]

    # ── Start new flow ───────────────────────────────────────────
    _active_leads[user_id] = LeadData(
        phone=phone,
        course=course or "General",
        lead_type=LeadType.ADMISSION_ENQUIRY.value,
        priority=LeadPriority.NORMAL.value,
        current_step=QualStep.ASK_NAME,
        started_at=time.time(),
    )

    # Score: course interest detected
    update_score(user_id, ScoreAction.ASKED_COURSES)

    logger.info("LEAD_QUAL_START | user=%s | course=%s", user_id, course)
    return QUAL_PROMPTS[QualStep.ASK_NAME]


def store_lead_data(user_id: str, value: str) -> str:
    """
    Store a validated answer for the current qualification step
    and advance to the next step.

    NOTE: Prefer using process_qualification_message() instead,
    which validates the input before calling this function.

    Returns the next prompt (or the completion message).
    """
    from rag.scoring import update_score, ScoreAction, get_score, get_lead_type
    from rag.segmentation import detect_segment, get_segment_focus

    lead = _active_leads.get(user_id)
    if lead is None:
        return ""

    step = lead.current_step
    value = value.strip()

    # Store the answer in the appropriate field
    if step == QualStep.ASK_NAME:
        lead.student_name = value
        update_score(user_id, ScoreAction.SHARED_NAME)

    elif step == QualStep.ASK_CLASS:
        lead.student_class = value
        update_score(user_id, ScoreAction.SHARED_CLASS)

        # Detect and store segment
        segment = detect_segment(value)
        lead.segment = segment.value
        logger.info("SEGMENT_DETECTED | user=%s | segment=%s", user_id, segment.value)

    elif step == QualStep.ASK_SCHOOL:
        lead.school = value
        update_score(user_id, ScoreAction.SHARED_SCHOOL)

    elif step == QualStep.ASK_PARENT_PHONE:
        lead.parent_phone = value
        update_score(user_id, ScoreAction.SHARED_PHONE)

    # Update lead score and type on the lead data
    lead.lead_score = get_score(user_id)
    lead.lead_score_type = get_lead_type(user_id).value

    # Advance to next step
    current_idx = QUAL_FLOW.index(step)
    next_step = QUAL_FLOW[current_idx + 1]
    lead.current_step = next_step

    logger.info(
        "LEAD_QUAL_STEP | user=%s | step=%s | value=%s | next=%s | score=%d",
        user_id, step.value, value[:50], next_step.value, lead.lead_score,
    )

    return QUAL_PROMPTS[next_step]


def process_qualification_message(user_id: str, message: str) -> Optional[str]:
    """
    Process a message from a user who is in the qualification flow.

    Applies intent detection and field validation before storing data.

    Returns:
        str  — a reply to send (next prompt or validation error)
        None — the message is a question; caller should run RAG + re-prompt
    """
    from rag.input_validator import detect_user_intent, validate_field

    lead = _active_leads.get(user_id)
    if lead is None:
        return ""

    step = lead.current_step
    msg = message.strip()

    # ── 1. Intent detection ──────────────────────────────────────
    intent = detect_user_intent(msg)

    if intent == "question":
        logger.info(
            "QUAL_QUESTION | user=%s | step=%s | msg=\"%s\"",
            user_id, step.value, msg[:60],
        )
        return None  # Signal: run RAG, then re-prompt

    if intent == "command":
        # Treat commands as pass-through — re-prompt the same field
        logger.info(
            "QUAL_COMMAND | user=%s | step=%s | msg=\"%s\"",
            user_id, step.value, msg[:60],
        )
        return QUAL_PROMPTS[step]

    # ── 2. Field validation ──────────────────────────────────────
    is_valid, cleaned_or_error = validate_field(step.value, msg)

    if not is_valid:
        logger.info(
            "QUAL_INVALID | user=%s | step=%s | msg=\"%s\"",
            user_id, step.value, msg[:60],
        )
        return cleaned_or_error  # This is the error/re-prompt message

    # ── 3. Store validated value and advance ──────────────────────
    return store_lead_data(user_id, cleaned_or_error)


def get_current_qual_prompt(user_id: str) -> str:
    """
    Get the current qualification prompt for a user
    (used to re-prompt after answering a mid-flow question).
    """
    lead = _active_leads.get(user_id)
    if lead is None:
        return ""
    return QUAL_PROMPTS.get(lead.current_step, "")


# Soft re-prompts — prefix with "Also, " so a mid-flow answered question
# transitions naturally back to the pending field instead of feeling
# like the bot ignored what the user just asked.
_SOFT_REPROMPTS = {
    QualStep.ASK_NAME: "Also, can I get the student's name to help you further?",
    QualStep.ASK_CLASS: "Also, which class is the student in?",
    QualStep.ASK_SCHOOL: "Also, which school does the student study in?",
    QualStep.ASK_PARENT_PHONE: "Also, please share a contact number so our counsellor can call.",
}


def get_soft_reprompt(user_id: str) -> str:
    """
    Return a "Also, can I get…"-style re-prompt for the user's current
    qualification step. Used after the bot has answered a mid-flow
    question and wants to gently steer back to the pending field.

    Returns "" if the user has no active qualification.
    """
    lead = _active_leads.get(user_id)
    if lead is None:
        return ""
    return _SOFT_REPROMPTS.get(lead.current_step, "")


def complete_lead(user_id: str) -> Optional[LeadData]:
    """
    Complete and return the lead data, then remove from active store.

    Only pops the lead if the qualification flow has reached COMPLETE.
    Returns the LeadData if completed, None otherwise.
    """
    lead = _active_leads.get(user_id)
    if lead and lead.current_step == QualStep.COMPLETE:
        _active_leads.pop(user_id, None)
        logger.info(
            "LEAD_COMPLETE | user=%s | name=%s | course=%s | score=%d | segment=%s",
            user_id, lead.student_name, lead.course,
            lead.lead_score, lead.segment,
        )
        return lead
    return None


def get_lead_data(user_id: str) -> Optional[LeadData]:
    """Get the current lead data for a user (read-only)."""
    return _active_leads.get(user_id)


# =====================================================================
# Counsellor Summary Generation
# =====================================================================

def generate_counsellor_summary(lead: LeadData) -> str:
    """
    Generate a formatted pre-brief summary for the academic counsellor.

    Includes all collected data + intelligence tags so the counsellor
    doesn't start the call blind.
    """
    lines = [
        "\U0001f4cb *Lead Summary for Counsellor*\n",
        f"*Name:* {lead.student_name or 'Not provided'}",
        f"*Class:* {lead.student_class or 'Not provided'}",
        f"*School:* {lead.school or 'Not provided'}",
        f"*Phone:* {lead.phone or 'Not provided'}",
        f"*Parent Phone:* {lead.parent_phone or 'Not provided'}",
        f"*Course Interest:* {lead.course or 'General'}",
        "",
        f"*Segment:* {lead.segment or 'Unknown'}",
        f"*Stage:* {lead.stage or 'Unknown'}",
        f"*Lead Score:* {lead.lead_score} ({lead.lead_score_type or 'N/A'})",
        f"*Persona:* {lead.persona or 'General'}",
        f"*Concern:* {lead.concern or 'None identified'}",
    ]

    if lead.internal_flags:
        lines.append(f"*Flags:* {', '.join(lead.internal_flags)}")

    lines.append("\n*Source:* WhatsApp AI Bot")
    return "\n".join(lines)


# =====================================================================
# Internal Flag Management
# =====================================================================

def add_internal_flag(user_id: str, flag: str) -> None:
    """Add an internal flag to the user's lead data."""
    lead = _active_leads.get(user_id)
    if lead and flag not in lead.internal_flags:
        lead.internal_flags.append(flag)
        logger.info("FLAG_ADDED | user=%s | flag=%s", user_id, flag)


def set_concern(user_id: str, concern: str) -> None:
    """Set the primary concern for the user's lead data."""
    lead = _active_leads.get(user_id)
    if lead:
        lead.concern = concern


def update_lead_intelligence(user_id: str, stage: str = "", persona: str = "") -> None:
    """Update intelligence fields on an active lead."""
    lead = _active_leads.get(user_id)
    if lead is None:
        return
    if stage:
        lead.stage = stage
    if persona:
        lead.persona = persona
    # Always sync latest score
    from rag.scoring import get_score, get_lead_type
    lead.lead_score = get_score(user_id)
    lead.lead_score_type = get_lead_type(user_id).value


# =====================================================================
# Supabase Lead Storage
# =====================================================================

# Columns that every ark_leads table is expected to have. If the insert
# with intelligence fields fails because the schema is older, we retry
# with just these core columns so the lead is never lost.
_CORE_LEAD_COLUMNS = {
    "phone", "student_name", "class", "school", "parent_phone",
    "course", "lead_type", "priority", "message", "created_at",
}


async def save_lead_to_db(lead: LeadData) -> bool:
    """
    Save a completed lead to the Supabase `ark_leads` table.

    First tries the full row (core + intelligence columns). If Supabase
    rejects it because the table is missing the intelligence columns
    (PGRST204 "Could not find the 'X' column"), retries with only the
    core columns so the lead still lands.

    Returns True if saved successfully, False otherwise.
    """
    from rag.retriever import get_supabase_client

    client = get_supabase_client()
    full_row = {
        "phone": lead.phone,
        "student_name": lead.student_name,
        "class": lead.student_class,
        "school": lead.school,
        "parent_phone": lead.parent_phone,
        "course": lead.course,
        "lead_type": lead.lead_type,
        "priority": lead.priority,
        "message": "",
        "created_at": datetime.now().isoformat(),
        # Intelligence fields
        "segment": lead.segment,
        "stage": lead.stage,
        "lead_score": lead.lead_score,
        "lead_score_type": lead.lead_score_type,
        "persona": lead.persona,
        "concern": lead.concern,
    }

    try:
        client.table("ark_leads").insert(full_row).execute()
        logger.info(
            "LEAD_SAVED | phone=%s | name=%s | score=%d",
            lead.phone, lead.student_name, lead.lead_score,
        )
        return True
    except Exception as e:
        err_msg = str(e)
        schema_mismatch = "PGRST204" in err_msg or "Could not find the" in err_msg
        if not schema_mismatch:
            logger.error("Failed to save lead to DB: %s", e)
            return False

        logger.warning(
            "LEAD_SAVE_SCHEMA_MISMATCH | %s — retrying with core columns only. "
            "Run scripts/add_ark_leads_intelligence_columns.sql on Supabase to "
            "stop losing intelligence fields.",
            err_msg[:200],
        )

    # Retry path: insert only columns that definitely exist in the table
    core_row = {k: v for k, v in full_row.items() if k in _CORE_LEAD_COLUMNS}
    try:
        client.table("ark_leads").insert(core_row).execute()
        logger.info(
            "LEAD_SAVED | phone=%s | name=%s | score=%d | mode=core_only",
            lead.phone, lead.student_name, lead.lead_score,
        )
        return True
    except Exception as e:
        logger.error("Failed to save lead to DB (core retry): %s", e)
        return False


# =====================================================================
# Admin Notification (qualified + hot leads)
# =====================================================================

async def notify_admin_qualified_lead(lead: LeadData) -> bool:
    """Send a qualified lead notification to the admin's WhatsApp."""
    from rag.escalation import _deliver_admin_message, _is_valid_admin_phone

    s = get_settings()
    admin_phone = s.ADMIN_WHATSAPP_NUMBER
    ok, reason = _is_valid_admin_phone(admin_phone)
    if not ok:
        logger.error("ADMIN_NOTIFY_BLOCKED | qualified_lead | %s", reason)
        return False
    if not s.AISENSY_API_KEY:
        logger.error("ADMIN_NOTIFY_BLOCKED | qualified_lead | AISENSY_API_KEY not set")
        return False

    admin_message = generate_counsellor_summary(lead)
    return await _deliver_admin_message(
        admin_phone, admin_message,
        context=f"qualified_lead phone={lead.phone} name={lead.student_name}",
    )


async def notify_admin_hot_lead(
    user_phone: str,
    user_message: str,
    lead_type: str = "",
) -> bool:
    """Send a hot lead notification to the admin's WhatsApp."""
    from rag.escalation import _deliver_admin_message, _is_valid_admin_phone

    s = get_settings()
    admin_phone = s.ADMIN_WHATSAPP_NUMBER
    ok, reason = _is_valid_admin_phone(admin_phone)
    if not ok:
        logger.error(
            "ADMIN_NOTIFY_BLOCKED | hot_lead | user=%s | %s",
            user_phone, reason,
        )
        return False
    if not s.AISENSY_API_KEY:
        logger.error(
            "ADMIN_NOTIFY_BLOCKED | hot_lead | user=%s | AISENSY_API_KEY not set",
            user_phone,
        )
        return False

    admin_message = (
        "\U0001f525 *HOT LEAD*\n\n"
        f"*Phone:* {user_phone}\n"
        f"*Message:* {user_message}\n"
        f"*Type:* {lead_type}\n\n"
        "*Priority:* HIGH\n\n"
        "*Contact immediately.*"
    )

    return await _deliver_admin_message(
        admin_phone, admin_message,
        context=f"hot_lead user={user_phone} type={lead_type}",
    )
