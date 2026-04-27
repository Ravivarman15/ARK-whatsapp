"""
rag/escalation.py
─────────────────
Human escalation detection and admin notification system.

Detects when a WhatsApp user wants to speak with a human (counsellor,
admin, fee negotiation, etc.) and forwards the lead to the admin's
WhatsApp number via the AiSensy API.

Includes:
  - Per-user cooldown to prevent duplicate admin notifications
  - Complaint / dissatisfaction detection
  - Repeated confusion escalation (after N unanswered queries)
"""

from __future__ import annotations

import logging
import re
import time

from config.settings import get_settings

logger = logging.getLogger("ark.escalation")

# =====================================================================
# Escalation Trigger Phrases
# =====================================================================

# These phrases signal that the user wants human assistance.
# Checked via substring match on the normalised (lowercased) message.
ESCALATION_PHRASES = [
    "fees negotiation",
    "fee negotiation",
    "negotiate fees",
    "negotiate fee",
    "call admin",
    "call the admin",
    "talk to someone",
    "talk to a person",
    "speak with counsellor",
    "speak with counselor",
    "speak to counsellor",
    "speak to counselor",
    "speak with a person",
    "speak to a person",
    "speak to someone",
    "contact admin",
    "contact the admin",
    "call me",
    "call me back",
    "callback",
    "call back",
    "i want to speak",
    "i want to talk",
    "need human help",
    "human help",
    "connect me",
    "connect to admin",
    "can i talk",
    "can i speak",
    "want to meet",
    "schedule a call",
    "book a call",
    "counsellor",
    "counselor",
    "admission help",
    "admission enquiry",
    "admission inquiry",
]

# =====================================================================
# Complaint / Dissatisfaction Phrases
# =====================================================================

COMPLAINT_PHRASES = [
    "not happy",
    "not satisfied",
    "unhappy",
    "disappointed",
    "complaint",
    "bad experience",
    "poor service",
    "waste of time",
    "waste of money",
    "not good",
    "worst",
    "terrible",
    "horrible",
    "cheat",
    "fraud",
    "false",
    "misleading",
    "pathetic",
    "useless",
    "refund",
    "money back",
]

COMPLAINT_PATTERNS = [
    r"\b(complaint|complain|dissatisf|disappoint)\b",
    r"\b(refund|money\s+back)\b",
    r"\b(waste\s+of\s+(time|money))\b",
    r"\b(not\s+(happy|satisfied|good))\b",
    r"\b(cheat|fraud|mislead)\b",
]

# =====================================================================
# Intent Detection
# =====================================================================

def detect_human_request(message: str) -> bool:
    """
    Detect whether the user message indicates a desire for human
    assistance or admin contact.

    Uses keyword/phrase matching on the normalised message.

    Args:
        message: The raw user message.

    Returns:
        True if escalation intent is detected, False otherwise.
    """
    normalised = message.lower().strip()

    # Direct phrase matching
    for phrase in ESCALATION_PHRASES:
        if phrase in normalised:
            return True

    # Regex patterns for common variations
    patterns = [
        r"\bcall\s+me\b",
        r"\bspeak\s+(to|with)\b",
        r"\btalk\s+(to|with)\b",
        r"\bcontact\s+(me|admin|number)\b",
        r"\bfees?\s+negoti",
        r"\bneed\s+(a\s+)?human\b",
        r"\bconnect\s+(me|to)\b",
    ]
    for pattern in patterns:
        if re.search(pattern, normalised):
            return True

    return False


def detect_complaint(message: str) -> bool:
    """
    Detect whether the message contains a complaint or expression
    of dissatisfaction that should trigger escalation.

    Args:
        message: The raw user message.

    Returns:
        True if complaint is detected.
    """
    normalised = message.lower().strip()

    for phrase in COMPLAINT_PHRASES:
        if phrase in normalised:
            return True

    for pattern in COMPLAINT_PATTERNS:
        if re.search(pattern, normalised):
            return True

    return False


# =====================================================================
# Repeated Confusion Tracker
# =====================================================================

# Stores {user_id: confusion_count}
_confusion_counts: dict[str, int] = {}


def record_confusion(user_id: str) -> int:
    """
    Record that the bot couldn't answer a user's question.

    Returns the updated confusion count.
    """
    _confusion_counts[user_id] = _confusion_counts.get(user_id, 0) + 1
    count = _confusion_counts[user_id]
    logger.debug("CONFUSION | user=%s | count=%d", user_id, count)
    return count


def reset_confusion(user_id: str) -> None:
    """Reset the confusion counter for a user (called on successful answer)."""
    _confusion_counts.pop(user_id, None)


def should_escalate_confusion(user_id: str) -> bool:
    """
    Check if the user has hit the confusion threshold and should
    be escalated to a human.
    """
    s = get_settings()
    count = _confusion_counts.get(user_id, 0)
    return count >= s.CONFUSION_ESCALATION_THRESHOLD


# =====================================================================
# Cooldown System (prevents duplicate notifications)
# =====================================================================

# Stores {phone_number: last_escalation_timestamp}
_cooldown_store: dict[str, float] = {}
COOLDOWN_SECONDS = 600  # 10 minutes


def _is_on_cooldown(phone: str) -> bool:
    """Check if this phone number is within the cooldown window."""
    last_time = _cooldown_store.get(phone)
    if last_time is None:
        return False
    return (time.time() - last_time) < COOLDOWN_SECONDS


def _set_cooldown(phone: str) -> None:
    """Record an escalation event for this phone number."""
    _cooldown_store[phone] = time.time()


# =====================================================================
# Admin Notification
# =====================================================================

def _is_valid_admin_phone(phone: str) -> tuple[bool, str]:
    """
    Validate that the configured admin phone is an actual E.164 number,
    not an empty string or a placeholder like '919xxxxxxxxx'.

    Returns:
        (is_valid, reason) — reason is populated only on failure.
    """
    if not phone:
        return False, "ADMIN_WHATSAPP_NUMBER is empty"
    digits = phone.lstrip("+")
    if not digits.isdigit():
        return False, f"ADMIN_WHATSAPP_NUMBER contains non-digits: {phone!r}"
    if len(digits) < 10:
        return False, f"ADMIN_WHATSAPP_NUMBER too short ({len(digits)} digits): {phone!r}"
    return True, ""


async def _deliver_admin_message(
    admin_phone: str,
    message: str,
    *,
    context: str,
) -> bool:
    """
    Deliver a message to the admin via the AiSensy Project Messages API
    using a pre-approved UTILITY template (``admin_alerts`` by default).

    Previously this routed through the Campaign API, which produced
    high dashboard failure rates on 1:1 transactional alerts. All admin
    alerts (escalations, hot leads, qualified-lead summaries) now share
    the Project API path with retry + safe text fallback — see
    ``rag.whatsapp_sender.send_admin_alert`` for the full strategy.
    """
    from rag.whatsapp_sender import send_admin_alert

    ok = await send_admin_alert(admin_phone, message)
    if ok:
        logger.info("ADMIN_NOTIFIED | %s", context)
    else:
        logger.error("ADMIN_NOTIFY_FAIL | %s", context)
    return ok


async def notify_admin(
    user_phone: str,
    user_message: str,
    user_name: str = "Unknown",
    lead_type: str = "General Enquiry",
    reason: str = "",
) -> bool:
    """
    Send a lead notification to the admin's WhatsApp number via AiSensy.

    Respects the cooldown — if the same user triggered escalation
    within the last 10 minutes, the duplicate notification is skipped.

    Args:
        user_phone:   The user's phone number.
        user_message: The original message that triggered escalation.
        user_name:    The user's name (if available from AiSensy payload).
        lead_type:    Classified lead type (Fee Negotiation, Callback, etc.).
        reason:       Additional escalation reason (complaint, confusion, etc.).

    Returns:
        True if notification was sent, False if skipped (cooldown or error).
    """
    s = get_settings()

    # Check cooldown
    if _is_on_cooldown(user_phone):
        logger.info(
            "ESCALATION_COOLDOWN | user=%s | skipped (within %ds window)",
            user_phone, COOLDOWN_SECONDS,
        )
        return False

    admin_phone = s.ADMIN_WHATSAPP_NUMBER
    ok, reason_invalid = _is_valid_admin_phone(admin_phone)
    if not ok:
        logger.error(
            "ADMIN_NOTIFY_BLOCKED | user=%s | %s — fix .env before alerts can be sent",
            user_phone, reason_invalid,
        )
        return False

    if not s.AISENSY_API_KEY:
        logger.error(
            "ADMIN_NOTIFY_BLOCKED | user=%s | AISENSY_API_KEY not set",
            user_phone,
        )
        return False

    # Build the admin notification message
    reason_line = f"\n*Reason:* {reason}" if reason else ""
    admin_message = (
        "\U0001f6a8 *New Lead Request*\n\n"
        f"*Type:* {lead_type}\n"
        f"*Name:* {user_name}\n"
        f"*Phone:* {user_phone}\n"
        f"{reason_line}\n"
        f"*Message:*\n{user_message}\n\n"
        "*Action Required:*\n"
        "Please contact this student."
    )

    sent = await _deliver_admin_message(
        admin_phone, admin_message,
        context=f"escalation user={user_phone} reason={reason or lead_type}",
    )
    if sent:
        _set_cooldown(user_phone)
    return sent


# =====================================================================
# User Escalation Reply
# =====================================================================

ESCALATION_REPLY = (
    "Thank you for your interest! \U0001f64f\n\n"
    "Our academic counsellor will contact you shortly. "
    "If urgent, you can also call us directly.\n\n"
    "— Team ARK Learning Arena"
)

COMPLAINT_REPLY = (
    "We're sorry to hear about your concern. \U0001f64f\n\n"
    "We take all feedback seriously. Our team will "
    "reach out to you shortly to address this.\n\n"
    "— Team ARK Learning Arena"
)
