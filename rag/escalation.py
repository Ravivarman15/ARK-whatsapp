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
from datetime import datetime
from typing import Optional

import httpx

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
    if not admin_phone:
        logger.warning("ADMIN_WHATSAPP_NUMBER not set — cannot notify admin.")
        return False

    if not s.AISENSY_API_KEY:
        logger.warning("AISENSY_API_KEY not set — cannot send admin notification.")
        return False

    # Build the admin notification message
    timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
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

    # Ensure admin phone has '+' prefix
    destination = admin_phone if admin_phone.startswith("+") else f"+{admin_phone}"

    url = "https://backend.aisensy.com/campaign/t1/api/v2"
    payload = {
        "apiKey": s.AISENSY_API_KEY,
        "campaignName": s.AISENSY_CAMPAIGN_NAME,
        "destination": destination,
        "userName": "Admin",
        "templateParams": [admin_message],
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code in (200, 201):
                _set_cooldown(user_phone)
                logger.info(
                    "ESCALATION_TRIGGERED | user=%s | reason=%s | admin_notified=True",
                    user_phone, reason or lead_type,
                )
                return True
            else:
                logger.error(
                    "AiSensy admin notification failed (%d): %s",
                    resp.status_code, resp.text,
                )
                return False
    except Exception as e:
        logger.error("Failed to send admin notification: %s", e)
        return False


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
