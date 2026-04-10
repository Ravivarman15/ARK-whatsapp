"""
rag/followup_manager.py
───────────────────────
Automated follow-up system for the ARK Learning Arena WhatsApp bot.

Tracks user activity timestamps in Supabase and sends staged follow-up
messages to users who stop responding during a conversation.

All follow-ups are within the 24-hour WhatsApp session window:
    0 = No follow-up sent
    1 = First follow-up sent  (after 30 min inactivity)
    2 = Second follow-up sent (after 5 hr inactivity)
    3 = Third follow-up sent  (after 18 hr inactivity)

Follow-ups are suppressed when:
    - The user has replied recently
    - Lead qualification is already completed / in-progress
    - Admin escalation or hot lead detection has occurred
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from config.settings import get_settings
from rag.lead_manager import is_in_qualification

logger = logging.getLogger("ark.followup")


# =====================================================================
# Follow-Up Message Templates
# =====================================================================

FOLLOWUP_MESSAGES = {
    1: (
        "Just checking — would you like to book your "
        "Academic Diagnostic Assessment? 📚\n\n"
        "It's free and helps us understand the student's current level.\n\n"
        "— Team ARK Learning Arena"
    ),
    2: (
        "We have limited slots available this week for the "
        "assessment. 🎯\n\n"
        "Would you like to reserve one?\n\n"
        "Reply 'Yes' to get started or ask any question.\n\n"
        "— Team ARK Learning Arena"
    ),
    3: (
        "Let us know if you need any guidance. "
        "We'll be happy to help you plan the next step. 🙏\n\n"
        "If you prefer, we can directly schedule a quick "
        "counselling call for you.\n\n"
        "— Team ARK Learning Arena"
    ),
}

# Drop-off recovery messages (when user stops mid-qualification)
DROPOFF_MESSAGES = {
    "ask_name": (
        "Hi! We noticed you were interested in our program. "
        "Could you share the student's name so we can guide you better?"
    ),
    "ask_class": (
        "Could you also share which class the student is studying in? "
        "This helps us recommend the right program."
    ),
    "ask_school": (
        "Which school is the student studying in? "
        "We'd like to understand their current academic environment."
    ),
    "ask_parent_phone": (
        "Please share a contact number so our counsellor can guide you properly. "
        "You can share it later, but we won't be able to assist you fully without it."
    ),
}


# =====================================================================
# Record User Activity
# =====================================================================

async def record_user_activity(phone: str) -> None:
    """
    Record that a user just sent a message.

    Upserts the `ark_followups` row for this phone number:
      - Resets `last_message_time` to now
      - Resets `followup_stage` to 0
      - Sets `status` to 'active'

    This ensures any pending follow-ups are cancelled when the user
    re-engages.
    """
    if not phone:
        return

    try:
        from rag.retriever import get_supabase_client

        client = get_supabase_client()
        now = datetime.now(timezone.utc).isoformat()

        row = {
            "phone": phone,
            "last_message_time": now,
            "followup_stage": 0,
            "status": "active",
        }

        # Upsert — insert if new, update if phone already exists
        client.table("ark_followups").upsert(
            row, on_conflict="phone"
        ).execute()

        logger.debug("FOLLOWUP_ACTIVITY | phone=%s | reset", phone)
    except Exception as e:
        logger.error("Failed to record user activity: %s", e)


# =====================================================================
# Skip Logic
# =====================================================================

def should_skip_followup(phone: str) -> bool:
    """
    Determine whether follow-ups should be skipped for this user.

    Returns True if:
      - The user is in an active lead qualification flow
      - (Escalation / hot lead state is handled by checking the
        'status' column in the DB — rows marked 'escalated' or
        'completed' are excluded from the query itself.)
    """
    user_id = phone  # user_id == phone in the webhook
    if is_in_qualification(user_id):
        return True
    return False


# =====================================================================
# Send Follow-Up Message
# =====================================================================

async def send_followup_message(phone: str, stage: int) -> bool:
    """
    Send a follow-up message to the user via AiSensy and update
    the `followup_stage` in the database.

    Args:
        phone: The user's phone number.
        stage: The follow-up stage (1, 2, or 3).

    Returns:
        True if the message was sent successfully.
    """
    # Import here to avoid circular imports
    from api.main import send_whatsapp_message
    from rag.retriever import get_supabase_client

    s = get_settings()

    message = FOLLOWUP_MESSAGES.get(stage)
    if not message:
        logger.warning("No follow-up message template for stage %d", stage)
        return False

    try:
        # Send the WhatsApp message
        await send_whatsapp_message(phone, message)

        # Update stage in database
        client = get_supabase_client()
        update_data = {"followup_stage": stage}

        # If this is the final stage, mark as completed
        if stage >= s.MAX_FOLLOWUP_STAGE:
            update_data["status"] = "completed"

        client.table("ark_followups").update(update_data).eq(
            "phone", phone
        ).execute()

        logger.info("FOLLOWUP_SENT | phone=%s | stage=%d", phone, stage)
        return True
    except Exception as e:
        logger.error("Failed to send follow-up (stage %d) to %s: %s", stage, phone, e)
        return False


# =====================================================================
# Mark User as Escalated (called from webhook on hot lead / escalation)
# =====================================================================

async def mark_followup_escalated(phone: str) -> None:
    """
    Mark a user's follow-up record as 'escalated' so no further
    follow-ups are sent. Called when a hot lead or human escalation
    is detected.
    """
    if not phone:
        return

    try:
        from rag.retriever import get_supabase_client

        client = get_supabase_client()
        client.table("ark_followups").update(
            {"status": "escalated"}
        ).eq("phone", phone).execute()

        logger.info("FOLLOWUP_ESCALATED | phone=%s", phone)
    except Exception as e:
        logger.error("Failed to mark follow-up escalated: %s", e)


# =====================================================================
# Mark User as Completed (called when lead qualification finishes)
# =====================================================================

async def mark_followup_completed(phone: str) -> None:
    """
    Mark a user's follow-up record as 'completed' so no further
    follow-ups are sent. Called when lead qualification completes.
    """
    if not phone:
        return

    try:
        from rag.retriever import get_supabase_client

        client = get_supabase_client()
        client.table("ark_followups").update(
            {"status": "completed"}
        ).eq("phone", phone).execute()

        logger.info("FOLLOWUP_COMPLETED | phone=%s", phone)
    except Exception as e:
        logger.error("Failed to mark follow-up completed: %s", e)


# =====================================================================
# Check Follow-Ups (called periodically by the scheduler)
# =====================================================================

async def check_followups() -> int:
    """
    Query the database for users who are due for a follow-up message.

    Logic:
      - Only check rows with status='active'
      - Stage 0 → if inactivity > STAGE1_DELAY  → send stage 1
      - Stage 1 → if inactivity > STAGE2_DELAY  → send stage 2
      - Stage 2 → if inactivity > STAGE3_DELAY  → send stage 3
      - Stage 3 → already maxed out (skipped by query)

    Returns:
        The number of follow-up messages sent.
    """
    from rag.retriever import get_supabase_client

    s = get_settings()
    sent_count = 0

    try:
        client = get_supabase_client()
        now = datetime.now(timezone.utc)

        # Fetch all active follow-up records that haven't maxed out
        result = (
            client.table("ark_followups")
            .select("phone, last_message_time, followup_stage")
            .eq("status", "active")
            .lt("followup_stage", s.MAX_FOLLOWUP_STAGE)
            .execute()
        )

        rows = result.data or []
        if not rows:
            return 0

        # Build delay mapping
        stage_delays = {
            1: s.FOLLOWUP_STAGE1_DELAY,
            2: s.FOLLOWUP_STAGE2_DELAY,
            3: s.FOLLOWUP_STAGE3_DELAY,
        }

        for row in rows:
            phone = row["phone"]
            stage = row["followup_stage"]
            last_msg_str = row["last_message_time"]

            # Parse the last message timestamp
            try:
                last_msg_time = datetime.fromisoformat(
                    last_msg_str.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                logger.warning(
                    "Invalid last_message_time for %s: %s", phone, last_msg_str
                )
                continue

            elapsed_seconds = (now - last_msg_time).total_seconds()

            # Determine if a follow-up is due
            next_stage = stage + 1
            delay = stage_delays.get(next_stage)

            if delay and elapsed_seconds >= delay:
                if not should_skip_followup(phone):
                    await send_followup_message(phone, next_stage)
                    sent_count += 1

        if sent_count:
            logger.info("FOLLOWUP_CHECK | sent=%d follow-ups", sent_count)

    except Exception as e:
        logger.error("Follow-up check failed: %s", e)

    return sent_count


# =====================================================================
# Background Scheduler (APScheduler)
# =====================================================================

_scheduler = None


def start_followup_scheduler() -> None:
    """
    Start the APScheduler background job that periodically calls
    `check_followups()`.

    Safe to call multiple times — only starts once.
    """
    global _scheduler

    if _scheduler is not None:
        logger.debug("Follow-up scheduler already running.")
        return

    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    s = get_settings()
    interval = s.FOLLOWUP_CHECK_INTERVAL

    _scheduler = AsyncIOScheduler()
    _scheduler.add_job(
        check_followups,
        trigger="interval",
        seconds=interval,
        id="followup_checker",
        name="Follow-up Checker",
        replace_existing=True,
    )
    _scheduler.start()

    logger.info(
        "Follow-up scheduler started (interval=%ds)", interval
    )


def stop_followup_scheduler() -> None:
    """Gracefully shut down the follow-up scheduler."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Follow-up scheduler stopped.")
