"""
rag/followup_manager.py
───────────────────────
Automated follow-up system for the ARK Learning Arena WhatsApp bot.

Tracks user activity timestamps in Supabase and sends staged follow-up
messages to users who stop responding during a conversation — all within
the 24-hour WhatsApp session window.

Stages (measured from last USER message):
    1 → 30 min inactivity
    2 → 4 hr  inactivity
    3 → 16 hr inactivity

Rules:
    - All sends happen WITHIN the 24-hr WhatsApp session window (<86400s)
    - Users outside the 24-hr window are marked 'expired' — no send
    - Stale stages (server restart catch-up) are skipped, not spammed:
      only the single most-appropriate stage is sent per check cycle
    - DB stage is updated BEFORE the message is sent to prevent duplicates
    - Follow-ups are suppressed when user is in active lead qualification,
      or when status is 'escalated' / 'completed' / 'expired'
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from config.settings import get_settings
from rag.lead_manager import is_in_qualification

logger = logging.getLogger("ark.followup")

# Maximum seconds past a stage's ideal send time before we skip it
# (prevents blasting 2-3 messages at once after server restart)
_MAX_STAGE_CATCHUP = 3600  # 1 hour — if missed by >1hr, skip to next

# WhatsApp 24-hour session window
_WHATSAPP_WINDOW = 86400  # seconds


# =====================================================================
# Follow-Up Message Templates
# =====================================================================

FOLLOWUP_MESSAGES = {
    1: (
        "Hi! Just checking in \U0001f44b\n"
        "Would you like to know more about our programs?\n"
        "Feel free to ask any question."
    ),
    2: (
        "We have limited slots this week \U0001f3af\n"
        "Would you like to reserve one?\n"
        "Reply 'Yes' to get started."
    ),
    3: (
        "Let us know if you need any help \U0001f64f\n"
        "We can schedule a quick counselling call for you."
    ),
}


# =====================================================================
# Record User Activity  (called on every incoming message)
# =====================================================================

async def record_user_activity(phone: str) -> None:
    """
    Record that a user just sent a message — resets their follow-up state.

    Upserts the `ark_followups` row:
      - last_message_time → now (UTC)
      - followup_stage    → 0  (restart the follow-up sequence)
      - status            → 'active'
    """
    if not phone:
        return
    try:
        from rag.retriever import get_supabase_client
        client = get_supabase_client()
        now = datetime.now(timezone.utc).isoformat()
        client.table("ark_followups").upsert(
            {
                "phone": phone,
                "last_message_time": now,
                "followup_stage": 0,
                "status": "active",
            },
            on_conflict="phone",
        ).execute()
        logger.debug("FOLLOWUP_ACTIVITY | phone=%s | reset to stage 0", phone)
    except Exception as e:
        logger.error("record_user_activity failed for %s: %s", phone, e)


# =====================================================================
# Mark helpers
# =====================================================================

async def mark_followup_escalated(phone: str) -> None:
    """Suppress all future follow-ups for this user (human escalation)."""
    if not phone:
        return
    try:
        from rag.retriever import get_supabase_client
        get_supabase_client().table("ark_followups").update(
            {"status": "escalated"}
        ).eq("phone", phone).execute()
        logger.info("FOLLOWUP_ESCALATED | phone=%s", phone)
    except Exception as e:
        logger.error("mark_followup_escalated failed for %s: %s", phone, e)


async def mark_followup_completed(phone: str) -> None:
    """Suppress all future follow-ups for this user (lead qualified)."""
    if not phone:
        return
    try:
        from rag.retriever import get_supabase_client
        get_supabase_client().table("ark_followups").update(
            {"status": "completed"}
        ).eq("phone", phone).execute()
        logger.info("FOLLOWUP_COMPLETED | phone=%s", phone)
    except Exception as e:
        logger.error("mark_followup_completed failed for %s: %s", phone, e)


# =====================================================================
# Core: Check and Send Follow-Ups
# =====================================================================

async def check_followups() -> int:
    """
    Query Supabase for users due for a follow-up and send messages.

    Called every FOLLOWUP_CHECK_INTERVAL seconds by the scheduler.

    For each active user:
      1. Skip if outside the 24-hr WhatsApp window (mark 'expired')
      2. Skip if user is in active lead qualification
      3. Find the highest stage that is overdue (avoids catch-up spam)
      4. Update DB stage FIRST, then send message

    Returns the number of messages sent.
    """
    from rag.retriever import get_supabase_client
    from rag.whatsapp_sender import send_whatsapp_message

    s = get_settings()
    sent_count = 0

    # Stage → seconds from last_message_time
    stage_delays = {
        1: s.FOLLOWUP_STAGE1_DELAY,
        2: s.FOLLOWUP_STAGE2_DELAY,
        3: s.FOLLOWUP_STAGE3_DELAY,
    }

    try:
        client = get_supabase_client()
        now = datetime.now(timezone.utc)

        result = (
            client.table("ark_followups")
            .select("phone, last_message_time, followup_stage")
            .eq("status", "active")
            .lt("followup_stage", s.MAX_FOLLOWUP_STAGE)
            .execute()
        )
        rows = result.data or []

        if not rows:
            logger.debug("FOLLOWUP_CHECK | no active users to check")
            return 0

        logger.debug("FOLLOWUP_CHECK | checking %d active user(s)", len(rows))

        for row in rows:
            phone: str = row["phone"]
            current_stage: int = row["followup_stage"]
            last_msg_str: str = row.get("last_message_time", "")

            # ── Parse timestamp ──────────────────────────────────
            try:
                last_msg_time = datetime.fromisoformat(
                    last_msg_str.replace("Z", "+00:00")
                )
                # Ensure timezone-aware (Supabase sometimes omits tz)
                if last_msg_time.tzinfo is None:
                    last_msg_time = last_msg_time.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError, AttributeError):
                logger.warning("FOLLOWUP_CHECK | bad timestamp for %s: %s", phone, last_msg_str)
                continue

            elapsed: float = (now - last_msg_time).total_seconds()

            # ── 24-hour window check ─────────────────────────────
            if elapsed >= _WHATSAPP_WINDOW:
                logger.info(
                    "FOLLOWUP_EXPIRED | phone=%s | elapsed=%.1fhr — marking expired",
                    phone, elapsed / 3600,
                )
                try:
                    client.table("ark_followups").update(
                        {"status": "expired"}
                    ).eq("phone", phone).execute()
                except Exception as e:
                    logger.error("Failed to mark expired for %s: %s", phone, e)
                continue

            # ── Skip if user is in active qualification ──────────
            if is_in_qualification(phone):
                logger.debug("FOLLOWUP_SKIP | phone=%s | in qualification", phone)
                continue

            # ── Determine which stage to send ────────────────────
            # Find the HIGHEST due stage (avoids sending stale stages
            # from before a server restart gap)
            target_stage: Optional[int] = None

            for stage in range(current_stage + 1, s.MAX_FOLLOWUP_STAGE + 1):
                delay = stage_delays.get(stage, 0)
                if elapsed < delay:
                    break  # Not yet due — higher stages also not due
                # Due. Check if it's so overdue we should skip it.
                next_delay = stage_delays.get(stage + 1)
                if next_delay and elapsed >= next_delay:
                    # This stage's send window has passed (next stage is
                    # already due too) — skip this stage to avoid spam
                    logger.debug(
                        "FOLLOWUP_SKIP_STALE | phone=%s | stage=%d | elapsed=%.1fhr",
                        phone, stage, elapsed / 3600,
                    )
                    continue
                target_stage = stage

            if target_stage is None:
                logger.debug(
                    "FOLLOWUP_CHECK | phone=%s | no stage due (elapsed=%.1fhr, stage=%d)",
                    phone, elapsed / 3600, current_stage,
                )
                continue

            # ── Update DB FIRST (prevents double-send on overlap) ─
            try:
                update_data: dict = {"followup_stage": target_stage}
                if target_stage >= s.MAX_FOLLOWUP_STAGE:
                    update_data["status"] = "completed"

                client.table("ark_followups").update(update_data).eq(
                    "phone", phone
                ).execute()
            except Exception as e:
                logger.error(
                    "FOLLOWUP_DB_UPDATE_FAILED | phone=%s | stage=%d | %s",
                    phone, target_stage, e,
                )
                continue  # Skip send if DB update failed

            # ── Send message ─────────────────────────────────────
            message = FOLLOWUP_MESSAGES.get(target_stage)
            if not message:
                logger.warning("No template for stage %d", target_stage)
                continue

            ok = await send_whatsapp_message(phone, message)

            if ok:
                logger.info(
                    "FOLLOWUP_SENT | phone=%s | stage=%d | elapsed=%.1fhr",
                    phone, target_stage, elapsed / 3600,
                )
                sent_count += 1
            else:
                # Revert DB stage so we retry next cycle
                logger.error(
                    "FOLLOWUP_SEND_FAILED | phone=%s | stage=%d — reverting DB",
                    phone, target_stage,
                )
                try:
                    revert: dict = {"followup_stage": current_stage}
                    if update_data.get("status") == "completed":
                        revert["status"] = "active"
                    client.table("ark_followups").update(revert).eq(
                        "phone", phone
                    ).execute()
                except Exception as e:
                    logger.error("FOLLOWUP_REVERT_FAILED | phone=%s | %s", phone, e)

    except Exception as e:
        logger.error("check_followups failed: %s", e)

    if sent_count:
        logger.info("FOLLOWUP_CHECK | sent=%d message(s) this cycle", sent_count)

    return sent_count


# =====================================================================
# Background Scheduler (APScheduler)
# =====================================================================

_scheduler = None


def start_followup_scheduler() -> None:
    """Start the APScheduler background job. Safe to call multiple times."""
    global _scheduler

    if _scheduler is not None:
        logger.debug("Follow-up scheduler already running.")
        return

    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    s = get_settings()
    _scheduler = AsyncIOScheduler()
    _scheduler.add_job(
        check_followups,
        trigger="interval",
        seconds=s.FOLLOWUP_CHECK_INTERVAL,
        id="followup_checker",
        name="Follow-up Checker",
        replace_existing=True,
        max_instances=1,        # never run two overlapping instances
        misfire_grace_time=30,  # if missed by <30s, still run; else skip
    )
    _scheduler.start()
    logger.info("Follow-up scheduler started (interval=%ds)", s.FOLLOWUP_CHECK_INTERVAL)


def stop_followup_scheduler() -> None:
    """Gracefully shut down the follow-up scheduler."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Follow-up scheduler stopped.")
