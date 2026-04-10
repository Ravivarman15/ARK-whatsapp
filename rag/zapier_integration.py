"""
rag/zapier_integration.py
─────────────────────────
Zapier webhook integration for sending captured leads to Google Sheets.

This is a fire-and-forget integration layer — errors are logged but
NEVER propagated, so the main bot flow is never affected.

Usage:
    # Async (api/main.py)
    await send_lead_to_zapier(build_lead_payload(lead, message))

    # Sync  (scripts/test_bot.py)
    send_lead_to_zapier_sync(build_lead_payload(lead, message))
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING

import httpx

from config.settings import get_settings

if TYPE_CHECKING:
    from rag.lead_manager import LeadData

logger = logging.getLogger("ark.zapier")

# =====================================================================
# Payload Builder
# =====================================================================

def build_lead_payload(
    lead: "LeadData",
    message: str = "",
    *,
    lead_type_override: str = "",
    priority_override: str = "",
) -> dict:
    """
    Build a standardised JSON payload from a LeadData object.

    Args:
        lead:               The LeadData dataclass instance.
        message:            The original user message (optional).
        lead_type_override: Override the lead_type field (e.g. "Hot Lead").
        priority_override:  Override the priority field (e.g. "HIGH").

    Returns:
        A flat dict matching the Google Sheets column layout.
    """
    return {
        "phone": lead.phone or "",
        "student_name": lead.student_name or "",
        "class": lead.student_class or "",
        "school": lead.school or "",
        "parent_phone": lead.parent_phone or "",
        "course": lead.course or "",
        "lead_type": lead_type_override or lead.lead_type or "",
        "lead_score": str(lead.lead_score),
        "stage": lead.stage or "",
        "segment": lead.segment or "",
        "priority": priority_override or lead.priority or "",
        "message": message,
        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
    }


def build_quick_payload(
    phone: str,
    message: str,
    lead_type: str,
    priority: str = "HIGH",
    user_name: str = "",
    score: int = 0,
    stage: str = "",
    segment: str = "",
) -> dict:
    """
    Build a lightweight payload when a full LeadData is not available
    (e.g. hot lead detection before qualification starts).
    """
    return {
        "phone": phone,
        "student_name": user_name,
        "class": "",
        "school": "",
        "parent_phone": "",
        "course": "",
        "lead_type": lead_type,
        "lead_score": str(score),
        "stage": stage,
        "segment": segment,
        "priority": priority,
        "message": message,
        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
    }


# =====================================================================
# Async Sender (for api/main.py)
# =====================================================================

MAX_RETRIES = 1
RETRY_DELAY = 2  # seconds


async def send_lead_to_zapier(data: dict) -> bool:
    """
    POST lead data to the Zapier Catch Hook webhook.

    - Reads ZAPIER_WEBHOOK_URL from settings.
    - Retries once on failure.
    - NEVER raises — all errors are caught and logged.

    Args:
        data: The JSON payload dict.

    Returns:
        True if the webhook accepted the data, False otherwise.
    """
    s = get_settings()
    url = s.ZAPIER_WEBHOOK_URL

    if not url:
        logger.debug("ZAPIER_SKIP | ZAPIER_WEBHOOK_URL not configured")
        return False

    for attempt in range(1 + MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(url, json=data)
                if resp.status_code in (200, 201):
                    logger.info(
                        "ZAPIER_SENT | phone=%s | type=%s | status=%d",
                        data.get("phone", "?"), data.get("lead_type", "?"),
                        resp.status_code,
                    )
                    return True
                else:
                    logger.warning(
                        "ZAPIER_ERROR | attempt=%d | status=%d | body=%s",
                        attempt + 1, resp.status_code, resp.text[:200],
                    )
        except Exception as e:
            logger.warning(
                "ZAPIER_ERROR | attempt=%d | error=%s", attempt + 1, e,
            )

        if attempt < MAX_RETRIES:
            await asyncio.sleep(RETRY_DELAY)

    logger.error("ZAPIER_FAILED | phone=%s | exhausted retries", data.get("phone", "?"))
    return False


# =====================================================================
# Sync Wrapper (for scripts/test_bot.py)
# =====================================================================

def send_lead_to_zapier_sync(data: dict) -> bool:
    """
    Synchronous wrapper around send_lead_to_zapier().

    Safe to call from non-async code (e.g. test_bot.py).
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already inside an async loop — schedule as a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, send_lead_to_zapier(data)).result()
        else:
            return loop.run_until_complete(send_lead_to_zapier(data))
    except RuntimeError:
        return asyncio.run(send_lead_to_zapier(data))
