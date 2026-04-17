"""
rag/whatsapp_sender.py
----------------------
Shared WhatsApp message sender via AiSensy Project API.

Extracted here to break the circular import between api/main.py
and rag/followup_manager.py.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from config.settings import get_settings

logger = logging.getLogger("ark.sender")

MAX_SEND_RETRIES = 1
FALLBACK_MESSAGE = "Sorry, I'm having trouble right now. Please try again in a moment."


async def send_whatsapp_message(phone: str, message: str) -> bool:
    """
    Send a message to a WhatsApp user via AiSensy Project API.

    Returns True on success, False on failure.
    Strips leading '+' from phone to match AiSensy format.
    """
    s = get_settings()
    if not s.AISENSY_API_KEY or not s.AISENSY_PROJECT_ID:
        logger.warning("AiSensy credentials not set — skipping send to %s", phone)
        return False

    destination = phone.lstrip("+")
    url = (
        f"https://apis.aisensy.com/project-apis/v1/"
        f"project/{s.AISENSY_PROJECT_ID}/messages"
    )
    headers = {
        "X-AiSensy-Project-API-Pwd": s.AISENSY_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "to": destination,
        "type": "text",
        "recipient_type": "individual",
        "text": {"body": message},
    }

    for attempt in range(1 + MAX_SEND_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code in (200, 201):
                    logger.info("SEND_OK | phone=%s", phone)
                    return True
                logger.error(
                    "SEND_FAIL | phone=%s | status=%d | body=%s",
                    phone, resp.status_code, resp.text[:200],
                )
        except Exception as e:
            logger.error("SEND_ERROR | phone=%s | attempt=%d | %s", phone, attempt + 1, e)

        if attempt < MAX_SEND_RETRIES:
            await asyncio.sleep(2)

    return False
