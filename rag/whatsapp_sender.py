"""
rag/whatsapp_sender.py
----------------------
Shared WhatsApp message sender via AiSensy Project API.

Exposes:
  - send_whatsapp_message(phone, message) — session text send (needs
    an open 24h customer-care window with the recipient).
  - send_admin_alert(phone, message)      — template send via the
    pre-approved UTILITY `admin_alert` template, with retry + safe
    text fallback. Use this for system alerts (admin notifications,
    hot-lead pings, qualified-lead summaries) — it works outside the
    24h window because WhatsApp allows approved templates anytime.

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

# ── Admin alert tunables ─────────────────────────────────────────────
ADMIN_ALERT_MAX_RETRIES = 2       # retries after the first attempt (total = 3 attempts)
ADMIN_ALERT_RETRY_DELAY = 2       # seconds between retries
ADMIN_ALERT_BODY_LIMIT = 900      # WhatsApp template body variable max is ~1024; stay safe
_RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}


async def send_whatsapp_message(phone: str, message: str) -> bool:
    """
    Send a message to a WhatsApp user via AiSensy Project API.

    Returns True on success, False on failure.
    Strips leading '+' from phone to match AiSensy format.
    Trims trailing whitespace / stray characters from the body so no
    accidental concatenation leaks into the outgoing WhatsApp message.
    """
    s = get_settings()
    if not s.AISENSY_API_KEY or not s.AISENSY_PROJECT_ID:
        logger.warning("AiSensy credentials not set — skipping send to %s", phone)
        return False

    if not isinstance(message, str):
        logger.warning(
            "SEND_COERCE | phone=%s | non-str message type=%s — stringifying",
            phone, type(message).__name__,
        )
        message = str(message)

    # Defensive: strip any trailing whitespace/newlines/control chars. An
    # isolated trailing digit or symbol ("…name?0") was reported in a
    # WhatsApp transcript without any code path producing it; rstrip
    # won't catch that single char, but will catch the common "\n" /
    # "​" tail that quietly breaks AiSensy rendering.
    body = message.rstrip()
    if not body:
        logger.warning("SEND_SKIP | phone=%s | empty body after strip", phone)
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
        "text": {"body": body},
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


# =====================================================================
# Admin alerts — AiSensy Campaign API + approved UTILITY template
# =====================================================================

def _normalize_admin_phone(phone: str) -> str | None:
    """
    Return the phone in AiSensy-expected 91XXXXXXXXXX digit form,
    or None if it doesn't look like a valid WhatsApp number.

    Strips a leading '+' and rejects anything that isn't purely digits
    or is shorter than 10 digits (country code + subscriber number).
    """
    if not phone:
        return None
    digits = phone.lstrip("+").strip()
    if not digits.isdigit() or len(digits) < 10:
        return None
    return digits


async def send_admin_alert(phone: str, message: str) -> bool:
    """
    Deliver a WhatsApp admin alert via the AiSensy Campaign API using a
    pre-approved UTILITY template wrapped in a campaign.

    Credentials
    -----------
    * ``AISENSY_CAMPAIGN_API_KEY`` — JWT (``eyJ...``), copied from
      AiSensy → Manage → API Keys. Passed as ``apiKey`` in the body.
    * ``AISENSY_CAMPAIGN_NAME`` — the **campaign name** in AiSensy that
      wraps the approved UTILITY template (e.g. ``admin_alerts``). This
      is NOT the raw template name.

    Payload shape matches the exact curl AiSensy's dashboard generates
    for the campaign — including the empty ``media/buttons/carouselCards/
    location/attributes`` objects and the ``paramsFallbackValue`` map,
    which AiSensy expects even when unused.

    Delivery strategy
    -----------------
    1. POST the campaign template message. Approved UTILITY templates
       are the only payload WhatsApp accepts outside the 24h customer-
       care window, so this is the reliable path for admin alerts.
    2. Retry up to ``ADMIN_ALERT_MAX_RETRIES`` times on transient
       failures (network errors + 408/429/5xx). 4xx responses (bad
       campaign, bad phone, auth) are permanent and fail fast.
    3. If the template send fully fails, attempt a plain-text session
       send as a best-effort fallback. That only lands if the admin
       messaged the bot within the last 24h — outside that window
       WhatsApp rejects it, which we log loudly rather than surface
       as a bot-wide error.

    Args:
        phone:   Admin phone in ``91XXXXXXXXXX`` form; a leading ``+``
                 is stripped. Non-digit / short numbers are rejected.
        message: Plain-text body. Becomes the single ``{{1}}`` template
                 parameter. Truncated to ``ADMIN_ALERT_BODY_LIMIT`` chars
                 to stay inside WhatsApp's body-variable size cap.

    Returns:
        True if the template OR the text fallback delivered successfully.
    """
    s = get_settings()

    if not s.AISENSY_CAMPAIGN_API_KEY:
        logger.error("ADMIN_ALERT_BLOCKED | reason=AISENSY_CAMPAIGN_API_KEY not set")
        return False
    if not s.AISENSY_CAMPAIGN_NAME:
        logger.error("ADMIN_ALERT_BLOCKED | reason=AISENSY_CAMPAIGN_NAME not set")
        return False

    destination = _normalize_admin_phone(phone)
    if destination is None:
        logger.error(
            "ADMIN_ALERT_BLOCKED | reason=invalid phone (expected 91XXXXXXXXXX) | raw=%r",
            phone,
        )
        return False

    body_param = (message or "").strip()
    if not body_param:
        logger.error("ADMIN_ALERT_BLOCKED | reason=empty message body")
        return False
    if len(body_param) > ADMIN_ALERT_BODY_LIMIT:
        body_param = body_param[: ADMIN_ALERT_BODY_LIMIT - 1] + "…"

    url = "https://backend.aisensy.com/campaign/t1/api/v2"
    payload = {
        "apiKey": s.AISENSY_CAMPAIGN_API_KEY,
        "campaignName": s.AISENSY_CAMPAIGN_NAME,
        "destination": destination,
        "userName": "ARK LEARNING ARENA",
        "templateParams": [body_param],
        "source": "ark-ai-bot",
        "media": {},
        "buttons": [],
        "carouselCards": [],
        "location": {},
        "attributes": {},
        "paramsFallbackValue": {"FirstName": "Admin"},
    }

    masked_to = destination[:4] + "****" + destination[-3:]
    logger.info(
        "ADMIN_ALERT_SEND | to=%s | campaign=%s | param_len=%d",
        masked_to, s.AISENSY_CAMPAIGN_NAME, len(body_param),
    )

    last_status: int | None = None
    last_body: str = ""
    total_attempts = ADMIN_ALERT_MAX_RETRIES + 1

    for attempt in range(1, total_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(url, json=payload)
            last_status = resp.status_code
            last_body = resp.text[:500]

            if resp.status_code in (200, 201, 202):
                logger.info(
                    "ADMIN_ALERT_OK | to=%s | attempt=%d | status=%d",
                    masked_to, attempt, resp.status_code,
                )
                return True

            if resp.status_code not in _RETRYABLE_STATUSES:
                logger.error(
                    "ADMIN_ALERT_FAIL_PERMANENT | to=%s | attempt=%d | status=%d | body=%s",
                    masked_to, attempt, resp.status_code, last_body,
                )
                break

            logger.warning(
                "ADMIN_ALERT_RETRY | to=%s | attempt=%d/%d | status=%d | body=%s",
                masked_to, attempt, total_attempts, resp.status_code, last_body,
            )
        except httpx.HTTPError as e:
            logger.warning(
                "ADMIN_ALERT_RETRY | to=%s | attempt=%d/%d | error=%s: %s",
                masked_to, attempt, total_attempts, type(e).__name__, e,
            )
        except Exception as e:
            logger.exception(
                "ADMIN_ALERT_ERROR | to=%s | attempt=%d | %s", masked_to, attempt, e
            )
            break

        if attempt < total_attempts:
            await asyncio.sleep(ADMIN_ALERT_RETRY_DELAY)

    # ── Safe fallback: session text send ──────────────────────────
    # Only delivers if the admin is inside the 24h customer-care window;
    # WhatsApp rejects it otherwise. Worth attempting so that an active-
    # session admin still gets the alert if the campaign config broke.
    logger.warning(
        "ADMIN_ALERT_FALLBACK_TEXT | to=%s | campaign_last_status=%s | campaign_body=%s",
        masked_to, last_status, last_body,
    )
    fallback_ok = await send_whatsapp_message(destination, body_param)
    if fallback_ok:
        logger.info(
            "ADMIN_ALERT_FALLBACK_OK | to=%s | delivered via session text",
            masked_to,
        )
        return True

    logger.error(
        "ADMIN_ALERT_DROPPED | to=%s | campaign + text fallback both failed "
        "(if admin is outside 24h window, only approved templates will land — "
        "verify AISENSY_CAMPAIGN_NAME points to a live campaign wrapping an "
        "APPROVED UTILITY template with exactly one body variable)",
        masked_to,
    )
    return False
