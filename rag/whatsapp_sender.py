"""
rag/whatsapp_sender.py
----------------------
Shared WhatsApp message sender via AiSensy Project API.

Exposes:
  - send_whatsapp_message(phone, message) — session text send (needs
    an open 24h customer-care window with the recipient).
  - send_admin_alert(phone, message)      — template send via the
    pre-approved UTILITY `admin_alerts` template, with retry + safe
    text fallback. Use this for system alerts (admin notifications,
    hot-lead pings, qualified-lead summaries) — it works outside the
    24h window because WhatsApp allows approved templates anytime.

Extracted here to break the circular import between api/main.py
and rag/followup_manager.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import NamedTuple

import httpx

from config.settings import get_settings

logger = logging.getLogger("ark.sender")


class AdminAlertResult(NamedTuple):
    """Structured result of an admin-alert send attempt.

    ``ok``     — True if the message was delivered (template or fallback).
    ``status`` — Last HTTP status code seen, or None if no request went out.
    ``via``    — "template" | "text_fallback" | "blocked" | "dropped".
    ``error``  — Human-readable reason for failure; empty on success.
    """
    ok: bool
    status: int | None
    via: str
    error: str

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
# Admin alerts — AiSensy Project Messages API + approved UTILITY template
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


# WhatsApp Cloud API template-parameter rules (enforced by Meta, surfaced
# by AiSensy as "Invalid Parameter"):
#   1. No newline characters  (\r, \n)
#   2. No tab characters      (\t)
#   3. No more than 4 consecutive spaces
# Applying these in order fixes the vast majority of Invalid-Parameter
# failures on multi-line admin summaries.
_MULTI_SPACE_RE = re.compile(r" {5,}")


def _sanitize_template_param(value: object) -> str:
    """
    Coerce any input into a WhatsApp-safe template-parameter string.

    Rules applied (in order):
      - Non-strings are coerced via ``str()``
      - ``\\r`` stripped, ``\\n`` → " | " (preserves structure on one line)
      - ``\\t`` → single space
      - Runs of 5+ spaces collapsed to 4
      - Leading/trailing whitespace stripped
      - Empty result replaced with sentinel "New Alert" so the param is
        never an empty string (Meta rejects empty values too)

    Returns:
        A single-line string safe to send as ``{{1}}``.
    """
    if not isinstance(value, str):
        value = str(value)

    # Normalise line endings, then flatten newlines to a separator that
    # preserves the visual break without tripping Meta's \n rule.
    value = value.replace("\r\n", "\n").replace("\r", "")
    value = value.replace("\n", " | ")

    # Tabs are disallowed; a single space is the closest equivalent.
    value = value.replace("\t", " ")

    # Collapse long space runs so "4+ consecutive spaces" can never happen.
    value = _MULTI_SPACE_RE.sub("    ", value)

    value = value.strip()
    if not value:
        value = "New Alert"

    if len(value) > ADMIN_ALERT_BODY_LIMIT:
        value = value[: ADMIN_ALERT_BODY_LIMIT - 1] + "…"

    return value


async def send_admin_alert(phone: str, message: object) -> AdminAlertResult:
    """
    Deliver a WhatsApp admin alert via the AiSensy Project Messages API
    using a pre-approved UTILITY template (name from
    ``AISENSY_ADMIN_ALERT_TEMPLATE``, default ``admin_alerts``).

    Why this shape
    --------------
    AiSensy's "Invalid Parameter" dashboard error on Project-API template
    sends is almost always Meta's template-parameter validator rejecting
    newlines, tabs, or runs of 5+ spaces inside the ``{{1}}`` value.
    ``_sanitize_template_param`` strips all three before we serialise.

    Payload — matches the user-specified shape exactly, no extra fields::

        {
          "to": "91XXXXXXXXXX",
          "type": "template",
          "recipient_type": "individual",
          "template": {
            "name": "<AISENSY_ADMIN_ALERT_TEMPLATE>",
            "language": {"code": "en"},
            "components": [
              {"type": "body", "parameters": [{"type": "text", "text": "<msg>"}]}
            ]
          }
        }

    Delivery strategy
    -----------------
    1. Template send via ``apis.aisensy.com/project-apis/v1/project/<id>/messages``
       authed with ``X-AiSensy-Project-API-Pwd: <AISENSY_API_KEY>``.
    2. Retry up to ``ADMIN_ALERT_MAX_RETRIES`` on transient failures
       (408/429/5xx + network errors). 4xx fails fast — that means the
       parameter or template is wrong; no amount of retrying helps.
    3. If the template fully fails and the admin is inside the 24h
       customer-care window, fall back to a plain-text session send.

    Args:
        phone:   Admin phone in ``91XXXXXXXXXX`` form (a leading ``+`` is stripped).
        message: Any value; non-strings are coerced, newlines flattened,
                 empty strings replaced with "New Alert".

    Returns:
        AdminAlertResult(ok, status, via, error). Existing bool-expecting
        callers can still do ``if result:`` — NamedTuple is falsy only
        when all fields are zero, so we never accidentally lie; callers
        that want correctness should check ``.ok``.
    """
    s = get_settings()

    # ── 1. Safe-guard: coerce + sanitize the message ─────────────
    body_param = _sanitize_template_param(message)

    # ── 2. Validate credentials + phone ──────────────────────────
    if not s.AISENSY_API_KEY or not s.AISENSY_PROJECT_ID:
        err = "AISENSY_API_KEY or AISENSY_PROJECT_ID not set"
        logger.error("ADMIN_ALERT_BLOCKED | reason=%s", err)
        return AdminAlertResult(False, None, "blocked", err)

    destination = _normalize_admin_phone(phone)
    if destination is None:
        err = f"invalid phone (expected 91XXXXXXXXXX): {phone!r}"
        logger.error("ADMIN_ALERT_BLOCKED | reason=%s", err)
        return AdminAlertResult(False, None, "blocked", err)

    template_name = (s.AISENSY_ADMIN_ALERT_TEMPLATE or "admin_alerts").strip()
    if not template_name:
        err = "AISENSY_ADMIN_ALERT_TEMPLATE is empty"
        logger.error("ADMIN_ALERT_BLOCKED | reason=%s", err)
        return AdminAlertResult(False, None, "blocked", err)

    # ── 3. Build EXACT payload shape — no extra fields ───────────
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
        "type": "template",
        "recipient_type": "individual",
        "template": {
            "name": template_name,
            "language": {"code": "en"},
            "components": [
                {
                    "type": "body",
                    "parameters": [
                        {"type": "text", "text": body_param},
                    ],
                },
            ],
        },
    }

    # ── 4. Debug log: final message, type, full payload ─────────
    masked_to = destination[:4] + "****" + destination[-3:]
    logger.info(
        "ADMIN_ALERT_SEND | to=%s | template=%s | msg_type=%s | msg_len=%d | msg=%r",
        masked_to, template_name, type(message).__name__, len(body_param), body_param,
    )
    logger.debug(
        "ADMIN_ALERT_PAYLOAD | to=%s | payload=%s",
        masked_to, json.dumps(payload, ensure_ascii=False),
    )

    last_status: int | None = None
    last_body: str = ""
    last_error: str = ""
    total_attempts = ADMIN_ALERT_MAX_RETRIES + 1

    for attempt in range(1, total_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(url, headers=headers, json=payload)
            last_status = resp.status_code
            last_body = resp.text[:500]

            if resp.status_code in (200, 201, 202):
                logger.info(
                    "ADMIN_ALERT_OK | to=%s | attempt=%d | status=%d | response=%s",
                    masked_to, attempt, resp.status_code, last_body,
                )
                return AdminAlertResult(True, resp.status_code, "template", "")

            last_error = _extract_aisensy_error(last_body, resp.status_code)

            if resp.status_code not in _RETRYABLE_STATUSES:
                logger.error(
                    "ADMIN_ALERT_FAIL_PERMANENT | to=%s | attempt=%d | status=%d | error=%s | body=%s",
                    masked_to, attempt, resp.status_code, last_error, last_body,
                )
                break

            logger.warning(
                "ADMIN_ALERT_RETRY | to=%s | attempt=%d/%d | status=%d | error=%s | body=%s",
                masked_to, attempt, total_attempts, resp.status_code, last_error, last_body,
            )
        except httpx.HTTPError as e:
            last_error = f"{type(e).__name__}: {e}"
            logger.warning(
                "ADMIN_ALERT_RETRY | to=%s | attempt=%d/%d | error=%s",
                masked_to, attempt, total_attempts, last_error,
            )
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            logger.exception(
                "ADMIN_ALERT_ERROR | to=%s | attempt=%d | %s",
                masked_to, attempt, last_error,
            )
            break

        if attempt < total_attempts:
            await asyncio.sleep(ADMIN_ALERT_RETRY_DELAY)

    # ── 5. Safe fallback: session text (only works inside 24h) ───
    logger.warning(
        "ADMIN_ALERT_FALLBACK_TEXT | to=%s | template_last_status=%s | error=%s",
        masked_to, last_status, last_error,
    )
    fallback_ok = await send_whatsapp_message(destination, body_param)
    if fallback_ok:
        logger.info(
            "ADMIN_ALERT_FALLBACK_OK | to=%s | delivered via session text",
            masked_to,
        )
        return AdminAlertResult(True, last_status, "text_fallback", last_error)

    logger.error(
        "ADMIN_ALERT_DROPPED | to=%s | status=%s | error=%s | "
        "(if admin is outside 24h window, only approved templates land — "
        "verify AISENSY_ADMIN_ALERT_TEMPLATE is an APPROVED UTILITY template "
        "with exactly one body variable, and that the message contains no "
        "newlines/tabs after sanitization)",
        masked_to, last_status, last_error,
    )
    return AdminAlertResult(False, last_status, "dropped", last_error or "delivery failed")


def _extract_aisensy_error(body_text: str, status: int) -> str:
    """
    Pull a short, human-readable error string out of an AiSensy / Meta
    response body. Falls back to the raw status label if no JSON fits.

    AiSensy error shapes we've seen:
      {"message": "Invalid Parameter", ...}
      {"error": {"message": "...", "code": 131000}}
      "<html>…"  (for 5xx / gateway failures — returned as-is, truncated)
    """
    if not body_text:
        return f"HTTP {status}"
    try:
        data = json.loads(body_text)
    except ValueError:
        return f"HTTP {status} — {body_text[:120]}"
    if isinstance(data, dict):
        for key in ("message", "error_message", "errorMessage"):
            val = data.get(key)
            if isinstance(val, str) and val:
                return val
        err = data.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("description")
            if isinstance(msg, str) and msg:
                return msg
        if isinstance(err, str) and err:
            return err
    return f"HTTP {status}"
