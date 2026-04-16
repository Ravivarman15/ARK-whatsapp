"""
api/main.py
-----------
Production FastAPI server for the ARK Learning Arena AI assistant.

Endpoints:
    GET  /health    - Liveness / readiness check
    POST /ask       - Direct question-answer endpoint
    POST /whatsapp  - AiSensy WhatsApp webhook (with full lead pipeline)

Features:
    - Async endpoints for low-latency responses
    - Smart intent routing (complaint > factual > admission > qualification > general)
    - Lead qualification conversation flow (state machine)
    - WhatsApp-friendly short responses (2-3 lines max)
    - Per-user state memory via phone number
    - Lead scoring + segmentation + stage detection
    - Complaint / confusion escalation
    - Admin notification for qualified leads
    - Structured logging
    - CORS middleware
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import get_settings
from rag.embeddings import close_client as close_embedding_client
from rag.retriever import ask_async
from rag.escalation import (
    detect_human_request, detect_complaint, notify_admin,
    record_confusion, reset_confusion, should_escalate_confusion,
    ESCALATION_REPLY, COMPLAINT_REPLY,
)
from rag.lead_manager import (
    classify_lead,
    detect_course_interest,
    is_in_qualification,
    start_lead_qualification,
    store_lead_data,
    process_qualification_message,
    get_current_qual_prompt,
    complete_lead,
    get_lead_data,
    save_lead_to_db,
    notify_admin_qualified_lead,
    notify_admin_hot_lead,
    generate_counsellor_summary,
    update_lead_intelligence,
    add_internal_flag,
    set_concern,
)
from rag.followup_manager import (
    record_user_activity,
    mark_followup_escalated,
    mark_followup_completed,
    start_followup_scheduler,
    stop_followup_scheduler,
)
from rag.zapier_integration import (
    send_lead_to_zapier, build_lead_payload, build_quick_payload,
)
from rag.scoring import score_from_message, get_score, get_lead_type, update_score, ScoreAction
from rag.stage_detector import detect_and_update_stage, get_stage
from rag.segmentation import detect_segment_from_message
from rag.persona_detector import detect_and_update_persona, get_persona
from rag.intent_router import classify_message, Route
from rag.response_formatter import format_whatsapp_response

# =====================================================================
# Logging
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-16s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ark.api")


# =====================================================================
# Lifespan (startup / shutdown)
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background jobs at startup; clean up at shutdown."""
    logger.info("Starting ARK AI Bot ...")
    logger.info("Using HuggingFace Inference API for embeddings (no local model).")
    start_followup_scheduler()
    logger.info("Server is live.")
    yield
    stop_followup_scheduler()
    await close_embedding_client()
    logger.info("Shutting down ARK AI Bot.")


# =====================================================================
# App
# =====================================================================

app = FastAPI(
    title="ARK AI Bot",
    description=(
        "Production RAG-powered AI assistant for ARK Learning Arena. "
        "Answers questions, captures leads, qualifies students, "
        "scores leads, detects decision stages, handles multi-intent, "
        "and notifies admin automatically."
    ),
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# Request / Response Models
# =====================================================================

class AskRequest(BaseModel):
    """Incoming question from the user."""
    question: str = Field(
        ..., min_length=1, max_length=1000,
        description="The user's natural-language question.",
    )
    user_id: Optional[str] = Field(
        default="anonymous",
        description="Optional user identifier for conversation memory.",
    )


class AskResponse(BaseModel):
    """AI-generated answer."""
    answer: str = Field(
        ..., description="The AI-generated answer grounded in the document.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error detail if the request failed.",
    )


class WhatsAppPayload(BaseModel):
    """Incoming AiSensy WhatsApp webhook payload.

    The outer structure wraps a flexible `data` dict that contains
    the nested `message` object from AiSensy.  Using `dict` for `data`
    keeps compatibility with all AiSensy payload variants while giving
    Swagger a proper request-body schema.

    Example::

        {
          "data": {
            "message": {
              "phone_number": "919999999999",
              "userName": "Test",
              "message_type": "TEXT",
              "message_content": { "text": "hello" }
            }
          }
        }
    """
    data: dict[str, Any] = Field(
        ...,
        description="AiSensy webhook data containing the message object.",
        json_schema_extra={
            "example": {
                "message": {
                    "phone_number": "919999999999",
                    "userName": "Test",
                    "message_type": "TEXT",
                    "message_content": {"text": "hello"},
                }
            }
        },
    )


# =====================================================================
# AiSensy Reply Helper
# =====================================================================

MAX_SEND_RETRIES = 1  # Retry once on failure
FALLBACK_MESSAGE = "Sorry, please try again later."


async def send_whatsapp_message(phone: str, message: str) -> None:
    """
    Send a reply message back to WhatsApp via the AiSensy Project API.

    Uses the session-message endpoint (free-text within 24-hour window).
    Includes retry logic for transient failures.
    """
    s = get_settings()
    if not s.AISENSY_API_KEY:
        logger.warning("AISENSY_API_KEY not set - skipping WhatsApp reply.")
        return

    if not s.AISENSY_PROJECT_ID:
        logger.warning("AISENSY_PROJECT_ID not set - skipping WhatsApp reply.")
        return

    # AiSensy expects bare numbers without + prefix (e.g. 917305801869)
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
                    logger.info("SEND_MESSAGE | phone=%s | status=sent", phone)
                    return
                else:
                    logger.error(
                        "SEND_MESSAGE | AiSensy API error (%d): %s",
                        resp.status_code, resp.text,
                    )
        except Exception as e:
            logger.error("SEND_MESSAGE | attempt=%d | error=%s", attempt + 1, e)

        if attempt < MAX_SEND_RETRIES:
            logger.info("SEND_MESSAGE | retrying in 2s ...")
            await asyncio.sleep(2)


# =====================================================================
# Payload Extraction Helper
# =====================================================================

def _extract_from_payload(payload: dict) -> tuple[str, str, str, str]:
    """
    Extract (message, phone, user_name, message_type) from an AiSensy
    webhook payload.

    Handles three payload shapes:
      1. Full AiSensy webhook:  body.data.message.{phone_number, userName, ...}
      2. Flat legacy:           { "Message": "...", "From": "...", ... }
      3. Partial nested:        { "message_content": { "text": "..." }, ... }

    Returns:
        (user_message, phone_number, user_name, message_type)
    """
    message = ""
    phone = ""
    user_name = "Unknown"
    message_type = "TEXT"

    if not isinstance(payload, dict):
        return message, phone, user_name, message_type

    # ── Shape 1: Full AiSensy webhook (data.message.*) ───────────
    data = payload.get("data") or payload.get("body", {}).get("data")
    if isinstance(data, dict):
        msg_obj = data.get("message", {})
        if isinstance(msg_obj, dict):
            phone = msg_obj.get("phone_number", "") or ""
            user_name = msg_obj.get("userName", "") or "Unknown"
            message_type = msg_obj.get("message_type", "TEXT") or "TEXT"

            msg_content = msg_obj.get("message_content", {})
            if isinstance(msg_content, dict):
                message = msg_content.get("text", "") or msg_content.get("body", "")
            elif isinstance(msg_content, str):
                message = msg_content

            # If we got data from the nested structure, return early
            if message or phone:
                phone = phone.lstrip("+") if phone else ""
                return message.strip(), phone, user_name, message_type.upper()

    # ── Shape 2: Flat webhook fields ─────────────────────────────
    message = payload.get("Message", "") or payload.get("message", "")
    phone = (
        payload.get("From", "")
        or payload.get("from", "")
        or payload.get("phone", "")
        or payload.get("phone_number", "")
    )
    user_name = payload.get("userName", "") or payload.get("user_name", "") or "Unknown"
    message_type = payload.get("message_type", "TEXT") or "TEXT"

    # ── Shape 3: Partial nested (message_content at top level) ───
    if not message:
        msg_content = payload.get("message_content", {})
        if isinstance(msg_content, dict):
            message = msg_content.get("text", "") or msg_content.get("body", "")
        elif isinstance(msg_content, str):
            message = msg_content

    sender = payload.get("sender", {})
    if isinstance(sender, dict):
        if not phone:
            phone = sender.get("phone_number", "") or sender.get("phone", "")
        if user_name == "Unknown":
            user_name = (
                sender.get("name", "")
                or sender.get("first_name", "")
                or "Unknown"
            )

    # Strip any leading '+' from phone for consistent internal usage
    phone = phone.lstrip("+") if phone else ""
    message = message.strip() if message else ""

    return message, phone, user_name, message_type.upper()


# =====================================================================
# Endpoints
# =====================================================================

@app.get("/health")
async def health():
    """Liveness / readiness check."""
    return {"status": "ok", "version": "4.0.0"}


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(body: AskRequest):
    """Accept a user question, run the RAG pipeline, and return an answer."""
    logger.info("ASK_REQUEST | user=%s | question=\"%s\"", body.user_id, body.question[:120])
    try:
        answer = await ask_async(
            question=body.question,
            user_id=body.user_id or "anonymous",
        )
        logger.info("ASK_SUCCESS | answer_len=%d", len(answer))
        return AskResponse(answer=answer)
    except Exception as e:
        logger.exception("ASK_ERROR | %s", e)
        return AskResponse(
            answer="Sorry, I am facing a temporary issue. Please try again.",
            error=str(e),
        )


@app.post("/whatsapp")
async def whatsapp_webhook(request: Request, body: Optional[WhatsAppPayload] = None):
    """
    AiSensy WhatsApp webhook — smart admission assistant.

    Intent priority (strict order):
      1. Complaint / escalation     → escalate immediately
      2. Human escalation request   → connect to counsellor
      3. Factual question (RAG)     → answer ONLY, no follow-up
      4. Admission intent           → start/resume qualification flow
      5. Active qualification flow  → continue collecting data
      6. General fallback           → short RAG answer
    """
    phone = ""  # Pre-init so fallback in except block can reference it
    try:
        # Use typed body if available, fall back to raw request parsing
        if body is not None:
            payload = {"data": body.data}
        else:
            payload = await request.json()
        logger.info("WEBHOOK_RECEIVED | payload_keys=%s", list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__)

        message, phone, user_name, message_type = _extract_from_payload(payload)

        logger.info(
            "USER_MESSAGE | phone=%s | name=%s | type=%s | message=\"%s\"",
            phone, user_name, message_type, message[:120] if message else "",
        )

        # ── Validation: only process TEXT messages ───────────────────
        if message_type != "TEXT":
            logger.info("IGNORED | phone=%s | reason=non-text (%s)", phone, message_type)
            return {"status": "ignored", "reason": f"unsupported message_type: {message_type}"}

        if not message:
            logger.warning("IGNORED | phone=%s | reason=empty message", phone)
            return {"status": "ignored", "reason": "empty message"}

        user_id = phone or "whatsapp_user"

        # ── Record activity for follow-up tracking ──────────────────
        if phone:
            await record_user_activity(phone)

        # ── Intelligence: scoring, stage, persona ───────────────────
        score_from_message(user_id, message)
        stage = detect_and_update_stage(user_id, message)
        persona = detect_and_update_persona(user_id, message)

        # Update lead intelligence if in qualification
        update_lead_intelligence(user_id, stage=stage.value, persona=persona.value)

        logger.info(
            "INTELLIGENCE | user=%s | score=%d | stage=%s | persona=%s",
            user_id, get_score(user_id), stage.value, persona.value,
        )

        # ── Classify intent using the unified router ─────────────────
        in_qual = is_in_qualification(user_id)
        has_complaint = detect_complaint(message)
        has_human_req = detect_human_request(message)

        route = classify_message(
            message,
            is_in_qualification=in_qual,
            has_complaint=has_complaint,
            has_human_request=has_human_req,
        )

        logger.info(
            "ROUTE | user=%s | route=%s | in_qual=%s",
            user_id, route.value, in_qual,
        )

        # =============================================================
        # PRIORITY 1: Complaint / Escalation
        # =============================================================
        if route == Route.COMPLAINT:
            lead_type = classify_lead(message).value
            logger.info("COMPLAINT | user=%s | message=\"%s\"", phone, message[:80])
            add_internal_flag(user_id, "COMPLAINT")
            set_concern(user_id, "Complaint / Dissatisfaction")

            await notify_admin(
                user_phone=phone or "unknown",
                user_message=message,
                user_name=user_name,
                lead_type=lead_type,
                reason="Complaint / Dissatisfaction",
            )
            reply = "We're sorry to hear that. Our team will reach out to you shortly \U0001f64f"
            if phone:
                await mark_followup_escalated(phone)
                await send_whatsapp_message(phone, reply)

            return {"status": "complaint", "answer": reply}

        # =============================================================
        # PRIORITY 2: Human Escalation
        # =============================================================
        if route == Route.HUMAN_ESCALATION:
            lead_type = classify_lead(message).value
            logger.info("ESCALATION | user=%s | type=%s", phone, lead_type)

            await notify_admin(
                user_phone=phone or "unknown",
                user_message=message,
                user_name=user_name,
                lead_type=lead_type,
            )

            try:
                await send_lead_to_zapier(build_quick_payload(
                    phone=phone or "unknown", message=message,
                    lead_type=lead_type, user_name=user_name,
                    score=get_score(user_id), stage=stage.value,
                ))
            except Exception:
                pass

            reply = "Our counsellor will contact you shortly \U0001f44d"
            if phone:
                await mark_followup_escalated(phone)
                await send_whatsapp_message(phone, reply)

            return {"status": "escalated", "answer": reply}

        # =============================================================
        # PRIORITY 3: Factual Question → RAG only, NO follow-up
        # =============================================================
        if route == Route.FACTUAL_QUESTION:
            # If user is mid-qualification and asks a question,
            # answer the question then re-prompt the current step
            if in_qual:
                rag_answer = await ask_async(question=message, user_id=user_id)
                rag_answer = format_whatsapp_response(rag_answer, is_factual=True)
                current_prompt = get_current_qual_prompt(user_id)
                # Send answer and re-prompt as separate messages for clarity
                if phone:
                    await send_whatsapp_message(phone, rag_answer)
                    await send_whatsapp_message(phone, current_prompt)
                reply = rag_answer
                logger.info("AI_RESPONSE | phone=%s | type=factual_midflow | response=\"%s\"", phone, reply[:120])
                return {"status": "factual_midflow", "answer": reply}

            # Pure factual question — answer only, no qualification trigger
            answer = await ask_async(question=message, user_id=user_id)
            answer = format_whatsapp_response(answer, is_factual=True)

            # Check for confusion escalation
            from rag.retriever import NO_CONTEXT_MSG
            if answer.startswith(NO_CONTEXT_MSG[:30]):
                confusion_count = record_confusion(user_id)
                if should_escalate_confusion(user_id):
                    await notify_admin(
                        user_phone=phone or "unknown",
                        user_message=message,
                        user_name=user_name,
                        lead_type="General Enquiry",
                        reason=f"Repeated confusion ({confusion_count} unanswered)",
                    )
            else:
                reset_confusion(user_id)

            if phone:
                await send_whatsapp_message(phone, answer)

            logger.info("AI_RESPONSE | phone=%s | type=factual | response=\"%s\"", phone, answer[:120])
            return {"status": "factual", "answer": answer}

        # =============================================================
        # PRIORITY 4: Admission Intent → start/resume qualification
        # =============================================================
        if route == Route.ADMISSION_INTENT:
            course = detect_course_interest(message)
            update_score(user_id, ScoreAction.ASKED_ADMISSION)

            logger.info(
                "ADMISSION_INTENT | user=%s | course=%s | score=%d",
                phone, course or "General", get_score(user_id),
            )

            # start_lead_qualification now resumes if already active
            qual_prompt = start_lead_qualification(
                user_id=user_id,
                phone=phone,
                course=course or "",
            )

            if phone:
                await send_whatsapp_message(phone, qual_prompt)

            return {"status": "qualification_started", "answer": qual_prompt}

        # =============================================================
        # PRIORITY 5: Active Qualification Flow
        # =============================================================
        if route == Route.QUALIFICATION:
            result = process_qualification_message(user_id, message)

            if result is None:
                # User asked a question mid-flow — answer via RAG + re-prompt
                rag_answer = await ask_async(question=message, user_id=user_id)
                rag_answer = format_whatsapp_response(rag_answer, is_factual=True)
                current_prompt = get_current_qual_prompt(user_id)
                if phone:
                    await send_whatsapp_message(phone, rag_answer)
                    await send_whatsapp_message(phone, current_prompt)
                reply = rag_answer
            else:
                reply = result
                if phone:
                    await send_whatsapp_message(phone, reply)

            # Check if qualification just completed
            lead = complete_lead(user_id)
            if lead:
                lead.stage = stage.value
                lead.persona = persona.value

                await save_lead_to_db(lead)
                await notify_admin_qualified_lead(lead)

                try:
                    await send_lead_to_zapier(build_lead_payload(lead, message))
                except Exception:
                    pass

                if phone:
                    await mark_followup_completed(phone)

            logger.info("AI_RESPONSE | phone=%s | type=qualification | response=\"%s\"", phone, reply[:120])
            return {"status": "qualification_step", "answer": reply}

        # =============================================================
        # PRIORITY 6: General Fallback → short RAG answer
        # =============================================================
        answer = await ask_async(question=message, user_id=user_id)
        answer = format_whatsapp_response(answer, is_factual=False)

        # Confusion escalation
        from rag.retriever import NO_CONTEXT_MSG
        if answer.startswith(NO_CONTEXT_MSG[:30]):
            confusion_count = record_confusion(user_id)
            if should_escalate_confusion(user_id):
                add_internal_flag(user_id, "REPEATED_CONFUSION")
                await notify_admin(
                    user_phone=phone or "unknown",
                    user_message=message,
                    user_name=user_name,
                    lead_type="General Enquiry",
                    reason=f"Repeated confusion ({confusion_count} unanswered)",
                )
        else:
            reset_confusion(user_id)

        if phone:
            await send_whatsapp_message(phone, answer)

        logger.info("AI_RESPONSE | phone=%s | type=general | response=\"%s\"", phone, answer[:120])
        return {"status": "ok", "answer": answer}

    except Exception as e:
        logger.exception("Error in /whatsapp webhook")
        if phone:
            try:
                await send_whatsapp_message(phone, FALLBACK_MESSAGE)
                logger.info("FALLBACK_SENT | phone=%s", phone)
            except Exception:
                logger.error("FALLBACK_FAILED | phone=%s", phone)
        return {"status": "error", "detail": str(e)}


# =====================================================================
# Local Development Entry Point
# =====================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
