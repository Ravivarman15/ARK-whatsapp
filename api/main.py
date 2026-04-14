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
    - Model pre-loading at startup
    - Lead qualification conversation flow
    - Lead scoring + segmentation + stage detection
    - Multi-intent handling + high-intent interruption
    - Parent persona detection
    - Complaint / confusion escalation
    - Hot lead detection and admin notification
    - Human escalation handling
    - ADA conversion engine
    - Psychological trigger rotation
    - Tamil language switching
    - Structured logging
    - CORS middleware
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

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
    detect_hot_lead,
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
from rag.intent_engine import (
    detect_intents, detect_high_intent, is_multi_intent,
    get_primary_intent, Intent, HIGH_INTENT_RESPONSE,
)

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

    # Ensure phone has country code prefix
    destination = phone if phone.startswith("+") else f"+{phone}"

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
    try:
        answer = await ask_async(
            question=body.question,
            user_id=body.user_id or "anonymous",
        )
        return AskResponse(answer=answer)
    except Exception as e:
        logger.exception("Error in /ask endpoint")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    """
    AiSensy WhatsApp webhook with full intelligent lead pipeline.

    Flow:
      1. Extract message, phone, name from payload
      2. Record activity for follow-up tracking
      3. Run intelligence engines (scoring, stage, persona)
      4. Check high-intent interruption -> capture phone + escalate
      5. If user is in qualification flow -> continue collecting data
      6. Check complaint -> escalate with reason
      7. Check hot lead intent -> notify admin immediately
      8. Check human escalation intent -> classify + notify admin
      9. Check course interest -> start qualification flow
     10. Normal RAG pipeline with intelligence context
    """
    phone = ""  # Pre-init so fallback in except block can reference it
    try:
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

        # ── Step 1: High-intent interruption ────────────────────────
        if detect_high_intent(message):
            logger.info(
                "HIGH_INTENT_INTERRUPT | user=%s | message=\"%s\"",
                phone, message[:80],
            )
            update_score(user_id, ScoreAction.ASKED_ADMISSION)

            # Notify admin immediately
            await notify_admin_hot_lead(
                user_phone=phone or "unknown",
                user_message=message,
                lead_type="Admission Intent (HIGH PRIORITY)",
            )
            if phone:
                await mark_followup_escalated(phone)
                await send_whatsapp_message(phone, HIGH_INTENT_RESPONSE)

            logger.info("AI_RESPONSE | phone=%s | type=high_intent | response=\"%s\"", phone, HIGH_INTENT_RESPONSE[:120])
            return {"status": "high_intent", "answer": HIGH_INTENT_RESPONSE}

        # ── Step 2: Active qualification flow ───────────────────────
        if is_in_qualification(user_id):
            result = process_qualification_message(user_id, message)

            if result is None:
                # User asked a question mid-flow — answer via RAG + re-prompt
                rag_answer = await ask_async(question=message, user_id=user_id)
                current_prompt = get_current_qual_prompt(user_id)
                reply = rag_answer + "\n\n" + current_prompt
            else:
                reply = result

            # Check if qualification just completed
            lead = complete_lead(user_id)
            if lead:
                # Update with latest intelligence
                lead.stage = stage.value
                lead.persona = persona.value

                # Save to database and notify admin
                await save_lead_to_db(lead)
                await notify_admin_qualified_lead(lead)

                # Send to Google Sheets via Zapier
                try:
                    await send_lead_to_zapier(build_lead_payload(lead, message))
                except Exception:
                    pass  # logged inside send_lead_to_zapier

                if phone:
                    await mark_followup_completed(phone)

            if phone:
                await send_whatsapp_message(phone, reply)

            logger.info("AI_RESPONSE | phone=%s | type=qualification | response=\"%s\"", phone, reply[:120])
            return {"status": "qualification_step", "answer": reply}

        # ── Step 3: Complaint detection ─────────────────────────────
        if detect_complaint(message):
            lead_type = classify_lead(message).value
            logger.info(
                "COMPLAINT | user=%s | message=\"%s\"",
                phone, message[:80],
            )
            add_internal_flag(user_id, "COMPLAINT")
            set_concern(user_id, "Complaint / Dissatisfaction")

            await notify_admin(
                user_phone=phone or "unknown",
                user_message=message,
                user_name=user_name,
                lead_type=lead_type,
                reason="Complaint / Dissatisfaction",
            )
            if phone:
                await mark_followup_escalated(phone)
                await send_whatsapp_message(phone, COMPLAINT_REPLY)

            logger.info("AI_RESPONSE | phone=%s | type=complaint | response=\"%s\"", phone, COMPLAINT_REPLY[:120])
            return {"status": "complaint", "answer": COMPLAINT_REPLY}

        # ── Step 4: Hot lead detection ──────────────────────────────
        if detect_hot_lead(message):
            lead_type = classify_lead(message).value
            logger.info(
                "HOT_LEAD | user=%s | type=%s | score=%d | message=\"%s\"",
                phone, lead_type, get_score(user_id), message[:80],
            )

            # Notify admin immediately with HIGH priority
            await notify_admin_hot_lead(
                user_phone=phone or "unknown",
                user_message=message,
                lead_type=lead_type,
            )

            # Send to Google Sheets via Zapier
            try:
                await send_lead_to_zapier(build_quick_payload(
                    phone=phone or "unknown", message=message,
                    lead_type=lead_type, priority="HIGH",
                    score=get_score(user_id), stage=stage.value,
                ))
            except Exception:
                pass  # logged inside send_lead_to_zapier

            if phone:
                await mark_followup_escalated(phone)

            # Also reply to user with escalation message
            if phone:
                await send_whatsapp_message(phone, ESCALATION_REPLY)

            logger.info("AI_RESPONSE | phone=%s | type=hot_lead | response=\"%s\"", phone, ESCALATION_REPLY[:120])
            return {"status": "hot_lead", "answer": ESCALATION_REPLY}

        # ── Step 5: Human escalation ────────────────────────────────
        if detect_human_request(message):
            lead_type = classify_lead(message).value
            logger.info(
                "ESCALATION | user=%s | type=%s | message=\"%s\"",
                phone, lead_type, message[:80],
            )

            await notify_admin(
                user_phone=phone or "unknown",
                user_message=message,
                user_name=user_name,
                lead_type=lead_type,
            )

            # Send to Google Sheets via Zapier
            try:
                await send_lead_to_zapier(build_quick_payload(
                    phone=phone or "unknown", message=message,
                    lead_type=lead_type, user_name=user_name,
                    score=get_score(user_id), stage=stage.value,
                ))
            except Exception:
                pass  # logged inside send_lead_to_zapier

            if phone:
                await mark_followup_escalated(phone)

            if phone:
                await send_whatsapp_message(phone, ESCALATION_REPLY)

            logger.info("AI_RESPONSE | phone=%s | type=escalation | response=\"%s\"", phone, ESCALATION_REPLY[:120])
            return {"status": "escalated", "answer": ESCALATION_REPLY}

        # ── Step 6: Course interest -> start qualification ──────────
        course = detect_course_interest(message)
        if course:
            logger.info(
                "COURSE_INTEREST | user=%s | course=%s | score=%d | message=\"%s\"",
                phone, course, get_score(user_id), message[:80],
            )

            first_prompt = start_lead_qualification(
                user_id=user_id,
                phone=phone,
                course=course,
            )

            if phone:
                # First send a RAG answer about the course, then start qualification
                rag_answer = await ask_async(question=message, user_id=user_id)
                combined_reply = rag_answer + "\n\n" + first_prompt
                await send_whatsapp_message(phone, combined_reply)
            else:
                combined_reply = first_prompt

            return {"status": "qualification_started", "answer": combined_reply}

        # ── Step 7: Normal RAG (with intelligence context) ──────────
        answer = await ask_async(question=message, user_id=user_id)

        # Check for confusion escalation (if answer is no-context)
        from rag.retriever import NO_CONTEXT_MSG
        if answer.startswith(NO_CONTEXT_MSG[:30]):
            confusion_count = record_confusion(user_id)
            if should_escalate_confusion(user_id):
                logger.info(
                    "CONFUSION_ESCALATION | user=%s | count=%d",
                    user_id, confusion_count,
                )
                add_internal_flag(user_id, "REPEATED_CONFUSION")
                await notify_admin(
                    user_phone=phone or "unknown",
                    user_message=message,
                    user_name=user_name,
                    lead_type="General Enquiry",
                    reason=f"Repeated confusion ({confusion_count} unanswered queries)",
                )
        else:
            reset_confusion(user_id)

        if phone:
            await send_whatsapp_message(phone, answer)

        logger.info("AI_RESPONSE | phone=%s | type=rag | response=\"%s\"", phone, answer[:120])
        return {"status": "ok", "answer": answer}

    except Exception as e:
        logger.exception("Error in /whatsapp webhook")
        # Send fallback message to user if we have their phone
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
