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

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import get_settings
from rag.whatsapp_sender import send_whatsapp_message
from rag.retriever import ask_async
from rag.escalation import (
    detect_human_request, detect_complaint, notify_admin,
    record_confusion, reset_confusion, should_escalate_confusion,
    ESCALATION_REPLY, COMPLAINT_REPLY,
)
from rag.lead_manager import (
    classify_lead,
    detect_course_interest,
    detect_hot_lead,
    is_in_qualification,
    start_lead_qualification,
    store_lead_data,
    process_qualification_message,
    get_current_qual_prompt,
    get_soft_reprompt,
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
    fire_and_forget as zapier_fire_and_forget,
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
    logger.info("Using page index (TF-IDF) for document search — no embeddings.")

    # Auto-build index if missing (safety net for first Render deploy)
    from rag.page_index import INDEX_PATH, build_index
    from pathlib import Path
    if not Path(INDEX_PATH).exists():
        logger.warning("Page index not found — building now from ark_details.docx ...")
        try:
            doc = Path(__file__).parent.parent / "documents" / "ark_details.docx"
            build_index(doc_path=doc)
            logger.info("Page index built successfully.")
        except Exception as _e:
            logger.error("Failed to build page index: %s", _e)

    # Zapier health check — surface misconfig at boot, not the first lead
    _s = get_settings()
    if _s.ZAPIER_WEBHOOK_URL:
        logger.info("Zapier webhook configured: %s...", _s.ZAPIER_WEBHOOK_URL[:60])
    else:
        logger.warning(
            "ZAPIER_WEBHOOK_URL is empty — leads will NOT be written to Google Sheet"
        )

    start_followup_scheduler()
    logger.info("Server is live.")
    yield
    stop_followup_scheduler()
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


FALLBACK_MESSAGE = "Sorry, I'm having trouble right now. Please try again in a moment."
RAG_ERROR_MESSAGE = "I couldn't process that right now. Please try again or ask a different question."


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


@app.post("/admin/test")
async def admin_test():
    """
    Diagnostic — fire a synthetic admin alert via the AiSensy Campaign
    API + approved UTILITY template path, so you can verify
    ADMIN_WHATSAPP_NUMBER, AISENSY_CAMPAIGN_API_KEY, and
    AISENSY_CAMPAIGN_NAME without having to simulate a real user message.

    Returns the delivery result + the config surface so misconfig is
    visible in the response (not just in logs).
    """
    from rag.escalation import _is_valid_admin_phone
    from rag.whatsapp_sender import send_admin_alert

    s = get_settings()
    ok, reason = _is_valid_admin_phone(s.ADMIN_WHATSAPP_NUMBER)
    if not ok:
        return {
            "sent": False,
            "admin_phone_valid": False,
            "reason": reason,
        }

    delivered = await send_admin_alert(
        phone=s.ADMIN_WHATSAPP_NUMBER,
        message=(
            "Diagnostic admin alert from /admin/test — if you see this "
            "in WhatsApp, the Campaign API + UTILITY template path is "
            "working."
        ),
    )
    return {
        "sent": delivered,
        "admin_phone_valid": True,
        "admin_phone_masked": s.ADMIN_WHATSAPP_NUMBER[:4] + "****" + s.ADMIN_WHATSAPP_NUMBER[-3:],
        "aisensy_campaign_api_key_set": bool(s.AISENSY_CAMPAIGN_API_KEY),
        "aisensy_campaign_name": s.AISENSY_CAMPAIGN_NAME or None,
        "aisensy_api_key_set": bool(s.AISENSY_API_KEY),
        "aisensy_project_id_set": bool(s.AISENSY_PROJECT_ID),
        "note": (
            "If sent=false, check logs for ADMIN_ALERT_FAIL_PERMANENT "
            "(likely: wrong campaign name, revoked JWT, or underlying "
            "template got reclassified to MARKETING). AISENSY_API_KEY / "
            "AISENSY_PROJECT_ID are shown because they're used by the "
            "session-text fallback inside the 24h window."
        ),
    }


@app.post("/zapier/test")
async def zapier_test():
    """
    Diagnostic endpoint — sends a synthetic lead to the configured
    Zapier webhook so you can verify the integration end-to-end
    without having to simulate a full WhatsApp conversation.

    Returns the raw True/False result so a misconfigured URL, 4xx,
    or timeout is immediately visible.
    """
    s = get_settings()
    sample = build_quick_payload(
        phone="919000000000",
        message="[diagnostic test from /zapier/test]",
        lead_type="Diagnostic",
        priority="LOW",
        user_name="Zapier Test",
        score=0,
        stage="TEST",
    )
    ok = await send_lead_to_zapier(sample)
    return {
        "sent": ok,
        "webhook_configured": bool(s.ZAPIER_WEBHOOK_URL),
        "webhook_prefix": s.ZAPIER_WEBHOOK_URL[:60] if s.ZAPIER_WEBHOOK_URL else None,
        "payload": sample,
    }


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
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    AiSensy WhatsApp webhook — smart admission assistant.

    IMPORTANT: AiSensy retries webhook calls that take longer than a few
    seconds, which was producing duplicate messages and cascading delays.
    So we read the payload, fire the full processing pipeline as a
    background task, and ACK AiSensy immediately. The user's reply is
    then sent via the outbound AiSensy Project API from the background
    task — usually within 1-2 seconds.
    """
    try:
        payload = await request.json()
    except Exception as e:
        logger.error("WEBHOOK_BAD_PAYLOAD | %s", e)
        return {"status": "bad_payload"}

    background_tasks.add_task(_process_incoming_message, payload)
    return {"status": "accepted"}


async def _process_incoming_message(payload: dict) -> None:
    """
    Actual pipeline — runs after the webhook has already returned.

    Intent priority (strict order):
      1. Complaint / escalation     → escalate immediately
      2. Human escalation request   → connect to counsellor
      3. Hot lead (admission-ready) → instant admin alert + qualification
      4. Multi-intent (Q + verb)    → answer via RAG, then start qual
      5. Factual question (RAG)     → answer; if mid-qual, soft re-prompt
      6. Admission intent           → start/resume qualification flow
      7. Active qualification flow  → continue collecting data
      8. General fallback           → short RAG answer
    """
    phone = ""  # Pre-init so fallback in except block can reference it
    try:
        logger.info(
            "WEBHOOK_RECEIVED | payload_keys=%s",
            list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__,
        )

        message, phone, user_name, message_type = _extract_from_payload(payload)

        logger.info(
            "USER_MESSAGE | phone=%s | name=%s | type=%s | message=\"%s\"",
            phone, user_name, message_type, message[:120] if message else "",
        )

        # ── Validation: only process TEXT messages ───────────────────
        if message_type != "TEXT":
            logger.info("IGNORED | phone=%s | reason=non-text (%s)", phone, message_type)
            return

        if not message:
            logger.warning("IGNORED | phone=%s | reason=empty message", phone)
            return

        user_id = phone or "whatsapp_user"

        # ── Record activity for follow-up tracking ──────────────────
        if phone:
            try:
                await record_user_activity(phone)
            except Exception as e:
                logger.error("ACTIVITY_RECORD_FAILED | phone=%s | %s", phone, e)

        # ── Intelligence: scoring, stage, persona ───────────────────
        # These are non-critical — if they fail, continue with defaults
        stage = None
        persona = None
        try:
            score_from_message(user_id, message)
            stage = detect_and_update_stage(user_id, message)
            persona = detect_and_update_persona(user_id, message)
            update_lead_intelligence(
                user_id,
                stage=stage.value if stage else "",
                persona=persona.value if persona else "",
            )
        except Exception as e:
            logger.error("INTELLIGENCE_FAILED | user=%s | %s", user_id, e)

        if stage and persona:
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
        # HOT-LEAD ALERT (runs alongside normal routing)
        # =============================================================
        # Fire a HIGH-priority admin ping the moment the user sends a
        # buying-intent message (e.g. "fees negotiation", "I want to join",
        # "call me now"). This must NOT block the user reply, so run it
        # in the background. We still let the normal route continue so
        # the user gets a proper conversational response.
        # Admission intent itself is a hot signal — covers Thanglish/Tamil
        # variants that detect_hot_lead() (phrase-based) can't enumerate.
        # Only treat it as hot when starting a NEW qualification, so we
        # don't re-ping admin for every mid-flow reply.
        is_admission_new = route == Route.ADMISSION_INTENT and not in_qual

        if (
            not has_complaint
            and not has_human_req
            and (detect_hot_lead(message) or is_admission_new)
        ):
            lead_type_hot = classify_lead(message).value
            add_internal_flag(user_id, "HOT_LEAD")
            logger.info(
                "HOT_LEAD | user=%s | phone=%s | type=%s",
                user_id, phone, lead_type_hot,
            )

            async def _safe_hot_lead_notify(p: str, m: str, t: str) -> None:
                try:
                    ok = await notify_admin_hot_lead(
                        user_phone=p, user_message=m, lead_type=t,
                    )
                    if not ok:
                        logger.error("HOT_LEAD_NOTIFY_FAILED | user=%s | type=%s", p, t)
                except Exception as exc:
                    logger.exception("HOT_LEAD_NOTIFY_ERROR | user=%s | %s", p, exc)

            asyncio.create_task(
                _safe_hot_lead_notify(phone or "unknown", message, lead_type_hot)
            )
            # Also log the hot lead to Google Sheet so it appears
            # in the pipeline even before qualification completes.
            zapier_fire_and_forget(build_quick_payload(
                phone=phone or "unknown",
                message=message,
                lead_type=lead_type_hot,
                priority="HIGH",
                user_name=user_name,
                score=get_score(user_id),
                stage=stage.value if stage else "",
            ))

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
            zapier_fire_and_forget(build_quick_payload(
                phone=phone or "unknown",
                message=message,
                lead_type=lead_type,
                priority="HIGH",
                user_name=user_name,
                score=get_score(user_id),
                stage=stage.value if stage else "",
            ))
            reply = "We're sorry to hear that. Our team will reach out to you shortly \U0001f64f"
            if phone:
                await mark_followup_escalated(phone)
                await send_whatsapp_message(phone, reply)
            return

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

            zapier_fire_and_forget(build_quick_payload(
                phone=phone or "unknown", message=message,
                lead_type=lead_type, user_name=user_name,
                score=get_score(user_id), stage=stage.value if stage else "",
            ))

            reply = "Our counsellor will contact you shortly \U0001f44d"
            if phone:
                await mark_followup_escalated(phone)
                await send_whatsapp_message(phone, reply)
            return

        # =============================================================
        # PRIORITY 3: Multi-Intent → answer via RAG, then start qual
        # =============================================================
        # User mixed a question with an explicit admission verb, e.g.
        # "I want to join NEET, what are fees?". Answer first so they
        # don't feel ignored, then start (or resume) the qualification
        # flow. start_lead_qualification handles the resume case so we
        # don't re-ask fields the user already provided.
        if route == Route.MULTI_INTENT:
            try:
                rag_answer = await ask_async(question=message, user_id=user_id)
                rag_answer = format_whatsapp_response(rag_answer, is_factual=True)
            except Exception as rag_err:
                logger.exception("RAG_ERROR (multi_intent) | %s: %s", type(rag_err).__name__, rag_err)
                rag_answer = RAG_ERROR_MESSAGE

            course = detect_course_interest(message)
            update_score(user_id, ScoreAction.ASKED_ADMISSION)
            qual_prompt = start_lead_qualification(
                user_id=user_id, phone=phone, course=course or "",
            )
            if phone:
                await send_whatsapp_message(phone, rag_answer)
                await send_whatsapp_message(phone, qual_prompt)
            logger.info(
                "AI_RESPONSE | phone=%s | type=multi_intent | answer=\"%s\" | qual=\"%s\"",
                phone, rag_answer[:80], qual_prompt[:80],
            )
            return

        # =============================================================
        # PRIORITY 4: Factual Question → RAG; if mid-qual, soft re-prompt
        # =============================================================
        if route == Route.FACTUAL_QUESTION:
            # Mid-qualification: answer the question, then politely
            # nudge back to the pending field with a soft "Also, can
            # I get…" re-prompt. The qual state is preserved in
            # `_active_leads` and the user can complete the flow
            # without restarting from name. Pure (non-qual) factual
            # questions get the answer alone — no qualification trigger.
            if in_qual:
                try:
                    rag_answer = await ask_async(question=message, user_id=user_id)
                    rag_answer = format_whatsapp_response(rag_answer, is_factual=True)
                except Exception as rag_err:
                    logger.exception("RAG_ERROR (factual_midflow) | %s: %s", type(rag_err).__name__, rag_err)
                    rag_answer = RAG_ERROR_MESSAGE
                soft_prompt = get_soft_reprompt(user_id)
                if phone:
                    await send_whatsapp_message(phone, rag_answer)
                    if soft_prompt:
                        await send_whatsapp_message(phone, soft_prompt)
                logger.info(
                    "AI_RESPONSE | phone=%s | type=factual_midflow | answer=\"%s\" | soft=\"%s\"",
                    phone, rag_answer[:80], soft_prompt[:80],
                )
                return

            # Pure factual question — answer only, no qualification trigger
            try:
                answer = await ask_async(question=message, user_id=user_id)
                answer = format_whatsapp_response(answer, is_factual=True)
            except Exception as rag_err:
                logger.exception("RAG_ERROR (factual) | %s: %s", type(rag_err).__name__, rag_err)
                answer = RAG_ERROR_MESSAGE

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
            return

        # =============================================================
        # PRIORITY 5: Admission Intent → start/resume qualification
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
            return

        # =============================================================
        # PRIORITY 6: Active Qualification Flow
        # =============================================================
        if route == Route.QUALIFICATION:
            result = process_qualification_message(user_id, message)

            if result is None:
                # User asked a question mid-flow — answer via RAG, then
                # send a soft "Also, can I get…" re-prompt instead of
                # the terse default prompt. Falls back to the default
                # prompt if the soft variant isn't defined for this step.
                try:
                    rag_answer = await ask_async(question=message, user_id=user_id)
                    rag_answer = format_whatsapp_response(rag_answer, is_factual=True)
                except Exception as rag_err:
                    logger.exception("RAG_ERROR (qual_midflow) | %s: %s", type(rag_err).__name__, rag_err)
                    rag_answer = RAG_ERROR_MESSAGE
                follow_prompt = get_soft_reprompt(user_id) or get_current_qual_prompt(user_id)
                if phone:
                    await send_whatsapp_message(phone, rag_answer)
                    if follow_prompt:
                        await send_whatsapp_message(phone, follow_prompt)
                reply = rag_answer
            else:
                reply = result
                if phone:
                    await send_whatsapp_message(phone, reply)

            # Check if qualification just completed
            lead = complete_lead(user_id)
            if lead:
                lead.stage = stage.value if stage else ""
                lead.persona = persona.value if persona else ""

                await save_lead_to_db(lead)
                await notify_admin_qualified_lead(lead)
                zapier_fire_and_forget(build_lead_payload(lead, message))

                if phone:
                    await mark_followup_completed(phone)

            logger.info("AI_RESPONSE | phone=%s | type=qualification | response=\"%s\"", phone, reply[:120])
            return

        # =============================================================
        # PRIORITY 7: General Fallback → short RAG answer
        # =============================================================
        try:
            answer = await ask_async(question=message, user_id=user_id)
            answer = format_whatsapp_response(answer, is_factual=False)
        except Exception as rag_err:
            logger.exception("RAG_ERROR (general) | %s: %s", type(rag_err).__name__, rag_err)
            answer = RAG_ERROR_MESSAGE

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
        return

    except Exception as e:
        logger.exception("Error processing WhatsApp message")
        if phone:
            try:
                await send_whatsapp_message(phone, FALLBACK_MESSAGE)
                logger.info("FALLBACK_SENT | phone=%s", phone)
            except Exception:
                logger.error("FALLBACK_FAILED | phone=%s", phone)


# =====================================================================
# Local Development Entry Point
# =====================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
