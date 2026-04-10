"""
scripts/test_bot.py
───────────────────
Terminal-based chatbot for testing the ARK AI Bot locally
without WhatsApp / AiSensy.

Reuses the same lead management + RAG + intelligence pipeline
as the /whatsapp endpoint, but runs everything synchronously
in a terminal loop.

Features:
    - Full conversation with RAG
    - Lead qualification with validation
    - Lead scoring display
    - Decision stage tracking
    - Student segmentation
    - Parent persona detection
    - Multi-intent handling
    - High-intent interruption
    - Complaint detection
    - Counsellor summary on completion
    - Status command to view current state
    - Escalation logging

Usage:
    python scripts/test_bot.py

Type 'exit' or 'quit' to stop.
Type 'reset' to clear conversation memory and lead state.
Type 'status' to see current user intelligence state.
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-16s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)

from rag.retriever import ask, get_memory
from rag.escalation import (
    detect_human_request, detect_complaint,
    record_confusion, reset_confusion, should_escalate_confusion,
    ESCALATION_REPLY, COMPLAINT_REPLY,
)
from rag.lead_manager import (
    classify_lead,
    detect_hot_lead,
    detect_course_interest,
    is_in_qualification,
    start_lead_qualification,
    process_qualification_message,
    get_current_qual_prompt,
    complete_lead,
    generate_counsellor_summary,
    update_lead_intelligence,
    add_internal_flag,
    set_concern,
)
from rag.scoring import (
    score_from_message, get_score, get_lead_type,
    update_score, ScoreAction, reset_score,
)
from rag.stage_detector import (
    detect_and_update_stage, get_stage, reset_stage,
)
from rag.segmentation import detect_segment_from_message
from rag.persona_detector import (
    detect_and_update_persona, get_persona, reset_persona,
)
from rag.intent_engine import (
    detect_intents, detect_high_intent, is_multi_intent,
    Intent, HIGH_INTENT_RESPONSE,
)
from rag.psychology_engine import reset_triggers
from rag.zapier_integration import (
    send_lead_to_zapier_sync, build_lead_payload, build_quick_payload,
)

# ── Config ────────────────────────────────────────────────────────────
TEST_USER_ID = "test_user"
TEST_PHONE = "919999999999"

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_banner():
    print()
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}   ARK AI Bot — Local Test Mode (v4.0){RESET}")
    print(f"{CYAN}   Full pipeline: RAG + Scoring + Segmentation + Persona{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"   Type {YELLOW}'exit'{RESET} to quit, {YELLOW}'reset'{RESET} to clear state,")
    print(f"   {YELLOW}'status'{RESET} to view intelligence state.")
    print()


def print_intelligence_bar(user_id: str):
    """Print a compact intelligence status bar after each response."""
    score = get_score(user_id)
    lead_type = get_lead_type(user_id).value
    stage = get_stage(user_id).value
    persona = get_persona(user_id).value

    # Color the lead type
    type_colors = {
        "COLD": DIM, "WARM": YELLOW, "HOT": RED, "VERY_HOT": f"{BOLD}{RED}",
    }
    type_color = type_colors.get(lead_type, "")

    print(f"   {DIM}┌─ Score: {score} ({type_color}{lead_type}{RESET}{DIM}) │ "
          f"Stage: {stage} │ Persona: {persona}{RESET}{DIM} ─┐{RESET}")


def print_status(user_id: str):
    """Print detailed intelligence state for the user."""
    from rag.lead_manager import get_lead_data

    score = get_score(user_id)
    lead_type = get_lead_type(user_id).value
    stage = get_stage(user_id).value
    persona = get_persona(user_id).value

    print(f"\n   {BOLD}{MAGENTA}{'─'*50}{RESET}")
    print(f"   {BOLD}{MAGENTA}User Intelligence State{RESET}")
    print(f"   {MAGENTA}{'─'*50}{RESET}")
    print(f"   {MAGENTA}Lead Score:   {score}{RESET}")
    print(f"   {MAGENTA}Lead Type:    {lead_type}{RESET}")
    print(f"   {MAGENTA}Stage:        {stage}{RESET}")
    print(f"   {MAGENTA}Persona:      {persona}{RESET}")

    lead = get_lead_data(user_id)
    if lead:
        print(f"   {MAGENTA}Segment:      {lead.segment or 'Not detected'}{RESET}")
        print(f"   {MAGENTA}Course:       {lead.course or 'Not detected'}{RESET}")
        print(f"   {MAGENTA}Current Step: {lead.current_step.value}{RESET}")
        print(f"   {MAGENTA}Name:         {lead.student_name or '-'}{RESET}")
        print(f"   {MAGENTA}Class:        {lead.student_class or '-'}{RESET}")
        print(f"   {MAGENTA}School:       {lead.school or '-'}{RESET}")
        print(f"   {MAGENTA}Phone:        {lead.parent_phone or '-'}{RESET}")
        if lead.internal_flags:
            print(f"   {MAGENTA}Flags:        {', '.join(lead.internal_flags)}{RESET}")
        if lead.concern:
            print(f"   {MAGENTA}Concern:      {lead.concern}{RESET}")

    print(f"   {MAGENTA}{'─'*50}{RESET}\n")


def process_message(message: str) -> str:
    """
    Run the same message processing pipeline as the /whatsapp endpoint.

    Flow:
      1. Intelligence signal detection (scoring, stage, persona)
      2. High-intent interruption check
      3. Qualification flow (if active)
      4. Complaint detection
      5. Hot lead detection
      6. Human escalation
      7. Course interest -> start qualification
      8. Normal RAG answer
    """
    user_id = TEST_USER_ID

    # ── Intelligence signals ────────────────────────────────────
    score_from_message(user_id, message)
    stage = detect_and_update_stage(user_id, message)
    persona = detect_and_update_persona(user_id, message)

    # Update lead intelligence if in qualification
    update_lead_intelligence(user_id, stage=stage.value, persona=persona.value)

    # Multi-intent detection
    intents = detect_intents(message)
    if is_multi_intent(message):
        print(f"\n   {CYAN}[MULTI-INTENT] Detected: {', '.join(i.value for i in intents)}{RESET}")

    # ── Step 1: High-intent interruption ────────────────────────
    if detect_high_intent(message):
        update_score(user_id, ScoreAction.ASKED_ADMISSION)
        print(f"\n   {RED}{BOLD}[HIGH-INTENT INTERRUPT] Admission intent detected!{RESET}")
        print(f"   {RED}(In production: admin would receive immediate WhatsApp alert){RESET}")
        return HIGH_INTENT_RESPONSE

    # ── Step 2: Active qualification flow ───────────────────────
    if is_in_qualification(user_id):
        result = process_qualification_message(user_id, message)

        if result is None:
            # User asked a question mid-flow — answer via RAG + re-prompt
            rag_answer = ask(question=message, user_id=user_id)
            current_prompt = get_current_qual_prompt(user_id)
            reply = rag_answer + "\n\n" + current_prompt
        else:
            reply = result

        lead = complete_lead(user_id)
        if lead:
            # Update with latest intelligence
            lead.stage = stage.value
            lead.persona = persona.value

            print(f"\n   {YELLOW}{'='*50}{RESET}")
            print(f"   {YELLOW}{BOLD}[LEAD QUALIFICATION COMPLETE]{RESET}")
            print(f"   {YELLOW}{'='*50}{RESET}")

            summary = generate_counsellor_summary(lead)
            for line in summary.split("\n"):
                # Strip WhatsApp bold markers for terminal
                line_clean = line.replace("*", "")
                print(f"   {YELLOW}{line_clean}{RESET}")

            print(f"   {YELLOW}{'='*50}{RESET}")
            print(f"   {YELLOW}(In production: counsellor receives this summary via WhatsApp){RESET}")

            # Send to Google Sheets via Zapier
            try:
                ok = send_lead_to_zapier_sync(build_lead_payload(lead, message))
                if ok:
                    print(f"   {CYAN}[ZAPIER] Lead sent to Google Sheets ✓{RESET}")
                else:
                    print(f"   {DIM}[ZAPIER] Webhook not configured or failed{RESET}")
            except Exception:
                pass

        return reply

    # ── Step 3: Complaint detection ─────────────────────────────
    if detect_complaint(message):
        lead_type = classify_lead(message).value
        print(f"\n   {RED}[COMPLAINT DETECTED] Type: {lead_type}{RESET}")
        print(f"   {RED}(In production: admin would be notified with complaint reason){RESET}")
        add_internal_flag(user_id, "COMPLAINT")
        set_concern(user_id, "Complaint / Dissatisfaction")
        return COMPLAINT_REPLY

    # ── Step 4: Hot lead detection ──────────────────────────────
    if detect_hot_lead(message):
        lead_type = classify_lead(message).value
        print(f"\n   {RED}[HOT LEAD] Type: {lead_type} | Score: {get_score(user_id)} | Priority: HIGH{RESET}")
        print(f"   {RED}(In production: admin would receive a WhatsApp alert){RESET}")

        # Send to Google Sheets via Zapier
        try:
            ok = send_lead_to_zapier_sync(build_quick_payload(
                phone=TEST_PHONE, message=message,
                lead_type=lead_type, priority="HIGH",
                score=get_score(user_id),
                stage=detect_and_update_stage(user_id, message).value,
            ))
            if ok:
                print(f"   {CYAN}[ZAPIER] Hot lead sent to Google Sheets ✓{RESET}")
        except Exception:
            pass

        return ESCALATION_REPLY

    # ── Step 5: Human escalation ────────────────────────────────
    if detect_human_request(message):
        lead_type = classify_lead(message).value
        print(f"\n   {YELLOW}[ESCALATION] Type: {lead_type}{RESET}")
        print(f"   {YELLOW}(In production: admin would be notified via WhatsApp){RESET}")

        # Send to Google Sheets via Zapier
        try:
            ok = send_lead_to_zapier_sync(build_quick_payload(
                phone=TEST_PHONE, message=message,
                lead_type=lead_type,
                score=get_score(user_id),
                stage=detect_and_update_stage(user_id, message).value,
            ))
            if ok:
                print(f"   {CYAN}[ZAPIER] Escalation sent to Google Sheets ✓{RESET}")
        except Exception:
            pass

        return ESCALATION_REPLY

    # ── Step 6: Course interest -> start qualification ──────────
    course = detect_course_interest(message)
    if course:
        print(f"\n   {CYAN}[COURSE INTEREST] Detected: {course} | Score: {get_score(user_id)}{RESET}")
        first_prompt = start_lead_qualification(
            user_id=user_id,
            phone=TEST_PHONE,
            course=course,
        )
        # Also get a RAG answer for the course question
        rag_answer = ask(question=message, user_id=user_id)
        return rag_answer + "\n\n" + first_prompt

    # ── Step 7: Normal RAG ──────────────────────────────────────
    answer = ask(question=message, user_id=user_id)

    # Confusion tracking
    from rag.retriever import NO_CONTEXT_MSG
    if answer.startswith(NO_CONTEXT_MSG[:30]):
        confusion_count = record_confusion(user_id)
        if should_escalate_confusion(user_id):
            print(f"\n   {RED}[CONFUSION ESCALATION] {confusion_count} unanswered queries{RESET}")
            print(f"   {RED}(In production: admin would be notified){RESET}")
            add_internal_flag(user_id, "REPEATED_CONFUSION")
    else:
        reset_confusion(user_id)

    return answer


def main():
    print_banner()

    while True:
        try:
            user_input = input(f"{BOLD}{GREEN}You: {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{CYAN}Goodbye!{RESET}\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print(f"\n{CYAN}Goodbye!{RESET}\n")
            break

        if user_input.lower() == "reset":
            get_memory().clear(TEST_USER_ID)
            reset_score(TEST_USER_ID)
            reset_stage(TEST_USER_ID)
            reset_persona(TEST_USER_ID)
            reset_triggers(TEST_USER_ID)
            print(f"{YELLOW}All state cleared (memory, score, stage, persona, triggers).{RESET}\n")
            continue

        if user_input.lower() == "status":
            print_status(TEST_USER_ID)
            continue

        try:
            response = process_message(user_input)
            print_intelligence_bar(TEST_USER_ID)
            print(f"{BOLD}{BLUE}Bot: {RESET}{response}\n")
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}\n")


if __name__ == "__main__":
    main()
