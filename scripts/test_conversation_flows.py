"""
scripts/test_conversation_flows.py
───────────────────────────────────
Test script to verify the chatbot conversation flow fixes.

Tests:
  1. Intent routing priority (factual > admission > general)
  2. Qualification state machine (start, resume, complete)
  3. Mid-flow question interruption
  4. Repeat intent handling
  5. Response formatting
"""

import sys
import os

# Add parent dir to path so we can import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_intent_router():
    """Test the unified intent router classifies correctly."""
    from rag.intent_router import classify_message, Route

    print("=" * 60)
    print("TEST: Intent Router Priority")
    print("=" * 60)

    cases = [
        # (message, is_in_qual, has_complaint, has_human_req, expected_route)
        ("I'm not happy with the service", False, True, False, Route.COMPLAINT),
        ("I want to speak to someone", False, False, True, Route.HUMAN_ESCALATION),
        ("Who is the founder?", False, False, False, Route.FACTUAL_QUESTION),
        ("What are the timings?", False, False, False, Route.FACTUAL_QUESTION),
        ("Where is ARK located?", False, False, False, Route.FACTUAL_QUESTION),
        ("I want to join NEET", False, False, False, Route.ADMISSION_INTENT),
        ("Tell me about fees", False, False, False, Route.ADMISSION_INTENT),
        ("I want admission", False, False, False, Route.ADMISSION_INTENT),
        ("Hello", False, False, False, Route.GENERAL),
        ("Good morning", False, False, False, Route.GENERAL),
        # Mid-qualification factual question
        # Mid-qualification: fees question should still route to admission (which resumes)
        ("What are the fees?", True, False, False, Route.ADMISSION_INTENT),
        # Mid-qualification answer (not a question)
        ("Rahul", True, False, False, Route.QUALIFICATION),
        ("Class 10", True, False, False, Route.QUALIFICATION),
    ]

    passed = 0
    failed = 0
    for msg, in_qual, complaint, human_req, expected in cases:
        result = classify_message(
            msg,
            is_in_qualification=in_qual,
            has_complaint=complaint,
            has_human_request=human_req,
        )
        status = "✅" if result == expected else "❌"
        if result != expected:
            failed += 1
            print(f"  {status} \"{msg}\" → {result.value} (expected: {expected.value})")
        else:
            passed += 1
            print(f"  {status} \"{msg}\" → {result.value}")

    print(f"\n  Results: {passed} passed, {failed} failed\n")
    return failed == 0


def test_factual_question_detection():
    """Test that factual questions are detected correctly."""
    from rag.intent_engine import is_factual_question

    print("=" * 60)
    print("TEST: Factual Question Detection")
    print("=" * 60)

    cases = [
        # (message, expected)
        ("Who is the founder?", True),
        ("What is ARK?", True),
        ("Where is the location?", True),
        ("How does the system work?", True),
        ("When was it established?", True),
        # These should NOT be factual (contain admission keywords)
        ("What are the fees?", False),  # contains "fees" → admission intent
        ("How to join NEET?", False),  # contains "join"
        ("Tell me about the course", False),  # contains "course"
        # These are not questions at all
        ("I want to join", False),
        ("Hello", False),
        ("Rahul", False),
    ]

    passed = 0
    failed = 0
    for msg, expected in cases:
        result = is_factual_question(msg)
        status = "✅" if result == expected else "❌"
        if result != expected:
            failed += 1
            print(f"  {status} \"{msg}\" → {result} (expected: {expected})")
        else:
            passed += 1
            print(f"  {status} \"{msg}\" → {result}")

    print(f"\n  Results: {passed} passed, {failed} failed\n")
    return failed == 0


def test_qualification_flow():
    """Test the qualification state machine."""
    from rag.lead_manager import (
        start_lead_qualification, is_in_qualification,
        process_qualification_message, complete_lead,
        _active_leads,
    )

    print("=" * 60)
    print("TEST: Qualification Flow (State Machine)")
    print("=" * 60)

    test_user = "test_qual_user_001"
    # Clean up any previous state
    _active_leads.pop(test_user, None)

    # Step 1: Start qualification
    prompt = start_lead_qualification(test_user, phone="9999999999", course="NEET")
    assert is_in_qualification(test_user), "Should be in qualification after start"
    print(f"  ✅ Started qualification → \"{prompt}\"")

    # Step 2: Provide name
    result = process_qualification_message(test_user, "Rahul Kumar")
    assert result is not None, "Name should be accepted"
    print(f"  ✅ Name accepted → \"{result}\"")

    # Step 3: Provide class
    result = process_qualification_message(test_user, "Class 11")
    assert result is not None, "Class should be accepted"
    print(f"  ✅ Class accepted → \"{result}\"")

    # Step 4: Ask a question mid-flow (should return None → caller runs RAG)
    result = process_qualification_message(test_user, "What are the fees?")
    assert result is None, "Question should interrupt flow (return None)"
    print(f"  ✅ Question mid-flow → returns None (caller runs RAG)")

    # Step 5: Still on school step
    assert is_in_qualification(test_user), "Should still be in qual after question"
    print(f"  ✅ Still in qualification after question")

    # Step 6: Provide school
    result = process_qualification_message(test_user, "DAV School")
    assert result is not None, "School should be accepted"
    print(f"  ✅ School accepted → \"{result}\"")

    # Step 7: Provide phone
    result = process_qualification_message(test_user, "9876543210")
    assert result is not None, "Phone should be accepted"
    print(f"  ✅ Phone accepted → \"{result}\"")

    # Step 8: Complete lead
    lead = complete_lead(test_user)
    assert lead is not None, "Lead should be completed"
    assert lead.student_name == "Rahul Kumar", f"Name should be 'Rahul Kumar', got '{lead.student_name}'"
    print(f"  ✅ Lead completed: name={lead.student_name}, class={lead.student_class}, school={lead.school}, phone={lead.parent_phone}")

    # Clean up
    _active_leads.pop(test_user, None)
    print()
    return True


def test_repeat_intent():
    """Test that repeating admission intent resumes flow, not restarts."""
    from rag.lead_manager import (
        start_lead_qualification, is_in_qualification,
        process_qualification_message, _active_leads,
    )

    print("=" * 60)
    print("TEST: Repeat Intent Handling")
    print("=" * 60)

    test_user = "test_repeat_user_001"
    _active_leads.pop(test_user, None)

    # Start qualification
    prompt1 = start_lead_qualification(test_user, phone="9999999999", course="NEET")
    print(f"  ✅ First start → \"{prompt1}\"")

    # Give name
    process_qualification_message(test_user, "Rahul")

    # Now repeat intent — should resume from CLASS step, not restart at NAME
    prompt2 = start_lead_qualification(test_user, phone="9999999999", course="NEET")
    assert "class" in prompt2.lower() or "which class" in prompt2.lower(), \
        f"Should resume at class step, got: '{prompt2}'"
    print(f"  ✅ Repeat intent → resumed at: \"{prompt2}\"")

    # Clean up
    _active_leads.pop(test_user, None)
    print()
    return True


def test_response_formatter():
    """Test the WhatsApp response formatter."""
    from rag.response_formatter import format_whatsapp_response

    print("=" * 60)
    print("TEST: Response Formatter")
    print("=" * 60)

    # Test 1: Long response gets truncated
    long_text = "\n".join([f"Line {i}: This is some content about ARK." for i in range(20)])
    formatted = format_whatsapp_response(long_text, max_lines=6)
    line_count = len(formatted.strip().split("\n"))
    assert line_count <= 6, f"Should be ≤6 lines, got {line_count}"
    print(f"  ✅ Long text truncated: {line_count} lines (max 6)")

    # Test 2: ADA block stripped from factual responses
    text_with_ada = "ARK was founded in 2015.\n\n📋 *We recommend starting with our free Academic Diagnostic Assessment (ADA).*\nWould you like to book a slot?"
    formatted = format_whatsapp_response(text_with_ada, is_factual=True)
    assert "diagnostic" not in formatted.lower() or "assessment" not in formatted.lower(), \
        "ADA should be stripped from factual responses"
    print(f"  ✅ ADA stripped from factual: \"{formatted[:80]}\"")

    # Test 3: Signature stripped
    text_with_sig = "We offer NEET coaching.\n\n— Team ARK Learning Arena"
    formatted = format_whatsapp_response(text_with_sig, is_factual=True)
    assert "Team ARK" not in formatted, "Signature should be stripped"
    print(f"  ✅ Signature stripped: \"{formatted[:80]}\"")

    # Test 4: Headers stripped
    text_with_headers = "## About ARK\nARK is a coaching institute."
    formatted = format_whatsapp_response(text_with_headers)
    assert "##" not in formatted, "Headers should be stripped"
    print(f"  ✅ Headers stripped: \"{formatted[:80]}\"")

    print()
    return True


def test_course_detection_tightened():
    """Test that course detection requires admission context."""
    from rag.lead_manager import detect_course_interest

    print("=" * 60)
    print("TEST: Course Detection (Tightened)")
    print("=" * 60)

    cases = [
        # Should trigger (has admission context + course keyword)
        ("I want to join NEET", "NEET"),
        ("Tell me about NEET fees", "NEET"),
        ("I'm interested in foundation course", "Foundation"),
        # Should NOT trigger (factual question, no admission context)
        ("What is NEET exam?", None),
        ("Tell me about the school", None),
        ("Which class is best?", None),
    ]

    passed = 0
    failed = 0
    for msg, expected in cases:
        result = detect_course_interest(msg)
        status = "✅" if result == expected else "❌"
        if result != expected:
            failed += 1
            print(f"  {status} \"{msg}\" → {result} (expected: {expected})")
        else:
            passed += 1
            print(f"  {status} \"{msg}\" → {result}")

    print(f"\n  Results: {passed} passed, {failed} failed\n")
    return failed == 0


def main():
    """Run all tests."""
    print("\n🧪 ARK AI Bot — Conversation Flow Tests\n")

    results = {
        "Intent Router": test_intent_router(),
        "Factual Question Detection": test_factual_question_detection(),
        "Qualification Flow": test_qualification_flow(),
        "Repeat Intent": test_repeat_intent(),
        "Response Formatter": test_response_formatter(),
        "Course Detection": test_course_detection_tightened(),
    }

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} | {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed — review output above.")
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
