"""
scripts/test_webhook.py
───────────────────────
Test script to simulate AiSensy webhook payloads against the local
FastAPI /whatsapp endpoint.

Usage:
    1. Start the server:  uvicorn api.main:app --reload
    2. Run this script:   python scripts/test_webhook.py

Options:
    --url       Base URL of the server (default: http://127.0.0.1:8000)
    --phone     Phone number to simulate (default: 917305801869)
    --message   Custom message to send (default: "Hello")
    --all       Run all built-in test cases
"""

from __future__ import annotations

import argparse
import json
import sys
import httpx

# ── ANSI Colors ──────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def build_aisensy_payload(
    message: str,
    phone: str = "917305801869",
    user_name: str = "Test User",
    message_type: str = "TEXT",
) -> dict:
    """Build a realistic AiSensy webhook payload matching the production format."""
    return {
        "id": "test_webhook_001",
        "created_at": "2026-04-10T06:28:16.591Z",
        "topic": "message.sender.user",
        "project_id": "69bb9e72352c2d552437c709",
        "delivery_attempt": "1",
        "data": {
            "message": {
                "type": "message",
                "id": "test_msg_001",
                "meta_data": [],
                "project_id": "69bb9e72352c2d552437c709",
                "phone_number": phone,
                "contact_id": "test_contact_001",
                "campaign": None,
                "sender": "USER",
                "message_content": {
                    "text": message,
                },
                "message_type": message_type,
                "status": "DELIVERED",
                "is_HSM": False,
                "chatbot_response": None,
                "agent_id": None,
                "sent_at": 1775802494000,
                "delivered_at": 1775802494000,
                "read_at": None,
                "userName": user_name,
                "countryCode": "91",
                "submitted_message_id": "",
                "message_price": 0,
                "deductionType": None,
                "mau_details": None,
                "whatsapp_conversation_details": None,
                "context": None,
                "messageId": "wamid.TEST123",
            }
        },
        "maxRetry": 15,
        "timeout": 10,
        "namespace": "ProjectWebhook",
        "groupId": "69bb9e72352c2d552437c709",
        "uniqueId": "test_webhook_001",
    }


# ── Built-in Test Cases ──────────────────────────────────────────────

TEST_CASES = [
    {
        "name": "Normal greeting",
        "message": "Hello",
        "message_type": "TEXT",
        "expect_status": "ok",
    },
    {
        "name": "Course enquiry (triggers qualification)",
        "message": "I want to know about JEE coaching",
        "message_type": "TEXT",
        "expect_status": None,  # could be qualification_started or ok
    },
    {
        "name": "Empty message",
        "message": "",
        "message_type": "TEXT",
        "expect_status": "ignored",
    },
    {
        "name": "Image message (should be ignored)",
        "message": "photo.jpg",
        "message_type": "IMAGE",
        "expect_status": "ignored",
    },
    {
        "name": "Fee enquiry",
        "message": "What are the fees for NEET coaching?",
        "message_type": "TEXT",
        "expect_status": None,
    },
    {
        "name": "Human escalation",
        "message": "I want to talk to a human counsellor",
        "message_type": "TEXT",
        "expect_status": "escalated",
    },
]


def send_webhook(base_url: str, payload: dict, test_name: str = "") -> dict | None:
    """Send a webhook payload and return the JSON response."""
    url = f"{base_url.rstrip('/')}/whatsapp"

    label = f" [{test_name}]" if test_name else ""
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}📨 Sending webhook{label}{RESET}")
    print(f"{DIM}   URL: {url}{RESET}")

    msg_text = payload.get("data", {}).get("message", {}).get("message_content", {}).get("text", "")
    msg_type = payload.get("data", {}).get("message", {}).get("message_type", "TEXT")
    print(f"{DIM}   Message: \"{msg_text}\" (type: {msg_type}){RESET}")

    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, json=payload)
            data = resp.json()

            status = data.get("status", "unknown")
            answer = data.get("answer", "")

            if status == "ok":
                color = GREEN
                icon = "✅"
            elif status == "ignored":
                color = YELLOW
                icon = "⏭️"
            elif status == "error":
                color = RED
                icon = "❌"
            else:
                color = CYAN
                icon = "📋"

            print(f"\n{color}{icon} Status: {status}{RESET}")
            if answer:
                # Truncate long answers for display
                display = answer[:300] + "..." if len(answer) > 300 else answer
                print(f"{color}   Answer: {display}{RESET}")
            if data.get("reason"):
                print(f"{YELLOW}   Reason: {data['reason']}{RESET}")
            if data.get("detail"):
                print(f"{RED}   Detail: {data['detail']}{RESET}")

            print(f"{DIM}   HTTP {resp.status_code}{RESET}")
            return data

    except httpx.ConnectError:
        print(f"\n{RED}❌ Connection failed — is the server running at {base_url}?{RESET}")
        print(f"{DIM}   Start it with: uvicorn api.main:app --reload{RESET}")
        return None
    except Exception as e:
        print(f"\n{RED}❌ Error: {e}{RESET}")
        return None


def run_single(base_url: str, message: str, phone: str):
    """Send a single custom message."""
    payload = build_aisensy_payload(message=message, phone=phone)
    send_webhook(base_url, payload, test_name="Custom message")


def run_all_tests(base_url: str, phone: str):
    """Run all built-in test cases."""
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}   AiSensy Webhook Test Suite{RESET}")
    print(f"{CYAN}   Running {len(TEST_CASES)} test cases{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")

    passed = 0
    failed = 0

    for i, tc in enumerate(TEST_CASES, 1):
        payload = build_aisensy_payload(
            message=tc["message"],
            phone=phone,
            message_type=tc["message_type"],
        )
        result = send_webhook(base_url, payload, test_name=f"{i}/{len(TEST_CASES)} {tc['name']}")

        if result is None:
            failed += 1
            continue

        # Check expected status if specified
        if tc["expect_status"] is not None:
            if result.get("status") == tc["expect_status"]:
                print(f"{GREEN}   ✓ Expected status '{tc['expect_status']}' — PASS{RESET}")
                passed += 1
            else:
                print(f"{RED}   ✗ Expected '{tc['expect_status']}', got '{result.get('status')}' — FAIL{RESET}")
                failed += 1
        else:
            print(f"{DIM}   ○ No expected status — skipped assertion{RESET}")
            passed += 1

    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}   Results: {GREEN}{passed} passed{RESET}, {RED}{failed} failed{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="Test AiSensy webhook endpoint")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of the FastAPI server")
    parser.add_argument("--phone", default="917305801869", help="Phone number to simulate")
    parser.add_argument("--message", default="Hello", help="Message to send")
    parser.add_argument("--all", action="store_true", help="Run all built-in test cases")

    args = parser.parse_args()

    # Quick health check
    print(f"{DIM}Checking server health at {args.url}...{RESET}")
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{args.url}/health")
            if resp.status_code == 200:
                print(f"{GREEN}✅ Server is healthy: {resp.json()}{RESET}")
            else:
                print(f"{YELLOW}⚠️ Server responded with {resp.status_code}{RESET}")
    except httpx.ConnectError:
        print(f"{RED}❌ Cannot connect to server at {args.url}{RESET}")
        print(f"{DIM}   Start it with: uvicorn api.main:app --reload{RESET}")
        sys.exit(1)

    if args.all:
        run_all_tests(args.url, args.phone)
    else:
        run_single(args.url, args.message, args.phone)


if __name__ == "__main__":
    main()
