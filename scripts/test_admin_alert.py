"""
scripts/test_admin_alert.py
───────────────────────────
Standalone diagnostic for ``send_admin_alert()``.

Why this exists
---------------
When AiSensy returns "Invalid Parameter" we need to isolate whether the
failure is:
  (a) **payload-shape** — wrong field, wrong template name, wrong
      language code, missing project-id, etc. — fails for ALL messages
      including a trivial "Test alert" string; OR
  (b) **content** — the message text contains a char Meta rejects
      (newline / tab / 5+ spaces, or an empty string) — fails only on
      complex/multi-line admin summaries.

The static-vs-dynamic two-phase test below collapses that ambiguity in
one run:

    static "Test alert" → OK   AND   multi-line dynamic → OK
        ⇒ everything healthy, ship it.

    static "Test alert" → OK   AND   multi-line dynamic → FAIL
        ⇒ payload is fine; the sanitizer missed a forbidden char.
          Inspect the ``ADMIN_ALERT_SEND | msg=...`` log line for the
          exact bytes Meta rejected.

    static "Test alert" → FAIL
        ⇒ payload itself is wrong. Likely culprits, in order:
            1. AISENSY_ADMIN_ALERT_TEMPLATE name doesn't match the
               approved template (case-sensitive!).
            2. Template was approved under a different language code
               (e.g. "en_US" rather than "en") — the code in the
               payload MUST match the approval exactly.
            3. AISENSY_API_KEY is the Campaign JWT instead of the
               Project API password.
            4. AISENSY_PROJECT_ID is missing or wrong.

Usage
-----
    python scripts/test_admin_alert.py
    python scripts/test_admin_alert.py --phone 919XXXXXXXXX
    python scripts/test_admin_alert.py --message "custom one-liner"
    python scripts/test_admin_alert.py --static-only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import get_settings  # noqa: E402
from rag.whatsapp_sender import send_admin_alert  # noqa: E402

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _configure_logging() -> None:
    """Surface every ADMIN_ALERT_* log line, including DEBUG payload dumps."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    # httpx is noisy at DEBUG — keep it at INFO so our own logs stand out.
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _print_result(label: str, result) -> None:
    colour = GREEN if result.ok else RED
    status = "OK" if result.ok else "FAIL"
    print(
        f"\n{BOLD}{colour}[{status}]{RESET} {label}\n"
        f"   ok={result.ok}  status={result.status}  via={result.via}\n"
        f"   error={result.error or '(none)'}"
    )


async def _run(phone: str, custom_msg: str | None, static_only: bool) -> int:
    s = get_settings()

    print(f"{BOLD}{CYAN}── Config ──{RESET}")
    print(f"   AISENSY_PROJECT_ID         = {s.AISENSY_PROJECT_ID or '(unset)'}")
    print(f"   AISENSY_API_KEY            = {'set' if s.AISENSY_API_KEY else '(unset)'}")
    print(f"   AISENSY_ADMIN_ALERT_TEMPLATE = {s.AISENSY_ADMIN_ALERT_TEMPLATE!r}")
    print(f"   target phone               = {phone}")

    if not (s.AISENSY_API_KEY and s.AISENSY_PROJECT_ID):
        print(f"\n{RED}Missing AISENSY_API_KEY or AISENSY_PROJECT_ID — aborting.{RESET}")
        return 2

    # ── Phase 1: static "Test alert" — isolates payload shape ──────
    print(f"\n{BOLD}{YELLOW}Phase 1: static 'Test alert' (payload-shape probe){RESET}")
    static_result = await send_admin_alert(phone, "Test alert")
    _print_result("static 'Test alert'", static_result)

    if static_only:
        return 0 if static_result.ok else 1

    # ── Phase 2: multi-line dynamic — exercises the sanitizer ─────
    dynamic_msg = custom_msg or (
        "New hot lead\n"
        "Name: Ravi\r\n"
        "Class: Grade 10\n"
        "Notes:\twants ICSE chemistry tuition,     prefers weekends\n"
        "Source: WhatsApp"
    )
    print(f"\n{BOLD}{YELLOW}Phase 2: multi-line dynamic (sanitizer probe){RESET}")
    print(f"   raw message = {dynamic_msg!r}")
    dynamic_result = await send_admin_alert(phone, dynamic_msg)
    _print_result("multi-line dynamic", dynamic_result)

    # ── Diagnosis ──────────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}── Diagnosis ──{RESET}")
    if static_result.ok and dynamic_result.ok:
        print(f"{GREEN}Both sends succeeded — payload + sanitizer healthy.{RESET}")
        return 0
    if static_result.ok and not dynamic_result.ok:
        print(
            f"{YELLOW}Static OK, dynamic FAILED ⇒ message-content issue.{RESET}\n"
            f"   Inspect 'ADMIN_ALERT_SEND | msg=...' above for the exact bytes\n"
            f"   sent. Likely a char the sanitizer missed (zero-width space,\n"
            f"   non-breaking space, control char, or empty-after-strip)."
        )
        return 1
    if not static_result.ok:
        print(
            f"{RED}Static FAILED ⇒ payload-shape / credential issue.{RESET}\n"
            f"   Error: {static_result.error}\n"
            f"   Check, in order:\n"
            f"     1. AISENSY_ADMIN_ALERT_TEMPLATE matches the approved template name exactly (case-sensitive).\n"
            f"     2. The template's approved language code = 'en' (try 'en_US' if it was approved that way).\n"
            f"     3. AISENSY_API_KEY is the Project API password, NOT the Campaign JWT.\n"
            f"     4. AISENSY_PROJECT_ID matches the project that owns the template."
        )
        return 1
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose AiSensy admin-alert template sends.")
    parser.add_argument(
        "--phone",
        default=None,
        help="Admin phone in 91XXXXXXXXXX form. Defaults to ADMIN_WHATSAPP_NUMBER from settings.",
    )
    parser.add_argument(
        "--message",
        default=None,
        help="Override the Phase-2 dynamic message body.",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Run only the Phase-1 static probe.",
    )
    args = parser.parse_args()

    _configure_logging()

    phone = args.phone or get_settings().ADMIN_WHATSAPP_NUMBER
    if not phone:
        print(f"{RED}No phone provided and ADMIN_WHATSAPP_NUMBER not set — aborting.{RESET}")
        return 2

    return asyncio.run(_run(phone, args.message, args.static_only))


if __name__ == "__main__":
    sys.exit(main())
