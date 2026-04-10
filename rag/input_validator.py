"""
rag/input_validator.py
──────────────────────
Smart input validation for the lead qualification flow.

Provides two layers of validation:
  1. Intent detection — determines if a message is a question, command, or answer
  2. Field-specific validators — validates name, class, school, and phone inputs

This prevents the bot from storing questions or junk as lead data fields.
"""

from __future__ import annotations

import re
import logging
from typing import Optional

logger = logging.getLogger("ark.validator")


# =====================================================================
# Intent Detection
# =====================================================================

# Words that signal the start of a question
QUESTION_STARTERS = [
    "what", "how", "why", "when", "where", "who", "whom", "whose",
    "which", "is", "are", "can", "could", "do", "does", "did",
    "will", "would", "should", "shall", "may", "might",
    "tell me", "explain", "describe", "show me",
    "kya", "kaise", "kab", "kahan", "kaun", "kitna", "kitne", "kitni",
    "konsa", "konsi", "kyu", "kyun", "kyunki", "bata", "batao",
]

# Regex patterns for question detection
QUESTION_PATTERNS = [
    r"\?",                           # Contains question mark
    r"^(what|how|why|when|where|who|which|is|are|can|do|does|will)\b",
    r"^(tell\s+me|explain|describe)\b",
    r"^(kya|kaise|kab|kahan|kaun|kitna|konsa|bata)\b",
    r"\b(fees?|cost|price|batch|timing|schedule|syllabus|result)\b.*\?",
]

# Command words
COMMAND_WORDS = [
    "hi", "hello", "hey", "help", "stop", "cancel", "quit", "exit",
    "menu", "restart", "start", "reset", "back", "go back",
]


def detect_user_intent(message: str) -> str:
    """
    Classify the user's message intent.

    Returns:
        'question' — the user is asking a question
        'command'  — the user is issuing a bot command
        'answer'   — the user is providing a field answer
    """
    msg = message.strip().lower()

    if not msg:
        return "answer"

    # Check for commands first (exact or near-exact matches)
    if msg in COMMAND_WORDS:
        return "command"

    # Check for question patterns
    for pattern in QUESTION_PATTERNS:
        if re.search(pattern, msg, re.IGNORECASE):
            return "question"

    # Check if starts with a question word
    first_words = msg.split()[:3]  # Check first 3 words
    for word in first_words:
        if word in QUESTION_STARTERS:
            # Only flag as question if message is long enough
            # (a single word like "who" could be a name answer in rare cases)
            if len(msg.split()) > 2:
                return "question"

    return "answer"


# =====================================================================
# Field Validators
# =====================================================================

def validate_name(message: str) -> tuple[bool, str]:
    """
    Validate a student name input.

    Rules:
      - 2–50 characters
      - Alphabetic + spaces + dots only
      - No question marks
      - No question words at the start
      - Not purely numeric

    Returns:
        (is_valid, cleaned_value)
    """
    value = message.strip()

    # Reject empty or too short
    if len(value) < 2:
        return False, value

    # Reject question marks
    if "?" in value:
        return False, value

    # Reject purely numeric
    if value.replace(" ", "").isdigit():
        return False, value

    # Allow only letters, spaces, dots, and hyphens
    cleaned = re.sub(r"[^a-zA-Z\s.\-]", "", value).strip()
    if len(cleaned) < 2:
        return False, value

    # Reject if it starts with a question word
    first_word = cleaned.lower().split()[0] if cleaned.split() else ""
    question_words = {"what", "how", "why", "when", "where", "who", "which",
                      "is", "are", "can", "do", "does", "will", "tell",
                      "kya", "kaise", "kab", "kahan", "kaun"}
    if first_word in question_words and len(cleaned.split()) > 2:
        return False, value

    # Truncate to 50 chars
    if len(cleaned) > 50:
        cleaned = cleaned[:50].strip()

    # Title-case the name
    cleaned = cleaned.title()

    return True, cleaned


def validate_class(message: str) -> tuple[bool, str]:
    """
    Validate a class/grade input.

    Accepts patterns like:
      - "10", "11", "12"
      - "class 10", "Class 11th"
      - "10th", "12th"
      - "IX", "X", "XI", "XII"
      - "9th class", "11 th"

    Returns:
        (is_valid, cleaned_value)
    """
    value = message.strip().lower()

    # Reject question marks
    if "?" in value:
        return False, value

    # Roman numeral mapping
    roman_map = {
        "viii": "8", "ix": "9", "x": "10", "xi": "11", "xii": "12",
        "vii": "7", "vi": "6", "v": "5", "iv": "4",
    }

    # Check roman numerals
    clean = re.sub(r"[^a-z]", "", value)
    if clean in roman_map:
        return True, f"Class {roman_map[clean]}"

    # Extract numeric class
    match = re.search(r"(\d{1,2})", value)
    if match:
        class_num = int(match.group(1))
        if 1 <= class_num <= 12:
            return True, f"Class {class_num}"

    # Check for common word patterns
    word_map = {
        "eight": "8", "ninth": "9", "nine": "9",
        "ten": "10", "tenth": "10",
        "eleven": "11", "eleventh": "11",
        "twelve": "12", "twelfth": "12",
    }
    for word, num in word_map.items():
        if word in value:
            return True, f"Class {num}"

    return False, message.strip()


def validate_school(message: str) -> tuple[bool, str]:
    """
    Validate a school name input.

    Rules:
      - 2–100 characters
      - Not purely numeric
      - No question marks
      - Allows alphanumeric, spaces, and common punctuation (.,-')

    Returns:
        (is_valid, cleaned_value)
    """
    value = message.strip()

    if len(value) < 2:
        return False, value

    if "?" in value:
        return False, value

    # Reject purely numeric
    if value.replace(" ", "").isdigit():
        return False, value

    # Reject if it looks like a question
    first_word = value.lower().split()[0] if value.split() else ""
    question_words = {"what", "how", "why", "when", "where", "who", "which",
                      "is", "are", "can", "do", "does", "will", "tell",
                      "kya", "kaise", "kab", "kahan", "kaun"}
    if first_word in question_words and len(value.split()) > 3:
        return False, value

    # Truncate to 100 chars
    if len(value) > 100:
        value = value[:100].strip()

    return True, value.title()


def validate_phone(message: str) -> tuple[bool, str]:
    """
    Validate an Indian phone number.

    Accepts:
      - "9876543210"
      - "+91 9876543210"
      - "91-9876543210"
      - "098765 43210"

    Returns:
        (is_valid, cleaned_10_digit_number)
    """
    value = message.strip()

    # Extract only digits
    digits = re.sub(r"[^\d]", "", value)

    # Remove country code prefix
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    elif digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]

    # Must be exactly 10 digits
    if len(digits) != 10:
        return False, value

    # Must start with 6-9 (Indian mobile numbers)
    if digits[0] not in "6789":
        return False, value

    return True, digits


# =====================================================================
# Validator Dispatcher
# =====================================================================

# Maps qualification step names to their validators
_STEP_VALIDATORS = {
    "ask_name": validate_name,
    "ask_class": validate_class,
    "ask_school": validate_school,
    "ask_parent_phone": validate_phone,
}

# Friendly error messages when validation fails
_VALIDATION_ERRORS = {
    "ask_name": "Hmm, that doesn't look like a name. Please share the *student's full name* (letters only).",
    "ask_class": "Please enter a valid class, e.g. *10*, *11th*, *Class 12*.",
    "ask_school": "Please share a valid *school name*.",
    "ask_parent_phone": "Please enter a valid *10-digit mobile number*, e.g. 9876543210.",
}


def validate_field(step: str, message: str) -> tuple[bool, str]:
    """
    Validate a message against the field expected for the given step.

    Args:
        step:    The qualification step name (e.g. 'ask_name').
        message: The user's raw message.

    Returns:
        (is_valid, cleaned_value_or_error_message)
        If invalid, the second element is the error/re-prompt message.
    """
    validator = _STEP_VALIDATORS.get(step)
    if validator is None:
        # No validator for this step — accept as-is
        return True, message.strip()

    is_valid, cleaned = validator(message)
    if is_valid:
        return True, cleaned
    else:
        error_msg = _VALIDATION_ERRORS.get(step, "Please provide a valid response.")
        return False, error_msg
