"""
rag/greeting_handler.py
───────────────────────
Detects greetings, acknowledgements, thanks, and farewells — short
"small-talk" messages that should NEVER hit the RAG pipeline (TF-IDF
returns no chunks for "hi", which then surfaces as the unhelpful
"I couldn't find relevant information" fallback).

Returns friendly canned replies in the user's language (English /
Thanglish / Tamil), so the bot behaves like a human assistant.

Detection is deliberately conservative: only short messages
(≤ 4 tokens after stripping punctuation) qualify, so substantive
questions that happen to start with "hi" still flow through RAG.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from rag.psychology_engine import detect_language


class SmallTalk(str, Enum):
    GREETING = "greeting"
    THANKS = "thanks"
    ACK = "ack"
    BYE = "bye"


# Single tokens that, on their own (or with another small-talk token),
# indicate a greeting.
_GREETING_TOKENS = {
    "hi", "hii", "hiii", "hey", "heyy", "heya",
    "hello", "helo", "hlo", "hellow",
    "yo", "hola", "namaste", "namaskar", "namaskaram",
    "vanakkam", "vanakam", "vannakam",
}

_GREETING_PHRASES = (
    "good morning", "good evening", "good afternoon", "good night",
    "gud morning", "gud evening", "gud night",
    "வணக்கம்",  # vanakkam in Tamil script
)

_THANKS_TOKENS = {
    "thanks", "thank", "thx", "thnx", "ty", "tnx",
    "nandri", "நன்றி",
}

_THANKS_PHRASES = (
    "thank you", "thanks a lot", "thanks so much", "many thanks",
    "thank u", "thanku",
)

# Acknowledgement / appreciation tokens. Detected only when EVERY token
# in a short message belongs to this set, so "ok send fees" doesn't
# get misclassified as a pure ack.
_ACK_TOKENS = {
    "ok", "okay", "okk", "okkk", "k", "kk", "kkk",
    "fine", "cool", "nice", "good", "great", "super",
    "perfect", "awesome", "alright", "sure",
    "yes", "ya", "yeah", "yep", "yup",
    "noted", "hmm", "hmmm", "mm",
    "👍", "👌", "🙏", "🙂", "😊",
    "sari", "seri", "sariya", "சரி",
}

_ACK_PHRASES = (
    "got it", "got that", "thats fine", "that's fine",
    "no problem", "no issue", "no worries", "all good",
)

_BYE_TOKENS = {
    "bye", "byee", "byeee", "goodbye", "tata", "cya",
    "ciao", "see you", "later",
}

_BYE_PHRASES = ("see you", "talk later", "good bye")


def _normalize(msg: str) -> str:
    """Lowercase + strip trailing punctuation/emoji noise."""
    return re.sub(r"[!.?,;:]+$", "", msg.strip()).lower()


def detect_small_talk(message: str) -> Optional[SmallTalk]:
    """
    Return the SmallTalk type if the message is a greeting / ack / thanks
    / bye, else None.

    Conservative rules:
      - Message must be ≤ 4 tokens after punctuation strip.
      - For ACK we require ALL tokens to be ack tokens, so combinations
        like "ok send fees" are NOT misclassified.
    """
    if not message:
        return None

    cleaned = _normalize(message)
    if not cleaned:
        return None

    words = cleaned.split()
    if len(words) > 4:
        return None

    # Multi-word phrases first (more specific).
    for phrase in _GREETING_PHRASES:
        if phrase in cleaned:
            return SmallTalk.GREETING
    for phrase in _THANKS_PHRASES:
        if phrase in cleaned:
            return SmallTalk.THANKS
    for phrase in _ACK_PHRASES:
        if phrase in cleaned:
            return SmallTalk.ACK
    for phrase in _BYE_PHRASES:
        if phrase in cleaned:
            return SmallTalk.BYE

    word_set = set(words)

    if word_set & _GREETING_TOKENS:
        return SmallTalk.GREETING
    if word_set & _THANKS_TOKENS:
        return SmallTalk.THANKS
    if word_set & _BYE_TOKENS:
        return SmallTalk.BYE
    if all(w in _ACK_TOKENS for w in words):
        return SmallTalk.ACK

    return None


_REPLIES: dict[SmallTalk, dict[str, str]] = {
    SmallTalk.GREETING: {
        "english": (
            "Hello! \U0001f44b I'm Priya from ARK Learning Arena. "
            "How can I help you today? \U0001f60a"
        ),
        "tamil": (
            "வணக்கம்! \U0001f44b நான் ARK Learning Arena-ல் இருந்து பிரியா. "
            "உங்களுக்கு எப்படி உதவ முடியும்? \U0001f60a"
        ),
        "thanglish": (
            "Vanakkam! \U0001f44b Naan Priya, ARK Learning Arena la irundhu. "
            "Eppadi help pannalam? \U0001f60a"
        ),
    },
    SmallTalk.THANKS: {
        "english": "You're welcome! \U0001f60a Let me know if you'd like more details.",
        "tamil": "பரவாயில்லை! \U0001f60a மேலும் ஏதாவது விவரம் வேண்டுமென்றால் சொல்லுங்கள்.",
        "thanglish": "Welcome! \U0001f60a Innum details venumna sollunga.",
    },
    SmallTalk.ACK: {
        "english": "Glad to hear that! \U0001f60a Let me know if you need anything else.",
        "tamil": "சரி! \U0001f60a மேலும் ஏதாவது தேவைப்பட்டால் சொல்லுங்கள்.",
        "thanglish": "Sari! \U0001f60a Edhuvachu venumna sollunga.",
    },
    SmallTalk.BYE: {
        "english": "Take care! \U0001f64f Reach out anytime — we're here to help.",
        "tamil": "நன்றி! \U0001f64f எப்போது வேண்டுமானாலும் தொடர்பு கொள்ளுங்கள்.",
        "thanglish": "Bye! \U0001f64f Edhuvachu venumna ping pannunga.",
    },
}


def get_small_talk_reply(kind: SmallTalk, message: str) -> str:
    """Return a friendly reply matched to the user's language."""
    lang = detect_language(message)
    return _REPLIES[kind].get(lang, _REPLIES[kind]["english"])
