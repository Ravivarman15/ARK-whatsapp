"""
rag/retriever.py
----------------
Production RAG pipeline with:

  - Configurable LLM (env var LLM_MODEL)
  - Vector-less page index search (TF-IDF + keyword scoring)
  - Question cache (in-memory / Redis)
  - Per-user conversation memory (last N turns)
  - Enhanced AI personality (Academic Advisor)
  - Stage-aware prompting
  - Persona-aware prompting
  - Psychological trigger injection
  - Tamil language detection + switching
  - Structured performance logging per request
  - HuggingFace InferenceClient

Latency target: <2 seconds end-to-end (no embedding network call).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from huggingface_hub import InferenceClient
from supabase import create_client, Client

from config.settings import get_settings
from rag.page_index import search_index, all_chunks
from rag.cache import get_cache

logger = logging.getLogger("ark.retriever")


# =====================================================================
# Supabase Client (singleton — used by lead_manager / followup_manager)
# =====================================================================

@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Initialise and cache the Supabase client (for leads/followups)."""
    s = get_settings()
    return create_client(s.SUPABASE_URL, s.SUPABASE_KEY)


# =====================================================================
# HuggingFace InferenceClient (singleton)
# =====================================================================

_hf_client: Optional[InferenceClient] = None


def get_hf_client() -> InferenceClient:
    """Create and cache a HuggingFace InferenceClient."""
    global _hf_client
    if _hf_client is None:
        s = get_settings()
        _hf_client = InferenceClient(api_key=s.HF_API_TOKEN)
        logger.info("HuggingFace InferenceClient initialised")
    return _hf_client


# =====================================================================
# Conversation Memory
# =====================================================================

@dataclass
class ConversationMemory:
    """Per-user sliding-window conversation memory (last N turns)."""
    max_turns: int = 3
    _store: dict = field(default_factory=lambda: defaultdict(list))

    def add_turn(self, user_id: str, question: str, answer: str) -> None:
        self._store[user_id].append({"q": question, "a": answer})
        if len(self._store[user_id]) > self.max_turns:
            self._store[user_id] = self._store[user_id][-self.max_turns:]

    def get_history(self, user_id: str) -> list[dict]:
        return list(self._store.get(user_id, []))

    def clear(self, user_id: str) -> None:
        self._store.pop(user_id, None)


_memory: Optional[ConversationMemory] = None


def get_memory() -> ConversationMemory:
    global _memory
    if _memory is None:
        _memory = ConversationMemory(max_turns=get_settings().MEMORY_MAX_TURNS)
    return _memory


# =====================================================================
# System Instruction (Enhanced Bot Personality)
# =====================================================================

SYSTEM_INSTRUCTION = (
    "You are Ms. Priya, the friendly Academic Counsellor for ARK Learning Arena — "
    "a results-focused coaching institute in Chennai, established in 2015, with "
    "500+ students mentored. You chat on WhatsApp with parents and students.\n\n"
    "HOW YOU SOUND:\n"
    "- Warm, caring, like a senior sister/counsellor who genuinely wants to help\n"
    "- Confident but never salesy or pushy\n"
    "- Conversational English, simple words, no jargon\n"
    "- Use 1 emoji max per reply (😊 🙌 ✨ 📘 🎯 🔥) — only when it adds warmth\n\n"
    "HOW YOU FORMAT (WhatsApp rules — MUST follow):\n"
    "- Short lines, no walls of text\n"
    "- Open with a small hook / acknowledgement (1 line)\n"
    "- Bullets for facts (use • not dashes)\n"
    "- For general questions → 2-3 crisp bullets\n"
    "- For overview questions (\"tell me about ARK\", \"who are you\", \"about institute\") "
    "→ list ALL the key points from context (intro line + every course/feature as a bullet)\n"
    "- For list questions (\"what courses\", \"what programs\", \"what classes\", \"what features\") "
    "→ list EVERY item from context — do NOT stop at 2-3\n"
    "- End with ONE soft, curious question that invites the next reply\n"
    "- Never include '— Team ARK' signatures\n\n"
    "HOW YOU ANSWER:\n"
    "- Use ONLY the CONTEXT below for facts — never invent details\n"
    "- When the context lists items (courses, programs, features), include EVERY item — never truncate a list\n"
    "- If the context does not cover it, say: \"Let me get our counsellor to share "
    "the exact details with you\" and gently ask for their phone/class\n"
    "- Never quote specific fee amounts — say \"our counsellor can walk you through "
    "the fee structure based on the class\" and offer a callback\n"
    "- Never guarantee marks/ranks; talk about ARK's proven system instead\n\n"
    "EXAMPLE REPLY (copy this vibe, not the content):\n"
    "\"Great question! 😊\n"
    "Here's what makes ARK different:\n"
    "• Structured weekly testing + performance tracking\n"
    "• Small batch sizes so every student gets attention\n"
    "• Mentored 500+ students since 2015\n"
    "Which class is your child in? I can share what fits best.\"\n"
)


# =====================================================================
# Message Builder
# =====================================================================

_LANG_DIRECTIVE = {
    "english": (
        "\nLANGUAGE LOCK: The user wrote in English. "
        "You MUST reply ONLY in English. "
        "Do NOT use Tamil script (Unicode) or Tamil transliteration words "
        "(evlo, enna, nalla, irukku, vanga, etc.). Pure English only."
    ),
    "thanglish": (
        "\nLANGUAGE LOCK: The user wrote in Thanglish (Tamil words in English letters). "
        "You MUST reply ONLY in Thanglish — Tamil words written in English letters. "
        "Do NOT use Tamil Unicode script. Do NOT reply in pure English. "
        "Example style: \"Vanga! Namma ARK la NEET batch iruku, fees details counsellor share pannuvanga.\""
    ),
    "tamil": (
        "\nLANGUAGE LOCK: The user wrote in Tamil (Unicode script). "
        "You MUST reply ONLY in Tamil Unicode script (தமிழ்). "
        "Do NOT reply in English or Thanglish (romanized Tamil)."
    ),
}


def _build_messages(
    question: str,
    context_chunks: list[str],
    history: Optional[list[dict]] = None,
    stage_instruction: str = "",
    persona_instruction: str = "",
    trigger_line: str = "",
    language: str = "english",
) -> list[dict]:
    sys_parts = [SYSTEM_INSTRUCTION]

    if stage_instruction:
        sys_parts.append(f"\nCURRENT USER STAGE:\n{stage_instruction}")

    if persona_instruction:
        sys_parts.append(f"\nPARENT TYPE:\n{persona_instruction}")

    sys_parts.append(_LANG_DIRECTIVE.get(language, _LANG_DIRECTIVE["english"]))

    if trigger_line:
        sys_parts.append(
            f"\nINCLUDE THIS LINE naturally in your response:\n\"{trigger_line}\""
        )

    messages = [{"role": "system", "content": "\n".join(sys_parts)}]

    if history:
        for turn in history:
            messages.append({"role": "user", "content": turn["q"]})
            messages.append({"role": "assistant", "content": turn["a"]})

    context_block = "\n\n".join(context_chunks)
    messages.append({"role": "user", "content": f"CONTEXT:\n{context_block}\n\nQUESTION: {question}"})
    return messages


# =====================================================================
# LLM Answer Generation
# =====================================================================

_HF_MAX_ATTEMPTS = 3
_HF_RETRY_BACKOFF = 0.7  # seconds; doubles each attempt


def generate_answer(
    question: str,
    context_chunks: list[str],
    history: Optional[list[dict]] = None,
    stage_instruction: str = "",
    persona_instruction: str = "",
    trigger_line: str = "",
    language: str = "english",
) -> str:
    s = get_settings()
    client = get_hf_client()
    messages = _build_messages(
        question, context_chunks, history,
        stage_instruction=stage_instruction,
        persona_instruction=persona_instruction,
        trigger_line=trigger_line,
        language=language,
    )

    last_err: Optional[Exception] = None
    for attempt in range(1, _HF_MAX_ATTEMPTS + 1):
        try:
            response = client.chat_completion(
                model=s.LLM_MODEL,
                messages=messages,
                max_tokens=s.MAX_NEW_TOKENS,
                temperature=0.3,
                top_p=0.9,
            )
            if response.choices:
                content = response.choices[0].message.content
                if content:
                    return content.strip()
            return "Sorry, I was unable to generate a response. Please try again."
        except Exception as e:
            last_err = e
            logger.warning(
                "LLM attempt %d/%d failed: %s: %s",
                attempt, _HF_MAX_ATTEMPTS, type(e).__name__, e,
            )
            if attempt < _HF_MAX_ATTEMPTS:
                time.sleep(_HF_RETRY_BACKOFF * attempt)

    logger.error("LLM generation failed after %d attempts: %s", _HF_MAX_ATTEMPTS, last_err)
    raise RuntimeError(f"HuggingFace API error: {last_err}")


async def generate_answer_async(
    question: str,
    context_chunks: list[str],
    history: Optional[list[dict]] = None,
    stage_instruction: str = "",
    persona_instruction: str = "",
    trigger_line: str = "",
    language: str = "english",
) -> str:
    # Run the blocking HF call in a worker thread so we don't freeze
    # the event loop — critical for concurrent webhook throughput.
    return await asyncio.to_thread(
        generate_answer,
        question, context_chunks, history,
        stage_instruction,
        persona_instruction,
        trigger_line,
        language,
    )


# =====================================================================
# High-Level RAG Pipeline
# =====================================================================

NO_CONTEXT_MSG = (
    "I'm sorry, I couldn't find relevant information in our knowledge base. "
    "Please contact ARK Learning Arena directly for assistance."
)


# Questions asking for a broad overview or a full list need more context than
# a 3-chunk TF-IDF window — bump top_k so every course/feature reaches the LLM.
_BROAD_QUERY_MARKERS = (
    "tell me about", "about ark", "about the institute", "about your institute",
    "about you", "who are you", "who r you", "introduce",
    "what courses", "which courses", "list courses", "all courses",
    "what programs", "what programmes", "list programs",
    "what classes", "what batches", "all batches",
    "what features", "what do you offer", "what do you provide",
    "courses you provide", "courses are you providing",
    "courses do you provide", "courses do you offer",
    "services", "facilities",
)


def _is_broad_query(question: str) -> bool:
    q = question.lower()
    return any(marker in q for marker in _BROAD_QUERY_MARKERS)


def _retrieve_context(question: str, top_k: int, language: str) -> list[str]:
    """
    TF-IDF retrieval with two fallbacks:
      1. Broad-query boost: list/overview questions use a larger k so the
         full course catalogue reaches the LLM instead of being cut to 3.
      2. Tamil/Thanglish empty-result fallback: the TF-IDF index is
         English-tokenised, so Tamil Unicode queries score 0 against every
         chunk. When retrieval comes back empty for a non-English query,
         hand the first few chunks to the LLM so it has *something* to
         answer from (language lock still enforces Tamil/Thanglish reply).
    """
    k = max(top_k, 6) if _is_broad_query(question) else top_k
    chunks = search_index(question, k=k)

    if not chunks and language in ("tamil", "thanglish"):
        chunks = all_chunks(limit=max(k, 5))
        if chunks:
            logger.info(
                "RETRIEVAL_FALLBACK | lang=%s | used all_chunks (n=%d) — TF-IDF empty",
                language, len(chunks),
            )

    return chunks


async def ask_async(
    question: str,
    user_id: str = "anonymous",
    top_k: int = 3,
) -> str:
    """
    Full async RAG pipeline using page index search (no embeddings/vector DB).

    Steps:
      1. Check cache
      2. Detect intelligence signals (stage, persona, language, triggers)
      3. Search page index (TF-IDF keyword search)
      4. Generate LLM answer
      5. Cache + memory
    """
    from rag.stage_detector import detect_and_update_stage, get_stage_instruction
    from rag.persona_detector import detect_and_update_persona, get_persona_instruction
    from rag.psychology_engine import (
        should_inject_trigger, get_next_trigger, detect_language,
    )
    from rag.scoring import score_from_message

    s = get_settings()
    cache = get_cache(redis_url=s.REDIS_URL, ttl=s.CACHE_TTL)
    memory = get_memory()
    total_start = time.perf_counter()

    # Step 1: Cache check
    cached = cache.get(question)
    if cached:
        logger.info("PERF | cache_hit=True | total_ms=%.1f", (time.perf_counter() - total_start) * 1000)
        return cached

    # Step 2: Intelligence signal detection
    stage = detect_and_update_stage(user_id, question)
    stage_inst = get_stage_instruction(stage)

    persona = detect_and_update_persona(user_id, question)
    persona_inst = get_persona_instruction(persona)

    language = detect_language(question)
    score_from_message(user_id, question)

    trigger_line = ""
    if should_inject_trigger(user_id, frequency=s.TRIGGER_ROTATION_FREQUENCY):
        trigger_line = get_next_trigger(user_id)

    # Step 3: Page index search (fast, in-memory)
    t0 = time.perf_counter()
    context_chunks = _retrieve_context(question, top_k=top_k, language=language)
    search_time = (time.perf_counter() - t0) * 1000

    if not context_chunks:
        return NO_CONTEXT_MSG

    # Step 4: Generate answer (hard wall-clock timeout so one slow HF
    # request never stalls the webhook — we prefer a graceful fallback
    # over making the user wait minutes)
    history = memory.get_history(user_id)
    t0 = time.perf_counter()
    try:
        answer = await asyncio.wait_for(
            generate_answer_async(
                question, context_chunks, history,
                stage_instruction=stage_inst,
                persona_instruction=persona_inst,
                trigger_line=trigger_line,
                language=language,
            ),
            timeout=s.HF_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error("LLM_TIMEOUT | user=%s | timeout=%ds", user_id, s.HF_TIMEOUT)
        return (
            "Thanks for reaching out! Our team will get back to you shortly \U0001f64f"
        )
    except RuntimeError as e:
        logger.error("LLM_FAILED | user=%s | %s", user_id, e)
        return (
            "Thanks for reaching out! Our team will get back to you shortly \U0001f64f"
        )
    llm_time = (time.perf_counter() - t0) * 1000

    total_time = (time.perf_counter() - total_start) * 1000

    # Step 5: Cache and memory
    cache.set(question, answer)
    memory.add_turn(user_id, question, answer)

    logger.info(
        "PERF | user=%s | stage=%s | persona=%s | lang=%s | "
        "search_ms=%.1f | llm_ms=%.1f | total_ms=%.1f",
        user_id, stage.value, persona.value, language,
        search_time, llm_time, total_time,
    )

    return answer


def ask(question: str, user_id: str = "anonymous", top_k: int = 3) -> str:
    """Synchronous RAG pipeline (for scripts and testing)."""
    from rag.stage_detector import detect_and_update_stage, get_stage_instruction
    from rag.persona_detector import detect_and_update_persona, get_persona_instruction
    from rag.psychology_engine import (
        should_inject_trigger, get_next_trigger, detect_language,
    )
    from rag.scoring import score_from_message

    s = get_settings()
    cache = get_cache(redis_url=s.REDIS_URL, ttl=s.CACHE_TTL)
    memory = get_memory()
    total_start = time.perf_counter()

    cached = cache.get(question)
    if cached:
        logger.info("PERF | cache_hit=True | total_ms=%.1f", (time.perf_counter() - total_start) * 1000)
        return cached

    stage = detect_and_update_stage(user_id, question)
    stage_inst = get_stage_instruction(stage)

    persona = detect_and_update_persona(user_id, question)
    persona_inst = get_persona_instruction(persona)

    language = detect_language(question)
    score_from_message(user_id, question)

    trigger_line = ""
    if should_inject_trigger(user_id, frequency=s.TRIGGER_ROTATION_FREQUENCY):
        trigger_line = get_next_trigger(user_id)

    t0 = time.perf_counter()
    context_chunks = _retrieve_context(question, top_k=top_k, language=language)
    search_time = (time.perf_counter() - t0) * 1000

    if not context_chunks:
        return NO_CONTEXT_MSG

    history = memory.get_history(user_id)
    t0 = time.perf_counter()
    answer = generate_answer(
        question, context_chunks, history,
        stage_instruction=stage_inst,
        persona_instruction=persona_inst,
        trigger_line=trigger_line,
        language=language,
    )
    llm_time = (time.perf_counter() - t0) * 1000

    total_time = (time.perf_counter() - total_start) * 1000
    cache.set(question, answer)
    memory.add_turn(user_id, question, answer)

    logger.info(
        "PERF | user=%s | stage=%s | persona=%s | lang=%s | "
        "search_ms=%.1f | llm_ms=%.1f | total_ms=%.1f",
        user_id, stage.value, persona.value, language,
        search_time, llm_time, total_time,
    )

    return answer
