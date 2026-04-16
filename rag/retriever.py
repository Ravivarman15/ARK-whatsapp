"""
rag/retriever.py
----------------
Production RAG pipeline with:

  - Configurable LLM (env var LLM_MODEL)
  - Supabase pgvector similarity search (top_k = 3)
  - Question cache (in-memory / Redis)
  - Per-user conversation memory (last N turns)
  - Enhanced AI personality (Academic Advisor)
  - Stage-aware prompting
  - Persona-aware prompting
  - Psychological trigger injection
  - ADA conversion injection
  - Tamil language detection + switching
  - Structured performance logging per request
  - HuggingFace InferenceClient (handles new router.huggingface.co)

Latency target: 2-3 seconds end-to-end.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from huggingface_hub import InferenceClient
from supabase import create_client, Client

from config.settings import get_settings
from rag.embeddings import embed_query
from rag.cache import get_cache

logger = logging.getLogger("ark.retriever")


# =====================================================================
# Supabase Client (singleton)
# =====================================================================

@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Initialise and cache the Supabase client."""
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
        _hf_client = InferenceClient(
            api_key=s.HF_API_TOKEN,
        )
        logger.info("HuggingFace InferenceClient initialised")
    return _hf_client


# =====================================================================
# Conversation Memory
# =====================================================================

@dataclass
class ConversationMemory:
    """
    Per-user sliding-window conversation memory.
    Stores the last N question/answer turns so the LLM can resolve
    follow-up questions.
    """
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
# Vector Similarity Search
# =====================================================================

def search_similar(query_embedding: list[float], top_k: int = 3) -> list[str]:
    """Cosine-similarity search via the Supabase match_ark_docs RPC."""
    client = get_supabase_client()
    response = client.rpc(
        "match_ark_docs",
        {"query_embedding": query_embedding, "match_count": top_k},
    ).execute()
    results = response.data or []
    return [row["content"] for row in results]


# =====================================================================
# System Instruction (Enhanced Bot Personality)
# =====================================================================

SYSTEM_INSTRUCTION = (
    "You are the Academic Advisor for ARK Learning Arena, "
    "a structured academic performance institute (NOT regular tuition) "
    "based in Chennai, established in 2015 with 500+ students trained.\n\n"
    "RESPONSE FORMAT (STRICT):\n"
    "- You are responding on WhatsApp\n"
    "- Keep ALL responses to 2-3 lines MAX\n"
    "- Use bullet points for lists\n"
    "- NO paragraphs, NO long explanations\n"
    "- Simple English, conversational tone\n"
    "- End with a relevant question only if needed\n\n"
    "PERSONALITY:\n"
    "- Confident and knowledgeable\n"
    "- Friendly like a human counsellor\n"
    "- Concise — say more with less\n"
    "- NEVER salesy or pushy\n\n"
    "RULES:\n"
    "- Use ONLY the context below to answer factual questions\n"
    "- If the answer is not in the context, say it's unavailable "
    "and suggest contacting ARK directly\n"
    "- Never guarantee results\n"
    "- Never discuss specific fee amounts; redirect to counsellor\n"
    "- Do NOT add 'book assessment' or promotional lines unless asked\n"
)

# =====================================================================
# Enhanced Message Builder
# =====================================================================


def _build_messages(
    question: str,
    context_chunks: list[str],
    history: Optional[list[dict]] = None,
    stage_instruction: str = "",
    persona_instruction: str = "",
    trigger_line: str = "",
    is_tamil: bool = False,
) -> list[dict]:
    """
    Build chat messages for the HuggingFace chat_completion API.

    Returns a list of {role, content} dicts:
      - system: grounding instruction + stage/persona modifiers
      - user/assistant: past conversation turns (if any)
      - user: current context + question
    """
    # Build enhanced system message
    sys_parts = [SYSTEM_INSTRUCTION]

    if stage_instruction:
        sys_parts.append(f"\nCURRENT USER STAGE:\n{stage_instruction}")

    if persona_instruction:
        sys_parts.append(f"\nPARENT TYPE:\n{persona_instruction}")

    if is_tamil:
        sys_parts.append(
            "\nLANGUAGE: The user is communicating in Tamil. "
            "Respond in Tamil (transliterated or Unicode) while maintaining "
            "the same authoritative and professional tone."
        )

    if trigger_line:
        sys_parts.append(
            f"\nINCLUDE THIS LINE naturally in your response:\n\"{trigger_line}\""
        )

    system_content = "\n".join(sys_parts)
    messages = [{"role": "system", "content": system_content}]

    # Add conversation history as alternating user/assistant turns
    if history:
        for turn in history:
            messages.append({"role": "user", "content": turn["q"]})
            messages.append({"role": "assistant", "content": turn["a"]})

    # Current question with context
    context_block = "\n\n".join(context_chunks)
    user_content = f"CONTEXT:\n{context_block}\n\nQUESTION: {question}"
    messages.append({"role": "user", "content": user_content})

    return messages


# =====================================================================
# LLM Answer Generation (via huggingface_hub InferenceClient)
# =====================================================================

def generate_answer(question: str, context_chunks: list[str],
                    history: Optional[list[dict]] = None,
                    stage_instruction: str = "",
                    persona_instruction: str = "",
                    trigger_line: str = "",
                    is_tamil: bool = False) -> str:
    """
    Send context + question to the configured LLM via HuggingFace
    InferenceClient chat_completion API (synchronous).
    """
    s = get_settings()
    client = get_hf_client()
    messages = _build_messages(
        question, context_chunks, history,
        stage_instruction=stage_instruction,
        persona_instruction=persona_instruction,
        trigger_line=trigger_line,
        is_tamil=is_tamil,
    )

    try:
        response = client.chat_completion(
            model=s.LLM_MODEL,
            messages=messages,
            max_tokens=s.MAX_NEW_TOKENS,
            temperature=0.3,
            top_p=0.9,
        )

        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                return content.strip()

        return "Sorry, I was unable to generate a response. Please try again."

    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        raise RuntimeError(f"HuggingFace API error: {e}")


async def generate_answer_async(question: str, context_chunks: list[str],
                                 history: Optional[list[dict]] = None,
                                 stage_instruction: str = "",
                                 persona_instruction: str = "",
                                 trigger_line: str = "",
                                 is_tamil: bool = False) -> str:
    """
    Async wrapper — calls the sync InferenceClient.
    HuggingFace InferenceClient handles HTTP internally.
    For true async, use AsyncInferenceClient (requires additional setup).
    """
    return generate_answer(
        question, context_chunks, history,
        stage_instruction=stage_instruction,
        persona_instruction=persona_instruction,
        trigger_line=trigger_line,
        is_tamil=is_tamil,
    )


# =====================================================================
# High-Level RAG Pipeline
# =====================================================================

NO_CONTEXT_MSG = (
    "I\'m sorry, I couldn\'t find relevant information in our knowledge base. "
    "Please contact ARK Learning Arena directly for assistance."
)


async def ask_async(
    question: str,
    user_id: str = "anonymous",
    top_k: int = 3,
) -> str:
    """
    Full async RAG pipeline with caching, memory, intelligence layering,
    and performance logging.

    Steps:
      1. Check cache -> return if hit
      2. Detect intelligence signals (stage, persona, language, triggers)
      3. Embed question
      4. Vector search in Supabase
      5. Generate LLM answer with enhanced context
      6. Inject ADA recommendation (if appropriate)
      7. Store in cache and memory
      8. Log all timings
    """
    from rag.stage_detector import detect_and_update_stage, get_stage_instruction
    from rag.persona_detector import detect_and_update_persona, get_persona_instruction
    from rag.psychology_engine import (
        should_inject_trigger, get_next_trigger,
        should_inject_ada, get_ada_line,
        detect_tamil,
    )
    from rag.scoring import score_from_message

    s = get_settings()
    cache = get_cache(redis_url=s.REDIS_URL, ttl=s.CACHE_TTL)
    memory = get_memory()
    total_start = time.perf_counter()

    # Step 1: Cache check
    cached = cache.get(question)
    if cached:
        total_time = (time.perf_counter() - total_start) * 1000
        logger.info(
            "PERF | cache_hit=True | total_ms=%.1f",
            total_time,
        )
        return cached

    # Step 2: Intelligence signal detection
    stage = detect_and_update_stage(user_id, question)
    stage_inst = get_stage_instruction(stage)

    persona = detect_and_update_persona(user_id, question)
    persona_inst = get_persona_instruction(persona)

    is_tamil = detect_tamil(question)

    # Score from message content
    score_from_message(user_id, question)

    # Trigger rotation
    trigger_line = ""
    if should_inject_trigger(user_id, frequency=s.TRIGGER_ROTATION_FREQUENCY):
        trigger_line = get_next_trigger(user_id)

    # Step 3: Embed question
    t0 = time.perf_counter()
    query_embedding = embed_query(question)
    embedding_time = (time.perf_counter() - t0) * 1000

    # Step 4: Vector search
    t0 = time.perf_counter()
    context_chunks = search_similar(query_embedding, top_k=top_k)
    search_time = (time.perf_counter() - t0) * 1000

    if not context_chunks:
        return NO_CONTEXT_MSG

    # Step 5: Generate answer with intelligence context
    history = memory.get_history(user_id)
    t0 = time.perf_counter()
    answer = await generate_answer_async(
        question, context_chunks, history,
        stage_instruction=stage_inst,
        persona_instruction=persona_inst,
        trigger_line=trigger_line,
        is_tamil=is_tamil,
    )
    llm_time = (time.perf_counter() - t0) * 1000

    # Step 6: ADA injection — DISABLED (now controlled by intent router)
    # Auto-injection removed to prevent random "book assessment" prompts.
    # ADA is only shown after explicit admission intent.

    total_time = (time.perf_counter() - total_start) * 1000

    # Step 7: Cache and memory
    cache.set(question, answer)
    memory.add_turn(user_id, question, answer)

    # Step 8: Log performance
    logger.info(
        "PERF | user=%s | stage=%s | persona=%s | tamil=%s | "
        "embedding_ms=%.1f | search_ms=%.1f | llm_ms=%.1f | total_ms=%.1f",
        user_id, stage.value, persona.value, is_tamil,
        embedding_time, search_time, llm_time, total_time,
    )

    return answer


def ask(question: str, user_id: str = "anonymous", top_k: int = 3) -> str:
    """
    Synchronous RAG pipeline (for scripts and testing).
    Same logic as ask_async but uses sync calls.
    """
    from rag.stage_detector import detect_and_update_stage, get_stage_instruction
    from rag.persona_detector import detect_and_update_persona, get_persona_instruction
    from rag.psychology_engine import (
        should_inject_trigger, get_next_trigger,
        should_inject_ada, get_ada_line,
        detect_tamil,
    )
    from rag.scoring import score_from_message

    s = get_settings()
    cache = get_cache(redis_url=s.REDIS_URL, ttl=s.CACHE_TTL)
    memory = get_memory()
    total_start = time.perf_counter()

    cached = cache.get(question)
    if cached:
        total_time = (time.perf_counter() - total_start) * 1000
        logger.info("PERF | cache_hit=True | total_ms=%.1f", total_time)
        return cached

    # Intelligence signals
    stage = detect_and_update_stage(user_id, question)
    stage_inst = get_stage_instruction(stage)

    persona = detect_and_update_persona(user_id, question)
    persona_inst = get_persona_instruction(persona)

    is_tamil = detect_tamil(question)
    score_from_message(user_id, question)

    trigger_line = ""
    if should_inject_trigger(user_id, frequency=s.TRIGGER_ROTATION_FREQUENCY):
        trigger_line = get_next_trigger(user_id)

    t0 = time.perf_counter()
    query_embedding = embed_query(question)
    embedding_time = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    context_chunks = search_similar(query_embedding, top_k=top_k)
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
        is_tamil=is_tamil,
    )
    llm_time = (time.perf_counter() - t0) * 1000

    # ADA injection — DISABLED (now controlled by intent router)

    total_time = (time.perf_counter() - total_start) * 1000

    cache.set(question, answer)
    memory.add_turn(user_id, question, answer)

    logger.info(
        "PERF | user=%s | stage=%s | persona=%s | tamil=%s | "
        "embedding_ms=%.1f | search_ms=%.1f | llm_ms=%.1f | total_ms=%.1f",
        user_id, stage.value, persona.value, is_tamil,
        embedding_time, search_time, llm_time, total_time,
    )

    return answer
