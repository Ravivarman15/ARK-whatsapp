"""
rag/embeddings.py
─────────────────
Embedding generation via HuggingFace InferenceClient.

Uses `BAAI/bge-small-en-v1.5` (384-dim) via the huggingface_hub
InferenceClient.  The old raw api-inference.huggingface.co endpoint
is deprecated (410 Gone) — InferenceClient handles routing automatically.

No local model is loaded — keeps RAM under 512 MB on Render.

Both sync and async interfaces are provided so that:
  - The FastAPI async endpoints use `get_embedding()` / `get_embeddings()`
  - CLI scripts (ingest) can call `embed_query()` / `embed_texts()` directly.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

from huggingface_hub import InferenceClient

logger = logging.getLogger("ark.embeddings")

# ── Configuration ─────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0          # seconds — doubles each retry

# ── Shared InferenceClient (singleton) ────────────────────────────────
_client: Optional[InferenceClient] = None


def _get_client() -> InferenceClient:
    """Return a lazily-created, long-lived InferenceClient."""
    global _client
    if _client is None:
        token = os.getenv("HF_API_TOKEN", "")
        if not token:
            # Fallback: try pydantic settings (imported lazily to avoid circular deps)
            try:
                from config.settings import get_settings
                token = get_settings().HF_API_TOKEN
            except Exception:
                pass
        _client = InferenceClient(
            provider="hf-inference",
            api_key=token,
        )
        logger.info(
            "HuggingFace InferenceClient initialised | model=%s",
            EMBEDDING_MODEL,
        )
    return _client


async def close_client() -> None:
    """No-op kept for backward compatibility with lifespan handler."""
    pass


# =====================================================================
# Core sync functions (InferenceClient is synchronous)
# =====================================================================

def _embed_single(text: str) -> list[float]:
    """
    Generate an embedding for a single text string.

    Args:
        text: The text to embed.

    Returns:
        A 384-dimensional embedding vector as a list of floats.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    logger.info("EMBEDDING_REQUEST | single text (%d chars)", len(text))
    client = _get_client()
    last_error: Optional[Exception] = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result = client.feature_extraction(
                text,
                model=EMBEDDING_MODEL,
            )

            # result can be: numpy array, nested list, or flat list
            if hasattr(result, "tolist"):
                embedding = result.tolist()
            elif isinstance(result, list):
                embedding = result
            else:
                embedding = list(result)

            # Flatten if nested [[...]]
            if embedding and isinstance(embedding[0], list):
                embedding = embedding[0]

            logger.info("EMBEDDING_SUCCESS | single | dim=%d", len(embedding))
            return embedding

        except Exception as exc:
            last_error = exc
            logger.warning(
                "EMBEDDING_ERROR | attempt %d/%d: %s",
                attempt, _MAX_RETRIES, exc,
            )
            if attempt < _MAX_RETRIES:
                wait = _RETRY_BACKOFF * (2 ** (attempt - 1))
                logger.info("Retrying in %.1fs …", wait)
                time.sleep(wait)

    raise RuntimeError(
        f"Embedding failed after {_MAX_RETRIES} retries: {last_error}"
    )


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts.

    Processes texts sequentially through the InferenceClient.

    Args:
        texts: List of text chunks to embed.

    Returns:
        A list of 384-dimensional embedding vectors.
    """
    if not texts:
        return []

    logger.info("EMBEDDING_REQUEST | batch of %d texts", len(texts))
    all_embeddings: list[list[float]] = []

    for i, text in enumerate(texts):
        emb = _embed_single(text)
        all_embeddings.append(emb)

        # Progress logging every 10 texts
        if (i + 1) % 10 == 0:
            logger.info(
                "EMBEDDING_PROGRESS | %d/%d texts embedded",
                i + 1, len(texts),
            )

    logger.info(
        "EMBEDDING_SUCCESS | total=%d embeddings (dim=%d)",
        len(all_embeddings),
        len(all_embeddings[0]) if all_embeddings else 0,
    )
    return all_embeddings


# =====================================================================
# Async wrappers (for FastAPI endpoints)
# =====================================================================

async def get_embedding(text: str) -> list[float]:
    """Async wrapper — offloads sync InferenceClient call to a thread."""
    return await asyncio.to_thread(_embed_single, text)


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Async wrapper — offloads sync batch embedding to a thread."""
    return await asyncio.to_thread(_embed_batch, texts)


# =====================================================================
# Sync wrappers (for CLI scripts & the sync retriever path)
# =====================================================================

def embed_query(query: str) -> list[float]:
    """
    Synchronous embedding for a single query.

    Used by the retriever's sync `ask()` function.
    """
    return _embed_single(query)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Synchronous embedding for multiple texts.

    Used by the ingest script.
    """
    return _embed_batch(texts)
