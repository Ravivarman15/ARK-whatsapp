"""
rag/embeddings.py
─────────────────
Embedding generation via HuggingFace Inference API.

Uses the same `all-MiniLM-L6-v2` model as before, but calls the
HuggingFace feature-extraction pipeline remotely instead of loading
the model into memory.  This keeps RAM usage under 512 MB on Render.

Both sync and async interfaces are provided so that:
  - The FastAPI async endpoints use `get_embedding()` / `get_embeddings()`
  - CLI scripts (ingest) can call `embed_query()` / `embed_texts()` via
    `asyncio.run()` wrappers.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger("ark.embeddings")

# ── Configuration ─────────────────────────────────────────────────────
HF_API_URL = (
    "https://api-inference.huggingface.co/pipeline/feature-extraction/"
    "sentence-transformers/all-MiniLM-L6-v2"
)
_MAX_RETRIES = 3
_RETRY_BACKOFF = 1.5          # seconds — doubles each retry
_REQUEST_TIMEOUT = 30.0       # seconds per request
_BATCH_SIZE = 64              # texts per API call (HF limit ≈ 128)

# ── Shared httpx AsyncClient (connection-pooled, reused) ──────────────
_client: Optional[httpx.AsyncClient] = None


def _get_headers() -> dict[str, str]:
    """Build authorisation headers from env."""
    token = os.getenv("HF_API_TOKEN", "")
    if not token:
        # Fallback: try pydantic settings (imported lazily to avoid circular deps)
        try:
            from config.settings import get_settings
            token = get_settings().HF_API_TOKEN
        except Exception:
            pass
    return {"Authorization": f"Bearer {token}"}


async def _get_client() -> httpx.AsyncClient:
    """Return a lazily-created, long-lived async HTTP client."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(_REQUEST_TIMEOUT),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
    return _client


async def close_client() -> None:
    """Gracefully close the shared httpx client (call at shutdown)."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None


# =====================================================================
# Core async functions
# =====================================================================

async def get_embedding(text: str) -> list[float]:
    """
    Generate an embedding for a single text string.

    Args:
        text: The text to embed.

    Returns:
        A 384-dimensional embedding vector as a list of floats.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    logger.debug("EMBEDDING_REQUEST | single text (%d chars)", len(text))
    client = await _get_client()
    headers = _get_headers()
    last_error: Optional[Exception] = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = await client.post(
                HF_API_URL,
                headers=headers,
                json={"inputs": text},
            )
            response.raise_for_status()
            data = response.json()

            # The API returns [[float, …]] for a single input
            embedding: list[float] = data[0] if isinstance(data[0], list) else data
            logger.info("EMBEDDING_SUCCESS | single | dim=%d", len(embedding))
            return embedding

        except httpx.TimeoutException as exc:
            last_error = exc
            logger.warning(
                "EMBEDDING_ERROR | timeout on attempt %d/%d: %s",
                attempt, _MAX_RETRIES, exc,
            )
        except httpx.HTTPStatusError as exc:
            last_error = exc
            status = exc.response.status_code
            # 503 = model loading, 429 = rate limit → retry
            if status in (429, 503):
                logger.warning(
                    "EMBEDDING_ERROR | HTTP %d on attempt %d/%d — retrying",
                    status, attempt, _MAX_RETRIES,
                )
            else:
                logger.error("EMBEDDING_ERROR | HTTP %d — not retryable", status)
                raise RuntimeError(f"HuggingFace API error {status}: {exc}") from exc
        except Exception as exc:
            last_error = exc
            logger.error(
                "EMBEDDING_ERROR | unexpected error on attempt %d/%d: %s",
                attempt, _MAX_RETRIES, exc,
            )

        if attempt < _MAX_RETRIES:
            wait = _RETRY_BACKOFF * (2 ** (attempt - 1))
            logger.info("Retrying in %.1fs …", wait)
            await asyncio.sleep(wait)

    raise RuntimeError(
        f"Embedding failed after {_MAX_RETRIES} retries: {last_error}"
    )


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts.

    Sends texts in batches via the HF API.  Falls back to sequential
    calls if the batch request fails.

    Args:
        texts: List of text chunks to embed.

    Returns:
        A list of 384-dimensional embedding vectors.
    """
    if not texts:
        return []

    logger.info("EMBEDDING_REQUEST | batch of %d texts", len(texts))
    client = await _get_client()
    headers = _get_headers()
    all_embeddings: list[list[float]] = []

    for batch_start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[batch_start : batch_start + _BATCH_SIZE]
        last_error: Optional[Exception] = None
        success = False

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = await client.post(
                    HF_API_URL,
                    headers=headers,
                    json={"inputs": batch},
                )
                response.raise_for_status()
                data = response.json()

                # data is [[float, …], [float, …], …]
                all_embeddings.extend(data)
                logger.info(
                    "EMBEDDING_SUCCESS | batch %d–%d of %d",
                    batch_start + 1,
                    min(batch_start + _BATCH_SIZE, len(texts)),
                    len(texts),
                )
                success = True
                break

            except httpx.TimeoutException as exc:
                last_error = exc
                logger.warning(
                    "EMBEDDING_ERROR | batch timeout attempt %d/%d",
                    attempt, _MAX_RETRIES,
                )
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status = exc.response.status_code
                if status in (429, 503):
                    logger.warning(
                        "EMBEDDING_ERROR | batch HTTP %d attempt %d/%d",
                        status, attempt, _MAX_RETRIES,
                    )
                else:
                    logger.error("EMBEDDING_ERROR | batch HTTP %d — aborting", status)
                    raise RuntimeError(
                        f"HuggingFace API error {status}: {exc}"
                    ) from exc
            except Exception as exc:
                last_error = exc
                logger.error(
                    "EMBEDDING_ERROR | batch unexpected error attempt %d/%d: %s",
                    attempt, _MAX_RETRIES, exc,
                )

            if attempt < _MAX_RETRIES:
                wait = _RETRY_BACKOFF * (2 ** (attempt - 1))
                await asyncio.sleep(wait)

        if not success:
            # Fallback: try one-by-one for this batch
            logger.warning(
                "Batch request failed — falling back to sequential for %d texts",
                len(batch),
            )
            for text in batch:
                emb = await get_embedding(text)
                all_embeddings.append(emb)

    logger.info(
        "EMBEDDING_SUCCESS | total=%d embeddings (dim=%d)",
        len(all_embeddings),
        len(all_embeddings[0]) if all_embeddings else 0,
    )
    return all_embeddings


# =====================================================================
# Sync wrappers (for CLI scripts & the sync retriever path)
# =====================================================================

def embed_query(query: str) -> list[float]:
    """
    Synchronous wrapper around `get_embedding()`.

    Used by the retriever's sync `ask()` function.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an already-running event loop (e.g. FastAPI).
        # Create a new thread to run the coroutine.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, get_embedding(query)).result()
    else:
        return asyncio.run(get_embedding(query))


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Synchronous wrapper around `get_embeddings()`.

    Used by the ingest script.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, get_embeddings(texts)).result()
    else:
        return asyncio.run(get_embeddings(texts))
