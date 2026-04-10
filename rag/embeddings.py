"""
rag/embeddings.py
─────────────────
Embedding generation using sentence-transformers.

Uses the lightweight, high-quality model `all-MiniLM-L6-v2`
which produces 384-dimensional vectors.

The model is loaded once at startup (singleton) and reused
for all queries — embedding a question takes ~5–10 ms.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

logger = logging.getLogger("ark.embeddings")

# Model identifier — fixed for consistency with stored vectors
MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load and cache the sentence-transformer model (singleton).

    Called once at server startup via the FastAPI lifespan event
    so the first user query doesn't pay the load cost.

    Returns:
        A SentenceTransformer model instance.
    """
    logger.info("Loading embedding model: %s …", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Embedding model loaded successfully.")
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of text strings.

    Used during document ingestion (not during queries).

    Args:
        texts: List of text chunks to embed.

    Returns:
        A list of embedding vectors (each is a list of 384 floats).
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    return [emb.tolist() for emb in embeddings]


def embed_query(query: str) -> list[float]:
    """
    Generate an embedding for a single query string.

    Optimised for speed — no progress bar, single item.

    Args:
        query: The user's question.

    Returns:
        A 384-dimensional embedding vector as a list of floats.
    """
    model = get_embedding_model()
    embedding = model.encode(query)
    return embedding.tolist()
