"""
rag/page_index.py
-----------------
Vector-less document index using word-based chunking + TF-IDF search.

Build:  python scripts/ingest_document.py
Search: search_index(query, k=3)

Index is loaded once into memory via lru_cache — zero I/O on repeated queries.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Optional

INDEX_PATH = Path(__file__).parent.parent / "data" / "page_index.json"

# ── Stopwords (kept minimal — domain terms must NOT be filtered) ──────
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "up", "down", "out", "off", "over", "under",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "both", "each", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "same", "so", "than", "too", "very",
    "just", "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "it", "they", "them", "their", "what", "which", "who", "this",
    "that", "these", "those", "and", "but", "if", "or", "also", "about",
    "us", "its",
}

# ── Keywords that get a score boost when matched ─────────────────────
PRIORITY_KEYWORDS = {
    "fee", "fees", "cost", "price", "amount", "charges", "payment",
    "course", "courses", "batch", "class", "classes", "program", "programme",
    "neet", "jee", "medical", "engineering", "biology", "physics", "chemistry",
    "contact", "phone", "mobile", "address", "location", "office",
    "timing", "timings", "time", "schedule", "hours",
    "admission", "enroll", "enrolment", "join", "registration",
    "duration", "result", "rank", "score", "marks", "percentage",
    "discount", "scholarship", "offer", "free",
    "ark", "arena", "learning",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase, extract alphanumeric tokens, remove stopwords."""
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def _chunk_by_words(
    text: str,
    min_words: int = 300,
    max_words: int = 500,
) -> list[str]:
    """
    Split text into chunks of 300–500 words.
    Tries to break on paragraph/sentence boundaries.
    Short trailing chunks are merged into the previous one.
    """
    # Split on paragraph breaks first, then sentence endings
    sentences = re.split(r"\n{2,}|(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    current_words: list[str] = []

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue

        # If adding this sentence would overflow max, flush first
        if len(current_words) + len(words) > max_words and len(current_words) >= min_words:
            chunks.append(" ".join(current_words))
            current_words = words
        else:
            current_words.extend(words)

    # Handle remainder
    if current_words:
        if chunks and len(current_words) < 80:
            # Too short — merge into previous chunk
            chunks[-1] = chunks[-1] + " " + " ".join(current_words)
        else:
            chunks.append(" ".join(current_words))

    return chunks


# =====================================================================
# Build Index
# =====================================================================

def build_index(
    doc_path: str | Path,
    output_path: str | Path = INDEX_PATH,
) -> int:
    """
    Extract text from doc_path, chunk it, compute TF-IDF weights,
    and save to output_path as JSON.

    Returns the number of chunks written.
    """
    from rag.chunking import extract_text

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Extracting text from: {doc_path}")
    text = extract_text(doc_path)
    print(f"      {len(text):,} characters extracted.")

    print("[2/3] Chunking text (300–500 words per chunk) ...")
    chunks = _chunk_by_words(text)
    print(f"      {len(chunks)} chunks created.")

    print("[3/3] Computing TF-IDF weights and saving index ...")
    df: Counter = Counter()
    chunk_records: list[dict] = []

    for i, chunk in enumerate(chunks):
        tokens = _tokenize(chunk)
        tf = Counter(tokens)
        df.update(set(tf.keys()))  # document frequency (count chunks, not token occurrences)
        chunk_records.append({
            "id": i,
            "text": chunk,
            "tf": dict(tf),
            "token_count": max(len(tokens), 1),
        })

    N = len(chunk_records)
    # BM25-inspired IDF: log((N - df + 0.5) / (df + 0.5) + 1)
    idf: dict[str, float] = {
        word: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
        for word, freq in df.items()
    }

    index = {
        "total_chunks": N,
        "idf": idf,
        "chunks": chunk_records,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

    print(f"      Index saved to {output_path} ({N} chunks).")
    return N


# =====================================================================
# Load Index (singleton — loaded once, stays in memory)
# =====================================================================

@lru_cache(maxsize=1)
def _load_index_cached(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_index(index_path: str | Path = INDEX_PATH) -> Optional[dict]:
    """Return the in-memory index, or None if the index file doesn't exist."""
    p = Path(index_path)
    if not p.exists():
        return None
    return _load_index_cached(str(p))


# =====================================================================
# Search
# =====================================================================

def search_index(
    query: str,
    k: int = 3,
    index_path: str | Path = INDEX_PATH,
) -> list[str]:
    """
    Return the top-k most relevant text chunks for *query*.

    Scoring:
      - TF-IDF (BM25-style IDF) — base relevance
      - Exact phrase match boost (+3.0 per phrase hit)
      - Priority keyword boost (+1.5 per domain keyword matched)

    Chunks with zero score are excluded entirely.
    """
    index = get_index(index_path)
    if index is None:
        return []

    chunks = index["chunks"]
    idf: dict[str, float] = index["idf"]

    query_lower = query.lower()
    query_tokens = _tokenize(query)

    # Also include raw query words (un-stopworded) for exact phrase check
    raw_query_words = re.findall(r"\b[a-zA-Z0-9]+\b", query_lower)

    if not query_tokens and not raw_query_words:
        return []

    scored: list[tuple[float, str]] = []

    for chunk in chunks:
        tf: dict[str, int] = chunk["tf"]
        token_count: int = chunk["token_count"]
        text: str = chunk["text"]
        text_lower = text.lower()

        # ── TF-IDF score ─────────────────────────────────────────
        tfidf = 0.0
        for token in query_tokens:
            if token in tf:
                tf_norm = tf[token] / token_count
                tfidf += tf_norm * idf.get(token, 1.0)

        # ── Exact phrase boost ────────────────────────────────────
        phrase_boost = 0.0
        if len(raw_query_words) >= 2 and query_lower in text_lower:
            phrase_boost = 3.0
        # Partial bigram matches (adjacent word pairs)
        for j in range(len(raw_query_words) - 1):
            bigram = raw_query_words[j] + " " + raw_query_words[j + 1]
            if bigram in text_lower:
                phrase_boost += 1.0

        # ── Priority keyword boost ────────────────────────────────
        priority_boost = 0.0
        for token in query_tokens:
            if token in PRIORITY_KEYWORDS and token in text_lower:
                priority_boost += 1.5

        total = tfidf + phrase_boost + priority_boost

        if total > 0:
            scored.append((total, text))

    # Sort descending, deduplicate, return top k
    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:k]]
