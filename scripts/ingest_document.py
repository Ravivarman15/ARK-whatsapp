"""
scripts/ingest_document.py
--------------------------
CLI script for incremental ingestion of the ARK Learning Arena
Word document into the Supabase vector database.

Usage:
    python scripts/ingest_document.py
    python scripts/ingest_document.py --doc path/to/custom.docx
    python scripts/ingest_document.py --full   (force full re-index)

Incremental Pipeline:
    1. Read the .docx file and chunk the text.
    2. Compute SHA-256 hash for each chunk.
    3. Fetch existing hashes from Supabase.
    4. Diff: identify added / removed / unchanged chunks.
    5. Delete only removed rows.
    6. Insert only new rows (with fresh embeddings).
    7. Log summary.

This avoids full re-indexing and keeps ingestion fast.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from supabase import create_client

from rag.chunking import extract_text, chunk_text_with_hashes
from rag.embeddings import embed_texts

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Default document path
DEFAULT_DOC = os.path.join(
    os.path.dirname(__file__), "..", "documents", "ark_details.docx"
)


def get_supabase():
    """Create a Supabase client for the ingestion run."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_KEY must be set in .env")
        sys.exit(1)
    return create_client(url, key)


# =====================================================================
# Full Ingestion (clear + re-insert everything)
# =====================================================================

def ingest_full(doc_path: str) -> None:
    """Full re-index: extract, chunk, embed, clear DB, insert all."""
    start = time.perf_counter()
    print(f"\n{'='*60}")
    print(f" FULL INGESTION")
    print(f"{'='*60}")

    # Step 1: Extract text
    print(f"\n[1/5] Reading document: {doc_path}")
    text = extract_text(doc_path)
    print(f"      Extracted {len(text):,} characters.")

    # Step 2: Chunk text with hashes
    print("[2/5] Chunking text ...")
    chunks_with_hashes = chunk_text_with_hashes(text, chunk_size=500, chunk_overlap=80)
    print(f"      Created {len(chunks_with_hashes)} chunks.")

    if not chunks_with_hashes:
        print("WARNING: No content found in the document. Aborting.")
        return

    chunks = [c for c, _ in chunks_with_hashes]
    hashes = [h for _, h in chunks_with_hashes]

    # Step 3: Generate embeddings
    print("[3/5] Generating embeddings ...")
    embeddings = embed_texts(chunks)
    print(f"      Generated {len(embeddings)} embeddings (dim={len(embeddings[0])}).")

    # Step 4: Clear existing data
    client = get_supabase()
    print("[4/5] Clearing existing ark_docs rows ...")
    client.table("ark_docs").delete().gt("id", 0).execute()
    print("      Cleared.")

    # Step 5: Insert new data
    print("[5/5] Inserting chunks into Supabase ...")
    rows = [
        {"content": chunk, "embedding": emb, "content_hash": h}
        for chunk, emb, h in zip(chunks, embeddings, hashes)
    ]

    BATCH_SIZE = 50
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        client.table("ark_docs").insert(batch).execute()
        print(f"      Inserted batch {i // BATCH_SIZE + 1} ({len(batch)} rows)")

    elapsed = time.perf_counter() - start
    print(f"\nDone! {len(chunks)} chunks stored in {elapsed:.1f}s.\n")


# =====================================================================
# Incremental Ingestion (hash-based diff)
# =====================================================================

def ingest_incremental(doc_path: str) -> None:
    """
    Smart incremental ingestion:
      - Only embed and insert NEW chunks.
      - Only delete REMOVED chunks.
      - Unchanged chunks are left intact.
    """
    start = time.perf_counter()
    print(f"\n{'='*60}")
    print(f" INCREMENTAL INGESTION")
    print(f"{'='*60}")

    # Step 1: Extract and chunk
    print(f"\n[1/5] Reading document: {doc_path}")
    text = extract_text(doc_path)
    print(f"      Extracted {len(text):,} characters.")

    print("[2/5] Chunking text ...")
    chunks_with_hashes = chunk_text_with_hashes(text, chunk_size=500, chunk_overlap=80)
    print(f"      Created {len(chunks_with_hashes)} chunks.")

    if not chunks_with_hashes:
        print("WARNING: No content found in the document. Aborting.")
        return

    new_hash_map = {h: c for c, h in chunks_with_hashes}  # hash -> content
    new_hashes = set(new_hash_map.keys())

    # Step 2: Fetch existing hashes from Supabase
    print("[3/5] Fetching existing hashes from Supabase ...")
    client = get_supabase()

    # Fetch all existing content_hash values
    existing_response = (
        client.table("ark_docs")
        .select("id, content_hash")
        .execute()
    )
    existing_rows = existing_response.data or []

    existing_hash_map = {}  # hash -> list of ids
    for row in existing_rows:
        h = row.get("content_hash", "")
        if h:
            existing_hash_map.setdefault(h, []).append(row["id"])

    existing_hashes = set(existing_hash_map.keys())

    # Step 3: Compute diff
    added_hashes = new_hashes - existing_hashes
    removed_hashes = existing_hashes - new_hashes
    unchanged_hashes = new_hashes & existing_hashes

    print(f"\n      Diff results:")
    print(f"        Added:     {len(added_hashes)} chunks")
    print(f"        Removed:   {len(removed_hashes)} chunks")
    print(f"        Unchanged: {len(unchanged_hashes)} chunks")

    # Step 4: Delete removed rows
    if removed_hashes:
        print("[4/5] Deleting removed chunks ...")
        ids_to_delete = []
        for h in removed_hashes:
            ids_to_delete.extend(existing_hash_map[h])
        # Delete in batches
        for i in range(0, len(ids_to_delete), 50):
            batch_ids = ids_to_delete[i : i + 50]
            client.table("ark_docs").delete().in_("id", batch_ids).execute()
        print(f"      Deleted {len(ids_to_delete)} rows.")
    else:
        print("[4/5] No chunks to delete.")

    # Step 5: Insert added rows
    if added_hashes:
        print("[5/5] Generating embeddings for new chunks ...")
        added_chunks = [new_hash_map[h] for h in added_hashes]
        added_hashes_list = list(added_hashes)
        embeddings = embed_texts(added_chunks)
        print(f"      Generated {len(embeddings)} new embeddings.")

        rows = [
            {"content": chunk, "embedding": emb, "content_hash": h}
            for chunk, emb, h in zip(added_chunks, embeddings, added_hashes_list)
        ]

        BATCH_SIZE = 50
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            client.table("ark_docs").insert(batch).execute()
            print(f"      Inserted batch {i // BATCH_SIZE + 1} ({len(batch)} rows)")
    else:
        print("[5/5] No new chunks to add.")

    elapsed = time.perf_counter() - start
    print(f"\nDone! Incremental update completed in {elapsed:.1f}s.")
    print(f"Total chunks in database: ~{len(unchanged_hashes) + len(added_hashes)}\n")


# =====================================================================
# CLI Entry Point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ingest an ARK Learning Arena .docx into Supabase vector DB."
    )
    parser.add_argument(
        "--doc", default=DEFAULT_DOC,
        help="Path to the .docx file (default: documents/ark_details.docx)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Force full re-indexing (clears all existing data first).",
    )
    args = parser.parse_args()

    if args.full:
        ingest_full(args.doc)
    else:
        ingest_incremental(args.doc)


if __name__ == "__main__":
    main()
