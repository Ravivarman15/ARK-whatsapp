"""
scripts/ingest_document.py
--------------------------
Build the ARK Learning Arena page index from a Word document.

No embeddings, no vector DB — produces a local JSON index used by
the TF-IDF search in rag/page_index.py.

Usage:
    python scripts/ingest_document.py
    python scripts/ingest_document.py --doc path/to/custom.docx
    python scripts/ingest_document.py --out data/custom_index.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from rag.page_index import build_index, INDEX_PATH

DEFAULT_DOC = os.path.join(
    os.path.dirname(__file__), "..", "documents", "ark_details.docx"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the ARK page index (TF-IDF) from a .docx file."
    )
    parser.add_argument(
        "--doc", default=DEFAULT_DOC,
        help="Path to the .docx source document.",
    )
    parser.add_argument(
        "--out", default=str(INDEX_PATH),
        help="Output path for the JSON index file.",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(" ARK PAGE INDEX BUILDER")
    print(f"{'='*60}\n")

    start = time.perf_counter()
    n = build_index(doc_path=args.doc, output_path=args.out)
    elapsed = time.perf_counter() - start

    print(f"\nDone! {n} chunks indexed in {elapsed:.2f}s.")
    print("Run the server — searches will use this index automatically.\n")


if __name__ == "__main__":
    main()
