"""
test_ingestion.py
─────────────────
Tests for the PDF ingestion pipeline.
Verifies chunk count, metadata integrity, and table extraction.

Run from myChatbot/:
    python tests/test_ingestion.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.pdf_loader import load_split

PDF_PATH = os.getenv("PDF_PATH", "./data/raw/nbc2020.pdf")
START    = int(os.getenv("PDF_START_PAGE", 430))
END      = int(os.getenv("PDF_END_PAGE", 557))


def test_chunks_produced():
    """At least one chunk must be produced from the PDF."""
    docs = load_split(PDF_PATH, start_page=START, end_page=END)
    assert len(docs) > 0, "No chunks produced — check PDF path and page range."
    print(f"✅ Chunks produced: {len(docs)}")
    return docs


def test_chunk_metadata(docs):
    """Every chunk must have source, page, and type metadata."""
    missing_source = [i for i, d in enumerate(docs) if not d.metadata.get("source")]
    missing_page   = [i for i, d in enumerate(docs) if not d.metadata.get("page")]
    missing_type   = [i for i, d in enumerate(docs) if not d.metadata.get("type")]

    assert not missing_source, f"Chunks missing 'source': indices {missing_source[:5]}"
    assert not missing_page,   f"Chunks missing 'page': indices {missing_page[:5]}"
    assert not missing_type,   f"Chunks missing 'type': indices {missing_type[:5]}"
    print(f"✅ All chunks have source, page, and type metadata.")


def test_page_range(docs):
    """All chunks must be within the configured page range."""
    out_of_range = [
        d.metadata.get("page") for d in docs
        if d.metadata.get("page") and not (START <= d.metadata["page"] <= END)
    ]
    assert not out_of_range, f"Chunks outside page range {START}–{END}: pages {out_of_range[:5]}"
    print(f"✅ All chunks within page range {START}–{END}.")


def test_chunk_content(docs):
    """No chunk should have empty page_content."""
    empty = [i for i, d in enumerate(docs) if not d.page_content.strip()]
    assert not empty, f"Empty chunks at indices: {empty[:5]}"
    print(f"✅ No empty chunks.")


def test_table_chunks_exist(docs):
    """At least some table chunks should be produced."""
    table_chunks = [d for d in docs if d.metadata.get("type") == "table"]
    assert len(table_chunks) > 0, (
        "No table chunks found. Check _extract_borderless_tables() in pdf_loader.py."
    )
    print(f"✅ Table chunks found: {len(table_chunks)}")


def test_text_chunk_size(docs):
    """Text chunks should not significantly exceed CHUNK_SIZE."""
    chunk_size  = int(os.getenv("CHUNK_SIZE", 600))
    max_allowed = chunk_size * 2   # allow some tolerance
    text_chunks = [d for d in docs if d.metadata.get("type") == "text"]
    oversized   = [d for d in text_chunks if len(d.page_content) > max_allowed]

    if oversized:
        print(f"⚠️  {len(oversized)} oversized text chunks (>{max_allowed} chars). "
              f"Sample page: {oversized[0].metadata.get('page')}")
    else:
        print(f"✅ All text chunks within size limits.")


def test_type_breakdown(docs):
    """Print a summary of chunk types."""
    from collections import Counter
    counts = Counter(d.metadata.get("type", "unknown") for d in docs)
    print(f"\n── Chunk type breakdown ──")
    for t, count in counts.items():
        print(f"   type={t:10} : {count}")


if __name__ == "__main__":
    print(f"=== Ingestion Tests (pages {START}–{END}) ===\n")

    docs = test_chunks_produced()
    test_chunk_metadata(docs)
    test_page_range(docs)
    test_chunk_content(docs)
    test_table_chunks_exist(docs)
    test_text_chunk_size(docs)
    test_type_breakdown(docs)

    print("\n=== All ingestion tests passed ===")