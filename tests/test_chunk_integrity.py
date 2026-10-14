"""
test_chunk_integrity.py
───────────────────────
Verifies that all chunks in the ChromaDB index are well-formed:
  - No empty chunks
  - No broken table chunks (table with no data rows)
  - No oversized text chunks
  - No chunks missing page metadata
  - No chunks outside the configured page range
  - No duplicate chunks

Run from myChatbot/:
    python tests/test_chunk_integrity.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.embedder import load_vector_store
from collections import Counter

START      = int(os.getenv("PDF_START_PAGE", 430))
END        = int(os.getenv("PDF_END_PAGE", 557))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 600))


def load_all_chunks():
    store = load_vector_store()
    all_chunks = store.get()
    total = len(all_chunks["ids"])
    print(f"Total chunks in index: {total}\n")
    return all_chunks, total


def test_no_empty_chunks(all_chunks):
    """No chunk should have empty page_content."""
    empty = [
        all_chunks["ids"][i]
        for i, content in enumerate(all_chunks["documents"])
        if not content.strip()
    ]
    if empty:
        print(f"❌ Empty chunks found: {len(empty)}")
        for id_ in empty[:3]:
            print(f"   ID: {id_}")
    else:
        print(f"✅ No empty chunks.")
    return len(empty)


def test_no_broken_tables(all_chunks):
    """Table chunks must have at least one data row."""
    broken = []
    for content, metadata in zip(all_chunks["documents"], all_chunks["metadatas"]):
        if metadata.get("type") != "table":
            continue
        lines     = [l.strip() for l in content.split("\n") if l.strip()]
        # Data rows contain | but are not separator rows (all dashes/pipes/spaces)
        data_rows = [
            l for l in lines
            if "|" in l and not all(c in "|-: " for c in l)
        ]
        if len(data_rows) < 2:   # header + at least 1 data row
            broken.append(metadata.get("page", "?"))

    if broken:
        print(f"❌ Broken table chunks: {len(broken)} (pages: {sorted(set(broken))[:10]})")
    else:
        print(f"✅ All table chunks have data rows.")
    return len(broken)


def test_no_oversized_text_chunks(all_chunks):
    """Text chunks should not exceed 2x chunk_size."""
    max_allowed = CHUNK_SIZE * 2
    oversized   = []
    for content, metadata in zip(all_chunks["documents"], all_chunks["metadatas"]):
        if metadata.get("type") == "text" and len(content) > max_allowed:
            oversized.append((metadata.get("page", "?"), len(content)))

    if oversized:
        print(f"⚠️  Oversized text chunks: {len(oversized)} "
              f"(>{max_allowed} chars). Sample pages: {[p for p, _ in oversized[:5]]}")
    else:
        print(f"✅ All text chunks within size limit ({max_allowed} chars).")
    return len(oversized)


def test_no_missing_page_metadata(all_chunks):
    """Every chunk must have a page number."""
    missing = [
        all_chunks["ids"][i]
        for i, metadata in enumerate(all_chunks["metadatas"])
        if not metadata.get("page")
    ]
    if missing:
        print(f"❌ Chunks missing page metadata: {len(missing)}")
    else:
        print(f"✅ All chunks have page metadata.")
    return len(missing)


def test_page_range_compliance(all_chunks):
    """All chunks must be within the configured page range."""
    out_of_range = [
        (all_chunks["ids"][i], metadata.get("page"))
        for i, metadata in enumerate(all_chunks["metadatas"])
        if metadata.get("page") and not (START <= metadata["page"] <= END)
    ]
    if out_of_range:
        pages = sorted(set(p for _, p in out_of_range))
        print(f"❌ Chunks outside page range {START}–{END}: {len(out_of_range)} chunks "
              f"on pages {pages[:10]}")
    else:
        print(f"✅ All chunks within page range {START}–{END}.")
    return len(out_of_range)


def test_no_duplicate_chunks(all_chunks):
    """No two chunks should have identical content."""
    keys = [content[:150].strip() for content in all_chunks["documents"]]
    duplicates = len(keys) - len(set(keys))
    if duplicates:
        print(f"⚠️  Duplicate chunks detected: {duplicates} near-identical chunks.")
        print("   Consider deleting the index and rebuilding cleanly.")
    else:
        print(f"✅ No duplicate chunks.")
    return duplicates


def print_type_breakdown(all_chunks):
    """Print chunk count by type."""
    counts = Counter(m.get("type", "unknown") for m in all_chunks["metadatas"])
    print(f"\n── Chunk type breakdown ──")
    for t, count in sorted(counts.items()):
        print(f"   type={t:10} : {count}")


def print_integrity_score(total, empty, broken, missing_page, out_of_range):
    """Print overall integrity score."""
    issues = empty + broken + missing_page + out_of_range
    score  = (total - issues) / total if total else 0
    print(f"\n── Integrity Score: {total - issues}/{total} = {score:.0%} ──")
    if score < 0.95:
        print("⚠️  Score below 95% — consider rebuilding the index.")
    else:
        print("✅ Index integrity is healthy.")


if __name__ == "__main__":
    print(f"=== Chunk Integrity Tests (pages {START}–{END}) ===\n")

    all_chunks, total = load_all_chunks()

    empty        = test_no_empty_chunks(all_chunks)
    broken       = test_no_broken_tables(all_chunks)
    oversized    = test_no_oversized_text_chunks(all_chunks)
    missing_page = test_no_missing_page_metadata(all_chunks)
    out_of_range = test_page_range_compliance(all_chunks)
    duplicates   = test_no_duplicate_chunks(all_chunks)

    print_type_breakdown(all_chunks)
    print_integrity_score(total, empty, broken, missing_page, out_of_range)