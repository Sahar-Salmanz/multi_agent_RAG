"""
test_retrieval.py
─────────────────
Basic retrieval smoke tests.
Verifies that the vector store loads, returns results,
and that both text and table chunks are retrievable.

Run from myChatbot/:
    python tests/test_retrieval.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retrieval.retriever import retrieve, format_context
from src.ingestion.embedder import load_vector_store

SAMPLE_QUERY = "What is the live load for residential areas?"


def test_vector_store_loads():
    """Vector store must load without error."""
    store = load_vector_store()
    assert store is not None, "Vector store failed to load."
    print("✅ Vector store loaded.")
    return store


def test_index_not_empty(store):
    """Index must contain chunks."""
    result = store.get()
    total  = len(result["ids"])
    assert total > 0, "Vector store is empty — run embedder.py first."
    print(f"✅ Index contains {total} chunks.")


def test_retrieve_returns_results():
    """retrieve() must return at least one chunk."""
    docs = retrieve(SAMPLE_QUERY)
    assert len(docs) > 0, f"retrieve() returned no results for: '{SAMPLE_QUERY}'"
    print(f"✅ retrieve() returned {len(docs)} chunks.")
    return docs


def test_retrieved_docs_have_metadata(docs):
    """Every retrieved doc must have page and type metadata."""
    for i, doc in enumerate(docs):
        assert doc.metadata.get("page"), f"Doc {i} missing 'page' metadata."
        assert doc.metadata.get("type"), f"Doc {i} missing 'type' metadata."
    print("✅ All retrieved docs have page and type metadata.")


def test_retrieved_docs_have_content(docs):
    """Every retrieved doc must have non-empty page_content."""
    for i, doc in enumerate(docs):
        assert doc.page_content.strip(), f"Doc {i} has empty page_content."
    print("✅ All retrieved docs have non-empty content.")


def test_table_chunks_retrievable():
    """Table chunks must be retrievable via filtered search."""
    store        = load_vector_store()
    table_results = store.similarity_search(
        SAMPLE_QUERY, k=4, filter={"type": "table"}
    )
    assert len(table_results) > 0, (
        "No table chunks retrieved. Check table extraction and indexing."
    )
    print(f"✅ Table-filtered search returned {len(table_results)} chunks.")


def test_format_context(docs):
    """format_context() must return a non-empty string with page labels."""
    context = format_context(docs)
    assert context.strip(), "format_context() returned empty string."
    assert "[Page" in context, "format_context() missing page labels."
    print("✅ format_context() returns properly labelled context string.")


def test_deduplication():
    """retrieve() must not return duplicate chunks."""
    docs  = retrieve(SAMPLE_QUERY, k=8)
    keys  = [doc.page_content[:150].strip() for doc in docs]
    assert len(keys) == len(set(keys)), (
        f"Duplicate chunks found in retrieve() results."
    )
    print("✅ No duplicate chunks in retrieve() results.")


def test_preview_results(docs):
    """Print a preview of retrieved chunks for manual inspection."""
    print(f"\n── Top {len(docs)} retrieved chunks ──")
    for i, doc in enumerate(docs, 1):
        page  = doc.metadata.get("page", "?")
        dtype = doc.metadata.get("type", "?")
        print(f"\n[{i}] page={page} type={dtype}")
        print(doc.page_content[:300])
        print("─" * 50)


if __name__ == "__main__":
    print(f"=== Retrieval Tests ===\n")
    print(f"Query: {SAMPLE_QUERY}\n")

    store = test_vector_store_loads()
    test_index_not_empty(store)
    docs  = test_retrieve_returns_results()
    test_retrieved_docs_have_metadata(docs)
    test_retrieved_docs_have_content(docs)
    test_table_chunks_retrievable()
    test_format_context(docs)
    test_deduplication()
    test_preview_results(docs)

    print("\n=== All retrieval tests passed ===")