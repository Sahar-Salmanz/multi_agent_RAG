"""
embedder.py
───────────
Embeds document chunks using nomic-embed-text (via Ollama)
and persists them to a local ChromaDB vector store.

Run once (or whenever the textbook changes):
  python -m src.ingestion.embedder

Safe to re-run: checks if the index already exists and skips
embedding if it does (avoids re-embedding unnecessarily).
Force a rebuild with --force:
  python -m src.ingestion.embedder --force
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from src.ingestion.pdf_loader import load_split

load_dotenv()

VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vectordatabase/chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
PDF_PATH = os.getenv("PDF_PATH", "./data/raw/nbc2020.pdf")
BATCH_SIZE = 50
PDF_START_PAGE = int(os.getenv("PDF_START_PAGE", 1))
PDF_END_PAGE   = int(os.getenv("PDF_END_PAGE", 0)) or None


def get_embeddings() -> OllamaEmbeddings:
    """
    Return the shared embeddings object.
    Always import and use this — never instantiate OllamaEmbeddings
    directly elsewhere, so the model is configured in one place only.
    """
    return OllamaEmbeddings(model=EMBED_MODEL)


def build_vector_store(pdf_path: str = PDF_PATH, force: bool = False) -> Chroma:
    """
    Embed chunks from the PDF and persist to ChromaDB.

    Args:
        pdf_path : Path to the source PDF.
        force    : If True, re-embed even if the index already exists.

    Returns:
        A loaded Chroma vector store ready for similarity search.
    """
    store_path = Path(VECTOR_STORE_PATH)

    # Skip if already built
    if store_path.exists() and any(store_path.iterdir()) and not force:
        print(f"Vector store already exists at {store_path}. Loading existing index.")
        print("(Run with --force to re-embed with updated PDF or loader.)")
        return load_vector_store()

    # Load and chunk the PDF
    chunks = load_split(pdf_path, start_page=PDF_START_PAGE, end_page=PDF_END_PAGE)
    if not chunks:
        raise ValueError("No chunks produced from PDF. Check the file and path.")

    total = len(chunks)
    print(f"\nEmbedding {total} chunks with {EMBED_MODEL}...")
    print("This may take several minutes for large PDFs.")

    embeddings = get_embeddings()

    # Embed first batch to create the store 
    vector_store = Chroma.from_documents(
        documents=chunks[:BATCH_SIZE],
        embedding=embeddings,
        persist_directory=str(store_path),
    )

    # Embed remaining chunks in batches
    for i in range(BATCH_SIZE, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        vector_store.add_documents(batch)
        print(f"Embedded {min(i + BATCH_SIZE, total)}/{total} chunks...", end="\r")

    print(f"\nDone. Index persisted to: {store_path}")
    return vector_store


def load_vector_store() -> Chroma:
    """
    Load an existing persisted ChromaDB index (no re-embedding).
    Call this everywhere outside of ingestion (retriever, chain, etc.).
    """
    store_path = Path(VECTOR_STORE_PATH)
    if not store_path.exists():
        raise FileNotFoundError(
            f"No vector store found at {store_path}. "
            "Run build_vector_store() first."
        )
    return Chroma(
        persist_directory=str(store_path),
        embedding_function=get_embeddings(),
    )


# Run standalone
if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    store = build_vector_store(force=force)

    # Sanity check 1: total chunk count by type
    print("\n── Sanity check 1: index composition ──")
    all_chunks = store.get()
    total = len(all_chunks["ids"])
    types = [m.get("type", "None") for m in all_chunks["metadatas"]]

    from collections import Counter
    type_counts = Counter(types)
    print(f"Total chunks : {total}")
    for t, count in type_counts.items():
        print(f"type={t:10} : {count}")

    if type_counts.get("table", 0) == 0:
        print("\nWARNING: No table chunks found in index.")
        print("Check pdf_loader.py ! _extract_borderless_tables() may not be firing.")

    # Sanity check 2: general similarity search
    print("\n── Sanity check 2: general similarity search ──")
    query   = "What is the live load for residential areas?"
    results = store.similarity_search(query, k=5)
    for i, doc in enumerate(results, 1):
        page  = doc.metadata.get("page", "?")
        dtype = doc.metadata.get("type", "?")
        print(f"\n[Result {i} — Page {page} | type={dtype}]")
        print(doc.page_content[:300])

    # Sanity check 3: table-only search 
    print("\n── Sanity check 3: table-only similarity search ──")
    try:
        table_results = store.similarity_search(
            query, k=5, filter={"type": "table"}
        )
        if not table_results:
            print("  No table chunks retrieved for this query.")
        for i, doc in enumerate(table_results, 1):
            page = doc.metadata.get("page", "?")
            print(f"\n[Table {i} — Page {page}]")
            print(doc.page_content[:300])
    except Exception as e:
        print(f"Table search failed: {e}")