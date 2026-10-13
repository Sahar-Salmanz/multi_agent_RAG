"""
retriever.py
-------------
Retrieves top-k relevant document chunks from the ChromaDB vector store using a query.

Run this to test retrieval:
    python -m src.retrieval.retriever "your questions here"

All other modules (chain.py, routes.py) should import 
get_retriever() from here. 
Never load ChromaDB directly
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.vectorstores import VectorStoreRetriever

from src.ingestion.embedder import load_vector_store

load_dotenv()
TOP_K = int(os.getenv("TOP_K", 8))


def get_retriever(k: int = TOP_K) -> VectorStoreRetriever:
    """
    Load the persisted ChromaDB index and return a LangChain retriever.

    :param k: Number of top-k relevant document chunks to retrieve, defaults to TOP_K in .env
    :return: LangChain VectorStoreRetriever ready to call with .invoke(query)
    """
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k": k}
    )
    return retriever


def retrieve(query: str, k: int = TOP_K) -> list:
    """
    Retrieve top-k relevant document chunks from the ChromaDB vector store using a query.

    :param query: The user's question or query string.
    :param k: Number of top-k relevant document chunks to retrieve, defaults to TOP_K in .env
    :return: List of retrieved documents (LangChain Document objects) with metadata and page_content.
    """
    store = load_vector_store()

    # 1. General similarity search (text + tables)
    general_docs = store.similarity_search(query, k=k)

    # 2. Dedicated table-only search
    try:
        table_docs = store.similarity_search(
            query,
            k=4,
            filter={"type": "table"},
        )
    except Exception:
        table_docs = []

    # 3. If query mentions a specific table number, fetch that page directly
    page_docs = []
    table_match = re.search(r"Table\s+([\d\.]+)", query, re.IGNORECASE)
    if table_match:
        try:
            # Known table → page mapping (add more as needed)
            TABLE_PAGES = {
                "4.1.5.3": [438, 439],
                "4.1.8.18": [502, 503]
            }
            table_num = table_match.group(1)
            pages     = TABLE_PAGES.get(table_num, [])
            if pages:
                raw = store.get(where={"page": {"$in": pages}})
                from langchain_core.documents import Document
                page_docs = [
                    Document(page_content=content, metadata=meta)
                    for content, meta in zip(raw["documents"], raw["metadatas"])
                    if meta.get("type") == "table"
                ]
        except Exception:
            pass

    # 4. Merge — deduplicate by first 150 chars
    seen, merged = set(), []
    for doc in general_docs + table_docs + page_docs:
        key = doc.page_content[:150].strip()
        if key not in seen:
            seen.add(key)
            merged.append(doc)

    return merged


def format_context(docs: list) -> str:
    """
    Format retrieved chunks into a single string context for the LLM prompt.

    :param docs: List of retrieved documents (LangChain Document objects).
    :return: Formatted string containing all relevant document chunks, ready to inject into the prompt.
    """
    sections = []
    for doc in docs:
        page = doc.metadata.get("page", "unknown")
        sections.append(f"[Page {page}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(sections)


# Test retrieval with a sample query
if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the live load for residential areas?"

    print(f"Query: {query}\n")
    docs = retrieve(query)

    print(f"Top {len(docs)} chunks retrieved:\n")
    for i, doc in enumerate(docs, 1):
        page  = doc.metadata.get("page", "unknown")
        print(f"-- Result {i} (Page {page}) ----------------------")
        print(doc.page_content[:600])
        print("\n")

    print("-- Formatted context (as seen by the LLM) ----------------------")
    print(format_context(docs))