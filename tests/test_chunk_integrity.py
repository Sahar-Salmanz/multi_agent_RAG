"""
test_retrieval_quality.py
─────────────────────────
Measures retrieval recall and precision against a ground-truth
evaluation set of questions with known correct pages.

Recall  = fraction of questions where the correct page was retrieved
Precision = fraction of retrieved chunks that were relevant

Run from myChatbot/:
    python tests/test_retrieval_quality.py

Expand EVAL_SET as you identify more ground-truth question/page pairs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retrieval.retriever import retrieve

# Ground truth: question → pages that must appear in top-k results
EVAL_SET = [
    {
        "question":       "What is the live load for residential bedrooms?",
        "expected_pages": [439],
        "description":    "Table 4.1.5.3 — residential bedrooms 1.9 kPa",
    },
    {
        "question":       "What is the live load for office areas on upper floors?",
        "expected_pages": [439],
        "description":    "Table 4.1.5.3 — office areas above ground 2.4 kPa",
    },
    {
        "question":       "What is the live load for factory floors?",
        "expected_pages": [439],
        "description":    "Table 4.1.5.3 — factories 6.0 kPa",
    },
    {
        "question":       "What is the live load for exit stairs and fire escapes?",
        "expected_pages": [439],
        "description":    "Table 4.1.5.3 — exits and fire escapes 4.8 kPa",
    },
    {
        "question":       "What is the live load for library stack rooms?",
        "expected_pages": [439],
        "description":    "Table 4.1.5.3 — library stack rooms 7.2 kPa",
    },
    {
        "question":       "What is the minimum roof live load?",
        "expected_pages": [439],
        "description":    "Table 4.1.5.3 — roofs 1.0 kPa",
    },
    {
        "question":       "What live load applies to assembly areas without fixed seats?",
        "expected_pages": [438, 439],
        "description":    "Table 4.1.5.3 — assembly areas 4.8 kPa",
    },
    {
        "question":       "What is the live load for garages with vehicles under 4000 kg?",
        "expected_pages": [439],
        "description":    "Table 4.1.5.3 — garages vehicles < 4000 kg 2.4 kPa",
    },
]


def evaluate_recall(eval_set, k=8):
    """
    Recall: for each question, did at least one expected page appear in top-k?
    """
    print("=== Retrieval Recall ===\n")
    hits  = 0
    total = len(eval_set)

    for item in eval_set:
        docs            = retrieve(item["question"], k=k)
        retrieved_pages = {doc.metadata.get("page") for doc in docs}
        expected_pages  = set(item["expected_pages"])
        hit             = bool(retrieved_pages & expected_pages)
        hits            += int(hit)

        status = "✅" if hit else "❌"
        print(f"{status} {item['description']}")
        print(f"   Expected pages : {sorted(expected_pages)}")
        print(f"   Retrieved pages: {sorted(p for p in retrieved_pages if p)}\n")

    recall = hits / total
    print(f"── Recall: {hits}/{total} = {recall:.0%} ──\n")
    return recall


def evaluate_precision(eval_set, k=8):
    """
    Precision: across all questions, what fraction of retrieved chunks
    came from an expected page?
    """
    print("=== Retrieval Precision ===\n")
    total_retrieved = 0
    total_relevant  = 0

    for item in eval_set:
        docs            = retrieve(item["question"], k=k)
        expected_pages  = set(item["expected_pages"])
        relevant        = sum(1 for doc in docs if doc.metadata.get("page") in expected_pages)
        total_retrieved += len(docs)
        total_relevant  += relevant

        precision = relevant / len(docs) if docs else 0
        print(f"Q: {item['question'][:60]}")
        print(f"   Relevant chunks: {relevant}/{len(docs)} = {precision:.0%}\n")

    overall = total_relevant / total_retrieved if total_retrieved else 0
    print(f"── Overall Precision: {total_relevant}/{total_retrieved} = {overall:.0%} ──\n")
    return overall


def evaluate_table_recall(eval_set, k=4):
    """
    Table recall: for table-heavy questions, does the table-filtered
    search return the correct page?
    """
    from src.ingestion.embedder import load_vector_store
    store = load_vector_store()

    print("=== Table-Filtered Recall ===\n")
    hits  = 0
    total = len(eval_set)

    for item in eval_set:
        try:
            table_docs      = store.similarity_search(
                item["question"], k=k, filter={"type": "table"}
            )
            retrieved_pages = {doc.metadata.get("page") for doc in table_docs}
            expected_pages  = set(item["expected_pages"])
            hit             = bool(retrieved_pages & expected_pages)
            hits            += int(hit)

            status = "✅" if hit else "❌"
            print(f"{status} {item['description']}")
            print(f"   Table pages retrieved: {sorted(p for p in retrieved_pages if p)}\n")
        except Exception as e:
            print(f"⚠️  Table search failed for: {item['question'][:50]} — {e}\n")

    recall = hits / total
    print(f"── Table Recall: {hits}/{total} = {recall:.0%} ──\n")
    return recall


if __name__ == "__main__":
    TOP_K = int(os.getenv("TOP_K", 8))
    print(f"Evaluating with TOP_K={TOP_K}\n")
    print("=" * 60)

    recall    = evaluate_recall(EVAL_SET, k=TOP_K)
    precision = evaluate_precision(EVAL_SET, k=TOP_K)
    t_recall  = evaluate_table_recall(EVAL_SET, k=4)

    print("=" * 60)
    print("── Final Scores ──")
    print(f"   Recall    : {recall:.0%}")
    print(f"   Precision : {precision:.0%}")
    print(f"   Table recall: {t_recall:.0%}")
    print("=" * 60)

    if recall < 0.7:
        print("\n⚠️  Recall below 70% — consider increasing TOP_K or checking table extraction.")
    if precision < 0.3:
        print("\n⚠️  Precision below 30% — consider reducing TOP_K or tightening chunk_size.")