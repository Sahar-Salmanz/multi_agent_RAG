"""
test_answer_quality.py
──────────────────────
End-to-end answer quality evaluation.
Runs the full multi-agent pipeline on a ground-truth eval set and
checks that answers contain the correct value, page citation,
and were approved by the Critic.

  Each question triggers 3+ Claude API calls (~$0.005–0.015 each).
Keep the eval set small during development.

Run from myChatbot/:
    python tests/test_answer_quality.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.supervisor import run

# Ground truth: question → expected value and page in the answer
EVAL_SET = [
    {
        "question":       "What is the live load for residential bedrooms?",
        "expected_value": "1.9",
        "expected_page":  439,
        "description":    "Residential bedrooms — 1.9 kPa",
    },
    {
        "question":       "What is the live load for office areas on floors above ground level?",
        "expected_value": "2.4",
        "expected_page":  439,
        "description":    "Office areas above ground — 2.4 kPa",
    },
    {
        "question":       "What is the live load for factory floors?",
        "expected_value": "6.0",
        "expected_page":  439,
        "description":    "Factories — 6.0 kPa",
    },
    {
        "question":       "What is the live load for library stack rooms?",
        "expected_value": "7.2",
        "expected_page":  439,
        "description":    "Library stack rooms — 7.2 kPa",
    },
    {
        "question":       "What is the live load for exit stairs and fire escapes?",
        "expected_value": "4.8",
        "expected_page":  439,
        "description":    "Exits and fire escapes — 4.8 kPa",
    },
]

# Out-of-scope questions — answer should indicate no information available
OUT_OF_SCOPE = [
    {
        "question":    "What are the fire egress requirements for high-rise buildings?",
        "description": "Fire egress — not in pages 430–557",
    },
    {
        "question":    "What is the boiling point of water?",
        "description": "Completely out of scope",
    },
]


def evaluate_answer_quality(eval_set):
    """
    For each question, check:
      1. Expected value appears in the answer
      2. Expected page is cited in the answer
      3. Critic approved the answer
    """
    print("=== Answer Quality Evaluation ===\n")
    correct    = 0
    total      = len(eval_set)
    iterations = []

    for item in eval_set:
        print(f"Q: {item['description']}")

        result      = run(question=item["question"], session_id="eval_quality")
        answer      = result["answer"]
        decision    = result["critic_decision"]
        iters       = result["iterations"]

        value_found = item["expected_value"] in answer
        page_found  = str(item["expected_page"]) in answer
        approved    = decision == "APPROVE"
        passed      = value_found and page_found and approved

        correct    += int(passed)
        iterations.append(iters)

        status = "✅" if passed else "❌"
        print(f"{status} Value '{item['expected_value']}' in answer : {value_found}")
        print(f"   Page {item['expected_page']} cited          : {page_found}")
        print(f"   Critic decision                : {decision} (iteration {iters})")

        if not passed:
            print(f"\n   Answer preview:\n   {answer[:300]}\n")
        print()

    accuracy       = correct / total
    avg_iterations = sum(iterations) / len(iterations) if iterations else 0

    print(f"── Answer Accuracy     : {correct}/{total} = {accuracy:.0%} ──")
    print(f"── Avg Critic iterations: {avg_iterations:.1f} ──\n")
    return accuracy


def evaluate_out_of_scope(out_of_scope):
    """
    Out-of-scope questions should produce honest 'I don't know' responses
    rather than hallucinated answers.
    """
    print("=== Out-of-Scope Handling ===\n")
    correct = 0
    total   = len(out_of_scope)

    # Phrases that indicate an honest "I don't know" response
    honest_indicators = [
        "don't have enough information",
        "not available",
        "not in the",
        "cannot answer",
        "outside",
        "not found",
        "no information",
    ]

    for item in out_of_scope:
        print(f"Q: {item['description']}")

        result  = run(question=item["question"], session_id="eval_oos")
        answer  = result["answer"].lower()

        honest  = any(phrase in answer for phrase in honest_indicators)
        correct += int(honest)

        status = "✅" if honest else "❌"
        print(f"{status} Honest response : {honest}")
        print(f"   Critic decision: {result['critic_decision']}")
        print(f"   Answer preview : {result['answer'][:200]}\n")

    accuracy = correct / total
    print(f"── Out-of-scope handling: {correct}/{total} = {accuracy:.0%} ──\n")
    return accuracy


def print_summary(in_scope_accuracy, oos_accuracy):
    print("=" * 60)
    print("── Final Summary ──")
    print(f"   In-scope accuracy    : {in_scope_accuracy:.0%}")
    print(f"   Out-of-scope handling: {oos_accuracy:.0%}")
    print("=" * 60)

    if in_scope_accuracy < 0.7:
        print("\n⚠️  In-scope accuracy below 70%.")
        print("   Check retrieval quality and Analyst prompt.")
    if oos_accuracy < 0.8:
        print("\n⚠️  Out-of-scope handling below 80%.")
        print("   Check Writer system prompt — 'I don't know' instruction may need strengthening.")


if __name__ == "__main__":
    print(f"⚠️  This test makes {len(EVAL_SET) + len(OUT_OF_SCOPE)} pipeline calls "
          f"(~{(len(EVAL_SET) + len(OUT_OF_SCOPE)) * 3} Claude API calls).\n")

    in_scope_acc = evaluate_answer_quality(EVAL_SET)
    oos_acc      = evaluate_out_of_scope(OUT_OF_SCOPE)
    print_summary(in_scope_acc, oos_acc)