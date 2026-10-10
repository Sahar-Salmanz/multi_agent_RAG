"""
critic.py (agent node)
──────────────────────
Critic Agent — reviews the Writer's answer against an explicit rubric
and either approves it or rejects it with targeted, actionable feedback.

On rejection, it tags the feedback with a "target" telling the Supervisor
exactly which agent to re-run:
  - "retrieval_insufficient" → Retriever (more/different chunks needed)
  - "interpretation_wrong"   → Analyst (misread the chunks)
  - "formatting_error"       → Writer only (chunks were fine, answer was not)

Max iterations is enforced in supervisor.py — the Critic does not need
to know about it.
"""

import os
import json
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.agents.supervisor import AgentState

load_dotenv()

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")

CRITIC_SYSTEM_PROMPT = """You are a strict quality reviewer for a building code chatbot.
You receive:
  1. The original question
  2. The retrieved context chunks (ground truth)
  3. The Writer's answer

Your job is to decide: APPROVE or REJECT.

REJECT if ANY of these criteria fail:
  CR1. The answer references a fact, value, or article NOT present in the retrieved context
  CR2. A numerical value (kPa, mm, %, factor) is stated but cannot be traced to a specific page in the context
  CR3. The answer says "I don't know" or "not available" but the context CLEARLY contains the answer
  CR4. A citation [Page X] is missing after a factual claim
  CR5. A table value is cited but the wrong row, column, or value was used
  CR6. The answer is internally contradictory

If REJECTING, you MUST:
  1. State which criterion failed (CR1–CR6)
  2. Quote the EXACT part of the answer that failed
  3. Quote the EXACT part of the context that contradicts or should have been used
  4. Specify the target agent:
       - "retrieval_insufficient" if the context simply doesn't contain enough info
       - "interpretation_wrong" if the context had the info but was misread
       - "formatting_error" if the answer structure/citations are wrong but the content is correct

Respond ONLY with valid JSON in this exact format:
{
  "decision": "APPROVE" or "REJECT",
  "criterion_failed": "CR1" or null,
  "quoted_answer_problem": "exact quote from answer" or null,
  "quoted_context_evidence": "exact quote from context" or null,
  "feedback": "specific actionable feedback" or null,
  "target": "retrieval_insufficient" | "interpretation_wrong" | "formatting_error" | null
}

Be strict but fair. Approve when the answer is accurate and well-cited, even if imperfect in style.
"""


def critic_node(state: AgentState) -> AgentState:
    """
    Reviews the Writer's answer and returns APPROVE or REJECT with feedback.
    Increments the iteration counter on each call.
    """
    question  = state["question"]
    context   = state["context"]
    answer    = state["answer"]
    iteration = state.get("iteration", 0) + 1

    print(f"[Critic] Reviewing answer (iteration {iteration})...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", CRITIC_SYSTEM_PROMPT),
        ("human", """Question: {question}

Retrieved context (ground truth):
{context}

Writer's answer:
{answer}

Review the answer and respond with JSON:""")
    ])

    llm   = ChatAnthropic(model=CLAUDE_MODEL, temperature=0, max_tokens=512)
    chain = prompt | llm | StrOutputParser()

    raw_response = chain.invoke({
        "question": question,
        "context":  context,
        "answer":   answer,
    })

    # ── Parse JSON response ──────────────────────────────────────────────────
    try:
        # Strip markdown code fences if present
        clean = raw_response.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        review = json.loads(clean.strip())
    except json.JSONDecodeError:
        # If JSON parsing fails, default to approve to avoid infinite loop
        print(f"[Critic] Failed to parse JSON response. Defaulting to APPROVE.")
        review = {"decision": "APPROVE", "feedback": None, "target": None}

    decision = review.get("decision", "APPROVE")
    feedback = review.get("feedback")
    target   = review.get("target")

    # Build human-readable feedback string
    feedback_str = ""
    if decision == "REJECT" and feedback:
        criterion       = review.get("criterion_failed", "")
        answer_problem  = review.get("quoted_answer_problem", "")
        context_evidence = review.get("quoted_context_evidence", "")

        feedback_str = (
            f"Failed criterion: {criterion}\n"
            f"Problem in answer: \"{answer_problem}\"\n"
            f"Context evidence: \"{context_evidence}\"\n"
            f"Feedback: {feedback}"
        )

    print(f"[Critic] Decision: {decision}"
          + (f" | Target: {target}" if target else "")
          + (f" | {criterion}" if decision == "REJECT" else ""))

    # Update best_answer only on approval
    best_answer = state.get("best_answer", "")
    if decision == "APPROVE":
        best_answer = answer

    return {
        **state,
        "critic_decision": decision,
        "critic_feedback": feedback_str,
        "critic_target":   target or "writer",
        "iteration":       iteration,
        "best_answer":     best_answer,
    }