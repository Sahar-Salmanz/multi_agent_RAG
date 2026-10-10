"""
writer.py (agent node)
──────────────────────
Writer Agent — takes the Analyst's structured interpretation and
produces a clean, cited, user-facing answer.

Responsibilities:
  - Write a clear, concise answer grounded strictly in the interpretation
  - Cite every factual claim with [Page X]
  - Format tables as readable markdown when relevant
  - End with a "Sources:" line listing all pages used
  - Never add information not present in the interpretation
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.agents.supervisor import AgentState

load_dotenv()

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")

WRITER_SYSTEM_PROMPT = """You are a technical writer producing answers about building codes.
You receive a structured analysis from an Analyst and must turn it into a clear, 
cited answer for the user.

Rules:
1. Answer ONLY from the information in the Analyst's interpretation — never add outside knowledge
2. Cite every factual claim with [Page X] immediately after the claim
3. If the answer involves a table, reproduce the relevant rows in markdown format
4. If the Analyst flagged gaps, say clearly what information was not available
5. End every answer with a "Sources:" line listing all page numbers used
6. Be concise — one paragraph maximum unless a table or list is needed
7. Never say "the analyst said" or "according to the analysis" — write directly to the user

Format:
[Your answer with inline citations]

Sources: Page X, Page Y
"""


def writer_node(state: AgentState) -> AgentState:
    """
    Produces the final user-facing answer from the Analyst's interpretation.
    On retry, incorporates Critic feedback to fix specific issues.
    """
    question        = state["question"]
    interpretation  = state["interpretation"]
    critic_feedback = state.get("critic_feedback", "")

    print("[Writer] Composing answer...")

    # Include critic feedback if this is a formatting retry
    feedback_section = ""
    if critic_feedback and state.get("critic_target") in ("writer", "formatting_error"):
        feedback_section = f"""
CRITIC FEEDBACK TO FIX:
{critic_feedback}

Your previous answer had the above issue. Fix it specifically.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", WRITER_SYSTEM_PROMPT),
        ("human", """Question: {question}

Analyst's interpretation:
{interpretation}
{feedback_section}

Write the answer:""")
    ])

    llm   = ChatAnthropic(model=CLAUDE_MODEL, temperature=0, max_tokens=1024)
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "question":        question,
        "interpretation":  interpretation,
        "feedback_section": feedback_section,
    })

    # Build sources list from retrieved docs metadata
    sources = [
        {
            "page":    doc.metadata.get("page", "unknown"),
            "excerpt": doc.page_content[:200],
        }
        for doc in state.get("retrieved_docs", [])
    ]

    # Track the best answer seen (used as fallback if critic rejects too many times)
    best_answer = state.get("best_answer", "") or answer

    print(f"[Writer] Answer composed ({len(answer)} chars).")

    return {
        **state,
        "answer":      answer,
        "sources":     sources,
        "best_answer": best_answer,
    }