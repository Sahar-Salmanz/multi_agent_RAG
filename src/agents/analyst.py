"""
analyst.py (agent node)
───────────────────────
Analyst Agent — interprets the retrieved chunks before the Writer
formats them into a user-facing answer.

Responsibilities:
  - Identify which retrieved chunks are actually relevant
  - Extract specific values from tables (kPa, mm, %)
  - Note applicable code factors or cross-references
  - Flag if retrieval seems insufficient

The Analyst outputs a structured interpretation, NOT a user-facing answer.
The Writer turns this into the final response.
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.agents.supervisor import AgentState

load_dotenv()

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")

ANALYST_SYSTEM_PROMPT = """You are a technical analyst specializing in building codes. 
Your job is to analyze retrieved textbook chunks and extract the precise information 
needed to answer a question.

You do NOT write the final answer — you produce a structured analysis that a Writer 
agent will use to compose the response.

Your analysis must include:
1. RELEVANT CHUNKS: List which chunks (by page number) directly answer the question
2. KEY VALUES: Extract any numerical values (kPa, mm, %, factors) with their page numbers
3. TABLE DATA: If a table is present, identify the exact row and value that applies
4. CROSS-REFERENCES: Note any articles or tables mentioned that should also be consulted
5. GAPS: If the chunks don't fully answer the question, state exactly what is missing

If the critic has sent feedback, address it specifically in your analysis.

Be precise and technical. Do not paraphrase loosely, and quote exact values and article numbers.
"""


def analyst_node(state: AgentState) -> AgentState:
    """
    Analyzes retrieved chunks and produces a structured interpretation
    for the Writer agent.
    """
    question = state["question"]
    context = state["context"]
    critic_feedback = state.get("critic_feedback", "")

    print(f"[Analyst] Analyzing {len(state['retrieved_docs'])} chunks...")

    # Include critic feedback in the prompt if this is a retry
    feedback_section = ""
    if critic_feedback and state.get("critic_target") in ("analyst", "interpretation_wrong"):
        feedback_section = f"""
CRITIC FEEDBACK TO ADDRESS:
{critic_feedback}

Make sure your analysis specifically fixes the issue identified above.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", ANALYST_SYSTEM_PROMPT),
        ("human", """Question: {question}

Retrieved context:
{context}
{feedback_section}

Provide your structured analysis:""")
    ])

    llm    = ChatAnthropic(model=CLAUDE_MODEL, temperature=0, max_tokens=1024)
    chain  = prompt | llm | StrOutputParser()

    interpretation = chain.invoke({
        "question":        question,
        "context":         context,
        "feedback_section": feedback_section,
    })

    print(f"[Analyst] Interpretation complete ({len(interpretation)} chars).")

    return {
        **state,
        "interpretation": interpretation,
    }