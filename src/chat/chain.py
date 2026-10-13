"""
chain.py
────────
Connects the retriever to Claude (Anthropic API) to produce
grounded, cited answers from the textbook.

This is the core RAG logic:
  question → retrieve chunks → build prompt → call Claude → answer + sources

Run standalone to test:
  python -m src.chat.chain
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.retrieval.retriever import retrieve, format_context

load_dotenv()

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
TOP_K = int(os.getenv("TOP_K", 4))

# System prompt
# This is the most important lever for hallucination control and citation quality.
# Keep it explicit about (1) grounding, (2) saying "I don't know", and (3) citation format.
SYSTEM_PROMPT = """You are a knowledgeable assistant answering questions about a specific textbook.

Rules you must follow:
1. Answer ONLY using the information in the provided context below. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, say "I don't have enough information in the textbook to answer that" — do not guess or make anything up.
3. After your answer, list the page numbers you used in a "Sources:" line, e.g. "Sources: Page 12, Page 45".
4. Be concise and accurate. Quote sparingly, prefer explaining in your own words.

Context from the textbook:
{context}
"""

HUMAN_PROMPT = "{question}"


def get_llm() -> ChatAnthropic:
    """Return the configured Claude chat model. Centralised here so
    every part of the app uses the same model and settings."""
    return ChatAnthropic(
        model=CLAUDE_MODEL,
        temperature=0,       # deterministic, factual answers — not creative
        max_tokens=1024,
    )


def build_chain():
    """
    Build the RAG chain using LangChain Expression Language (LCEL).
    Returns a runnable that accepts {"context": ..., "question": ...}.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])
    llm = get_llm()
    parser = StrOutputParser()

    chain = prompt | llm | parser
    return chain


def ask(question: str, k: int = TOP_K) -> dict:
    """
    Full RAG pipeline: retrieve chunks, generate a cited answer.

    Args:
        question : The user's question.
        k        : Number of chunks to retrieve.

    Returns:
        dict with:
          - answer  : Claude's generated answer (string, includes "Sources:" line)
          - sources : list of {"page": int, "excerpt": str} used as context
    """
    # 1. Retrieve relevant chunks
    docs = retrieve(question, k=k)

    if not docs:
        return {
            "answer": "I don't have enough information in the textbook to answer that.",
            "sources": [],
        }

    context = format_context(docs)

    # 2. Run the chain
    chain = build_chain()
    answer = chain.invoke({"context": context, "question": question})

    # 3. Build a structured sources list from metadata (independent of
    # whatever Claude wrote in the "Sources:" line — this is the
    # ground truth of what was actually retrieved)
    sources = [
        {"page": doc.metadata.get("page", "unknown"), "excerpt": doc.page_content[:200]}
        for doc in docs
    ]

    return {"answer": answer, "sources": sources}


# Run standalone to test
if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the live load for residential areas?"

    print(f"Question: {question}\n")
    result = ask(question)

    print("── Answer ──────────────────────────────")
    print(result["answer"])

    print("\n── Retrieved sources (ground truth) ─────")
    for s in result["sources"]:
        print(f"  Page {s['page']}: {s['excerpt']}...")