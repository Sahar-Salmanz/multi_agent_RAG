"""
retriever.py (agent node)
─────────────────────────
Retriever Agent — wraps the existing retrieve() and format_context()
functions as a LangGraph node.

No LLM call — pure vector search.
If the critic sends it back with "retrieval_insufficient",
it increases k by 2 to fetch more candidates.
"""
import os
from dotenv import load_dotenv
from src.agents.supervisor import AgentState
from src.retrieval.retriever import retrieve, format_context


load_dotenv()
TOP_K = int(os.getenv("TOP_K", 4))

def retriever_node(state: AgentState) -> AgentState:
    """
    Retrieves relevant chunks from ChromaDB for the user's question.
    On retry (iteration > 0), increases k to fetch more candidates.
    """
    question  = state["question"]
    iteration = state.get("iteration", 0)

    # Increase k on retries triggered by retrieval_insufficient
    k = TOP_K + (2 * iteration) if state.get("critic_target") == "retrieval_insufficient" else TOP_K

    print(f"[Retriever] Fetching top-{k} chunks for: '{question[:60]}...'")

    docs = retrieve(question, k=k)
    context = format_context(docs)

    print(f"[Retriever] Retrieved {len(docs)} chunks.")

    return {
        **state,
        "retrieved_docs": docs,
        "context":        context,
    }