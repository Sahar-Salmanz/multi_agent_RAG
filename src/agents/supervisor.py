"""
supervisor.py
─────────────
Defines the LangGraph StateGraph for the multi-agent RAG chatbot.

Graph flow:
    START → retriever → analyst → writer → critic → END (if approved)
                                                   ↘ analyst (if interpretation_wrong)
                                                   ↘ retriever (if retrieval_insufficient)
                                                   ↘ writer (if formatting_error)

State is passed between all agents as a typed dict.
Max 3 critic iterations before returning best answer.
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
import operator


# Shared State 
class AgentState(TypedDict):
    # Input
    question:         str                    # original user question
    session_id:       str                    # for chat history

    # Retriever output
    retrieved_docs:   list                   # LangChain Document objects
    context:          str                    # formatted context string

    # Analyst output
    interpretation:   str                    # structured interpretation of chunks

    # Writer output
    answer:           str                    # final user-facing answer
    sources:          list                   # list of {"page": int, "excerpt": str}

    # Critic output
    critic_decision:  str                    # "APPROVE" | "REJECT"
    critic_feedback:  str                    # detailed feedback if REJECT
    critic_target:    str                    # "retriever" | "analyst" | "writer"

    # Control
    iteration:        int                    # critic iteration count
    best_answer:      str                    # best answer seen so far (fallback)


MAX_ITERATIONS = 3


def route_after_critic(state: AgentState) -> str:
    """
    Routing function called after the Critic node.
    Decides which node to visit next based on critic decision.
    """
    if state["critic_decision"] == "APPROVE":
        return END

    if state["iteration"] >= MAX_ITERATIONS:
        # Too many retries — return best answer seen
        return END

    target = state.get("critic_target", "writer")
    if target == "retrieval_insufficient":
        return "retriever"
    elif target == "interpretation_wrong":
        return "analyst"
    else:
        return "writer"


def build_graph():
    """
    Build and compile the LangGraph StateGraph.
    Returns a compiled graph ready to invoke.
    """
    from src.agents.retriever import retriever_node
    from src.agents.analyst  import analyst_node
    from src.agents.writer   import writer_node
    from src.agents.critic   import critic_node

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("retriever", retriever_node)
    graph.add_node("analyst",   analyst_node)
    graph.add_node("writer",    writer_node)
    graph.add_node("critic",    critic_node)

    # Define edges
    graph.add_edge(START,       "retriever")
    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst",   "writer")
    graph.add_edge("writer",    "critic")

    # Conditional routing after critic
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "retriever": "retriever",
            "analyst":   "analyst",
            "writer":    "writer",
            END:          END,
        }
    )

    return graph.compile()


# Convenience run function
def run(question: str, session_id: str = "default") -> dict:
    """
    Run the full multi-agent pipeline for a question.

    Args:
        question   : User's question.
        session_id : Chat session identifier.

    Returns:
        dict with answer, sources, critic_decision, iteration count.
    """
    graph = build_graph()

    initial_state = AgentState(
        question=question,
        session_id=session_id,
        retrieved_docs=[],
        context="",
        interpretation="",
        answer="",
        sources=[],
        critic_decision="",
        critic_feedback="",
        critic_target="",
        iteration=0,
        best_answer="",
    )

    final_state = graph.invoke(initial_state)

    return {
        "answer":          final_state.get("best_answer") or final_state.get("answer"),
        "sources":         final_state.get("sources", []),
        "critic_decision": final_state.get("critic_decision"),
        "iterations":      final_state.get("iteration", 0),
    }