"""
history.py
──────────
Manages per-user conversation history using SQLite via LangChain's
SQLChatMessageHistory. Each conversation is keyed by a session_id
so multiple users (and multiple sessions per user) are supported.

Session ID convention:
  "{user_id}_{session_name}"  e.g. "sahar_session1"

Run standalone to test:
  python -m src.chat.history
"""

import os
from dotenv import load_dotenv
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from src.chat.chain import get_llm

from src.retrieval.retriever import retrieve, format_context

load_dotenv()

DB_PATH      = os.getenv("DB_PATH", "./database/chat_history.db")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
TOP_K        = int(os.getenv("TOP_K", 4))

# ── System prompt (same grounding rules as chain.py) ────────────────────────
SYSTEM_PROMPT = """You are a knowledgeable assistant answering questions about a specific textbook.

Rules you must follow:
1. Answer ONLY using the information in the provided context below. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, say "I don't have enough information in the textbook to answer that" — do not guess or make anything up.
3. After your answer, list the page numbers you used in a "Sources:" line, e.g. "Sources: Page 12, Page 45".
4. Be concise and accurate.

Context from the textbook:
{context}
"""


def get_session_history(session_id: str) -> SQLChatMessageHistory:
    """
    Return the SQLChatMessageHistory for a given session.
    SQLite file is created automatically on first use.

    Args:
        session_id: Unique key for the conversation, e.g. "sahar_session1".
    """
    # Ensure database directory exists
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True)

    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=f"sqlite:///{DB_PATH}",
    )


def build_chain_with_history():
    """
    Build a RAG chain wrapped with message history.
    The chain receives {context, question} and maintains
    the conversation history automatically per session_id.
    """
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),   # ← injected automatically
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,                  # called with session_id each time
        input_messages_key="question",
        history_messages_key="history",
    )

    return chain_with_history


def ask_with_history(question: str, session_id: str, k: int = TOP_K) -> dict:
    """
    Full RAG pipeline with persistent chat history.

    Args:
        question   : The user's question.
        session_id : Identifies the conversation, e.g. "sahar_session1".
        k          : Number of chunks to retrieve.

    Returns:
        dict with:
          - answer     : Claude's response (string)
          - sources    : list of {"page": int, "excerpt": str}
          - session_id : echoed back for convenience
    """
    # 1. Retrieve relevant chunks for this question
    docs = retrieve(question, k=k)
    context = format_context(docs) if docs else "No relevant context found."

    # 2. Run chain — history is loaded and saved automatically
    chain = build_chain_with_history()
    answer = chain.invoke(
        {"context": context, "question": question},
        config={"configurable": {"session_id": session_id}},
    )

    # 3. Build sources list from retrieved chunk metadata
    sources = [
        {"page": doc.metadata.get("page", "unknown"), "excerpt": doc.page_content[:200]}
        for doc in docs
    ]

    return {"answer": answer, "sources": sources, "session_id": session_id}


def get_history(session_id: str) -> list:
    """
    Return the full message history for a session as a list of dicts.

    Args:
        session_id: The session to retrieve history for.

    Returns:
        List of {"role": "human"|"ai", "content": str}
    """
    history = get_session_history(session_id)
    messages = []
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "ai", "content": msg.content})
    return messages


def clear_history(session_id: str) -> None:
    """
    Delete all messages for a session.

    Args:
        session_id: The session to clear.
    """
    history = get_session_history(session_id)
    history.clear()
    print(f"History cleared for session: {session_id}")


# Run standalone to test
if __name__ == "__main__":
    SESSION = "test_user_session1"

    # Clear any previous test history
    clear_history(SESSION)

    print("=== Multi-turn conversation test ===\n")

    # Turn 1
    q1 = "What is the live load for residential areas?"
    print(f"You: {q1}")
    r1 = ask_with_history(q1, SESSION)
    print(f"Bot: {r1['answer']}\n")

    # Turn 2 — follow-up that relies on prior context
    q2 = "Can you elaborate on that?"
    print(f"You: {q2}")
    r2 = ask_with_history(q2, SESSION)
    print(f"Bot: {r2['answer']}\n")

    # Turn 3 — another follow-up
    q3 = "What are the key concepts introduced in this chapter?"
    print(f"You: {q3}")
    r3 = ask_with_history(q3, SESSION)
    print(f"Bot: {r3['answer']}\n")

    # Print stored history
    print("=== Stored history ===")
    for msg in get_history(SESSION):
        role = "You" if msg["role"] == "human" else "Bot"
        print(f"{role}: {msg['content'][:120]}...")