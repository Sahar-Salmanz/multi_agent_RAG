"""
app.py
──────
Chainlit UI — now powered by the multi-agent LangGraph pipeline.
Shows which agents ran, how many critic iterations occurred,
and displays citation cards per source.

Run:
  chainlit run app.py --watch
"""

import os
import chainlit as cl
from dotenv import load_dotenv
from src.agents.supervisor import run as agent_run
from src.chat.history import get_session_history

load_dotenv()

DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "sahar")


@cl.on_chat_start
async def on_chat_start():
    session_name = cl.user_session.get("id", "session1")
    session_id   = f"{DEFAULT_USER_ID}_{session_name}"
    cl.user_session.set("session_id", session_id)

    await cl.Message(
        content=(
            "Hello! I'm your building code assistant powered by a multi-agent pipeline.\n\n"
            "**Agents:** Retriever → Analyst → Writer → Critic\n\n"
            "Ask me anything about the NBC 2020."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    question   = message.content

    # ── Show agent progress steps ────────────────────────────────────────────
    async with cl.Step(name="Retriever") as step:
        step.input = question
        step.output = "Searching vector store..."

    async with cl.Step(name="Analyst") as step:
        step.input = "Retrieved chunks"
        step.output = "Interpreting relevant passages..."

    async with cl.Step(name="Writer") as step:
        step.input = "Analyst interpretation"
        step.output = "Composing cited answer..."

    async with cl.Step(name="Critic") as step:
        step.input = "Writer's answer"
        step.output = "Reviewing for accuracy..."

    # ── Run the full agent pipeline ──────────────────────────────────────────
    result = agent_run(question=question, session_id=session_id)

    # ── Persist to history ───────────────────────────────────────────────────
    history = get_session_history(session_id)
    from langchain_core.messages import HumanMessage, AIMessage
    history.add_message(HumanMessage(content=question))
    history.add_message(AIMessage(content=result["answer"]))

    # ── Build citation elements ──────────────────────────────────────────────
    citation_elements = []
    for i, source in enumerate(result["sources"], 1):
        citation_elements.append(
            cl.Text(
                name=f"Source {i} — Page {source['page']}",
                content=source["excerpt"],
                display="side",
            )
        )

    # ── Send answer ──────────────────────────────────────────────────────────
    critic_badge = (
        f"✅ Approved in {result['iterations']} iteration(s)"
        if result["critic_decision"] == "APPROVE"
        else f"⚠️ Best answer after {result['iterations']} iteration(s)"
    )

    await cl.Message(
        content=f"{result['answer']}\n\n*{critic_badge}*",
        elements=citation_elements if citation_elements else [],
    ).send()