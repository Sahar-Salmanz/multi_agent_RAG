"""
routes.py
─────────
FastAPI endpoints — powered by the multi-agent LangGraph pipeline.
"""

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel

from src.api.auth import get_current_user
from src.agents.supervisor import run as agent_run
from src.chat.history import get_history, clear_history, get_session_history

router = APIRouter()


# Schemas
class ChatRequest(BaseModel):
    session_name: str
    message:      str


class SourceItem(BaseModel):
    page:    int | str
    excerpt: str


class ChatResponse(BaseModel):
    session_id:      str
    answer:          str
    sources:         list[SourceItem]
    critic_decision: str
    iterations:      int


class MessageItem(BaseModel):
    role:    str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    messages:   list[MessageItem]


class SessionsResponse(BaseModel):
    user_id:  str
    sessions: list[str]


# Routes
@router.post("/chat", response_model=ChatResponse)
def chat(
    body:    ChatRequest,
    user_id: str = Depends(get_current_user),
):
    session_id = f"{user_id}_{body.session_name}"

    # Run the multi-agent pipeline
    result = agent_run(
        question=body.message,
        session_id=session_id,
    )

    # Persist to chat history
    history = get_session_history(session_id)
    from langchain_core.messages import HumanMessage, AIMessage
    history.add_message(HumanMessage(content=body.message))
    history.add_message(AIMessage(content=result["answer"]))

    return ChatResponse(
        session_id=session_id,
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]],
        critic_decision=result["critic_decision"],
        iterations=result["iterations"],
    )


@router.get("/history/{session_name}", response_model=HistoryResponse)
def history(
    session_name: str,
    user_id:      str = Depends(get_current_user),
):
    session_id = f"{user_id}_{session_name}"
    messages   = get_history(session_id)
    return HistoryResponse(
        session_id=session_id,
        messages=[MessageItem(**m) for m in messages],
    )


@router.delete("/history/{session_name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_history(
    session_name: str,
    user_id:      str = Depends(get_current_user),
):
    clear_history(f"{user_id}_{session_name}")


@router.get("/sessions", response_model=SessionsResponse)
def list_sessions(user_id: str = Depends(get_current_user)):
    from sqlalchemy import create_engine, text
    import os
    db_path = os.getenv("DB_PATH", "./database/chat_history.db")
    engine  = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT session_id FROM message_store WHERE session_id LIKE :prefix"),
            {"prefix": f"{user_id}_%"},
        ).fetchall()
    prefix   = f"{user_id}_"
    sessions = [row[0].removeprefix(prefix) for row in rows]
    return SessionsResponse(user_id=user_id, sessions=sessions)