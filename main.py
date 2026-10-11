"""
main.py
───────
Boots the FastAPI application.

Start the server:
  uvicorn main:app --reload

API docs (auto-generated):
  http://localhost:8000/docs
"""

from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(
    title="Domain-Specific RAG Chatbot",
    description="A chatbot grounded in a NBCC 2020 using RAG and Claude.",
    version="0.1.0",
)

app.include_router(router)


@app.get("/")
def root():
    return {"status": "ok", "message": "RAG Chatbot API is running."}