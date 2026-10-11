# Multi-Agent RAG Chatbot
A production-ready, domain-specific question-answering chatbot grounded in the National Building Code of Canada (NBCC 2020). Built with a multi-agent LangGraph pipeline for answer quality control, Retrieval-Augmented Generation (RAG) to prevent hallucinations, and source citations on every response.


## Features
- __Multi-agent pipeline:__ Retriever → Analyst → Writer → Critic, each with a single responsibility
- __Hallucination-free answers:__ Claude only answers from retrieved textbook passages
- __Source citations:__ every answer includes the page numbers it was drawn from
- __Quality control loop:__ Critic agent reviews every answer against an explicit rubric before it reaches the user; rejects with targeted feedback and re-runs the appropriate agent
- __Borderless table extraction:__ custom word-coordinate snapping reconstructs tables that standard PDF parsers miss entirely
- __Multi-turn conversation:__ persistent SQLite-backed chat history per user session
- __Multi-user ready:__ sessions namespaced by user_id from day one; swap SQLite for PostgreSQL to scale
- __REST API:__ FastAPI layer with Bearer token authentication
- __Browser UI:__ Chainlit chat interface with agent step display and citation cards


## Architecture
```
User Question
      │
      ▼
 Supervisor (LangGraph StateGraph)
      │
      ├──► Retriever Agent    ← vector similarity search (ChromaDB, no LLM)
      │         │
      ├──► Analyst Agent      ← interprets chunks, extracts table values (Claude)
      │         │
      ├──► Writer Agent       ← formats cited user-facing answer (Claude)
      │         │
      └──► Critic Agent       ← reviews against rubric CR1–CR6, approves or rejects (Claude)
                │
                ├── APPROVE ──► answer returned to user
                └── REJECT  ──► targeted feedback → re-run Retriever / Analyst / Writer
                                 (max 3 iterations)
```


## Critic Rubric
|Criterion|Description|Re-runs|
|---------|-----------|-------|
|CR1|Fact not present in retrieved context|Analyst|
|CR2|Number stated without page citation|Writer|
|CR3|Says "I don't know" but context has the answer|Retriever|
|CR4|Missing inline citation|Writer|
|CR5|Wrong table row used|Analyst|
|CR6|Internal contradiction|Writer|



## Tech Stack
|Layer|Tool|
|-----|----|
|LLM|Claude Sonnet (`claude-sonnet-4-5`) via Anthropic API|
|Embeddings|`nomic-embed-text` via Ollama (local, free)|
|Vector store|ChromaDB (persisted to disk)|
|Multi-agent orchestration|LangGraph|
|Orchestration|LangChain (LCEL)|
|PDF parsing|pdfplumber (borderless table extraction)|
|Chat history|SQLite via `SQLChatMessageHistory`|
|API|FastAPI + Uvicorn|
|UI|Chainlit|
|IDE|VSCode + Python|



## Project Structure
```
multi_agent_RAG/
├── data/
│   └── raw/                      # Original PDF textbook
├── vectorstore/
│   └── chroma_db/                # Persisted ChromaDB index (auto-generated)
├── database/
│   └── chat_history.db           # SQLite chat history (auto-generated)
├── src/
│   ├── ingestion/
│   │   ├── pdf_loader.py         # Load, chunk, extract tables from PDF
│   │   └── embedder.py           # Embed chunks → ChromaDB
│   ├── retrieval/
│   │   └── retriever.py          # Dual similarity search + deduplication
│   ├── chat/
│   │   ├── chain.py              # Single-turn RAG chain + get_llm() factory
│   │   └── history.py            # Multi-turn chain with SQLite session history
│   ├── agents/
│   │   ├── supervisor.py         # AgentState, LangGraph StateGraph, routing
│   │   ├── retriever.py          # Retriever agent node (wraps src/retrieval)
│   │   ├── analyst.py            # Analyst agent node (Claude API)
│   │   ├── writer.py             # Writer agent node (Claude API)
│   │   └── critic.py             # Critic agent node (Claude API + JSON rubric)
│   └── api/
│       ├── routes.py             # FastAPI endpoints
│       └── auth.py               # API key authentication
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_agents.py
├── app.py                        # Chainlit browser UI
├── main.py                       # FastAPI entrypoint
├── .env                          # API keys and config (never committed)
├── .gitignore
└── requirements.txt
```
