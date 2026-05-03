# Helix SROP

A production-ready **Stateful RAG Orchestration Pipeline** that powers an AI Support Concierge for Helix — a B2B developer tools platform.

The system handles two distinct workflows within a single ongoing conversation: answering product documentation questions via retrieval-augmented generation, and resolving account queries via internal tooling — all with full session persistence, structured tracing, and async-first architecture.

**Stack:** Python 3.12 · FastAPI · Google ADK · SQLite (SQLAlchemy 2.x async) · ChromaDB · Gemini 2.0 Flash

---

## Setup

```bash
git clone <repo-url>
cd helix-srop-assignment
pip install -e ".[dev]"

cp .env.example .env          # add GOOGLE_API_KEY

python -m app.rag.ingest --path docs/
uvicorn app.main:app --reload
```

Get the API key at [aistudio.google.com](https://aistudio.google.com).

---

## Usage

```bash
# Create a session
SESSION=$(curl -s -X POST http://localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u1", "plan_tier": "pro"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

# Knowledge question — answered via RAG
curl -s -X POST http://localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content": "How do I rotate a deploy key?"}' | python3 -m json.tool

# Account question — answered via tools
curl -s -X POST http://localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content": "Show me my last 3 failed builds"}' | python3 -m json.tool

# Follow-up — state and history carry across turns
curl -s -X POST http://localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content": "What is my current plan tier?"}' | python3 -m json.tool
```

---

## Tests

```bash
pytest -q
```

LLM is mocked at the `_call_adk` boundary — the full suite runs instantly without a real API key.

---

## Architecture

```
POST /v1/chat/{session_id}
         │
         ▼
┌─────────────────────────────────────────────┐
│  SROP Pipeline                              │
│  1. Load SessionState from SQLite           │
│  2. Re-hydrate last 10 messages             │
│  3. Build root agent (state + history)      │
│  4. asyncio.wait_for → InMemoryRunner       │
│  5. Stream events → routing + tool traces   │
│  6. Persist state / messages / trace to DB  │
└──────────────────┬──────────────────────────┘
                   │  AgentTool (LLM-driven routing)
         ┌─────────┴──────────┐
         ▼                    ▼
   KnowledgeAgent        AccountAgent
   └─ search_docs()      └─ get_recent_builds()
          │                  get_account_status()
          ▼
    ChromaDB · cosine similarity
    Google text-embedding-004
```

**Database schema**

| Table | Purpose |
|---|---|
| `users` | user_id, plan_tier |
| `sessions` | session_id, state (JSON), timestamps |
| `messages` | per-turn user/assistant messages with trace_id |
| `agent_traces` | routed_to, tool_calls, chunk_ids, latency_ms |

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/sessions` | Create session · `{user_id, plan_tier}` → `{session_id}` |
| `POST` | `/v1/chat/{session_id}` | Send message → `{reply, routed_to, trace_id}` |
| `GET` | `/v1/traces/{trace_id}` | Fetch structured turn trace |
| `GET` | `/healthz` | Health check |

Errors follow [RFC 7807](https://datatracker.ietf.org/doc/html/rfc7807) — `{type, title, status, detail}`.

---

## Design Decisions

### State Persistence — Pattern 3 with History Re-hydration

Session state (`user_id`, `plan_tier`, `last_agent`, `turn_count`) is serialized to a JSON column in SQLite on every commit. The last 10 messages are loaded from the `messages` table and injected into the root agent's system prompt on each turn, giving the agent conversational memory without a custom `BaseSessionService` or ADK session replay.

This survives process restarts by design — the next request loads everything from the DB. The ADK `InMemoryRunner` is intentionally ephemeral; all durability lives in SQLAlchemy.

### Routing — AgentTool, not String Parsing

The root orchestrator uses ADK's `AgentTool` pattern. The LLM selects `knowledge_agent` or `account_agent` as a structured function call. Agent names are declared as module-level constants and matched exactly against `event.author` in the event stream — no substring heuristics.

### RAG Pipeline

- **Chunking:** Heading-aware markdown splitting (`# ## ###` boundaries) keeps procedural sections intact. Long sections are sub-chunked at 512 characters with 64-char overlap.
- **Embeddings:** Google `text-embedding-004` (768-dim) at both ingest and query time with correct `task_type` (`retrieval_document` vs `retrieval_query`).
- **Retrieval:** Cosine similarity via ChromaDB. Chunks scoring below `0.45` are discarded before being passed to the agent.
- **Chunk IDs:** `SHA-256(filepath::index)[:16]` — deterministic and deduplication-safe on re-ingest.

### Performance

- `genai.Client` and ChromaDB collection are `@lru_cache` singletons — initialized once, not per request.
- `AgentTool` wrappers for sub-agents are module-level — only the root agent's instruction string is rebuilt per turn to inject fresh session context.
- All LLM and vector store calls are wrapped in `asyncio.wait_for` with configurable timeouts.

---

## Project Structure

```
app/
├── main.py                      # FastAPI app, lifespan, error handlers
├── settings.py                  # Pydantic Settings from .env
├── agents/
│   ├── orchestrator.py          # Root agent + singleton AgentTools
│   ├── knowledge.py             # KnowledgeAgent (RAG)
│   ├── account.py               # AccountAgent
│   └── tools/
│       ├── search_docs.py       # ChromaDB retrieval, ContextVar trace hook
│       └── account_tools.py     # Build and account data tools
├── rag/
│   └── ingest.py                # CLI: chunk → embed → upsert
├── srop/
│   ├── pipeline.py              # Core orchestration loop
│   └── state.py                 # SessionState model
├── api/
│   ├── routes_sessions.py
│   ├── routes_chat.py
│   ├── routes_traces.py
│   └── errors.py                # HelixError + RFC 7807 handler
├── db/
│   ├── models.py                # SQLAlchemy ORM models
│   └── session.py               # Async engine + get_db dependency
└── obs/
    └── logging.py               # structlog JSON logging

tests/
├── conftest.py                  # In-memory DB, async client, mock_adk
├── test_api.py                  # Integration tests
└── test_retriever.py            # Unit tests — chunker + metadata
```
