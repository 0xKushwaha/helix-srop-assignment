# Helix SROP — Ayush Kushwaha

**Stateful RAG Orchestration Pipeline** — AI Support Concierge for Helix, a B2B dev-tools platform.

Handles two workflows in a single ongoing conversation:
- **Knowledge questions** ("How do I rotate a deploy key?") → answered via RAG over product docs
- **Account lookups** ("Show my last 3 failed builds") → answered via internal tools

Built with **FastAPI · Google ADK · SQLite (SQLAlchemy 2.x async) · ChromaDB**.

---

## Quick Start (< 5 minutes)

### 1. Clone and install

```bash
git clone <your-repo-url>
cd helix-srop-assignment
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Open .env and set GOOGLE_API_KEY=your-key-here
```

Get a free API key at [aistudio.google.com](https://aistudio.google.com).

### 3. Ingest product docs into the vector store

```bash
python -m app.rag.ingest --path docs/
# Found 11 markdown files in docs/
# Ingest complete. Total chunks: 87
```

### 4. Start the server

```bash
uvicorn app.main:app --reload
# Uvicorn running on http://127.0.0.1:8000
```

### 5. Try it

```bash
# Create a session
SESSION=$(curl -s -X POST http://localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u_demo", "plan_tier": "pro"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

# Ask a knowledge question
curl -s -X POST http://localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content": "How do I rotate a deploy key?"}' | python3 -m json.tool

# Ask an account question
curl -s -X POST http://localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content": "Show me my last 3 failed builds"}' | python3 -m json.tool

# Healthcheck
curl http://localhost:8000/healthz
```

---

## Running Tests

```bash
pytest -q
# 13 passed, 1 skipped
```

The skipped test (`test_search_docs_returns_results_with_chunk_ids`) requires a seeded vector store. After running ingest, all 14 tests pass.

LLM is mocked at the ADK boundary (`_call_adk`) so tests run instantly without hitting the Gemini API.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/sessions` | Create session. Body: `{"user_id": str, "plan_tier": "free"\|"pro"\|"enterprise"}` |
| `POST` | `/v1/chat/{session_id}` | Send message. Body: `{"content": str}`. Returns `{reply, routed_to, trace_id}` |
| `GET` | `/v1/traces/{trace_id}` | Fetch structured trace for one turn |
| `GET` | `/healthz` | Health check |

**Error responses** follow [RFC 7807](https://datatracker.ietf.org/doc/html/rfc7807):
```json
{"type": "...", "title": "SESSION_NOT_FOUND", "status": 404, "detail": "..."}
```

---

## Architecture

```
POST /v1/chat/{session_id}
         │
         ▼
┌─────────────────────────────────────────┐
│  pipeline.run()                         │
│  1. Load SessionState from SQLite       │
│  2. Build root agent (state in prompt)  │
│  3. asyncio.wait_for → ADK run          │
│  4. Collect events (routing + tools)    │
│  5. Persist state + messages + trace    │
└────────────────┬────────────────────────┘
                 │  AgentTool (LLM-driven routing)
       ┌─────────┴──────────┐
       ▼                    ▼
 KnowledgeAgent        AccountAgent
 └─ search_docs()      └─ get_recent_builds()
       │                   get_account_status()
       ▼
 ChromaDB (cosine)
 text-embedding-004
```

```
SQLite schema
├── users          (user_id PK, plan_tier, created_at)
├── sessions       (session_id PK, user_id FK, state JSON, timestamps)
├── messages       (message_id PK, session_id FK, role, content, trace_id)
└── agent_traces   (trace_id PK, session_id, routed_to, tool_calls JSON,
                    retrieved_chunk_ids JSON, latency_ms)
```

---

## Design Decisions

### State Persistence — Pattern 3 (SessionState injection)

I used **Pattern 3**: store only `SessionState` (user_id, plan_tier, last_agent, turn_count) in the DB as a JSON column and inject it into the root agent's system instruction on every turn.

**Why:** The agent needs user context (plan tier, who they are) to route and respond correctly — not full conversation history. Pattern 3 is the lightest approach: no custom `BaseSessionService`, no full message history replay, no context window waste. Each turn creates a fresh `InMemoryRunner` with the user context baked into the instruction.

**Tradeoff:** Prior conversation turns aren't visible to the agent. Users can't say "tell me more about step 2" across turns. Acceptable for a support concierge focused on routing; a chat assistant would need Pattern 2.

**Why state survives restarts:** `SessionState` is serialized to SQLite on every commit. On server restart, the next request loads it from DB — no in-memory state is lost.

### Chunking — Heading-aware Markdown

Split on `# ## ###` headings to keep each section coherent, then sub-chunk long sections with fixed-size + overlap (512 chars, 64 overlap).

**Why:** The `docs/` directory is structured markdown with clear heading boundaries. Heading-aware chunking ensures a chunk about "Rotating a Deploy Key" stays together instead of being cut mid-procedure. Better retrieval relevance than pure character splitting.

### Vector Store — ChromaDB + Google text-embedding-004

ChromaDB with cosine similarity, persisted to `./chroma_db`. Embedding model: Google `text-embedding-004` (768-dim).

**Why:** Simplest persistent vector store — no server, single `PersistentClient` call. Consistent with the Gemini ADK stack. Stable chunk IDs (SHA-256 of `filepath::index`) prevent duplicates on re-ingest.

---

## Project Structure

```
app/
├── main.py                    # FastAPI app, lifespan, error handlers
├── settings.py                # Pydantic Settings (reads .env)
├── agents/
│   ├── orchestrator.py        # Root agent with AgentTool routing
│   ├── knowledge.py           # KnowledgeAgent (RAG)
│   ├── account.py             # AccountAgent (builds/status)
│   └── tools/
│       ├── search_docs.py     # ChromaDB retrieval tool
│       └── account_tools.py   # Mock build/account data tools
├── rag/
│   └── ingest.py              # CLI: chunk → embed → upsert to ChromaDB
├── srop/
│   ├── pipeline.py            # Core: load state → ADK → save state+trace
│   └── state.py               # SessionState Pydantic model
├── api/
│   ├── routes_sessions.py     # POST /v1/sessions
│   ├── routes_chat.py         # POST /v1/chat/{session_id}
│   ├── routes_traces.py       # GET /v1/traces/{trace_id}
│   └── errors.py              # HelixError hierarchy + RFC 7807 handler
├── db/
│   ├── models.py              # SQLAlchemy ORM models
│   └── session.py             # Async engine + get_db dependency
└── obs/
    └── logging.py             # structlog JSON logging

tests/
├── conftest.py                # Fixtures: in-memory DB, async client, mock_adk
├── test_api.py                # Integration tests (8 tests)
└── test_retriever.py          # Unit tests for chunker + metadata (6 tests)

docs/                          # Product docs ingested into ChromaDB
```

---

## Known Limitations

- **No cross-turn conversation context** — agent sees user state but not prior messages. Follow-up questions like "tell me more" won't work across turns.
- **Mock account data** — `get_recent_builds` and `get_account_status` return hardcoded data.
- **No authentication** — all endpoints are open.
- **Single-file SQLite** — not suitable for multi-process deployments without a shared DB.

---

## What I'd Do With More Time

- **E2 Escalation agent** — `create_ticket` tool writing to a `tickets` table (already scaffolded)
- **E3 SSE streaming** — `StreamingResponse` yielding ADK events as server-sent events
- **E6 Docker** — `docker-compose.yml` with app + ingest as a one-command setup
- **Pattern 2 state** — re-hydrate message history for multi-turn contextual conversations
- **Score threshold filtering** — discard chunks below 0.6 cosine similarity

---

## Time Breakdown

| Phase | Time |
|-------|------|
| Setup + DB + FastAPI routes | ~45 min |
| RAG ingest + search_docs tool | ~45 min |
| ADK agents (knowledge, account, orchestrator) | ~30 min |
| pipeline.py + state persistence | ~40 min |
| Tests + conftest mock | ~30 min |
| README + docs | ~10 min |
| **Total** | **~3.5 hrs** |

---

## Extensions Completed

- [ ] E1: Idempotency (`Idempotency-Key` header)
- [ ] E2: Escalation agent
- [ ] E3: Streaming SSE
- [ ] E4: LLM-as-judge reranker
- [ ] E5: Guardrails (out-of-scope refusal + PII redaction)
- [ ] E6: Docker + docker-compose
- [ ] E7: Eval harness
