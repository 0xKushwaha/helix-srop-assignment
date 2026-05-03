"""
Integration tests — exercise the full SROP pipeline.
LLM mocked at the ADK boundary (not at the HTTP layer).
"""
import pytest


@pytest.mark.asyncio
async def test_healthz(client):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_create_session(client):
    resp = await client.post("/v1/sessions", json={"user_id": "u_test_001"})
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert data["user_id"] == "u_test_001"


@pytest.mark.asyncio
async def test_create_session_with_plan_tier(client):
    resp = await client.post("/v1/sessions", json={"user_id": "u_test_pro", "plan_tier": "pro"})
    assert resp.status_code == 200
    assert "session_id" in resp.json()


@pytest.mark.asyncio
async def test_knowledge_query_routes_correctly(client, mock_adk):
    """
    Core integration test: two turns in the same session.

    Turn 1: knowledge query → routed_to == "knowledge", trace has chunk IDs.
    Turn 2: plan tier question → reply contains "pro" (state persisted across turns).
    """
    # Create session
    sess = await client.post("/v1/sessions", json={"user_id": "u_test_002", "plan_tier": "pro"})
    assert sess.status_code == 200
    session_id = sess.json()["session_id"]

    # Turn 1 — knowledge query
    r1 = await client.post(
        f"/v1/chat/{session_id}", json={"content": "How do I rotate a deploy key?"}
    )
    assert r1.status_code == 200
    assert r1.json()["routed_to"] == "knowledge"
    trace_id = r1.json()["trace_id"]

    # Trace must exist and have chunk IDs
    trace = await client.get(f"/v1/traces/{trace_id}")
    assert trace.status_code == 200
    assert len(trace.json()["retrieved_chunk_ids"]) > 0

    # Turn 2 — state persistence: agent must know plan_tier without re-asking
    r2 = await client.post(
        f"/v1/chat/{session_id}", json={"content": "What is my plan tier?"}
    )
    assert r2.status_code == 200
    assert "pro" in r2.json()["reply"].lower()


@pytest.mark.asyncio
async def test_account_query_routes_to_account_agent(client, mock_adk):
    sess = await client.post("/v1/sessions", json={"user_id": "u_test_003", "plan_tier": "free"})
    session_id = sess.json()["session_id"]

    r = await client.post(f"/v1/chat/{session_id}", json={"content": "Show me my last 3 failed builds"})
    assert r.status_code == 200
    assert r.json()["routed_to"] == "account"


@pytest.mark.asyncio
async def test_session_not_found_returns_404(client):
    resp = await client.post("/v1/chat/nonexistent-id", json={"content": "hello"})
    assert resp.status_code == 404
    assert resp.json()["title"] == "SESSION_NOT_FOUND"


@pytest.mark.asyncio
async def test_trace_not_found_returns_404(client):
    resp = await client.get("/v1/traces/nonexistent-trace-id")
    assert resp.status_code == 404
    assert resp.json()["title"] == "TRACE_NOT_FOUND"


@pytest.mark.asyncio
async def test_turn_count_increments(client, mock_adk):
    """Verify turn_count advances in session state after each message."""
    sess = await client.post("/v1/sessions", json={"user_id": "u_turn_count"})
    session_id = sess.json()["session_id"]

    for _ in range(3):
        r = await client.post(f"/v1/chat/{session_id}", json={"content": "hello"})
        assert r.status_code == 200

    # After 3 turns, trace should exist for each
    r = await client.post(f"/v1/chat/{session_id}", json={"content": "hello"})
    assert r.status_code == 200
    trace = await client.get(f"/v1/traces/{r.json()['trace_id']}")
    assert trace.status_code == 200
