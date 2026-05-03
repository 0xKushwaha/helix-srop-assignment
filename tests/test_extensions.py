"""
Tests for extensions E1 (idempotency), E2 (escalation), E5 (guardrails).
"""
import pytest
from sqlalchemy import select

from app.db.models import IdempotencyKey, Ticket
from app.srop.guardrails import is_out_of_scope, redact_pii


# ──────────────────────────────────────────────────────────────────────────────
# E5 — Guardrails
# ──────────────────────────────────────────────────────────────────────────────

def test_redact_pii_email():
    assert redact_pii("Contact me at jane@example.com") == "Contact me at [EMAIL]"


def test_redact_pii_ssn():
    assert "[SSN]" in redact_pii("My SSN is 123-45-6789")


def test_redact_pii_credit_card():
    assert "[CARD]" in redact_pii("Card: 4111 1111 1111 1111")


def test_redact_pii_idempotent():
    """Redacting twice yields the same result."""
    once = redact_pii("Email: a@b.com SSN: 111-22-3333")
    twice = redact_pii(once)
    assert once == twice


def test_redact_pii_empty():
    assert redact_pii("") == ""
    assert redact_pii("no pii here") == "no pii here"


def test_out_of_scope_detection():
    assert is_out_of_scope("Tell me a joke")
    assert is_out_of_scope("What's the weather today?")
    assert not is_out_of_scope("How do I rotate a deploy key?")
    assert not is_out_of_scope("Show me my failed builds")


@pytest.mark.asyncio
async def test_out_of_scope_query_skips_llm(client):
    """Out-of-scope queries should be refused without invoking the agent."""
    sess = await client.post("/v1/sessions", json={"user_id": "u_oos"})
    session_id = sess.json()["session_id"]

    r = await client.post(f"/v1/chat/{session_id}", json={"content": "Tell me a joke"})
    assert r.status_code == 200
    assert r.json()["routed_to"] == "guardrail"
    assert "Helix" in r.json()["reply"]


@pytest.mark.asyncio
async def test_pii_redacted_in_persisted_message(client, mock_adk, db):
    """User PII should be redacted before being persisted to messages table."""
    from app.db.models import Message

    sess = await client.post("/v1/sessions", json={"user_id": "u_pii"})
    session_id = sess.json()["session_id"]

    await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "How do I email jane.doe@helix.example?"},
    )

    result = await db.execute(select(Message).where(Message.role == "user"))
    msgs = result.scalars().all()
    assert any("[EMAIL]" in m.content for m in msgs)
    assert not any("jane.doe@helix.example" in m.content for m in msgs)


# ──────────────────────────────────────────────────────────────────────────────
# E2 — Escalation agent
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_escalation_routes_to_escalation_agent(client, mock_adk):
    sess = await client.post("/v1/sessions", json={"user_id": "u_esc"})
    session_id = sess.json()["session_id"]

    r = await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "I want to escalate this issue and file a ticket"},
    )
    assert r.status_code == 200
    assert r.json()["routed_to"] == "escalation"
    assert "tkt_" in r.json()["reply"]


@pytest.mark.asyncio
async def test_escalation_writes_ticket_row(client, mock_adk, db):
    sess = await client.post("/v1/sessions", json={"user_id": "u_ticket"})
    session_id = sess.json()["session_id"]

    await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "Please file a ticket — my deploy is broken"},
    )

    result = await db.execute(select(Ticket).where(Ticket.session_id == session_id))
    tickets = result.scalars().all()
    assert len(tickets) == 1
    assert tickets[0].user_id == "u_ticket"
    assert tickets[0].priority in ("low", "normal", "high")


# ──────────────────────────────────────────────────────────────────────────────
# E1 — Idempotency
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_idempotency_returns_cached_response(client, mock_adk):
    """Same Idempotency-Key + same body returns the cached response."""
    sess = await client.post("/v1/sessions", json={"user_id": "u_idem"})
    session_id = sess.json()["session_id"]

    headers = {"Idempotency-Key": "key-001"}
    body = {"content": "How do I rotate a deploy key?"}

    r1 = await client.post(f"/v1/chat/{session_id}", json=body, headers=headers)
    r2 = await client.post(f"/v1/chat/{session_id}", json=body, headers=headers)

    assert r1.status_code == 200
    assert r2.status_code == 200
    # Same trace_id proves the second call hit the cache, not the pipeline
    assert r1.json()["trace_id"] == r2.json()["trace_id"]
    assert r1.json()["reply"] == r2.json()["reply"]


@pytest.mark.asyncio
async def test_idempotency_conflict_on_different_body(client, mock_adk):
    """Same key + different body returns 409."""
    sess = await client.post("/v1/sessions", json={"user_id": "u_conflict"})
    session_id = sess.json()["session_id"]

    headers = {"Idempotency-Key": "key-002"}
    r1 = await client.post(f"/v1/chat/{session_id}", json={"content": "Question A"}, headers=headers)
    assert r1.status_code == 200

    r2 = await client.post(f"/v1/chat/{session_id}", json={"content": "Question B"}, headers=headers)
    assert r2.status_code == 409
    assert r2.json()["title"] == "IDEMPOTENCY_CONFLICT"


@pytest.mark.asyncio
async def test_idempotency_key_persisted(client, mock_adk, db):
    sess = await client.post("/v1/sessions", json={"user_id": "u_persist"})
    session_id = sess.json()["session_id"]

    await client.post(
        f"/v1/chat/{session_id}",
        json={"content": "hello"},
        headers={"Idempotency-Key": "key-003"},
    )

    result = await db.execute(select(IdempotencyKey).where(IdempotencyKey.key == "key-003"))
    row = result.scalar_one_or_none()
    assert row is not None
    assert row.session_id == session_id


@pytest.mark.asyncio
async def test_no_idempotency_key_means_no_caching(client, mock_adk):
    """Two calls without the header produce two distinct turns."""
    sess = await client.post("/v1/sessions", json={"user_id": "u_no_idem"})
    session_id = sess.json()["session_id"]

    body = {"content": "hello"}
    r1 = await client.post(f"/v1/chat/{session_id}", json=body)
    r2 = await client.post(f"/v1/chat/{session_id}", json=body)

    assert r1.json()["trace_id"] != r2.json()["trace_id"]
