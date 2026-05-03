"""
SROP pipeline — called by the chat route.

State persistence: Pattern 3.
  - Load SessionState from DB each turn.
  - Load last N messages and inject as conversation history into the root
    agent's instruction so context carries across turns.
  - After the turn, write updated state + messages + trace back to DB.
  - ADK session is ephemeral (InMemoryRunner); cross-turn state lives in DB.

Timeout: asyncio.wait_for wraps the entire ADK generator iteration.
"""
import asyncio
import time
import uuid
from dataclasses import dataclass

import structlog
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.orchestrator import (
    ACCOUNT_AGENT_NAME,
    ESCALATION_AGENT_NAME,
    KNOWLEDGE_AGENT_NAME,
    build_root_agent,
)
from app.agents.tools.escalation_tools import reset_ticket_context, set_ticket_context
from app.agents.tools.search_docs import _chunk_ids_var
from app.api.errors import SessionNotFoundError, UpstreamTimeoutError
from app.db.models import AgentTrace, Message, Session
from app.settings import settings
from app.srop.guardrails import OUT_OF_SCOPE_REPLY, is_out_of_scope, redact_pii
from app.srop.state import SessionState

log = structlog.get_logger()

HISTORY_WINDOW = 10  # number of prior messages to re-hydrate into context


@dataclass
class PipelineResult:
    content: str
    routed_to: str
    trace_id: str


async def _load_history(session_id: str, db: AsyncSession) -> list[dict]:
    """Load the last HISTORY_WINDOW messages for context re-hydration."""
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(HISTORY_WINDOW)
    )
    messages = result.scalars().all()
    return [{"role": m.role, "content": m.content} for m in reversed(messages)]


async def _call_adk(
    session_id: str,
    user_message: str,
    state: SessionState,
    history: list[dict],
    db: AsyncSession | None = None,
) -> tuple[str, str, list[dict], list[str]]:
    """Run one ADK turn. Returns (reply, routed_to, tool_calls, chunk_ids)."""
    agent = build_root_agent(
        user_id=state.user_id,
        plan_tier=state.plan_tier,
        last_agent=state.last_agent,
        turn_count=state.turn_count,
        history=history,
    )
    runner = InMemoryRunner(agent=agent, app_name="helix_srop")

    new_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_message)],
    )

    routed_to = "smalltalk"
    tool_calls: list[dict] = []
    chunk_ids: list[str] = []
    reply_text = ""

    chunk_token = _chunk_ids_var.set(chunk_ids)
    ticket_token = set_ticket_context(db, session_id, state.user_id) if db is not None else None
    try:
        async for event in runner.run_async(
            user_id=state.user_id,
            session_id=session_id,
            new_message=new_message,
        ):
            for fc in event.get_function_calls():
                tool_calls.append({
                    "tool_name": fc.name,
                    "args": dict(fc.args or {}),
                    "result": None,
                })

            if event.is_final_response():
                if event.content and event.content.parts:
                    reply_text = event.content.parts[0].text or ""
                author = event.author or ""
                if author == KNOWLEDGE_AGENT_NAME:
                    routed_to = "knowledge"
                elif author == ACCOUNT_AGENT_NAME:
                    routed_to = "account"
                elif author == ESCALATION_AGENT_NAME:
                    routed_to = "escalation"
                else:
                    routed_to = "smalltalk"
    finally:
        _chunk_ids_var.reset(chunk_token)
        if ticket_token is not None:
            reset_ticket_context(ticket_token)

    return reply_text, routed_to, tool_calls, chunk_ids


async def run(session_id: str, user_message: str, db: AsyncSession) -> PipelineResult:
    trace_id = str(uuid.uuid4())

    result = await db.execute(select(Session).where(Session.session_id == session_id))
    db_session = result.scalar_one_or_none()
    if db_session is None:
        raise SessionNotFoundError(f"Session {session_id} not found")

    state = SessionState.from_db_dict(db_session.state)
    history = await _load_history(session_id, db)

    structlog.contextvars.bind_contextvars(
        session_id=session_id,
        trace_id=trace_id,
        user_id=state.user_id,
    )
    log.info("pipeline_started", turn=state.turn_count + 1, message_len=len(user_message))

    # Inbound guardrail — redact PII before anything reaches the LLM.
    redacted_message = redact_pii(user_message)

    start_ms = time.monotonic()

    # Out-of-scope shortcut — skip the LLM entirely for clearly off-topic queries.
    if is_out_of_scope(redacted_message):
        log.info("guardrail_out_of_scope")
        reply = OUT_OF_SCOPE_REPLY
        routed_to = "guardrail"
        tool_calls: list[dict] = []
        chunk_ids: list[str] = []
        latency_ms = int((time.monotonic() - start_ms) * 1000)
    else:
        try:
            reply, routed_to, tool_calls, chunk_ids = await asyncio.wait_for(
                _call_adk(session_id, redacted_message, state, history, db),
                timeout=settings.llm_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise UpstreamTimeoutError(
                f"LLM did not respond within {settings.llm_timeout_seconds}s"
            )

        # Outbound guardrail — strip PII the model may have echoed back.
        reply = redact_pii(reply)
        latency_ms = int((time.monotonic() - start_ms) * 1000)

    state.turn_count += 1
    state.last_agent = routed_to  # type: ignore[assignment]
    db_session.state = state.to_db_dict()
    db.add(db_session)

    db.add(Message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="user",
        content=redacted_message,
        trace_id=trace_id,
    ))
    db.add(Message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="assistant",
        content=reply,
        trace_id=trace_id,
    ))
    db.add(AgentTrace(
        trace_id=trace_id,
        session_id=session_id,
        routed_to=routed_to,
        tool_calls=tool_calls,
        retrieved_chunk_ids=chunk_ids,
        latency_ms=latency_ms,
    ))

    await db.commit()
    log.info("pipeline_done", routed_to=routed_to, latency_ms=latency_ms)

    return PipelineResult(content=reply, routed_to=routed_to, trace_id=trace_id)
