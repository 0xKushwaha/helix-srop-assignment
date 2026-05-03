"""
SROP entrypoint — called by the chat route.

State persistence: Pattern 3.
  - Load SessionState from DB each turn.
  - Inject it into the root agent's instruction (dynamic system prompt).
  - After the turn, write updated state back to DB.
  - The ADK session is ephemeral (InMemoryRunner) — no cross-turn ADK state.
  - Full message history is stored in the messages table for auditability,
    but not re-hydrated into ADK (keeps context window lean for routing tasks).

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

from app.agents.orchestrator import build_root_agent
from app.agents.tools.search_docs import _chunk_ids_var
from app.api.errors import SessionNotFoundError, UpstreamTimeoutError
from app.db.models import AgentTrace, Message, Session
from app.settings import settings
from app.srop.state import SessionState

log = structlog.get_logger()


@dataclass
class PipelineResult:
    content: str
    routed_to: str
    trace_id: str


async def _call_adk(
    session_id: str,
    user_message: str,
    state: SessionState,
) -> tuple[str, str, list[dict], list[str]]:
    """Run one ADK turn. Returns (reply, routed_to, tool_calls, chunk_ids)."""
    agent = build_root_agent(
        user_id=state.user_id,
        plan_tier=state.plan_tier,
        last_agent=state.last_agent,
        turn_count=state.turn_count,
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

    # Set up chunk ID collector via ContextVar
    token = _chunk_ids_var.set(chunk_ids)
    try:
        async for event in runner.run_async(
            user_id=state.user_id,
            session_id=session_id,
            new_message=new_message,
        ):
            # Capture function calls for trace
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
                if "knowledge" in author:
                    routed_to = "knowledge"
                elif "account" in author:
                    routed_to = "account"
                else:
                    routed_to = "smalltalk"
    finally:
        _chunk_ids_var.reset(token)

    return reply_text, routed_to, tool_calls, chunk_ids


async def run(session_id: str, user_message: str, db: AsyncSession) -> PipelineResult:
    trace_id = str(uuid.uuid4())

    # Load session
    result = await db.execute(select(Session).where(Session.session_id == session_id))
    db_session = result.scalar_one_or_none()
    if db_session is None:
        raise SessionNotFoundError(f"Session {session_id} not found")

    state = SessionState.from_db_dict(db_session.state)
    structlog.contextvars.bind_contextvars(
        session_id=session_id,
        trace_id=trace_id,
        user_id=state.user_id,
    )
    log.info("pipeline_started", turn=state.turn_count + 1, message_len=len(user_message))

    start_ms = time.monotonic()

    try:
        reply, routed_to, tool_calls, chunk_ids = await asyncio.wait_for(
            _call_adk(session_id, user_message, state),
            timeout=settings.llm_timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise UpstreamTimeoutError(
            f"LLM did not respond within {settings.llm_timeout_seconds}s"
        )

    latency_ms = int((time.monotonic() - start_ms) * 1000)

    # Update session state
    state.turn_count += 1
    state.last_agent = routed_to  # type: ignore[assignment]
    db_session.state = state.to_db_dict()
    db.add(db_session)

    # Persist messages
    user_msg = Message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="user",
        content=user_message,
        trace_id=trace_id,
    )
    assistant_msg = Message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="assistant",
        content=reply,
        trace_id=trace_id,
    )
    db.add(user_msg)
    db.add(assistant_msg)

    # Write trace
    trace = AgentTrace(
        trace_id=trace_id,
        session_id=session_id,
        routed_to=routed_to,
        tool_calls=tool_calls,
        retrieved_chunk_ids=chunk_ids,
        latency_ms=latency_ms,
    )
    db.add(trace)

    await db.commit()
    log.info("pipeline_done", routed_to=routed_to, latency_ms=latency_ms)

    return PipelineResult(content=reply, routed_to=routed_to, trace_id=trace_id)
