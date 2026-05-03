"""
POST /v1/chat/{session_id} — send a user message, get assistant reply.

Idempotency (E1): if the client supplies an `Idempotency-Key` header, the
same key + same request body returns the cached response. Different body
under the same key returns 409 IDEMPOTENCY_CONFLICT.
"""
import hashlib
import json

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import IdempotencyConflictError
from app.db.models import IdempotencyKey
from app.db.session import get_db
from app.srop import pipeline

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    content: str


class ChatResponse(BaseModel):
    reply: str
    routed_to: str
    trace_id: str


def _hash_request(session_id: str, body: ChatRequest) -> str:
    payload = json.dumps({"session_id": session_id, "content": body.content}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


@router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(
    session_id: str,
    body: ChatRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> ChatResponse:
    """
    Run one turn of the SROP pipeline.

    Errors:
      404 SESSION_NOT_FOUND   — session does not exist
      504 UPSTREAM_TIMEOUT    — LLM exceeded llm_timeout_seconds
      409 IDEMPOTENCY_CONFLICT — same key, different request body
    """
    request_hash = _hash_request(session_id, body) if idempotency_key else None

    if idempotency_key:
        result = await db.execute(
            select(IdempotencyKey).where(IdempotencyKey.key == idempotency_key)
        )
        cached = result.scalar_one_or_none()
        if cached is not None:
            if cached.request_hash != request_hash:
                raise IdempotencyConflictError(
                    f"Idempotency-Key {idempotency_key!r} was already used with a different request body"
                )
            return ChatResponse(**cached.response_body)

    result = await pipeline.run(session_id, body.content, db)
    response = ChatResponse(
        reply=result.content,
        routed_to=result.routed_to,
        trace_id=result.trace_id,
    )

    if idempotency_key:
        db.add(IdempotencyKey(
            key=idempotency_key,
            session_id=session_id,
            request_hash=request_hash,
            response_body=response.model_dump(),
        ))
        await db.commit()

    return response
