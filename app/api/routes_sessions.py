"""
POST /v1/sessions — create a session.
"""
import uuid
from typing import Literal

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Session, User
from app.db.session import get_db
from app.srop.state import SessionState

log = structlog.get_logger()
router = APIRouter(tags=["sessions"])


class CreateSessionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    plan_tier: Literal["free", "pro", "enterprise"] = "free"


class CreateSessionResponse(BaseModel):
    session_id: str
    user_id: str


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(
    body: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
) -> CreateSessionResponse:
    """Create a new session. Upsert the user if not seen before."""
    # Upsert user
    result = await db.execute(select(User).where(User.user_id == body.user_id))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(user_id=body.user_id, plan_tier=body.plan_tier)
        db.add(user)
    else:
        user.plan_tier = body.plan_tier

    session_id = str(uuid.uuid4())
    initial_state = SessionState(user_id=body.user_id, plan_tier=body.plan_tier)

    session = Session(
        session_id=session_id,
        user_id=body.user_id,
        state=initial_state.to_db_dict(),
    )
    db.add(session)
    await db.commit()

    log.info("session_created", session_id=session_id, user_id=body.user_id, plan_tier=body.plan_tier)
    return CreateSessionResponse(session_id=session_id, user_id=body.user_id)
