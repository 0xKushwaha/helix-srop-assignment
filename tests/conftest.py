"""
Test fixtures.
"""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base
from app.db.session import get_db
from app.main import app
from app.srop.pipeline import PipelineResult

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db() -> AsyncSession:
    async with TestSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def client(db):
    """Async test client with DB overridden to in-memory SQLite."""
    app.dependency_overrides[get_db] = lambda: db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def mock_adk(monkeypatch):
    """
    Patch pipeline._call_adk at the ADK boundary so tests don't call the real LLM.

    Account keywords are checked first so "plan tier" / "builds" take priority
    over general question words like "what".
    The real pipeline.run() still executes (DB writes, state updates,
    trace recording) — only the LLM call is replaced.
    """
    async def fake_call_adk(
        session_id: str,
        user_message: str,
        state,
    ) -> tuple[str, str, list[dict], list[str]]:
        msg = user_message.lower()
        # Account keywords checked first to avoid "what" swallowing plan-tier queries
        if any(kw in msg for kw in ("plan tier", "plan", "build", "pipeline", "failed", "status", "account")):
            if "plan" in msg or "tier" in msg:
                return (
                    f"Your current plan tier is {state.plan_tier}.",
                    "account",
                    [{"tool_name": "get_account_status", "args": {"user_id": state.user_id}, "result": None}],
                    [],
                )
            return (
                "Your last 3 failed builds were bld_001, bld_002, bld_003.",
                "account",
                [{"tool_name": "get_recent_builds", "args": {"user_id": state.user_id, "limit": 3}, "result": None}],
                [],
            )
        if any(kw in msg for kw in ("rotate", "deploy", "how", "what", "doc", "configure")):
            return (
                "To rotate a deploy key, generate a new key pair with ssh-keygen "
                "and update it in Settings → Security → Deploy Keys. "
                "See [chunk_abc123] for full steps.",
                "knowledge",
                [{"tool_name": "search_docs", "args": {"query": user_message, "k": 5}, "result": None}],
                ["chunk_abc123", "chunk_def456"],
            )
        return ("Hello! How can I help you today?", "smalltalk", [], [])

    monkeypatch.setattr("app.srop.pipeline._call_adk", fake_call_adk)
