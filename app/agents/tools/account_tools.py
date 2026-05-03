"""
Account tools — used by AccountAgent.
Mock data simulates a real DB query for the take-home assignment.
"""
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone


@dataclass
class BuildSummary:
    build_id: str
    pipeline: str
    status: str
    branch: str
    started_at: str
    duration_seconds: int


@dataclass
class AccountStatus:
    user_id: str
    plan_tier: str
    concurrent_builds_used: int
    concurrent_builds_limit: int
    storage_used_gb: float
    storage_limit_gb: float


_MOCK_BUILDS: list[BuildSummary] = [
    BuildSummary("bld_001", "deploy-prod", "failed", "main", "2026-05-02T10:15:00Z", 142),
    BuildSummary("bld_002", "test-suite", "failed", "feature/auth", "2026-05-02T09:30:00Z", 88),
    BuildSummary("bld_003", "test-suite", "failed", "main", "2026-05-01T22:10:00Z", 95),
    BuildSummary("bld_004", "deploy-staging", "passed", "main", "2026-05-01T18:00:00Z", 201),
    BuildSummary("bld_005", "test-suite", "passed", "main", "2026-05-01T17:45:00Z", 76),
]

_PLAN_LIMITS = {
    "free": (1, 5.0),
    "pro": (5, 50.0),
    "enterprise": (20, 500.0),
}


async def get_recent_builds(user_id: str, limit: int = 5) -> str:
    """Return the most recent builds for the user, newest first.

    Args:
        user_id: The user's ID.
        limit: Maximum number of builds to return (default 5).

    Returns:
        JSON string with list of build records including build_id, pipeline,
        status (passed/failed/cancelled), branch, started_at, and duration_seconds.
    """
    builds = _MOCK_BUILDS[:limit]
    return json.dumps([asdict(b) for b in builds], indent=2)


async def get_account_status(user_id: str) -> str:
    """Return current account status including plan tier and usage limits.

    Args:
        user_id: The user's ID.

    Returns:
        JSON string with plan_tier, concurrent_builds_used/limit,
        storage_used_gb, and storage_limit_gb.
    """
    concurrent_limit, storage_limit = _PLAN_LIMITS.get("pro", (5, 50.0))
    status = AccountStatus(
        user_id=user_id,
        plan_tier="pro",
        concurrent_builds_used=2,
        concurrent_builds_limit=concurrent_limit,
        storage_used_gb=12.4,
        storage_limit_gb=storage_limit,
    )
    return json.dumps(asdict(status), indent=2)
