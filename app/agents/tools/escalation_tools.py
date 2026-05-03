"""
Escalation tools (E2) — used by EscalationAgent to file support tickets.

The pipeline sets a ContextVar with (db, session_id, user_id) before running
the agent. The tool reads it at call time, writes a ticket row, and returns
the ticket_id. This avoids passing DB handles through the ADK tool signature.
"""
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from app.db.models import Ticket


@dataclass
class _TicketContext:
    db: object  # AsyncSession — typed loosely to avoid circular imports
    session_id: str
    user_id: str


_ticket_context: ContextVar[_TicketContext | None] = ContextVar(
    "ticket_context", default=None
)


def set_ticket_context(db, session_id: str, user_id: str) -> object:
    """Pipeline calls this before running the agent."""
    return _ticket_context.set(_TicketContext(db=db, session_id=session_id, user_id=user_id))


def reset_ticket_context(token) -> None:
    _ticket_context.reset(token)


async def create_ticket(
    subject: str,
    body: str,
    priority: Literal["low", "normal", "high"] = "normal",
) -> str:
    """Create a Helix support ticket on behalf of the user.

    Use this tool when the user explicitly asks to escalate, file a ticket,
    talk to a human, or report an issue that the knowledge or account agents
    cannot resolve.

    Args:
        subject: Short summary of the issue (under 200 chars).
        body: Full description of the problem the user is reporting.
        priority: One of "low", "normal", or "high". Default is "normal".

    Returns:
        Confirmation string with the new ticket_id.
    """
    ctx = _ticket_context.get(None)
    if ctx is None:
        return "Ticket creation is unavailable — no active session context."

    ticket_id = "tkt_" + uuid.uuid4().hex[:12]
    ticket = Ticket(
        ticket_id=ticket_id,
        session_id=ctx.session_id,
        user_id=ctx.user_id,
        subject=subject[:256],
        body=body,
        priority=priority,
    )
    ctx.db.add(ticket)
    # Flush so the row exists even before the pipeline's final commit.
    await ctx.db.flush()
    return f"Ticket {ticket_id} created with priority={priority}. A support engineer will follow up via email."
