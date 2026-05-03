"""
Guardrails (E5) — out-of-scope refusal + PII redaction.

Two-stage protection:
  1. Inbound: redact PII from user messages BEFORE sending to the LLM.
  2. Outbound: redact PII from agent replies BEFORE returning to the user.

Out-of-scope refusal is handled in the orchestrator instruction; this module
exposes a quick keyword check used as a guardrail for traceability.
"""
import re

# Order matters — credit cards before generic 16-digit numbers, etc.
_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[CARD]"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b(?:sk|pk|api)[-_][A-Za-z0-9]{16,}\b"), "[API_KEY]"),
]

# Topics outside the Helix support scope — agent must refuse instead of guessing.
_OUT_OF_SCOPE_PATTERNS = [
    re.compile(r"\b(weather|stock\s+price|sports\s+score|recipe|joke|poem)\b", re.IGNORECASE),
]


def redact_pii(text: str) -> str:
    """Replace PII with structural placeholders. Idempotent."""
    if not text:
        return text
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def is_out_of_scope(text: str) -> bool:
    """Return True if the message is clearly outside the Helix support scope."""
    return any(p.search(text) for p in _OUT_OF_SCOPE_PATTERNS)


OUT_OF_SCOPE_REPLY = (
    "I'm the Helix Support Concierge, so I can only help with questions about "
    "Helix products, your account, or your builds. For anything else, please "
    "use a different assistant."
)
