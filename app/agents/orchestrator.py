"""
SROP Root Orchestrator — Google ADK agent.

Routes every user turn to KnowledgeAgent or AccountAgent via AgentTool.
The LLM selects the tool — no string parsing.

Performance: KnowledgeAgent and AccountAgent are module-level singletons.
The root agent is rebuilt per turn only to inject the dynamic user context
block (plan_tier, turn count, message history). The sub-agents and their
tools are reused across all turns.

State persistence: Pattern 3 — SessionState is loaded from DB and injected
into the root agent's instruction at runtime. The last N messages from the
DB are included as conversation history so the agent retains context across
turns without needing a custom BaseSessionService.
"""
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from app.agents.account import account_agent
from app.agents.knowledge import knowledge_agent
from app.settings import settings

# Canonical agent names — used for routing detection in pipeline.py
KNOWLEDGE_AGENT_NAME = "knowledge_agent"
ACCOUNT_AGENT_NAME = "account_agent"
ROOT_AGENT_NAME = "srop_root"

# Pre-built AgentTools — reused across all turns (singleton)
_knowledge_tool = AgentTool(agent=knowledge_agent)
_account_tool = AgentTool(agent=account_agent)

ROOT_INSTRUCTION = """You are the Helix Support Concierge — a routing agent.
Call the correct specialist tool based on the user's intent.

Routing rules:
- HOW to do something, WHAT something is, docs/feature questions → knowledge_agent
- Their builds, pipelines, account status, plan tier, usage → account_agent
- Greetings, thanks, off-topic → respond directly without calling a tool

Always call a tool when intent matches. Never answer knowledge or account questions yourself.
Use the user context and conversation history provided above when relevant."""


def build_root_agent(
    user_id: str,
    plan_tier: str,
    last_agent: str | None,
    turn_count: int,
    history: list[dict] | None = None,
) -> LlmAgent:
    """Build the root agent with current session state and message history injected."""
    history_block = ""
    if history:
        lines = []
        for msg in history:
            role = msg["role"].capitalize()
            lines.append(f"{role}: {msg['content']}")
        history_block = "\n\nConversation so far:\n" + "\n".join(lines) + "\n"

    context_block = (
        f"Current user context:\n"
        f"- user_id: {user_id}\n"
        f"- plan_tier: {plan_tier}\n"
        f"- last_agent_used: {last_agent or 'none'}\n"
        f"- conversation_turn: {turn_count + 1}\n"
        f"{history_block}"
    )

    return LlmAgent(
        name=ROOT_AGENT_NAME,
        model=settings.adk_model,
        instruction=context_block + "\n" + ROOT_INSTRUCTION,
        tools=[_knowledge_tool, _account_tool],
    )
