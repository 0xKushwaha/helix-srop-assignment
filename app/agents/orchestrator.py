"""
SROP Root Orchestrator — Google ADK agent.

Routes every user turn to KnowledgeAgent or AccountAgent via ADK's AgentTool.
The LLM decides which tool to call — no string parsing.

State persistence: Pattern 3 — SessionState is loaded from DB and injected into
the root agent's instruction at runtime. This is the lightest approach: no full
message history replay, no custom BaseSessionService. Each turn gets a fresh
InMemoryRunner with the current user context baked into the system prompt.
Tradeoff: prior conversation turns are not visible to the agent across requests,
but the critical stateful fields (plan_tier, last_agent, turn_count) persist.
"""
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from app.agents.account import account_agent
from app.agents.knowledge import knowledge_agent

ROOT_INSTRUCTION = """
You are the Helix Support Concierge — a routing agent.
Call the correct specialist tool based on the user's intent.

Routing rules:
- HOW to do something, WHAT something is, docs/feature questions → knowledge_agent
- Their builds, pipelines, account status, plan tier, usage → account_agent
- Greetings, thanks, off-topic → respond directly, no tool call

Always call a tool when intent matches. Never answer knowledge or account questions yourself.
User context is provided above — use it when relevant (e.g. mentioning their plan_tier).
"""


def build_root_agent(user_id: str, plan_tier: str, last_agent: str | None, turn_count: int) -> LlmAgent:
    """Build root agent with current session state injected into the instruction."""
    context_block = f"""
Current user context:
- user_id: {user_id}
- plan_tier: {plan_tier}
- last_agent_used: {last_agent or "none"}
- conversation_turn: {turn_count + 1}
"""
    return LlmAgent(
        name="srop_root",
        model="gemini-2.0-flash",
        instruction=context_block + "\n" + ROOT_INSTRUCTION,
        tools=[
            AgentTool(agent=knowledge_agent),
            AgentTool(agent=account_agent),
        ],
    )
