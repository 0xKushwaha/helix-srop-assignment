"""
AccountAgent — handles user account and build queries.
"""
from google.adk.agents import LlmAgent

from app.agents.tools.account_tools import get_account_status, get_recent_builds
from app.settings import settings

ACCOUNT_INSTRUCTION = """
You are the Helix account specialist. You help users check their build history and account status.

When the user asks about builds, pipelines, failures, or account limits, call the appropriate tool.
Always present results in a clear, readable format.
If the user asks about their plan tier, call get_account_status.
"""

account_agent = LlmAgent(
    name="account_agent",
    model=settings.adk_model,
    instruction=ACCOUNT_INSTRUCTION,
    tools=[get_recent_builds, get_account_status],
)
