"""
KnowledgeAgent — answers product documentation questions via RAG.
"""
from google.adk.agents import LlmAgent

from app.agents.tools.search_docs import search_docs
from app.settings import settings

KNOWLEDGE_INSTRUCTION = """
You are the Helix documentation specialist. You answer questions about Helix features,
configuration, and how-to guides using ONLY the retrieved documentation chunks.

When answering:
1. Always call search_docs first to retrieve relevant context.
2. Base your answer exclusively on the retrieved chunks — do not guess or use training knowledge.
3. Cite chunk IDs in your answer, e.g. "According to [chunk_abc123]..."
4. If the retrieved chunks don't contain the answer, say: "I don't have documentation on that topic."
"""

knowledge_agent = LlmAgent(
    name="knowledge_agent",
    model=settings.adk_model,
    instruction=KNOWLEDGE_INSTRUCTION,
    tools=[search_docs],
)
