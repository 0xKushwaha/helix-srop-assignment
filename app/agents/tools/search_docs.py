"""
search_docs tool — used by KnowledgeAgent to retrieve relevant doc chunks.

Singletons: genai.Client and ChromaDB collection are initialized once at
module load to avoid per-call overhead.

Uses a contextvars.ContextVar to expose retrieved chunk IDs to the pipeline
for trace recording without coupling the tool to pipeline internals.
"""
import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache

import chromadb
from google import genai
from google.genai import types as genai_types

from app.settings import settings

_chunk_ids_var: ContextVar[list[str] | None] = ContextVar("chunk_ids", default=None)

SCORE_THRESHOLD = 0.45  # discard chunks below this cosine similarity


@dataclass
class DocChunk:
    chunk_id: str
    score: float
    content: str
    metadata: dict


@lru_cache(maxsize=1)
def _get_genai_client() -> genai.Client:
    return genai.Client(api_key=settings.google_api_key)


@lru_cache(maxsize=1)
def _get_collection() -> chromadb.Collection:
    from app.rag.ingest import COLLECTION_NAME
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _embed_query(query: str) -> list[float]:
    response = _get_genai_client().models.embed_content(
        model="models/text-embedding-004",
        contents=query,
        config=genai_types.EmbedContentConfig(task_type="retrieval_query"),
    )
    return list(response.embeddings[0].values)


def _format_chunks(chunks: list[DocChunk]) -> str:
    if not chunks:
        return "No relevant documentation found for that query."
    parts = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        parts.append(
            f"[{chunk.chunk_id}] (score: {chunk.score:.2f}, source: {source})\n{chunk.content}"
        )
    return "\n\n---\n\n".join(parts)


async def search_docs(query: str, k: int = 5) -> str:
    """Search Helix product documentation and return relevant chunks with IDs.

    Args:
        query: The user's question or search query.
        k: Number of top results to return (default 5).

    Returns:
        Formatted string with chunk IDs, scores, and content. Always cite
        the chunk_id (e.g. [chunk_abc123]) in your answer.
    """
    query_embedding = await asyncio.to_thread(_embed_query, query)

    def _query() -> dict:
        return _get_collection().query(query_embeddings=[query_embedding], n_results=k)

    results = await asyncio.to_thread(_query)

    chunks: list[DocChunk] = []
    if results["ids"] and results["ids"][0]:
        for chunk_id, distance, doc, meta in zip(
            results["ids"][0],
            results["distances"][0],
            results["documents"][0],
            results["metadatas"][0],
        ):
            score = round(max(0.0, min(1.0, 1.0 - float(distance))), 4)
            if score >= SCORE_THRESHOLD:
                chunks.append(DocChunk(chunk_id=chunk_id, score=score, content=doc, metadata=meta or {}))

    chunks.sort(key=lambda c: c.score, reverse=True)

    collector = _chunk_ids_var.get(None)
    if collector is not None:
        collector.extend(c.chunk_id for c in chunks)

    return _format_chunks(chunks)
