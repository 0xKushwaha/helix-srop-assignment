"""
RAG ingest CLI.

Usage:
    python -m app.rag.ingest --path docs/
    python -m app.rag.ingest --path docs/ --chunk-size 512 --chunk-overlap 64

Strategy: heading-aware markdown chunking. Splits on ## / ### headings to keep
sections coherent, then sub-chunks long sections by fixed-size with overlap.
This preserves topic boundaries better than pure character splitting for docs.
"""
import argparse
import asyncio
import hashlib
import re
from pathlib import Path
from typing import Any

import chromadb
import yaml
from google import genai
from google.genai import types as genai_types

from app.settings import settings

COLLECTION_NAME = "helix_docs"


def get_chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def make_chunk_id(relative_path: str, chunk_index: int) -> str:
    raw = f"{relative_path}::{chunk_index}"
    return "chunk_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def extract_metadata(file_path: Path, text: str) -> dict[str, Any]:
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        return {"source": file_path.name}
    try:
        meta: dict[str, Any] = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    meta["source"] = file_path.name
    # ChromaDB metadata values must be str/int/float/bool — flatten lists
    if "tags" in meta and isinstance(meta["tags"], list):
        meta["tags"] = ",".join(str(t) for t in meta["tags"])
    return {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}


def chunk_markdown(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    # Strip YAML frontmatter before chunking
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)

    # Split on markdown headings (# ## ###) — keeps each section together
    sections = re.split(r"\n(?=#{1,3} )", text)

    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            # Sub-chunk long sections with fixed-size + overlap
            start = 0
            while start < len(section):
                end = min(start + chunk_size, len(section))
                chunks.append(section[start:end])
                start += chunk_size - overlap

    return [c for c in chunks if c.strip()]


def embed_texts(texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
    client = genai.Client(api_key=settings.google_api_key)
    embeddings: list[list[float]] = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=batch,
            config=genai_types.EmbedContentConfig(task_type=task_type),
        )
        embeddings.extend([list(e.values) for e in response.embeddings])
    return embeddings


async def ingest_directory(docs_path: Path, chunk_size: int, chunk_overlap: int) -> None:
    md_files = list(docs_path.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files in {docs_path}")

    collection = get_chroma_collection()
    total_chunks = 0

    for file_path in md_files:
        text = file_path.read_text(encoding="utf-8")
        metadata = extract_metadata(file_path, text)
        chunks = chunk_markdown(text, chunk_size, chunk_overlap)
        if not chunks:
            continue
        print(f"  {file_path.name}: {len(chunks)} chunks")

        relative = str(file_path.relative_to(docs_path))
        ids = [make_chunk_id(relative, i) for i in range(len(chunks))]
        metadatas = [dict(metadata) for _ in chunks]

        embeddings = await asyncio.to_thread(embed_texts, chunks)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)

    print(f"Ingest complete. Total chunks: {total_chunks}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into the vector store")
    parser.add_argument("--path", type=Path, required=True, help="Directory containing .md files")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    args = parser.parse_args()

    asyncio.run(ingest_directory(args.path, args.chunk_size, args.chunk_overlap))


if __name__ == "__main__":
    main()
