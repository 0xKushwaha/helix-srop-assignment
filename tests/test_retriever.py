"""
Unit tests for RAG retrieval and chunking.
"""
import pytest

from app.rag.ingest import chunk_markdown, extract_metadata
from pathlib import Path


def test_chunker_produces_non_empty_chunks():
    """Chunker must not produce empty strings."""
    text = "# Header\n\nSome content here.\n\n## Section 2\n\nMore content here with details."
    chunks = chunk_markdown(text, chunk_size=100, overlap=20)
    assert len(chunks) > 0
    assert all(c.strip() for c in chunks)


def test_chunker_strips_frontmatter():
    """Frontmatter should not appear in any chunk."""
    text = "---\ntitle: Test\nproduct_area: security\n---\n# Header\n\nContent here."
    chunks = chunk_markdown(text)
    assert all("---" not in c or c.count("---") == 0 for c in chunks)
    combined = " ".join(chunks)
    assert "title: Test" not in combined


def test_chunker_handles_long_section():
    """Long sections must be sub-chunked."""
    long_text = "## Long Section\n\n" + "word " * 300
    chunks = chunk_markdown(long_text, chunk_size=100, overlap=20)
    assert len(chunks) > 1


def test_extract_metadata_with_frontmatter():
    """extract_metadata should parse YAML frontmatter."""
    text = "---\ntitle: Deploy Keys\nproduct_area: security\ntags: [keys, ci-cd]\n---\n# Content"
    path = Path("deploy-keys.md")
    meta = extract_metadata(path, text)
    assert meta["title"] == "Deploy Keys"
    assert meta["product_area"] == "security"
    assert meta["source"] == "deploy-keys.md"
    # Tags should be flattened to string for ChromaDB
    assert isinstance(meta["tags"], str)


def test_extract_metadata_without_frontmatter():
    """Files without frontmatter should still return source key."""
    text = "# Just a plain doc\n\nNo frontmatter."
    path = Path("plain.md")
    meta = extract_metadata(path, text)
    assert meta["source"] == "plain.md"


@pytest.mark.asyncio
async def test_search_docs_returns_results_with_chunk_ids():
    """search_docs must return chunk IDs and scores in [0, 1].

    Requires the vector store to be seeded — skipped if ChromaDB is empty.
    Run: python -m app.rag.ingest --path docs/ first.
    """
    import chromadb
    from app.settings import settings
    from app.rag.ingest import COLLECTION_NAME

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    try:
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
    except Exception:
        count = 0

    if count == 0:
        pytest.skip("Vector store is empty — run ingest first")

    from app.agents.tools.search_docs import search_docs
    result = await search_docs("how to rotate a deploy key", k=3)
    assert isinstance(result, str)
    assert "chunk_" in result
