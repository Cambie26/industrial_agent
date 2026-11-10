"""
Chunks -> index: BGE embeddings and a BM25 index, built once and saved.

The index is built in Colab by `scripts/build_index.py` and written to
`data/index/` (gitignored). Search loads it rather than rebuilding.

Before touching this, know that sentence-transformers and rank-bm25 are part
of the `[rag]` extra and must be imported inside the functions.

Not built yet - this is PR 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunk import Chunk

INDEX_DIR = Path("data/index")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# BGE retrieval wants this prefix on queries, not on passages.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class Index:
    """Everything search needs: the chunks, their embeddings, and BM25 over them."""

    chunks: list[Chunk]
    embeddings: Any
    bm25: Any


def build_index(chunks: list[Chunk], index_dir: Path | str = INDEX_DIR) -> Index:
    """Embed the chunks, build BM25 over them, and save both to `index_dir`."""
    raise NotImplementedError("build_index is PR 3 (chunking and index).")


def load_index(index_dir: Path | str = INDEX_DIR) -> Index:
    """Load a saved index. Raises if `build_index` hasn't been run."""
    raise NotImplementedError("load_index is PR 3 (chunking and index).")
