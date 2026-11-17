"""
Chunks -> index: BGE embeddings and a BM25 index, built once and saved.

The index is built in Colab by `scripts/build_index.py` and written to
`data/index/` (gitignored): `chunks.json` (the chunks and their metadata),
`embeddings.npy` (one L2-normalised row per chunk, so a dot product is the
cosine similarity) and `manifest.json` (which model made them). BM25 isn't
saved - rebuilding it from the chunk text takes well under a second, and it
saves pickling a third-party object.

Before touching this, know that:
  - numpy, sentence-transformers and rank-bm25 are part of the `[rag]` extra
    and must be imported inside the functions.
  - Queries need BGE's QUERY_PREFIX and passages must not have it. Embedding
    a query lives in search.py, which reads the model name from the manifest
    so the two never drift apart.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .chunk import Chunk

INDEX_DIR = Path("data/index")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_BATCH_SIZE = 32

# BGE retrieval wants this prefix on queries, not on passages.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CHUNKS_FILE = "chunks.json"
EMBEDDINGS_FILE = "embeddings.npy"
MANIFEST_FILE = "manifest.json"

# BM25 tokens: runs of lower-case letters and digits.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

Embedder = Callable[[list[str]], Any]


@dataclass
class Index:
    """Everything search needs: the chunks, their embeddings, and BM25 over them."""

    chunks: list[Chunk]
    embeddings: Any
    bm25: Any
    embedding_model: str


def tokenise(text: str) -> list[str]:
    """Lower-cased word tokens for BM25."""
    return TOKEN_PATTERN.findall(text.lower())


def build_bm25(chunks: list[Chunk]) -> Any:
    """A `rank_bm25.BM25Okapi` over the chunks' text."""
    from rank_bm25 import BM25Okapi

    return BM25Okapi([tokenise(chunk.text) for chunk in chunks])


def bge_embedder(model_name: str = EMBEDDING_MODEL) -> Embedder:
    """A function that embeds passages with a sentence-transformers model,
    L2-normalised. Downloads the model on first use."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> Any:
        return model.encode(texts, batch_size=EMBEDDING_BATCH_SIZE,
                            normalize_embeddings=True, show_progress_bar=True)

    return embed


def build_index(
    chunks: list[Chunk],
    index_dir: Path | str = INDEX_DIR,
    embed: Embedder | None = None,
    embedding_model: str = EMBEDDING_MODEL,
) -> Index:
    """Embed the chunks, build BM25 over them, and save the index to disk.

    Args:
        chunks: every chunk in the corpus.
        index_dir: folder to write to; created if missing, files overwritten.
        embed: texts -> 2-D array of normalised embeddings. Defaults to BGE
            via `bge_embedder(embedding_model)`; tests pass a fake.
        embedding_model: name recorded in the manifest, and the model loaded
            when `embed` isn't given.

    Returns:
        The index, ready to search without loading it back.
    """
    import numpy as np

    if not chunks:
        raise ValueError("No chunks to index.")
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Chunk ids are not unique; did a chapter get chunked twice?")

    # Embed every chunk and check one row came back per chunk.
    embed = embed or bge_embedder(embedding_model)
    embeddings = np.asarray(embed([chunk.text for chunk in chunks]), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError(f"Expected {len(chunks)} embedding rows, got shape "
                         f"{embeddings.shape}.")

    # Write the chunks, the embeddings and a manifest saying what made them.
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / CHUNKS_FILE).write_text(
        json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False))
    np.save(index_dir / EMBEDDINGS_FILE, embeddings)
    (index_dir / MANIFEST_FILE).write_text(json.dumps({
        "embedding_model": embedding_model,
        "chunk_count": len(chunks),
        "dimensions": int(embeddings.shape[1]),
    }, indent=2))

    return Index(chunks, embeddings, build_bm25(chunks), embedding_model)


def load_index(index_dir: Path | str = INDEX_DIR) -> Index:
    """Load a saved index. Raises if `build_index` hasn't been run there."""
    import numpy as np

    index_dir = Path(index_dir)
    missing = [name for name in (CHUNKS_FILE, EMBEDDINGS_FILE, MANIFEST_FILE)
               if not (index_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"No index in {index_dir.resolve()} (missing {', '.join(missing)}). "
            f"Run scripts/build_index.py first."
        )

    manifest = json.loads((index_dir / MANIFEST_FILE).read_text())
    chunks = [Chunk(**fields)
              for fields in json.loads((index_dir / CHUNKS_FILE).read_text())]
    embeddings = np.load(index_dir / EMBEDDINGS_FILE)
    if embeddings.shape[0] != len(chunks):
        raise ValueError(f"{index_dir} is inconsistent: {len(chunks)} chunks but "
                         f"{embeddings.shape[0]} embedding rows. Rebuild it.")
    return Index(chunks, embeddings, build_bm25(chunks), manifest["embedding_model"])
