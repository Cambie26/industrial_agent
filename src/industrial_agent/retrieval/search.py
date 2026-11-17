"""
Query -> ranked chunks: hybrid dense + BM25 search with a cross-encoder rerank.

The pipeline, each stage switchable so `eval/run_eval.py` can ablate it:
  1. Filter: with `turbine_only`, reciprocating-engine chunks are removed
     before ranking (general and turbine chunks stay).
  2. Retrieve: dense cosine similarity on BGE embeddings, and BM25. Each
     keeps its top CANDIDATES_PER_RETRIEVER.
  3. Fuse: reciprocal rank fusion of the two lists (`hybrid=False` uses the
     dense list alone).
  4. Rerank: the top RERANK_CANDIDATES fused hits are rescored by a BGE
     cross-encoder, which reads query and passage together.

Before touching this, know that:
  - A hit's score means different things depending on the last stage that
    ran: a 0-1 relevance from the reranker (sigmoid of its logit), an RRF
    score, or a cosine similarity. Only the reranker score is meant to be
    compared against a fixed "is this relevant enough" threshold; the agent's
    refusal rule (PR 5) relies on that. `SearchHit.scored_by` says which.
  - The models are loaded once per process and cached. Tests replace
    `query_embedder` and `cross_encoder` with fakes rather than loading them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from .chunk import Chunk
from .index import INDEX_DIR, QUERY_PREFIX, Index, load_index, tokenise

RERANKER_MODEL = "BAAI/bge-reranker-base"

# Standard reciprocal-rank-fusion constant.
RRF_K = 60

# Candidates pulled from each retriever before fusion.
CANDIDATES_PER_RETRIEVER = 50

# Fused candidates passed to the cross-encoder; it is the slow step.
RERANK_CANDIDATES = 20

ScoredBy = Literal["rerank", "rrf", "dense"]


@dataclass
class SearchHit:
    """One search result, the score it was ranked by, and which stage gave it."""

    chunk: Chunk
    score: float
    scored_by: ScoredBy


@lru_cache(maxsize=2)
def cached_index(index_dir: str) -> Index:
    """The index in `index_dir`, loaded once per process."""
    return load_index(index_dir)


@lru_cache(maxsize=2)
def query_embedder(model_name: str) -> Callable[[str], Any]:
    """A function that embeds one query with BGE's query prefix, L2-normalised.
    Loaded once per model and process."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(query: str) -> Any:
        return model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]

    return embed


@lru_cache(maxsize=2)
def cross_encoder(model_name: str) -> Callable[[str, list[str]], list[float]]:
    """A function that scores (query, passage) pairs with a cross-encoder,
    returning raw logits. Loaded once per model and process."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)

    def score(query: str, passages: list[str]) -> list[float]:
        return [float(logit) for logit in model.predict(
            [(query, passage) for passage in passages])]

    return score


def allowed_positions(index: Index, turbine_only: bool) -> list[int]:
    """Positions of the chunks that survive the engine-type filter."""
    return [position for position, chunk in enumerate(index.chunks)
            if not turbine_only or chunk.engine_type != "reciprocating"]


def top_positions(scores: Any, allowed: list[int], limit: int) -> list[int]:
    """The `limit` allowed positions with the highest scores, best first."""
    return sorted(allowed, key=lambda position: -float(scores[position]))[:limit]


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = RRF_K
) -> dict[int, float]:
    """Fuse ranked position lists: each list adds 1 / (k + rank) to a position.

    Returns position -> fused score; rank starts at 1.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, position in enumerate(ranking, start=1):
            fused[position] = fused.get(position, 0.0) + 1.0 / (k + rank)
    return fused


def sigmoid(logit: float) -> float:
    """Logistic function, turning a reranker logit into a 0-1 relevance."""
    return 1.0 / (1.0 + math.exp(-logit))


def search(
    query: str,
    top_k: int = 5,
    hybrid: bool = True,
    rerank: bool = True,
    turbine_only: bool = True,
    index: Index | None = None,
) -> list[SearchHit]:
    """Search the maintenance handbook.

    Args:
        query: the question, in plain language.
        top_k: number of hits to return.
        hybrid: fuse BM25 with dense retrieval; False is dense only.
        rerank: rerank the fused candidates with the cross-encoder.
        turbine_only: drop chunks tagged as reciprocating-engine content.
        index: a loaded index; loaded from `data/index/` if not given.

    Returns:
        Up to `top_k` hits, best first.
    """
    if not query.strip():
        raise ValueError("Empty query.")
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}.")
    index = index or cached_index(str(Path(INDEX_DIR)))

    allowed = allowed_positions(index, turbine_only)
    if not allowed:
        raise ValueError("No chunks left after the engine-type filter.")

    # Rank every allowed chunk by cosine similarity to the query.
    query_vector = query_embedder(index.embedding_model)(query)
    dense_scores = index.embeddings @ query_vector
    dense_ranking = top_positions(dense_scores, allowed, CANDIDATES_PER_RETRIEVER)

    # Fuse with BM25, or keep the dense ranking on its own.
    if hybrid:
        bm25_scores = index.bm25.get_scores(tokenise(query))
        bm25_ranking = top_positions(bm25_scores, allowed, CANDIDATES_PER_RETRIEVER)
        fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking])
        ranked = sorted(fused, key=lambda position: -fused[position])
        hits = [SearchHit(index.chunks[position], fused[position], "rrf")
                for position in ranked]
    else:
        hits = [SearchHit(index.chunks[position], float(dense_scores[position]),
                          "dense")
                for position in dense_ranking]

    if not rerank:
        return hits[:top_k]

    # Rescore the best fused candidates with the cross-encoder.
    candidates = hits[:RERANK_CANDIDATES]
    passages = [hit.chunk.text for hit in candidates]
    logits = cross_encoder(RERANKER_MODEL)(query, passages)
    reranked = [SearchHit(hit.chunk, sigmoid(logit), "rerank")
                for hit, logit in zip(candidates, logits, strict=True)]
    reranked.sort(key=lambda hit: -hit.score)
    return reranked[:top_k]
