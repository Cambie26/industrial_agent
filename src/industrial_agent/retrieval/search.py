"""
Query -> ranked chunks: hybrid dense + BM25 search with a cross-encoder rerank.

Dense and BM25 results are fused with reciprocal rank fusion, filtered to
turbine content by default, then reranked. Each switch exists so the
ablations in `eval/run_eval.py` can turn it off.

Not built yet - this is PR 4.
"""

from __future__ import annotations

from dataclasses import dataclass

from .chunk import Chunk
from .index import Index

RERANKER_MODEL = "BAAI/bge-reranker-base"

# Standard reciprocal-rank-fusion constant.
RRF_K = 60

# Candidates pulled from each retriever before fusion and reranking.
CANDIDATES_PER_RETRIEVER = 50


@dataclass
class SearchHit:
    """One search result and the score it was ranked by."""

    chunk: Chunk
    score: float


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
        Hits, best first.
    """
    raise NotImplementedError("search is PR 4 (hybrid search and rerank).")
