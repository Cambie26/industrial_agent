"""
Retrieval over the FAA Powerplant Handbook: parse -> chunk -> index -> search.

Heavy dependencies (PyMuPDF, sentence-transformers, rank-bm25) live in the
`[rag]` extra and are imported inside functions, so this subpackage - and
`industrial_agent` as a whole - imports without them.
"""

from .search import SearchHit, search

__all__ = ["SearchHit", "search"]
