"""
Blocks -> chunks: structure-aware chunking of the parsed handbook.

Section boundaries come from each chapter's contents (see
`parse.table_of_contents`), never from a fixed window. Tables are always
their own chunk. Every chunk carries the metadata that search filters and
the agent's citations rely on: chapter, section, page and engine_type.

Not built yet - this is PR 3. The signatures here are the contract the index
and search modules are written against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .parse import Block, BlockKind

# Upper bound on a body-text chunk before a section is split further.
MAX_CHUNK_TOKENS = 400

# Words that mark a passage as turbine (vs reciprocating) content.
TURBINE_KEYWORDS = (
    "turbine", "turbofan", "turbojet", "turboprop", "turboshaft",
    "compressor", "combustor", "nozzle", "hpc", "lpc", "hpt", "lpt", "egt",
)
RECIPROCATING_KEYWORDS = (
    "piston", "cylinder", "crankshaft", "carburettor", "carburetor",
    "magneto", "spark plug", "valve train",
)

EngineType = Literal["turbine", "reciprocating", "general"]


@dataclass
class Chunk:
    """One retrievable passage, with the metadata needed to filter and cite it."""

    chunk_id: str
    text: str
    chapter: int
    section: str
    page: int
    kind: BlockKind
    engine_type: EngineType


def tag_engine_type(text: str) -> EngineType:
    """Classify a passage as turbine, reciprocating or general content."""
    raise NotImplementedError("tag_engine_type is PR 3 (chunking and index).")


def chunk_blocks(blocks: list[Block], toc: list[tuple[int, str, int]]) -> list[Chunk]:
    """Group one chapter's blocks into section-aligned chunks; tables stand alone."""
    raise NotImplementedError("chunk_blocks is PR 3 (chunking and index).")
