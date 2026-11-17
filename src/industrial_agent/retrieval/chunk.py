"""
Blocks -> chunks: structure-aware chunking of the parsed handbook.

Section boundaries come from the heading blocks `parse.py` finds by font
(level 1 section, 2 subsection, 3 below that) - never from a fixed window,
and not from the PDF bookmarks, which are useless in this handbook. Body
text and captions under one heading path are packed into chunks of up to
MAX_CHUNK_WORDS; a chunk never straddles two headings. Tables are always
chunks of their own.

Every chunk carries what search filters on and the agent cites: chapter,
section, subsection, page range and engine_type. Its text starts with the
heading path ("Turbine Engine Maintenance > Turbine Blade Inspection") so
BM25 and the embeddings both see the context.

Before touching this, know that engine_type is sticky: a level 1 heading
that names an engine type switches it, and everything after inherits it
until the next switch. In Chapter 10 that makes pages 1-43 reciprocating
and 44-65 turbine, which keyword counting per chunk gets wrong for generic
sections like "Engine Ratings".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .parse import Block

# Upper bound on a chunk's body, in words, before it is split further.
MAX_CHUNK_WORDS = 300

# Section name for text that comes before a chapter's first heading.
INTRODUCTION_SECTION = "Introduction"

# Words that mark a passage as turbine or reciprocating content.
TURBINE_KEYWORDS = (
    "turbine", "turbofan", "turbojet", "turboprop", "turboshaft",
    "compressor", "combustor", "nozzle", "hpc", "lpc", "hpt", "lpt", "egt",
)
RECIPROCATING_KEYWORDS = (
    "reciprocating", "piston", "cylinder", "crankshaft", "carburettor",
    "carburetor", "magneto", "spark plug", "valve train",
)

# A level 1 heading containing one of these switches the sticky engine type.
TURBINE_HEADING_PATTERN = re.compile(
    r"\bturbo(?:fan|jet|prop|shaft)\b|\bturbine\b", re.IGNORECASE)
RECIPROCATING_HEADING_PATTERN = re.compile(r"\breciprocating\b", re.IGNORECASE)

# Splits a markdown table row on pipes that aren't escaped.
CELL_SEPARATOR = re.compile(r"(?<!\\)\|")

EngineType = Literal["turbine", "reciprocating", "general"]
ChunkKind = Literal["text", "table"]


@dataclass
class Chunk:
    """One retrievable passage, with the metadata needed to filter and cite it."""

    chunk_id: str
    text: str
    source: str
    chapter: int
    section: str
    subsection: str
    page: int
    page_end: int
    kind: ChunkKind
    engine_type: EngineType

    @property
    def citation(self) -> str:
        """Chapter and printed page(s), e.g. "Ch. 10, p. 10-59" or "pp. 10-59-60"."""
        if self.page_end == self.page:
            return f"Ch. {self.chapter}, p. {self.chapter}-{self.page}"
        return (f"Ch. {self.chapter}, pp. {self.chapter}-{self.page}"
                f"-{self.page_end}")


def keyword_count(text: str, keywords: tuple[str, ...]) -> int:
    """How many times any of `keywords` appears in `text` as a whole word."""
    lowered = text.lower()
    return sum(len(re.findall(rf"\b{re.escape(keyword)}\b", lowered))
               for keyword in keywords)


def tag_engine_type(text: str) -> EngineType:
    """Classify a passage by keyword counts. "general" if neither side wins."""
    turbine_hits = keyword_count(text, TURBINE_KEYWORDS)
    reciprocating_hits = keyword_count(text, RECIPROCATING_KEYWORDS)
    if turbine_hits > reciprocating_hits:
        return "turbine"
    if reciprocating_hits > turbine_hits:
        return "reciprocating"
    return "general"


def heading_engine_type(heading: str) -> EngineType | None:
    """Engine type a section heading switches to, or None if it names neither."""
    if RECIPROCATING_HEADING_PATTERN.search(heading):
        return "reciprocating"
    if TURBINE_HEADING_PATTERN.search(heading):
        return "turbine"
    return None


def split_row(line: str) -> list[str]:
    """Cells of one markdown table row, stripped."""
    return [cell.strip() for cell in CELL_SEPARATOR.split(line.strip())[1:-1]]


def join_row(cells: list[str]) -> str:
    """One markdown table row from its cells."""
    return "| " + " | ".join(cells) + " |"


def tidy_table(markdown: str) -> tuple[list[str], list[list[str]]]:
    """Header and body rows of a markdown table, repaired for retrieval.

    Two neighbouring columns that never both hold text in a row, with no
    header on the right one, are merged - PyMuPDF often splits one column in
    two and puts the header over the wrong half. Then blank leading cells are
    filled from the row above, so a continuation row ("|  | • Overpriming. |")
    still says which trouble it belongs to. Merging has to come first, or the
    fill-down copies values into the half that is about to be merged.
    """
    lines = markdown.strip().splitlines()
    header = split_row(lines[0])
    body = [split_row(line) for line in lines[2:]]

    # Merge each column into its right-hand neighbour where the two never clash.
    column = 0
    while column < len(header) - 1:
        clash = any(row[column] and row[column + 1] for row in body)
        if header[column] and not header[column + 1] and not clash:
            header[column + 1] = header[column]
            for row in body:
                row[column + 1] = row[column] or row[column + 1]
            del header[column]
            for row in body:
                del row[column]
            continue
        column += 1

    # Fill each row's leading blanks from the row above.
    for row_index in range(1, len(body)):
        for column, cell in enumerate(body[row_index]):
            if cell:
                break
            body[row_index][column] = body[row_index - 1][column]
    return header, body


def table_pieces(markdown: str) -> list[str]:
    """A tidied table split by rows into pieces of up to MAX_CHUNK_WORDS words,
    each repeating the header."""
    header, body = tidy_table(markdown)
    header_lines = [join_row(header), "|" + " --- |" * len(header)]
    header_words = len(" ".join(header).split())

    pieces: list[str] = []
    current_rows: list[str] = []
    current_words = header_words

    # Add rows until the next one would take the piece over the word cap.
    for row in body:
        row_words = len(" ".join(row).split())
        if current_rows and current_words + row_words > MAX_CHUNK_WORDS:
            pieces.append("\n".join(header_lines + current_rows))
            current_rows, current_words = [], header_words
        current_rows.append(join_row(row))
        current_words += row_words
    if current_rows or not pieces:
        pieces.append("\n".join(header_lines + current_rows))
    return pieces


def chunk_blocks(blocks: list[Block]) -> list[Chunk]:
    """Group one chapter's blocks into heading-aligned chunks; tables stand alone.

    Raises if the blocks come from more than one chapter, or if the chapter
    has no section headings to chunk along.
    """
    chapters = {block.chapter for block in blocks}
    if len(chapters) != 1:
        raise ValueError(f"chunk_blocks takes one chapter at a time, got {chapters}.")
    if not any(block.kind == "heading" and block.level == 1 for block in blocks):
        raise ValueError(
            f"No section headings found in chapter {chapters.pop()}. Check the "
            f"heading font rules in parse.py against this chapter."
        )

    chapter = blocks[0].chapter
    source = blocks[0].source
    headings: dict[int, str] = {}
    engine_type: EngineType = "general"
    chunks: list[Chunk] = []
    pending: list[Block] = []

    def heading_path() -> str:
        return " > ".join(headings[level] for level in sorted(headings))

    def add_chunk(kind: ChunkKind, body: str, first_page: int, last_page: int) -> None:
        # Wrap one finished body in a Chunk with the current heading metadata.
        text = f"{heading_path() or INTRODUCTION_SECTION}\n\n{body}"
        chunk_engine_type = engine_type
        if chunk_engine_type == "general":
            chunk_engine_type = tag_engine_type(text)
        chunks.append(Chunk(
            chunk_id=f"ch{chapter:02d}-p{first_page:03d}-{len(chunks):03d}",
            text=text,
            source=source,
            chapter=chapter,
            section=headings.get(1, INTRODUCTION_SECTION),
            subsection=" > ".join(headings[level] for level in sorted(headings)
                                  if level > 1),
            page=first_page,
            page_end=last_page,
            kind=kind,
            engine_type=chunk_engine_type,
        ))

    def flush_text() -> None:
        # Pack the pending paragraphs into chunks, cutting at the word cap.
        piece: list[Block] = []
        piece_words = 0
        for block in pending:
            block_words = len(block.text.split())
            if piece and piece_words + block_words > MAX_CHUNK_WORDS:
                add_chunk("text", "\n\n".join(item.text for item in piece),
                          piece[0].page, piece[-1].page)
                piece, piece_words = [], 0
            piece.append(block)
            piece_words += block_words
        if piece:
            add_chunk("text", "\n\n".join(item.text for item in piece),
                      piece[0].page, piece[-1].page)
        pending.clear()

    # Walk the chapter in reading order, closing a chunk at every heading.
    for block in blocks:
        if block.kind == "heading":
            flush_text()
            headings = {level: text for level, text in headings.items()
                        if level < block.level}
            headings[block.level] = block.text
            if block.level == 1:
                engine_type = heading_engine_type(block.text) or engine_type
        elif block.kind == "table":
            for piece in table_pieces(block.text):
                add_chunk("table", piece, block.page, block.page)
        else:
            pending.append(block)
    flush_text()
    return chunks
