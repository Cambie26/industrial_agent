"""
PDF -> blocks: the first stage of the maintenance-docs pipeline.

Each handbook chapter is its own PDF in `data/raw/`, named with its chapter
number (`ch10.pdf`, `chapter_10.pdf`, the FAA's own `12_amtp_ch10_0.pdf`
and `10_turbine_engines.pdf` all work). A page
comes back as a list of `Block`s in reading order, each tagged as body text, a
table (rendered as markdown) or a figure caption. Figures themselves are
dropped - only their captions are kept.

Before touching this, know that:
  - PyMuPDF is part of the `[rag]` extra, so it is imported inside the
    functions. The package must still import without it.
  - The handbook is laid out in two columns with full-width headings and
    figures. Reading order is rebuilt by `reading_order`; sorting blocks by
    plain (y, x) interleaves the columns line by line.
  - `table_aware=False` is the naive baseline for the ablation, not a
    fallback. Keep it naive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

RAW_DIR = Path("data/raw")

# "Figure 10-12. Compressor stall ..." - the handbook numbers figures per chapter.
CAPTION_PATTERN = re.compile(r"^Figure\s+\d+-\d+")

# "ch10" / "chapter_10" in a file name wins; otherwise the first number is the
# chapter. The FAA's downloads are named like "12_amtp_ch10_0.pdf".
LABELLED_CHAPTER_PATTERN = re.compile(r"ch(?:apter)?[_\- ]?(\d+)", re.IGNORECASE)
CHAPTER_PATTERN = re.compile(r"(\d+)")

# A block that reaches this far (in points) either side of the page's centre
# line crosses the gutter, so it spans both columns.
GUTTER_HALF_WIDTH = 10

# Blocks entirely inside the top/bottom margin are running headers and page
# numbers ("10-3"), not content.
MARGIN_RATIO = 0.05

# The handbook's page footers ("10-50") sit low but not always inside the margin.
PAGE_NUMBER_PATTERN = re.compile(r"^\d+-\d+$")

BlockKind = Literal["text", "table", "caption"]


@dataclass
class Block:
    """One piece of a page: a paragraph, a table or a figure caption."""

    source: str
    chapter: int
    page: int
    kind: BlockKind
    text: str


def chapter_number(pdf_path: Path | str) -> int:
    """Chapter number taken from the file name. Raises if there isn't one."""
    stem = Path(pdf_path).stem
    match = LABELLED_CHAPTER_PATTERN.search(stem) or CHAPTER_PATTERN.search(stem)
    if match is None:
        raise ValueError(
            f"Can't tell which chapter {Path(pdf_path).name} is. Name chapter "
            f"PDFs with their number, e.g. ch10.pdf."
        )
    return int(match.group(1))


def clean_text(raw_text: str) -> str:
    """Join a block's lines into one string, undoing end-of-line hyphenation."""
    joined = re.sub(r"-\n(?=[a-z])", "", raw_text)
    return re.sub(r"\s+", " ", joined).strip()


def inside(inner: tuple[float, ...], outer: tuple[float, ...]) -> bool:
    """True if the centre of box `inner` falls within box `outer`."""
    centre_x = (inner[0] + inner[2]) / 2
    centre_y = (inner[1] + inner[3]) / 2
    return outer[0] <= centre_x <= outer[2] and outer[1] <= centre_y <= outer[3]


def table_to_markdown(rows: list[list[str | None]]) -> str:
    """Render table rows as markdown, first row as the header. Columns that are
    empty in every row are dropped; returns "" for a table with no text.

    Written here rather than using PyMuPDF's `to_markdown`, which invents
    "Col2"-style headers for empty columns.
    """
    # Tidy every cell and keep only the columns that hold something.
    cells = [[clean_text(cell or "").replace("|", "\\|") for cell in row]
             for row in rows]
    column_count = max((len(row) for row in cells), default=0)
    cells = [row + [""] * (column_count - len(row)) for row in cells]
    kept_columns = [column for column in range(column_count)
                    if any(row[column] for row in cells)]
    if not kept_columns:
        return ""

    lines = []
    for row_index, row in enumerate(cells):
        lines.append("| " + " | ".join(row[column] for column in kept_columns) + " |")
        if row_index == 0:
            lines.append("|" + " --- |" * len(kept_columns))
    return "\n".join(lines)


def table_cell_text(page: Any, table: Any) -> list[list[str | None]]:
    """Text of every cell of a PyMuPDF table, row by row, header row first.

    Reads each cell's box off the page rather than using `table.extract()`,
    which drops the spaces between words in the handbook's fonts.
    """
    import pymupdf

    rows = [row.cells for row in table.rows]
    if table.header.external:
        rows = [table.header.cells, *rows]
    return [[page.get_textbox(pymupdf.Rect(cell)) if cell else None
             for cell in row]
            for row in rows]


def reading_order(
    placed: list[tuple[tuple[float, ...], Block]], page_width: float
) -> list[Block]:
    """Order a page's blocks for a two-column layout.

    Full-width blocks - anything that crosses the gutter, such as headings,
    wide figures and tables - split the page into bands. Within each band
    the left column is read top to bottom, then the right.

    Args:
        placed: (bounding box, block) pairs; a box is (x0, y0, x1, y1).
        page_width: width of the page, in the same units as the boxes.

    Returns:
        The blocks, in reading order.
    """
    ordered: list[Block] = []
    left_column: list[tuple[float, Block]] = []
    right_column: list[tuple[float, Block]] = []

    def flush_band() -> None:
        # Empty the current band into the output: left column, then right.
        for column in (left_column, right_column):
            by_height = sorted(column, key=lambda pair: pair[0])
            ordered.extend(block for _, block in by_height)
            column.clear()

    # Walk down the page; every full-width block closes the band above it.
    centre = page_width / 2
    for box, block in sorted(placed, key=lambda pair: (pair[0][1], pair[0][0])):
        crosses_gutter = (box[0] < centre - GUTTER_HALF_WIDTH
                          and box[2] > centre + GUTTER_HALF_WIDTH)
        if crosses_gutter:
            flush_band()
            ordered.append(block)
        elif box[0] < centre:
            left_column.append((box[1], block))
        else:
            right_column.append((box[1], block))
    flush_band()
    return ordered


def parse_page(page: Any, source: str, chapter: int, page_number: int) -> list[Block]:
    """Table-aware parse of one PyMuPDF page into blocks, in reading order.

    Args:
        page: a `pymupdf.Page`.
        source: file name the page came from.
        chapter: chapter number.
        page_number: 1-based page number within the PDF.

    Returns:
        Text, table and caption blocks. Images are dropped.
    """
    width, height = page.rect.width, page.rect.height
    placed: list[tuple[tuple[float, ...], Block]] = []

    # Pull out the tables first, as markdown, and remember where they sit.
    table_boxes = []
    for table in page.find_tables().tables:
        markdown = table_to_markdown(table_cell_text(page, table))
        if not markdown:
            continue
        table_boxes.append(tuple(table.bbox))
        placed.append((tuple(table.bbox),
                       Block(source, chapter, page_number, "table", markdown)))

    # Then every text block that isn't inside a table or the page margins.
    for x0, y0, x1, y1, raw_text, _, block_type in page.get_text("blocks"):
        box = (x0, y0, x1, y1)
        if block_type != 0:  # 1 = image
            continue
        if y1 < MARGIN_RATIO * height or y0 > (1 - MARGIN_RATIO) * height:
            continue
        if any(inside(box, table_box) for table_box in table_boxes):
            continue
        text = clean_text(raw_text)
        if not text or PAGE_NUMBER_PATTERN.match(text):
            continue
        kind: BlockKind = "caption" if CAPTION_PATTERN.match(text) else "text"
        placed.append((box, Block(source, chapter, page_number, kind, text)))

    return reading_order(placed, width)


def parse_pdf(pdf_path: Path | str, table_aware: bool = True) -> list[Block]:
    """Parse one chapter PDF into blocks, page by page.

    With `table_aware=False` each page is one text block from PyMuPDF's plain
    text extraction - the naive baseline for the ablation.
    """
    import pymupdf

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"No PDF at {pdf_path}.")
    chapter = chapter_number(pdf_path)

    blocks: list[Block] = []
    with pymupdf.open(pdf_path) as document:
        # Loop through each page, numbering from 1 to match the printed PDF.
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            if table_aware:
                blocks.extend(parse_page(page, pdf_path.name, chapter, page_number))
                continue
            text = page.get_text("text").strip()
            if text:
                blocks.append(Block(pdf_path.name, chapter, page_number, "text", text))

    if not blocks:
        raise ValueError(f"No text found in {pdf_path.name}. Is it a scanned PDF?")
    return blocks


def chapter_pdfs(raw_dir: Path | str = RAW_DIR) -> list[Path]:
    """Every chapter PDF in `raw_dir`, sorted by chapter. Raises if there are none."""
    raw_dir = Path(raw_dir)
    pdf_paths = sorted(raw_dir.glob("*.pdf"), key=chapter_number)
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs in {raw_dir.resolve()}. Put one PDF per chapter there."
        )
    return pdf_paths


def parse_corpus(
    raw_dir: Path | str = RAW_DIR, table_aware: bool = True
) -> list[Block]:
    """Parse every chapter PDF in `raw_dir`, in chapter order."""
    blocks: list[Block] = []
    for pdf_path in chapter_pdfs(raw_dir):
        blocks.extend(parse_pdf(pdf_path, table_aware=table_aware))
    return blocks


def table_of_contents(pdf_path: Path | str) -> list[tuple[int, str, int]]:
    """The PDF's bookmarks as (level, title, page). Raises if it has none.

    Chunking takes its section boundaries from this. A chapter PDF without
    bookmarks will need its contents page parsed instead.
    """
    import pymupdf

    with pymupdf.open(Path(pdf_path)) as document:
        toc = document.get_toc()
    if not toc:
        raise ValueError(
            f"{Path(pdf_path).name} has no bookmarks, so the contents page "
            f"will have to be parsed for section boundaries."
        )
    return [(int(level), str(title).strip(), int(page)) for level, title, page in toc]


def to_markdown(blocks: list[Block]) -> str:
    """Render blocks as one markdown document with page markers, for checking
    a parse by eye."""
    lines: list[str] = []
    current_page = None
    for block in blocks:
        # Start a new section whenever the page changes.
        if (block.source, block.page) != current_page:
            current_page = (block.source, block.page)
            lines.append(f"\n---\n**{block.source} - page {block.page}**\n")
        if block.kind == "caption":
            lines.append(f"*{block.text}*\n")
        else:
            lines.append(block.text + "\n")
    return "\n".join(lines).strip() + "\n"
