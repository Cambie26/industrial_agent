"""
PDF -> blocks: the first stage of the maintenance-docs pipeline.

Each handbook chapter is its own PDF in `data/raw/`, named with its chapter
number (`ch10.pdf`, `chapter_10.pdf`, the FAA's own `12_amtp_ch10_0.pdf`
and `10_turbine_engines.pdf` all work). A page
comes back as a list of `Block`s in reading order, each tagged as a heading
(level 1 section or level 2 subsection), body text, a table (rendered as
markdown) or a figure caption. Figures themselves are dropped - only their
captions are kept, and the small labels drawn inside figures are filtered out.

Before touching this, know that:
  - PyMuPDF is part of the `[rag]` extra, so it is imported inside the
    functions. The package must still import without it.
  - The handbook is laid out in two columns with full-width headings and
    figures. Reading order is rebuilt by `reading_order`; sorting blocks by
    plain (y, x) interleaves the columns line by line.
  - Headings are found by font, not by the PDF's bookmarks (those are
    tagged-PDF structure with whole paragraphs in them). In Chapter 10 a
    section heading is Arial-Bold 11 pt, a subsection Helvetica-Bold 10 pt,
    a sub-subsection Times Bold-Italic 11 pt, body Times 10 pt, and figure
    labels Helvetica 4-8 pt or white text on dark artwork. The rules below
    go by weight, slant, size and colour, not font names, so other chapters
    on the same template should parse the same way.
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

# A line set entirely in bold at this size or above is a heading: bold italic
# is level 3, bold at SECTION_HEADING_MIN_SIZE or above level 1, other bold
# level 2. Headings need a few letters so bold axis numbers on charts don't
# count.
HEADING_MIN_SIZE = 9.5
SECTION_HEADING_MIN_SIZE = 10.5
HEADING_MIN_LETTERS = 3

# Body text is 10 pt; anything smaller (outside tables and captions) is a
# label drawn inside a figure.
BODY_MIN_SIZE = 9.0

# Text blocks with fewer letters than this are stray chart labels and numbers.
MIN_TEXT_LETTERS = 10

# White text (sRGB integer) only appears as labels on dark figure artwork.
WHITE = 0xFFFFFF

# PyMuPDF span flag bits.
ITALIC_FLAG = 2
BOLD_FLAG = 16

BlockKind = Literal["heading", "text", "table", "caption"]


@dataclass
class Block:
    """One piece of a page: a heading, a paragraph, a table or a figure caption.

    `level` is 1 for a section heading, 2 for a subsection and 3 below that;
    0 for anything that isn't a heading.
    """

    source: str
    chapter: int
    page: int
    kind: BlockKind
    text: str
    level: int = 0


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


def letter_count(text: str) -> int:
    """Number of alphabetic characters in `text`."""
    return sum(character.isalpha() for character in text)


def visible_spans(line: dict[str, Any]) -> list[dict[str, Any]]:
    """A PyMuPDF line's spans that hold text and aren't white figure labels."""
    return [span for span in line["spans"]
            if span["text"].strip() and span["color"] != WHITE]


def is_bold(span: dict[str, Any]) -> bool:
    """True if a span is set in a bold face."""
    return bool(span["flags"] & BOLD_FLAG) or "bold" in span["font"].lower()


def is_italic(span: dict[str, Any]) -> bool:
    """True if a span is set in an italic or oblique face."""
    font = span["font"].lower()
    return bool(span["flags"] & ITALIC_FLAG) or "ital" in font or "oblique" in font


def heading_level(line: dict[str, Any]) -> int:
    """Heading level (1-3) of a PyMuPDF text line, or 0 if it isn't a heading."""
    spans = visible_spans(line)
    if not spans or not all(is_bold(span) for span in spans):
        return 0
    smallest = min(span["size"] for span in spans)
    letters = letter_count("".join(span["text"] for span in spans))
    if smallest < HEADING_MIN_SIZE or letters < HEADING_MIN_LETTERS:
        return 0
    if all(is_italic(span) for span in spans):
        return 3
    return 1 if smallest >= SECTION_HEADING_MIN_SIZE else 2


def largest_font_size(line: dict[str, Any]) -> float:
    """Largest font size among a line's visible spans; 0 if there are none."""
    return max((span["size"] for span in visible_spans(line)), default=0.0)


def split_text_block(
    lines: list[dict[str, Any]],
) -> list[tuple[str, int, tuple[float, ...], str]]:
    """Split one PyMuPDF text block into heading and body segments.

    A block often fuses a heading with the paragraph under it; this separates
    them and drops lines set too small to be body text (figure labels).
    Internal to `parse_page`.

    Args:
        lines: the block's lines from `page.get_text("dict")`.

    Returns:
        (kind, level, bounding box, raw text) per segment, in order. `kind` is
        "heading" or "text"; a caption is returned as "text" and tagged later.
    """
    segments: list[tuple[str, int, tuple[float, ...], str]] = []
    current_level = None
    current_lines: list[dict[str, Any]] = []

    def close_segment() -> None:
        # Turn the lines gathered so far into one segment with a joint box.
        if not current_lines:
            return
        box = (min(line["bbox"][0] for line in current_lines),
               min(line["bbox"][1] for line in current_lines),
               max(line["bbox"][2] for line in current_lines),
               max(line["bbox"][3] for line in current_lines))
        raw_text = "\n".join("".join(span["text"] for span in line["spans"])
                             for line in current_lines)
        kind = "heading" if current_level else "text"
        segments.append((kind, current_level or 0, box, raw_text))
        current_lines.clear()

    # A caption's "Figure 10-1." is set at 8 pt, so keep caption blocks whole.
    block_text = clean_text("\n".join("".join(span["text"] for span in line["spans"])
                                      for line in lines))
    is_caption = bool(CAPTION_PATTERN.match(block_text))

    # Walk the lines, starting a new segment whenever heading level changes.
    for line in lines:
        level = 0 if is_caption else heading_level(line)
        if not level and not is_caption and largest_font_size(line) < BODY_MIN_SIZE:
            continue
        if level != current_level:
            close_segment()
            current_level = level
        current_lines.append(line)
    close_segment()
    return segments


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
        Heading, text, table and caption blocks. Images are dropped.
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

    # Then every text block that isn't an image, inside a table or in the margins.
    for text_block in page.get_text("dict")["blocks"]:
        if text_block["type"] != 0:  # 1 = image
            continue
        for kind, level, box, raw_text in split_text_block(text_block["lines"]):
            if box[3] < MARGIN_RATIO * height or box[1] > (1 - MARGIN_RATIO) * height:
                continue
            if any(inside(box, table_box) for table_box in table_boxes):
                continue
            text = clean_text(raw_text)
            if not text or PAGE_NUMBER_PATTERN.match(text):
                continue
            if CAPTION_PATTERN.match(text):
                kind = "caption"
            elif kind == "text" and letter_count(text) < MIN_TEXT_LETTERS:
                continue
            placed.append((box, Block(source, chapter, page_number, kind, text,
                                      level)))

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

    Kept for inspection only: the handbook's bookmarks are tagged-PDF
    structure, not a contents list, so chunking uses heading blocks instead.
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
        if block.kind == "heading":
            lines.append(f"{'#' * (block.level + 1)} {block.text}\n")
        elif block.kind == "caption":
            lines.append(f"*{block.text}*\n")
        else:
            lines.append(block.text + "\n")
    return "\n".join(lines).strip() + "\n"
