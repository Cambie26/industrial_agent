"""Retrieval tests. The parse tests build a small two-column PDF on the fly;
swap in tests/fixtures/ch10_sample.pdf once it exists."""

from pathlib import Path

import pytest
from eval.run_eval import ABLATIONS, FULL_PIPELINE, load_queries, recall_at_k

from industrial_agent.retrieval.parse import (
    chapter_number,
    chapter_pdfs,
    clean_text,
    table_to_markdown,
)

PAGE_WIDTH, PAGE_HEIGHT = 612, 792
TABLE_ROWS = [["Symptom", "Likely cause"], ["High EGT", "Compressor fouling"]]


def make_chapter_pdf(pdf_path: Path) -> Path:
    """Write a one-page, two-column PDF with a heading, a ruled table and a
    figure caption. Internal to these tests."""
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)

    # Full-width heading, then two columns of body text.
    page.insert_textbox((50, 60, 560, 90), "Turbine Engine Troubleshooting "
                        "and Inspection Procedures", fontsize=14)
    page.insert_textbox((50, 110, 290, 160), "Alpha paragraph in the left column.")
    page.insert_textbox((50, 170, 290, 220), "Bravo paragraph in the left column.")
    page.insert_textbox((320, 110, 560, 160), "Charlie paragraph in the right column.")

    # A ruled 2 x 2 table across both columns: draw the grid, then fill the cells.
    column_edges, row_edges = (50, 305, 560), (300, 330, 360)
    grid = page.new_shape()
    for x in column_edges:
        grid.draw_line((x, row_edges[0]), (x, row_edges[-1]))
    for y in row_edges:
        grid.draw_line((column_edges[0], y), (column_edges[-1], y))
    grid.finish(color=(0, 0, 0), width=1)
    grid.commit()
    for row_index, row in enumerate(TABLE_ROWS):
        for column_index, cell in enumerate(row):
            page.insert_text((column_edges[column_index] + 5,
                              row_edges[row_index] + 20), cell)

    page.insert_textbox((50, 400, 290, 440), "Figure 10-1. Axial-flow compressor.")
    document.save(pdf_path)
    document.close()
    return pdf_path


@pytest.fixture()
def chapter_blocks(tmp_path):
    from industrial_agent.retrieval.parse import parse_pdf

    return parse_pdf(make_chapter_pdf(tmp_path / "ch10.pdf"))


def test_chapter_number_comes_from_the_file_name():
    assert chapter_number("ch10.pdf") == 10
    assert chapter_number("chapter_03_engine_ignition.pdf") == 3


def test_chapter_number_fails_loudly_without_a_number():
    with pytest.raises(ValueError, match="chapter"):
        chapter_number("powerplant.pdf")


def test_chapter_pdfs_fails_loudly_on_an_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="No PDFs"):
        chapter_pdfs(tmp_path)


def test_clean_text_undoes_hyphenation_and_joins_lines():
    assert clean_text("compres-\nsor stall\n  occurs") == "compressor stall occurs"


def test_parse_reads_left_column_then_right_then_full_width_table(chapter_blocks):
    texts = [block.text for block in chapter_blocks]
    order = [next(i for i, text in enumerate(texts) if word in text)
             for word in ("Turbine Engine", "Alpha", "Bravo", "Charlie",
                          "High EGT", "Figure 10-1")]
    assert order == sorted(order)


def test_parse_returns_table_as_markdown(chapter_blocks):
    tables = [block for block in chapter_blocks if block.kind == "table"]
    assert [table.text for table in tables] == [
        "| Symptom | Likely cause |\n"
        "| --- | --- |\n"
        "| High EGT | Compressor fouling |"
    ]
    # Table cells must not also leak out as body text.
    assert not any("High EGT" in block.text for block in chapter_blocks
                   if block.kind == "text")


def test_parse_tags_figure_captions(chapter_blocks):
    captions = [block.text for block in chapter_blocks if block.kind == "caption"]
    assert captions == ["Figure 10-1. Axial-flow compressor."]


def test_parse_tags_every_block_with_chapter_and_page(chapter_blocks):
    assert {(block.chapter, block.page) for block in chapter_blocks} == {(10, 1)}


def test_naive_parse_gives_one_text_block_per_page(tmp_path):
    from industrial_agent.retrieval.parse import parse_pdf

    blocks = parse_pdf(make_chapter_pdf(tmp_path / "ch10.pdf"), table_aware=False)
    assert [block.kind for block in blocks] == ["text"]


def test_table_to_markdown_drops_empty_columns_and_escapes_pipes():
    rows = [["Part", None, "Limit"], ["Blade", "", "0.5 mm | max"]]
    assert table_to_markdown(rows) == (
        "| Part | Limit |\n| --- | --- |\n| Blade | 0.5 mm \\| max |"
    )
    assert table_to_markdown([[None, ""]]) == ""


def test_recall_at_k_counts_a_hit_only_within_the_top_k():
    retrieved = [(10, "Inspection"), (10, "Troubleshooting"), (9, "Ignition")]
    assert recall_at_k(retrieved, (10, "Troubleshooting"), k=2) == 1.0
    assert recall_at_k(retrieved, (9, "Ignition"), k=2) == 0.0


def test_recall_at_k_rejects_k_below_one():
    with pytest.raises(ValueError):
        recall_at_k([], (10, "Inspection"), k=0)


def test_every_ablation_switches_off_a_real_pipeline_setting():
    for overrides in ABLATIONS.values():
        assert set(overrides) <= set(FULL_PIPELINE)


def test_labelled_queries_file_loads_with_the_expected_fields():
    expected_fields = {"id", "question", "source_chapter", "source_section",
                       "source_page", "needs_data", "answerable"}
    for query in load_queries():
        assert set(query) == expected_fields
