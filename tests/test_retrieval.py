"""Retrieval tests. The parse tests build a small two-column PDF on the fly;
swap in tests/fixtures/ch10_sample.pdf once it exists. Chunk tests build
blocks by hand, and index tests use a fake embedder, so no model download.
"""

from pathlib import Path

import pytest
from eval.run_eval import ABLATIONS, FULL_PIPELINE, load_queries, recall_at_k

from industrial_agent.retrieval import chunk
from industrial_agent.retrieval.chunk import (
    chunk_blocks,
    table_pieces,
    tag_engine_type,
    tidy_table,
)
from industrial_agent.retrieval.parse import (
    Block,
    chapter_number,
    chapter_pdfs,
    clean_text,
    table_to_markdown,
)

PAGE_WIDTH, PAGE_HEIGHT = 612, 792
TABLE_ROWS = [["Symptom", "Likely cause"], ["High EGT", "Compressor fouling"]]


def make_chapter_pdf(pdf_path: Path) -> Path:
    """Write a one-page, two-column PDF with section and subsection headings,
    a heading fused onto its paragraph, a small figure label, a ruled table
    and a figure caption. Internal to these tests."""
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)

    # Full-width bold section heading, then two columns of body text.
    page.insert_textbox((50, 60, 560, 90), "Turbine Engine Troubleshooting "
                        "and Inspection Procedures", fontname="hebo", fontsize=11)
    page.insert_textbox((50, 110, 290, 160), "Alpha paragraph in the left column.",
                        fontname="tiro", fontsize=10)
    page.insert_textbox((50, 170, 290, 220), "Bravo paragraph in the left column.",
                        fontname="tiro", fontsize=10)
    page.insert_textbox((320, 110, 560, 160), "Charlie paragraph in the right column.",
                        fontname="tiro", fontsize=10)

    # A subsection heading set tight above its paragraph, so PyMuPDF fuses them.
    page.insert_text((50, 240), "Combustor Checks", fontname="hebo", fontsize=10)
    page.insert_text((50, 253), "Delta paragraph under a fused heading.",
                     fontname="tiro", fontsize=10)

    # A small label of the kind drawn inside figures.
    page.insert_text((330, 230), "Ridge removed by grinding", fontsize=7)

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
    page.insert_textbox((280, 700, 340, 720), "10-1")
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
    assert chapter_number("10_turbine_engines.pdf") == 10


def test_chapter_number_prefers_the_ch_label_in_faa_file_names():
    assert chapter_number("12_amtp_ch10_0.pdf") == 10


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
             for word in ("Turbine Engine", "Alpha", "Bravo", "Combustor",
                          "Delta", "Charlie", "High EGT", "Figure 10-1")]
    assert order == sorted(order)


def test_parse_finds_headings_and_their_levels(chapter_blocks):
    headings = [(block.level, block.text) for block in chapter_blocks
                if block.kind == "heading"]
    assert headings == [
        (1, "Turbine Engine Troubleshooting and Inspection Procedures"),
        (2, "Combustor Checks"),
    ]


def test_parse_splits_a_heading_off_its_paragraph(chapter_blocks):
    assert any(block.kind == "text" and block.text ==
               "Delta paragraph under a fused heading." for block in chapter_blocks)


def test_parse_drops_small_figure_labels(chapter_blocks):
    assert not any("Ridge removed" in block.text for block in chapter_blocks)


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


def test_parse_drops_page_number_footers(chapter_blocks):
    assert not any(block.text == "10-1" for block in chapter_blocks)


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


def make_block(kind: str, text: str, page: int = 1, level: int = 0) -> Block:
    """A chapter 10 block, for building chunker input by hand."""
    return Block("ch10.pdf", 10, page, kind, text, level)


def words(count: int, word: str = "inspect") -> str:
    """A paragraph of `count` copies of one word."""
    return " ".join([word] * count)


@pytest.fixture()
def chapter_chunks():
    return chunk_blocks([
        make_block("text", "Engine Maintenance & Operation"),
        make_block("heading", "Reciprocating Engine Overhaul", 1, level=1),
        make_block("text", "Overhaul every cylinder at the stated interval.", 1),
        make_block("heading", "Engine Ratings", 2, level=1),
        make_block("text", "Ratings are set by the manufacturer.", 2),
        make_block("heading", "Turbine Engine Maintenance", 3, level=1),
        make_block("heading", "Blade Inspection", 3, level=2),
        make_block("text", "Look for nicks on the leading edge.", 3),
        make_block("caption", "Figure 10-64. Stress rupture cracks.", 4),
        make_block("heading", "Engine Shutdown", 5, level=1),
        make_block("text", "Retard the throttle to idle first.", 5),
    ])


def test_chunks_follow_headings_and_carry_section_metadata(chapter_chunks):
    sections = [(item.section, item.subsection) for item in chapter_chunks]
    assert sections == [
        ("Introduction", ""),
        ("Reciprocating Engine Overhaul", ""),
        ("Engine Ratings", ""),
        ("Turbine Engine Maintenance", "Blade Inspection"),
        ("Engine Shutdown", ""),
    ]


def test_chunk_text_starts_with_its_heading_path(chapter_chunks):
    assert chapter_chunks[3].text.startswith(
        "Turbine Engine Maintenance > Blade Inspection\n\n")


def test_engine_type_sticks_until_the_next_section_that_names_one(chapter_chunks):
    engine_types = [item.engine_type for item in chapter_chunks]
    assert engine_types == ["general", "reciprocating", "reciprocating",
                            "turbine", "turbine"]


def test_captions_join_their_section_and_widen_its_page_range(chapter_chunks):
    blade_chunk = chapter_chunks[3]
    assert "Figure 10-64" in blade_chunk.text
    assert (blade_chunk.page, blade_chunk.page_end) == (3, 4)
    assert blade_chunk.citation == "Ch. 10, pp. 10-3-4"
    assert chapter_chunks[4].citation == "Ch. 10, p. 10-5"


def test_chunk_ids_are_unique(chapter_chunks):
    chunk_ids = [item.chunk_id for item in chapter_chunks]
    assert len(set(chunk_ids)) == len(chunk_ids)
    assert chunk_ids[1].startswith("ch10-p001-")


def test_long_sections_split_at_paragraphs_under_the_word_cap():
    paragraphs = [make_block("text", words(120), page) for page in range(1, 6)]
    chunks = chunk_blocks([make_block("heading", "Cleaning", 1, level=1),
                           *paragraphs])
    assert len(chunks) == 3
    for item in chunks:
        body = item.text.split("\n\n", 1)[1]
        assert len(body.split()) <= chunk.MAX_CHUNK_WORDS


def test_tables_become_their_own_chunks():
    chunks = chunk_blocks([
        make_block("heading", "Troubleshooting Turbine Engines", 60, level=1),
        make_block("text", "Common malfunctions follow.", 60),
        make_block("table", "| Trouble | Remedy |\n| --- | --- |\n"
                            "| High EGT | Wash compressor |", 60),
    ])
    assert [item.kind for item in chunks] == ["table", "text"]
    assert chunks[0].engine_type == "turbine"


def test_tidy_table_merges_split_columns_and_fills_down_continuations():
    markdown = ("| Trouble |  | Probable causes |\n| --- | --- | --- |\n"
                "|  | Engine fails to start. | • Lack of fuel. |\n"
                "|  |  | • Underpriming. |\n"
                "| Low oil pressure. |  | • Worn bearings. |")
    header, body = tidy_table(markdown)
    assert header == ["Trouble", "Probable causes"]
    assert body == [["Engine fails to start.", "• Lack of fuel."],
                    ["Engine fails to start.", "• Underpriming."],
                    ["Low oil pressure.", "• Worn bearings."]]


def test_long_tables_split_by_rows_and_repeat_the_header(monkeypatch):
    monkeypatch.setattr(chunk, "MAX_CHUNK_WORDS", 10)
    rows = "\n".join(f"| Fault {i} | {words(4, 'check')} |" for i in range(6))
    pieces = table_pieces(f"| Trouble | Remedy |\n| --- | --- |\n{rows}")
    assert len(pieces) > 1
    assert all(piece.startswith("| Trouble | Remedy |\n| --- | --- |\n")
               for piece in pieces)
    assert sum(piece.count("| Fault") for piece in pieces) == 6


def test_chunk_blocks_fails_loudly_without_section_headings():
    with pytest.raises(ValueError, match="No section headings"):
        chunk_blocks([make_block("text", "Just prose.")])


def test_chunk_blocks_takes_one_chapter_at_a_time():
    other_chapter = Block("ch11.pdf", 11, 1, "heading", "Ignition", 1)
    with pytest.raises(ValueError, match="one chapter"):
        chunk_blocks([make_block("heading", "Cleaning", level=1), other_chapter])


def test_tag_engine_type_counts_keywords():
    assert tag_engine_type("Check the compressor and turbine blades.") == "turbine"
    assert tag_engine_type("Remove the piston from the cylinder.") == "reciprocating"
    assert tag_engine_type("Record the work in the logbook.") == "general"


def fake_embedder(texts: list[str]):
    """Deterministic 4-d unit vectors from text length. Internal to these tests."""
    numpy = pytest.importorskip("numpy")
    vectors = numpy.array([[len(text), len(text.split()), 1.0, 2.0]
                           for text in texts], dtype=numpy.float32)
    return vectors / numpy.linalg.norm(vectors, axis=1, keepdims=True)


def test_index_round_trips_through_disk(tmp_path, chapter_chunks):
    pytest.importorskip("rank_bm25")
    from industrial_agent.retrieval.index import build_index, load_index

    built = build_index(chapter_chunks, tmp_path, embed=fake_embedder,
                        embedding_model="fake-model")
    loaded = load_index(tmp_path)
    assert loaded.chunks == chapter_chunks
    assert loaded.embedding_model == "fake-model"
    assert loaded.embeddings.shape == built.embeddings.shape == (5, 4)
    assert loaded.bm25.get_scores(["throttle"]).argmax() == 4


def test_load_index_fails_loudly_when_nothing_was_built(tmp_path):
    from industrial_agent.retrieval.index import load_index

    with pytest.raises(FileNotFoundError, match="build_index"):
        load_index(tmp_path)


def test_build_index_rejects_duplicate_chunk_ids(tmp_path, chapter_chunks):
    from industrial_agent.retrieval.index import build_index

    with pytest.raises(ValueError, match="unique"):
        build_index(chapter_chunks * 2, tmp_path, embed=fake_embedder)


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
