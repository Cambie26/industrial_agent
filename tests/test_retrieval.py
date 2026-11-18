"""Retrieval tests. The parse tests build a small two-column PDF on the fly;
swap in tests/fixtures/ch10_sample.pdf once it exists. Chunk tests build
blocks by hand, and index tests use a fake embedder, so no model download.
"""

import importlib
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


def make_chunk(position: int, text: str, engine_type: str) -> chunk.Chunk:
    """A chapter 10 chunk for search tests, one per page."""
    return chunk.Chunk(chunk_id=f"ch10-p{position:03d}-{position:03d}", text=text,
                       source="ch10.pdf", chapter=10, section="Section",
                       subsection="", page=position, page_end=position,
                       kind="text", engine_type=engine_type)


@pytest.fixture()
def search_index(monkeypatch):
    """A four-chunk index with hand-set embeddings, and fake models that need
    no download. The query always embeds to the first axis."""
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("rank_bm25")
    from industrial_agent.retrieval.index import Index, build_bm25

    # The package exports the search() function under the module's own name,
    # so fetch the module itself to patch it.
    search_module = importlib.import_module("industrial_agent.retrieval.search")

    chunks = [
        make_chunk(0, "Piston ring gap check", "reciprocating"),
        make_chunk(1, "Compressor wash procedure", "turbine"),
        make_chunk(2, "High EGT troubleshooting", "turbine"),
        make_chunk(3, "Logbook entries after maintenance", "general"),
    ]
    embeddings = numpy.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0],
                              [0.0, 1.0, 0.0], [0.1, 0.0, 0.99]], dtype=numpy.float32)
    embeddings /= numpy.linalg.norm(embeddings, axis=1, keepdims=True)

    query_vector = numpy.array([1.0, 0.0, 0.0], dtype=numpy.float32)
    monkeypatch.setattr(search_module, "query_embedder",
                        lambda model_name: lambda query: query_vector)
    # The fake reranker loves anything mentioning EGT.
    monkeypatch.setattr(search_module, "cross_encoder",
                        lambda model_name: lambda query, passages: [
                            5.0 if "EGT" in passage else -5.0 for passage in passages])
    return Index(chunks, embeddings, build_bm25(chunks), "fake-model")


def hit_pages(hits) -> list[int]:
    """The page of each hit, best first."""
    return [hit.chunk.page for hit in hits]


def test_reciprocal_rank_fusion_rewards_agreement_between_rankings():
    from industrial_agent.retrieval.search import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion([[0, 1], [1, 2]], k=60)
    assert max(fused, key=fused.get) == 1
    assert fused[1] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0] == pytest.approx(1 / 61)


def test_turbine_filter_drops_reciprocating_chunks_only(search_index):
    from industrial_agent.retrieval.search import search

    pages = hit_pages(search("gap", top_k=4, hybrid=False, rerank=False,
                             index=search_index))
    assert 0 not in pages
    assert set(pages) == {1, 2, 3}
    unfiltered = search("gap", top_k=4, hybrid=False, rerank=False,
                        turbine_only=False, index=search_index)
    assert hit_pages(unfiltered)[0] == 0


def test_dense_only_ranks_by_cosine_similarity(search_index):
    from industrial_agent.retrieval.search import search

    hits = search("EGT", top_k=2, hybrid=False, rerank=False, index=search_index)
    assert hit_pages(hits) == [1, 3]
    assert {hit.scored_by for hit in hits} == {"dense"}


def test_hybrid_lets_bm25_rescue_an_exact_term_the_embeddings_miss(search_index):
    from industrial_agent.retrieval.search import search

    hits = search("EGT", top_k=2, rerank=False, index=search_index)
    assert hit_pages(hits) == [1, 2]
    assert {hit.scored_by for hit in hits} == {"rrf"}


def test_reranker_reorders_candidates_and_scores_between_zero_and_one(search_index):
    from industrial_agent.retrieval.search import search

    hits = search("EGT", top_k=3, index=search_index)
    assert hit_pages(hits)[0] == 2
    assert all(hit.scored_by == "rerank" and 0 < hit.score < 1 for hit in hits)
    assert hits[0].score > 0.99 > 0.01 > hits[-1].score


def test_reranker_probabilities_pass_through_without_a_second_sigmoid():
    from industrial_agent.retrieval.search import relevances

    assert relevances([0.0, 0.58, 1.0]) == [0.0, 0.58, 1.0]


def test_reranker_logits_are_turned_into_probabilities():
    from industrial_agent.retrieval.search import relevances

    assert relevances([0.0, 5.0, -5.0]) == pytest.approx([0.5, 0.9933, 0.0067],
                                                         abs=1e-4)


def test_search_rejects_bad_arguments(search_index):
    from industrial_agent.retrieval.search import search

    with pytest.raises(ValueError, match="Empty query"):
        search("   ", index=search_index)
    with pytest.raises(ValueError, match="top_k"):
        search("EGT", top_k=0, index=search_index)


def test_search_fails_loudly_when_the_filter_leaves_nothing(search_index):
    from industrial_agent.retrieval.index import Index
    from industrial_agent.retrieval.search import search

    only_reciprocating = Index(search_index.chunks[:1], search_index.embeddings[:1],
                               search_index.bm25, "fake-model")
    with pytest.raises(ValueError, match="engine-type filter"):
        search("piston", index=only_reciprocating)


def scored_hit(score: float, scored_by: str = "rerank", page: int = 60):
    """A search hit on a turbine troubleshooting chunk."""
    from industrial_agent.retrieval.search import SearchHit

    hit_chunk = make_chunk(page, "Section\n\nHigh EGT at takeoff.", "turbine")
    hit_chunk.subsection = "Turbine Engine Analyzer Uses"
    return SearchHit(hit_chunk, score, scored_by)


def test_confidence_band_follows_the_thresholds():
    from industrial_agent.retrieval.search import confidence_band

    assert confidence_band(scored_hit(0.739)) == "confident"
    assert confidence_band(scored_hit(0.260)) == "cautious"
    assert confidence_band(scored_hit(0.000)) == "refuse"
    assert confidence_band(scored_hit(0.260), cautious_below=0.2) == "confident"


def test_confidence_band_is_unscored_without_the_reranker():
    from industrial_agent.retrieval.search import confidence_band

    assert confidence_band(scored_hit(0.03, scored_by="rrf")) == "unscored"


def test_app_table_and_verdict_describe_the_hits():
    from industrial_agent.retrieval.app import hits_table, verdict

    hits = [scored_hit(0.579), scored_hit(0.02, page=55)]
    rows = hits_table(hits, refuse_below=0.15, cautious_below=0.4)
    assert rows[0] == [1, 0.579, "confident", "Ch. 10, p. 10-60", "turbine", "text",
                       "Section > Turbine Engine Analyzer Uses"]
    assert rows[1][2] == "refuse"
    assert verdict(hits, 0.15, 0.4).startswith("**Answer**")
    assert verdict(hits, 0.15, 0.6).startswith("**Answer cautiously**")


def test_app_hit_text_drops_the_heading_line():
    from industrial_agent.retrieval.app import hits_markdown

    text = hits_markdown([scored_hit(0.579)])
    assert text.startswith("### 1. Ch. 10, p. 10-60 - 0.579")
    assert "High EGT at takeoff." in text
    assert "Section\n\nHigh" not in text


def test_labelled_query_records_the_chosen_hit_as_the_source():
    from industrial_agent.retrieval.app import labelled_query, source_choices

    hits = [scored_hit(0.739, page=54), scored_hit(0.579, page=60)]
    record = labelled_query("High EGT at takeoff?", source_choices(hits)[1], hits,
                            needs_data=False, query_id="q01")
    assert record == {"id": "q01", "question": "High EGT at takeoff?",
                      "source_chapter": 10, "source_section": "Section",
                      "source_page": 60, "needs_data": False, "answerable": True}


def test_labelled_query_marks_not_covered_questions_unanswerable():
    from industrial_agent.retrieval.app import NOT_COVERED, labelled_query

    record = labelled_query("CFM56 fan bolt torque?", NOT_COVERED,
                            [scored_hit(0.0)], needs_data=False, query_id="q02")
    assert record["answerable"] is False
    assert record["source_section"] is None


def test_labelled_query_refuses_without_a_question_or_a_choice():
    from industrial_agent.retrieval.app import labelled_query

    with pytest.raises(ValueError, match="question"):
        labelled_query("  ", "anything", [], needs_data=False, query_id="q01")
    with pytest.raises(ValueError, match="Pick"):
        labelled_query("Why?", None, [scored_hit(0.5)], needs_data=False,
                       query_id="q01")


def test_saved_labels_number_on_and_load_as_eval_queries(tmp_path):
    from industrial_agent.retrieval.app import (
        NOT_COVERED,
        append_query,
        labelled_query,
        next_query_id,
    )

    queries_path = tmp_path / "drive" / "queries.jsonl"
    assert next_query_id(queries_path) == "q01"
    for question in ("First?", "Second?"):
        append_query(queries_path, labelled_query(
            question, NOT_COVERED, [], needs_data=True,
            query_id=next_query_id(queries_path)))
    assert next_query_id(queries_path) == "q03"
    assert [query["id"] for query in load_queries(queries_path)] == ["q01", "q02"]


def test_app_builds_without_launching(search_index):
    pytest.importorskip("gradio")
    from industrial_agent.retrieval.app import build_app

    assert build_app(search_index) is not None
