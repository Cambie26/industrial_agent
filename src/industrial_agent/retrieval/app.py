"""
A browser page for trying handbook search and tuning it: a Gradio app.

Launched from a Colab cell with `launch(index_dir, queries_path)`; Gradio
prints a public link that opens in its own tab and lasts while the runtime
runs. Type a question, flip the search stages on and off, drag the
confidence thresholds, and see the hits re-ranked and re-banded. When a
result is clearly right (or the handbook clearly has nothing), save the
question as a labelled example; it is appended to `queries_path` in the
`eval/queries.jsonl` format, so testing builds the evaluation set.

Before touching this, know that:
  - Gradio is part of the `[rag]` extra and is imported inside the functions
    that need it. The helpers above `build_app` are plain Python so tests
    can check them without a browser.
  - Moving a threshold slider only re-bands the hits already found; it
    doesn't search again. Toggling a stage or the hit count does.
  - Bands only mean something with the reranker on (see
    `search.confidence_band`); without it every hit shows "unscored".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .index import INDEX_DIR, Index, load_index
from .search import (
    CAUTIOUS_BELOW,
    REFUSE_BELOW,
    SearchHit,
    confidence_band,
    search,
)

HIT_COLUMNS = ["#", "Score", "Band", "Citation", "Engine", "Kind", "Section"]
MAX_HITS = 10
DEFAULT_HITS = 5

# The dropdown option for marking a question the handbook doesn't answer.
NOT_COVERED = "None of these - the handbook doesn't cover it"

VERDICTS = {
    "refuse": "**Refuse** - the best match is too weak; the agent would say the "
              "handbook doesn't cover this.",
    "cautious": "**Answer cautiously** - the best match only touches on the "
                "question; the agent would say coverage is thin.",
    "confident": "**Answer** - the best match looks relevant; the agent would "
                 "answer with citations.",
    "unscored": "**No verdict** - turn the reranker on to get 0-1 scores.",
}


def heading_path(hit: SearchHit) -> str:
    """Section and subsection of a hit, joined with " > "."""
    return " > ".join(part for part in (hit.chunk.section, hit.chunk.subsection)
                      if part)


def hits_table(
    hits: list[SearchHit], refuse_below: float, cautious_below: float
) -> list[list[Any]]:
    """One row per hit, in HIT_COLUMNS order."""
    return [[rank, round(hit.score, 3),
             confidence_band(hit, refuse_below, cautious_below),
             hit.chunk.citation, hit.chunk.engine_type, hit.chunk.kind,
             heading_path(hit)]
            for rank, hit in enumerate(hits, start=1)]


def verdict(hits: list[SearchHit], refuse_below: float, cautious_below: float) -> str:
    """What the agent would do with these hits, judged on the best one."""
    if not hits:
        return "No hits."
    band = confidence_band(hits[0], refuse_below, cautious_below)
    return f"{VERDICTS[band]} Top score {hits[0].score:.3f}."


def hits_markdown(hits: list[SearchHit]) -> str:
    """Every hit's full text as markdown, so tables render as tables."""
    sections = []
    for rank, hit in enumerate(hits, start=1):
        # The chunk text starts with its heading path; show only the body.
        body = hit.chunk.text.split("\n\n", 1)[-1]
        sections.append(f"### {rank}. {hit.chunk.citation} - {hit.score:.3f}\n"
                        f"*{heading_path(hit)}*\n\n{body}")
    return "\n\n---\n\n".join(sections)


def source_choices(hits: list[SearchHit]) -> list[str]:
    """Dropdown labels for marking which hit answers the question."""
    labels = [f"{rank}. {hit.chunk.citation} - {heading_path(hit)}"
              for rank, hit in enumerate(hits, start=1)]
    return [*labels, NOT_COVERED]


def next_query_id(queries_path: Path | str) -> str:
    """The next free "qNN" id in a labelled-questions file; "q01" if it's new."""
    queries_path = Path(queries_path)
    numbers = [0]
    if queries_path.exists():
        for line in queries_path.read_text().splitlines():
            query_id = json.loads(line)["id"] if line.strip() else ""
            if query_id.startswith("q") and query_id[1:].isdigit():
                numbers.append(int(query_id[1:]))
    return f"q{max(numbers) + 1:02d}"


def labelled_query(
    question: str,
    choice: str,
    hits: list[SearchHit],
    needs_data: bool,
    query_id: str,
) -> dict[str, Any]:
    """A record in the eval/queries.jsonl format from the user's choice of source.

    Args:
        question: the question as typed.
        choice: the chosen dropdown label, from `source_choices(hits)`.
        hits: the hits the labels were made from.
        needs_data: True if answering also needs the sensor tools.
        query_id: the id to store, e.g. from `next_query_id`.

    Returns:
        A dict with exactly the fields `eval/run_eval.py` reads.
    """
    if not question.strip():
        raise ValueError("Type a question before saving it.")
    labels = source_choices(hits)
    if choice not in labels:
        raise ValueError("Pick which hit answers the question (or 'None of these').")

    record = {"id": query_id, "question": question.strip(), "source_chapter": None,
              "source_section": None, "source_page": None,
              "needs_data": bool(needs_data), "answerable": choice != NOT_COVERED}
    if record["answerable"]:
        chunk = hits[labels.index(choice)].chunk
        record.update(source_chapter=chunk.chapter, source_section=chunk.section,
                      source_page=chunk.page)
    return record


def append_query(queries_path: Path | str, record: dict[str, Any]) -> None:
    """Add one labelled question as a new line at the end of the file."""
    queries_path = Path(queries_path)
    queries_path.parent.mkdir(parents=True, exist_ok=True)
    with queries_path.open("a") as queries_file:
        queries_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_app(index: Index, queries_path: Path | str | None = None) -> Any:
    """The Gradio page, wired to `index`. Saving labels needs `queries_path`."""
    import gradio as gr
    import pandas as pd

    def show(hits, refuse_below, cautious_below):
        # Everything that depends on the thresholds, recomputed on each change.
        table = pd.DataFrame(hits_table(hits, refuse_below, cautious_below),
                             columns=HIT_COLUMNS)
        return verdict(hits, refuse_below, cautious_below), table

    def run_search(question, hybrid, rerank, turbine_only, top_k,
                   refuse_below, cautious_below):
        # Search, then fill every output and reset the labelling dropdown.
        # Toggling a stage with no question yet shouldn't pop up an error.
        if not question.strip():
            return ("Type a question first.", None, "", gr.Dropdown(choices=[]),
                    [], "")
        try:
            hits = search(question, top_k=int(top_k), hybrid=hybrid,
                          rerank=rerank, turbine_only=turbine_only, index=index)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        verdict_text, table = show(hits, refuse_below, cautious_below)
        dropdown = gr.Dropdown(choices=source_choices(hits), value=None)
        return verdict_text, table, hits_markdown(hits), dropdown, hits, ""

    def save(question, choice, needs_data, hits):
        # Append the labelled question to the file and say what was written.
        if queries_path is None:
            raise gr.Error("No queries_path was given to launch(), so there's "
                           "nowhere to save to.")
        try:
            record = labelled_query(question, choice, hits, needs_data,
                                    next_query_id(queries_path))
        except ValueError as error:
            raise gr.Error(str(error)) from error
        append_query(queries_path, record)
        return f"Saved `{record['id']}` to `{queries_path}`:\n\n```json\n" \
               f"{json.dumps(record, ensure_ascii=False)}\n```"

    with gr.Blocks(title="Handbook search") as app:
        gr.Markdown("# FAA Powerplant Handbook search\n"
                    "Ask a maintenance question, then tune the search below.")
        hits_state = gr.State([])

        with gr.Row():
            question = gr.Textbox(label="Question", scale=5, placeholder=(
                "Engine has high EGT at takeoff power. What should I check?"))
            search_button = gr.Button("Search", variant="primary", scale=1)

        with gr.Row():
            hybrid = gr.Checkbox(value=True, label="Hybrid (BM25 + dense)")
            rerank = gr.Checkbox(value=True, label="Reranker")
            turbine_only = gr.Checkbox(value=True, label="Turbine only")
            top_k = gr.Slider(1, MAX_HITS, value=DEFAULT_HITS, step=1, label="Hits")
        with gr.Row():
            refuse_below = gr.Slider(0, 1, value=REFUSE_BELOW, step=0.01,
                                     label="Refuse below")
            cautious_below = gr.Slider(0, 1, value=CAUTIOUS_BELOW, step=0.01,
                                       label="Answer cautiously below")

        verdict_box = gr.Markdown()
        table = gr.Dataframe(headers=HIT_COLUMNS, interactive=False, wrap=True)
        with gr.Accordion("Full text of each hit", open=False):
            details = gr.Markdown()

        with gr.Accordion("Save as a labelled question", open=False):
            source = gr.Dropdown(label="Which hit answers it?", choices=[])
            needs_data = gr.Checkbox(label="Also needs sensor data")
            save_button = gr.Button("Save")
            save_status = gr.Markdown()

        # Searching fills everything; moving a threshold only re-bands.
        search_inputs = [question, hybrid, rerank, turbine_only, top_k,
                         refuse_below, cautious_below]
        search_outputs = [verdict_box, table, details, source, hits_state,
                          save_status]
        search_button.click(run_search, search_inputs, search_outputs)
        question.submit(run_search, search_inputs, search_outputs)
        for control in (hybrid, rerank, turbine_only, top_k):
            control.change(run_search, search_inputs, search_outputs)
        for slider in (refuse_below, cautious_below):
            slider.change(show, [hits_state, refuse_below, cautious_below],
                          [verdict_box, table])
        save_button.click(save, [question, source, needs_data, hits_state],
                          [save_status])
    return app


def launch(
    index_dir: Path | str = INDEX_DIR,
    queries_path: Path | str | None = None,
    share: bool = True,
) -> Any:
    """Load the index and open the search page.

    Args:
        index_dir: folder `scripts/build_index.py` wrote to.
        queries_path: labelled-questions file to append to; saving is off
            without it. On Colab, put it on Drive so it survives a restart.
        share: give the page a public gradio.live link (needed from Colab).

    Returns:
        The running Gradio app.
    """
    app = build_app(load_index(index_dir), queries_path)
    app.launch(share=share)
    return app
