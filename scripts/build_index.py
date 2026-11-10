"""
Build the retrieval index from the chapter PDFs in `data/raw/`.

Run from the repo root in Colab: `!cd repo && python scripts/build_index.py`.
Parses each chapter, chunks it along its contents, then embeds and saves the
lot to `data/index/`. Needs the `[rag]` extra.

Parsing works now; chunking and indexing land in PR 3, so this stops there
with NotImplementedError until then.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from industrial_agent.retrieval.chunk import chunk_blocks
from industrial_agent.retrieval.index import INDEX_DIR, build_index
from industrial_agent.retrieval.parse import (
    RAW_DIR,
    chapter_pdfs,
    parse_pdf,
    table_of_contents,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    arguments = parser.parse_args()

    # Parse and chunk each chapter on its own, since sections are per chapter.
    chunks = []
    for pdf_path in chapter_pdfs(arguments.raw_dir):
        blocks = parse_pdf(pdf_path)
        chapter_chunks = chunk_blocks(blocks, table_of_contents(pdf_path))
        print(f"{pdf_path.name}: {len(blocks)} blocks -> {len(chapter_chunks)} chunks")
        chunks.extend(chapter_chunks)

    build_index(chunks, arguments.index_dir)
    print(f"Indexed {len(chunks)} chunks to {arguments.index_dir}")


if __name__ == "__main__":
    main()
