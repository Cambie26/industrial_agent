"""
Retrieval evaluation: recall@k over the labelled questions, and the ablations.

Questions live in `eval/queries.jsonl`, one JSON object per line:
  id, question, source_chapter, source_section, source_page,
  needs_data (also needs the sensor tools), answerable (False = should refuse).

A hit counts when a retrieved chunk comes from the labelled chapter and
section. Unanswerable questions are left out of recall - they test refusal,
which is the agent's job (PR 5), not retrieval's.

Run from the repo root in Colab: `!cd repo && python eval/run_eval.py`.
`recall_at_k` works now; `run_ablations` needs search (PR 4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

QUERIES_PATH = Path(__file__).parent / "queries.jsonl"
K_VALUES = (1, 3, 5, 10)

# The full pipeline, and each variant as the one switch it turns off.
FULL_PIPELINE = {"table_aware": True, "hybrid": True, "rerank": True,
                 "turbine_only": True}
ABLATIONS: dict[str, dict[str, bool]] = {
    "full": {},
    "naive_parse": {"table_aware": False},
    "dense_only": {"hybrid": False},
    "no_rerank": {"rerank": False},
    "no_filter": {"turbine_only": False},
}


def load_queries(path: Path | str = QUERIES_PATH) -> list[dict[str, Any]]:
    """Read the labelled questions. Skips blank lines; raises on a malformed one."""
    queries = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            queries.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{path} line {line_number} is not valid JSON: {error}"
            ) from error
    if not queries:
        raise ValueError(f"No questions in {path}.")
    return queries


def recall_at_k(
    retrieved: list[tuple[int, str]], relevant: tuple[int, str], k: int
) -> float:
    """1.0 if the labelled (chapter, section) is in the top k retrieved, else 0.0.

    Args:
        retrieved: (chapter, section) of each hit, best first.
        relevant: the labelled (chapter, section) for the question.
        k: how many of the top hits to look at.

    Returns:
        1.0 or 0.0; average over questions for recall@k.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}.")
    return 1.0 if relevant in retrieved[:k] else 0.0


def run_ablations(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recall@k for every variant in ABLATIONS, one row per variant."""
    raise NotImplementedError("run_ablations needs search (PR 4).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    arguments = parser.parse_args()

    # One row per variant, one column per k.
    for row in run_ablations(load_queries(arguments.queries)):
        print(row)


if __name__ == "__main__":
    main()
