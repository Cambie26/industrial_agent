# industrial_agent

## Runtime
- Runtime is Google Colab; GitHub is the source of truth.
- Don't add tooling that assumes a local checkout: Docker, Makefiles,
  pre-commit hooks, editor configs, shell scripts meant to be run by hand.
- If something can't be done from a browser, say so plainly rather than
  working around it.

## Structure
- All logic lives in `src/industrial_agent/`. Notebooks under `notebooks/`
  are a thin demo layer that imports the package — if a cell needs more
  than ~15 lines of logic, the logic is in the wrong place.
- Layout: `README.md`, `notebooks/`, `src/industrial_agent/`, `data/`,
  `tests/`, `pyproject.toml`, `requirements.txt`, `.gitignore`.
- Every notebook opens with the clone + editable install cell.
  README starts with an "Open in Colab" badge.
- `data/` is gitignored. Never commit datasets. Test fixtures go in
  `tests/fixtures/` and stay under a few hundred rows.

## Constraints
<!-- fill in once the structure exists — the untestable boundaries,
     the config approach, anything you had to decide rather than default to -->

## Working style
- Run `pytest` and `ruff check .` before pushing. Don't open a PR on red tests.
- One module or one clear change per task.
- Don't edit `.ipynb` files unless the task explicitly asks. Notebooks are
  edited in Colab.
- Solid Python — no need to explain basics. British spelling, metric units,
  type hints on public functions.
- When something needs to happen on my side (merge a PR, restart the runtime,
  mount Drive), say it as a numbered step.
