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
- Prefer fewer, larger files over many small ones. Don't split a module
  just because it's getting long — split when there's a real reason
  (a genuinely separate concern, a different dependency, a piece reused
  from several places). One new file per change is usually the ceiling;
  if a change wants three, say why.
- Same goes for docs and config: one `README.md`, one `pyproject.toml`.
  Don't scatter notes across extra markdown files.
- Every notebook opens with the clone + editable install cell.
  README starts with an "Open in Colab" badge.
- `data/` is gitignored. Never commit datasets. Test fixtures go in
  `tests/fixtures/` and stay under a few hundred rows.

## Code style
- Names are spelled out. No single-letter or cryptic variables —
  `df_failures` not `d`, `sensor_column` not `sc`. Loop variables included;
  `for engine_id in ...` not `for e in ...`. The only exceptions are `i`
  in a trivial index loop and the conventional `_` for a discard.
- Comments describe *what* the code is doing, in plain blunt terms —
  "loop through each engine and take the last cycle", "drop the sensors
  that never move". A couple per longer function is right; a short obvious
  function needs none.
- Only explain *why* when the reasoning is genuinely non-obvious (a
  workaround, a numerical trap, an ordering that matters). Don't justify
  ordinary code.
- Every module opens with a docstring saying what it's for and what the
  reader should know before touching it.
- Type hints on public functions. British spelling, metric units.
- Constants (URLs, column lists, thresholds) go at the top of the module in
  CAPS, not buried as magic numbers inside functions.
- Fail loudly with a clear message rather than silently returning an empty
  frame or a default.

## Constraints
<!-- fill in once the structure exists — the untestable boundaries,
     the config approach, anything you had to decide rather than default to -->

## Working style
- Run `pytest` and `ruff check .` before pushing. Don't open a PR on red tests.
- One module or one clear change per task.
- Keep the diff to the task. Don't reformat, rename, or tidy unrelated code
  in the same commit — it makes review much harder.
- Don't edit `.ipynb` files unless the task explicitly asks. Notebooks are
  edited in Colab.
- Tests mirror the module (`tests/test_health.py` for `health.py`) and are
  named for the behaviour being checked, not `test_1`.
- Prefer the standard library and what's already in `requirements.txt`.
  Flag any new dependency before adding it.
- Solid Python — no need to explain basics.
- When reporting back: say what changed, what you deliberately left alone,
  and anything you weren't sure about.
- When something needs to happen on my side (merge a PR, restart the runtime,
  mount Drive), say it as a numbered step.
