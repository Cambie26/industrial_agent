# industrial_agent

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Cambie26/industrial_agent/blob/main/notebooks/demo.ipynb)

An agent that answers maintenance questions about a fleet of 100 turbofan
engines. Ask it "which engines need attention?" and it decides which tools to
call, calls them, and comes back with a judgement.

No retrieval, no fine-tuning, no framework — a hand-written tool-use loop over
the [Claude Messages API](https://docs.anthropic.com/en/api/messages) and the
NASA C-MAPSS turbofan degradation dataset (subset FD001).

**[Open the demo notebook](notebooks/demo.ipynb)** to see it working — each
cell is one question, with every tool call the model made before answering.
Running it also writes `demo_transcript.md`, the same conversation as plain
markdown.

```
------------------------------------------------------------------------
Q: Which engines need attention?
------------------------------------------------------------------------
  -> rank_fleet(top_n=5)          # the model chose this tool and these args
     100 engines in service. 10 at 70+ wear index (act now).

     The 5 most worn:
       unit  76: wear  86/100, 205 cycles in service
       unit  81: wear  80/100, 213 cycles in service
       ...
```

## How it works

```
question ──> Claude ──> tool_use? ──no──> answer
               ^                │
               │                yes
               └── tool results ─┘
```

Three tools, one per question an engineer actually asks:

| Tool | Answers |
|---|---|
| `rank_fleet` | Where should I look first? |
| `assess_health` | How worn is this engine, on a scale that means something? |
| `get_current_readings` | What is this engine doing right now? |

## Design notes

**Tools return interpreted results, not raw rows.** A model reasons better over
`sensor_11 is 3.2 sd above healthy` than over twenty floats, and the context
stays small. The numeric work happens in Python; the model does the judgement.

**The wear index is calibrated, not invented.** Scoring is the mean absolute
z-score of an engine's last 10 cycles against a healthy norm — pooled from the
first 20 cycles of all 100 run-to-failure engines, which gives a stable noise
estimate and works for units with short histories. That raw statistic is then
divided by the score those engines reached *at failure* (3.77 sd, computed in
`build_baseline`), so **100 means "as worn as engines that failed"**. The
calibration constant is never surfaced: an engineer sees `wear index 86`.

**The index is not clamped at the bottom.** A new engine reads in the teens
rather than 0, because sensors are noisy and units vary at manufacture.
Clamping flattened a third of the healthy fleet to exactly zero and broke the
ranking.

**Seven of the 21 sensors are dropped.** Sensors 1, 5, 10, 16, 18 and 19 are
constant across FD001; sensor 6 is effectively binary (sd = 0.002), which makes
it useless as a signal and dangerous as a z-score denominator.

**Failures are data, not crashes.** A question about unit 250 gets
`No unit 250 in the fleet. Valid units are 1-100.` back as a tool result, so
the model can recover and say something useful. Unexpected exceptions come back
as `is_error` results rather than ending the turn.

## Layout

```
src/industrial_agent/
  data.py     download and load FD001; nothing is read at import time
  health.py   pure scoring — baseline, z-scores, wear index, verdict
  tools.py    the three tools and their JSON schemas
  agent.py    the tool-use loop; yields events instead of printing
  chat.py     prints the exchange and saves a transcript
  plots.py    sensor traces, for context
notebooks/
  demo.ipynb              the demo above
  Industrial_Agent.ipynb  development scratchpad
tests/                    fixtures sliced from FD001; no network, no API key
```

The demo prints plain text rather than rendering an `ipywidgets` chat box
deliberately: widget state is not saved into a `.ipynb`, so a widget-based demo
shows a visitor browsing GitHub nothing at all.

## Running it

In Colab, click the badge above, add an `ANTHROPIC_API_KEY` secret, and run all.

Locally:

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
python -c "
from industrial_agent import default_chat
chat = default_chat()
chat.ask('Which engines need attention?')
"
pytest && ruff check .
```

The dataset (~4 MB) downloads on first use into a gitignored `data/`.
Default model is `claude-opus-5`; override with `default_chat(model=...)`.

## Data

[NASA C-MAPSS Turbofan Engine Degradation Simulation](https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data),
subset FD001, via
[hankroark/Turbofan-Engine-Degradation](https://github.com/hankroark/Turbofan-Engine-Degradation).
100 engines run to failure for calibration, 100 in service for the agent to
reason about.
