# industrial_agent

Demo project to show agentic AI capability - LLM SDKs and tooling use. 

Implemented is an agent that answers maintenance questions about a fleet of 100 turbofan
engines. One can ask it "which engines need attention?" and it decides which tools to
call, calls them, and comes back with a judgement.

The engine data used was taken from the NASA C-MAPSS turbofan degradation dataset (subset FD001). 
The agent deployed uses anthropic model to interpret requests, initiate actions, and serve results.

**[Read the demo notebook](notebooks/demo.ipynb)** — which is saved with its
cell outputs. Each cell is one question, showing every tool call the model made
before it answered. [`demo_transcript.md`](demo_transcript.md) is the same
conversation as plain markdown.

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

**Tools return interpreted results, not raw rows.** As a model reasons better over
`sensor_11 is 3.2 sd above healthy` than over twenty floats. 
The numeric work happens in Python; the model does the judgement.

**The wear index is calibrated, not invented.** Scoring is the mean absolute
z-score of an engine's last 10 cycles against a healthy norm — pooled from the
first 20 cycles of all 100 run-to-failure engines, which gives a stable noise
estimate. That raw statistic is then
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

## Running it

To run the demo notebook yourself you will need your own Anthropic API key - add an `ANTHROPIC_API_KEY` secret.

Note: the dataset (~4 MB) downloads on first use into a gitignored `data/`.
Default model is `claude-opus-5`.

## Data

[NASA C-MAPSS Turbofan Engine Degradation Simulation](https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data),
subset FD001, via
[hankroark/Turbofan-Engine-Degradation](https://github.com/hankroark/Turbofan-Engine-Degradation).
100 engines run to failure for calibration, 100 in service for the agent to
reason about.
