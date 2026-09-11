"""
The agent: a tool-use loop over the Claude Messages API.

Deliberately a hand-written loop rather than the SDK's tool runner. The point
of this repo is to show what an agent *is* - the model picks a tool, we run it,
the result goes back, and it decides whether it has enough to answer. That
cycle is worth reading.

`Agent.run()` yields events rather than printing. The notebook renders them;
tests assert on them.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from .tools import TOOL_SCHEMAS, FleetTools

MODEL = "claude-opus-5"
MAX_TURNS = 8
MAX_TOKENS = 16000

SYSTEM = """You are a maintenance analyst for a fleet of turbofan engines.

You answer questions using the tools provided. Never invent sensor values or
unit numbers - if a tool cannot give you something, say so plainly.

Engineers want a judgement, not a data dump. Lead with the answer, then the
evidence. Keep it short. If a reading is ambiguous, say what would settle it.
"""

EventKind = Literal["question", "tool_call", "tool_result", "answer", "error"]


@dataclass
class Event:
    """One observable step of a turn."""

    kind: EventKind
    text: str
    detail: dict[str, Any] = field(default_factory=dict)


def resolve_api_key() -> str | None:
    """Colab secret first, then the environment. Returns None if neither."""
    try:  # pragma: no cover - Colab only
        from google.colab import userdata

        # Try each secret name in turn; a missing secret raises, so skip it.
        for secret_name in ("ANTHROPIC_API_KEY", "api-key-sep26"):
            try:
                api_key = userdata.get(secret_name)
            except Exception:
                continue
            if api_key:
                return api_key
    except ImportError:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


def build_client() -> Any:
    """An Anthropic client, with the API key resolved for Colab or local use."""
    import anthropic

    api_key = resolve_api_key()
    if not api_key:
        raise RuntimeError(
            "No API key found. In Colab, add a secret named ANTHROPIC_API_KEY "
            "and grant this notebook access. Locally, export ANTHROPIC_API_KEY."
        )
    return anthropic.Anthropic(api_key=api_key)


class Agent:
    """A maintenance analyst backed by the fleet tools.

    Args:
        tools: FleetTools - the fleet the agent can query.
        client: Any | None - an Anthropic client, or None to build one.
        model: str - model ID to call.
        system: str - system prompt.
        max_turns: int - how many model turns before giving up on an answer.
        max_tokens: int - output token cap per turn.
    """

    def __init__(
        self,
        tools: FleetTools,
        client: Any | None = None,
        model: str = MODEL,
        system: str = SYSTEM,
        max_turns: int = MAX_TURNS,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        self.tools = tools
        self.client = client if client is not None else build_client()
        self.model = model
        self.system = system
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.messages: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Forget the conversation. The fleet data and client are kept."""
        self.messages = []

    def run(self, question: str) -> Iterator[Event]:
        """One question to a final answer, yielding each step as it happens.

        Args:
            question: str - what to ask. Appended to the running conversation,
                so earlier turns still apply until `reset()`.

        Yields:
            Event - the question, then each tool call and result, then either
            an answer or an error if the turn ran out of turns.
        """
        yield Event("question", question)
        self.messages.append({"role": "user", "content": question})

        # Each pass is one model turn: it either answers or calls tools.
        for _ in range(self.max_turns):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system,
                tools=TOOL_SCHEMAS,
                messages=self.messages,
            )
            self.messages.append(
                {"role": "assistant", "content": response.content}
            )

            # A server-side tool hit its iteration limit; ask it to carry on.
            if response.stop_reason == "pause_turn":
                continue

            # Anything other than a tool call means the model is done.
            if response.stop_reason != "tool_use":
                yield Event(
                    "answer",
                    "".join(block.text for block in response.content
                            if block.type == "text"),
                )
                return

            yield from self.run_tools(response)

        yield Event(
            "error",
            f"Stopped after {self.max_turns} turns without a final answer.",
        )

    def run_tools(self, response: Any) -> Iterator[Event]:
        """Execute every tool call in one assistant turn, then post the
        results back as a single user message.

        Internal to the loop in `run()`; kept separate because the fan-out over
        parallel tool calls is the part worth reading on its own.

        Args:
            response: Any - the assistant message, which may carry several
                tool_use blocks.

        Yields:
            Event - a tool_call then a tool_result for each block executed.
        """
        registry = self.tools.registry
        results: list[dict[str, Any]] = []

        # One assistant turn can request several tools; run them all.
        for block in response.content:
            if block.type != "tool_use":
                continue

            # Tool inputs arrive as JSON; never string-match the raw form.
            args = dict(block.input)
            yield Event(
                "tool_call",
                f"{block.name}({format_args(args)})",
                {"name": block.name, "input": args},
            )

            try:
                output = registry[block.name](**args)
                is_error = False
            except Exception as exc:  # a tool failure is data, not a crash
                output = f"Tool failed: {exc}"
                is_error = True

            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
                "is_error": is_error,
            })
            yield Event(
                "tool_result",
                output,
                {"name": block.name, "is_error": is_error},
            )

        # All results go back in ONE user message - splitting them teaches the
        # model to stop making parallel calls.
        self.messages.append({"role": "user", "content": results})


def format_args(args: dict[str, Any]) -> str:
    """Render tool arguments as a readable `name=value` call signature."""
    return ", ".join(
        f"{name}={json.dumps(value)}" for name, value in args.items()
    )
