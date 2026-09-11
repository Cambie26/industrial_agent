"""
Chat surface for the demo notebook.

Prints plain text rather than rendering widgets, deliberately: ipywidget state
is not saved into a .ipynb, so a widget-based demo shows a visitor nothing on
GitHub. Printed output is saved with the notebook and mirrored to a markdown
transcript.
"""

from __future__ import annotations

from pathlib import Path

from .agent import Agent, Event
from .data import download, load_failed, load_fleet
from .health import build_baseline
from .tools import FleetTools

RULE = "-" * 72
TRANSCRIPT_TITLE = "# Industrial agent - demo transcript"


class Chat:
    """Ask questions, see the tool calls, keep a transcript."""

    def __init__(self, agent: Agent, show_tool_output: bool = True) -> None:
        self.agent = agent
        self.show_tool_output = show_tool_output
        self.transcript: list[str] = []

    def ask(self, question: str) -> None:
        """Run one question and print the exchange."""
        for event in self.agent.run(question):
            for line in self.render(event):
                print(line)
                self.transcript.append(line)
        print()
        self.transcript.append("")

    def reset(self) -> None:
        """Start a fresh conversation. The transcript is kept."""
        self.agent.reset()
        note = "[history cleared]"
        print(note, "\n")
        self.transcript += [note, ""]

    def save(self, path: str | Path = "demo_transcript.md") -> Path:
        """Write the whole session to a markdown file."""
        path = Path(path)
        body = "\n".join(self.transcript).rstrip()
        path.write_text(
            f"{TRANSCRIPT_TITLE}\n\n"
            f"Model: `{self.agent.model}`\n\n"
            f"```\n{body}\n```\n",
            encoding="utf-8",
        )
        print(f"Wrote {path} ({len(self.transcript)} lines)")
        return path

    def render(self, event: Event) -> list[str]:
        """Format one event as printable lines. Empty when it is suppressed."""
        if event.kind == "question":
            return [RULE, f"Q: {event.text}", RULE]
        if event.kind == "tool_call":
            return [f"  -> {event.text}"]
        if event.kind == "tool_result":
            if not self.show_tool_output:
                return []
            marker = "  !! " if event.detail.get("is_error") else "     "
            return [(marker + line).rstrip()
                    for line in event.text.splitlines()]
        if event.kind == "error":
            return ["", f"!! {event.text}"]
        return ["", event.text]


def default_chat(show_tool_output: bool = True, **agent_kwargs) -> Chat:
    """Download the data, calibrate, wire up the agent. One call to start.

    Args:
        show_tool_output: bool - print each tool result, not just the call.
        **agent_kwargs: Any - passed straight to `Agent` (client, model,
            system, max_turns, max_tokens).

    Returns:
        Chat - ready to take questions.
    """
    download()
    tools = FleetTools(load_fleet(), build_baseline(load_failed()))
    return Chat(Agent(tools, **agent_kwargs), show_tool_output=show_tool_output)
