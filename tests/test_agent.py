"""Agent loop tests, driven by a stub client - no network, no API key."""

from types import SimpleNamespace

import pytest

from industrial_agent.agent import Agent


def text(s):
    return SimpleNamespace(type="text", text=s)


def tool_use(name, tool_input, block_id="tu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input,
                           id=block_id)


def reply(content, stop_reason):
    return SimpleNamespace(content=content, stop_reason=stop_reason)


class StubClient:
    """Replays scripted responses and records what it was sent."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.script.pop(0)


def agent_with(script, tools, **kwargs):
    return Agent(tools, client=StubClient(script), **kwargs)


def kinds(events):
    return [e.kind for e in events]


def test_answer_without_tools(tools):
    a = agent_with([reply([text("The hangar temperature is not something I "
                               "can measure.")], "end_turn")], tools)
    events = list(a.run("What's the hangar temperature?"))
    assert kinds(events) == ["question", "answer"]
    assert "not something I can measure" in events[-1].text


def test_single_tool_call_then_answer(tools):
    a = agent_with([
        reply([tool_use("assess_health", {"unit": 1})], "tool_use"),
        reply([text("Unit 1 is fine.")], "end_turn"),
    ], tools)
    events = list(a.run("How is unit 1?"))
    assert kinds(events) == ["question", "tool_call", "tool_result", "answer"]
    assert events[1].text == "assess_health(unit=1)"
    assert "Wear index" in events[2].text


def test_parallel_tool_calls_return_in_one_message(tools):
    """Splitting results across messages teaches the model to stop
    parallelising, so all results must land in a single user turn."""
    a = agent_with([
        reply([tool_use("assess_health", {"unit": 1}, "a"),
               tool_use("assess_health", {"unit": 2}, "b")], "tool_use"),
        reply([text("Unit 2 is worse.")], "end_turn"),
    ], tools)
    list(a.run("Compare units 1 and 2"))

    results = [m for m in a.messages
               if m["role"] == "user" and isinstance(m["content"], list)]
    assert len(results) == 1
    assert [b["tool_use_id"] for b in results[0]["content"]] == ["a", "b"]


def test_tool_exception_is_reported_not_raised(tools):
    a = agent_with([
        reply([tool_use("assess_health", {"nonsense": 1})], "tool_use"),
        reply([text("I could not read that engine.")], "end_turn"),
    ], tools)
    events = list(a.run("How is unit ???"))
    failure = next(e for e in events if e.kind == "tool_result")
    assert failure.detail["is_error"] is True
    assert failure.text.startswith("Tool failed:")


def test_unknown_unit_is_a_normal_result(tools):
    """A unit outside the fleet is answered by the tool, not an exception."""
    a = agent_with([
        reply([tool_use("assess_health", {"unit": 250})], "tool_use"),
        reply([text("There is no unit 250.")], "end_turn"),
    ], tools)
    events = list(a.run("How is unit 250?"))
    result = next(e for e in events if e.kind == "tool_result")
    assert result.detail["is_error"] is False
    assert "No unit 250" in result.text


def test_pause_turn_is_resumed(tools):
    a = agent_with([
        reply([text("")], "pause_turn"),
        reply([text("Done.")], "end_turn"),
    ], tools)
    events = list(a.run("Anything?"))
    assert kinds(events) == ["question", "answer"]
    assert events[-1].text == "Done."


def test_loop_stops_at_max_turns(tools):
    a = agent_with(
        [reply([tool_use("rank_fleet", {})], "tool_use")] * 3, tools,
        max_turns=3,
    )
    events = list(a.run("Loop forever"))
    assert events[-1].kind == "error"
    assert "Stopped after 3 turns" in events[-1].text


def test_request_carries_tools_and_system(tools):
    a = agent_with([reply([text("hi")], "end_turn")], tools)
    list(a.run("hello"))
    sent = a.client.calls[0]
    assert sent["model"] == a.model
    assert [t["name"] for t in sent["tools"]] == list(tools.registry)
    assert "maintenance analyst" in sent["system"]


def test_history_persists_across_questions_until_reset(tools):
    a = agent_with([reply([text("one")], "end_turn"),
                    reply([text("two")], "end_turn")], tools)
    list(a.run("first"))
    list(a.run("second"))
    assert [m["role"] for m in a.messages] == ["user", "assistant"] * 2

    a.reset()
    assert a.messages == []


def test_build_client_without_key(monkeypatch):
    from industrial_agent import agent as agent_module

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(agent_module, "resolve_api_key", lambda: None)
    with pytest.raises(RuntimeError, match="No API key"):
        agent_module.build_client()
