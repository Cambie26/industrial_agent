from types import SimpleNamespace

from tests.test_agent import agent_with, reply, text, tool_use

from industrial_agent.chat import Chat


def test_ask_prints_question_tools_and_answer(tools, capsys):
    a = agent_with([
        reply([tool_use("assess_health", {"unit": 1})], "tool_use"),
        reply([text("Unit 1 is fine.")], "end_turn"),
    ], tools)
    chat = Chat(a)
    chat.ask("How is unit 1?")

    out = capsys.readouterr().out
    assert "Q: How is unit 1?" in out
    assert "-> assess_health(unit=1)" in out
    assert "Unit 1 is fine." in out
    assert "\n".join(chat.transcript).count("Q: How is unit 1?") == 1


def test_tool_output_can_be_hidden(tools, capsys):
    a = agent_with([
        reply([tool_use("assess_health", {"unit": 1})], "tool_use"),
        reply([text("Fine.")], "end_turn"),
    ], tools)
    Chat(a, show_tool_output=False).ask("How is unit 1?")

    out = capsys.readouterr().out
    assert "-> assess_health(unit=1)" in out
    assert "Wear index" not in out


def test_save_writes_markdown(tools, tmp_path, capsys):
    a = agent_with([reply([text("Hello.")], "end_turn")], tools)
    chat = Chat(a)
    chat.ask("Hi")
    path = chat.save(tmp_path / "t.md")

    body = path.read_text()
    assert body.startswith("# Industrial agent - demo transcript")
    assert "Q: Hi" in body and "Hello." in body
    assert f"`{a.model}`" in body


def test_reset_clears_agent_history(tools, capsys):
    a = agent_with([reply([text("Hello.")], "end_turn")], tools)
    chat = Chat(a)
    chat.ask("Hi")
    chat.reset()
    assert a.messages == []
    assert "[history cleared]" in capsys.readouterr().out


def test_stub_shapes_match_sdk_blocks():
    """Guard the stub: real SDK blocks expose .type/.text/.name/.input/.id."""
    block = tool_use("rank_fleet", {"top_n": 3})
    assert (block.type, block.name, block.input, block.id) == (
        "tool_use", "rank_fleet", {"top_n": 3}, "tu_1")
    assert isinstance(reply([], "end_turn"), SimpleNamespace)
