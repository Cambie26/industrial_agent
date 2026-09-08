from industrial_agent.tools import MAX_RANK, TOOL_SCHEMAS


def test_registry_matches_schemas(tools):
    assert set(tools.registry) == {s["name"] for s in TOOL_SCHEMAS}


def test_schemas_are_well_formed():
    for schema in TOOL_SCHEMAS:
        assert schema["description"].strip()
        assert schema["input_schema"]["type"] == "object"
        for name in schema["input_schema"].get("required", []):
            assert name in schema["input_schema"]["properties"]


def test_assess_health_reports_index_and_verdict(tools):
    out = tools.assess_health(1)
    assert "Unit 1:" in out
    assert "Wear index" in out
    assert "Largest deviations from healthy:" in out


def test_unknown_unit_is_answered_not_raised(tools):
    """The model must be told the unit does not exist, not handed a traceback."""
    for out in (tools.assess_health(250), tools.get_current_readings(250)):
        assert "No unit 250" in out
        assert "Valid units are" in out


def test_current_readings_list_every_live_sensor(tools, baseline):
    out = tools.get_current_readings(1)
    for sensor in baseline.sensors:
        assert f"  {sensor}:" in out


def test_rank_fleet_clamps_top_n(tools):
    assert tools.rank_fleet(top_n=999).count("  unit ") <= MAX_RANK
    assert tools.rank_fleet(top_n=0).count("  unit ") == 1


def test_rank_fleet_orders_by_wear(tools):
    def wears(text):
        return [int(line.split("wear")[1].split("/")[0])
                for line in text.splitlines() if line.startswith("  unit ")]

    assert wears(tools.rank_fleet(3)) == sorted(wears(tools.rank_fleet(3)),
                                                reverse=True)
    healthiest = wears(tools.rank_fleet(3, worst_first=False))
    assert healthiest == sorted(healthiest)
