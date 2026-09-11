"""
Tools exposed to the agent.

Three tools, one per question an engineer actually asks:
  rank_fleet           - where should I look first?
  get_current_readings - what is this engine doing right now?
  assess_health        - how worn is it, on a scale that means something?

Design principle: tools return interpreted results, not raw rows. A model
reasons better over "sensor_11 is 3.2 sd above healthy" than over 20 floats,
and context stays small.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from .health import Baseline, deviations, recent_means, verdict, wear_index

MAX_RANK = 25
WORST_SENSORS = 4


class FleetTools:
    """The agent's view of the fleet.

    Takes its data by argument rather than reading globals, so tests can drive
    it from a small fixture and the package imports without the dataset.
    """

    def __init__(self, fleet: pd.DataFrame, baseline: Baseline) -> None:
        self.fleet = fleet
        self.baseline = baseline

    # -- helpers ----------------------------------------------------------
    def history(self, unit: int) -> pd.DataFrame:
        """Every cycle recorded for one engine. Empty if the unit is unknown."""
        return self.fleet[self.fleet.unit == unit]

    def unknown_unit_message(self, unit: int) -> str:
        """Message for a unit that isn't in the fleet, naming the valid range."""
        units = sorted(int(unit_id) for unit_id in self.fleet.unit.unique())
        return (f"No unit {unit} in the fleet. "
                f"Valid units are {units[0]}-{units[-1]}.")

    # -- tools ------------------------------------------------------------
    def rank_fleet(self, top_n: int = 10, worst_first: bool = True) -> str:
        """Rank every in-service engine by wear, to decide where to look.

        Returns a ranked summary as text, capped at MAX_RANK entries.
        """
        top_n = max(1, min(int(top_n), MAX_RANK))

        # Score every engine in the fleet, keeping its ID and service life.
        scored = []
        for unit in self.fleet.unit.unique():
            history = self.history(unit)
            scored.append((wear_index(history, self.baseline), int(unit),
                           int(history.cycle.max())))

        scored.sort(reverse=worst_first)
        urgent_count = sum(1 for wear_pct, _, _ in scored if wear_pct >= 70)

        end_of_range = "most worn" if worst_first else "healthiest"
        lines = [f"{len(scored)} engines in service. "
                 f"{urgent_count} at 70+ wear index (act now).",
                 f"\nThe {top_n} {end_of_range}:"]

        # One line per engine, from the requested end of the ranking.
        for wear_pct, unit, cycles in scored[:top_n]:
            lines.append(f"  unit {unit:>3}: wear {wear_pct:>3.0f}/100, "
                         f"{cycles} cycles in service")
        return "\n".join(lines)

    def get_current_readings(self, unit: int) -> str:
        """Latest sensor values for one in-service engine, with deviation from
        the healthy norm."""
        history = self.history(unit)
        if history.empty:
            return self.unknown_unit_message(unit)

        recent = recent_means(history, self.baseline)
        z_scores = deviations(history, self.baseline)

        lines = [
            f"Unit {unit}, {int(history.cycle.max())} cycles in service.",
            f"Mean of last {self.baseline.window} cycles, with deviation from "
            f"healthy norm:",
        ]

        # One line per sensor: current mean, healthy mean, and the gap in sd.
        for sensor in self.baseline.sensors:
            lines.append(f"  {sensor}: {recent[sensor]:.2f}  "
                         f"(healthy {self.baseline.healthy_mean[sensor]:.2f}, "
                         f"z={z_scores[sensor]:+.1f})")
        return "\n".join(lines)

    def assess_health(self, unit: int) -> str:
        """Overall wear of one in-service engine, calibrated against 100
        engines that ran to failure."""
        history = self.history(unit)
        if history.empty:
            return self.unknown_unit_message(unit)

        wear_pct = wear_index(history, self.baseline)
        recent = recent_means(history, self.baseline)
        z_scores = deviations(history, self.baseline)

        # Only the sensors furthest from healthy are worth reporting.
        worst_sensors = z_scores.abs().nlargest(WORST_SENSORS).index

        lines = [
            f"Unit {unit}: {verdict(wear_pct)}.",
            f"Wear index {wear_pct:.0f}/100.",
            f"In service {int(history.cycle.max())} cycles.",
            "",
            "Largest deviations from healthy:",
        ]
        for sensor in worst_sensors:
            lines.append(
                f"  {sensor}: {recent[sensor]:.2f} vs healthy "
                f"{self.baseline.healthy_mean[sensor]:.2f} "
                f"(z={z_scores[sensor]:+.1f})"
            )
        return "\n".join(lines)

    @property
    def registry(self) -> dict[str, Callable[..., str]]:
        """Tool name -> bound method, as the agent loop dispatches them."""
        return {
            "rank_fleet": self.rank_fleet,
            "get_current_readings": self.get_current_readings,
            "assess_health": self.assess_health,
        }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "rank_fleet",
        "description": (
            "Rank all 100 in-service engines by wear index (0 = as-new, "
            "100 = the level at which comparable engines failed). Use for "
            "fleet-wide questions: which engines are worst, what to inspect, "
            "how many are near failure. Returns a ranked list, not detail on "
            "any one engine."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "top_n": {"type": "integer",
                          "description": "How many to return. Default 10, max 25."},
                "worst_first": {"type": "boolean",
                                "description": "True for most worn (default), "
                                               "False for healthiest."},
            },
            "required": [],
        },
    },
    {
        "name": "get_current_readings",
        "description": (
            "Latest sensor values for one in-service engine, averaged over the "
            "last 10 cycles, each shown against the healthy fleet norm. Use "
            "when asked what an engine's readings are, or about a specific "
            "sensor."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "unit": {"type": "integer",
                         "description": "Engine unit ID, 1-100."}
            },
            "required": ["unit"],
        },
    },
    {
        "name": "assess_health",
        "description": (
            "Wear assessment for one in-service engine. Returns a wear index "
            "from 0 (as-new) to 100 (the level at which comparable engines "
            "failed), a plain-language verdict, and the sensors driving it. "
            "Use for questions about condition, degradation, whether an engine "
            "needs attention, or how worn it is."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "unit": {"type": "integer",
                         "description": "Engine unit ID, 1-100."}
            },
            "required": ["unit"],
        },
    },
]
