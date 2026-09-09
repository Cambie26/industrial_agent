"""Exploratory plots. Context for the demo, not used by the agent."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

DEFAULT_SENSORS = ["sensor_11", "sensor_4", "sensor_12", "sensor_9"]
DEFAULT_UNITS = [3, 24, 69]


def sensor_traces(
    fleet: pd.DataFrame,
    sensors: list[str] | None = None,
    units: list[int] | None = None,
) -> Figure:
    """Sensor traces over cycles for a few engines - degradation is visible
    to the eye long before failure.

    Args:
        fleet: pd.DataFrame - fleet history, one row per unit per cycle.
        sensors: list[str] | None - sensors to plot, one panel each.
        units: list[int] | None - engines to overlay within each panel.

    Returns:
        Figure - a grid of panels, two columns wide.
    """
    sensors = sensors or DEFAULT_SENSORS
    units = units or DEFAULT_UNITS

    # Two columns, and however many rows that needs (ceiling division).
    column_count = 2 if len(sensors) > 1 else 1
    row_count = -(-len(sensors) // column_count)
    figure, panel_grid = plt.subplots(
        row_count,
        column_count,
        figsize=(6 * column_count, 3.5 * row_count),
        squeeze=False,
    )

    # One panel per sensor, with every requested engine drawn on it.
    panels = panel_grid.flat
    for panel, sensor in zip(panels, sensors, strict=False):
        for unit in units:
            unit_history = fleet[fleet.unit == unit]
            panel.plot(unit_history.cycle, unit_history[sensor],
                       alpha=0.7, label=f"unit {unit}")
        panel.set_title(sensor)
        panel.set_xlabel("cycle")

    # Hide the trailing empty panel left over when the sensor count is odd.
    for panel in list(panels)[len(sensors):]:
        panel.set_visible(False)

    panel_grid.flat[0].legend()
    figure.tight_layout()
    return figure
