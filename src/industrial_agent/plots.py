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
    to the eye long before failure."""
    sensors = sensors or DEFAULT_SENSORS
    units = units or DEFAULT_UNITS

    cols = 2 if len(sensors) > 1 else 1
    rows = -(-len(sensors) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows),
                             squeeze=False)
    flat = axes.flat
    for ax, sensor in zip(flat, sensors, strict=False):
        for unit in units:
            d = fleet[fleet.unit == unit]
            ax.plot(d.cycle, d[sensor], alpha=0.7, label=f"unit {unit}")
        ax.set_title(sensor)
        ax.set_xlabel("cycle")
    for ax in list(flat)[len(sensors):]:   # trailing empty panel, odd count
        ax.set_visible(False)
    axes.flat[0].legend()
    fig.tight_layout()
    return fig
