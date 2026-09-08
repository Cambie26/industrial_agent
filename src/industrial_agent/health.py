"""
Health scoring: turning 14 noisy sensor traces into one number an engineer
can act on.

Pure numerics - no file I/O, no network, no module-level state. Everything
takes a `Baseline` so the same maths runs against the real fleet or a test
fixture.

The method, in three steps:
  1. Pool the first 20 cycles of every run-to-failure engine to get a healthy
     norm (mean and sd per sensor).
  2. Score an engine as the mean absolute z-score of its last 10 cycles.
  3. Divide by the score comparable engines reached at failure, so 100 means
     "as worn as engines that failed".
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data import LIVE_SENSORS

HEALTHY_CYCLES = 20
WINDOW = 10


@dataclass(frozen=True)
class Baseline:
    """Healthy norm and failure calibration, derived from failed engines."""

    mu: pd.Series
    sd: pd.Series
    failure_score: float
    sensors: list[str]
    window: int = WINDOW


def build_baseline(
    failed: pd.DataFrame,
    sensors: list[str] | None = None,
    healthy_cycles: int = HEALTHY_CYCLES,
    window: int = WINDOW,
) -> Baseline:
    """Calibrate against engines whose full run-to-failure history is known.

    Pooling the early cycles of every engine beats a per-unit baseline here:
    it gives a stable noise estimate, and it works for in-service units with
    short histories.
    """
    sensors = list(sensors) if sensors is not None else list(LIVE_SENSORS)

    early = failed[failed.cycle <= healthy_cycles]
    mu = early[sensors].mean()
    sd = early[sensors].std()

    # What the statistic reads at the moment of failure, averaged over every
    # engine that got there. This is the scale, not a threshold - it is never
    # surfaced to the user.
    at_failure = [
        _raw_score(failed[failed.unit == u], mu, sd, sensors, window)
        for u in failed.unit.unique()
    ]
    failure_score = float(pd.Series(at_failure).mean())

    return Baseline(mu=mu, sd=sd, failure_score=failure_score,
                    sensors=sensors, window=window)


def _raw_score(
    history: pd.DataFrame,
    mu: pd.Series,
    sd: pd.Series,
    sensors: list[str],
    window: int,
) -> float:
    recent = history.tail(window)[sensors].mean()
    return float(((recent - mu) / sd).abs().mean())


def deviations(history: pd.DataFrame, baseline: Baseline) -> pd.Series:
    """Per-sensor z-score of the recent window against the healthy norm."""
    recent = recent_means(history, baseline)
    return (recent - baseline.mu) / baseline.sd


def recent_means(history: pd.DataFrame, baseline: Baseline) -> pd.Series:
    """Mean sensor values over the engine's most recent cycles."""
    return history.tail(baseline.window)[baseline.sensors].mean()


def raw_score(history: pd.DataFrame, baseline: Baseline) -> float:
    """Mean absolute deviation from the healthy norm, in standard deviations."""
    return _raw_score(history, baseline.mu, baseline.sd,
                      baseline.sensors, baseline.window)


def wear_index(history: pd.DataFrame, baseline: Baseline) -> float:
    """Wear on a 0-100 scale, where 100 is the level at which comparable
    engines failed.

    Not clamped at the bottom: a new engine reads in the teens rather than 0,
    because sensors are noisy and units vary at manufacture. Clamping flattened
    a third of the healthy fleet to exactly zero and broke the ranking.
    """
    return min(100.0, raw_score(history, baseline) / baseline.failure_score * 100)


def verdict(pct: float) -> str:
    """Plain-language reading of a wear index."""
    if pct < 25:
        return "healthy"
    if pct < 50:
        return "early wear"
    if pct < 70:
        return "significant wear, schedule inspection"
    return "close to failure, act now"
