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
    """Healthy norm and failure calibration, derived from failed engines.

    Fields:
        healthy_mean: mean value per sensor across healthy early cycles.
        healthy_sd: standard deviation per sensor over the same cycles.
        failure_score: raw score comparable engines reached at failure.
        sensors: sensor columns the baseline covers.
        window: how many recent cycles a score averages over.
    """

    healthy_mean: pd.Series
    healthy_sd: pd.Series
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

    Args:
        failed: pd.DataFrame - run-to-failure histories, one row per cycle.
        sensors: list[str] | None - sensor columns to score, defaults to
            LIVE_SENSORS.
        healthy_cycles: int - how many opening cycles count as healthy.
        window: int - how many recent cycles a score averages over.

    Returns:
        Baseline - the healthy norm plus the failure-point calibration.
    """
    sensors = list(sensors) if sensors is not None else list(LIVE_SENSORS)

    # The healthy norm: pool the opening cycles of every engine.
    early = failed[failed.cycle <= healthy_cycles]
    healthy_mean = early[sensors].mean()
    healthy_sd = early[sensors].std()

    # Score every engine over its final cycles, i.e. at the point it failed.
    # This is the scale, not a threshold - it is never surfaced to the user.
    scores_at_failure = [
        score_against_norm(
            failed[failed.unit == unit], healthy_mean, healthy_sd, sensors, window
        )
        for unit in failed.unit.unique()
    ]
    failure_score = float(pd.Series(scores_at_failure).mean())

    return Baseline(
        healthy_mean=healthy_mean,
        healthy_sd=healthy_sd,
        failure_score=failure_score,
        sensors=sensors,
        window=window,
    )


def score_against_norm(
    history: pd.DataFrame,
    healthy_mean: pd.Series,
    healthy_sd: pd.Series,
    sensors: list[str],
    window: int,
) -> float:
    """Mean absolute z-score of an engine's recent cycles against a healthy norm.

    Takes the norm as loose arguments rather than a `Baseline` so
    `build_baseline` can call it while the baseline is still being assembled.

    Args:
        history: pd.DataFrame - one engine's rows, oldest cycle first.
        healthy_mean: pd.Series - mean per sensor when healthy.
        healthy_sd: pd.Series - standard deviation per sensor when healthy.
        sensors: list[str] - sensor columns to score.
        window: int - how many trailing cycles to average.

    Returns:
        float - deviation from healthy, in standard deviations.
    """
    recent = history.tail(window)[sensors].mean()
    return float(((recent - healthy_mean) / healthy_sd).abs().mean())


def deviations(history: pd.DataFrame, baseline: Baseline) -> pd.Series:
    """Per-sensor z-score of the recent window against the healthy norm."""
    recent = recent_means(history, baseline)
    return (recent - baseline.healthy_mean) / baseline.healthy_sd


def recent_means(history: pd.DataFrame, baseline: Baseline) -> pd.Series:
    """Mean sensor values over the engine's most recent cycles."""
    return history.tail(baseline.window)[baseline.sensors].mean()


def raw_score(history: pd.DataFrame, baseline: Baseline) -> float:
    """Mean absolute deviation from the healthy norm, in standard deviations."""
    return score_against_norm(
        history,
        baseline.healthy_mean,
        baseline.healthy_sd,
        baseline.sensors,
        baseline.window,
    )


def wear_index(history: pd.DataFrame, baseline: Baseline) -> float:
    """Wear on a 0-100 scale, where 100 is the level at which comparable
    engines failed.

    Not clamped at the bottom: a new engine reads in the teens rather than 0,
    because sensors are noisy and units vary at manufacture. Clamping flattened
    a third of the healthy fleet to exactly zero and broke the ranking.
    """
    return min(100.0, raw_score(history, baseline) / baseline.failure_score * 100)


def verdict(wear_pct: float) -> str:
    """Plain-language reading of a wear index."""
    if wear_pct < 25:
        return "healthy"
    if wear_pct < 50:
        return "early wear"
    if wear_pct < 70:
        return "significant wear, schedule inspection"
    return "close to failure, act now"
