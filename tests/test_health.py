import numpy as np
import pytest

from industrial_agent.data import LIVE_SENSORS
from industrial_agent.health import (
    build_baseline,
    deviations,
    raw_score,
    verdict,
    wear_index,
)


def test_baseline_covers_every_live_sensor(baseline):
    assert baseline.sensors == LIVE_SENSORS
    assert set(baseline.healthy_mean.index) == set(LIVE_SENSORS)
    assert baseline.failure_score > 0


def test_baseline_excludes_dead_sensors(baseline):
    """Dead sensors have sd = 0 and would produce infinite z-scores."""
    assert "sensor_1" not in baseline.sensors
    assert "sensor_6" not in baseline.sensors
    assert (baseline.healthy_sd > 0).all()


def test_wear_rises_towards_failure(failed, baseline):
    """The whole premise: an engine reads worse near the end of its life."""
    unit = failed[failed.unit == failed.unit.iloc[0]]
    early = wear_index(unit[unit.cycle <= 30], baseline)
    late = wear_index(unit, baseline)
    assert late > early


def test_wear_index_is_bounded_above(failed, baseline):
    for u in failed.unit.unique():
        assert wear_index(failed[failed.unit == u], baseline) <= 100.0


def test_deviations_are_finite(fleet, baseline):
    z = deviations(fleet[fleet.unit == 1], baseline)
    assert np.isfinite(z.to_numpy()).all()


def test_raw_score_of_healthy_window_is_small(failed, baseline):
    """First cycles are the baseline itself, so deviation should be near zero."""
    unit = failed[failed.unit == failed.unit.iloc[0]]
    assert raw_score(unit[unit.cycle <= 20], baseline) < 1.5


@pytest.mark.parametrize(
    "pct,expected",
    [(0, "healthy"), (24.9, "healthy"), (25, "early wear"),
     (49.9, "early wear"), (50, "significant wear, schedule inspection"),
     (69.9, "significant wear, schedule inspection"),
     (70, "close to failure, act now"), (100, "close to failure, act now")],
)
def test_verdict_thresholds(pct, expected):
    assert verdict(pct) == expected


def test_window_size_is_honoured(failed):
    b = build_baseline(failed, window=3)
    assert b.window == 3
