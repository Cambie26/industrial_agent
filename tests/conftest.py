"""Fixtures sliced from FD001: 2 run-to-failure engines, 3 in service.

The failed engines keep their full histories - the failure calibration is
meaningless without the final cycles.
"""

from pathlib import Path

import pytest

from industrial_agent.data import read_cmapss
from industrial_agent.health import build_baseline
from industrial_agent.tools import FleetTools

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def failed():
    return read_cmapss(FIXTURES / "failed_sample.txt")


@pytest.fixture(scope="session")
def fleet():
    return read_cmapss(FIXTURES / "fleet_sample.txt")


@pytest.fixture(scope="session")
def baseline(failed):
    return build_baseline(failed)


@pytest.fixture()
def tools(fleet, baseline):
    return FleetTools(fleet, baseline)
