"""
Loading the NASA C-MAPSS turbofan degradation dataset (subset FD001).

Two files matter:
  train_FD001.txt - 100 engines run to failure, complete histories
  test_FD001.txt  - 100 engines still in service, histories truncated

Nothing here is loaded at import time. The dataset is ~4 MB and lives in a
gitignored `data/` directory, so importing the package must work without it.
"""

from __future__ import annotations

import os
import urllib.request
from functools import lru_cache
from pathlib import Path

import pandas as pd

BASE_URL = (
    "https://raw.githubusercontent.com/hankroark/"
    "Turbofan-Engine-Degradation/master/CMAPSSData"
)
FILES = ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")

COLS: list[str] = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Sensors 1, 5, 10, 16, 18 and 19 are constant across FD001. Sensor 6 is
# effectively binary (sd = 0.002), which makes it useless as a signal and
# numerically dangerous as a z-score denominator.
DEAD_SENSORS: list[int] = [1, 5, 6, 10, 16, 18, 19]
LIVE_SENSORS: list[str] = [
    f"sensor_{i}" for i in range(1, 22) if i not in DEAD_SENSORS
]


def data_dir() -> Path:
    """Where the dataset lives. Override with INDUSTRIAL_AGENT_DATA."""
    return Path(os.environ.get("INDUSTRIAL_AGENT_DATA", "data"))


def download(dest: Path | None = None) -> Path:
    """Fetch the FD001 files if they are not already on disk. Idempotent."""
    dest = Path(dest) if dest is not None else data_dir()
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = dest / name
        if not target.exists():
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", target)
    return dest


def read_cmapss(path: str | Path) -> pd.DataFrame:
    """Read one whitespace-delimited C-MAPSS file into a labelled frame."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS)


@lru_cache(maxsize=8)
def _load_cached(path_str: str) -> pd.DataFrame:
    return read_cmapss(path_str)


def load_failed(dest: Path | None = None) -> pd.DataFrame:
    """The 100 engines that ran to failure. Used to calibrate health scoring."""
    dest = Path(dest) if dest is not None else data_dir()
    return _load_cached(str(dest / "train_FD001.txt"))


def load_fleet(dest: Path | None = None) -> pd.DataFrame:
    """The 100 engines currently in service. What the agent is asked about."""
    dest = Path(dest) if dest is not None else data_dir()
    return _load_cached(str(dest / "test_FD001.txt"))
