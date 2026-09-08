"""An agentic maintenance analyst for a fleet of turbofan engines."""

from .agent import MODEL, SYSTEM, Agent, Event
from .chat import Chat, default_chat
from .data import LIVE_SENSORS, download, load_failed, load_fleet
from .health import Baseline, build_baseline, verdict, wear_index
from .plots import sensor_traces
from .tools import TOOL_SCHEMAS, FleetTools

__version__ = "0.1.0"

__all__ = [
    "Agent", "Event", "MODEL", "SYSTEM",
    "Chat", "default_chat",
    "download", "load_failed", "load_fleet", "LIVE_SENSORS",
    "Baseline", "build_baseline", "wear_index", "verdict",
    "FleetTools", "TOOL_SCHEMAS",
    "sensor_traces",
    "__version__",
]
