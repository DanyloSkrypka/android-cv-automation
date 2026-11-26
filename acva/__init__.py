"""acva - Android computer-vision automation: screenshots in, taps out, logs for everything."""
from .device import AdbDevice, DeviceError, FakeDevice
from .flow import Ctx, RunResult, StepFailed, run_flow
from .fleet import FleetReport, run_fleet
from .vision import Match, color_share, find_color, find_template

__all__ = ["AdbDevice", "DeviceError", "FakeDevice", "Ctx", "RunResult", "StepFailed", "run_flow",
           "FleetReport", "run_fleet", "Match", "color_share", "find_color", "find_template"]
