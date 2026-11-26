"""A flow is a list of named steps run against one device.

Each step gets a Ctx (device + helpers) and either returns normally or raises.
The runner records per-step timing, and on failure saves the last screenshot,
so a failed run on device #23 at 03:00 can be understood in the morning.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from .device import Device
from .vision import Match


class StepFailed(RuntimeError):
    pass


@dataclass
class Ctx:
    device: Device
    out_dir: str
    variant: str = "A"                   # A/B branch assigned by the fleet runner
    run_id: int = 0
    last_shot: np.ndarray | None = None
    data: dict = field(default_factory=dict)

    def shot(self) -> np.ndarray:
        self.last_shot = self.device.screenshot()
        return self.last_shot

    def wait_for(self, check: Callable[[np.ndarray], Match | list | bool | None], timeout: float = 10.0,
                 poll: float = 0.25, what: str = "condition") -> Match | list | bool:
        """Take screenshots until `check(screen)` returns something truthy, or raise StepFailed."""
        end = time.monotonic() + timeout
        while True:
            res = check(self.shot())
            if res:
                return res
            if time.monotonic() >= end:
                raise StepFailed(f"timeout {timeout:.0f}s waiting for {what}")
            time.sleep(poll)

    def tap_match(self, m: Match | list) -> None:
        m = m[0] if isinstance(m, list) else m
        self.device.tap(*m.center)


Step = tuple[str, Callable[[Ctx], None]]


@dataclass
class RunResult:
    serial: str
    run_id: int
    variant: str
    ok: bool
    seconds: float
    steps: list[tuple[str, float]]
    error: str = ""
    failure_shot: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def run_flow(device: Device, steps: list[Step], out_dir: str, run_id: int = 0, variant: str = "A") -> RunResult:
    os.makedirs(out_dir, exist_ok=True)
    ctx = Ctx(device, out_dir, variant, run_id)
    t0 = time.monotonic()
    timings: list[tuple[str, float]] = []
    for name, fn in steps:
        ts = time.monotonic()
        try:
            fn(ctx)
        except Exception as e:  # any error = failed run; the fleet decides whether to retry
            shot_path = ""
            if ctx.last_shot is not None:
                shot_path = os.path.join(out_dir, f"fail_{device.serial.replace(':', '_')}_{run_id}_{name}.png")
                cv2.imwrite(shot_path, ctx.last_shot)
            timings.append((name, round(time.monotonic() - ts, 3)))
            return RunResult(device.serial, run_id, variant, False, round(time.monotonic() - t0, 3), timings,
                             f"{name}: {type(e).__name__}: {e}", shot_path)
        timings.append((name, round(time.monotonic() - ts, 3)))
    return RunResult(device.serial, run_id, variant, True, round(time.monotonic() - t0, 3), timings)
