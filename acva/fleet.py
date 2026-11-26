"""Run one flow on many devices in parallel.

- one worker thread per device (adb calls are I/O-bound, threads are enough)
- every run is appended to logs/runs.jsonl (one JSON per line - easy to grep, load into pandas or a dashboard)
- a device that hangs (`max_consecutive_fail` timeouts/adb errors in a row) is taken out of rotation, the rest keep going
- A/B: each run gets a variant from `ab.assign(run_id)`, so experiments run on the whole fleet at once
"""
from __future__ import annotations

import itertools
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from .ab import assign
from .device import Device
from .flow import RunResult, Step, run_flow


@dataclass
class FleetReport:
    results: list[RunResult] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)

    def summary(self) -> str:
        by_dev: dict[str, list[RunResult]] = {}
        for r in self.results:
            by_dev.setdefault(r.serial, []).append(r)
        lines = [f"{'device':16} {'runs':>5} {'ok':>5} {'ok%':>6} {'avg s':>7}"]
        for s, rs in sorted(by_dev.items()):
            ok = sum(r.ok for r in rs)
            avg = sum(r.seconds for r in rs) / len(rs)
            flag = "  DISABLED" if s in self.disabled else ""
            lines.append(f"{s:16} {len(rs):5} {ok:5} {100 * ok / len(rs):5.1f}% {avg:7.2f}{flag}")
        tot = len(self.results)
        ok = sum(r.ok for r in self.results)
        lines.append(f"{'TOTAL':16} {tot:5} {ok:5} {100 * ok / max(1, tot):5.1f}%")
        return "\n".join(lines)


def is_device_fault(res: RunResult) -> bool:
    """A hung/offline device (timeouts, adb errors) - not a normal test failure like 'error banner shown'.
    Only device faults count towards taking a device out of rotation."""
    return not res.ok and ("timeout" in res.error or "DeviceError" in res.error)


def run_fleet(devices: list[Device], steps: list[Step], runs_per_device: int, out_dir: str = "logs",
              variants: tuple[str, ...] = ("A",), max_consecutive_fail: int = 3,
              on_result: Callable[[RunResult], None] | None = None, pause: float = 0.0) -> FleetReport:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "runs.jsonl")
    lock = threading.Lock()
    counter = itertools.count(1)          # global run id -> deterministic A/B assignment
    report = FleetReport()

    def worker(dev: Device) -> None:
        fails = 0
        for _ in range(runs_per_device):
            with lock:
                run_id = next(counter)
            variant = assign(run_id, variants)
            res = run_flow(dev, steps, out_dir, run_id, variant)
            with lock:
                report.results.append(res)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(res.to_json() + "\n")
            if on_result:
                on_result(res)
            fails = fails + 1 if is_device_fault(res) else 0
            if fails >= max_consecutive_fail:
                with lock:
                    report.disabled.append(dev.serial)
                return
            if pause:
                time.sleep(pause)

    with ThreadPoolExecutor(max_workers=max(1, len(devices))) as ex:
        list(ex.map(worker, devices))
    return report
