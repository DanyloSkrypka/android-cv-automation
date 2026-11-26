"""Device layer: one small interface, two implementations.

AdbDevice  - a real phone, emulator or cloud phone reachable through `adb -s SERIAL`.
FakeDevice - an in-memory "phone" that shows pre-drawn screens; used by tests and the offline demo.

All coordinates are FRACTIONS of the screen (0..1), so one script works on 720p, 1080p and 1440p devices.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import cv2
import numpy as np


class Device(Protocol):
    serial: str

    def screenshot(self) -> np.ndarray: ...
    def tap(self, fx: float, fy: float) -> None: ...
    def swipe(self, fx1: float, fy1: float, fx2: float, fy2: float, ms: int = 300) -> None: ...


class DeviceError(RuntimeError):
    pass


class AdbDevice:
    """Thin, defensive wrapper around the adb CLI.

    - every call has a timeout (a hung adb must not freeze the whole fleet)
    - screenshots are retried, then raise DeviceError so the runner can restart the device
    - landscape/portrait is handled by reading the real frame size
    """

    def __init__(self, serial: str, adb: str | None = None, shot_timeout: float = 3.0, retries: int = 3):
        self.serial = serial
        self.adb = adb or shutil.which("adb") or "adb"
        self.shot_timeout = shot_timeout
        self.retries = retries
        self.size: tuple[int, int] | None = None  # (w, h) of the last frame

    def _run(self, *args: str, timeout: float = 15.0) -> bytes:
        try:
            r = subprocess.run([self.adb, "-s", self.serial, *args], capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise DeviceError(f"{self.serial}: adb {' '.join(args[:2])} timed out after {timeout}s") from e
        if r.returncode != 0:
            raise DeviceError(f"{self.serial}: adb {' '.join(args[:2])} failed: {r.stderr.decode(errors='ignore')[:200]}")
        return r.stdout

    def screenshot(self) -> np.ndarray:
        last = None
        for _ in range(self.retries):
            try:
                raw = self._run("exec-out", "screencap", "-p", timeout=self.shot_timeout)
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    self.size = (img.shape[1], img.shape[0])
                    return img
                last = DeviceError(f"{self.serial}: empty screenshot")
            except DeviceError as e:
                last = e
            time.sleep(0.3)
        raise last or DeviceError(f"{self.serial}: screenshot failed")

    def _px(self, fx: float, fy: float) -> tuple[int, int]:
        if self.size is None:
            self.screenshot()
        w, h = self.size  # type: ignore[misc]
        return int(fx * w), int(fy * h)

    def tap(self, fx: float, fy: float) -> None:
        x, y = self._px(fx, fy)
        self._run("shell", "input", "tap", str(x), str(y))

    def swipe(self, fx1: float, fy1: float, fx2: float, fy2: float, ms: int = 300) -> None:
        x1, y1 = self._px(fx1, fy1)
        x2, y2 = self._px(fx2, fy2)
        self._run("shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(ms))

    def key(self, keycode: str) -> None:
        self._run("shell", "input", "keyevent", keycode)

    def start_activity(self, action_or_component: str) -> None:
        flag = "-n" if "/" in action_or_component else "-a"
        self._run("shell", "am", "start", flag, action_or_component)

    @staticmethod
    def list_serials(adb: str | None = None) -> list[str]:
        out = subprocess.run([adb or shutil.which("adb") or "adb", "devices"], capture_output=True, text=True, timeout=10).stdout
        return [ln.split("\t")[0] for ln in out.splitlines()[1:] if ln.endswith("\tdevice")]


@dataclass
class FakeDevice:
    """A scripted phone for tests and demos.

    screens: name -> image. `on_tap(device, fx, fy)` decides which screen comes next,
    so the fake behaves like a tiny state machine (login -> loading -> home ...).
    latency / fail_rate let you simulate slow or flaky devices.
    """

    serial: str
    screens: dict[str, np.ndarray]
    current: str
    on_tap: Callable[["FakeDevice", float, float], None] | None = None
    latency: float = 0.0
    taps: list[tuple[float, float]] = field(default_factory=list)
    pending: list[tuple[float, str]] = field(default_factory=list)  # (time, screen) scheduled transitions

    def go(self, screen: str, after: float = 0.0) -> None:
        if after <= 0:
            self.current = screen
        else:
            self.pending.append((time.monotonic() + after, screen))

    def screenshot(self) -> np.ndarray:
        now = time.monotonic()
        due = [p for p in self.pending if p[0] <= now]
        if due:
            self.current = max(due)[1]
            self.pending = [p for p in self.pending if p[0] > now]
        if self.latency:
            time.sleep(self.latency)
        return self.screens[self.current].copy()

    def tap(self, fx: float, fy: float) -> None:
        self.taps.append((fx, fy))
        if self.on_tap:
            self.on_tap(self, fx, fy)

    def swipe(self, fx1: float, fy1: float, fx2: float, fy2: float, ms: int = 300) -> None:
        self.taps.append((fx2, fy2))
