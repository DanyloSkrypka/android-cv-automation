"""A drawn "shop app" with 4 screens, rendered at any resolution.

Used by the offline demo and tests, so the project runs without a phone.
login  -> tap "Sign in" -> loading -> home (green "Online" badge)
                                   \\-> error banner (red) with "Retry" -> loading -> home
"""
from __future__ import annotations

import random

import cv2
import numpy as np

from acva.device import FakeDevice

# element boxes in fractions (x1, y1, x2, y2) - the fake uses them to hit-test taps
SIGN_IN = (0.15, 0.62, 0.85, 0.70)
RETRY = (0.62, 0.135, 0.92, 0.185)

WHITE, GREY, DARK = (250, 250, 250), (225, 225, 225), (40, 40, 40)
BLUE, GREEN, RED = (230, 120, 30), (80, 190, 60), (60, 60, 220)  # BGR


def _box(img, f, color, text="", text_color=(255, 255, 255)):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = int(f[0] * w), int(f[1] * h), int(f[2] * w), int(f[3] * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
    if text:
        scale = (y2 - y1) / 50
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, int(scale * 2)))
        cv2.putText(img, text, (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, max(1, int(scale * 2)), cv2.LINE_AA)


def _text(img, fx, fy, text, size=1.0, color=DARK):
    h, w = img.shape[:2]
    s = size * w / 720
    cv2.putText(img, text, (int(fx * w), int(fy * h)), cv2.FONT_HERSHEY_SIMPLEX, s, color, max(1, int(s * 2)), cv2.LINE_AA)


def render(w: int = 1080, h: int = 1920) -> dict[str, np.ndarray]:
    login = np.full((h, w, 3), WHITE, np.uint8)
    _text(login, 0.15, 0.25, "ShopDemo", 2.0)
    _box(login, (0.15, 0.40, 0.85, 0.46), GREY, "email", DARK)
    _box(login, (0.15, 0.50, 0.85, 0.56), GREY, "password", DARK)
    _box(login, SIGN_IN, BLUE, "Sign in")

    loading = np.full((h, w, 3), WHITE, np.uint8)
    cv2.circle(loading, (w // 2, h // 2), w // 10, GREY, max(2, w // 60), cv2.LINE_AA)
    _text(loading, 0.36, 0.62, "Loading...", 1.0)

    home = np.full((h, w, 3), WHITE, np.uint8)
    _text(home, 0.06, 0.07, "Orders", 1.4)
    _box(home, (0.70, 0.035, 0.94, 0.075), GREEN, "Online")
    for i in range(6):
        _box(home, (0.06, 0.12 + i * 0.12, 0.94, 0.21 + i * 0.12), GREY, f"Order #{1040 + i}", DARK)

    error = home.copy()
    _box(error, (0.0, 0.11, 1.0, 0.21), RED)
    _text(error, 0.06, 0.17, "Network error", 1.0, (255, 255, 255))
    _box(error, RETRY, BLUE, "Retry")
    return {"login": login, "loading": loading, "home": home, "error": error}


def _inside(fx, fy, box):
    return box[0] <= fx <= box[2] and box[1] <= fy <= box[3]


def make_device(serial: str, w: int, h: int, error_rate: float = 0.25, load_s: float = 0.3,
                frozen: bool = False, seed: int | None = None) -> FakeDevice:
    """frozen=True simulates a hung device (stuck on loading forever)."""
    rnd = random.Random(seed if seed is not None else serial)

    def on_tap(d: FakeDevice, fx: float, fy: float) -> None:
        if d.current == "login" and _inside(fx, fy, SIGN_IN):
            d.go("loading")
            if not frozen:
                d.go("error" if rnd.random() < error_rate else "home", after=load_s)
        elif d.current == "error" and _inside(fx, fy, RETRY):
            d.go("loading")
            d.go("home", after=load_s)

    return FakeDevice(serial, render(w, h), "login", on_tap)
