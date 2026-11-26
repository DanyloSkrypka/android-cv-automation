"""Real-device smoke test: open Android Settings on every connected device, scroll, go home.

    python examples/settings_smoke.py                 # all devices from `adb devices`
    python examples/settings_smoke.py --serial emulator-5554 --runs 3

Uses only stock Android (no app needed), so it is a safe first check that adb, screenshots,
taps and the parallel runner work on your fleet. Run it only on devices you own/manage.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acva import AdbDevice, Ctx, StepFailed, run_fleet  # noqa: E402


def changed(a: np.ndarray, b: np.ndarray, min_share: float = 0.05) -> bool:
    """True if more than `min_share` of the screen differs - 'did my action do anything?'"""
    if a.shape != b.shape:
        return True
    d = cv2.absdiff(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    return np.count_nonzero(d > 25) / d.size > min_share


def home(ctx: Ctx):
    ctx.device.key("KEYCODE_HOME")
    ctx.data["home"] = ctx.shot()


def open_settings(ctx: Ctx):
    ctx.device.start_activity("android.settings.SETTINGS")
    ctx.wait_for(lambda s: changed(ctx.data["home"], s), timeout=8, what="Settings to open")


def scroll(ctx: Ctx):
    before = ctx.shot()
    ctx.device.swipe(0.5, 0.75, 0.5, 0.30, 350)
    ctx.wait_for(lambda s: changed(before, s, 0.02), timeout=4, what="list to scroll")
    cv2.imwrite(os.path.join(ctx.out_dir, f"settings_{ctx.device.serial.replace(':', '_')}_{ctx.run_id}.png"), ctx.last_shot)


def back_home(ctx: Ctx):
    ctx.device.key("KEYCODE_HOME")
    if changed(ctx.data["home"], ctx.shot(), 0.20):
        raise StepFailed("did not return to the home screen")


STEPS = [("home", home), ("open_settings", open_settings), ("scroll", scroll), ("back_home", back_home)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", action="append", help="device serial (repeat for several); default: all")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--out", default="logs")
    a = ap.parse_args()
    serials = a.serial or AdbDevice.list_serials()
    if not serials:
        sys.exit("no devices: check `adb devices`")
    rep = run_fleet([AdbDevice(s) for s in serials], STEPS, a.runs, a.out)
    print(rep.summary())
