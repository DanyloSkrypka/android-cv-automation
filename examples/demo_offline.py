"""Offline demo: 6 simulated phones (3 resolutions, 1 of them hangs), one flow, A/B test.

    python examples/demo_offline.py            # ~20 s, writes logs/runs.jsonl, logs/ab_report.json, logs/detection.png
    python examples/demo_offline.py --picture docs/detection.png   # refresh the README picture

Variant A: if the app shows a "Network error" banner, the run fails.
Variant B: the bot recognises the red banner, taps "Retry" and continues.
The A/B report shows whether B is really better, with a significance test.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from acva import Ctx, StepFailed, find_color, find_template, run_fleet  # noqa: E402
from acva.ab import compare  # noqa: E402
from acva.vision import BLUE, GREEN, RED_LOW, crop  # noqa: E402
from fake_app import SIGN_IN, make_device, render  # noqa: E402

# Template cut ONCE from a 1080x1920 screenshot; reused on 720p and 1440p phones (multi-scale matching).
_ref = render(1080, 1920)["login"]
SIGN_IN_TPL, _, _ = crop(_ref, SIGN_IN)


def home_badge(s):
    return find_color(s, *GREEN, roi=(0.6, 0.0, 1.0, 0.1))


def error_banner(s):
    return find_color(s, *RED_LOW, roi=(0.0, 0.09, 1.0, 0.23), min_area=2000)


# ---- flow steps -------------------------------------------------------------
def reset(ctx: Ctx):
    # on a real phone: `am force-stop` + `am start` the app under test
    ctx.device.pending.clear()
    ctx.device.go("login")


def tap_sign_in(ctx: Ctx):
    m = ctx.wait_for(lambda s: find_template(s, SIGN_IN_TPL, 0.8), timeout=3, what="'Sign in' button")
    ctx.tap_match(m)


def wait_result(ctx: Ctx):
    ctx.wait_for(lambda s: home_badge(s) or error_banner(s), timeout=3, what="home screen or error banner")
    if error_banner(ctx.last_shot):
        if ctx.variant == "A":
            raise StepFailed("network error banner")
        btn = ctx.wait_for(lambda s: find_color(s, *BLUE, roi=(0.55, 0.11, 1.0, 0.21)), timeout=2, what="Retry button")
        ctx.tap_match(btn)


def check_home(ctx: Ctx):
    ctx.wait_for(home_badge, timeout=3, what="green 'Online' badge")


STEPS = [("reset", reset), ("tap_sign_in", tap_sign_in), ("wait_result", wait_result), ("check_home", check_home)]


def save_detection_picture(path: str):
    """Side-by-side picture for the README: template found on a 720p screen + colour badge on 1440p."""
    a = render(720, 1280)["login"]
    m = find_template(a, SIGN_IN_TPL, 0.8)
    cv2.rectangle(a, (m.x, m.y), (m.x + m.w, m.y + m.h), (0, 0, 255), 4)
    cv2.putText(a, f"template {m.score:.2f}", (m.x, m.y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    b = render(1440, 2560)["error"]
    for blob, label in ((home_badge(b)[0], "green badge"), (error_banner(b)[0], "red banner")):
        cv2.rectangle(b, (blob.x, blob.y), (blob.x + blob.w, blob.y + blob.h), (0, 200, 255), 8)
        cv2.putText(b, label, (blob.x + 10, blob.y + blob.h + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 140, 255), 4)
    b = cv2.resize(b, (720, 1280))
    pic = np.hstack([a, np.full((1280, 20, 3), 255, np.uint8), b])
    cv2.imwrite(path, cv2.resize(pic, (730, 637)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20, help="runs per device")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "logs"))
    ap.add_argument("--picture", default=None, help="where to save the detection picture (default: <out>/detection.png)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    open(os.path.join(args.out, "runs.jsonl"), "w").close()   # fresh log for the demo

    devices = [
        make_device("emu-720p-1", 720, 1280), make_device("emu-720p-2", 720, 1280),
        make_device("phone-1080p-1", 1080, 1920), make_device("phone-1080p-2", 1080, 1920),
        make_device("tablet-1440p", 1440, 2560),
        make_device("cloud-frozen", 1080, 1920, frozen=True),  # hangs on loading -> disabled after 3 fails
    ]
    rep = run_fleet(devices, STEPS, args.runs, args.out, variants=("A", "B"), max_consecutive_fail=3)
    print(rep.summary())
    ab = compare([r.__dict__ for r in rep.results if r.serial not in rep.disabled])
    a, b, t = ab["A"], ab["B"], ab["test"]
    print(f"\nA/B (A = fail on error banner, B = tap Retry):  A {100 * a['rate']:.1f}% of {a['runs']}"
          f" · B {100 * b['rate']:.1f}% of {b['runs']}  ->  z {t['z']}, p {t['p']}, {t['verdict']}")
    with open(os.path.join(args.out, "ab_report.json"), "w", encoding="utf-8") as f:
        json.dump(ab, f, indent=2)
    fails = [r for r in rep.results if not r.ok]
    if fails:
        print(f"\nexample failure: {fails[0].serial} run {fails[0].run_id}: {fails[0].error}\n  screenshot: {os.path.relpath(fails[0].failure_shot)}")
    save_detection_picture(args.picture or os.path.join(args.out, "detection.png"))


if __name__ == "__main__":
    main()
