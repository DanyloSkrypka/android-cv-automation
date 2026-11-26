"""Render docs/demo.gif from REAL runs of the library (no hand-made numbers).

    pip install pillow            # only needed for this script
    python tools/make_demo_gif.py

Part 1: one flow on a simulated 720p phone, variant B: template match -> tap -> error banner
        detected by colour -> tap Retry -> green badge. Scores, taps and timings come from the code.
Part 2: the same flow on 6 simulated devices in parallel (one of them hangs), live table + A/B result.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

from acva import find_color, find_template, run_flow, run_fleet  # noqa: E402
from acva.ab import compare  # noqa: E402
from acva.vision import BLUE  # noqa: E402
import demo_offline as demo  # noqa: E402
from fake_app import make_device, render  # noqa: E402

W, H = 960, 540
BG, PANEL, FG, DIM = (34, 30, 28), (48, 43, 40), (235, 235, 235), (150, 150, 150)
OK, BAD, ACC, WARN = (110, 200, 90), (80, 80, 235), (60, 200, 255), (0, 165, 255)  # BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX
PH_X, PH_Y, PH_W, PH_H = 48, 30, 270, 480          # phone screen on the canvas
SCALE = PH_W / 720

frames: list[tuple[np.ndarray, int]] = []           # (image, duration ms)


def text(img, s, x, y, size=0.55, color=FG, bold=1):
    cv2.putText(img, s, (x, y), FONT, size, color, bold, cv2.LINE_AA)


def canvas(title: str) -> np.ndarray:
    c = np.full((H, W, 3), BG, np.uint8)
    text(c, title, 360, 52, 0.75, FG, 2)
    return c


def phone(c: np.ndarray, screen: np.ndarray, boxes=(), tap=None, tap_r=0):
    s = cv2.resize(screen, (PH_W, PH_H), interpolation=cv2.INTER_AREA)
    for (m, label, color, *below) in boxes:
        x1, y1 = int(m.x * SCALE), int(m.y * SCALE)
        x2, y2 = int((m.x + m.w) * SCALE), int((m.y + m.h) * SCALE)
        cv2.rectangle(s, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.42, 1)
        ly = y2 + th + 8 if below or y1 <= 20 else y1 - 6
        x1 = min(x1 + (3 if below else 0), PH_W - tw - 8)
        cv2.rectangle(s, (x1, ly - th - 4), (x1 + tw + 6, ly + 3), color, -1)
        text(s, label, x1 + 3, ly, 0.42, (255, 255, 255))
    if tap is not None:
        cx, cy = int(tap[0] * PH_W), int(tap[1] * PH_H)
        cv2.circle(s, (cx, cy), tap_r, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(s, (cx, cy), max(3, tap_r // 3), (40, 40, 40), -1, cv2.LINE_AA)
    cv2.rectangle(c, (PH_X - 10, PH_Y - 12), (PH_X + PH_W + 10, PH_Y + PH_H + 12), (20, 20, 20), -1)
    c[PH_Y:PH_Y + PH_H, PH_X:PH_X + PH_W] = s


def log(c: np.ndarray, lines: list[tuple[str, tuple]]):
    cv2.rectangle(c, (350, 75), (930, 510), PANEL, -1)
    for i, (s, color) in enumerate(lines[-14:]):
        text(c, s, 366, 108 + i * 29, 0.55, color)


def add(img, ms):
    frames.append((img, ms))


def part1():
    scr = render(720, 1280)
    dev = make_device("emu-720p", 720, 1280, error_rate=1.0, load_s=0.3)
    res = run_flow(dev, demo.STEPS, tempfile.mkdtemp(), 7, "B")          # real run, for the timings
    assert res.ok, res.error
    t = dict(res.steps)

    m_btn = find_template(scr["login"], demo.SIGN_IN_TPL, 0.8)
    m_red = demo.error_banner(scr["error"])[0]
    m_retry = find_color(scr["error"], *BLUE, roi=(0.55, 0.11, 1.0, 0.21))[0]
    m_ok = demo.home_badge(scr["home"])[0]
    title = "One flow, variant B, 720p phone"
    L: list[tuple[str, tuple]] = [("run 7  device=emu-720p  variant=B", DIM)]

    def frame(screen, boxes=(), tap=None, r=0, ms=120):
        c = canvas(title)
        phone(c, screen, boxes, tap, r)
        log(c, L)
        add(c, ms)

    frame(scr["login"], ms=900)
    L.append(("find_template('Sign in')  [template cut on 1080p]", FG))
    frame(scr["login"], ms=600)
    L.append((f"  match {m_btn.score:.2f}  at {m_btn.w}x{m_btn.h}px", OK))
    frame(scr["login"], [(m_btn, f"template {m_btn.score:.2f}", BAD)], ms=900)
    L.append((f"tap ({m_btn.center[0]:.2f}, {m_btn.center[1]:.2f})  [screen fractions]", FG))
    for r in (6, 12, 18, 24):
        frame(scr["login"], [(m_btn, f"template {m_btn.score:.2f}", BAD)], m_btn.center, r, 70)
    L.append((f"tap_sign_in        {t['tap_sign_in']:.2f} s", DIM))
    L.append(("wait_for: home badge OR error banner", FG))
    frame(scr["loading"], ms=700)
    L.append(("  red banner found (HSV colour blob)", WARN))
    frame(scr["error"], [(m_red, "red banner", WARN, 1)], ms=1000)
    L.append(("  variant B -> find blue 'Retry', tap", FG))
    frame(scr["error"], [(m_red, "red banner", WARN, 1), (m_retry, "Retry", ACC, 1)], ms=700)
    for r in (6, 12, 18, 24):
        frame(scr["error"], [(m_red, "red banner", WARN, 1), (m_retry, "Retry", ACC, 1)], m_retry.center, r, 70)
    L.append((f"wait_result        {t['wait_result']:.2f} s", DIM))
    frame(scr["loading"], ms=600)
    L.append(("  green 'Online' badge found", OK))
    L.append((f"check_home         {t['check_home']:.2f} s", DIM))
    L.append((f"PASS  4 steps  {res.seconds:.2f} s", OK))
    frame(scr["home"], [(m_ok, "Online badge", OK)], ms=2200)


def part2():
    devices = [
        make_device("emu-720p-1", 720, 1280), make_device("emu-720p-2", 720, 1280),
        make_device("phone-1080p-1", 1080, 1920), make_device("phone-1080p-2", 1080, 1920),
        make_device("tablet-1440p", 1440, 2560), make_device("cloud-frozen", 1080, 1920, frozen=True),
    ]
    order = [d.serial for d in devices]
    events = []
    t0 = time.monotonic()
    rep = run_fleet(devices, demo.STEPS, 20, tempfile.mkdtemp(), variants=("A", "B"),
                    max_consecutive_fail=3, on_result=lambda r: events.append((time.monotonic() - t0, r)))
    total = events[-1][0]
    ab = compare([r.__dict__ for r in rep.results if r.serial not in rep.disabled])

    def draw(upto: float, final: bool):
        c = np.full((H, W, 3), BG, np.uint8)
        text(c, "Same flow on 6 devices in parallel (A/B on every device)", 40, 52, 0.75, FG, 2)
        text(c, f"t = {min(upto, total):4.1f} s", 820, 52, 0.6, DIM)
        cv2.rectangle(c, (30, 75), (930, 510), PANEL, -1)
        hx = (50, 250, 330, 400, 480)
        for x, h in zip(hx, ("device", "runs", "ok", "ok %", "progress")):
            text(c, h, x, 108, 0.55, DIM)
        seen = [r for (ts, r) in events if ts <= upto]
        for i, s in enumerate(order):
            rs = [r for r in seen if r.serial == s]
            y = 150 + i * 40
            ok = sum(r.ok for r in rs)
            dis = s in rep.disabled and len(rs) == len([r for r in rep.results if r.serial == s])
            text(c, s, hx[0], y, 0.58, BAD if dis else FG)
            text(c, f"{len(rs):3}", hx[1], y, 0.58)
            text(c, f"{ok:3}", hx[2], y, 0.58)
            text(c, f"{100 * ok / len(rs):5.1f}" if rs else "   -", hx[3], y, 0.58)
            x0, x1 = hx[4], hx[4] + 300
            cv2.rectangle(c, (x0, y - 14), (x1, y), (70, 64, 60), -1)
            for j, r in enumerate(rs):
                cx = x0 + j * 15
                cv2.rectangle(c, (cx + 1, y - 13), (cx + 13, y - 1), OK if r.ok else BAD, -1)
            if dis:
                text(c, "DISABLED (hung)", x1 + 14, y, 0.5, BAD)
        if final:
            a, b, tst = ab["A"], ab["B"], ab["test"]
            text(c, "A  fail on error banner", 50, 425, 0.6, FG)
            text(c, f"{100 * a['rate']:.1f}% of {a['runs']} runs", 330, 425, 0.6, FG)
            text(c, "B  tap Retry", 50, 455, 0.6, FG)
            text(c, f"{100 * b['rate']:.1f}% of {b['runs']} runs", 330, 455, 0.6, FG)
            text(c, f"z = {tst['z']}   p = {tst['p']}   ->  {tst['verdict']}", 50, 490, 0.65, OK, 2)
        return c

    steps = 26
    for k in range(1, steps + 1):
        add(draw(total * k / steps, False), 160)
    add(draw(total + 1, True), 3800)


if __name__ == "__main__":
    part1()
    part2()
    out = os.path.join(ROOT, "docs", "demo.gif")
    imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).quantize(colors=64, method=Image.Quantize.MEDIANCUT)
            for f, _ in frames]
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=[d for _, d in frames], loop=0, optimize=True)
    print(f"{out}: {len(frames)} frames, {sum(d for _, d in frames) / 1000:.1f} s, {os.path.getsize(out) / 1e6:.2f} MB")
