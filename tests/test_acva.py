import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

from acva import find_color, find_template, run_flow, run_fleet  # noqa: E402
from acva.ab import assign, compare  # noqa: E402
from acva.fleet import is_device_fault  # noqa: E402
from acva.vision import GREEN, RED_LOW, crop  # noqa: E402
import demo_offline as demo  # noqa: E402
from fake_app import SIGN_IN, make_device, render  # noqa: E402


@pytest.mark.parametrize("w,h", [(720, 1280), (1080, 1920), (1440, 2560)])
def test_template_found_on_every_resolution(w, h):
    m = find_template(render(w, h)["login"], demo.SIGN_IN_TPL, 0.8)
    assert m is not None
    cx, cy = m.center
    assert SIGN_IN[0] < cx < SIGN_IN[2] and SIGN_IN[1] < cy < SIGN_IN[3]


def test_template_absent_returns_none():
    assert find_template(render()["home"], demo.SIGN_IN_TPL, 0.8) is None


def test_color_blobs():
    s = render()
    assert find_color(s["home"], *GREEN, roi=(0.6, 0, 1, 0.1))
    assert not find_color(s["login"], *GREEN)
    assert find_color(s["error"], *RED_LOW, min_area=2000)


def test_roi_crop_offsets():
    img = render(1000, 2000)["login"]
    part, ox, oy = crop(img, (0.1, 0.5, 0.6, 1.0))
    assert (ox, oy) == (100, 1000) and part.shape[:2] == (1000, 500)


def test_flow_success_and_failure_screenshot(tmp_path):
    ok = run_flow(make_device("d1", 1080, 1920, error_rate=0.0), demo.STEPS, str(tmp_path), 1, "A")
    assert ok.ok and [s for s, _ in ok.steps] == ["reset", "tap_sign_in", "wait_result", "check_home"]
    bad = run_flow(make_device("d2", 1080, 1920, error_rate=1.0), demo.STEPS, str(tmp_path), 2, "A")
    assert not bad.ok and "error banner" in bad.error and os.path.exists(bad.failure_shot)
    fixed = run_flow(make_device("d3", 1080, 1920, error_rate=1.0), demo.STEPS, str(tmp_path), 3, "B")
    assert fixed.ok  # variant B taps Retry


def test_fleet_disables_hung_device_only(tmp_path):
    devs = [make_device("good", 720, 1280, error_rate=1.0), make_device("hung", 720, 1280, frozen=True)]
    rep = run_fleet(devs, demo.STEPS, 4, str(tmp_path), variants=("A",), max_consecutive_fail=2)
    assert rep.disabled == ["hung"]
    assert sum(r.serial == "good" for r in rep.results) == 4      # app errors do not disable a device
    assert all(is_device_fault(r) for r in rep.results if r.serial == "hung")
    assert len(open(tmp_path / "runs.jsonl").read().splitlines()) == len(rep.results)


def test_ab_assignment_balanced_and_deterministic():
    v = [assign(i) for i in range(1, 2001)]
    assert v == [assign(i) for i in range(1, 2001)]
    assert 0.45 < v.count("A") / len(v) < 0.55


def test_ab_compare_significance():
    rows = [{"variant": "A", "ok": i < 80} for i in range(100)] + [{"variant": "B", "ok": i < 95} for i in range(100)]
    r = compare(rows)
    assert r["test"]["verdict"] == "B better" and r["test"]["p"] < 0.05
    same = compare([{"variant": "A", "ok": i % 2 == 0} for i in range(40)] + [{"variant": "B", "ok": i % 2 == 0} for i in range(40)])
    assert same["test"]["verdict"] == "no significant difference"
