"""Screen understanding without access to the app's code or view tree.

Two tools cover most real cases:
  find_template - "where is this button/icon?" (multi-scale, works across resolutions)
  find_color    - "is there a green status badge / red error banner, and where?" (HSV ranges, blobs)

Both accept an optional region of interest in fractions (x1, y1, x2, y2): searching a small
area is faster and gives fewer false positives.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Roi = tuple[float, float, float, float]


@dataclass(frozen=True)
class Match:
    x: int          # top-left, pixels in the full screenshot
    y: int
    w: int
    h: int
    score: float    # template: correlation 0..1; color: share of the blob's bounding box that matches
    screen_w: int
    screen_h: int

    @property
    def center(self) -> tuple[float, float]:
        """Center as fractions of the screen - pass straight to device.tap(*m.center)."""
        return (self.x + self.w / 2) / self.screen_w, (self.y + self.h / 2) / self.screen_h


def crop(img: np.ndarray, roi: Roi | None) -> tuple[np.ndarray, int, int]:
    if roi is None:
        return img, 0, 0
    h, w = img.shape[:2]
    x1, y1, x2, y2 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
    return img[y1:y2, x1:x2], x1, y1


def find_template(screen: np.ndarray, template: np.ndarray, threshold: float = 0.85,
                  roi: Roi | None = None, scales: tuple[float, ...] = (1.0, 0.9, 1.1, 0.75, 1.25, 0.67, 1.33, 0.5, 1.5, 2.0)) -> Match | None:
    """Best match of `template` on `screen`, or None if below threshold.

    Templates are cut once from one device; `scales` lets the same template work on
    phones with other DPIs. Matching is done in grayscale (robust to small color shifts)."""
    area, ox, oy = crop(screen, roi)
    g = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY) if area.ndim == 3 else area
    t0 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim == 3 else template
    best: Match | None = None
    for s in scales:
        t = t0 if s == 1.0 else cv2.resize(t0, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        if t.shape[0] > g.shape[0] or t.shape[1] > g.shape[1] or min(t.shape[:2]) < 4:
            continue
        res = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if best is None or score > best.score:
            best = Match(loc[0] + ox, loc[1] + oy, t.shape[1], t.shape[0], float(score), screen.shape[1], screen.shape[0])
        if score >= 0.98:
            break
    return best if best is not None and best.score >= threshold else None


def find_color(screen: np.ndarray, hsv_low: tuple[int, int, int], hsv_high: tuple[int, int, int],
               min_area: int = 150, roi: Roi | None = None) -> list[Match]:
    """All blobs whose color is inside the HSV range, biggest first.

    OpenCV hue is 0..179. Tip: red wraps around 0, so call twice (0-8 and 170-179)."""
    area, ox, oy = crop(screen, roi)
    hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_low, np.uint8), np.array(hsv_high, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_area:
            out.append(Match(int(x + ox), int(y + oy), int(w), int(h), float(a) / float(w * h), screen.shape[1], screen.shape[0]))
    return sorted(out, key=lambda m: m.w * m.h, reverse=True)


def color_share(screen: np.ndarray, hsv_low: tuple[int, int, int], hsv_high: tuple[int, int, int], roi: Roi | None = None) -> float:
    """Share of pixels (0..1) in the range - cheap "is the screen mostly X?" check (loading, error overlay)."""
    area, _, _ = crop(screen, roi)
    mask = cv2.inRange(cv2.cvtColor(area, cv2.COLOR_BGR2HSV), np.array(hsv_low, np.uint8), np.array(hsv_high, np.uint8))
    return float(np.count_nonzero(mask)) / mask.size


# Ready-made ranges (OpenCV HSV). Tune per app: sample a pixel with `python -m acva.hsv shot.png X Y`, widen hue +-10.
GREEN = ((45, 120, 100), (80, 255, 255))
BLUE = ((95, 120, 100), (125, 255, 255))
RED_LOW = ((0, 150, 120), (8, 255, 255))
RED_HIGH = ((170, 150, 120), (179, 255, 255))
