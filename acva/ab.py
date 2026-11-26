"""Honest A/B testing for automation flows.

assign()  - deterministic branch per run id (no bias from "device 3 is slow" or "nights are different":
            branches alternate on every device all day long).
compare() - success rate per branch + two-proportion z-test, so "B is better" is only said when the
            difference is bigger than noise.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict


def assign(run_id: int, variants: tuple[str, ...] = ("A", "B")) -> str:
    # hash-mixing instead of run_id % n, so the branch does not line up with device order
    x = (run_id * 2654435761) & 0xFFFFFFFF
    return variants[(x >> 16) % len(variants)]


def compare(results: list[dict], metric: str = "ok") -> dict:
    """results: dicts with keys 'variant' and `metric` (bool). Returns per-variant stats and z / p for A vs B."""
    n: dict[str, int] = defaultdict(int)
    k: dict[str, int] = defaultdict(int)
    for r in results:
        n[r["variant"]] += 1
        k[r["variant"]] += bool(r[metric])
    out: dict = {v: {"runs": n[v], "success": k[v], "rate": round(k[v] / n[v], 3)} for v in sorted(n)}
    if len(n) == 2:
        a, b = sorted(n)
        p = (k[a] + k[b]) / (n[a] + n[b])
        se = math.sqrt(p * (1 - p) * (1 / n[a] + 1 / n[b])) if 0 < p < 1 else 0.0
        z = (k[b] / n[b] - k[a] / n[a]) / se if se else 0.0
        pval = math.erfc(abs(z) / math.sqrt(2))      # two-sided
        out["test"] = {"z": round(z, 2), "p": float(f"{pval:.2g}"),
                       "verdict": (f"{b} better" if z > 0 else f"{a} better") if pval < 0.05 else "no significant difference"}
    return out


def compare_file(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return compare([json.loads(line) for line in f if line.strip()])


if __name__ == "__main__":
    import sys
    print(json.dumps(compare_file(sys.argv[1] if len(sys.argv) > 1 else "logs/runs.jsonl"), indent=2))
