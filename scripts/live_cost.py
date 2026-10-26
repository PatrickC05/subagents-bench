"""Live spend of the final study's runs against issue #23's expected cost, from the calls the
running sessions have logged so far (``<trial>/agent/calls.jsonl``, priced with configs/pricing.yaml).

    .venv/bin/python scripts/live_cost.py [--jobs jobs] [--factor 1.5]

One line per model whose jobs dirs exist (jobs/final_levels-<model>, jobs/final_isolated-<model>):
dollars so far, the expected total, the share spent, and the share of its sessions finished. A
model is flagged OVER when it has spent more than ``--factor`` times its expected total, or more
than ``--factor`` times its expected share at the sessions finished so far (a projection: spend /
finished share, once a fifth of its sessions are done). Exit 1 if any model is flagged, else 0.
Prices are list prices; the bill can differ (Prime's own record: `prime wallet`).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from subagents_bench.analysis.pricing import call_cost, load_prices  # noqa: E402

# issue #23: expected USD and sessions per model (levels sweep, isolated sweep)
EXPECTED = {
    "claude-opus-5.5": (613, 54, 36), "gpt-5.6-luna": (34, 54, 36), "deepseek-v4.1-flash-1017": (34, 54, 36),
    "gpt-5.6-sol": (360, 18, 36), "gpt-5.6-terra": (180, 18, 36), "claude-sonnet-5.5": (130, 18, 36),
}


def spend(dirs: list[Path], prices: dict) -> tuple[float, int, int]:
    """(USD so far, calls without a price, trials finished) over these jobs dirs."""
    usd, unpriced, done = 0.0, 0, 0
    for d in dirs:
        for calls in d.glob("*/*/agent/calls.jsonl"):
            for line in calls.read_text().splitlines():
                try:
                    c = call_cost(json.loads(line), prices)
                except (ValueError, KeyError, TypeError):
                    continue  # a line being written
                if math.isnan(c):
                    unpriced += 1
                else:
                    usd += c
            done += (calls.parent.parent / "result.json").is_file()
    return usd, unpriced, done


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--jobs", type=Path, default=ROOT / "jobs")
    p.add_argument("--factor", type=float, default=1.5)
    a = p.parse_args(argv)
    prices, flagged, total = load_prices(), False, 0.0
    for model, (expected, n_levels, n_isolated) in EXPECTED.items():
        dirs = [a.jobs / f"final_levels-{model}", a.jobs / f"final_isolated-{model}"]
        if not any(d.is_dir() for d in dirs):
            continue
        usd, unpriced, done = spend([d for d in dirs if d.is_dir()], prices)
        n = n_levels + n_isolated
        finished = min(done / n, 1.0)
        projected = usd / finished if finished >= 0.2 else math.nan
        over = usd > a.factor * expected or (not math.isnan(projected) and projected > a.factor * expected)
        flagged |= over
        total += usd
        print(f"{'OVER' if over else 'ok  '} {model:26s} ${usd:8.2f} of ${expected:5d} expected ({usd / expected:5.0%}); "
              f"{done}/{n} sessions done" + (f"; projected ${projected:,.0f}" if not math.isnan(projected) else "")
              + (f"; {unpriced} calls without a price" if unpriced else ""))
    print(f"total so far: ${total:,.2f}")
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
