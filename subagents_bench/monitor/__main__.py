"""``python -m subagents_bench.monitor JOBS_DIR [--compare OTHER_JOBS_DIR] [--spec SPEC] --out OUT_DIR
[--no-transcripts]``: writes OUT_DIR/index.html and OUT_DIR/transcripts/*.json (serve the folder, or
publish it: the page fetches a transcript when its toggle is opened). Reads the jobs dirs only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .build import build


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m subagents_bench.monitor", description=__doc__.split("\n\n")[0])
    p.add_argument("jobs_dir", type=Path, help="the sweep's jobs dir (its sweep.json gives the design)")
    p.add_argument("--compare", type=Path, help="an earlier sweep's jobs dir (the same tasks and first bugs)")
    p.add_argument("--spec", type=Path, help="the sweep's spec, instead of JOBS_DIR/sweep.json")
    p.add_argument("--out", type=Path, required=True, help="output dir: index.html and transcripts/")
    p.add_argument("--no-transcripts", action="store_true", help="skip exporting transcripts/ (no toggles)")
    a = p.parse_args(argv)
    try:
        page, tx, wrote = build(a.jobs_dir, a.out, a.compare, a.spec, not a.no_transcripts)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not a.no_transcripts:
        print(f"transcripts: {len(tx)} files, {wrote} rewritten, {sum(e['bytes'] for e in tx.values()) / 1e6:.1f} MB")
    print(page)
    return 0


if __name__ == "__main__":
    sys.exit(main())
