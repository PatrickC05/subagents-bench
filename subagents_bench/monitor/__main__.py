"""``python -m subagents_bench.monitor JOBS_DIR [--compare OTHER_JOBS_DIR] [--spec SPEC] --out OUT_DIR
[--page NAME] [--nav LABEL=HREF ...] [--no-transcripts]``: writes OUT_DIR/index.html (or OUT_DIR/NAME)
and OUT_DIR/transcripts/*.json (serve the folder, or publish it: the page fetches a transcript when its
toggle is opened). Pages of several sweeps can share OUT_DIR: write each with its own --page and the
same --nav tabs. Reads the jobs dirs only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .build import build


def nav_tab(s: str) -> tuple[str, str]:
    label, eq, href = s.partition("=")
    if not (eq and label.strip() and href.strip()):
        raise argparse.ArgumentTypeError(f"{s!r}: expected LABEL=HREF, e.g. 'Requests (archive)=requests.html'")
    return label.strip(), href.strip()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m subagents_bench.monitor", description=__doc__.split("\n\n")[0])
    p.add_argument("jobs_dir", type=Path, help="the sweep's jobs dir (its sweep.json gives the design)")
    p.add_argument("--compare", type=Path, help="an earlier sweep's jobs dir (the same tasks and first bugs)")
    p.add_argument("--spec", type=Path, help="the sweep's spec, instead of JOBS_DIR/sweep.json")
    p.add_argument("--out", type=Path, required=True, help="output dir: the page and transcripts/")
    p.add_argument("--page", default="index.html", help="the page's file name in --out (default index.html; any "
                   "other page gets its own doctype and viewport: the host wraps only index.html)")
    p.add_argument("--nav", type=nav_tab, action="append", metavar="LABEL=HREF",
                   help="a tab of the bar at the top of the page, in order (repeat it); the tab whose HREF is "
                        "--page is the current one")
    p.add_argument("--no-transcripts", action="store_true", help="skip exporting transcripts/ (no toggles)")
    a = p.parse_args(argv)
    try:
        page, tx, wrote = build(a.jobs_dir, a.out, a.compare, a.spec, not a.no_transcripts, a.page, a.nav)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not a.no_transcripts:
        print(f"transcripts: {len(tx)} files, {wrote} rewritten, {sum(e['bytes'] for e in tx.values()) / 1e6:.1f} MB")
    print(page)
    return 0


if __name__ == "__main__":
    sys.exit(main())
