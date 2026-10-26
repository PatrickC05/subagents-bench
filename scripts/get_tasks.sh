#!/usr/bin/env bash
# Fetch the SWE-sweep task data the sweeps run on: facebookresearch/swe-sweep at the commit the
# study used, cloned to ../class-6-swe-sweep/swe-sweep (the sweeps' dataset_path). An existing
# checkout is checked out at that commit (fails if it cannot be, e.g. with local changes). SWE_SWEEP_DIR (env) clones somewhere else; the sweeps then
# need a dataset_path pointing there.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${SWE_SWEEP_DIR:-$ROOT/../class-6-swe-sweep/swe-sweep}"
COMMIT=5968c69  # "Update paper citation", the tasks every sweep in this repo ran

if [ ! -d "$DEST/.git" ]; then
  if [ -e "$DEST" ]; then echo "[FAIL] $DEST exists but is not a git checkout of swe-sweep" >&2; exit 1; fi
  mkdir -p "$(dirname "$DEST")"
  git clone https://github.com/facebookresearch/swe-sweep.git "$DEST"
fi
if [ "$(git -C "$DEST" rev-parse HEAD)" != "$(git -C "$DEST" rev-parse "$COMMIT^{commit}" 2>/dev/null)" ]; then
  # a checkout at another commit (upstream HEAD, or a clone interrupted before this step)
  git -C "$DEST" fetch -q origin 2>/dev/null || true
  if ! git -C "$DEST" checkout -q "$COMMIT"; then
    echo "[FAIL] $DEST is not at $COMMIT and cannot be checked out there (local changes?)" >&2
    exit 1
  fi
fi
echo "[OK] $DEST/tasks at $COMMIT"
