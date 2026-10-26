#!/usr/bin/env bash
# Fetch the SWE-sweep task data the sweeps run on: facebookresearch/swe-sweep at the commit the
# study used, cloned to ../class-6-swe-sweep/swe-sweep (the sweeps' dataset_path). Skips the clone
# when that directory already exists. SWE_SWEEP_DIR (env) clones somewhere else; the sweeps then
# need a dataset_path pointing there.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${SWE_SWEEP_DIR:-$ROOT/../class-6-swe-sweep/swe-sweep}"
COMMIT=5968c69  # "Update paper citation", the tasks every sweep in this repo ran

if [ -d "$DEST/tasks" ]; then
  echo "[OK] $DEST/tasks exists ($(git -C "$DEST" rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout'))"
  exit 0
fi
mkdir -p "$(dirname "$DEST")"
git clone https://github.com/facebookresearch/swe-sweep.git "$DEST"
git -C "$DEST" checkout -q "$COMMIT"
echo "[OK] $DEST/tasks at $COMMIT"
