#!/usr/bin/env bash
# Analyze a sweep's jobs: validate the records, report field completeness,
# aggregate into tables and draw the figures.
#
#   scripts/analyze.sh [JOBS_DIR] [OUT]
#
#   JOBS_DIR   the sweep's jobs dir, default jobs/pilot
#   OUT        output name, default basename of JOBS_DIR; tables go to
#              analysis/<OUT>/ and figures (+ index.md) to figures/<OUT>/
#
# Steps (in order): check-run, completeness (--strict: FAIL if a field that
# should be populated never is; full report in analysis/<OUT>/completeness.csv),
# aggregate, plot. Each step's
# output is also saved under analysis/<OUT>/_logs/. A failing check-run or
# completeness is reported but does not stop aggregation; plot needs a
# successful aggregate. Ends with a PASS/FAIL checklist; exits nonzero on any
# FAIL. Safe to re-run (outputs are overwritten); works on a partial sweep.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

JOBS_DIR="${1:-jobs/pilot}"
JOBS_DIR="${JOBS_DIR%/}"
OUT="${2:-$(basename "$JOBS_DIR")}"
BIN="$ROOT/.venv/bin/subagents-bench"
ANALYSIS="analysis/$OUT"
FIGURES="figures/$OUT"
LOGS="$ANALYSIS/_logs"

if [ ! -x "$BIN" ]; then
  echo "[FAIL] $BIN not found: run scripts/setup.sh first" >&2
  exit 1
fi
if [ ! -d "$JOBS_DIR" ]; then
  echo "[FAIL] jobs dir not found: $JOBS_DIR (run scripts/run_experiment.sh first)" >&2
  exit 1
fi
mkdir -p "$LOGS"

RESULTS=()
record() { RESULTS+=("$1|$2|$3"); }
step() { # NAME CMD...
  local name="$1"
  shift
  printf '\n== %s: %s\n' "$name" "$*"
  if "$@" 2>&1 | tee "$LOGS/$name.txt"; then
    record PASS "$name" "log: $LOGS/$name.txt"
    return 0
  fi
  record FAIL "$name" "see $LOGS/$name.txt"
  return 1
}

step check-run "$BIN" check-run "$JOBS_DIR" || true
step completeness "$BIN" completeness "$JOBS_DIR" --strict --out "$ANALYSIS/completeness.csv" || true
if step aggregate "$BIN" aggregate "$JOBS_DIR" --out "$ANALYSIS"; then
  step plot "$BIN" plot "$ANALYSIS" --out "$FIGURES" || true
else
  record SKIP plot "needs a successful aggregate"
fi

n_fail=0
printf '\n================ analysis checklist (%s) ================\n' "$JOBS_DIR"
for r in "${RESULTS[@]}"; do
  IFS='|' read -r status name detail <<<"$r"
  printf '  [%-4s] %-13s %s\n' "$status" "$name" "$detail"
  if [ "$status" = "FAIL" ]; then n_fail=$((n_fail + 1)); fi
done
printf '\nTables:  %s/\nFigures: %s/ (captions in %s/index.md)\n' "$ANALYSIS" "$FIGURES" "$FIGURES"
if [ "$n_fail" -gt 0 ]; then exit 1; fi
