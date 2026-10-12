#!/usr/bin/env bash
# Analyze a sweep's jobs: validate the records, report field completeness,
# aggregate into tables and draw the figures.
#
#   scripts/analyze.sh [JOBS_DIR...]
#   OUT=NAME scripts/analyze.sh [JOBS_DIR...]
#
#   JOBS_DIR     one or more sweep jobs dirs, default jobs/pilot (relative
#                paths: to the current directory, else to the repo root).
#                Several dirs are analyzed as one data set, e.g. the size
#                sweep next to the pilot:
#                  OUT=size scripts/analyze.sh jobs/pilot jobs/prime_size
#   OUT          (environment) output name, default the basename of the last
#                JOBS_DIR; tables go to analysis/<OUT>/ and figures
#                (+ index.md) to figures/<OUT>/
#   -h, --help   print this help
#
# Steps (in order): check-run, completeness (--strict: FAIL if a field that
# should be populated never is, or if a subagent condition never delegated or
# an arbiter condition never arbitrated, i.e. the treatment was not
# exercised; full report in analysis/<OUT>/completeness.csv),
# aggregate, plot. Each step's output is also saved under
# analysis/<OUT>/_logs/. A failing check-run or completeness is reported but
# does not stop aggregation; plot needs a successful aggregate. Ends with a
# PASS/FAIL checklist; exits nonzero on any FAIL. Safe to re-run (outputs are
# overwritten); works on a partial sweep.
set -euo pipefail

CALLER_PWD="$(pwd)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() { awk 'NR > 1 && /^set -euo/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"; }

# abspath PATH: absolute PATH; a relative one resolves against the caller's
# directory if it exists there, else against the repo root.
abspath() {
  local p="$1"
  case "$p" in
    /*) ;;
    *) if [ -e "$CALLER_PWD/$p" ]; then p="$CALLER_PWD/$p"; else p="$ROOT/$p"; fi ;;
  esac
  if [ -d "$p" ]; then p="$(cd "$p" && pwd)"; fi
  printf '%s\n' "$p"
}

JOBS_DIRS=()
for arg in "$@"; do
  case "$arg" in
    -h | --help) usage; exit 0 ;;
    -*) echo "[FAIL] unknown option: $arg (see scripts/analyze.sh --help)" >&2; exit 2 ;;
    *) JOBS_DIRS+=("$(abspath "$arg")") ;;
  esac
done
if [ ${#JOBS_DIRS[@]} -eq 0 ]; then JOBS_DIRS=("$ROOT/jobs/pilot"); fi
for d in "${JOBS_DIRS[@]}"; do
  if [ ! -d "$d" ]; then
    echo "[FAIL] jobs dir not found: ${d#"$ROOT"/} (run scripts/run_experiment.sh first; to name the output, set OUT=NAME)" >&2
    exit 1
  fi
done

cd "$ROOT"
OUT="${OUT:-$(basename "${JOBS_DIRS[${#JOBS_DIRS[@]}-1]}")}"
BIN="$ROOT/.venv/bin/subagents-bench"
ANALYSIS="analysis/$OUT"
FIGURES="figures/$OUT"
LOGS="$ANALYSIS/_logs"
SHOW_DIRS=""
for d in "${JOBS_DIRS[@]}"; do SHOW_DIRS="$SHOW_DIRS ${d#"$ROOT"/}"; done
SHOW_DIRS="${SHOW_DIRS# }"

if [ ! -x "$BIN" ]; then
  echo "[FAIL] $BIN not found: run scripts/setup.sh first" >&2
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

step check-run "$BIN" check-run "${JOBS_DIRS[@]}" || true
step completeness "$BIN" completeness "${JOBS_DIRS[@]}" --strict --out "$ANALYSIS/completeness.csv" || true
if step aggregate "$BIN" aggregate "${JOBS_DIRS[@]}" --out "$ANALYSIS"; then
  step plot "$BIN" plot "$ANALYSIS" --out "$FIGURES" || true
else
  record SKIP plot "needs a successful aggregate"
fi

n_fail=0
printf '\n================ analysis checklist (%s) ================\n' "$SHOW_DIRS"
for r in "${RESULTS[@]}"; do
  IFS='|' read -r status name detail <<<"$r"
  printf '  [%-4s] %-13s %s\n' "$status" "$name" "$detail"
  if [ "$status" = "FAIL" ]; then n_fail=$((n_fail + 1)); fi
done
printf '\nTables:  %s/\nFigures: %s/ (captions in %s/index.md)\n' "$ANALYSIS" "$FIGURES" "$FIGURES"
if [ "$n_fail" -gt 0 ]; then exit 1; fi
