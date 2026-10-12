#!/usr/bin/env bash
# Aggregate a sweep's jobs into tables, draw the figures, then apply the delegation gate.
#
#   scripts/analyze.sh [JOBS_DIR...]      (run from the repo root)
#
#   JOBS_DIR   sweep jobs dir(s), analyzed as one data set (default: jobs/pilot)
#   OUT        (env) output name, default the last JOBS_DIR's name: tables and
#              summary.json go to analysis/OUT/, figures and index.md to figures/OUT/
#   PYTHON     (env) interpreter, default the repo's .venv/bin/python
#
# Exit 3 if a subagent condition never delegated in any of its valid episodes
# (tables and figures are still written), 2 on a usage error or no episodes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
usage() { awk 'NR > 1 && /^set -euo/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"; }

DIRS=()
for arg in "$@"; do
  case "$arg" in
    -h | --help) usage; exit 0 ;;
    -*) echo "[FAIL] unknown option: $arg (see --help)" >&2; exit 2 ;;
    *) DIRS+=("$arg") ;;
  esac
done
if [ ${#DIRS[@]} -eq 0 ]; then DIRS=(jobs/pilot); fi
for d in "${DIRS[@]}"; do
  if [ ! -d "$d" ]; then echo "[FAIL] jobs dir not found: $d" >&2; exit 2; fi
done
OUT="${OUT:-$(basename "${DIRS[${#DIRS[@]}-1]}")}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
if ! command -v "$PY" >/dev/null; then echo "[FAIL] $PY not found: run scripts/setup.sh or set PYTHON" >&2; exit 2; fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"  # this checkout's package

echo "== aggregate: ${DIRS[*]} -> analysis/$OUT/"
rc=0
"$PY" -m subagents_bench.analysis "${DIRS[@]}" --out "analysis/$OUT" || rc=$?
if [ "$rc" -ne 0 ] && [ "$rc" -ne 3 ]; then exit "$rc"; fi
echo "== plots: figures/$OUT/"
"$PY" -m subagents_bench.plots "analysis/$OUT" --out "figures/$OUT"
if [ "$rc" -eq 3 ]; then
  echo "[FAIL] delegation gate failed (see above); tables and figures were written anyway"
fi
exit "$rc"
