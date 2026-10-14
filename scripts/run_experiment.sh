#!/usr/bin/env bash
# Run a sweep end to end: preflight (with harbor's oracle agent on every task), then every
# (round, condition) harbor job. Resumable: re-run the same command after any stop.
#
#   scripts/run_experiment.sh [SPEC] [sweep options]
#
#   SPEC             sweep spec, default configs/sweeps/pilot.yaml (a relative path that does
#                    not exist in the current directory is taken from the repository root)
#   sweep options    passed to `subagents-bench sweep`: --dry-run (print the jobs, skip
#                    preflight), --only COND..., --rounds 1,2
#
# SKIP_PREFLIGHT=1 skips preflight. On macOS the sweep runs under `caffeinate -i` so the
# machine does not sleep (and drop the network); SB_NO_CAFFEINATE=1 turns that off.
# Exit codes: 0 done, 1 failed jobs or preflight, 2 spec or config error, 3 billing error,
# 4 provider outage (the ABORT line names the model and error), 130 interrupted.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1  # progress lines appear at once in a log file too
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e '$d' -e 's/^# \{0,1\}//'; }

SPEC="configs/sweeps/pilot.yaml"
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
  SPEC="$1"
  shift
fi
case "$SPEC" in
  /*) ;;
  *) [ -e "$SPEC" ] || SPEC="$ROOT/$SPEC" ;;
esac

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    -h | --help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1 ;;
  esac
done

BIN="$ROOT/.venv/bin/subagents-bench"
if [ ! -x "$BIN" ]; then
  echo "[FAIL] $BIN not found: run scripts/setup.sh first" >&2
  exit 1
fi

if [ "$DRY_RUN" = 0 ] && [ "${SKIP_PREFLIGHT:-0}" != 1 ]; then
  "$BIN" preflight "$SPEC" --oracle || exit $?
fi

AWAKE=()
if [ "$DRY_RUN" = 0 ] && [ "${SB_NO_CAFFEINATE:-0}" != 1 ] && command -v caffeinate >/dev/null 2>&1; then
  AWAKE=(caffeinate -i)
fi
exec ${AWAKE[@]+"${AWAKE[@]}"} "$BIN" sweep "$SPEC" "$@"
