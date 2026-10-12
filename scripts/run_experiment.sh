#!/usr/bin/env bash
# Run a sweep end to end: preflight, then every (round x condition) harbor job.
# Resumable: re-run the same command after an interruption and it continues
# where it stopped (complete jobs are skipped, partial ones moved aside and
# rerun).
#
#   scripts/run_experiment.sh [SPEC] [sweep options...]
#
#   SPEC             sweep spec, default configs/sweeps/pilot.yaml
#   sweep options    passed to `subagents-bench sweep`, e.g.
#                    --only single_5step gemma31b_parallel_hard   --rounds 1
#
# Preflight runs harbor's oracle agent on every task once (results kept in
# <jobs_dir>/_oracle and reused afterwards) to prove the tasks are solvable on
# this machine. SKIP_PREFLIGHT=1 skips preflight entirely (e.g. when resuming).
#
# --dry-run (passed through to `sweep`) only prints the plan, every
# `harbor run` command, the config validation and a ROUGH cost estimate; it
# skips preflight, launches no container and makes no API request.
#
# Exit codes: 0 done, 1 failed jobs/preflight, 2 bad spec/config,
# 3 aborted on a provider billing error, 130 interrupted.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SPEC="${1:-configs/sweeps/pilot.yaml}"
if [ $# -gt 0 ]; then shift; fi
BIN="$ROOT/.venv/bin/subagents-bench"

if [ ! -x "$BIN" ]; then
  echo "[FAIL] $BIN not found: run scripts/setup.sh first" >&2
  exit 1
fi
if [ ! -f "$SPEC" ]; then
  echo "[FAIL] sweep spec not found: $SPEC" >&2
  exit 2
fi

DRY_RUN=0
for arg in "$@"; do
  if [ "$arg" = "--dry-run" ]; then DRY_RUN=1; fi
done

if [ "$DRY_RUN" = "1" ]; then
  echo "== dry run: skipping preflight (no containers, no API requests)"
elif [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  printf '\n== preflight: %s\n' "$SPEC"
  if ! "$BIN" preflight "$SPEC" --oracle; then
    echo "[FAIL] preflight failed: fix the [FAIL] items above (details in <jobs_dir>/preflight.json)" >&2
    exit 1
  fi
fi

printf '\n== sweep: %s %s\n' "$SPEC" "$*"
set +e
"$BIN" sweep "$SPEC" "$@"
rc=$?
set -e
case "$rc" in
  0) if [ "$DRY_RUN" = "1" ]; then echo "[PASS] dry run: plan printed, nothing was run"; else echo "[PASS] sweep complete"; fi ;;
  3) echo "[FAIL] sweep aborted on a billing error: top up the provider account, then re-run this command to resume" ;;
  130) echo "[WARN] sweep interrupted: re-run this command to resume" ;;
  *) echo "[FAIL] sweep exited with $rc: see the output above and <jobs_dir>/sweep_manifest.jsonl; re-run to retry" ;;
esac
exit "$rc"
