#!/usr/bin/env bash
# Run a sweep end to end: preflight, then every (round x condition) harbor job.
# Resumable: re-run the same command after an interruption and it continues
# where it stopped (complete jobs are skipped; partial jobs, and finished jobs
# with trials that did not measure their condition, are moved aside and rerun).
#
#   scripts/run_experiment.sh [SPEC] [sweep options...]
#
#   SPEC             sweep spec, default configs/sweeps/pilot.yaml (relative
#                    paths: to the current directory, else to the repo root)
#   sweep options    passed to `subagents-bench sweep`, e.g.
#                    --only single_5step gemma31b_parallel_hard   --rounds 1
#                    --env-file FILE   (also used by preflight)
#                    --allow-task-change   --allow-code-change   --dry-run
#   -h, --help       print this help
#
# Preflight runs harbor's oracle agent on every task (results kept in
# <jobs_dir>/_oracle and reused while they pass; a failed oracle job is rerun)
# to prove the tasks are solvable on this machine. SKIP_PREFLIGHT=1 skips
# preflight entirely (e.g. when resuming).
#
# --dry-run only prints the plan, every `harbor run` command, the config
# validation, the resume checks and a ROUGH cost estimate; it skips
# preflight, launches no container and makes no API request.
#
# Exit codes: 0 done, 1 failed jobs/preflight (or jobs that still need a
# retry), 2 bad spec/config, a refused resume (changed task list, harbor
# arguments or code) or another sweep running on the same jobs dir,
# 3 aborted on a provider billing error, 4 aborted on a provider outage (a
# trial hit a provider failure and the models still do not answer; usually
# within minutes of the outage), 130 interrupted.
#
# Each trial's outcome is printed as harbor finishes it (reward, exit reason,
# wall clock, provider errors), so failures show while the job runs. On macOS
# the sweep runs under `caffeinate -i` so the machine does not idle-sleep
# (and lose the network) mid-sweep; SB_NO_CAFFEINATE=1 turns that off.
set -euo pipefail

CALLER_PWD="$(pwd)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Progress lines appear at once also when the output goes to a log file.
export PYTHONUNBUFFERED=1

usage() { awk 'NR > 1 && /^set -euo/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"; }

# abspath PATH: absolute PATH; a relative one resolves against the caller's
# directory if it exists there, else against the repo root.
abspath() {
  local p="$1"
  case "$p" in
    /*) ;;
    *) if [ -e "$CALLER_PWD/$p" ]; then p="$CALLER_PWD/$p"; else p="$ROOT/$p"; fi ;;
  esac
  if [ -d "$(dirname "$p")" ]; then p="$(cd "$(dirname "$p")" && pwd)/$(basename "$p")"; fi
  printf '%s\n' "$p"
}

SPEC_ARG=""
if [ $# -gt 0 ]; then
  case "$1" in
    -h | --help) usage; exit 0 ;;
    -*) ;; # options only: default SPEC
    *) SPEC_ARG="$1"; shift ;;
  esac
fi
SPEC_ARG="${SPEC_ARG:-configs/sweeps/pilot.yaml}"
SPEC="$(abspath "$SPEC_ARG")"

SWEEP_ARGS=()
PREFLIGHT_ARGS=()
DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    -h | --help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1; SWEEP_ARGS+=("$1") ;;
    --env-file | --env-file=*)
      if [ "$1" = "--env-file" ]; then
        if [ $# -lt 2 ]; then echo "[FAIL] --env-file needs a FILE" >&2; exit 2; fi
        env_file="$2"
        shift
      else
        env_file="${1#--env-file=}"
      fi
      env_file="$(abspath "$env_file")"
      SWEEP_ARGS+=(--env-file "$env_file")
      PREFLIGHT_ARGS+=(--env-file "$env_file")
      ;;
    *) SWEEP_ARGS+=("$1") ;;
  esac
  shift
done

cd "$ROOT"
BIN="$ROOT/.venv/bin/subagents-bench"
SHOW_SPEC="${SPEC#"$ROOT"/}"

if [ ! -x "$BIN" ]; then
  echo "[FAIL] $BIN not found: run scripts/setup.sh first" >&2
  exit 1
fi
if [ ! -f "$SPEC" ]; then
  echo "[FAIL] sweep spec not found: $SPEC_ARG (looked in $CALLER_PWD and $ROOT)" >&2
  exit 2
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "== dry run: skipping preflight (no containers, no API requests)"
elif [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  printf '\n== preflight: %s\n' "$SHOW_SPEC"
  set +e
  "$BIN" preflight "$SPEC" --oracle ${PREFLIGHT_ARGS[@]+"${PREFLIGHT_ARGS[@]}"}
  prc=$?
  set -e
  if [ "$prc" = "2" ]; then
    # Nothing was checked: a spec error, or another sweep holds the jobs dir.
    echo "[FAIL] preflight refused to run: see the reason above (spec problem, or another sweep on this jobs dir)" >&2
    exit 2
  elif [ "$prc" != "0" ]; then
    echo "[FAIL] preflight failed: fix the [FAIL] items above (details in <jobs_dir>/preflight.json)" >&2
    exit 1
  fi
fi

# Keep macOS from idle-sleeping during the sweep: sleep drops the network,
# and every model call and harbor job needs it (SB_NO_CAFFEINATE=1: don't).
AWAKE=()
if [ "$DRY_RUN" != "1" ] && [ "${SB_NO_CAFFEINATE:-0}" != "1" ] && command -v caffeinate >/dev/null 2>&1; then
  AWAKE=(caffeinate -i)
fi

printf '\n== sweep: %s %s\n' "$SHOW_SPEC" "${SWEEP_ARGS[*]-}"
set +e
${AWAKE[@]+"${AWAKE[@]}"} "$BIN" sweep "$SPEC" ${SWEEP_ARGS[@]+"${SWEEP_ARGS[@]}"}
rc=$?
set -e
case "$rc" in
  0) if [ "$DRY_RUN" = "1" ]; then echo "[PASS] dry run: plan printed, nothing was run"; else echo "[PASS] sweep complete"; fi ;;
  2) echo "[FAIL] sweep refused to run: see the reasons above (spec or config problem, a resume that would mix task lists, harbor arguments or code, or another sweep on this jobs dir)" ;;
  3) echo "[FAIL] sweep aborted on a billing error: top up the provider account, then re-run this command to resume (the affected job is rerun)" ;;
  4) echo "[FAIL] sweep aborted: a provider is failing (outage or rejected key; the ABORT line above names the model and error); re-run this command later to resume (the job is rerun)" ;;
  130) echo "[WARN] sweep interrupted: re-run this command to resume" ;;
  *) echo "[FAIL] sweep exited with $rc: see the output above and <jobs_dir>/sweep_manifest.jsonl; re-run to retry" ;;
esac
exit "$rc"
