#!/usr/bin/env bash
# Run the final study (issue #23) for one model, end to end: the sessions, the grading and the
# tables. Resumable: re-run the same command after any stop; finished jobs and grades are kept.
#
#   scripts/run_final.sh [--l0-only] [--levels SPEC] [--isolated SPEC] [--jobs-at-once N] [--dry-run]
#
#   --l0-only        experiment 1 only: the three settings at L0 (18 sessions) and the isolated
#                    baseline (36). Without it, experiment 2 as well: the three settings at every
#                    level, L0-L3 (72 sessions, L0 included), and the isolated baseline.
#   --levels SPEC    the levels sweep, default configs/sweeps/final_levels.yaml (DeepSeek-V4.1-Flash-1017
#                    on Sail); another model's spec extends it with its own `name` and `base`
#   --isolated SPEC  the isolated baseline, default configs/sweeps/final_isolated.yaml
#   --jobs-at-once N harbor jobs at once per sweep, default 12 (a job runs its 6 tasks at once)
#   --dry-run        print the jobs of both sweeps; run and grade nothing
#
# Steps: copy the fixed question sets (data/questions) into the levels sweep's hints dir, so every
# model is asked the same questions; run the levels sweep (preflight with harbor's oracle first) and
# the isolated sweep; replay each trial's snapshots to time every fix (`fix-times`, no LLM call);
# judge the answers to questions (`grade-answers`, the levels sweep's model); write the tables and
# figures with scripts/analyze.sh to analysis/<sweep>/ and figures/<sweep>/; then experiment 1's
# cost-vs-fixed table and plot (subagents_bench.analysis.final) to analysis/<levels sweep>_l0/,
# with the isolated bugs also graded at the time each had in the shared sessions.
# Needs scripts/setup.sh, scripts/get_tasks.sh, SAIL_API_KEY in .env and a Modal token.
# Exit codes: those of scripts/run_experiment.sh (0 done, 1 failed jobs, 2 config error, 3 billing,
# 4 provider outage, 130 interrupted), or of the grading step that failed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e '$d' -e 's/^# \{0,1\}//'; }

LEVELS=configs/sweeps/final_levels.yaml
ISOLATED=configs/sweeps/final_isolated.yaml
L0_ONLY=0 DRY_RUN=0 AT_ONCE=12
while [ $# -gt 0 ]; do
  case "$1" in
    -h | --help) usage; exit 0 ;;
    --l0-only) L0_ONLY=1 ;;
    --levels) LEVELS="$2"; shift ;;
    --isolated) ISOLATED="$2"; shift ;;
    --jobs-at-once) AT_ONCE="$2"; shift ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "[FAIL] unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

BIN="$ROOT/.venv/bin/subagents-bench"
[ -x "$BIN" ] || { echo "[FAIL] $BIN not found: run scripts/setup.sh first" >&2; exit 1; }
jobs_dir() { "$ROOT/.venv/bin/python" -c 'import sys; from subagents_bench.sweep import load_spec; print(load_spec(sys.argv[1]).jobs_dir)' "$1"; }
LEVELS_JOBS="$(jobs_dir "$LEVELS")"
ISOLATED_JOBS="$(jobs_dir "$ISOLATED")"

ONLY=()
if [ "$L0_ONLY" = 1 ]; then
  ONLY=(--only background_hinted_baseline manager_background_baseline manager_owner_baseline)
fi
DRY=()
[ "$DRY_RUN" = 1 ] && DRY=(--dry-run)

if [ "$L0_ONLY" = 0 ]; then
  mkdir -p "$LEVELS_JOBS/hints"
  for f in data/questions/*.questions.json; do
    [ -e "$LEVELS_JOBS/hints/$(basename "$f")" ] || cp "$f" "$LEVELS_JOBS/hints/"
  done
  [ "$DRY_RUN" = 1 ] || "$BIN" questions "$LEVELS"  # a no-op when every task has its questions
fi

echo "== levels sweep: $LEVELS -> $LEVELS_JOBS"
scripts/run_experiment.sh "$LEVELS" ${ONLY[@]+"${ONLY[@]}"} --jobs-at-once "$AT_ONCE" ${DRY[@]+"${DRY[@]}"}
echo "== isolated baseline: $ISOLATED -> $ISOLATED_JOBS"
SKIP_PREFLIGHT=1 scripts/run_experiment.sh "$ISOLATED" --jobs-at-once "$AT_ONCE" ${DRY[@]+"${DRY[@]}"}
[ "$DRY_RUN" = 1 ] && exit 0

echo "== grading"
"$BIN" fix-times "$LEVELS"
"$BIN" fix-times "$ISOLATED"
[ "$L0_ONLY" = 1 ] || "$BIN" grade-answers "$LEVELS"

echo "== tables and figures"
LV="$(basename "$LEVELS_JOBS")" ISO="$(basename "$ISOLATED_JOBS")"
for d in "$LEVELS_JOBS" "$ISOLATED_JOBS"; do
  rc=0
  scripts/analyze.sh "$d" || rc=$?
  [ "$rc" -eq 0 ] || [ "$rc" -eq 3 ] || exit "$rc"  # 3: the delegation gate; the tables are written
done
.venv/bin/python -m subagents_bench.analysis.final "analysis/$LV" "analysis/$ISO" --out "analysis/${LV}_l0"
echo "[OK] analysis/$LV, analysis/$ISO and analysis/${LV}_l0 (experiment 1: l0.csv, l0.png)"
