#!/usr/bin/env bash
# Run the final study (issue #23) for one model, end to end: the sessions, the grading and the
# tables. Resumable: re-run the same command after any stop; finished jobs and grades are kept.
#
#   scripts/run_final.sh [--model FILE] [--l0-only | --l1l2-only] [--slow] [--parallel] [--grade-only]
#                        [--jobs-at-once N] [--dry-run]
#
#   --model FILE     the model, a configs/prime/models/*.yaml file, default
#                    configs/prime/models/deepseek-v4.1-flash-1017.yaml; the orchestrator and its
#                    subagents run it, in jobs/final_levels-<stem> and jobs/final_isolated-<stem>
#   --l0-only        experiment 1 only: the three settings at L0 (18 sessions) and the isolated
#                    baseline (36). Without it, experiment 2 as well: the three settings at
#                    L0-L2 (54 sessions, L0 included), and the isolated baseline.
#   --l1l2-only      only the three settings at L1 and L2 (36 sessions; no L0, no isolated baseline):
#                    the other half of experiment 2, for splitting a model between two machines
#                    (one runs --l0-only, the other --l1l2-only)
#   --slow           2 sessions at a time, one job at a time (configs/sweeps/final_*_slow.yaml, in
#                    jobs/final_*_slow-<stem>), for a model Prime throttles at 6 sessions per job
#                    (GPT-5.6 Luna: 16 requests at once per model and account); not with --parallel
#   --parallel       run the two sweeps at the same time (after preflight), each logging to
#                    logs/final/<stem>-levels.log and -isolated.log: every session of the model at
#                    once, so the sessions take about 30 minutes in all
#   --jobs-at-once N harbor jobs at once per sweep, default 12 (a job runs its 6 tasks at once;
#                    12 runs the levels sweep's 9 jobs and the isolated sweep's 6 all at once)
#   --grade-only     run no session: grade and tabulate the trials the sweeps have (after a run
#                    that ended with failed trials, without rerunning them)
#   --dry-run        print the jobs of both sweeps; run and grade nothing
#
# Trials that fail (a provider error past every retry, an agent crash) are not rerun: when a sweep ends with failed
# trials (exit 1), the grading and tables still run, and the analysis counts a session that ended
# in a provider error or an agent crash as a failure (its bugs not fixed, its unanswered requests never answered).
# Rerunning the same command without --grade-only does rerun them.
#
# Steps: copy the fixed question sets (data/questions) into the levels sweep's own hints dir, which
# every model's run reads, so every model is asked the same questions (a different file already
# there stops the run; --dry-run writes nothing); run the levels sweep
# (preflight with harbor's oracle first) and the isolated sweep (configs/sweeps/final_*.yaml);
# replay each trial's snapshots to time every fix (`fix-times`, no LLM call); judge the answers to
# questions (`grade-answers`, with the sweep's own model, DeepSeek-V4.1-Flash on Sail, for every
# model); write the tables and figures with scripts/analyze.sh to analysis/<jobs dir name>/; then
# experiment 1's cost-vs-fixed table (subagents_bench.analysis.final) to analysis/final_l0-<stem>/,
# with the isolated bugs also graded at the time each had in the shared sessions.
# Needs scripts/setup.sh --final: PRIME_API_KEY (PRIME_TEAM_ID to bill a team) and SAIL_API_KEY
# (the judge) in .env, the SWE-sweep tasks and a Modal token.
# Exit codes: those of scripts/run_experiment.sh (0 done, 1 failed jobs, 2 config error, 3 billing,
# 4 provider outage, 130 interrupted), or of the grading step that failed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e '$d' -e 's/^# \{0,1\}//'; }

LEVELS=configs/sweeps/final_levels.yaml
ISOLATED=configs/sweeps/final_isolated.yaml
MODEL=configs/prime/models/deepseek-v4.1-flash-1017.yaml
L0_ONLY=0 L1L2_ONLY=0 SLOW=0 DRY_RUN=0 PARALLEL=0 GRADE_ONLY=0 AT_ONCE=
while [ $# -gt 0 ]; do
  case "$1" in
    -h | --help) usage; exit 0 ;;
    --model) MODEL="$2"; shift ;;
    --l0-only) L0_ONLY=1 ;;
    --l1l2-only) L1L2_ONLY=1 ;;
    --slow) SLOW=1 ;;
    --jobs-at-once) AT_ONCE="$2"; shift ;;
    --dry-run) DRY_RUN=1 ;;
    --parallel) PARALLEL=1 ;;
    --grade-only) GRADE_ONLY=1 ;;
    *) echo "[FAIL] unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
[ -f "$MODEL" ] || { echo "[FAIL] model file not found: $MODEL" >&2; exit 2; }
[ "$L0_ONLY" = 0 ] || [ "$L1L2_ONLY" = 0 ] || { echo "[FAIL] --l0-only and --l1l2-only exclude each other" >&2; exit 2; }
[ "$SLOW" = 0 ] || [ "$PARALLEL" = 0 ] || { echo "[FAIL] --slow runs one job at a time: drop --parallel" >&2; exit 2; }
if [ "$SLOW" = 1 ]; then
  LEVELS=configs/sweeps/final_levels_slow.yaml
  ISOLATED=configs/sweeps/final_isolated_slow.yaml
  AT_ONCE="${AT_ONCE:-1}"
fi
AT_ONCE="${AT_ONCE:-12}"

BIN="$ROOT/.venv/bin/subagents-bench"
[ -x "$BIN" ] || { echo "[FAIL] $BIN not found: run scripts/setup.sh --final first" >&2; exit 1; }
jobs_dir() {  # the jobs dir of a spec, with --model when given
  "$ROOT/.venv/bin/python" -c 'import sys; from subagents_bench.sweep import load_spec
print(load_spec(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None).jobs_dir)' "$@"
}
HINTS="$(jobs_dir "$LEVELS")/hints"  # the sweep's own: every model's run reads its questions there
LEVELS_JOBS="$(jobs_dir "$LEVELS" "$MODEL")"
ISOLATED_JOBS="$(jobs_dir "$ISOLATED" "$MODEL")"
STEM="$(basename "$MODEL" .yaml)"

ONLY=()
if [ "$L0_ONLY" = 1 ]; then
  ONLY=(--only background_hinted_baseline manager_background_baseline manager_owner_baseline)
elif [ "$L1L2_ONLY" = 1 ]; then
  ONLY=(--only background_hinted_eta manager_background_eta manager_owner_eta
        background_hinted_questions manager_background_questions manager_owner_questions)
fi
DRY=()
[ "$DRY_RUN" = 1 ] && DRY=(--dry-run)

# Every model must be asked data/questions: a different file already in the hints dir is an error
# (preflight checks every condition of the levels sweep, L2 included, so --l0-only needs them too).
for f in data/questions/*.questions.json; do
  dest="$HINTS/$(basename "$f")"
  if [ -e "$dest" ]; then
    cmp -s "$f" "$dest" || { echo "[FAIL] $dest differs from $f: every model must get the same questions" \
      "(remove it to use $f)" >&2; exit 2; }
  elif [ "$DRY_RUN" = 1 ]; then
    echo "would copy $f -> $HINTS/"
  else
    mkdir -p "$HINTS"
    cp "$f" "$dest"
  fi
done

rc=0
if [ "$GRADE_ONLY" = 1 ]; then
  echo "== $STEM: --grade-only: no session runs"
elif [ "$PARALLEL" = 1 ] && [ "$DRY_RUN" = 0 ]; then
  if [ "${SKIP_PREFLIGHT:-0}" != 1 ]; then
    echo "== $STEM: preflight"
    "$BIN" preflight "$LEVELS" --oracle --model "$MODEL"
  fi
  echo "== $STEM: both sweeps at once -> $LEVELS_JOBS, $ISOLATED_JOBS"
  # shellcheck source=scripts/parallel.sh
  . scripts/parallel.sh
  par_start "logs/final/$STEM-levels.log" env SKIP_PREFLIGHT=1 scripts/run_experiment.sh "$LEVELS" --model "$MODEL" \
    ${ONLY[@]+"${ONLY[@]}"} --jobs-at-once "$AT_ONCE"
  par_start "logs/final/$STEM-isolated.log" env SKIP_PREFLIGHT=1 scripts/run_experiment.sh "$ISOLATED" --model "$MODEL" \
    --jobs-at-once "$AT_ONCE"
  par_wait || rc=$?
else
  echo "== $STEM: levels sweep -> $LEVELS_JOBS"
  scripts/run_experiment.sh "$LEVELS" --model "$MODEL" ${ONLY[@]+"${ONLY[@]}"} --jobs-at-once "$AT_ONCE" \
    ${DRY[@]+"${DRY[@]}"} || rc=$?
  if { [ "$rc" -eq 0 ] || [ "$rc" -eq 1 ]; } && [ "$L1L2_ONLY" = 0 ]; then
    echo "== $STEM: isolated baseline -> $ISOLATED_JOBS"
    rc2=0
    SKIP_PREFLIGHT=1 scripts/run_experiment.sh "$ISOLATED" --model "$MODEL" --jobs-at-once "$AT_ONCE" \
      ${DRY[@]+"${DRY[@]}"} || rc2=$?
    [ "$rc2" -eq 0 ] || [ "$rc" -ne 0 ] || rc=$rc2
    [ "$rc2" -le 1 ] || rc=$rc2
  fi
fi
[ "$DRY_RUN" = 1 ] && exit "$rc"
case "$rc" in
  0) ;;
  1) echo "[WARN] $STEM: some trials failed; grading what the sweeps have (a provider error or an agent crash counts as a failure)" ;;
  *) exit "$rc" ;;  # interrupted, billing, config error, outage: stop
esac

echo "== $STEM: grading"
"$BIN" fix-times "$LEVELS" --model "$MODEL"
[ "$L1L2_ONLY" = 1 ] || "$BIN" fix-times "$ISOLATED" --model "$MODEL"
[ "$L0_ONLY" = 1 ] || "$BIN" grade-answers "$LEVELS" --model "$MODEL"

echo "== $STEM: tables and figures"
LV="$(basename "$LEVELS_JOBS")" ISO="$(basename "$ISOLATED_JOBS")"
DIRS=("$LEVELS_JOBS")
[ "$L1L2_ONLY" = 1 ] || DIRS+=("$ISOLATED_JOBS")
for d in "${DIRS[@]}"; do
  arc=0
  scripts/analyze.sh "$d" || arc=$?
  [ "$arc" -eq 0 ] || [ "$arc" -eq 3 ] || exit "$arc"  # 3: the delegation gate; the tables are written
done
if [ "$L1L2_ONLY" = 1 ]; then
  echo "[OK] analysis/$LV (L1-L2 only: experiment 1 comes from the --l0-only half)"
  exit "$rc"
fi
L0OUT="analysis/final_l0-$STEM"
[ "$SLOW" = 0 ] || L0OUT="analysis/final_l0_slow-$STEM"
.venv/bin/python -m subagents_bench.analysis.final --model "$STEM=analysis/$LV:analysis/$ISO" --out "$L0OUT"
echo "[OK] analysis/$LV, analysis/$ISO and $L0OUT (experiment 1: l0.csv, l0.png)"
exit "$rc"  # 1 if trials failed: graded as failures, rerun the command (without --grade-only) to rerun them
