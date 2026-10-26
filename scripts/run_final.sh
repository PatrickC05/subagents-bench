#!/usr/bin/env bash
# Run the final study (issue #23) for one model, end to end: the sessions, the grading and the
# tables. Resumable: re-run the same command after any stop; finished jobs and grades are kept.
#
#   scripts/run_final.sh [--model FILE] [--l0-only] [--jobs-at-once N] [--dry-run]
#
#   --model FILE     the model, a configs/prime/models/*.yaml file, default
#                    configs/prime/models/deepseek-v4.1-flash-1017.yaml; the orchestrator and its
#                    subagents run it, in jobs/final_levels-<stem> and jobs/final_isolated-<stem>
#   --l0-only        experiment 1 only: the three settings at L0 (18 sessions) and the isolated
#                    baseline (36). Without it, experiment 2 as well: the three settings at every
#                    level, L0-L3 (72 sessions, L0 included), and the isolated baseline.
#   --jobs-at-once N harbor jobs at once per sweep, default 12 (a job runs its 6 tasks at once)
#   --dry-run        print the jobs of both sweeps; run and grade nothing
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
L0_ONLY=0 DRY_RUN=0 AT_ONCE=12
while [ $# -gt 0 ]; do
  case "$1" in
    -h | --help) usage; exit 0 ;;
    --model) MODEL="$2"; shift ;;
    --l0-only) L0_ONLY=1 ;;
    --jobs-at-once) AT_ONCE="$2"; shift ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "[FAIL] unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
[ -f "$MODEL" ] || { echo "[FAIL] model file not found: $MODEL" >&2; exit 2; }

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
fi
DRY=()
[ "$DRY_RUN" = 1 ] && DRY=(--dry-run)

# Every model must be asked data/questions: a different file already in the hints dir is an error
# (preflight checks every condition of the levels sweep, L2-L3 included, so --l0-only needs them too).
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

echo "== $STEM: levels sweep -> $LEVELS_JOBS"
scripts/run_experiment.sh "$LEVELS" --model "$MODEL" ${ONLY[@]+"${ONLY[@]}"} --jobs-at-once "$AT_ONCE" \
  ${DRY[@]+"${DRY[@]}"}
echo "== $STEM: isolated baseline -> $ISOLATED_JOBS"
SKIP_PREFLIGHT=1 scripts/run_experiment.sh "$ISOLATED" --model "$MODEL" --jobs-at-once "$AT_ONCE" ${DRY[@]+"${DRY[@]}"}
[ "$DRY_RUN" = 1 ] && exit 0

echo "== $STEM: grading"
"$BIN" fix-times "$LEVELS" --model "$MODEL"
"$BIN" fix-times "$ISOLATED" --model "$MODEL"
[ "$L0_ONLY" = 1 ] || "$BIN" grade-answers "$LEVELS" --model "$MODEL"

echo "== $STEM: tables and figures"
LV="$(basename "$LEVELS_JOBS")" ISO="$(basename "$ISOLATED_JOBS")"
for d in "$LEVELS_JOBS" "$ISOLATED_JOBS"; do
  rc=0
  scripts/analyze.sh "$d" || rc=$?
  [ "$rc" -eq 0 ] || [ "$rc" -eq 3 ] || exit "$rc"  # 3: the delegation gate; the tables are written
done
.venv/bin/python -m subagents_bench.analysis.final --model "$STEM=analysis/$LV:analysis/$ISO" --out "analysis/final_l0-$STEM"
echo "[OK] analysis/$LV, analysis/$ISO and analysis/final_l0-$STEM (experiment 1: l0.csv, l0.png)"
