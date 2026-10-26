#!/usr/bin/env bash
# Run the whole final study of issue #23, model after model, then draw the cross-model figures.
# Resumable: re-run the same command after any stop; finished jobs and grades are kept.
#
#   scripts/run_all_final.sh [--only STEM...] [--parallel] [--jobs-at-once N] [--dry-run]
#
#   --only STEM...   run only these models (configs/prime/models/<STEM>.yaml), or `none` to only
#                    draw the figures; the figures use every model whose tables exist
#   --parallel       run every model at once, each with its two sweeps at once (run_final.sh
#                    --parallel), logging to logs/final/<stem>.log: the sessions take about 30
#                    minutes in all, plus preflight before and grading after (about 1 hour end to
#                    end). Ctrl-C stops every run.
#   --jobs-at-once N passed to scripts/run_final.sh, default 12
#   --dry-run        list every model's jobs; run nothing
#
# What runs (issue #23 is the source of truth):
#   experiments 1 and 2 (L0-L2 + isolated, 90 sessions each):  Opus 5.5, Luna, DeepSeek-V4.1-Flash-1017
#   experiment 1 only (L0 + isolated, 54 sessions each):        Terra, Sol, Sonnet 5.5
# Gemini 4 Argon and Kimi K3 are left out (we have no access to them), and Claude Haiku 5.5 and
# MiMo V2.6 are not on Prime. scripts/run_final_patrick.sh and scripts/run_final_jonathan.sh split
# the models between two machines.
# A model whose jobs fail (exit 1) or whose provider is down (4) does not stop the others; an
# interrupt (130), a billing error (3: every model bills the same Prime account) or a config error
# (2) does. The exit code is the first failure's (0: all done).
# Figures: analysis/final_l0/ (experiment 1, every model) and figures/final/ (experiment 2).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e '$d' -e 's/^# \{0,1\}//'; }

FULL=(claude-opus-5.5 gpt-5.6-luna deepseek-v4.1-flash-1017)
L0_ONLY=(gpt-5.6-sol gpt-5.6-terra claude-sonnet-5.5)
name() {  # the figures' label of a model file stem
  case "$1" in
    claude-opus-5.5) echo "Opus 5.5" ;; claude-sonnet-5.5) echo "Sonnet 5.5" ;;
    gpt-5.6-luna) echo "GPT-5.6 Luna" ;; gpt-5.6-terra) echo "GPT-5.6 Terra" ;; gpt-5.6-sol) echo "GPT-5.6 Sol" ;;
    deepseek-v4.1-flash-1017) echo "DeepSeek-V4.1-Flash" ;; *) echo "$1" ;;
  esac
}

ONLY=() PASS=() DRY_RUN=0 PARALLEL=0
while [ $# -gt 0 ]; do
  case "$1" in
    -h | --help) usage; exit 0 ;;
    --only) shift; while [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; do ONLY+=("$1"); shift; done; continue ;;
    --jobs-at-once) PASS+=(--jobs-at-once "$2"); shift ;;
    --dry-run) DRY_RUN=1; PASS+=(--dry-run) ;;
    --parallel) PARALLEL=1 ;;
    *) echo "[FAIL] unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
for m in ${ONLY[@]+"${ONLY[@]}"}; do  # a typo would silently run nothing; `none` draws the figures only
  case " none ${FULL[*]} ${L0_ONLY[*]} " in *" $m "*) ;; *) echo "[FAIL] --only: unknown model $m" >&2; exit 2 ;; esac
done
wanted() { [ ${#ONLY[@]} -eq 0 ] && return 0; for m in "${ONLY[@]}"; do [ "$m" = "$1" ] && return 0; done; return 1; }

first_rc=0
run() {  # STEM [--l0-only]
  wanted "$1" || return 0
  local rc=0
  scripts/run_final.sh --model "configs/prime/models/$1.yaml" "${@:2}" ${PASS[@]+"${PASS[@]}"} || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[FAIL] $1: scripts/run_final.sh exited $rc"
    [ "$first_rc" -ne 0 ] || first_rc=$rc
  fi
  case "$rc" in  # stop: interrupted (Ctrl-C), billing (every model bills the same account), config error
    130 | 3 | 2) echo "[ABORT] not starting the other models (exit $rc)"; exit "$rc" ;;
  esac
}
if [ "$PARALLEL" = 1 ] && [ "$DRY_RUN" = 0 ]; then
  # shellcheck source=scripts/parallel.sh
  . scripts/parallel.sh
  for m in "${FULL[@]}" "${L0_ONLY[@]}"; do
    wanted "$m" || continue
    l0=()
    case " ${L0_ONLY[*]} " in *" $m "*) l0=(--l0-only) ;; esac
    par_start "logs/final/$m.log" scripts/run_final.sh --model "configs/prime/models/$m.yaml" ${l0[@]+"${l0[@]}"} \
      --parallel ${PASS[@]+"${PASS[@]}"}
  done
  echo "follow along: tail -f logs/final/*.log"
  par_wait || first_rc=$?
else
  for m in "${FULL[@]}"; do run "$m"; done
  for m in "${L0_ONLY[@]}"; do run "$m" --l0-only; done
fi
[ "$DRY_RUN" = 1 ] && exit "$first_rc"

echo "== cross-model figures"
L0_ARGS=() EXP2_ARGS=()
for m in "${FULL[@]}" "${L0_ONLY[@]}"; do
  lv="analysis/final_levels-$m" iso="analysis/final_isolated-$m"
  if [ -f "$lv/episodes.csv" ] && [ -f "$lv/fixes.csv" ] && [ -f "$iso/episodes.csv" ] && [ -f "$iso/fixes.csv" ]; then
    L0_ARGS+=(--model "$(name "$m")=$lv:$iso")
  fi
done
for m in "${FULL[@]}"; do
  if [ -f "analysis/final_levels-$m/messages.csv" ]; then EXP2_ARGS+=(--model "$(name "$m")=analysis/final_levels-$m"); fi
done
figure() {  # a failed figure is reported, and the other still drawn
  local rc=0
  .venv/bin/python -m "$@" || rc=$?
  if [ "$rc" -ne 0 ]; then echo "[FAIL] $1 exited $rc"; [ "$first_rc" -ne 0 ] || first_rc=$rc; fi
}
if [ ${#L0_ARGS[@]} -gt 0 ]; then figure subagents_bench.analysis.final "${L0_ARGS[@]}" --out analysis/final_l0; fi
if [ ${#EXP2_ARGS[@]} -gt 0 ]; then figure subagents_bench.plots.final "${EXP2_ARGS[@]}" --out figures/final; fi
exit "$first_rc"
