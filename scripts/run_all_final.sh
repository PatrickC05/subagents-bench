#!/usr/bin/env bash
# Run the whole final study of issue #23, model after model, then draw the cross-model figures.
# Resumable: re-run the same command after any stop; finished jobs and grades are kept.
#
#   scripts/run_all_final.sh [--only STEM...] [--jobs-at-once N] [--dry-run]
#
#   --only STEM...   run only these models (configs/prime/models/<STEM>.yaml); the figures still
#                    use every model whose tables exist
#   --jobs-at-once N passed to scripts/run_final.sh, default 12
#   --dry-run        list every model's jobs; run nothing
#
# What runs (issue #23 is the source of truth):
#   experiments 1 and 2 (L0-L3 + isolated, 108 sessions each): Opus 5.5, Luna, DeepSeek-V4.1-Flash-1017
#   experiment 1 only (L0 + isolated, 54 sessions each):        Terra, Sol, Sonnet 5.5, Argon, Kimi K3
# Claude Haiku 5.5 and MiMo V2.6 are not on Prime and are left out.
# A model that fails does not stop the others; the exit code is the first failure's (0: all done).
# Figures: analysis/final_l0/ (experiment 1, every model) and figures/final/ (experiment 2).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e '$d' -e 's/^# \{0,1\}//'; }

FULL=(claude-opus-5.5 gpt-5.6-luna deepseek-v4.1-flash-1017)
L0_ONLY=(gpt-5.6-terra gpt-5.6-sol claude-sonnet-5.5 gemini-4-argon kimi-k3)
name() {  # the figures' label of a model file stem
  case "$1" in
    claude-opus-5.5) echo "Opus 5.5" ;; claude-sonnet-5.5) echo "Sonnet 5.5" ;;
    gpt-5.6-luna) echo "GPT-5.6 Luna" ;; gpt-5.6-terra) echo "GPT-5.6 Terra" ;; gpt-5.6-sol) echo "GPT-5.6 Sol" ;;
    deepseek-v4.1-flash-1017) echo "DeepSeek-V4.1-Flash" ;; gemini-4-argon) echo "Gemini 4 Argon" ;;
    kimi-k3) echo "Kimi K3" ;; *) echo "$1" ;;
  esac
}

ONLY=() PASS=() DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    -h | --help) usage; exit 0 ;;
    --only) shift; while [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; do ONLY+=("$1"); shift; done; continue ;;
    --jobs-at-once) PASS+=(--jobs-at-once "$2"); shift ;;
    --dry-run) DRY_RUN=1; PASS+=(--dry-run) ;;
    *) echo "[FAIL] unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
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
}
for m in "${FULL[@]}"; do run "$m"; done
for m in "${L0_ONLY[@]}"; do run "$m" --l0-only; done
[ "$DRY_RUN" = 1 ] && exit "$first_rc"

echo "== cross-model figures"
L0_ARGS=() EXP2_ARGS=()
for m in "${FULL[@]}" "${L0_ONLY[@]}"; do
  lv="analysis/final_levels-$m" iso="analysis/final_isolated-$m"
  if [ -f "$lv/episodes.csv" ] && [ -f "$iso/episodes.csv" ]; then L0_ARGS+=(--model "$(name "$m")=$lv:$iso"); fi
done
for m in "${FULL[@]}"; do
  if [ -f "analysis/final_levels-$m/messages.csv" ]; then EXP2_ARGS+=(--model "$(name "$m")=analysis/final_levels-$m"); fi
done
[ ${#L0_ARGS[@]} -eq 0 ] || .venv/bin/python -m subagents_bench.analysis.final "${L0_ARGS[@]}" --out analysis/final_l0
[ ${#EXP2_ARGS[@]} -eq 0 ] || .venv/bin/python -m subagents_bench.plots.final "${EXP2_ARGS[@]}" --out figures/final
exit "$first_rc"
