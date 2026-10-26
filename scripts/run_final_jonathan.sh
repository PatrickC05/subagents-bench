#!/usr/bin/env bash
# Jonathan's share of the final study (issue #23), every model at once:
#   Claude Opus 5.5: experiments 1 and 2 (108 sessions)
#   GPT-5.6 Sol, GPT-5.6 Terra: experiment 1 (54 sessions each)
# 216 sessions, about $1.3k on Prime. The sessions take about 30 minutes (every model and both
# its sweeps at once: scripts/run_all_final.sh --parallel), about 1 hour with preflight and
# grading. Logs: logs/final/<model>.log (tail -f logs/final/*.log). Ctrl-C stops every run.
# The other share is scripts/run_final_patrick.sh, on another machine, at the same time (each
# model has its own jobs dirs).
#
#   scripts/run_final_jonathan.sh [--jobs-at-once N] [--dry-run]
#
# Resumable: re-run after any stop. Options and exit codes: scripts/run_all_final.sh. The
# cross-model figures need both shares' analysis/final_* and jobs/final_* dirs in one checkout;
# then `scripts/run_all_final.sh --only none` draws them without running anything.
set -euo pipefail
case "${1:-}" in -h | --help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e '$d' -e 's/^# \{0,1\}//'; exit 0 ;; esac
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_all_final.sh" --parallel --only claude-opus-5.5 gpt-5.6-sol gpt-5.6-terra "$@"
