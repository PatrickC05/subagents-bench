#!/usr/bin/env bash
# One-time setup; safe to re-run.
#
#   scripts/setup.sh [--final] [-h | --help]
#
# Installs .venv (Python 3.13) with the exact versions pinned in uv.lock, harbor 0.24.0
# included; checks harbor and Docker; downloads the Terminal-Bench 2.0 tasks into harbor's
# task cache; creates .env from .env.example if missing and checks that PRIME_API_KEY is set
# (never printing it; SAIL_API_KEY too, for the configs/sail sweeps and their judge). Ends with
# exit 1 if any check failed.
# --final: for the final study (scripts/run_final.sh), which runs on Modal: skips Docker and
# Terminal-Bench, fetches the SWE-sweep tasks (scripts/get_tasks.sh), checks for a Modal token,
# and needs SAIL_API_KEY as well (the questions' judge is DeepSeek on Sail for every model).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e '$d' -e 's/^# \{0,1\}//'; }
FINAL=0
case "${1:-}" in
  -h | --help) usage; exit 0 ;;
  --final) FINAL=1 ;;
  "") ;;
  *) echo "unknown argument: $1 (see scripts/setup.sh --help)" >&2; exit 2 ;;
esac
cd "$ROOT"

failed=0
pass() { printf '[PASS] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*"; failed=1; }

if ! command -v uv >/dev/null 2>&1; then
  fail "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
if uv sync --locked --all-extras --python 3.13; then
  pass "dependencies installed exactly from uv.lock"
else
  fail "uv sync --locked failed (see above)"
fi

hv="$(.venv/bin/harbor --version 2>/dev/null || true)"
if [ "$hv" = "0.24.0" ]; then pass "harbor $hv"; else fail "harbor 0.24.0 not in .venv (found: ${hv:-none})"; fi

if [ "$FINAL" = 1 ]; then
  if scripts/get_tasks.sh; then pass "SWE-sweep tasks"; else fail "scripts/get_tasks.sh failed (see above)"; fi
  # the same check preflight makes (both MODAL_TOKEN_ID and MODAL_TOKEN_SECRET, or ~/.modal.toml)
  if msg="$(.venv/bin/python -c 'from subagents_bench.sweep import modal_ready
ok, msg = modal_ready(); print(msg); raise SystemExit(0 if ok else 1)')"; then
    pass "Modal: $msg"
  else
    fail "Modal: $msg"
  fi
else
  if docker info >/dev/null 2>&1; then pass "Docker running"; else fail "Docker is not running: start Docker Desktop or dockerd"; fi

  if .venv/bin/harbor dataset download terminal-bench@2.0 --cache >/dev/null; then
    pass "terminal-bench@2.0 tasks in harbor's cache"
  else
    fail "dataset download failed (it needs network access to GitHub)"
  fi
fi

if [ ! -f .env ]; then
  cp .env.example .env
  chmod 600 .env
  echo "created .env from .env.example"
fi
if [ -n "${PRIME_API_KEY:-}" ] || grep -Eq '^(export )?PRIME_API_KEY=.' .env; then
  pass "PRIME_API_KEY set"
else
  fail "PRIME_API_KEY missing: add it to .env"
fi
if [ -n "${SAIL_API_KEY:-}" ] || grep -Eq '^(export )?SAIL_API_KEY=.' .env; then
  pass "SAIL_API_KEY set"
else
  if [ "$FINAL" = 1 ]; then
    fail "SAIL_API_KEY missing: the final study's judge (grade-answers) runs on Sail; add it to .env"
  else
    printf '[INFO] %s\n' "SAIL_API_KEY not set: the sweeps on configs/sail/base.yaml (and their questions and judge, under --model too) need it"
  fi
fi

if [ "$failed" = 0 ] && [ "$FINAL" = 1 ]; then
  echo "Setup OK. Next: scripts/run_all_final.sh --dry-run"
elif [ "$failed" = 0 ]; then
  echo "Setup OK. Next: a sweep on a Prime model (README, *Models*), e.g."
  echo "  scripts/run_experiment.sh configs/sweeps/swe_interactive_levels_smoke.yaml --model configs/prime/models/gpt-5.6-luna.yaml --job 1:single_baseline"
  echo "(the sweeps without --model, like configs/sweeps/smoke.yaml, run on Sail: SAIL_API_KEY)"
fi
exit "$failed"
