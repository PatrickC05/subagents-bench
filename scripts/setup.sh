#!/usr/bin/env bash
# One-time setup; safe to re-run.
#
#   scripts/setup.sh [-h | --help]
#
# Installs .venv (Python 3.13) with the exact versions pinned in uv.lock, harbor 0.24.0
# included; checks harbor and Docker; downloads the Terminal-Bench 2.0 tasks into harbor's
# task cache; creates .env from .env.example if missing and checks that SAIL_API_KEY is set
# (never printing it). Ends with exit 1 if any check failed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e '$d' -e 's/^# \{0,1\}//'; }
case "${1:-}" in
  -h | --help) usage; exit 0 ;;
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

if docker info >/dev/null 2>&1; then pass "Docker running"; else fail "Docker is not running: start Docker Desktop or dockerd"; fi

if .venv/bin/harbor dataset download terminal-bench@2.0 --cache >/dev/null; then
  pass "terminal-bench@2.0 tasks in harbor's cache"
else
  fail "dataset download failed (it needs network access to GitHub)"
fi

if [ ! -f .env ]; then
  cp .env.example .env
  chmod 600 .env
  echo "created .env from .env.example"
fi
if [ -n "${SAIL_API_KEY:-}" ] || grep -Eq '^(export )?SAIL_API_KEY=.' .env; then
  pass "SAIL_API_KEY set"
else
  fail "SAIL_API_KEY missing: add it to .env"
fi

if [ "$failed" = 0 ]; then echo "Setup OK. Next: scripts/run_experiment.sh configs/sweeps/smoke.yaml"; fi
exit "$failed"
