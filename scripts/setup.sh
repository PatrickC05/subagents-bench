#!/usr/bin/env bash
# One-time setup for subagents-bench. Idempotent: safe to re-run at any time.
#
#   scripts/setup.sh            (no arguments; -h/--help prints this help)
#
# 1. checks uv and Docker (installed and running);
# 2. creates .venv with Python 3.13 and installs the package plus the exact
#    dependency versions pinned in uv.lock (harbor==0.24.0 included);
# 3. creates .env from .env.example if missing and checks the credentials
#    (reports set/missing only -- never prints a value);
# 4. runs the offline unit tests (no network, no Docker);
# 5. pre-downloads the Terminal-Bench 2.0 task definitions.
# It ends with a PASS/WARN/FAIL checklist and exits nonzero on any FAIL.
#
# Env: SKIP_TESTS=1 skips step 4; SUBAGENTS_BENCH_DATASET_CACHE=DIR uses (and
# downloads to) DIR instead of .cache/datasets.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() { awk 'NR > 1 && /^set -euo/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"; }
for arg in "$@"; do
  case "$arg" in
    -h | --help) usage; exit 0 ;;
    *) echo "[FAIL] unknown argument: $arg (setup.sh takes no arguments; see scripts/setup.sh --help)" >&2; exit 2 ;;
  esac
done

cd "$ROOT"

PYTHON_VERSION="3.13"
DATASET="terminal-bench@2.0"
VENV="$ROOT/.venv"
VENV_PY="$VENV/bin/python"
# uv installs the project into .venv no matter what the caller's shell says.
export UV_PROJECT_ENVIRONMENT="$VENV"

RESULTS=()
record() { # STATUS NAME DETAIL
  RESULTS+=("$1|$2|$3")
  printf '  [%s] %s: %s\n' "$1" "$2" "$3"
}
section() { printf '\n== %s\n' "$1"; }
summary() {
  local n_fail=0 status name detail
  printf '\n================ setup checklist ================\n'
  for r in "${RESULTS[@]}"; do
    IFS='|' read -r status name detail <<<"$r"
    printf '  [%-4s] %-22s %s\n' "$status" "$name" "$detail"
    if [ "$status" = "FAIL" ]; then n_fail=$((n_fail + 1)); fi
  done
  if [ "$n_fail" -eq 0 ]; then
    printf '\nSetup OK. Next:\n  scripts/run_experiment.sh configs/sweeps/smoke.yaml   # ~24 episodes, end-to-end check\n  scripts/run_experiment.sh                              # the pilot sweep\n'
  else
    printf '\n%d check(s) FAILED. Fix them and re-run scripts/setup.sh.\n' "$n_fail"
  fi
  return "$n_fail"
}

# --- 1. uv and Docker ----------------------------------------------------------
section "uv"
if command -v uv >/dev/null 2>&1; then
  record PASS "uv" "$(uv --version)"
else
  record FAIL "uv" "not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh  (then open a new shell)"
  summary || exit 1
fi

section "Docker"
if ! command -v docker >/dev/null 2>&1; then
  record FAIL "docker" "not installed (https://docs.docker.com/get-docker/)"
elif docker info >/dev/null 2>&1; then
  record PASS "docker" "daemon running ($(docker info --format '{{.ServerVersion}}, {{.Architecture}}, {{.NCPU}} CPUs' 2>/dev/null || echo '?'))"
else
  record FAIL "docker" "installed but the daemon is not running: start Docker Desktop (macOS) or dockerd (Linux)"
fi

# --- 2. Python environment ------------------------------------------------------
section "Python environment (.venv, Python $PYTHON_VERSION, uv.lock)"
if [ ! -x "$VENV_PY" ]; then
  uv venv --python "$PYTHON_VERSION" "$VENV"
fi
if [ ! -f uv.lock ]; then
  record FAIL "dependencies" "uv.lock missing (it is committed with the repo)"
elif uv sync --locked --all-extras --python "$PYTHON_VERSION"; then
  lock_sha="$( (sha256sum uv.lock 2>/dev/null || shasum -a 256 uv.lock) | cut -c1-12)"
  record PASS "dependencies" "installed exactly from uv.lock (sha256 $lock_sha...)"
else
  record FAIL "dependencies" "uv sync --locked failed (see output above; a stale uv.lock needs 'uv lock')"
fi

if [ -x "$VENV_PY" ]; then
  pyver="$("$VENV_PY" -c 'import platform; print(platform.python_version())')"
  case "$pyver" in
    "$PYTHON_VERSION".*) record PASS "python" "$pyver ($VENV_PY)" ;;
    *) record FAIL "python" "$pyver in .venv, expected $PYTHON_VERSION.x: delete .venv and re-run" ;;
  esac
  hv="$("$VENV_PY" -c 'import importlib.metadata as m; print(m.version("harbor"))' 2>/dev/null || echo missing)"
  if [ "$hv" = "0.24.0" ] && [ -x "$VENV/bin/harbor" ]; then
    record PASS "harbor" "0.24.0 ($VENV/bin/harbor)"
  else
    record FAIL "harbor" "found '$hv', expected 0.24.0"
  fi
  if "$VENV/bin/subagents-bench" --help >/dev/null 2>&1; then
    record PASS "subagents-bench CLI" "$VENV/bin/subagents-bench"
  else
    record FAIL "subagents-bench CLI" "'subagents-bench --help' failed: run it to see the error"
  fi
else
  record FAIL "python" "$VENV_PY missing"
fi

# --- 3. .env and credentials ------------------------------------------------------
section "Credentials (.env)"
if [ ! -f .env ]; then
  cp .env.example .env
  chmod 600 .env
  record WARN ".env" "created from .env.example: put your SAIL_API_KEY in .env, then re-run this script"
else
  record PASS ".env" "present"
fi

# Prints NAME=set|missing lines only; values never leave Python.
CRED_PY='
from pathlib import Path
from subagents_bench.credentials import describe_credentials, load_dotenv
load_dotenv(Path(".env"), override=False)
for name, state in describe_credentials(["SAIL_API_KEY", "PRIME_API_KEY", "PRIME_TEAM_ID"]).items():
    print(name + "=" + state)
'
cred_out="$("$VENV_PY" -c "$CRED_PY" 2>&1 || true)"
state_of() { printf '%s\n' "$cred_out" | sed -nE "s/^$1=(set|missing)$/\1/p" | head -1; }
sail="$(state_of SAIL_API_KEY)"
pkey="$(state_of PRIME_API_KEY)"
pteam="$(state_of PRIME_TEAM_ID)"
if [ -z "$sail" ]; then
  record FAIL "credentials" "could not check (subagents_bench.credentials failed to import)"
else
  if [ "$sail" = "set" ]; then
    record PASS "SAIL_API_KEY" "set"
  else
    record FAIL "SAIL_API_KEY" "missing: add it to .env (required for every sweep)"
  fi
  if [ "$pkey" = "set" ] && [ "$pteam" = "set" ]; then
    record PASS "Prime credentials" "PRIME_API_KEY and PRIME_TEAM_ID set (env, .env or prime CLI login)"
  else
    record WARN "Prime credentials" "missing (only needed for configs/sweeps/prime_size.yaml): 'prime login' or set PRIME_API_KEY + PRIME_TEAM_ID in .env"
  fi
fi

# --- 4. offline unit tests ---------------------------------------------------------
section "Unit tests (offline)"
if [ "${SKIP_TESTS:-0}" = "1" ]; then
  record WARN "unit tests" "skipped (SKIP_TESTS=1)"
elif test_out="$("$VENV_PY" -m pytest -q -m "not live and not docker" -p no:cacheprovider 2>&1)"; then
  record PASS "unit tests" "$(printf '%s\n' "$test_out" | tail -1)"
else
  printf '%s\n' "$test_out" | grep -E '^(FAILED|ERROR)' | head -30 || true
  record FAIL "unit tests" "$(printf '%s\n' "$test_out" | tail -1)"
fi

# --- 5. dataset ----------------------------------------------------------------------
section "Dataset ($DATASET)"
if ds_out="$("$VENV_PY" -m subagents_bench.sweep tasks "$DATASET" 2>&1)"; then
  record PASS "dataset" "$(printf '%s\n' "$ds_out" | head -1)"
else
  # The cause is in harbor's own message (network, proxy, GitHub rate limit,
  # missing git), which comes before the final "not available" line.
  printf '%s\n' "$ds_out" | head -20 | sed 's/^/    /' >&2
  cause="$(printf '%s\n' "$ds_out" | grep -m1 'download failed' || printf '%s\n' "$ds_out" | tail -1)"
  record FAIL "dataset" "$(printf '%s' "$cause" | cut -c1-240) (output above; the download needs network access to GitHub, or set SUBAGENTS_BENCH_DATASET_CACHE to an existing export)"
fi

summary || exit 1
