# Run commands at once, each in its own process group with its output in a log file; sourced by
# scripts/run_final.sh and scripts/run_all_final.sh (--parallel).
#
#   par_start LOG CMD...   start CMD in the background, output to LOG
#   par_wait               wait for every started command; prints each exit code; returns the first
#                          non-zero one (0 if all succeeded)
#
# Without job control a background command ignores SIGINT, so a Ctrl-C would leave the sweeps
# running. With it (set -m) each command gets its own process group, which the terminal's Ctrl-C
# does not reach, so the trap forwards SIGINT to every group: each sweep then stops its harbor jobs
# as on a Ctrl-C of its own (cancelling the trials, removing the containers) and exits 130.

set -m
PAR_PIDS=() PAR_NAMES=()
par_start() {
  local log="$1"
  shift
  mkdir -p "$(dirname "$log")"
  "$@" >"$log" 2>&1 &
  PAR_PIDS+=($!)
  PAR_NAMES+=("$log")
  echo "started: $* (log: $log)"
}
par_interrupt() {
  echo "Ctrl-C: stopping every run (each removes its containers; this can take a few minutes)" >&2
  local p
  for p in ${PAR_PIDS[@]+"${PAR_PIDS[@]}"}; do kill -INT -- "-$p" 2>/dev/null || true; done
}
trap par_interrupt INT TERM
par_wait() {
  local i rc first=0
  for i in "${!PAR_PIDS[@]}"; do
    while true; do  # a trapped signal ends `wait` early: wait again until the command has exited
      rc=0
      wait "${PAR_PIDS[$i]}" || rc=$?
      kill -0 "${PAR_PIDS[$i]}" 2>/dev/null || break
    done
    if [ "$rc" -eq 0 ]; then echo "[OK] ${PAR_NAMES[$i]}"; else echo "[FAIL] exit $rc: ${PAR_NAMES[$i]}"; fi
    [ "$first" -ne 0 ] || first=$rc
  done
  return "$first"
}
