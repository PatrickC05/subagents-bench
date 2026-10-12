"""Agent tools (bash, read_file, list_dir, search, edit_file) and their registry.

Every tool is ``fn(ctx: ToolContext, args: dict) -> ToolOutput``; ``ctx``
carries the environment, the agent id, the agent's starting directory, the
truncation limit and the command-timeout budget. :func:`run_tool` is the
runtime's entry point and never raises except
:class:`~subagents_bench.types.AgentStopped`.

**Per-agent shell state.** Each agent has a state dir
``<sb_root>/<agent_id>/`` (``/tmp/.sb/<agent_id>/`` in containers) holding its
persisted ``cwd`` and ``env.sh`` (``export -p``). Every shell-backed tool call
is one ``env.exec`` of an *outer* script that:

1. writes the tool's script, base64-encoded, to ``cmd_<nonce>.sh`` and a
   generated wrapper to ``wrap_<nonce>.sh`` (byte-exact: heredocs and quotes in
   model commands need no escaping);
2. runs ``timeout -k 5 T bash wrap_<nonce>.sh`` (plain ``bash`` if the sandbox
   has no compatible ``timeout``, probed once per env) with stdin from
   /dev/null and stdout+stderr going to a file in the state dir -- so a
   background job that keeps its stdout open cannot hang the call;
3. prints the output (head and tail only, if huge) followed by a sentinel line
   carrying the exit code and the cwd before/after.

The wrapper restores the agent's cwd (falling back to ``ctx.initial_cwd``)
and, for ``bash``, its exported variables, sources the command in that same
shell, and on EXIT saves ``pwd`` and ``export -p`` (bash only; the read-only
tools never change persisted state). It uses only shell builtins, so a
command that breaks ``PATH`` cannot break state saving.

**Timeouts.** The host waits ``T + HOST_TIMEOUT_GRACE_SEC`` (15 s). An exit
code of 124/137 after at least ~T seconds, or a host-side timeout, sets
``timed_out``. After a host-side timeout a follow-up exec fetches the partial
output and, on Linux, kills the command's process tree via ``/proc`` (the
fallback when there is no ``timeout`` binary).
"""

from __future__ import annotations

import base64
import functools
import posixpath
import shlex
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from . import workspace
from .environment import ExecResult, check_stopped, plan_b64_upload
from .types import AgentStopped, ToolContext, ToolOutput, ToolSpec

# Extra seconds the host waits beyond the in-sandbox timeout before giving up.
HOST_TIMEOUT_GRACE_SEC = 15
# timeout -k: SIGKILL this long after the SIGTERM.
KILL_AFTER_SEC = 5
# Exit codes of `timeout` (TERM worked / needed KILL).
TIMED_OUT_EXIT_CODES = (124, 137)
# Timeout for the post-timeout follow-up exec.
FOLLOWUP_TIMEOUT_SEC = 30

READ_MAX_LINES = 2000
READ_MAX_LINE_CHARS = 2000
LIST_MAX_DEPTH = 5
LIST_MAX_ENTRIES = 500
SEARCH_MAX_LINES = 200
SEARCH_MAX_LINE_CHARS = 400
EDIT_MAX_BYTES = 10_000_000
# Paths skipped by list_dir expansion and grep-based search.
_SKIP_DIRS = (".git", "node_modules", "__pycache__")


class ToolArgError(ValueError):
    """Bad tool arguments from the model (reported back as a tool error)."""


# ---------------------------------------------------------------------------
# truncation
# ---------------------------------------------------------------------------


def _elision_marker(n: int) -> str:
    return f"\n[... {n} characters elided ...]\n"


def _truncate_parts(head: str, tail: str, elided: int, limit: int) -> tuple[str, bool]:
    """Keep the start of ``head`` and the end of ``tail`` within ``limit``
    chars; ``elided`` chars between them were already dropped upstream."""
    limit = max(limit, 0)
    if not elided and not tail:
        if len(head) <= limit:
            return head, False
        keep_h = limit // 2
        keep_t = limit - keep_h
        dropped = len(head) - keep_h - keep_t
        return head[:keep_h] + _elision_marker(dropped) + head[len(head) - keep_t :], True
    keep_h = min(len(head), limit // 2)
    keep_t = min(len(tail), limit - keep_h)
    dropped = len(head) + elided + len(tail) - keep_h - keep_t
    return head[:keep_h] + _elision_marker(dropped) + tail[len(tail) - keep_t :], True


def truncate_output(text: str, limit: int) -> tuple[str, bool]:
    """Head + tail of ``text`` within ``limit`` chars, with a marker giving
    the exact number of elided characters. Returns (text, truncated)."""
    return _truncate_parts(text, "", 0, limit)


# ---------------------------------------------------------------------------
# exec accounting
# ---------------------------------------------------------------------------


@dataclass
class _Meter:
    """Counts exec round-trips and the seconds spent inside them."""

    n: int = 0
    sec: float = 0.0

    def exec(self, env: Any, command: str, timeout_sec: float) -> ExecResult:
        check_stopped(env)
        t0 = time.monotonic()
        try:
            return env.exec(command, timeout_sec=timeout_sec)
        finally:
            self.n += 1
            self.sec += time.monotonic() - t0


def _tool(fn: Callable[[ToolContext, dict, _Meter], ToolOutput]) -> Callable[[ToolContext, dict], ToolOutput]:
    """Give ``fn`` a meter, convert failures to error outputs, fill n_exec/exec_sec."""

    @functools.wraps(fn)
    def wrapper(ctx: ToolContext, args: dict) -> ToolOutput:
        meter = _Meter()
        try:
            if not isinstance(args, dict):
                raise ToolArgError(f"arguments must be a JSON object, got {type(args).__name__}")
            out = fn(ctx, args, meter)
        except AgentStopped:
            raise
        except ToolArgError as e:
            out = ToolOutput(text=f"[error] {e}", is_error=True, error=str(e))
        except Exception as e:  # noqa: BLE001 - surfaced to the model, never crashes the agent
            msg = f"{type(e).__name__}: {e}"
            out = ToolOutput(text=f"[error] {fn.__name__} failed: {msg}", is_error=True, error=msg)
        out.n_exec = meter.n
        out.exec_sec = meter.sec
        return out

    return wrapper


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def _arg_str(args: dict, name: str, *, required: bool = False, default: str | None = None) -> str | None:
    v = args.get(name)
    if v is None:
        if required:
            raise ToolArgError(f"missing required argument {name!r}")
        return default
    if not isinstance(v, str):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return str(v)
        raise ToolArgError(f"argument {name!r} must be a string")
    return v


def _arg_int(args: dict, name: str, *, default: int | None = None) -> int | None:
    v = args.get(name)
    if v is None or v == "":
        return default
    if isinstance(v, bool):
        raise ToolArgError(f"argument {name!r} must be an integer")
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        try:
            return int(v.strip())
        except ValueError:
            pass
    raise ToolArgError(f"argument {name!r} must be an integer, got {v!r}")


# ---------------------------------------------------------------------------
# the per-agent shell
# ---------------------------------------------------------------------------


def _safe_agent(agent_id: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in agent_id) or "_"


def agent_state_dir(env: Any, agent_id: str) -> str:
    """``<sb_root>/<agent_id>``: the agent's persisted cwd/env live here."""
    return f"{workspace.sb_root(env)}/{_safe_agent(agent_id)}"


def _initial_cwd(ctx: ToolContext) -> str:
    return ctx.initial_cwd or getattr(ctx.env, "workdir", None) or "/"


_WRAPPER = """\
# subagents-bench per-agent shell wrapper (generated)
__sb_S={S}
__sb_init={INIT}
echo "$$" > "$__sb_S/pid_{N}"
__sb_cwd=; [ -f "$__sb_S/cwd" ] && read -r __sb_cwd < "$__sb_S/cwd"
[ -n "$__sb_cwd" ] || __sb_cwd=$__sb_init
{RESTORE_ENV}
if ! cd -- "$__sb_cwd" 2>/dev/null; then
  echo "[note: working directory $__sb_cwd does not exist; running in $__sb_init instead]"
  cd -- "$__sb_init" 2>/dev/null || cd /
fi
echo "$PWD" > "$__sb_S/start_{N}"
{SAVE_TRAP}
. "$__sb_S/cmd_{N}.sh"
"""

_RESTORE_ENV = 'if [ -f "$__sb_S/env.sh" ]; then . "$__sb_S/env.sh" > /dev/null 2>&1; fi'

# Builtins only: works even after the command broke PATH. A command that
# replaces this trap or exec()s away skips saving (state stays as before).
_SAVE_TRAP = """\
__sb_save() {
  pwd > "$__sb_S/cwd" 2>/dev/null
  export -p > "$__sb_S/env.sh" 2>/dev/null
  exit "$__sb_rc"
}
# Capture the status, then drop the command's -e/-u/-x so they cannot
# affect (or trace) the save; the group's stderr hides its own trace.
trap '{ __sb_rc=$?; set +eux; } 2>/dev/null; __sb_save' EXIT"""

# Shared by the outer script and the post-timeout follow-up.
_EMIT_FN = """\
__sb_emit() {
  __sb_sz=$(wc -c < "$1" 2>/dev/null); __sb_sz=$((__sb_sz + 0))
  if [ "$2" -le 0 ] || [ "$__sb_sz" -le "$2" ]; then cat "$1" 2>/dev/null
  else
    __sb_h=$(( $2 / 2 ))
    head -c "$__sb_h" "$1"
    printf '\\n__SB_ELIDED_%s %s\\n' "$__sb_nonce" $((__sb_sz - 2 * __sb_h))
    tail -c "$__sb_h" "$1"
  fi
}
__sb_enc() { printf '%s' "$1" | base64 | tr -d '\\n'; }"""


def _after_cwd(init: str, persist: bool) -> str:
    # The cwd the agent's next call will start in: the persisted one, else
    # the initial cwd (nothing saved yet, e.g. the first call was SIGKILLed).
    if not persist:
        return "__sb_a=$__sb_b"
    return (
        '__sb_a=; [ -f "$__sb_S/cwd" ] && read -r __sb_a < "$__sb_S/cwd"\n'
        f'[ -n "$__sb_a" ] || __sb_a={shlex.quote(init)}'
    )


def _outer_script(
    state: str, nonce: str, init: str, stage_cmd: str, stage_wrap: str, runner: str, cap_bytes: int, persist: bool
) -> str:
    after = _after_cwd(init, persist)
    files = " ".join(f'"$__sb_S/{f}_{nonce}{ext}"' for f, ext in (("out", ""), ("cmd", ".sh"), ("wrap", ".sh"), ("pid", ""), ("start", "")))
    return f"""\
__sb_S={shlex.quote(state)}
__sb_nonce={nonce}
{_EMIT_FN}
mkdir -p "$__sb_S" 2>/dev/null || {{ echo "[sb-error] cannot create state dir $__sb_S"; exit 97; }}
{stage_cmd} || {{ echo "[sb-error] failed to stage the command (is base64 available?)"; exit 97; }}
{stage_wrap} || {{ echo "[sb-error] failed to stage the wrapper"; exit 97; }}
{runner} "$__sb_S/wrap_{nonce}.sh" > "$__sb_S/out_{nonce}" 2>&1 < /dev/null
__sb_rc=$?
__sb_b=; [ -f "$__sb_S/start_{nonce}" ] && read -r __sb_b < "$__sb_S/start_{nonce}"
{after}
__sb_emit "$__sb_S/out_{nonce}" {cap_bytes}
rm -f {files}
printf '\\n__SB_META_%s %s x%s x%s\\n' "$__sb_nonce" "$__sb_rc" "$(__sb_enc "$__sb_b")" "$(__sb_enc "$__sb_a")"
"""


def _followup_script(state: str, nonce: str, init: str, cap_bytes: int, persist: bool) -> str:
    """After a host-side timeout: kill the command's process tree (Linux
    /proc walk, only if the recorded pid is still our wrapper), then emit the
    partial output and clean up."""
    after = _after_cwd(init, persist)
    files = " ".join(f'"$__sb_S/{f}_{nonce}{ext}"' for f, ext in (("out", ""), ("cmd", ".sh"), ("wrap", ".sh"), ("pid", ""), ("start", "")))
    return f"""\
__sb_S={shlex.quote(state)}
__sb_nonce={nonce}
{_EMIT_FN}
__sb_children() {{
  local p=$1 st line rest ppid
  for st in /proc/[0-9]*/stat; do
    read -r line < "$st" 2>/dev/null || continue
    rest=${{line##*) }}; ppid=${{rest#* }}; ppid=${{ppid%% *}}
    [ "$ppid" = "$p" ] && echo "${{line%% *}}"
  done
}}
__sb_tree() {{ local c; echo "$1"; for c in $(__sb_children "$1"); do __sb_tree "$c"; done; }}
if [ -f "$__sb_S/pid_{nonce}" ] && [ -d /proc/self ]; then
  read -r __sb_p < "$__sb_S/pid_{nonce}"
  case "$(tr '\\000' ' ' < "/proc/$__sb_p/cmdline" 2>/dev/null)" in
    *wrap_{nonce}*) kill -KILL $(__sb_tree "$__sb_p") 2>/dev/null ;;
  esac
fi
__sb_b=; [ -f "$__sb_S/start_{nonce}" ] && read -r __sb_b < "$__sb_S/start_{nonce}"
{after}
__sb_emit "$__sb_S/out_{nonce}" {cap_bytes}
rm -f {files}
printf '\\n__SB_META_%s - x%s x%s\\n' "$__sb_nonce" "$(__sb_enc "$__sb_b")" "$(__sb_enc "$__sb_a")"
"""


@dataclass
class ShellRun:
    """Parsed result of one command run through the agent's shell."""

    head: str  # output (all of it, unless the sandbox elided the middle)
    tail: str = ""  # output after the sandbox-side elision
    elided: int = 0  # bytes elided in the sandbox (huge outputs only)
    exit_code: int | None = None
    cwd_before: str | None = None
    cwd_after: str | None = None
    timed_out: bool = False
    host_timed_out: bool = False
    error: str | None = None  # the wrapper itself could not run
    notes: list[str] = field(default_factory=list)

    @property
    def raw_chars(self) -> int:
        return len(self.head) + self.elided + len(self.tail)

    @property
    def text(self) -> str:
        return self.head + self.tail


def _dec(field_: str) -> str | None:
    if not field_.startswith("x") or len(field_) == 1:
        return None
    try:
        return base64.b64decode(field_[1:]).decode("utf-8", errors="replace")
    except (ValueError, TypeError):
        return None


def _parse(stdout: str, nonce: str) -> tuple[str, str, int, tuple[str | None, str | None, str | None] | None]:
    meta = None
    tag = f"\n__SB_META_{nonce} "
    i = stdout.rfind(tag)
    if i >= 0:
        body = stdout[:i]
        fields = stdout[i + len(tag) :].split("\n", 1)[0].split(" ")
        fields += [""] * (3 - len(fields))
        meta = (fields[0], _dec(fields[1]), _dec(fields[2]))
    else:
        body = stdout
    tail, elided = "", 0
    etag = f"\n__SB_ELIDED_{nonce} "
    j = body.find(etag)
    if j >= 0:
        rest = body[j + len(etag) :]
        count, _, tail = rest.partition("\n")
        body = body[:j]
        try:
            elided = int(count)
        except ValueError:
            elided = 0
    return body, tail, elided, meta


def run_in_agent_shell(
    ctx: ToolContext,
    meter: _Meter,
    script: str,
    timeout_sec: int,
    *,
    persist: bool,
    cap_bytes: int = 0,
) -> ShellRun:
    """Run ``script`` in ``ctx.agent_id``'s shell (see the module docstring).

    ``persist``: restore exported env and save cwd/env afterwards (bash).
    ``cap_bytes``: if > 0, outputs larger than this keep only their first and
    last ``cap_bytes/2`` bytes (in the sandbox, before transfer).
    """
    env = ctx.env
    state = agent_state_dir(env, ctx.agent_id)
    nonce = uuid.uuid4().hex[:12]
    use_timeout = workspace.has_timeout_binary(env, runner=lambda c, t: meter.exec(env, c, t))
    wrapper = _WRAPPER.format(
        S=shlex.quote(state),
        INIT=shlex.quote(_initial_cwd(ctx)),
        N=nonce,
        RESTORE_ENV=_RESTORE_ENV if persist else "",
        SAVE_TRAP=_SAVE_TRAP if persist else "",
    )
    xfer = f"{workspace.sb_root(env)}/xfer"
    pre_c, stage_cmd = plan_b64_upload(script.encode("utf-8", errors="replace"), f"{state}/cmd_{nonce}.sh", xfer)
    pre_w, stage_wrap = plan_b64_upload(wrapper.encode("utf-8"), f"{state}/wrap_{nonce}.sh", xfer)
    for cmd in pre_c + pre_w:
        res = meter.exec(env, cmd, timeout_sec + HOST_TIMEOUT_GRACE_SEC)
        if res.return_code != 0:
            return ShellRun(head="", error=f"failed to upload the command: {res.output}")
    runner = f"timeout -k {KILL_AFTER_SEC} {timeout_sec} bash" if use_timeout else "bash"
    init = _initial_cwd(ctx)
    outer = _outer_script(state, nonce, init, stage_cmd, stage_wrap, runner, cap_bytes, persist)

    t0 = time.monotonic()
    res = meter.exec(env, outer, timeout_sec + HOST_TIMEOUT_GRACE_SEC)
    elapsed = time.monotonic() - t0
    head, tail, elided, meta = _parse(res.stdout, nonce)

    if meta is None and res.timed_out:
        # Host-side timeout: fetch what the command printed and kill it.
        run = ShellRun(head=head, tail=tail, elided=elided, timed_out=True, host_timed_out=True)
        try:
            fres = meter.exec(env, _followup_script(state, nonce, init, cap_bytes, persist), FOLLOWUP_TIMEOUT_SEC)
            fh, ft, fe, fmeta = _parse(fres.stdout, nonce)
            if fmeta is not None:
                run.head, run.tail, run.elided = fh, ft, fe
                run.cwd_before, run.cwd_after = fmeta[1], fmeta[2]
        except AgentStopped:
            raise
        except Exception as e:  # noqa: BLE001 - best effort
            run.notes.append(f"follow-up after timeout failed: {type(e).__name__}: {e}")
        return run

    if meta is None:
        detail = (head + tail).strip() or res.stderr.strip() or f"exit status {res.return_code}"
        return ShellRun(head=head, tail=tail, elided=elided, exit_code=res.return_code, error=detail)

    rc_s, before, after = meta
    try:
        rc: int | None = int(rc_s)
    except (TypeError, ValueError):
        rc = None
    # 124/137 is a timeout only if the command actually ran ~T seconds (a
    # user's own nested `timeout`, or an OOM kill, also exit 124/137).
    timed_out = bool(use_timeout and rc in TIMED_OUT_EXIT_CODES and elapsed >= 0.9 * timeout_sec)
    return ShellRun(
        head=head,
        tail=tail,
        elided=elided,
        exit_code=rc,
        cwd_before=before,
        cwd_after=after,
        timed_out=timed_out or res.timed_out,
        host_timed_out=res.timed_out,
    )


def _resolve_timeout(ctx: ToolContext, requested: int | None) -> tuple[int, str | None]:
    max_t = max(int(ctx.max_command_timeout_sec), 1)
    if requested is None:
        return min(max(int(ctx.command_timeout_sec), 1), max_t), None
    if requested < 1:
        return 1, "[timeout_sec raised to 1s]"
    if requested > max_t:
        return max_t, f"[timeout_sec capped at {max_t}s]"
    return requested, None


def _server_cap(ctx: ToolContext) -> int:
    # Bytes kept in the sandbox for huge outputs: comfortably more than what
    # the model will see, so the final head/tail cut happens on the host.
    return max(65_536, 8 * int(ctx.max_output_chars))


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


@_tool
def bash(ctx: ToolContext, args: dict, meter: _Meter) -> ToolOutput:
    command = _arg_str(args, "command", required=True)
    if not command.strip():
        raise ToolArgError("argument 'command' is empty")
    timeout, tnote = _resolve_timeout(ctx, _arg_int(args, "timeout_sec"))
    run = run_in_agent_shell(ctx, meter, command, timeout, persist=True, cap_bytes=_server_cap(ctx))
    if run.error is not None:
        return ToolOutput(
            text=f"[error] could not run the command: {run.error}",
            is_error=True,
            error=run.error,
            exit_code=run.exit_code,
            timeout_sec=timeout,
        )
    body, truncated = _truncate_parts(run.head, run.tail, run.elided, ctx.max_output_chars)
    body = body.rstrip("\n")
    notes = [tnote] if tnote else []
    if run.timed_out:
        notes.append(f"[command timed out after {timeout}s; it was killed]")
    elif run.exit_code not in (0, None):
        notes.append(f"[exit code: {run.exit_code}]")
    text = body if body else "(no output)"
    if notes:
        text += "\n" + "\n".join(notes)
    return ToolOutput(
        text=text,
        exit_code=run.exit_code,
        timed_out=run.timed_out,
        timeout_sec=timeout,
        output_chars_raw=run.raw_chars,
        truncated=truncated,
        cwd_before=run.cwd_before,
        cwd_after=run.cwd_after,
        error="; ".join(run.notes) or None,
    )


def _shell_tool_output(run: ShellRun, timeout: int, text: str, *, is_error: bool, error: str | None, raw: int, truncated: bool) -> ToolOutput:
    return ToolOutput(
        text=text,
        is_error=is_error,
        exit_code=run.exit_code,
        timed_out=run.timed_out,
        timeout_sec=timeout,
        output_chars_raw=raw,
        truncated=truncated,
        cwd_before=run.cwd_before,
        cwd_after=run.cwd_after,
        error=error,
    )


def _fail(run: ShellRun, timeout: int, msg: str) -> ToolOutput:
    return _shell_tool_output(run, timeout, f"[error] {msg}", is_error=True, error=msg, raw=run.raw_chars, truncated=False)


def _run_readonly(ctx: ToolContext, meter: _Meter, script: str) -> tuple[ShellRun, int]:
    timeout = min(max(int(ctx.command_timeout_sec), 1), max(int(ctx.max_command_timeout_sec), 1))
    run = run_in_agent_shell(ctx, meter, script, timeout, persist=False)
    return run, timeout


def _status_error(run: ShellRun, timeout: int, what: str) -> ToolOutput | None:
    if run.error is not None:
        return _fail(run, timeout, f"could not run {what}: {run.error}")
    if run.timed_out:
        return _fail(run, timeout, f"{what} timed out after {timeout}s")
    return None


@_tool
def read_file(ctx: ToolContext, args: dict, meter: _Meter) -> ToolOutput:
    path = _arg_str(args, "path", required=True)
    if not path:
        raise ToolArgError("argument 'path' is empty")
    start = _arg_int(args, "start_line", default=1)
    end = _arg_int(args, "end_line")
    if start < 1:
        raise ToolArgError("start_line must be >= 1")
    if end is not None and end < start:
        raise ToolArgError(f"end_line ({end}) is before start_line ({start})")
    n_take = READ_MAX_LINES if end is None else min(end - start + 1, READ_MAX_LINES)
    byte_cap = max(4 * int(ctx.max_output_chars), 65_536)
    q = shlex.quote(path)
    script = f"""\
__p={q}
case "$__p" in /*) __a=$__p ;; *) __a=$PWD/$__p ;; esac
if [ -d "$__p" ]; then echo "__SB_ISDIR $__a"; exit 0; fi
if [ ! -e "$__p" ]; then echo "__SB_MISSING $__a"; exit 0; fi
if [ ! -r "$__p" ]; then echo "__SB_UNREADABLE $__a"; exit 0; fi
__sz=$(wc -c < "$__p"); __n=$(wc -l < "$__p")
__lastnl=1; if [ $((__sz + 0)) -gt 0 ] && [ -n "$(tail -c 1 "$__p")" ]; then __lastnl=0; fi
__all=$(head -c 8192 "$__p" | wc -c); __nonul=$(head -c 8192 "$__p" | tr -d '\\000' | wc -c)
echo "__SB_RF $((__sz + 0)) $((__n + 0)) $__lastnl $((__all - __nonul)) $__a"
if [ $((__all - __nonul)) -gt 0 ]; then exit 0; fi
tail -n +{start} "$__p" | head -n {n_take} | head -c {byte_cap}
"""
    run, timeout = _run_readonly(ctx, meter, script)
    err = _status_error(run, timeout, "read_file")
    if err:
        return err
    out = run.text
    first, _, content = out.partition("\n")
    kind, _, rest = first.partition(" ")
    if kind == "__SB_ISDIR":
        return _fail(run, timeout, f"{rest} is a directory (use list_dir)")
    if kind == "__SB_MISSING":
        return _fail(run, timeout, f"file not found: {rest}")
    if kind == "__SB_UNREADABLE":
        return _fail(run, timeout, f"permission denied: {rest}")
    if kind != "__SB_RF":
        return _fail(run, timeout, f"unexpected output from read_file: {out[:200]!r}")
    size_s, nl_s, lastnl_s, nul_s, abspath = rest.split(" ", 4)
    size, total = int(size_s), int(nl_s) + (1 if lastnl_s == "0" else 0)
    if int(nul_s) > 0:
        text = f"[{abspath} looks binary ({size} bytes); inspect it with bash, e.g. `xxd {shlex.quote(path)} | head`]"
        return _shell_tool_output(run, timeout, text, is_error=False, error=None, raw=size, truncated=False)
    if total == 0:
        return _shell_tool_output(run, timeout, f"({abspath} is empty)", is_error=False, error=None, raw=0, truncated=False)
    if start > total:
        text = f"[{abspath} has {total} lines; start_line {start} is past the end]"
        return _shell_tool_output(run, timeout, text, is_error=False, error=None, raw=0, truncated=False)

    lines = content.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    elif len(lines) > 1 and start + len(lines) - 1 < total:
        lines.pop()  # cut mid-line by the byte cap
    rendered: list[str] = []
    used = 0
    budget = int(ctx.max_output_chars)
    char_capped = False
    for i, line in enumerate(lines):
        if len(line) > READ_MAX_LINE_CHARS:
            line = line[:READ_MAX_LINE_CHARS] + f" [... line truncated, {len(line)} chars total]"
        entry = f"{start + i:6d}\t{line}"
        if rendered and used + len(entry) + 1 > budget:
            char_capped = True
            break
        rendered.append(entry)
        used += len(entry) + 1
    last = start + len(rendered) - 1
    want_end = total if end is None else min(end, total)
    text = "\n".join(rendered)
    truncated = last < want_end
    if truncated:
        why = "line limit" if not char_capped and len(lines) == n_take else "output size limit"
        text += f"\n[showing lines {start}-{last} of {total} ({why}); call read_file with start_line={last + 1} to continue]"
    elif start > 1 or last < total:
        text += f"\n[lines {start}-{last} of {total}]"
    return _shell_tool_output(run, timeout, text, is_error=False, error=None, raw=len(content), truncated=truncated)


@_tool
def list_dir(ctx: ToolContext, args: dict, meter: _Meter) -> ToolOutput:
    path = _arg_str(args, "path", default=".") or "."
    depth = _arg_int(args, "depth", default=2)
    depth = min(max(depth, 1), LIST_MAX_DEPTH)
    names = " -o ".join(f"-name {shlex.quote(n)}" for n in _SKIP_DIRS)
    split = f"__SB_SPLIT_{uuid.uuid4().hex[:8]}"
    script = f"""\
__p={shlex.quote(path)}
if [ ! -e "$__p" ]; then echo "__SB_MISSING $__p"; exit 0; fi
if [ ! -d "$__p" ]; then echo "__SB_NOTDIR $__p"; exit 0; fi
cd -- "$__p" || exit 4
echo "__SB_LD $PWD"
__t="$__sb_S/ls_$$"
{{ find . -mindepth 1 -maxdepth {depth} \\( {names} \\) -prune -type d -print -o -type d -print 2>/dev/null
  echo {split}
  find . -mindepth 1 -maxdepth {depth} \\( {names} \\) -prune -o ! -type d -print 2>/dev/null; }} > "$__t"
echo "__SB_COUNT $(wc -l < "$__t")"
head -n {LIST_MAX_ENTRIES * 6} "$__t"
rm -f "$__t"
"""
    run, timeout = _run_readonly(ctx, meter, script)
    err = _status_error(run, timeout, "list_dir")
    if err:
        return err
    out = run.text
    lines = out.split("\n")
    kind, _, rest = lines[0].partition(" ")
    if kind == "__SB_MISSING":
        return _fail(run, timeout, f"no such directory: {rest}")
    if kind == "__SB_NOTDIR":
        return _fail(run, timeout, f"{rest} is not a directory (use read_file)")
    if kind != "__SB_LD" or len(lines) < 2 or not lines[1].startswith("__SB_COUNT"):
        return _fail(run, timeout, f"unexpected output from list_dir: {out[:200]!r}")
    abspath = rest
    total = int(lines[1].split()[1]) - 1  # minus the split line
    entries: list[str] = []
    is_dir = True
    for line in lines[2:]:
        if line == split:
            is_dir = False
            continue
        if not line.startswith("./"):
            continue
        entries.append(line[2:] + ("/" if is_dir else ""))
    entries.sort()
    shown = entries[:LIST_MAX_ENTRIES]
    body = "\n".join(shown)
    text, truncated = truncate_output(body, ctx.max_output_chars)
    header = f"{abspath} (depth {depth}; {total} entries"
    header += f", showing {len(shown)})" if total > len(shown) else ")"
    if not entries:
        text = "(empty directory)"
    if total > len(shown):
        truncated = True
        text += f"\n[... {total - len(shown)} more entries not shown; list a subdirectory or use a smaller depth]"
    skipped = ", ".join(_SKIP_DIRS)
    text = f"{header}\n{text}\n[contents of {skipped} are not expanded]" if entries else f"{header}\n{text}"
    return _shell_tool_output(run, timeout, text, is_error=False, error=None, raw=len(body), truncated=truncated)


@_tool
def search(ctx: ToolContext, args: dict, meter: _Meter) -> ToolOutput:
    pattern = _arg_str(args, "pattern", required=True)
    if pattern == "":
        raise ToolArgError("argument 'pattern' is empty")
    path = _arg_str(args, "path", default=".") or "."
    glob = _arg_str(args, "glob")
    rg_glob = f"-g {shlex.quote(glob)} " if glob else ""
    grep_glob = f"--include={shlex.quote(glob)} " if glob else ""
    excl = " ".join(f"--exclude-dir={shlex.quote(d)}" for d in _SKIP_DIRS)
    script = f"""\
__pat={shlex.quote(pattern)}
__p={shlex.quote(path)}
if [ ! -e "$__p" ]; then echo "__SB_MISSING $__p"; exit 0; fi
__t="$__sb_S/search_$$"
if command -v rg >/dev/null 2>&1; then
  __eng=rg; rg -n -H --no-heading --color never {rg_glob}-e "$__pat" -- "$__p" > "$__t" 2> "$__t.err"
else
  __eng=grep; grep -rnHI -E {excl} {grep_glob}-e "$__pat" -- "$__p" > "$__t" 2> "$__t.err"
fi
__rc=$?
echo "__SB_SEARCH $__eng $__rc $(wc -l < "$__t")"
head -n {SEARCH_MAX_LINES} "$__t"
echo "__SB_ERR"
head -c 2000 "$__t.err"
rm -f "$__t" "$__t.err"
"""
    run, timeout = _run_readonly(ctx, meter, script)
    err = _status_error(run, timeout, "search")
    if err:
        return err
    out = run.text
    first, _, rest = out.partition("\n")
    kind, _, info = first.partition(" ")
    if kind == "__SB_MISSING":
        return _fail(run, timeout, f"no such file or directory: {info}")
    if kind != "__SB_SEARCH":
        return _fail(run, timeout, f"unexpected output from search: {out[:200]!r}")
    engine, rc_s, n_s = info.split()
    rc, n_total = int(rc_s), int(n_s)
    run.exit_code = rc
    matches_part, _, err_part = rest.rpartition("__SB_ERR\n")
    err_text = err_part.strip()
    matches = [m for m in matches_part.split("\n") if m]
    if not matches:
        if rc >= 2:
            return _fail(run, timeout, f"{engine} failed (exit {rc}): {err_text or 'no details'}")
        return _shell_tool_output(run, timeout, "No matches found.", is_error=False, error=None, raw=0, truncated=False)
    clipped = [m if len(m) <= SEARCH_MAX_LINE_CHARS else m[:SEARCH_MAX_LINE_CHARS] + " [...]" for m in matches]
    body = "\n".join(clipped)
    text, truncated = truncate_output(body, ctx.max_output_chars)
    if n_total > len(matches):
        truncated = True
        text += f"\n[showing the first {len(matches)} of {n_total} matching lines; narrow the pattern, path or glob]"
    if rc >= 2 and err_text:
        text += f"\n[{engine} reported errors: {err_text[:300]}]"
    return _shell_tool_output(run, timeout, text, is_error=False, error=None, raw=len(matches_part), truncated=truncated)


def _line_of(data: bytes, offset: int) -> int:
    return data.count(b"\n", 0, offset) + 1


@_tool
def edit_file(ctx: ToolContext, args: dict, meter: _Meter) -> ToolOutput:
    path = _arg_str(args, "path", required=True)
    if not path:
        raise ToolArgError("argument 'path' is empty")
    old = _arg_str(args, "old", required=True)
    new = _arg_str(args, "new", required=True)
    create = old == ""
    q = shlex.quote(path)
    # Resolve against the agent's cwd and (unless creating) fetch the bytes.
    script = f"""\
__p={q}
case "$__p" in /*) __a=$__p ;; *) __a=$PWD/$__p ;; esac
printf '__SB_EF %s\\n' "$(printf '%s' "$__a" | base64 | tr -d '\\n')"
if [ -d "$__a" ]; then echo __SB_ISDIR; exit 0; fi
if [ ! -e "$__a" ]; then echo __SB_MISSING; exit 0; fi
__sz=$(wc -c < "$__a"); echo "__SB_SIZE $((__sz + 0))"
{"exit 0" if create else f'if [ $((__sz + 0)) -gt {EDIT_MAX_BYTES} ]; then exit 0; fi'}
base64 < "$__a" | tr -d '\\n'; echo
"""
    run, timeout = _run_readonly(ctx, meter, script)
    err = _status_error(run, timeout, "edit_file")
    if err:
        return err
    lines = run.text.split("\n")
    if not lines[0].startswith("__SB_EF "):
        return _fail(run, timeout, f"unexpected output from edit_file: {run.text[:200]!r}")
    abspath = posixpath.normpath(base64.b64decode(lines[0][8:]).decode("utf-8", errors="replace"))
    status = lines[1] if len(lines) > 1 else ""
    out_kw = dict(timeout_sec=timeout, cwd_before=run.cwd_before, cwd_after=run.cwd_after)

    def error(msg: str) -> ToolOutput:
        return ToolOutput(text=f"[error] {msg}", is_error=True, error=msg, **out_kw)

    if status == "__SB_ISDIR":
        return error(f"{abspath} is a directory")
    new_b = new.encode("utf-8", errors="replace")
    if create:
        existed = status.startswith("__SB_SIZE")
        _write(ctx, meter, abspath, new_b)
        n_lines = len(new_b.splitlines())
        if existed:
            old_size = int(status.split()[1])
            msg = f"Overwrote existing file {abspath} ({old_size} -> {len(new_b)} bytes, {n_lines} lines)."
        else:
            msg = f"Created {abspath} ({len(new_b)} bytes, {n_lines} lines)."
        return ToolOutput(text=msg, files_written=[abspath], **out_kw)
    if status == "__SB_MISSING":
        return error(f"file not found: {abspath} (to create it, pass an empty 'old')")
    size = int(status.split()[1])
    if size > EDIT_MAX_BYTES:
        return error(f"{abspath} is {size} bytes, too large for edit_file (limit {EDIT_MAX_BYTES}); use bash")
    data = base64.b64decode(lines[2] if len(lines) > 2 else "")
    if len(data) != size:
        return error(f"could not read {abspath} completely ({len(data)} of {size} bytes)")
    old_b = old.encode("utf-8", errors="replace")
    count = data.count(old_b)
    if count == 0:
        hint = ""
        if b"\r\n" in data and b"\n" in old_b and b"\r\n" not in old_b:
            hint = " The file uses CRLF line endings; include the \\r characters in 'old'."
        return error(f"'old' text not found in {abspath}; it must match exactly, including whitespace.{hint}")
    if count > 1:
        offsets, pos = [], data.find(old_b)
        while pos >= 0 and len(offsets) < 10:
            offsets.append(_line_of(data, pos))
            pos = data.find(old_b, pos + 1)
        where = ", ".join(map(str, offsets))
        return error(f"'old' text occurs {count} times in {abspath} (at lines {where}); include more context to make it unique")
    idx = data.find(old_b)
    updated = data[:idx] + new_b + data[idx + len(old_b) :]
    _write(ctx, meter, abspath, updated)
    first = _line_of(data, idx)
    old_n = max(len(old_b.splitlines()), 1)
    new_n = len(new_b.splitlines())
    msg = f"Edited {abspath}: replaced {old_n} line(s) at line {first} with {new_n} line(s)."
    return ToolOutput(text=msg, files_written=[abspath], **out_kw)


def _write(ctx: ToolContext, meter: _Meter, abspath: str, data: bytes) -> None:
    """Byte-exact write in place (keeps mode/owner), parent dirs created."""
    env = ctx.env
    timeout = max(int(ctx.command_timeout_sec), 1) + HOST_TIMEOUT_GRACE_SEC
    pre, final = plan_b64_upload(data, abspath, f"{workspace.sb_root(env)}/xfer", mkdirs=True)
    for cmd in pre + [final]:
        res = meter.exec(env, cmd, timeout)
        if res.timed_out:
            raise TimeoutError(f"writing {abspath} timed out")
        if res.return_code != 0:
            raise OSError(f"failed to write {abspath}: {res.output}")


# ---------------------------------------------------------------------------
# specs and registry
# ---------------------------------------------------------------------------

_BASH_DESC = (
    "Run a bash command in the task environment and return its output (stdout and stderr interleaved) "
    "and exit code. Your working directory and exported environment variables persist between your "
    "bash calls; shell functions, aliases, shell options and unexported variables do not. Commands run "
    "non-interactively with no TTY and stdin closed, so interactive programs (editors, pagers, prompts) "
    "will not work: use non-interactive flags such as -y. Each command has a timeout{timeout}; on "
    "timeout the command and its child processes are killed. Start long-running processes such as "
    "servers in the background with `nohup CMD > log 2>&1 &` and check on them in later calls. Long "
    "output is truncated in the middle."
)


def _bash_spec(command_timeout_sec: int | None = None, max_command_timeout_sec: int | None = None) -> ToolSpec:
    if command_timeout_sec and max_command_timeout_sec:
        t = f" (default {command_timeout_sec}s; set timeout_sec to change it, up to {max_command_timeout_sec}s)"
    else:
        t = " (a harness default; set timeout_sec to change it, up to a harness maximum)"
    return ToolSpec(
        name="bash",
        description=_BASH_DESC.format(timeout=t),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command (may span several lines)."},
                "timeout_sec": {"type": "integer", "description": "Optional timeout in seconds for this command."},
            },
            "required": ["command"],
        },
    )


TOOL_SPECS: dict[str, ToolSpec] = {
    "bash": _bash_spec(),
    "read_file": ToolSpec(
        name="read_file",
        description=(
            "Read a text file, returned with line numbers. Relative paths are resolved against your "
            f"current working directory. Use start_line/end_line (1-based, inclusive) to read part of a "
            f"file; at most {READ_MAX_LINES} lines are returned per call."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File to read."},
                "start_line": {"type": "integer", "description": "First line to show (default 1)."},
                "end_line": {"type": "integer", "description": "Last line to show (default: end of file)."},
            },
            "required": ["path"],
        },
    ),
    "list_dir": ToolSpec(
        name="list_dir",
        description=(
            f"List files and directories under a path, up to `depth` levels deep (default 2, max "
            f"{LIST_MAX_DEPTH}). Directories end with '/'. Relative paths are resolved against your current "
            "working directory. The contents of .git, node_modules and __pycache__ are not expanded."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to list (default: current directory)."},
                "depth": {"type": "integer", "description": "How many levels to descend (default 2)."},
            },
            "required": [],
        },
    ),
    "search": ToolSpec(
        name="search",
        description=(
            "Search file contents for a regular expression and return matching lines as "
            f"path:line:text (at most {SEARCH_MAX_LINES} lines). Uses ripgrep if it is installed (it skips "
            "hidden files and files ignored by .gitignore), otherwise grep -rE (extended regex; skips "
            ".git, node_modules and __pycache__). `glob` restricts the file names searched, e.g. '*.py'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression to search for."},
                "path": {"type": "string", "description": "File or directory to search (default: current directory)."},
                "glob": {"type": "string", "description": "Optional file-name glob, e.g. '*.py'."},
            },
            "required": ["pattern"],
        },
    ),
    "edit_file": ToolSpec(
        name="edit_file",
        description=(
            "Edit a file by replacing an exact piece of text. `old` must match the file exactly (including "
            "whitespace and indentation) and occur exactly once; include enough surrounding lines to make it "
            "unique. To create a file, or replace a file's entire content, pass an empty `old` and the full "
            "content as `new`. Relative paths are resolved against your current working directory."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File to edit or create."},
                "old": {"type": "string", "description": "Exact text to replace; empty string to create/overwrite the file."},
                "new": {"type": "string", "description": "Replacement text (or the full file content when old is empty)."},
            },
            "required": ["path", "old", "new"],
        },
    ),
}

TOOL_FNS: dict[str, Callable[[ToolContext, dict], ToolOutput]] = {
    "bash": bash,
    "read_file": read_file,
    "list_dir": list_dir,
    "search": search,
    "edit_file": edit_file,
}

# Without finish/spawn_subagents (the orchestrator adds those itself).
ORCHESTRATOR_TOOLS: list[str] = ["bash", "edit_file", "read_file", "list_dir", "search"]

# Subagent tool sets by delegation scope, without submit.
SCOPE_TOOLS: dict[str, list[str]] = {
    "easy": ["read_file", "list_dir", "search"],
    "medium": ["read_file", "list_dir", "search", "bash"],
    "hard": ["read_file", "list_dir", "search", "bash", "edit_file"],
}


def specs_for(
    names: list[str],
    *,
    command_timeout_sec: int | None = None,
    max_command_timeout_sec: int | None = None,
) -> list[ToolSpec]:
    """Specs for ``names``, in order; with timeouts given, the bash
    description states the actual default and maximum."""
    out = []
    for n in names:
        if n == "bash" and command_timeout_sec and max_command_timeout_sec:
            out.append(_bash_spec(command_timeout_sec, max_command_timeout_sec))
        else:
            out.append(TOOL_SPECS[n])
    return out


def run_tool(ctx: ToolContext, name: str, args: dict) -> ToolOutput:
    """Run tool ``name``. Never raises except AgentStopped."""
    fn = TOOL_FNS.get(name)
    if fn is None:
        msg = f"unknown tool {name!r}; available: {', '.join(TOOL_FNS)}"
        return ToolOutput(text=f"[error] {msg}", is_error=True, error=msg)
    try:
        return fn(ctx, args)
    except AgentStopped:
        raise
    except Exception as e:  # noqa: BLE001 - last resort; the tool wrappers catch first
        msg = f"{type(e).__name__}: {e}"
        return ToolOutput(text=f"[error] {name} failed: {msg}", is_error=True, error=msg)
