"""Agent tools (bash, edit_file, read_file, list_dir, search), run through each agent's shell.

Each agent has a state dir ``<state_root>/<agent_id>/`` with its persisted cwd and exports.
A tool call is one exec of a script that stages the command and a wrapper, runs the wrapper
as its own process group (``set -m``) next to a watchdog that kills the group at the timeout,
and prints the output (head and tail if long) and the exit status. Its own stderr is dropped:
harbor merges stderr into stdout, and job-control notices ("[1]- Done ...") would land there. While a command runs, its
pid (= process group) is in ``pid_<n>``: harbor's cancel kills what is left running from it.
"""

from __future__ import annotations

import base64
import posixpath
import re
import shlex
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from .config import BudgetConfig
from .environment import AgentStopped
from .llm import ToolSpec

SCOPE_TOOLS = {
    "easy": ["read_file", "list_dir", "search"],
    "medium": ["read_file", "list_dir", "search", "bash"],
    "hard": ["read_file", "list_dir", "search", "bash", "edit_file"],
}
KILL_AFTER_SEC = 5  # SIGKILL this long after the SIGTERM
HOST_GRACE_SEC = 20  # the exec's own timeout is the command's plus this
LIST_MAX_ENTRIES = 500
SEARCH_MAX_LINES = 200


@dataclass
class ToolContext:
    env: Any
    agent_id: str
    cwd: str  # where the agent's shell starts
    max_output_chars: int = 12000
    command_timeout_sec: int = 180
    max_command_timeout_sec: int = 900
    remap: tuple[str, str] | None = None  # (workdir, private copy) for arbiter candidates


@dataclass
class ToolOutput:
    text: str
    is_error: bool = False
    exit_code: int | None = None
    timed_out: bool = False
    output_chars: int = 0  # before truncation


def truncate(text: str, limit: int) -> str:
    """Head and tail of ``text`` within about ``limit`` chars."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n[... {len(text) - 2 * half} characters elided ...]\n{text[-half:]}"


_WRAPPER = """S={state}
echo "$$" > "$S/pid_{n}"
{restore}
__sb_c={init}; [ -f "$S/cwd" ] && read -r __sb_c < "$S/cwd"
cd -- "$__sb_c" 2>/dev/null || cd -- {init} 2>/dev/null || cd /
{save}
. "$S/cmd_{n}.sh"
"""
_RESTORE = '[ -f "$S/env.sh" ] && . "$S/env.sh" > /dev/null 2>&1'
# Builtins only, so a command that breaks PATH cannot break saving.
_SAVE = "trap '__sb_rc=$?; { set +eux; pwd > \"$S/cwd\"; export -p > \"$S/env.sh\"; } 2>/dev/null; exit $__sb_rc' EXIT"
_OUTER = """S={state}
mkdir -p "$S" && printf '%s' {cmd} | base64 -d > "$S/cmd_{n}.sh" || exit 97
cat > "$S/wrap_{n}.sh" <<'__SB_WRAP_{n}'
{wrapper}__SB_WRAP_{n}
exec 2>/dev/null
set -m
bash "$S/wrap_{n}.sh" > "$S/out_{n}" 2>&1 < /dev/null &
__sb_p=$!
( sleep {timeout}; touch "$S/to_{n}"; kill -TERM -- -$__sb_p; sleep {kill_after}; kill -KILL -- -$__sb_p ) > /dev/null 2>&1 &
__sb_w=$!
wait $__sb_p
__sb_rc=$?
kill -- -$__sb_w 2>/dev/null
__sb_sz=$(( $(wc -c < "$S/out_{n}") ))
if [ $__sb_sz -gt {cap} ]; then head -c {half} "$S/out_{n}"; printf '\\n__SB_CUT_{n}\\n'; tail -c {half} "$S/out_{n}"
else cat "$S/out_{n}"; fi
__sb_to=0; [ -f "$S/to_{n}" ] && __sb_to=1
rm -f "$S/out_{n}" "$S/pid_{n}" "$S/to_{n}" "$S/cmd_{n}.sh" "$S/wrap_{n}.sh"
printf '\\n__SB_END_{n} %s %s %s\\n' "$__sb_rc" "$__sb_to" "$__sb_sz"
"""


def _shell(ctx: ToolContext, script: str, timeout: int, *, persist: bool, cap: int = 10**15) -> tuple[str, int, bool, int]:
    """Run ``script`` in the agent's shell -> (output, exit code, timed out, output bytes).
    ``persist``: restore and save the agent's exports and cwd (bash)."""
    n = uuid.uuid4().hex[:10]
    q = shlex.quote
    state = q(f"{ctx.env.state_root}/{ctx.agent_id}")
    wrapper = _WRAPPER.format(state=state, n=n, init=q(ctx.cwd), restore=_RESTORE if persist else "",
                              save=_SAVE if persist else "")
    cmd = q(base64.b64encode(script.encode("utf-8", "surrogateescape")).decode())
    outer = _OUTER.format(state=state, n=n, cmd=cmd, wrapper=wrapper, timeout=timeout, kill_after=KILL_AFTER_SEC,
                          cap=cap, half=cap // 2)
    res = ctx.env.exec(outer, timeout + KILL_AFTER_SEC + HOST_GRACE_SEC)
    body, sep, meta = res.stdout.rpartition(f"\n__SB_END_{n} ")
    if not sep:
        why = "timed out" if res.timed_out else f"exit {res.return_code}"
        raise RuntimeError(f"the agent shell failed ({why}): {(res.stdout + res.stderr).strip()[-500:]}")
    rc, timed_out, size = meta.split()[:3]
    head, cut, tail = body.partition(f"\n__SB_CUT_{n}\n")
    text = f"{head}\n[... {int(size) - 2 * (cap // 2)} bytes elided ...]\n{tail}" if cut else head
    return text, int(rc), timed_out == "1", int(size)


def _int(value: Any, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _error(msg: str) -> ToolOutput:
    return ToolOutput(f"[error] {msg}", is_error=True)


def bash(ctx: ToolContext, args: dict) -> ToolOutput:
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        return _error("'command' must be a non-empty string")
    timeout = min(max(_int(args.get("timeout_sec"), ctx.command_timeout_sec), 1), ctx.max_command_timeout_sec)
    text, rc, timed_out, size = _shell(ctx, command, timeout, persist=True, cap=ctx.max_output_chars)
    text = text.rstrip("\n") or "(no output)"
    if timed_out:
        text += f"\n[command timed out after {timeout}s and was killed]"
    elif rc:
        text += f"\n[exit code: {rc}]"
    return ToolOutput(text, exit_code=rc, timed_out=timed_out, output_chars=size)


def _readonly(ctx: ToolContext, script: str) -> ToolOutput:
    text, rc, timed_out, size = _shell(ctx, script, ctx.command_timeout_sec, persist=False, cap=ctx.max_output_chars)
    text = text.rstrip("\n") or "(no output)"
    if rc or timed_out:
        text = "[error] " + (f"timed out after {ctx.command_timeout_sec}s" if timed_out else text)
    return ToolOutput(text, bool(rc or timed_out), rc, timed_out, size)


def read_file(ctx: ToolContext, args: dict) -> ToolOutput:
    path = args.get("path")
    if not isinstance(path, str) or not path:
        return _error("'path' must be a non-empty string")
    start, end = max(_int(args.get("start_line"), 1), 1), max(_int(args.get("end_line"), 0), 0)
    return _readonly(ctx, f"""p={shlex.quote(path)}
if [ -d "$p" ]; then echo "$p is a directory (use list_dir)"; exit 2; fi
if [ ! -f "$p" ]; then echo "no such file: $p"; exit 2; fi
awk -v s={start} -v e={end} 'NR >= s && (e == 0 || NR <= e) {{ printf "%6d\\t%s\\n", NR, $0 }}' "$p"
""")


def list_dir(ctx: ToolContext, args: dict) -> ToolOutput:
    path = args.get("path") if isinstance(args.get("path"), str) and args.get("path") else "."
    depth = min(max(_int(args.get("depth"), 2), 1), 5)
    return _readonly(ctx, f"""p={shlex.quote(path)}
if [ ! -d "$p" ]; then echo "not a directory: $p"; exit 2; fi
cd -- "$p" || exit 2
find . -mindepth 1 -maxdepth {depth} \\( -name .git -o -name node_modules -o -name __pycache__ \\) -prune -print -o -print \\
  | LC_ALL=C sort | head -n {LIST_MAX_ENTRIES} | while IFS= read -r f; do
  if [ -d "$f" ]; then printf '%s/\\n' "${{f#./}}"; else printf '%s\\n' "${{f#./}}"; fi
done
""")


def search(ctx: ToolContext, args: dict) -> ToolOutput:
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return _error("'pattern' must be a non-empty string")
    path = args.get("path") if isinstance(args.get("path"), str) and args.get("path") else "."
    return _readonly(ctx, f"""out=$(grep -rnIE --exclude-dir=.git --exclude-dir=node_modules -e {shlex.quote(pattern)} \\
  -- {shlex.quote(path)} 2>&1 | head -n {SEARCH_MAX_LINES})
if [ -n "$out" ]; then printf '%s\\n' "$out"; else echo "No matches."; fi
""")


def edit_file(ctx: ToolContext, args: dict) -> ToolOutput:
    path, old, new = args.get("path"), args.get("old", ""), args.get("new")
    if not isinstance(path, str) or not path or not isinstance(old, str) or not isinstance(new, str):
        return _error("'path' and 'new' must be strings ('old' too, if given)")
    text, _, _, _ = _shell(ctx, f"""p={shlex.quote(path)}
case "$p" in /*) ;; *) p="$PWD/$p" ;; esac
printf '%s\\n' "$p"
if [ -d "$p" ]; then echo DIR; elif [ -f "$p" ]; then echo FILE; base64 < "$p"; else echo MISSING; fi
""", ctx.command_timeout_sec, persist=False)
    abspath, kind, data = (text.split("\n", 2) + ["", ""])[:3]
    abspath = posixpath.normpath(abspath)
    if kind == "DIR":
        return _error(f"{abspath} is a directory")
    content = new
    if old:
        if kind != "FILE":
            return _error(f"no such file: {abspath} (pass an empty 'old' to create it)")
        current = base64.b64decode(data).decode("utf-8", "surrogateescape")
        count = current.count(old)
        if count != 1:
            hint = "include more surrounding lines" if count else "it must match exactly, including whitespace"
            return _error(f"'old' occurs {count} times in {abspath}; {hint}")
        content = current.replace(old, new, 1)
    b64 = base64.b64encode(content.encode("utf-8", "surrogateescape")).decode()
    q = shlex.quote(abspath)
    res = ctx.env.exec(f"mkdir -p -- \"$(dirname -- {q})\" && printf '%s' {shlex.quote(b64)} | base64 -d > {q}",
                       ctx.command_timeout_sec)
    if res.return_code != 0:
        return _error(f"could not write {abspath}: {(res.stdout + res.stderr).strip()}")
    return ToolOutput(f"{'Edited' if old else 'Wrote'} {abspath}.")


# A workdir path in a command is remapped only as a whole path prefix: not the tail of a
# longer path or name (/srv/app, $HOME/app), and followed by a separator or delimiter.
_BEFORE = r"(?<![\w.~/$@%+\-})])"
_AFTER = r"""(?=$|[/\s'"`;&|()<>:,}\]\\])"""


def remap_args(name: str, args: dict, src: str, dst: str) -> dict:
    """Point an arbiter candidate's tool arguments at its private copy ``dst`` of ``src``."""
    if name == "bash" and isinstance(args.get("command"), str):
        alts = "|".join(re.escape(p) for p in sorted({src, dst}, key=len, reverse=True))
        command = re.sub(f"{_BEFORE}({alts}){_AFTER}", lambda m: dst if m.group(1) == src else m.group(1), args["command"])
        return {**args, "command": command}
    path = args.get("path")
    if isinstance(path, str) and path.startswith("/"):
        p = posixpath.normpath(path)
        in_dst = p == dst or p.startswith(dst + "/")
        if not in_dst and (p == src or p.startswith(src + "/")):
            return {**args, "path": dst + p[len(src):]}
    return args


TOOLS: dict[str, Callable[[ToolContext, dict], ToolOutput]] = {
    "bash": bash, "edit_file": edit_file, "read_file": read_file, "list_dir": list_dir, "search": search,
}


def run_tool(ctx: ToolContext, name: str, args: Any) -> ToolOutput:
    """Run tool ``name``; any failure becomes an error output (only AgentStopped propagates)."""
    if name not in TOOLS:
        return _error(f"unknown tool {name!r}")
    if not isinstance(args, dict):
        return _error("arguments must be a JSON object")
    note = ""
    if ctx.remap:
        mapped = remap_args(name, args, *ctx.remap)
        if mapped != args:
            args, note = mapped, f"\n[note: {ctx.remap[0]} was redirected to your private copy {ctx.remap[1]}]"
    try:
        out = TOOLS[name](ctx, args)
    except AgentStopped:
        raise
    except Exception as e:  # surfaced to the model
        out = _error(f"{name} failed: {type(e).__name__}: {e}")
    out.text += note
    return out


def _schema(required: list[str], **props: tuple[str, str]) -> dict:
    return {"type": "object", "properties": {k: {"type": t, "description": d} for k, (t, d) in props.items()},
            "required": required}


def tool_specs(names: list[str], budget: BudgetConfig) -> list[ToolSpec]:
    """Specs for ``names``; the bash description states the real timeouts."""
    specs = {
        "bash": ToolSpec("bash", (
            "Run a bash command in the task container; returns its output (stdout and stderr) and exit code. "
            "Your working directory and exported variables persist between calls. No TTY, stdin is closed: "
            f"use non-interactive flags. Default timeout {budget.command_timeout_sec}s (set timeout_sec, up to "
            f"{budget.max_command_timeout_sec}s); on timeout the command and its children are killed. Start "
            "servers with `nohup CMD > log 2>&1 &`. Long output is truncated in the middle."),
            _schema(["command"], command=("string", "The command (may span lines)."),
                    timeout_sec=("integer", "Timeout in seconds for this command."))),
        "edit_file": ToolSpec("edit_file", (
            "Replace an exact snippet `old` (must occur exactly once, whitespace included) with `new`. "
            "With an empty `old`, create or overwrite the file with `new`."),
            _schema(["path", "old", "new"], path=("string", "File to edit or create."),
                    old=("string", "Exact text to replace; empty to create/overwrite."),
                    new=("string", "Replacement text, or the whole file."))),
        "read_file": ToolSpec("read_file", "Read a text file with line numbers (start_line/end_line are 1-based).",
                              _schema(["path"], path=("string", "File to read."),
                                      start_line=("integer", "First line (default 1)."),
                                      end_line=("integer", "Last line (default: end of file)."))),
        "list_dir": ToolSpec("list_dir", f"List a directory tree (directories end with '/'; at most "
                             f"{LIST_MAX_ENTRIES} entries; .git and node_modules not expanded).",
                             _schema([], path=("string", "Directory (default: current)."),
                                     depth=("integer", "Levels to descend, 1-5 (default 2)."))),
        "search": ToolSpec("search", f"grep -rE for a regular expression in file contents; returns path:line:text "
                           f"(at most {SEARCH_MAX_LINES} lines).",
                           _schema(["pattern"], pattern=("string", "Extended regular expression."),
                                   path=("string", "File or directory (default: current)."))),
    }
    return [specs[n] for n in names]
