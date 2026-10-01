"""Tool definitions shared by the orchestrator and subagents.

Both agent tiers get the same base kit -- ``bash``, ``edit_file``,
``search`` -- so a subagent is a full-fledged worker, not a crippled helper.
The orchestrator additionally gets ``spawn_subagents`` and ``finish`` (wired in
:mod:`subagents_bench.orchestrator`); subagents get ``submit`` to return a
result.

Each tool is a pure function of (Environment, args) -> str, which keeps them
trivially testable against :class:`LocalShellEnvironment`.
"""

from __future__ import annotations

import shlex
from typing import Any

from .environment import Environment
from .llm import ToolSpec

MAX_TOOL_OUTPUT = 12_000


def _truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n... [truncated {len(text) - limit} chars] ...\n{tail}"


BASH_SPEC = ToolSpec(
    name="bash",
    description=(
        "Run a bash command in the task sandbox. State (cwd, files) persists "
        "across calls within an agent. Returns combined stdout/stderr and exit code."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The bash command to run."},
            "timeout_sec": {"type": "integer", "description": "Optional per-command timeout."},
        },
        "required": ["command"],
    },
)

EDIT_SPEC = ToolSpec(
    name="edit_file",
    description=(
        "Replace an exact substring in a file with new text (or create the file "
        "when 'old' is empty). 'old' must match exactly once."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old": {"type": "string", "description": "Exact text to replace; empty to create/overwrite."},
            "new": {"type": "string"},
        },
        "required": ["path", "new"],
    },
)

SEARCH_SPEC = ToolSpec(
    name="search",
    description="Search the repository for a regex pattern (ripgrep if available, else grep -r).",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "Directory or file to search; defaults to cwd."},
        },
        "required": ["pattern"],
    },
)


def run_bash(env: Environment, args: dict[str, Any]) -> str:
    cmd = args["command"]
    res = env.exec(cmd, timeout_sec=args.get("timeout_sec"))
    prefix = "" if res.return_code == 0 else f"[exit {res.return_code}]\n"
    return _truncate(prefix + res.output)


def run_edit(env: Environment, args: dict[str, Any]) -> str:
    path = args["path"]
    new = args["new"]
    old = args.get("old", "")
    if not old:
        env.write_file(path, new)
        return f"Wrote {len(new)} chars to {path}."
    try:
        current = env.read_file(path)
    except FileNotFoundError:
        return f"[error] file not found: {path}"
    count = current.count(old)
    if count == 0:
        return f"[error] 'old' text not found in {path}."
    if count > 1:
        return f"[error] 'old' text matches {count} times in {path}; make it unique."
    env.write_file(path, current.replace(old, new, 1))
    return f"Edited {path} (1 replacement)."


def run_search(env: Environment, args: dict[str, Any]) -> str:
    pattern = args["pattern"]
    path = args.get("path", ".")
    q = shlex.quote(pattern)
    p = shlex.quote(path)
    cmd = f"(rg -n --no-heading {q} {p} 2>/dev/null || grep -rn {q} {p}) | head -n 200"
    res = env.exec(cmd)
    return _truncate(res.output)


BASE_TOOLS: dict[str, Any] = {
    "bash": run_bash,
    "edit_file": run_edit,
    "search": run_search,
}

BASE_SPECS: list[ToolSpec] = [BASH_SPEC, EDIT_SPEC, SEARCH_SPEC]
