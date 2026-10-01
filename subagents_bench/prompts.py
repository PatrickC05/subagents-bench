"""System prompts and the subtask-type prefix.

The ``subtask_type`` config knob changes only the *prefix* prepended to the
orchestrator's system prompt -- it steers *what* the orchestrator delegates
without changing its tools or budget. This is the lever the study uses to ask
"how does performance depend on the tasks the orchestrator assigns?".

The exact subtask wordings are deliberately left thin for now (the project note
says "leave this to be done later"); each is a single instruction block that
can be iterated on without touching the agent loop.
"""

from __future__ import annotations

from .config import SubtaskType

ORCHESTRATOR_BASE = """You are the orchestrator of a software-engineering session. \
You have a shell, file editing, and search tools in a sandboxed container that \
holds the repository at its base commit. Your job is to resolve the given issue \
so that the hidden test suite passes.

You operate in a small fixed number of high-level steps. Work efficiently: \
investigate, make precise edits, and verify with the repository's own tests \
before finishing. When you are confident the issue is fixed, call `finish`.
"""

ORCHESTRATOR_SUBAGENT_NOTE = """
You may delegate work to subagents with `spawn_subagents`. Each subagent is an \
independent worker with the same shell/edit/search tools in the SAME container, \
so file changes they make are visible to you and to each other. Give each a \
self-contained instruction and the context it needs; they cannot see your \
conversation. Subagents are useful for parallel investigation, localized edits, \
or verification. You remain responsible for the final patch and must call \
`finish` yourself.
"""

# Prefixes selected by SubtaskType. Kept short; refine later.
SUBTASK_PREFIXES: dict[SubtaskType, str] = {
    SubtaskType.NONE: "",
    SubtaskType.PARALLEL_EXPLORATION: (
        "Delegation policy: spawn several subagents IN PARALLEL to explore "
        "different hypotheses or regions of the codebase, then synthesise their "
        "findings yourself before editing.\n\n"
    ),
    SubtaskType.LOCALIZE_THEN_FIX: (
        "Delegation policy: first dispatch a subagent to localise the root cause "
        "(files, functions, failing assertions); once localised, dispatch a "
        "subagent to implement the fix in that region.\n\n"
    ),
    SubtaskType.VERIFY: (
        "Delegation policy: after you make a change, dispatch a subagent to "
        "independently verify it by running the relevant tests and reporting back, "
        "before you call finish.\n\n"
    ),
    SubtaskType.DECOMPOSE: (
        "Delegation policy: decompose the issue into independent subtasks and "
        "assign one subagent per subtask, then integrate their patches.\n\n"
    ),
}

SUBAGENT_BASE = """You are a subagent working inside a shared repository sandbox. \
You have shell, file editing, and search tools. You were dispatched by an \
orchestrator with a specific instruction below. Do exactly that task, make any \
file edits directly in the container, and when done call `submit` with a concise \
report of what you found or changed (include file paths and key results). Be \
economical with steps.
"""


def orchestrator_system(subtask: SubtaskType, with_subagents: bool) -> str:
    prefix = SUBTASK_PREFIXES.get(subtask, "")
    body = ORCHESTRATOR_BASE
    if with_subagents:
        body = body + ORCHESTRATOR_SUBAGENT_NOTE
    return prefix + body


def subagent_system() -> str:
    return SUBAGENT_BASE
