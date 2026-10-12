"""System prompts: orchestrator, subagents (per scope), arbiter judge, debate.

Every prompt describes the runtime honestly -- the step budget, what each tool
really does (non-interactive shell, persisted cwd/exports, timeouts, nohup for
servers), what subagents can and cannot do under the configured strategy and
scope, and the per-call and per-episode dispatch caps -- because a model that
misjudges its budget or tools wastes steps, and wasted steps are what the
study measures.

The legacy ``subtask_type`` knob still selects a prefix from
:data:`SUBTASK_PREFIXES` (``none`` in the study's sweeps).
"""

from __future__ import annotations

from typing import Any

from .config import Scope, Strategy, SubtaskType

# Prefixes selected by SubtaskType (legacy delegation nudges).
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

# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

ORCHESTRATOR_BASE = """\
You are an autonomous agent solving a task inside a Linux container. The task \
is described in the first user message. Nobody will answer questions or give \
feedback. When you stop, automated tests check the final state of the \
container (files, programs, services); only that final state is graded, so \
the work must actually be done in the container, not just described.

## Step budget
You have exactly {steps} steps. A step is one response from you. In one \
response you may call several tools; they run in the order you list them, and \
you see all their results at the start of your next step. After each step a \
note like "[step k/{steps} complete; ...]" tells you how many steps remain. \
When the steps run out the episode ends, whether or not you called `finish`. \
Plan for this: batch independent tool calls into one step, and verify your \
result (for example by running the relevant program or tests) before your \
last step.

## Tools
{tool_lines}{subagent_tools}{manager_note}
Relative paths are resolved against your shell's current directory, which \
starts in {workdir}.
"""

# One line per orchestrator tool, in prompt order; only the configured
# tools (``tools.orchestrator_tools``) are listed.
ORCHESTRATOR_TOOL_LINES: dict[str, str] = {
    "bash": (
        "- `bash`: run a shell command in the container and see its output and exit "
        "code. Your working directory and exported environment variables persist "
        "between your bash calls (shell functions, aliases and unexported variables do "
        "not). Commands run non-interactively: there is no TTY and stdin is closed, so "
        "interactive programs (editors, pagers, prompts) will not work; use "
        "non-interactive flags such as `-y` or `--batch`. Each command has a timeout "
        "(default {cmd_timeout}s; pass `timeout_sec` up to {max_cmd_timeout}s for long "
        "builds or tests); on timeout the command and its children are killed. Start "
        "long-running processes such as servers in the background with "
        "`nohup CMD > /tmp/server.log 2>&1 &` and check them in a later call. Long "
        "output is truncated in the middle, so filter it (grep, head, tail)."
    ),
    "read_file": "- `read_file`: read a file with line numbers (use start_line/end_line for large files).",
    "list_dir": "- `list_dir`: list a directory tree.",
    "search": "- `search`: search file contents with a regular expression.",
    "edit_file": (
        "- `edit_file`: replace an exact, unique snippet in a file, or create a file "
        "(empty `old`)."
    ),
}

FINISH_TOOL_LINE = (
    "- `finish`: declare the task complete. It ends the episode immediately; tools "
    "listed after it in the same response are not run. Call it only when the "
    "container is in its final, working state."
)

# Added when the orchestrator has neither bash nor edit_file of its own.
MANAGER_NOTE = (
    "\nYou cannot run commands or change files yourself. Everything that runs a "
    "command or modifies the container (including building, testing and "
    "verifying) must be delegated to subagents with `spawn_subagents`.\n"
)

SUBAGENT_TOOL_LINE = (
    "- `spawn_subagents`: delegate work to subagents (see \"Subagents\" below). "
    "Your step waits until every spawned subagent has finished."
)

ORCHESTRATOR_SUBAGENT_NOTE = """
## Subagents
`spawn_subagents(instructions)` takes a list of instructions. Subagents are \
smaller, cheaper models running in the SAME container as you. Each one starts \
fresh: it sees only its instruction, not your conversation or the task \
description, so make every instruction self-contained (the goal, relevant \
paths, commands, constraints, and what to report). Each subagent runs at most \
{max_steps} steps of its own and returns a written report; you receive all \
reports as the result of the spawn_subagents call.

Strategy: {strategy_text}

Scope: {scope_text}

Limits: {limits_text}
"""

STRATEGY_TEXT: dict[Strategy, str] = {
    Strategy.PARALLEL: (
        "parallel. Each instruction starts one subagent, and all of them run at "
        "the same time. They share the filesystem, so give concurrent subagents "
        "independent work that does not edit the same files."
    ),
    Strategy.SEQUENTIAL: (
        "sequential. Each instruction starts one subagent; they run one at a "
        "time, in the order given, so each one sees the file changes made by "
        "the ones before it."
    ),
    Strategy.ARBITER: (
        "arbiter. Each instruction is given to {n} independent candidate "
        "subagents that attempt it at the same time. An arbiter model then "
        "compares their reports{diffs} and picks the best one; if it is not "
        "confident, the candidates first see each other's reports and revise "
        "theirs. You receive only the winning report, with the arbiter's "
        "confidence and whether the candidates agreed.{isolation}"
    ),
}

SCOPE_TEXT: dict[Scope, str] = {
    Scope.EASY: (
        "easy. Subagents can only read: `read_file`, `list_dir` and `search`. "
        "They cannot run commands or change files. Use them to locate and "
        "retrieve information and summarize it for you."
    ),
    Scope.MEDIUM: (
        "medium. Subagents can read (`read_file`, `list_dir`, `search`) and "
        "run shell commands (`bash`), so they can execute programs and tests, "
        "observe the output and diagnose problems. They have no `edit_file` "
        "and are told to investigate and report rather than modify project "
        "files, but the commands they run can change the container."
    ),
    Scope.HARD: (
        "hard. Subagents have the same tools as you except spawning and "
        "finishing: `read_file`, `list_dir`, `search`, `bash` and `edit_file`. "
        "They can plan, implement changes and verify them by running tests."
    ),
}


def _limits_text(strategy: Strategy, max_concurrent: int, max_dispatches: int, n_candidates: int) -> str:
    if strategy is Strategy.ARBITER:
        per_call = max(1, max_concurrent // max(n_candidates, 1))
        return (
            f"at most {per_call} instruction(s) per spawn_subagents call (each uses "
            f"{n_candidates} candidate subagents); extra instructions in a call are not "
            f"run. Over the whole episode at most {max_dispatches} subagent runs are "
            "allowed, and every candidate counts as one run. The remaining budget is "
            "shown after each step."
        )
    order = "run" if strategy is Strategy.PARALLEL else "run, in order"
    return (
        f"at most {max_concurrent} instructions per spawn_subagents call are {order}; "
        f"extra instructions in a call are not run. Over the whole episode at most "
        f"{max_dispatches} subagent runs are allowed. The remaining budget is shown "
        "after each step."
    )


def orchestrator_system(
    subtask: SubtaskType = SubtaskType.NONE,
    with_subagents: bool = False,
    *,
    steps: int = 5,
    strategy: Strategy | str = Strategy.PARALLEL,
    scope: Scope | str = Scope.HARD,
    max_concurrent: int = 3,
    max_dispatches: int | None = None,
    subagent_max_steps: int = 8,
    n_candidates: int = 3,
    workdir: str = "the task's working directory",
    command_timeout_sec: int = 180,
    max_command_timeout_sec: int = 900,
    tools: list[str] | None = None,
) -> str:
    """The orchestrator's system prompt (legacy positional args still work).

    ``tools`` is the orchestrator's own tool list (None = all of them)."""
    strategy = Strategy(strategy)
    scope = Scope(scope)
    names = [t for t in ORCHESTRATOR_TOOL_LINES if tools is None or t in tools]
    lines = [
        ORCHESTRATOR_TOOL_LINES[t].format(cmd_timeout=command_timeout_sec, max_cmd_timeout=max_command_timeout_sec)
        for t in names
    ]
    lines.append(FINISH_TOOL_LINE)
    manager = with_subagents and not {"bash", "edit_file"} & set(names)
    body = ORCHESTRATOR_BASE.format(
        steps=steps,
        tool_lines="\n".join(lines) + "\n",
        subagent_tools=SUBAGENT_TOOL_LINE + "\n" if with_subagents else "",
        manager_note=MANAGER_NOTE if manager else "",
        workdir=workdir,
    )
    if with_subagents:
        md = max_dispatches if max_dispatches is not None else max_concurrent * steps
        strategy_text = STRATEGY_TEXT[strategy]
        if strategy is Strategy.ARBITER:
            hard = scope is Scope.HARD
            strategy_text = strategy_text.format(
                n=n_candidates,
                diffs=" (and, for edits, their file changes)" if hard else "",
                isolation=(
                    " Each candidate edits a private copy of the working directory (if it is "
                    "small enough to copy); only the winner's changes are applied to the real "
                    "working directory."
                    if hard
                    else ""
                ),
            )
        body += ORCHESTRATOR_SUBAGENT_NOTE.format(
            max_steps=subagent_max_steps,
            strategy_text=strategy_text,
            scope_text=(
                SCOPE_TEXT[scope].replace("the same tools as you except spawning and finishing", "these tools")
                if manager
                else SCOPE_TEXT[scope]
            ),
            limits_text=_limits_text(strategy, max_concurrent, md, n_candidates),
        )
    return SUBTASK_PREFIXES.get(subtask, "") + body


def orchestrator_system_for(config: Any, workdir: str) -> str:
    """:func:`orchestrator_system` filled in from an ExperimentConfig."""
    from .config import Topology  # local: keep the module's import surface small

    s = config.subagent
    return orchestrator_system(
        config.subtask_type,
        config.topology is Topology.SUBAGENTS,
        steps=config.budget.orchestrator_steps,
        strategy=s.strategy,
        scope=s.scope,
        max_concurrent=s.max_concurrent,
        max_dispatches=config.resolved_max_dispatches(),
        subagent_max_steps=s.max_steps,
        n_candidates=s.arbiter.n_candidates,
        workdir=workdir,
        command_timeout_sec=config.budget.command_timeout_sec,
        max_command_timeout_sec=config.budget.max_command_timeout_sec,
        tools=config.resolved_orchestrator_tools(),
    )


def step_note(step: int, steps: int, dispatches_left: int | None) -> str:
    """The runtime note appended after the last tool result of each step."""
    note = f"[step {step}/{steps} complete; {steps - step} remaining"
    if dispatches_left is not None:
        note += f"; dispatch budget left: {dispatches_left}"
    return note + "]"


# ---------------------------------------------------------------------------
# subagents
# ---------------------------------------------------------------------------

SUBAGENT_BASE = """\
You are a subagent: a worker that an orchestrator agent dispatched to do one \
specific job inside a Linux container. Your instruction is in the first user \
message; you cannot see the orchestrator's conversation and nobody will answer \
questions, so work with what you have.

You have at most {max_steps} steps. A step is one response; you may call \
several tools in one response and they run in order. End by calling \
`submit(report)`: the orchestrator sees ONLY your report, so make it concise \
and concrete -- what you found or did, file paths, commands you ran and their \
key output, and anything left unresolved. If you run out of steps without \
calling submit, your last message is used as the report.

{scope_text}

Relative paths are resolved against your shell's current directory, which \
starts in {workdir}.
"""

SUBAGENT_SCOPE_TEXT: dict[Scope, str] = {
    Scope.EASY: (
        "Your tools: `read_file` (numbered lines), `list_dir` and `search` "
        "(regular expressions over file contents). You cannot run commands or "
        "modify files. Your job is to locate and retrieve the information the "
        "instruction asks for and summarize it accurately (quote the relevant "
        "lines with their paths and line numbers)."
    ),
    Scope.MEDIUM: (
        "Your tools: `read_file`, `list_dir`, `search` and `bash`. With bash you "
        "can run programs and tests and inspect their output; your job is to "
        "investigate, execute, observe and diagnose. Do not modify project files "
        "unless the instruction explicitly asks you to. {bash_text}"
    ),
    Scope.HARD: (
        "Your tools: `read_file`, `list_dir`, `search`, `bash` and `edit_file`. "
        "You can plan, implement changes and verify them by running the relevant "
        "programs or tests; report what you changed and how you verified it. "
        "{bash_text}"
    ),
}

BASH_TEXT = (
    "bash: your working directory and exported variables persist between your "
    "calls; commands run non-interactively with no TTY and stdin closed; each "
    "command has a timeout (default {cmd_timeout}s, up to {max_cmd_timeout}s via "
    "timeout_sec) and is killed when it expires; start servers with "
    "`nohup CMD > log 2>&1 &`; long output is truncated in the middle."
)

PRIVATE_COPY_NOTE = """
Isolation: your private working copy of {orig} is {copy}. Make all edits \
there and do not modify {orig}: wherever the instruction mentions a path under \
{orig}, use the corresponding path under {copy}. Other candidates are working \
on the same instruction in their own copies; afterwards an arbiter compares \
the results and only the best copy's changes are applied to {orig}.
"""

LAST_STEP_NOTE = "[note: you have 1 step left; call submit with your report in that step]"


def subagent_system(
    scope: Scope | str = Scope.HARD,
    *,
    max_steps: int = 8,
    workdir: str = "the task's working directory",
    private_copy: str | None = None,
    original_workdir: str | None = None,
    command_timeout_sec: int = 180,
    max_command_timeout_sec: int = 900,
) -> str:
    scope = Scope(scope)
    bash_text = BASH_TEXT.format(cmd_timeout=command_timeout_sec, max_cmd_timeout=max_command_timeout_sec)
    text = SUBAGENT_BASE.format(
        max_steps=max_steps,
        scope_text=SUBAGENT_SCOPE_TEXT[scope].format(bash_text=bash_text),
        workdir=workdir,
    )
    if private_copy:
        text += PRIVATE_COPY_NOTE.format(orig=original_workdir or "the working directory", copy=private_copy)
    return text


# ---------------------------------------------------------------------------
# arbiter judge and debate
# ---------------------------------------------------------------------------

ARBITER_SYSTEM = """\
You are an impartial arbiter. Several candidate subagents independently \
attempted the same instruction inside a Linux container. Decide which \
candidate's result best accomplishes the instruction: correctness first, then \
completeness, then the strength of the evidence (commands run, test output, \
file changes shown). A claim without evidence counts for less than a verified \
result, and a candidate that ran out of steps or failed is usually worse.

Call the tool `submit_judgement` exactly once with:
- winner: the 0-based index of the best candidate;
- confidence: a number from 0 to 1, your probability that the winner is \
correct and the best choice;
- agreement: true if the candidates reached substantially the same result or \
conclusion, false otherwise;
- rationale: one to three sentences.
If you cannot call the tool, reply with only a JSON object with those four keys.
"""


def arbiter_user(instruction: str, candidates: list[dict[str, Any]], round_no: int) -> str:
    """Judge input. ``candidates``: dicts with index, status, steps, report and
    optional diff (the candidate's file changes)."""
    parts = [f"Instruction given to every candidate:\n{instruction}\n"]
    if round_no > 1:
        parts.append(
            "The candidates have seen each other's reports and revised theirs "
            f"(debate round {round_no - 1}). Judge the revised reports.\n"
        )
    for c in candidates:
        parts.append(f"## Candidate {c['index']} (status: {c['status']}, {c['steps']} steps)")
        parts.append(f"Report:\n{c['report'] or '(empty report)'}")
        if c.get("diff") is not None:
            parts.append(f"File changes in its private copy:\n{c['diff']}")
        parts.append("")
    parts.append(f"Pick the winner among candidates 0..{len(candidates) - 1} with submit_judgement.")
    return "\n".join(parts)


DEBATE_SYSTEM = """\
You are candidate {index} of {n}. You worked on the instruction below and \
wrote a report; the other candidates worked on the same instruction \
independently. Read their reports and re-examine your own conclusions \
critically: keep what is correct and supported by evidence, fix what is \
wrong, and state remaining disagreements with the evidence for each side. You \
cannot run tools now. Reply with your revised final report only.
"""


def debate_user(instruction: str, own_report: str, others: list[tuple[int, str]]) -> str:
    parts = [
        f"Instruction:\n{instruction}\n",
        f"Your report:\n{own_report or '(empty report)'}\n",
        "Other candidates' reports:",
    ]
    for idx, rep in others:
        parts.append(f"## Candidate {idx}\n{rep or '(empty report)'}\n")
    parts.append("Write your revised report.")
    return "\n".join(parts)
