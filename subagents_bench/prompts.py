"""System prompts and runtime notes.

The orchestrator's description of subagents never names the subagent model ("separate
model instances"), so the same-model control reads the same text, and the default
(all-tools) prompt never asks it to delegate; the manager prompt says it must, and
``subagent.delegation_hint`` adds a paragraph stating the step trade-off.
"""

from __future__ import annotations

from .config import ExperimentConfig

ORCHESTRATOR = """\
You are an autonomous agent solving a task inside a Linux container. The task is in the \
first user message. Nobody will answer questions. When you stop, automated tests check the \
final state of the container (files, programs, services), so the work must be done in the \
container, not just described.

## Step budget
You have exactly {steps} steps. A step is one response; you may call several tools in it, \
they run in order, and you see all results at the start of your next step. A note after each \
step says how many remain. When the steps run out the episode ends, finished or not, so batch \
independent tool calls and verify your result before your last step.

## Tools
{tools}
Relative paths are resolved against your shell's current directory, which starts in {workdir}.
"""

TOOL_LINES = {
    "bash": ("- `bash`: run a shell command (non-interactive, no TTY; cwd and exported variables persist; "
             "default timeout {t}s, up to {tmax}s via timeout_sec; start servers with `nohup CMD > log 2>&1 &`; "
             "long output is truncated in the middle, so filter it)."),
    "edit_file": "- `edit_file`: replace an exact, unique snippet in a file, or create a file (empty `old`).",
    "read_file": "- `read_file`: read a file with line numbers.",
    "list_dir": "- `list_dir`: list a directory tree.",
    "search": "- `search`: search file contents with a regular expression.",
}
SPAWN_LINE = "- `spawn_subagents`: delegate work to subagents (see Subagents below); your step waits for all of them."
FINISH_LINE = "- `finish`: declare the task complete; ends the episode at once (later tools in that response do not run)."
MANAGER_NOTE = ("\nYou cannot run commands or change files yourself: everything that runs a command or modifies "
                "the container (building, testing, verifying included) must be delegated with `spawn_subagents`.")

SUBAGENTS = """
## Subagents
`spawn_subagents(instructions)` takes a list of instructions. Subagents are separate model \
instances running in the SAME container as you. Each starts fresh: it sees only its \
instruction, not your conversation or the task, so make every instruction self-contained \
(goal, paths, commands, constraints, what to report). Each runs at most {max_steps} steps and \
returns a written report; you get all reports as the result of the call. One spawn_subagents \
call uses one of your steps, like any other tool call; the subagents' steps do not count.
{hint}
Strategy: {strategy}

Scope: {scope}

Limits: at most {per_call} instruction(s) per call are run{order}; extra instructions are not \
run. At most {max_dispatches} subagent runs per episode{per_run}; the budget left is shown after \
each step.
"""

STRATEGY = {
    "parallel": "parallel: each instruction starts one subagent and all run at the same time; they share "
                "the filesystem, so give them independent work that does not edit the same files.",
    "sequential": "sequential: each instruction starts one subagent; they run one at a time in the order "
                  "given, so each sees the changes of the ones before it.",
    "arbiter": "arbiter: each instruction goes to {n} independent candidate subagents at the same time. An "
               "arbiter model compares their reports{diffs} and picks the best; if it is not confident, the "
               "candidates first read each other's reports and revise theirs. You get the winning report.{copy}",
}
DELEGATION_HINT = ("\nYour steps are the scarce resource: you have {steps}, and most tasks need more work than "
                   "that. Subagent steps do not count against yours, so delegate as much of the work as their scope "
                   "(below) allows and keep your own steps for planning, combining their reports and checking the "
                   "result.\n")
ARBITER_COPY = (" Each candidate edits a private copy of the working directory and only the winner's changes "
                "are applied to it; changes outside the working directory (/etc, packages, services) are made "
                "directly by every candidate.")

SCOPE = {
    "easy": "easy: subagents can only read (`read_file`, `list_dir`, `search`); they cannot run commands or "
            "change files. Use them to locate information and summarize it.",
    "medium": "medium: subagents can read and run commands (`bash`) to execute programs and tests and diagnose "
              "problems, but they do not edit project files, even if asked: they report the exact change "
              "for you to apply. Their commands can still change the container.",
    "hard": "hard: subagents have `read_file`, `list_dir`, `search`, `bash` and `edit_file`; they can "
            "implement changes and verify them by running tests.",
}


def orchestrator_prompt(cfg: ExperimentConfig, workdir: str) -> str:
    b, s = cfg.budget, cfg.subagent
    names = cfg.orchestrator_tools
    lines = [TOOL_LINES[t].format(t=b.command_timeout_sec, tmax=b.max_command_timeout_sec) for t in names]
    subagents = cfg.topology == "subagents"
    if subagents:
        lines.append(SPAWN_LINE)
    lines.append(FINISH_LINE)
    if subagents and not {"bash", "edit_file"} & set(names):
        lines.append(MANAGER_NOTE)
    text = ORCHESTRATOR.format(steps=b.orchestrator_steps, tools="\n".join(lines), workdir=workdir)
    if not subagents:
        return text
    n = s.arbiter.n_candidates
    arbiter = s.strategy == "arbiter"
    hard = s.scope == "hard"
    return text + SUBAGENTS.format(
        max_steps=s.max_steps,
        hint=DELEGATION_HINT.format(steps=b.orchestrator_steps) if s.delegation_hint else "",
        strategy=STRATEGY[s.strategy].format(n=n, diffs=" (and their file changes)" if hard else "",
                                             copy=ARBITER_COPY if hard else ""),
        scope=SCOPE[s.scope],
        per_call=max(1, s.max_concurrent // n) if arbiter else s.max_concurrent,
        order=f", each by {n} candidates" if arbiter else " (in order)" if s.strategy == "sequential" else "",
        max_dispatches=s.max_dispatches,
        per_run="; every candidate counts as one run" if arbiter else "",
    )


def step_note(step: int, steps: int, dispatches_left: int | None) -> str:
    note = f"[step {step}/{steps} complete; {steps - step} remaining"
    return note + (f"; dispatch budget left: {dispatches_left}]" if dispatches_left is not None else "]")


SUBAGENT = """\
You are a subagent: an orchestrator agent dispatched you to do one job inside a Linux \
container. Your instruction is in the first user message; you cannot see the orchestrator's \
conversation and nobody will answer questions.

You have at most {max_steps} steps (one response each; several tool calls per response run \
in order). End by calling `submit(report)`: the orchestrator sees ONLY your report, so make it \
concise and concrete (what you found or did, file paths, commands and key output, what is left). \
If you run out of steps, your last message is the report.

{scope}

Relative paths are resolved against your shell's current directory, which starts in {cwd}.
"""
SUBAGENT_SCOPE = {
    "easy": "Your tools: `read_file`, `list_dir`, `search`. You cannot run commands or modify files: locate "
            "what the instruction asks for and summarize it accurately (quote paths and line numbers).",
    "medium": "Your tools: `read_file`, `list_dir`, `search`, `bash`. Investigate, run programs and tests, "
              "diagnose. Do not modify project files, even if asked; report the exact change (a diff or the "
              "commands) for the orchestrator to apply.",
    "hard": "Your tools: `read_file`, `list_dir`, `search`, `bash`, `edit_file`. Implement changes and verify "
            "them by running the relevant programs or tests; report what you changed and how you verified it.",
}
PRIVATE_COPY = """
Isolation: your private copy of {orig} is {copy}. Work there: paths under {orig} in your tool \
arguments and commands are redirected to {copy}. Only {orig} is copied; anything outside it \
(/etc, packages, services) is shared with the other candidates working on the same instruction. \
Afterwards an arbiter compares the results and only the best copy's changes are applied to {orig}.
"""
LAST_STEP_NOTE = "[note: you have 1 step left; call submit with your report in that step]"
# A reply cut off at the output-token limit before any tool call: the step is used and the
# agent goes on (DeepSeek's cut-off replies had no visible text).
LENGTH_CUTOFF_NOTE = ("[note: your last reply reached the output token limit before any tool call, so nothing "
                      "ran. Reason more briefly and call a tool in your next reply.]")
LENGTH_CUTOFF_REPLY = "[reply cut off at the output token limit]"


def subagent_prompt(scope: str, max_steps: int, cwd: str, workdir: str | None = None) -> str:
    """``workdir`` is given when ``cwd`` is a candidate's private copy of it."""
    text = SUBAGENT.format(max_steps=max_steps, scope=SUBAGENT_SCOPE[scope], cwd=cwd)
    return text + (PRIVATE_COPY.format(orig=workdir, copy=cwd) if workdir else "")


ARBITER = """\
You are an impartial arbiter. Several candidate subagents independently attempted the same \
instruction inside a Linux container. Pick the candidate whose result best accomplishes it: \
correctness first, then completeness, then the strength of the evidence (commands run, test \
output, file changes). Claims without evidence count less; a candidate that ran out of steps \
or failed is usually worse.

Call `submit_judgement` once with: winner (0-based index), confidence (0 to 1, your probability \
that the winner is correct and best), agreement (true if the candidates reached substantially \
the same result), rationale (one to three sentences). If you cannot call the tool, reply with \
only a JSON object with those four keys.
"""
JUDGE_RETRY_NOTE = ("Your previous attempt reached the output token limit before calling submit_judgement. "
                    "Reason briefly, then call submit_judgement right away.")


def judge_prompt(instruction: str, candidates: list[dict], round_no: int) -> str:
    """``candidates``: dicts with status, steps, report and diff (None without private copies)."""
    parts = [f"Instruction given to every candidate:\n{instruction}\n"]
    if round_no > 1:
        parts.append("The candidates have read each other's reports and revised theirs, without tools: "
                     "nothing in the container changed since they finished.\n")
    for k, c in enumerate(candidates):
        parts.append(f"## Candidate {k} (status: {c['status']}, {c['steps']} steps)\nReport:\n{c['report'] or '(empty)'}")
        if c.get("diff") is not None:
            parts.append(f"File changes in its private copy:\n{c['diff']}")
        parts.append("")
    parts.append(f"Pick the winner among candidates 0..{len(candidates) - 1} with submit_judgement.")
    return "\n".join(parts)


DEBATE = """\
You are candidate {k} of {n}. You worked on the instruction below and wrote a report; the \
others worked on it independently. Read their reports and re-examine yours critically: keep \
what is correct and supported by evidence, fix what is wrong, and state remaining disagreements \
with the evidence for each side. You cannot run tools now and nothing you did can change: \
describe only work you actually did. Reply with your revised report only.
"""


def debate_prompt(instruction: str, own: str, others: list[tuple[int, str]]) -> str:
    parts = [f"Instruction:\n{instruction}\n", f"Your report:\n{own or '(empty)'}\n", "Other candidates' reports:"]
    parts += [f"## Candidate {k}\n{r or '(empty)'}\n" for k, r in others]
    return "\n".join(parts + ["Write your revised report."])


def report_with_revision(original: str, revised: str) -> str:
    return (f"{original}\n\n[post-debate revision, written without tools after reading the other candidates' "
            f"reports; the report above is what was actually done]\n{revised}")
