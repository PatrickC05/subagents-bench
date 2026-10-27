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
NO_READ_NOTE = ("You cannot read or search the files either: subagents read the code for you, so ask them for "
                "what you need to know.")
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
SUBAGENT_HELPERS = """
You can start helper subagents of your own with `spawn_subagents(instructions)`: one helper per \
instruction, all running at the same time; the call returns when they have all finished, with \
their reports. Helpers start fresh in the same container and see only their instruction, so make \
each one self-contained (goal, paths, what to report). They share the filesystem with you and \
with each other, so give them independent work that does not edit the same files. Helpers cannot \
start helpers. At most {n} helper runs in all; the number left is shown after each call.
"""
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


def subagent_prompt(scope: str, max_steps: int, cwd: str, workdir: str | None = None, helpers: int = 0) -> str:
    """``workdir`` is given when ``cwd`` is a candidate's private copy of it; ``helpers``: the helper
    subagents it may start (``subagent.max_children``; 0 for a helper itself)."""
    text = SUBAGENT.format(max_steps=max_steps, scope=SUBAGENT_SCOPE[scope], cwd=cwd)
    text += SUBAGENT_HELPERS.format(n=helpers) if helpers else ""
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


# Interactive episodes (``interactive:`` in the config): a timed session in which the task's
# subtasks arrive as user messages, with no step budget and no `finish`.
INTERACTIVE = """\
You are an autonomous agent working inside a Linux container, in a session with a user. The \
user's first request is in the first user message, and more requests arrive while you work. \
The user does not answer questions. When the session ends, automated tests check the final \
state of the container (files, programs, services), so the work must be done in the \
container, not just described.

## Session
The session lasts {minutes} minutes from the first message, then ends wherever you are. New \
user messages{and_reports} are delivered between your steps: a step is one response with its \
tool calls, and anything that arrives during a step waits until all of its tools have \
finished. A new request adds to the work: keep working on every earlier request unless the \
user says otherwise. A note after each step shows the time used{running}. When you have \
nothing to do until {until}, call `wait`.

## Tools
{tools}
Relative paths are resolved against your shell's current directory, which starts in {workdir}.
"""
REPLY_LINE = "- `reply`: send a message to the user about one of their requests (see Replying to the user below)."
REPLIES = """
## Replying to the user
Every user message has an id (#3). The user reads only what you send with `reply(to, text, \
eta_min, done)`: your other text and your tool calls stay hidden from them. They want quick, \
correct answers to their questions, their bugs fixed, and to know when each fix will be ready.
- A question: answer it with `reply` (to its id). Answer from the code; a short answer is fine. \
Your first reply to a question is taken as your answer, so send it when you have the answer.
- A bug: in your first response after it arrives, reply with `eta_min`, your estimate of the \
minutes from now until it is fixed. Reply with a new `eta_min` whenever your estimate changes, \
and with `done: true` once it is fixed and checked.
- A status check: reply to each open bug with an updated `eta_min`.
- An urgent bug matters more than the earlier ones: fix it before them.
"""
# interactive.change_of_plan: a bullet for the messages that take back a bug (only with them, so the
# prompt of every other condition stays as it was)
CHANGE_REPLY = ("- A request to forget or undo an earlier bug: do what it asks, and reply to its id with `done: "
                "true` once that code is as the user wants it.\n")
# interactive.bundle > 1: a message can list several bugs; replies name the item
REPLIES_BUNDLE = """\
- A message that lists several bugs (1., 2., ...): each item is a bug of its own. Reply about \
each one with `item` set to its number (an eta_min or done: true for that bug alone); a reply \
without `item` is about every bug of the message.
"""
WAIT_LINE = "- `wait`: pause until {until} (or timeout_sec passes); replying without a tool call also waits."
STOP_LINE = ("- `stop_subagents`: stop running background subagents by id (sub-3); each stops at its next step "
             "and its report so far arrives as a message.")
BACKGROUND_SPAWN_LINE = ("- `spawn_subagents`: start subagents in the background (see Subagents below); "
                         "it returns at once and each report arrives as a message.")
SUBAGENTS_BACKGROUND = """
## Subagents
`spawn_subagents(instructions)` takes a list of instructions and starts one subagent per \
instruction in the background; the call returns at once. Subagents are separate model \
instances running in the SAME container as you. Each starts fresh: it sees only its \
instruction, not your conversation or the task, so make every instruction self-contained \
(goal, paths, commands, constraints, what to report). Each runs at most {max_steps} steps; its \
written report arrives as a message when it finishes, and you keep working (or `wait`) \
meanwhile. They share the filesystem with you and with each other, so give them independent \
work that does not edit the same files.
{hint}
Scope: {scope}

Limits: at most {max_concurrent} subagents run at once; instructions beyond the free slots are \
not run. At most {max_dispatches} subagent runs per episode; the budget left is shown after each \
step.
"""
SUBAGENTS_BLOCKING = """
## Subagents
`spawn_subagents(instructions)` takes a list of instructions and starts one subagent per \
instruction; the call returns when all of them have finished, with their reports. Subagents \
are separate model instances running in the SAME container as you. Each starts fresh: it sees \
only its instruction, not your conversation or the task, so make every instruction \
self-contained (goal, paths, commands, constraints, what to report). Each runs at most \
{max_steps} steps. They run at the same time and share the filesystem with you and with each \
other, so give them independent work that does not edit the same files.

Scope: {scope}

Limits: at most {max_concurrent} instruction(s) per call are run; extra instructions are not \
run. At most {max_dispatches} subagent runs per episode; the budget left is shown after each step.
"""
INTERACTIVE_DELEGATION_HINT = ("\nYour attention is the scarce resource: while your own tools run, new requests "
                               "wait until they finish. Subagents run in the background, so delegate long work "
                               "(investigating and fixing a bug, running a test suite) and keep yourself free to "
                               "respond to new requests and to combine reports.\n")
INTERACTIVE_AVAILABILITY_HINT = ("\nStay available to the user: new messages reach you only between your steps, so "
                                 "every long step, long thinking included, keeps the user waiting. Keep each step "
                                 "short. Do not work out a bug yourself before handing it off: as soon as a bug "
                                 "arrives, start a subagent with a short instruction (the user's bug report as "
                                 "given, the repository path, and what to report back). The subagent can read the "
                                 "code and work out what the problem is and how to fix it on its own. Then reply to "
                                 "the user and go back to waiting.\n")
INTERACTIVE_OWNER_HINT = ("\nPut one subagent in charge of each bug: the subagent you start for a bug owns it end "
                          "to end (finding the cause, fixing it, testing the fix and reporting back), so say in its "
                          "instruction that it is in charge of that bug. It can start helper subagents of its own to "
                          "split the work. You coordinate: reply to the user and pass on what the reports say.\n")
FIRST_HINT = "{instruction}{note}\n\n{request}"
# interactive.scope "reported": the task's own goal (SWE-sweep: find and fix every bug) is set aside,
# so a bug the agent fixes before it is sent cannot pass for one it never started.
REPORTED_SCOPE = ("{instruction}\n\n---\nIn this session you do not search the repository for bugs on your "
                  "own: the user points you to the bugs to fix, one message at a time. Fix the bugs the user "
                  "reports; every rule above still applies.{note}\n\n{request}")
NEXT_HINT = "Another one, while you keep working on the earlier ones.\n\nHint: {hint}"
# interactive.replies: every message carries its id
FIRST_REQUEST = "Request #{id}: a bug.\n\nHint: {hint}"
NEXT_REQUEST = "Request #{id}: another bug, while you keep working on the earlier ones.\n\nHint: {hint}"
URGENT_REQUEST = "Request #{id}: another bug, and this one is urgent: fix it before the earlier ones.\n\nHint: {hint}"
# interactive.bundle: several bugs in one message, {hint} their numbered list
FIRST_BUNDLE = "Request #{id}: a few things are broken.\n\n{hint}"
NEXT_BUNDLE = "Request #{id}: a few more things are broken, while you keep working on the earlier ones.\n\n{hint}"
URGENT_BUNDLE = ("Request #{id}: a few more things are broken, and these are urgent: fix them before the earlier "
                 "ones.\n\n{hint}")
QUESTION_REQUEST = "Question #{id}: {question}"
# interactive.change_of_plan; {target}: the id of the bug taken back
CANCEL_REQUEST = ("Request #{id}: forget request #{target}: that behaviour is intended. Leave that code as it was "
                  "before you started (undo any change you made for it).")
UNDO_REQUEST = "Request #{id}: please undo your fix for #{target}: we need the old behaviour after all."
# {ids}: every bug sent before the check (``hints.schedule`` puts them in the check's text), so the
# message is the same in every condition whatever the agent fixed or said done
STATUS_REQUEST = ("Status check #{id}: how long until each of your bug requests so far ({ids}) is fixed? Reply to "
                  "each one's own id with eta_min, or with done: true if it is fixed.")
USER_EVENT = "[{at} elapsed] Message from the user:\n{text}"
REPORT_EVENT = "[{at} elapsed] Report from subagent {agent} ({status}, {steps} steps) on: {instruction}\n{report}"
WAIT_RESULT = "[waited {sec:.0f}s]"


def clock(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def interactive_prompt(cfg: ExperimentConfig, workdir: str) -> str:
    b, s, i = cfg.budget, cfg.subagent, cfg.interactive
    names = cfg.orchestrator_tools
    subagents = cfg.topology == "subagents"
    background = subagents and s.spawn_mode == "background"
    until = "a subagent report or a new user message arrives" if background else "a new user message arrives"
    lines = [TOOL_LINES[t].format(t=b.command_timeout_sec, tmax=b.max_command_timeout_sec) for t in names]
    if subagents:
        lines.append(BACKGROUND_SPAWN_LINE if background else SPAWN_LINE)
    if background and s.stop_tool:
        lines.append(STOP_LINE)
    lines.append(WAIT_LINE.format(until=until))
    if i.replies:
        lines.append(REPLY_LINE)
    if subagents and not {"bash", "edit_file"} & set(names):
        lines.append(MANAGER_NOTE + (" " + NO_READ_NOTE if not names else ""))
    text = INTERACTIVE.format(minutes=f"{i.wall_clock_sec / 60:g}", and_reports=" and subagent reports" if background else "",
                              running=" and the subagents running" if background else "", until=until,
                              tools="\n".join(lines), workdir=workdir)
    if i.replies:
        text += REPLIES + (CHANGE_REPLY if i.change_of_plan else "") + (REPLIES_BUNDLE if i.bundle > 1 else "")
    if not subagents:
        return text
    if background:
        hint = (INTERACTIVE_DELEGATION_HINT if s.delegation_hint else "") + \
            (INTERACTIVE_AVAILABILITY_HINT if s.availability_hint else "") + \
            (INTERACTIVE_OWNER_HINT if s.owner_hint else "")
        return text + SUBAGENTS_BACKGROUND.format(max_steps=s.max_steps, hint=hint, scope=SCOPE[s.scope],
                                                  max_concurrent=s.max_concurrent, max_dispatches=s.max_dispatches)
    if s.strategy == "parallel":  # the background text but for the blocking return, so only that differs
        return text + SUBAGENTS_BLOCKING.format(max_steps=s.max_steps, scope=SCOPE[s.scope],
                                                max_concurrent=s.max_concurrent, max_dispatches=s.max_dispatches)
    n = s.arbiter.n_candidates
    arbiter, hard = s.strategy == "arbiter", s.scope == "hard"
    return text + SUBAGENTS.format(
        max_steps=s.max_steps, hint="",
        strategy=STRATEGY[s.strategy].format(n=n, diffs=" (and their file changes)" if hard else "",
                                             copy=ARBITER_COPY if hard else ""),
        scope=SCOPE[s.scope],
        per_call=max(1, s.max_concurrent // n) if arbiter else s.max_concurrent,
        order=f", each by {n} candidates" if arbiter else " (in order)" if s.strategy == "sequential" else "",
        max_dispatches=s.max_dispatches, per_run="; every candidate counts as one run" if arbiter else "",
    )


def interactive_note(elapsed: float, total: float, running: list[tuple[str, float]] | None,
                     dispatches_left: int | None) -> str:
    """``running``: (agent id, seconds running) of the live background subagents, or None."""
    note = f"[{clock(elapsed)} of {clock(total)} elapsed"
    if running is not None:
        note += "; subagents running: " + (", ".join(f"{a} ({clock(t)})" for a, t in running) or "none")
    return note + (f"; dispatch budget left: {dispatches_left}]" if dispatches_left is not None else "]")


def request_text(kind: str, msg_id: int | None, text: str, urgent: bool = False, first: bool = False,
                 n_items: int = 1) -> str:
    """A scheduled user message as the agent reads it (after the elapsed-time header); a status
    check's ``text`` is the ids of the bugs sent before it (``#1, #2, #4``), a cancel's or undo's the
    id of the bug it takes back (``#4``); a bug message with ``n_items`` > 1 (``interactive.bundle``)
    has their numbered list as ``text``."""
    if msg_id is None:  # without interactive.replies: a bug, as before
        return f"Hint: {text}" if first else NEXT_HINT.format(hint=text)
    if kind == "question":
        return QUESTION_REQUEST.format(id=msg_id, question=text)
    if kind == "status":
        return STATUS_REQUEST.format(id=msg_id, ids=text)
    if kind in ("cancel", "undo"):
        return (CANCEL_REQUEST if kind == "cancel" else UNDO_REQUEST).format(id=msg_id, target=text.lstrip("#"))
    if n_items > 1:
        return (FIRST_BUNDLE if first else URGENT_BUNDLE if urgent else NEXT_BUNDLE).format(id=msg_id, hint=text)
    return (FIRST_REQUEST if first else URGENT_REQUEST if urgent else NEXT_REQUEST).format(id=msg_id, hint=text)
