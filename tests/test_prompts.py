"""Prompt text, snapshot style: the orchestrator's Subagents section (neutral
about the subagent model, identical in the same-model control, plain step
accounting, no delegation nudge in the default prompt), the manager note,
every scope's text for the orchestrator and the subagent (medium never
applies changes), the arbiter's strategy/isolation text, the private-copy
note, and the judge/debate texts used after a debate."""

from __future__ import annotations

import pytest

from fakes import make_config
from subagents_bench.config import Scope, Strategy
from subagents_bench.prompts import (
    DEBATE_FROZEN_NOTE,
    MANAGER_NOTE,
    POST_DEBATE_HEADER,
    POST_DEBATE_NOTE,
    arbiter_user,
    debate_system,
    orchestrator_system,
    orchestrator_system_for,
    report_with_revision,
    subagent_system,
)
from subagents_bench.tools import truncate_output

SUBAGENTS_SECTION = """
## Subagents
`spawn_subagents(instructions)` takes a list of instructions. Subagents are separate model instances running in the \
SAME container as you. Each one starts fresh: it sees only its instruction, not your conversation or the task \
description, so make every instruction self-contained (the goal, relevant paths, commands, constraints, and what to \
report). Each subagent runs at most 8 steps of its own and returns a written report; you receive all reports as the \
result of the spawn_subagents call.

Step accounting: one spawn_subagents call uses one of your steps, like any other tool call; the subagents' own steps \
do not count against your step budget.
"""

ORCH_SCOPE = {
    "easy": (
        "Scope: easy. Subagents can only read: `read_file`, `list_dir` and `search`. They cannot run commands or "
        "change files. Use them to locate and retrieve information and summarize it for you."
    ),
    "medium": (
        "Scope: medium. Subagents can read (`read_file`, `list_dir`, `search`) and run shell commands (`bash`), so "
        "they can execute programs and tests, observe the output and diagnose problems. They have no `edit_file` and "
        "will not apply changes to project files, even if you ask them to: they report the exact change (a diff or "
        "the commands to run) for you to apply. The commands they run (builds, tests, installs) can still change the "
        "container."
    ),
    "hard": (
        "Scope: hard. Subagents have the same tools as you except spawning and finishing: `read_file`, `list_dir`, "
        "`search`, `bash` and `edit_file`. They can plan, implement changes and verify them by running tests."
    ),
}

BASH_TEXT = (
    "bash: your working directory and exported variables persist between your calls; commands run non-interactively "
    "with no TTY and stdin closed; each command has a timeout (default 180s, up to 900s via timeout_sec) and is "
    "killed when it expires; start servers with `nohup CMD > log 2>&1 &`; long output is truncated in the middle."
)

SUB_SCOPE = {
    "easy": (
        "Your tools: `read_file` (numbered lines), `list_dir` and `search` (regular expressions over file contents). "
        "You cannot run commands or modify files. Your job is to locate and retrieve the information the instruction "
        "asks for and summarize it accurately (quote the relevant lines with their paths and line numbers)."
    ),
    "medium": (
        "Your tools: `read_file`, `list_dir`, `search` and `bash`. With bash you can run programs and tests and "
        "inspect their output; your job is to investigate, execute, observe and diagnose. Do not modify project "
        "files, even if the instruction asks you to; instead, report the exact change (a diff or the commands to "
        f"run) for the orchestrator to apply. {BASH_TEXT}"
    ),
    "hard": (
        "Your tools: `read_file`, `list_dir`, `search`, `bash` and `edit_file`. You can plan, implement changes and "
        "verify them by running the relevant programs or tests; report what you changed and how you verified it. "
        f"{BASH_TEXT}"
    ),
}

ARBITER_HARD = (
    "Strategy: arbiter. Each instruction is given to 3 independent candidate subagents that attempt it at the same "
    "time. An arbiter model then compares their reports (and, for edits, their file changes) and picks the best one; "
    "if it is not confident, the candidates first see each other's reports and revise theirs. You receive only the "
    "winning report, with the arbiter's confidence and whether the candidates agreed. Each candidate edits a private "
    "copy of the working directory (if it is small enough to copy); only the winner's changes are applied to the real "
    "working directory. Only the working directory is copied: changes outside it (for example under /etc, installed "
    "packages or running services) are made directly in the container by every candidate."
)

PRIVATE_COPY = """
Isolation: your private working copy of /app is /c/cand0. Make all edits there and do not modify /app: wherever the \
instruction mentions a path under /app, use the corresponding path under /c/cand0. Paths under /app in your tool \
arguments and commands are redirected to /c/cand0 automatically, and changes made in /app by other means are undone. \
Only /app is copied: anything outside it (for example /etc, /var, installed packages and running services) is shared \
with the other candidates and changes immediately. Other candidates are working on the same instruction in their own \
copies; afterwards an arbiter compares the results and only the best copy's changes are applied to /app (mentions of \
/c/cand0 inside its files become /app).
"""


def orch(**kw):
    base = dict(with_subagents=True, steps=5, max_concurrent=3, max_dispatches=15, workdir="/app")
    return orchestrator_system(**{**base, **kw})


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", list(Strategy))
@pytest.mark.parametrize("scope", list(Scope))
def test_subagents_section_is_neutral_and_states_step_accounting(strategy, scope):
    text = orch(strategy=strategy, scope=scope)
    assert SUBAGENTS_SECTION in text
    assert f"\n{ORCH_SCOPE[scope.value]}\n" in text
    for word in ("smaller", "cheaper", "larger", "weaker", "Gemma", "DeepSeek"):
        assert word not in text
    # The default (all-tools) prompt never asks the orchestrator to delegate.
    assert MANAGER_NOTE not in text and "must be delegated" not in text and "must delegate" not in text


def test_single_topology_has_no_subagents_section():
    text = orchestrator_system(with_subagents=False, steps=5, workdir="/app")
    assert "## Subagents" not in text and "spawn_subagents" not in text and "Step accounting" not in text


def test_same_model_control_gets_the_same_prompt(tmp_path):
    gemma = make_config(tmp_path, subagent={"model": {"model": "nvidia/Gemma-4-31B-IT-NVFP4", "provider": "sail"}})
    same = make_config(tmp_path, subagent={"model": None})
    assert same.resolved_subagent_model().model != gemma.resolved_subagent_model().model
    assert orchestrator_system_for(gemma, "/app") == orchestrator_system_for(same, "/app")


def test_manager_prompt():
    text = orch(strategy="parallel", scope="hard", tools=["read_file", "list_dir", "search"])
    assert MANAGER_NOTE == (
        "\nYou cannot run commands or change files yourself. Everything that runs a command or modifies the container "
        "(including building, testing and verifying) must be delegated to subagents with `spawn_subagents`.\n"
    )
    assert MANAGER_NOTE in text and SUBAGENTS_SECTION in text
    assert "- `bash`" not in text and "- `edit_file`" not in text
    assert (
        "\nScope: hard. Subagents have these tools: `read_file`, `list_dir`, `search`, `bash` and `edit_file`. They can "
        "plan, implement changes and verify them by running tests.\n"
    ) in text


def test_arbiter_strategy_text():
    assert f"\n{ARBITER_HARD}\n" in orch(strategy="arbiter", scope="hard")
    for scope in ("easy", "medium"):
        text = orch(strategy="arbiter", scope=scope)
        assert "compares their reports and picks the best one" in text
        assert "private copy" not in text and "file changes" not in text
        assert "whether the candidates agreed.\n" in text


# ---------------------------------------------------------------------------
# subagents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scope", list(Scope))
def test_subagent_scope_text(scope):
    text = subagent_system(scope, workdir="/app")
    assert f"\n{SUB_SCOPE[scope.value]}\n" in text
    assert "Isolation:" not in text
    assert text.endswith("Relative paths are resolved against your shell's current directory, which starts in /app.\n")


def test_medium_subagents_never_apply_changes():
    text = subagent_system("medium", workdir="/app")
    assert "even if the instruction asks you to" in text and "unless the instruction" not in text


def test_private_copy_note():
    text = subagent_system("hard", workdir="/c/cand0", private_copy="/c/cand0", original_workdir="/app")
    assert text.endswith(PRIVATE_COPY)


# ---------------------------------------------------------------------------
# arbiter judge and debate
# ---------------------------------------------------------------------------


def test_debate_system_text():
    plain = debate_system(1, 3)
    assert plain == (
        "You are candidate 1 of 3. You worked on the instruction below and wrote a report; the other candidates worked "
        "on the same instruction independently. Read their reports and re-examine your own conclusions critically: "
        "keep what is correct and supported by evidence, fix what is wrong, and state remaining disagreements with the "
        "evidence for each side. You cannot run tools now, and nothing you did can be changed any more: describe only "
        "work you actually did, and do not claim commands, edits or results you did not make. Reply with your revised "
        "final report only.\n"
    )
    frozen = debate_system(1, 3, private_copy=True)
    assert DEBATE_FROZEN_NOTE == (
        " The files in your private copy are frozen as you left them: the arbiter compares those files, and if you win "
        "they are applied exactly as they are, whatever your revised report says."
    )
    assert frozen == plain.replace(" Reply with", DEBATE_FROZEN_NOTE + " Reply with")


def test_judge_text_after_a_debate():
    cands = [{"index": k, "status": "submitted", "steps": 2, "report": f"r{k}", "diff": f"A f{k}"} for k in range(2)]
    first = arbiter_user("do it", cands, 1)
    assert "revised" not in first and "File changes in its private copy:\nA f0" in first
    second = arbiter_user("do it", cands, 2)
    assert (
        "The candidates have seen each other's reports and revised theirs (debate round 1). The revisions were written "
        "without tools: nothing in the container changed after the candidates finished. The file changes shown are "
        "unchanged since then, and they are what would be applied: weigh them over claims in the revised reports.\n"
    ) in second
    shared = arbiter_user("do it", [{**c, "diff": None} for c in cands], 2)
    assert "nothing in the container changed after the candidates finished. Judge the revised reports" in shared
    assert "weigh them over claims" not in shared and "File changes" not in shared


def test_report_with_revision():
    assert POST_DEBATE_NOTE == (
        "[post-debate revision, written without tools after reading the other candidates' reports. No files changed "
        "after the candidates finished: the report above and the applied files listed below are what was actually done]"
    )
    assert report_with_revision("did A", "now B") == f"{POST_DEBATE_HEADER}\ndid A\n\n{POST_DEBATE_NOTE}\nnow B"
    # Real-size reports (smoke fix-git: 7312-char report, 7285-char revision)
    # overflow the orchestrator's 12000-char per-report cap, which elides the
    # middle; the header still says which part is which.
    shown, truncated = truncate_output(report_with_revision("A" * 7312, "B" * 7285), 12000)
    assert truncated and POST_DEBATE_NOTE not in shown and shown.startswith(f"{POST_DEBATE_HEADER}\nAAA")
