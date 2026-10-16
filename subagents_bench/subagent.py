"""Subagents: fresh-context tool loops with their scope's tools plus ``submit(report)``."""

from __future__ import annotations

import shlex
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .config import ExperimentConfig
from .environment import AgentStopped
from .llm import LLMError, Message, ToolResult, ToolSpec
from .prompts import LAST_STEP_NOTE, LENGTH_CUTOFF_NOTE, LENGTH_CUTOFF_REPLY, subagent_prompt
from .records import EpisodeRecorder, Who
from .tools import SCOPE_TOOLS, ToolContext, ToolOutput, run_tool, tool_specs, truncate

SUBMIT_SPEC = ToolSpec(
    "submit",
    "Finish and return your report to the orchestrator, which sees only this report.",
    {"type": "object", "properties": {"report": {"type": "string", "description": "Your concise, concrete report."}},
     "required": ["report"]},
)
HELPER_SPEC = ToolSpec(
    "spawn_subagents",
    "Start helper subagents in the same container, one per self-contained instruction, all at once; returns "
    "their reports when all have finished.",
    {"type": "object", "properties": {"instructions": {"type": "array", "items": {"type": "string"},
                                                       "description": "One self-contained instruction per helper."}},
     "required": ["instructions"]},
)
MIN_HELPER_REPORT_CHARS = 2000  # per helper report in a spawn result
FILES_CAP = 200
INSTRUCTION_CHARS = 4000  # of the instruction in the dispatch record (the transcript has it whole)


@dataclass
class Runtime:
    """Per-episode state every agent shares."""

    config: ExperimentConfig
    env: Any
    rec: EpisodeRecorder
    workdir: str
    orch_client: Any
    sub_client: Any = None

    @property
    def stopped(self) -> bool:
        return self.rec.stop_event.is_set()


@dataclass
class Dispatch:
    instruction: str
    instruction_index: int
    group_id: str
    step: int
    dispatch_id: str
    agent_id: str
    cwd: str
    arbitration_id: str | None = None
    candidate_index: int | None = None
    private_copy: bool = False  # cwd is an arbiter candidate's copy of the workdir
    parent: str | None = None  # the agent_id of the subagent that started this helper

    @property
    def who(self) -> Who:
        return Who("subagent", self.agent_id, self.dispatch_id, self.group_id, self.arbitration_id)


@dataclass
class DispatchResult:
    status: str  # submitted | no_tool_call | max_steps | error | cancelled
    report: str
    steps_used: int


def new_dispatch(rt: Runtime, instruction: str, index: int, group_id: str, step: int, *, cwd: str | None = None,
                 **kw: Any) -> Dispatch:
    """Allocate a dispatch's ids at queue time (so a spawn lists it even if it never runs)."""
    did = rt.rec.new_id("dispatch")
    return Dispatch(instruction, index, group_id, step, did, "sub-" + did.split("-")[1], cwd or rt.workdir, **kw)


def run_dispatch(rt: Runtime, d: Dispatch) -> DispatchResult:
    """Run one subagent and write its dispatch record; never raises."""
    cfg, s = rt.config, rt.config.subagent
    names = SCOPE_TOOLS[s.scope]
    helpers_left = s.max_children if d.parent is None and not d.private_copy else 0
    specs = tool_specs(names, cfg.budget) + ([HELPER_SPEC] if helpers_left else []) + [SUBMIT_SPEC]
    system = subagent_prompt(s.scope, s.max_steps, d.cwd, rt.workdir if d.private_copy else None, helpers_left)
    ctx = ToolContext(rt.env, d.agent_id, d.cwd, s.max_output_chars, cfg.budget.command_timeout_sec,
                      cfg.budget.max_command_timeout_sec, (rt.workdir, d.cwd) if d.private_copy else None)
    who, rec = d.who, rt.rec
    messages: list[Message] = []

    def add(m: Message, call_id: str | None = None) -> None:
        messages.append(m)
        rec.transcript(who, m, call_id)

    t_start = time.time()
    status, report, error, last_text, steps, marker = "max_steps", "", None, "", 0, None
    try:
        marker = None if s.scope == "easy" else _marker(rt, d.dispatch_id)
        rec.transcript(who, Message("system", system))
        add(Message("user", d.instruction))
        for step in range(1, s.max_steps + 1):
            steps = step
            r, call_id = rec.llm(who, rt.sub_client, system, messages, specs, step)
            cut_off = not r.tool_calls and r.finish_reason == "length"
            add(Message("assistant", r.text or (LENGTH_CUTOFF_REPLY if cut_off else ""), r.tool_calls,
                        reasoning=r.reasoning or None), call_id)
            if r.text.strip():
                last_text = r.text
            warn = step == s.max_steps - 1
            if not r.tool_calls:
                if not cut_off:
                    status, report = "no_tool_call", r.text
                    break
                add(Message("user", LENGTH_CUTOFF_NOTE + (f"\n\n{LAST_STEP_NOTE}" if warn else "")))
                continue
            results: list[ToolResult] = []
            for tc in r.tool_calls:
                if status == "submitted":
                    results.append(ToolResult(tc.id, "[not run: submit was called earlier in this step]", True))
                    continue
                if tc.name == "submit":
                    rep = tc.arguments.get("report")
                    report, status = (rep if isinstance(rep, str) else r.text), "submitted"
                    out = rec.tool(who, step, call_id, tc, lambda: ToolOutput("Report submitted."))
                elif tc.name == "spawn_subagents" and s.max_children and d.parent is None and not d.private_copy:
                    out, helpers_left = _helpers(rt, d, tc, step, call_id, helpers_left)
                elif tc.name in names:
                    out = rec.tool(who, step, call_id, tc, lambda tc=tc: run_tool(ctx, tc.name, tc.arguments))
                else:
                    msg = f"[error] tool {tc.name!r} is not available to you; your tools: {', '.join(names)}, submit"
                    out = rec.tool(who, step, call_id, tc, lambda msg=msg: ToolOutput(msg, is_error=True))
                results.append(ToolResult(tc.id, out.text, out.is_error))
                if rt.stopped:
                    raise AgentStopped("episode stopped")
            if warn and status != "submitted":
                last = results[-1]
                results[-1] = ToolResult(last.call_id, f"{last.content}\n\n{LAST_STEP_NOTE}", last.is_error)
            add(Message("tool", tool_results=results))
            if status == "submitted":
                break
    except AgentStopped:
        status = "cancelled"
    except LLMError as e:
        status, error = "error", f"{type(e).__name__}: {e}"
    except Exception:  # a harness bug fails this dispatch, not the episode
        status, error = "error", traceback.format_exc()[-2000:]
    report = report or last_text or f"(no report: the subagent ended with status {status})"
    files: list[str] = []
    if marker and not rt.stopped:
        try:
            files = _modified_since(rt, marker, d.cwd)
        except AgentStopped:
            pass
    rec.write("dispatches", {
        "dispatch_id": d.dispatch_id, "group_id": d.group_id, "agent_id": d.agent_id,
        "arbitration_id": d.arbitration_id, "candidate_index": d.candidate_index,
        "instruction_index": d.instruction_index, "instruction": d.instruction[:INSTRUCTION_CHARS],
        "parent_agent_id": d.parent,
        "status": status, "steps_used": steps, "t_start": t_start,
        "t_end": time.time(), "report_chars": len(report), "files_modified": files, "error": error,
    })
    return DispatchResult(status, report, steps)


def _helpers(rt: Runtime, d: Dispatch, tc: Any, step: int, call_id: str, left: int) -> tuple[ToolOutput, int]:
    """A subagent's spawn_subagents: run min(requested, left) helpers at once, blocking, and return their reports."""
    instructions = tc.arguments.get("instructions")
    if isinstance(instructions, str):
        instructions = [instructions]
    if not isinstance(instructions, list) or not instructions or not all(isinstance(x, str) and x.strip()
                                                                         for x in instructions):
        out = rt.rec.tool(d.who, step, call_id, tc,
                          lambda: ToolOutput("[error] 'instructions' must be a non-empty list of strings", True))
        return out, left
    granted = min(len(instructions), left)
    group_id = rt.rec.new_id("group")
    kids = [new_dispatch(rt, text, i, group_id, step, cwd=d.cwd, parent=d.agent_id)
            for i, text in enumerate(instructions[:granted])]

    def run() -> ToolOutput:
        results = []
        if kids:
            with ThreadPoolExecutor(len(kids)) as pool:
                results = list(pool.map(lambda k: run_dispatch(rt, k), kids))
        parts = [f"[spawn_subagents: ran {granted} of {len(instructions)} helper(s); helper runs left: {left - granted}]"]
        if granted < len(instructions):
            parts.append(f"[note: instruction(s) {granted + 1}-{len(instructions)} were NOT run: no helper runs left]")
        cap = max(MIN_HELPER_REPORT_CHARS, rt.config.subagent.max_output_chars // max(granted, 1))
        parts += [f"### Helper {i + 1} ({k.agent_id}: {r.status}, {r.steps_used} steps)\n{truncate(r.report, cap)}"
                  for i, (k, r) in enumerate(zip(kids, results))]
        text = "\n\n".join(parts)
        return ToolOutput(text, output_chars=len(text))

    return rt.rec.tool(d.who, step, call_id, tc, run), left - granted


def _marker(rt: Runtime, name: str) -> str:
    path = f"{rt.env.state_root}/markers/{name}"
    rt.env.exec(f"mkdir -p -- {shlex.quote(rt.env.state_root)}/markers && touch -- {shlex.quote(path)}", 60)
    return path


def _modified_since(rt: Runtime, marker: str, root: str) -> list[str]:
    """Files under ``root`` (harness state excluded) newer than ``marker``, capped."""
    q = shlex.quote
    res = rt.env.exec(f"find {q(root)} -path {q(rt.env.state_root)} -prune -o -type f -newer {q(marker)} -print "
                      f"2>/dev/null | head -n {FILES_CAP}", 300)
    return sorted(p for p in res.stdout.split("\n") if p)
