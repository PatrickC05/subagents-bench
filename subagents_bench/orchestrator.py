"""The orchestrator: up to ``orchestrator_steps`` steps of one LLM call plus its tool calls.

Tools: its own (all five, or ``tools.orchestrator_tools`` for a manager), ``finish``, and with
subagents ``spawn_subagents``. Exit reasons: finished, max_steps, no_tool_call, cancelled,
provider_error, billing_error, error. A provider or billing failure of any role stops the
episode at once (the recorder sets the stop event); ``run`` always finalizes the records.
"""

from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .arbiter import run_arbitration
from .config import ExperimentConfig
from .environment import AgentStopped
from .llm import BillingError, ChatClient, LLMError, Message, ToolCall, ToolResult, ToolSpec
from .prompts import LENGTH_CUTOFF_NOTE, LENGTH_CUTOFF_REPLY, orchestrator_prompt, step_note
from .records import EpisodeRecorder, Who
from .subagent import Runtime, new_dispatch, run_dispatch
from .tools import ToolContext, ToolOutput, run_tool, tool_specs, truncate

SPAWN_SPEC = ToolSpec(
    "spawn_subagents",
    "Delegate work to subagents in the same container; each instruction must be self-contained. Returns "
    "their reports when all have finished. The system prompt describes the strategy, tools and limits.",
    {"type": "object", "properties": {"instructions": {"type": "array", "items": {"type": "string"},
                                                       "description": "One self-contained instruction per subagent."}},
     "required": ["instructions"]},
)
FINISH_SPEC = ToolSpec(
    "finish",
    "Declare the task complete and end the episode; the container is then graded by automated tests.",
    {"type": "object", "properties": {"summary": {"type": "string", "description": "What you did and how you verified it."}},
     "required": []},
)
MIN_REPORT_CHARS = 2000  # per report in a spawn result


class Orchestrator:
    def __init__(self, config: ExperimentConfig, env: Any, rec: EpisodeRecorder, workdir: str,
                 client_factory: Callable[[Any], Any] = ChatClient):
        self.cfg, self.env, self.rec, self.workdir = config, env, rec, workdir
        env.stop_event = rec.stop_event
        self.client_factory = client_factory
        self.subagents = config.topology == "subagents"
        self.own_tools = config.orchestrator_tools
        self.specs = tool_specs(self.own_tools, config.budget) + ([SPAWN_SPEC] if self.subagents else []) + [FINISH_SPEC]
        self.system = orchestrator_prompt(config, workdir)
        b = config.budget
        self.ctx = ToolContext(env, "orchestrator", workdir, config.tools.max_output_chars, b.command_timeout_sec,
                               b.max_command_timeout_sec)
        self.who = Who("orchestrator", "orchestrator")
        self.dispatches_left = config.subagent.max_dispatches
        self.messages: list[Message] = []
        self.has_finish = True  # interactive episodes have no `finish`

    def run(self, instruction: str) -> str:
        """Run the episode, finalize its records and return the exit reason."""
        clients: list[Any] = []
        exit_reason, error = "max_steps", None
        try:
            clients.append(self.client_factory(self.cfg.orchestrator))
            if self.subagents:
                clients.append(self.client_factory(self.cfg.subagent_model))
            self.rt = Runtime(self.cfg, self.env, self.rec, self.workdir, *clients)
            self.rec.transcript(self.who, Message("system", self.system))
            self._add(Message("user", instruction))
            for step in range(1, self.cfg.budget.orchestrator_steps + 1):
                outcome = self._step(step)
                if outcome:
                    exit_reason = outcome
                    break
        except (AgentStopped, LLMError) as e:
            failure = self.rec.failure
            if isinstance(failure, BillingError):
                exit_reason = "billing_error"
            elif failure is not None:
                exit_reason = "provider_error"
            else:
                exit_reason = "cancelled" if isinstance(e, AgentStopped) else "error"  # e.g. ContextLengthError
            cause = failure or e
            error = None if exit_reason == "cancelled" else f"{type(cause).__name__}: {cause}"
        except BaseException:
            self.rec.finalize("error", traceback.format_exc()[-4000:])
            raise
        finally:
            for c in clients:
                c.close()
        self.rec.finalize(exit_reason, error)
        return exit_reason

    def _add(self, m: Message, call_id: str | None = None) -> None:
        self.messages.append(m)
        self.rec.transcript(self.who, m, call_id)

    def _note(self, step: int) -> str:
        return step_note(step, self.cfg.budget.orchestrator_steps, self.dispatches_left if self.subagents else None)

    def _step(self, step: int) -> str | None:
        """One step -> an exit reason, or None to go on. Its step record is written in any case."""
        t_start = time.time()
        t_llm_end = call_id = None
        names: list[str] = []
        groups: list[str] = []
        finished = False
        try:
            try:
                r, call_id = self.rec.llm(self.who, self.rt.orch_client, self.system, self.messages, self.specs, step)
            finally:
                t_llm_end = time.time()
            cut_off = not r.tool_calls and r.finish_reason == "length"
            self._add(Message("assistant", r.text or (LENGTH_CUTOFF_REPLY if cut_off else ""), r.tool_calls,
                              reasoning=r.reasoning or None, reasoning_details=r.reasoning_details or None), call_id)
            if not r.tool_calls:
                if not cut_off:
                    return "no_tool_call"
                self._add(Message("user", f"{LENGTH_CUTOFF_NOTE}\n\n{self._note(step)}"))
                return None
            results: list[ToolResult] = []
            for tc in r.tool_calls:
                names.append(tc.name)
                if finished:
                    results.append(ToolResult(tc.id, "[not run: finish was called earlier in this step]", True))
                    continue
                out = self._tool(tc, step, call_id, groups)
                finished = tc.name == "finish" and self.has_finish
                results.append(ToolResult(tc.id, out.text, out.is_error))
                if self.rec.stop_event.is_set():
                    raise AgentStopped("episode stopped")
            last = results[-1]
            results[-1] = ToolResult(last.call_id, f"{last.content}\n\n{self._note(step)}", last.is_error)
            self._add(Message("tool", tool_results=results))
            return "finished" if finished else None
        finally:
            self.rec.write("steps", {
                "step": step, "call_id": call_id, "t_start": t_start, "t_llm_end": t_llm_end, "t_end": time.time(),
                "tool_names": names, "group_ids": groups, "finished": finished,
            })

    def _tool(self, tc: ToolCall, step: int, call_id: str, groups: list[str]) -> ToolOutput:
        def record(run: Callable[[], ToolOutput], who: Who = self.who) -> ToolOutput:
            return self.rec.tool(who, step, call_id, tc, run)

        if tc.name == "finish" and self.has_finish:
            return record(lambda: ToolOutput("Episode finished."))
        if tc.name == "spawn_subagents" and self.subagents:
            instructions = tc.arguments.get("instructions")
            if isinstance(instructions, str):
                instructions = [instructions]
            if not isinstance(instructions, list) or not instructions or not all(
                    isinstance(x, str) and x.strip() for x in instructions):
                return record(lambda: ToolOutput("[error] 'instructions' must be a non-empty list of strings", True))
            group_id = self.rec.new_id("group")
            groups.append(group_id)
            return record(lambda: self._spawn(instructions, step, group_id), self.who.with_group(group_id))
        if tc.name in self.own_tools:
            return record(lambda: run_tool(self.ctx, tc.name, tc.arguments))
        msg = f"[error] unknown tool {tc.name!r}; available: {', '.join(s.name for s in self.specs)}"
        return record(lambda: ToolOutput(msg, is_error=True))

    def _spawn(self, instructions: list[str], step: int, group_id: str) -> ToolOutput:
        """Grant min(requested, per-call limit, budget) instructions and run them by strategy."""
        s = self.cfg.subagent
        per = s.arbiter.n_candidates if s.strategy == "arbiter" else 1
        per_call = max(1, s.max_concurrent // per) if per > 1 else s.max_concurrent
        granted = min(len(instructions), per_call, self.dispatches_left // per)
        self.dispatches_left -= granted * per
        t_start = time.time()
        dispatch_ids: list[str] = []
        arbitration_ids: list[str] = []
        sections: list[tuple[str, str]] = []
        try:
            if s.strategy == "arbiter":
                for i, text in enumerate(instructions[:granted]):
                    outcome = run_arbitration(self.rt, text, i, group_id, step, dispatch_ids, arbitration_ids)
                    sections.append(outcome)
            else:
                ds = [new_dispatch(self.rt, text, i, group_id, step) for i, text in enumerate(instructions[:granted])]
                dispatch_ids += [d.dispatch_id for d in ds]
                if s.strategy == "parallel" and len(ds) > 1:
                    with ThreadPoolExecutor(len(ds)) as pool:
                        results = list(pool.map(lambda d: run_dispatch(self.rt, d), ds))
                else:
                    results = [run_dispatch(self.rt, d) for d in ds]
                sections = [(f"{d.agent_id}: {r.status}, {r.steps_used} steps", r.report) for d, r in zip(ds, results)]
        finally:
            self.rec.write("spawns", {
                "group_id": group_id, "step": step, "strategy": s.strategy, "scope": s.scope, "spawn_mode": "blocking",
                "requested": len(instructions), "granted": granted, "dispatch_ids": dispatch_ids,
                "arbitration_ids": arbitration_ids, "t_start": t_start, "t_end": time.time(),
            })
        parts = [f"[spawn_subagents: {s.strategy}, scope {s.scope}; ran {granted} of {len(instructions)} "
                 f"instruction(s); dispatch budget left: {self.dispatches_left}]"]
        if granted < len(instructions):
            parts.append(f"[note: instruction(s) {granted + 1}-{len(instructions)} were NOT run: the per-call "
                         "limit or the episode's dispatch budget]")
        cap = max(MIN_REPORT_CHARS, self.cfg.tools.max_output_chars // max(granted, 1))
        parts += [f"### Instruction {i + 1} ({head})\n{truncate(report, cap)}" for i, (head, report) in enumerate(sections)]
        text = "\n\n".join(parts)
        return ToolOutput(text, output_chars=len(text))
