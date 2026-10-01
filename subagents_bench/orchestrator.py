"""The orchestrator agent: a frontier model directing subagents.

The orchestrator runs a bounded tool loop (``budget.orchestrator_steps`` steps,
fixed at 5 for the study). On top of the base tool kit it gets:

* ``spawn_subagents`` -- dispatch one or more subagents, each with its own
  instruction. Dispatches run concurrently (threads; the LLM calls are
  I/O-bound) up to ``subagent.max_concurrent`` at a time, and are capped over
  the episode by ``subagent.max_dispatches``. Every dispatch after the first is
  counted as a re-dispatch for telemetry.
* ``finish`` -- declare the issue resolved and end the episode.

In the ``single`` topology the subagent tools are withheld, giving baseline 1
(orchestrator working alone) for free.

The loop also enforces the two hard budgets -- total token budget and wall-clock
timeout -- ending the episode with the appropriate ``exit_reason`` when either
is hit.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from .config import ExperimentConfig, Topology
from .environment import Environment
from .llm import LLMClient, Message, ToolCall, ToolResult, ToolSpec, make_client
from .prompts import orchestrator_system
from .subagent import Subagent
from .telemetry import CallEvent, Telemetry
from .tools import BASE_SPECS, BASE_TOOLS

SPAWN_SPEC = ToolSpec(
    name="spawn_subagents",
    description=(
        "Dispatch one or more subagents to work in the shared sandbox, in "
        "parallel. Provide a list of instructions; each becomes one subagent. "
        "Returns each subagent's report."
    ),
    parameters={
        "type": "object",
        "properties": {
            "instructions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One self-contained instruction per subagent to spawn.",
            }
        },
        "required": ["instructions"],
    },
)

FINISH_SPEC = ToolSpec(
    name="finish",
    description="Declare the issue resolved and end the session.",
    parameters={
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "What you changed and why it resolves the issue."}
        },
        "required": [],
    },
)


class Orchestrator:
    def __init__(self, config: ExperimentConfig, env: Environment, telemetry: Telemetry):
        self.config = config
        self.env = env
        self.telemetry = telemetry
        self.client: LLMClient = make_client(config.orchestrator)

        self.with_subagents = config.topology is Topology.SUBAGENTS
        self._sub_client: LLMClient | None = None
        if self.with_subagents:
            self._sub_client = make_client(config.resolved_subagent_model())

        self._dispatch_count = 0
        self._sub_counter = 0
        # Instruction-following log: requested vs. actually-spawned per call.
        self.spawn_log: list[dict] = []

    @property
    def _max_dispatches(self) -> int:
        cap = self.config.subagent.max_dispatches
        if cap is not None:
            return cap
        return self.config.subagent.max_concurrent * self.config.budget.orchestrator_steps

    def _specs(self) -> list[ToolSpec]:
        specs = [*BASE_SPECS, FINISH_SPEC]
        if self.with_subagents:
            specs.insert(-1, SPAWN_SPEC)
        return specs

    def run(self, issue: str) -> str:
        system = orchestrator_system(self.config.subtask_type, self.with_subagents)
        messages: list[Message] = [Message(role="user", content=issue)]
        specs = self._specs()
        budget = self.config.budget
        final_summary = ""
        exit_reason = "max_steps"

        self.telemetry.start_episode()
        deadline = time.time() + budget.timeout_sec

        for step in range(1, budget.orchestrator_steps + 1):
            if time.time() > deadline:
                exit_reason = "timeout"
                break
            if self.telemetry.total_tokens >= budget.total_token_budget:
                exit_reason = "token_budget"
                break

            self.telemetry.record_orchestrator_step()
            resp = self.client.complete(system, messages, specs)
            self.telemetry.record_call(
                CallEvent(
                    role="orchestrator",
                    model=self.client.cfg.model,
                    agent_id="orchestrator",
                    step=step,
                    t_start=resp.t_start,
                    t_end=resp.t_end,
                    input_tokens=resp.usage.input_tokens,
                    output_tokens=resp.usage.output_tokens,
                    cache_read_tokens=resp.usage.cache_read_tokens,
                    cost_usd=resp.cost_usd,
                    n_tool_calls=len(resp.tool_calls),
                )
            )
            messages.append(
                Message(role="assistant", content=resp.text, tool_calls=resp.tool_calls)
            )

            if not resp.tool_calls:
                # Model stopped without finishing; nudge once by ending.
                exit_reason = "no_tool_call"
                final_summary = resp.text
                break

            results: list[ToolResult] = []
            finished = False
            for tc in resp.tool_calls:
                if tc.name == "finish":
                    final_summary = tc.arguments.get("summary", resp.text)
                    finished = True
                    exit_reason = "finished"
                    results.append(ToolResult(call_id=tc.id, content="Session ended."))
                    continue
                results.append(self._dispatch_tool(tc, step))
            messages.append(Message(role="tool", tool_results=results))
            if finished:
                break

        self.telemetry.end_episode(exit_reason)
        self.telemetry.spawn_log = self.spawn_log
        return final_summary

    # ---- tool dispatch -----------------------------------------------------

    def _dispatch_tool(self, tc: ToolCall, step: int) -> ToolResult:
        if tc.name == "spawn_subagents":
            return self._spawn(tc, step)
        fn = BASE_TOOLS.get(tc.name)
        if fn is None:
            return ToolResult(call_id=tc.id, content=f"[error] unknown tool: {tc.name}", is_error=True)
        try:
            return ToolResult(call_id=tc.id, content=fn(self.env, tc.arguments))
        except Exception as e:  # noqa: BLE001
            return ToolResult(call_id=tc.id, content=f"[error] {type(e).__name__}: {e}", is_error=True)

    def _spawn(self, tc: ToolCall, step: int) -> ToolResult:
        if not self.with_subagents:
            return ToolResult(call_id=tc.id, content="[error] subagents disabled in this config.", is_error=True)

        requested = list(tc.arguments.get("instructions", []))
        remaining = self._max_dispatches - self._dispatch_count
        if remaining <= 0:
            return ToolResult(
                call_id=tc.id,
                content=f"[error] subagent dispatch cap reached ({self._max_dispatches}).",
                is_error=True,
            )
        granted = requested[: min(remaining, self.config.subagent.max_concurrent)]

        self.spawn_log.append(
            {"step": step, "requested": len(requested), "granted": len(granted)}
        )

        results = self._run_subagents_parallel(granted)

        for r in results:
            is_redispatch = self._dispatch_count > 0
            self._dispatch_count += 1
            self.telemetry.record_dispatch(is_redispatch=is_redispatch, n_steps=r.steps_used)

        report = "\n\n".join(
            f"### Subagent {r.agent_id} "
            f"({'done' if r.completed else 'incomplete'}, {r.steps_used} steps)\n{r.report}"
            for r in results
        )
        if len(granted) < len(requested):
            report = (
                f"[note] requested {len(requested)} subagents; granted {len(granted)} "
                f"(concurrency/budget cap).\n\n" + report
            )
        return ToolResult(call_id=tc.id, content=report)

    # ------------------------------------------------------------------

    def _run_subagents_parallel(self, instructions: list[str]):
        assert self._sub_client is not None
        subagents = []
        for instr in instructions:
            self._sub_counter += 1
            sa = Subagent(
                agent_id=f"sub-{self._sub_counter}",
                client=self._sub_client,
                env=self.env,
                telemetry=self.telemetry,
                max_steps=self.config.subagent.max_steps,
            )
            subagents.append((sa, instr))

        if len(subagents) == 1:
            sa, instr = subagents[0]
            return [sa.run(instr)]

        with ThreadPoolExecutor(max_workers=len(subagents)) as pool:
            futures = [pool.submit(sa.run, instr) for sa, instr in subagents]
            return [f.result() for f in futures]
