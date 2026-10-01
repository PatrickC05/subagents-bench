"""A subagent: a self-contained tool-using worker.

A subagent runs its own bounded tool loop against the shared environment, using
the subagent model. It starts from a single instruction string (it cannot see
the orchestrator's conversation) and ends by calling ``submit`` with a report,
or by exhausting its step budget. Its token usage is attributed to the
"subagent" role in telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass

from .environment import Environment
from .llm import LLMClient, Message, ToolCall, ToolResult, ToolSpec
from .prompts import subagent_system
from .telemetry import CallEvent, Telemetry
from .tools import BASE_SPECS, BASE_TOOLS

SUBMIT_SPEC = ToolSpec(
    name="submit",
    description="Finish the task and return your report to the orchestrator.",
    parameters={
        "type": "object",
        "properties": {
            "report": {"type": "string", "description": "Concise summary of findings/changes."}
        },
        "required": ["report"],
    },
)


@dataclass
class SubagentResult:
    agent_id: str
    report: str
    steps_used: int
    completed: bool  # True if it called submit; False if it ran out of steps.


class Subagent:
    def __init__(
        self,
        agent_id: str,
        client: LLMClient,
        env: Environment,
        telemetry: Telemetry,
        max_steps: int,
    ):
        self.agent_id = agent_id
        self.client = client
        self.env = env
        self.telemetry = telemetry
        self.max_steps = max_steps
        self._specs = [*BASE_SPECS, SUBMIT_SPEC]

    def run(self, instruction: str) -> SubagentResult:
        system = subagent_system()
        messages: list[Message] = [Message(role="user", content=instruction)]
        report = ""
        completed = False
        step = 0

        while step < self.max_steps:
            step += 1
            resp = self.client.complete(system, messages, self._specs)
            self.telemetry.record_call(
                CallEvent(
                    role="subagent",
                    model=self.client.cfg.model,
                    agent_id=self.agent_id,
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
                # No tool call and no submit: treat text as the report.
                report = resp.text
                completed = True
                break

            results: list[ToolResult] = []
            did_submit = False
            for tc in resp.tool_calls:
                if tc.name == "submit":
                    report = tc.arguments.get("report", resp.text)
                    completed = True
                    did_submit = True
                    results.append(ToolResult(call_id=tc.id, content="ok"))
                    continue
                results.append(self._dispatch_tool(tc))
            messages.append(Message(role="tool", tool_results=results))
            if did_submit:
                break

        if not completed and not report:
            report = "(subagent exhausted its step budget without submitting)"
        return SubagentResult(
            agent_id=self.agent_id, report=report, steps_used=step, completed=completed
        )

    def _dispatch_tool(self, tc: ToolCall) -> ToolResult:
        fn = BASE_TOOLS.get(tc.name)
        if fn is None:
            return ToolResult(call_id=tc.id, content=f"[error] unknown tool: {tc.name}", is_error=True)
        try:
            out = fn(self.env, tc.arguments)
            return ToolResult(call_id=tc.id, content=out)
        except Exception as e:  # noqa: BLE001 - surface tool errors to the model
            return ToolResult(call_id=tc.id, content=f"[error] {type(e).__name__}: {e}", is_error=True)
