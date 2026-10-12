"""Subagents: bounded tool loops run by a (usually smaller) model.

A subagent starts from one instruction string (it cannot see the
orchestrator's conversation), works with its scope's tool set
(:data:`subagents_bench.tools.SCOPE_TOOLS`) plus ``submit(report)``, and ends
with one of these statuses (written to ``dispatches.jsonl``):

* ``submitted`` -- it called ``submit``;
* ``no_tool_call`` -- it answered without a tool call (the text is the report);
* ``max_steps`` -- it used every step (the last assistant text is the report);
* ``context_overflow`` -- the provider rejected the prompt as too long even
  after an aggressive trim;
* ``error`` -- a provider error after retries (its failed call counts in
  ``EpisodeRecord.n_provider_errors`` and the dispatch ``error`` starts with
  ``LLMError:``), or a harness bug;
* ``cancelled`` -- the episode's stop event fired;
* ``token_budget`` / ``timeout`` -- the episode's token or wall-clock safety
  cap (or the per-subagent ``token_budget``) was reached before a step.

**Context guard** (models with ``context_window``): before each call the
prompt is estimated as the last reported ``prompt_tokens`` plus
``chars added since / 3``. If estimate + ``max_output_tokens`` exceeds 92% of
the window, the oldest tool-result contents (all but the latest two) are
replaced by ``[output elided to fit context: N chars]`` until it fits; each
such pass is one ``context_trims`` and its elided chars go on the next
CallRecord as ``context_trimmed_chars``. On ``ContextLengthError`` one
aggressive trim (only the latest tool result kept) and one retry are made.

``files_modified`` of a dispatch = its ``edit_file`` targets plus files under
its start directory (the workdir, or its private copy) whose mtime is newer
than a marker made when it started (``find -newer``). With concurrent
subagents sharing the workdir this window also catches siblings' changes, so
it over-approximates for the parallel strategy; it is exact for sequential
runs and private copies. Easy scope (read-only tools) is not scanned.

This module also holds :class:`Runtime`, the per-episode state every agent
shares (config, environment, recorder, stop event, budgets, LLM clients), and
:class:`MeteredEnv`, which counts the exec round-trips the runtime itself makes
(markers, scans, copies) so they can be reported on the tool that caused them.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from . import workspace
from .config import ExperimentConfig, Scope
from .prompts import LAST_STEP_NOTE, subagent_system
from .telemetry import AgentLog, EpisodeRecorder, sum_or_none
from .tools import SCOPE_TOOLS, run_tool, specs_for
from .types import (
    AgentStopped,
    BillingError,
    ContextLengthError,
    LLMError,
    Message,
    ToolCall,
    ToolContext,
    ToolOutput,
    ToolResult,
    ToolSpec,
)

log = logging.getLogger(__name__)

SUBMIT_SPEC = ToolSpec(
    name="submit",
    description=(
        "Finish your work and return your report to the orchestrator. The orchestrator sees only this "
        "report, so include what you found or changed, file paths, commands run and key results."
    ),
    parameters={
        "type": "object",
        "properties": {"report": {"type": "string", "description": "Your concise, concrete report."}},
        "required": ["report"],
    },
)

# Fraction of the context window the guard keeps the request under.
CONTEXT_GUARD_FRACTION = 0.92
# Chars per token assumed by the guard's estimate.
GUARD_CHARS_PER_TOKEN = 3.0
# Tool results the guard never elides (the latest ones).
GUARD_KEEP_LATEST = 2
ELIDED_PREFIX = "[output elided to fit context:"


class MeteredEnv:
    """Environment proxy counting exec round-trips and their seconds.

    Used for the runtime's own helper commands (markers, file scans, copies,
    applies); every other attribute is forwarded to the wrapped environment.
    ``also`` is another meter that every exec through this one is added to
    as well (e.g. the episode-wide measurement meter).

    ``sec`` is the plain sum of the execs' durations; :meth:`union_sec` is the
    wall-clock time covered by at least one of them, which is less when
    execs from concurrent threads (parallel dispatches, arbiter candidates)
    overlap.
    """

    def __init__(self, env: Any, also: "MeteredEnv | None" = None):
        self._env = env
        self._also = also
        self._lock = threading.Lock()
        self.n = 0
        self.sec = 0.0
        # (start, end) time.monotonic() of every exec, for union_sec().
        self._intervals: list[tuple[float, float]] = []

    def add(self, n: int, sec: float, interval: tuple[float, float] | None = None) -> None:
        with self._lock:
            self.n += n
            self.sec += sec
            if interval is not None:
                self._intervals.append(interval)

    def union_sec(self) -> float:
        """Seconds during which at least one exec ran (overlaps counted once)."""
        with self._lock:
            intervals = sorted(self._intervals)
        total = 0.0
        cur: list[float] | None = None
        for t0, t1 in intervals:
            if cur is None or t0 > cur[1]:
                if cur is not None:
                    total += cur[1] - cur[0]
                cur = [t0, t1]
            else:
                cur[1] = max(cur[1], t1)
        if cur is not None:
            total += cur[1] - cur[0]
        return total

    def exec(self, command: str, timeout_sec: float | None = None, cwd: str | None = None) -> Any:
        t0 = time.monotonic()
        try:
            if cwd is None:
                return self._env.exec(command, timeout_sec=timeout_sec)
            return self._env.exec(command, timeout_sec=timeout_sec, cwd=cwd)
        finally:
            t1 = time.monotonic()
            self.add(1, t1 - t0, (t0, t1))
            if self._also is not None:
                self._also.add(1, t1 - t0, (t0, t1))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


def text_output(text: str, *, is_error: bool = False, error: str | None = None) -> ToolOutput:
    """A runtime-generated tool result (no exec): raw size = returned size."""
    return ToolOutput(text=text, is_error=is_error, error=error, output_chars_raw=len(text))


def measure(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a measurement helper; None if it fails or the episode stopped."""
    try:
        return fn(*args, **kwargs)
    except AgentStopped:
        return None
    except Exception:  # noqa: BLE001 - measurements never break an episode
        log.debug("measurement %s failed", getattr(fn, "__name__", fn), exc_info=True)
        return None


@dataclass
class Runtime:
    """Per-episode state shared by the orchestrator, subagents and arbiter."""

    config: ExperimentConfig
    env: Any
    recorder: EpisodeRecorder
    workdir: str
    stop_event: threading.Event
    deadline: float  # time.time() at which the timeout safety cap fires
    orchestrator_client: Any
    subagent_client: Any = None
    arbiter_client: Any = None
    # Meter for the harness's own measurement commands (markers, modified-file
    # scans, git diff --stat): EpisodeRecord.harness_exec_n, harness_exec_sec
    # (union_sec) and harness_exec_sum_sec (sec).
    harness: MeteredEnv | None = None

    def harness_exec_fields(self) -> dict[str, Any]:
        """The EpisodeRecord harness_exec_* values (None when not metered)."""
        h = self.harness
        if h is None:
            return {"harness_exec_n": None, "harness_exec_sec": None, "harness_exec_sum_sec": None}
        return {"harness_exec_n": h.n, "harness_exec_sec": h.union_sec(), "harness_exec_sum_sec": h.sec}

    @property
    def measure_env(self) -> Any:
        """The environment measurement helpers should use (metered if possible)."""
        return self.harness if self.harness is not None else self.env

    def stopped(self) -> bool:
        return self.stop_event.is_set()

    def cap_reason(self) -> str | None:
        """'cancelled' | 'timeout' | 'token_budget' | None (in that priority)."""
        if self.stop_event.is_set():
            return "cancelled"
        if time.time() >= self.deadline:
            return "timeout"
        if self.recorder.total_tokens() >= self.config.budget.total_token_budget:
            return "token_budget"
        return None


@dataclass
class DispatchSpec:
    """What one subagent run is asked to do and where."""

    instruction: str
    instruction_index: int
    group_id: str
    orchestrator_step: int
    strategy: str
    t_queued: float
    initial_cwd: str
    isolation: str = "shared"  # shared | copy | shared_fallback
    private_copy: str | None = None
    arbitration_id: str | None = None
    candidate_index: int | None = None


@dataclass
class DispatchResult:
    dispatch_id: str
    agent_id: str
    status: str
    report: str
    steps_used: int
    t_start: float
    t_end: float
    error: str | None = None
    record: dict[str, Any] = field(default_factory=dict)


def _chars(obj: Any) -> int:
    return len(json.dumps(obj, default=str, ensure_ascii=False))


def _message_chars(m: Message) -> int:
    n = len(m.content or "") + len(m.reasoning or "")
    n += sum(len(tc.name) + _chars(tc.arguments) for tc in m.tool_calls)
    n += sum(len(tr.content) for tr in m.tool_results)
    return n


class Subagent:
    """One dispatch. Ids are allocated at construction (queue time)."""

    def __init__(self, rt: Runtime, spec: DispatchSpec):
        self.rt = rt
        self.spec = spec
        cfg = rt.config
        self.scope: Scope = cfg.subagent.scope
        self.max_steps = cfg.subagent.max_steps
        self.agent_id = f"sub-{rt.recorder.next_number('sub')}"
        self.dispatch_id = rt.recorder.new_id("dispatch")
        self.client = rt.subagent_client
        self.tool_names = list(SCOPE_TOOLS[self.scope.value])
        self.specs = specs_for(
            self.tool_names,
            command_timeout_sec=cfg.budget.command_timeout_sec,
            max_command_timeout_sec=cfg.budget.max_command_timeout_sec,
        ) + [SUBMIT_SPEC]
        self.log = AgentLog(
            rt.recorder,
            self.agent_id,
            "subagent",
            dispatch_id=self.dispatch_id,
            group_id=spec.group_id,
            arbitration_id=spec.arbitration_id,
            stop_event=rt.stop_event,
        )
        self.ctx = ToolContext(
            env=rt.env,
            agent_id=self.agent_id,
            initial_cwd=spec.initial_cwd,
            max_output_chars=cfg.subagent.max_output_chars,
            command_timeout_sec=cfg.budget.command_timeout_sec,
            max_command_timeout_sec=cfg.budget.max_command_timeout_sec,
        )
        self.messages: list[Message] = []
        self.system = subagent_system(
            self.scope,
            max_steps=self.max_steps,
            workdir=spec.initial_cwd,
            private_copy=spec.private_copy,
            original_workdir=rt.workdir if spec.private_copy else None,
            command_timeout_sec=cfg.budget.command_timeout_sec,
            max_command_timeout_sec=cfg.budget.max_command_timeout_sec,
        )
        # Context guard state.
        self.context_trims = 0
        self._pending_trimmed = 0
        self._last_prompt_tokens = 0
        self._chars_since = 0
        self._files_written: set[str] = set()

    # ---- context guard ---------------------------------------------------------

    def _context_window(self) -> int | None:
        return getattr(self.client.cfg, "context_window", None)

    def _estimate(self) -> float:
        return self._last_prompt_tokens + self._chars_since / GUARD_CHARS_PER_TOKEN

    def _tool_result_slots(self) -> list[tuple[int, int]]:
        slots = []
        for mi, m in enumerate(self.messages):
            for ri, tr in enumerate(m.tool_results):
                if not tr.content.startswith(ELIDED_PREFIX):
                    slots.append((mi, ri))
        return slots

    def _elide(self, slot: tuple[int, int]) -> int:
        mi, ri = slot
        m = self.messages[mi]
        old = m.tool_results[ri]
        new_text = f"{ELIDED_PREFIX} {len(old.content)} chars]"
        results = list(m.tool_results)
        results[ri] = ToolResult(call_id=old.call_id, content=new_text, is_error=old.is_error)
        self.messages[mi] = Message(role=m.role, content=m.content, tool_calls=m.tool_calls, tool_results=results, reasoning=m.reasoning)
        return max(len(old.content) - len(new_text), 0)

    def _guard(self) -> None:
        """Elide old tool output until the estimated request fits."""
        cw = self._context_window()
        if not cw:
            return
        limit = CONTEXT_GUARD_FRACTION * cw
        max_tokens = int(getattr(self.client.cfg, "max_output_tokens", 0) or 0)
        if self._estimate() + max_tokens <= limit:
            return
        slots = self._tool_result_slots()
        elidable = slots[:-GUARD_KEEP_LATEST] if len(slots) > GUARD_KEEP_LATEST else []
        trimmed = 0
        for slot in elidable:
            if self._estimate() - trimmed / GUARD_CHARS_PER_TOKEN + max_tokens <= limit:
                break
            trimmed += self._elide(slot)
        if trimmed:
            self.context_trims += 1
            self._pending_trimmed += trimmed
            self._chars_since -= trimmed

    def _aggressive_trim(self) -> int:
        """Keep only the latest tool result (after a ContextLengthError)."""
        slots = self._tool_result_slots()
        trimmed = sum(self._elide(s) for s in slots[:-1])
        if trimmed:
            self.context_trims += 1
            self._pending_trimmed += trimmed
        return trimmed

    # ---- the loop ------------------------------------------------------------------

    def _append(self, m: Message, call_id: str | None = None) -> None:
        self.messages.append(m)
        self._chars_since += _message_chars(m)
        self.log.transcript(m, call_id)

    def _call(self, step: int) -> tuple[Any, str]:
        """One LLM step with the context guard and the overflow retry."""
        self._guard()
        trimmed, self._pending_trimmed = self._pending_trimmed, 0
        try:
            return self.log.llm(self.client, self.system, self.messages, self.specs, step=step, context_trimmed_chars=trimmed)
        except ContextLengthError:
            if self._aggressive_trim() == 0:
                raise
        trimmed, self._pending_trimmed = self._pending_trimmed, 0
        return self.log.llm(self.client, self.system, self.messages, self.specs, step=step, context_trimmed_chars=trimmed)

    def _own_tokens(self) -> int:
        rec = self.rt.recorder
        with rec._lock:  # noqa: SLF001 - read-only snapshot of our own calls
            return sum(
                c["prompt_tokens"] + c["completion_tokens"]
                for c in rec.calls
                if c["dispatch_id"] == self.dispatch_id and c["role"] == "subagent"
            )

    def _cap_status(self) -> str | None:
        reason = self.rt.cap_reason()
        if reason == "cancelled":
            raise AgentStopped("episode stopped")
        if reason is not None:
            return reason
        tb = self.rt.config.subagent.token_budget
        if tb is not None and self._own_tokens() >= tb:
            return "token_budget"
        return None

    def _run_tool(self, tc: ToolCall, step: int, call_id: str) -> ToolOutput:
        if tc.name in self.tool_names:
            out = self.log.tool(tc, lambda ev: run_tool(self.ctx, tc.name, tc.arguments), step=step, call_id=call_id)
            self._files_written.update(out.files_written)
            return out
        avail = ", ".join(self.tool_names + ["submit"])
        msg = f"tool {tc.name!r} is not available to you (scope {self.scope.value}); available: {avail}"
        return self.log.tool(
            tc, lambda ev: text_output(f"[error] {msg}", is_error=True, error=msg), step=step, call_id=call_id
        )

    def run(self) -> DispatchResult:
        """Run to completion and write the DispatchRecord. Re-raises only
        BillingError (after recording); every other failure is a status."""
        spec, rt = self.spec, self.rt
        scan_root = spec.private_copy or rt.workdir
        # Easy scope has only read-only tools: nothing to detect, so skip the
        # marker/scan round-trips (files_modified is then [] by construction).
        read_only = "bash" not in self.tool_names and "edit_file" not in self.tool_names
        marker = None if read_only else measure(workspace.make_marker, rt.measure_env, f"dispatch-{self.dispatch_id}")
        t_start = time.time()
        status, report, error = "max_steps", "", None
        steps_used = 0
        last_text = ""
        billing: BillingError | None = None
        try:
            # The system prompt goes to the transcript only (clients take it separately).
            self.log.transcript(Message(role="system", content=self.system))
            self._chars_since = len(self.system) + sum(
                len(s.name) + len(s.description) + _chars(s.parameters) for s in self.specs
            )
            self._append(Message(role="user", content=spec.instruction))
            for step in range(1, self.max_steps + 1):
                cap = self._cap_status()
                if cap is not None:
                    status = cap
                    break
                steps_used = step
                try:
                    resp, call_id = self._call(step)
                except ContextLengthError as e:
                    status, error = "context_overflow", f"{type(e).__name__}: {e}"
                    break
                if resp.usage is not None:
                    self._last_prompt_tokens = resp.usage.prompt_tokens
                self._chars_since = 0
                self._append(
                    Message(role="assistant", content=resp.text, tool_calls=resp.tool_calls, reasoning=resp.reasoning or None),
                    call_id,
                )
                if resp.text.strip():
                    last_text = resp.text
                if not resp.tool_calls:
                    status, report = "no_tool_call", resp.text
                    break
                results: list[ToolResult] = []
                submitted = False
                for tc in resp.tool_calls:
                    if submitted:
                        results.append(ToolResult(tc.id, "[not run: submit was called earlier in this step]", True))
                        continue
                    if tc.name == "submit":
                        rep = tc.arguments.get("report") if isinstance(tc.arguments, dict) else None
                        report = rep if isinstance(rep, str) else (json.dumps(rep) if rep is not None else resp.text)
                        out = self.log.tool(tc, lambda ev: text_output("Report submitted."), step=step, call_id=call_id)
                        submitted = True
                    else:
                        out = self._run_tool(tc, step, call_id)
                    results.append(ToolResult(call_id=tc.id, content=out.text, is_error=out.is_error))
                    if rt.stopped():
                        raise AgentStopped("episode stopped")
                if not submitted and step == self.max_steps - 1 and results:
                    last = results[-1]
                    results[-1] = ToolResult(last.call_id, f"{last.content}\n\n{LAST_STEP_NOTE}", last.is_error)
                self._append(Message(role="tool", tool_results=results))
                if submitted:
                    status = "submitted"
                    break
        except AgentStopped:
            status = "cancelled"
        except BillingError as e:
            status, error, billing = "error", f"{type(e).__name__}: {e}", e
        except LLMError as e:
            status, error = "error", f"{type(e).__name__}: {e}"
        except Exception:  # noqa: BLE001 - a harness bug fails this dispatch, not the episode
            status, error = "error", traceback.format_exc()[-4000:]
            log.exception("subagent %s crashed", self.agent_id)
        t_end = time.time()
        if status == "max_steps" and not report:
            report = last_text or f"(no report: the subagent used all {self.max_steps} steps without calling submit)"
        elif status not in ("submitted", "no_tool_call") and not report:
            report = last_text or f"(no report: the subagent ended with status {status})"
        files: tuple[list[str], int] | None = ([], 0) if read_only else None
        if marker is not None:
            files = measure(workspace.files_modified_since, rt.measure_env, marker, scan_root, 200)
        rec = self._write_record(status, report, error, steps_used, t_start, t_end, files)
        result = DispatchResult(
            dispatch_id=self.dispatch_id,
            agent_id=self.agent_id,
            status=status,
            report=report,
            steps_used=steps_used,
            t_start=t_start,
            t_end=t_end,
            error=error,
            record=rec or {},
        )
        if billing is not None:
            raise billing
        return result

    def skip(self, status: str, error: str) -> DispatchResult:
        """Record this dispatch as never run (its group was aborted first)."""
        now = time.time()
        rec = self._write_record(status, f"(not run: {error})", error, 0, now, now, None)
        return DispatchResult(
            dispatch_id=self.dispatch_id, agent_id=self.agent_id, status=status, report="", steps_used=0,
            t_start=now, t_end=now, error=error, record=rec or {},
        )

    def _write_record(
        self, status: str, report: str, error: str | None, steps_used: int, t_start: float, t_end: float,
        files: tuple[list[str], int] | None,
    ) -> dict[str, Any] | None:
        from .schema import DispatchRecord  # noqa: PLC0415

        rec = self.rt.recorder
        with rec._lock:  # noqa: SLF001 - consistent snapshot of this dispatch's events
            calls = [c for c in rec.calls if c["dispatch_id"] == self.dispatch_id and c["role"] == "subagent"]
            tools = [t for t in rec.tools if t["dispatch_id"] == self.dispatch_id and t["role"] == "subagent"]
        work_tools = [t for t in tools if t["name"] != "submit"]
        counts: dict[str, int] = {}
        for t in work_tools:
            counts[t["name"]] = counts.get(t["name"], 0) + 1
        reasoning = [c["reasoning_tokens"] for c in calls if c["reasoning_tokens"] is not None]
        if files is not None:
            found, n_total = files
            merged = sorted(set(found) | self._files_written)
            files_list: list[str] | None = merged[:200]
            n_files: int | None = max(n_total, len(merged))
        else:
            files_list, n_files = None, None
        spec = self.spec
        return rec.record_dispatch(
            DispatchRecord(
                episode_id=rec.episode_id,
                dispatch_id=self.dispatch_id,
                agent_id=self.agent_id,
                group_id=spec.group_id,
                arbitration_id=spec.arbitration_id,
                candidate_index=spec.candidate_index,
                instruction_index=spec.instruction_index,
                orchestrator_step=spec.orchestrator_step,
                strategy=spec.strategy,
                scope=self.scope.value,
                model=self.client.cfg.model,
                tools=self.tool_names + ["submit"],
                instruction=spec.instruction,
                report=report,
                status=status,
                steps_used=steps_used,
                max_steps=self.max_steps,
                t_queued=spec.t_queued,
                t_start=t_start,
                t_end=t_end,
                duration_sec=t_end - t_start,
                queue_sec=t_start - spec.t_queued,
                n_llm_calls=len(calls),
                prompt_tokens=sum(c["prompt_tokens"] for c in calls),
                completion_tokens=sum(c["completion_tokens"] for c in calls),
                cached_tokens=sum(c["cached_tokens"] for c in calls),
                reasoning_tokens=sum(reasoning) if reasoning else None,
                cost_usd=sum_or_none(c["cost_usd"] for c in calls),
                llm_sec=float(sum(c["latency_sec"] for c in calls)),
                tool_sec=float(sum(t["duration_sec"] for t in tools)),
                n_tool_calls=len(work_tools),
                tool_counts=counts,
                n_tool_errors=sum(1 for t in work_tools if t["is_error"]),
                n_timeouts=sum(1 for t in work_tools if t["timed_out"]),
                context_trims=self.context_trims,
                workdir=spec.initial_cwd,
                isolation=spec.isolation,
                files_modified=files_list,
                n_files_modified=n_files,
                error=error,
            )
        )
