"""The orchestrator: a large model working a task in ``budget.orchestrator_steps``
steps, optionally delegating to subagents.

Each step is one LLM call followed by every requested tool, executed in
order. Tools: ``bash, edit_file, read_file, list_dir, search``, ``finish``,
and ``spawn_subagents`` when ``topology: subagents``. After the last tool
result of each step the runtime appends ``[step k/N complete; N-k remaining;
dispatch budget left: D]`` to that result (to the tool result rather than as
a separate user turn, so every provider accepts the message sequence). Tools
listed after ``finish`` in the same response are not run.

``spawn_subagents(instructions)`` grants (``docs/DESIGN.md`` section 4):

* parallel / sequential: ``min(requested, max_concurrent, remaining)``
  instructions, one subagent each, all concurrently / one at a time in order;
* arbiter: ``min(requested, max(1, max_concurrent // n), remaining // n)``
  instructions with ``n = n_candidates`` concurrent candidates each
  (:mod:`subagents_bench.arbiter`); arbitrations run concurrently.

The group's ``files_modified`` comes from a marker file made at group start
and ``find -newer`` at the end.

A reply without a tool call ends the episode (``no_tool_call``), except one
cut off at the output-token limit (``finish_reason`` ``length``): that step
is used, the orchestrator is told to be briefer and call a tool
(:data:`~subagents_bench.prompts.LENGTH_CUTOFF_NOTE`), and the episode goes on.

Exit reasons: ``finished``, ``max_steps``, ``no_tool_call``, ``token_budget``
and ``timeout`` (safety caps, checked before each step), ``cancelled`` (the
stop event), ``billing_error`` (recorded, then re-raised), ``provider_error``
and ``error``. A provider failure that survived every retry (a plain
:class:`LLMError`: retries exhausted, a non-retryable 4xx, a missing key)
ends the episode with ``provider_error`` without raising; the exception is
kept on :attr:`EpisodeResult.cause` so the harbor agent can raise
:class:`~subagents_bench.types.ProviderUnavailableError` after finalizing.
A :class:`ContextLengthError` of the orchestrator ends it with ``error`` (its
context outgrew the model: an outcome of the condition, not an outage). Any
other exception is a harness bug: it is recorded with its traceback and
re-raised (``raise_on_error=False`` suppresses that, for tests and callers
that only need the record). The episode is finalized in every case before
:meth:`Orchestrator.run` returns or raises.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import workspace
from .arbiter import ArbitrationOutcome, raise_first, run_arbitration
from .config import ExperimentConfig, Strategy, Topology
from .llm import make_client
from .prompts import LENGTH_CUTOFF_NOTE, LENGTH_CUTOFF_REPLY, cut_off, orchestrator_system_for, step_note
from .schema import SpawnRecord, StepRecord
from .subagent import DispatchResult, DispatchSpec, MeteredEnv, Runtime, Subagent, measure, text_output
from .telemetry import AgentLog, EpisodeRecorder, ToolEvent, is_provider_error
from .tools import run_tool, specs_for, truncate_output
from .types import (
    AgentStopped,
    BillingError,
    LLMError,
    Message,
    ToolCall,
    ToolContext,
    ToolOutput,
    ToolResult,
    ToolSpec,
)

log = logging.getLogger(__name__)

SPAWN_SPEC = ToolSpec(
    name="spawn_subagents",
    description=(
        "Delegate work to subagents. Each instruction in the list is given to a subagent (or, with the "
        "arbiter strategy, to several competing candidates) that works in the same container and returns a "
        "report. Subagents cannot see your conversation, so each instruction must be self-contained. "
        "Returns every report once all of them have finished. See the system prompt for the strategy, "
        "the subagents' tools and the limits."
    ),
    parameters={
        "type": "object",
        "properties": {
            "instructions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One self-contained instruction per subagent.",
            }
        },
        "required": ["instructions"],
    },
)

FINISH_SPEC = ToolSpec(
    name="finish",
    description=(
        "Declare the task complete and end the episode immediately. Tools listed after it in the same "
        "response are not run. The container's final state is then graded by automated tests."
    ),
    parameters={
        "type": "object",
        "properties": {"summary": {"type": "string", "description": "What you did and how you verified it."}},
        "required": [],
    },
)

# Episode-level files_modified cap (DESIGN: capped at 500).
EPISODE_FILES_CAP = 500
GROUP_FILES_CAP = 200
# Minimum chars kept per subagent report in the spawn result.
MIN_REPORT_CHARS = 2000


def llm_error_text(e: LLMError) -> str:
    """``'<ExceptionType>: message'``, with the HTTP status appended when the
    message does not already carry it."""
    text = f"{type(e).__name__}: {e}"
    if e.status is not None and f"HTTP {e.status}" not in text:
        text += f" [HTTP {e.status}]"
    return text


@dataclass
class EpisodeResult:
    exit_reason: str
    final_summary: str | None
    error: str | None
    episode_id: str
    episode_dir: Path
    record: dict[str, Any]
    recorder: EpisodeRecorder | None = None
    # The provider failure behind exit_reason "provider_error" (not raised).
    cause: LLMError | None = None


class Orchestrator:
    """Runs one episode. ``client_factory(model_config) -> client`` defaults
    to :func:`subagents_bench.llm.make_client`; one client is made per
    distinct model config and shared by every agent that uses it."""

    def __init__(
        self,
        config: ExperimentConfig,
        env: Any,
        recorder: EpisodeRecorder,
        *,
        workdir: str | None = None,
        stop_event: threading.Event | None = None,
        client_factory: Callable[[Any], Any] | None = None,
    ):
        self.config = config
        self.env = env
        self.recorder = recorder
        self.with_subagents = config.topology is Topology.SUBAGENTS
        self.stop_event = stop_event or getattr(env, "stop_event", None) or threading.Event()
        if getattr(env, "stop_event", None) is None:
            try:
                env.stop_event = self.stop_event
            except AttributeError:
                pass
        self.workdir = workdir or getattr(env, "workdir", None) or recorder.record.get("workdir") or "/"
        self._factory = client_factory or (lambda cfg: make_client(cfg))
        self._clients: dict[str, Any] = {}
        # Clients are made at the start of run(), inside its error handling,
        # so a construction failure is recorded like any other error.
        self.rt = Runtime(
            config=config,
            env=env,
            recorder=recorder,
            workdir=self.workdir,
            stop_event=self.stop_event,
            deadline=recorder.t_start + config.budget.timeout_sec,
            orchestrator_client=None,
            harness=MeteredEnv(env),
        )
        self.max_dispatches = config.resolved_max_dispatches()
        self.used_dispatches = 0
        self.log = AgentLog(recorder, "orchestrator", "orchestrator", stop_event=self.stop_event)
        self.ctx = ToolContext(
            env=env,
            agent_id="orchestrator",
            initial_cwd=self.workdir,
            max_output_chars=config.tools.max_output_chars,
            command_timeout_sec=config.budget.command_timeout_sec,
            max_command_timeout_sec=config.budget.max_command_timeout_sec,
        )
        self.own_tools = config.resolved_orchestrator_tools()
        self.specs = specs_for(
            self.own_tools,
            command_timeout_sec=config.budget.command_timeout_sec,
            max_command_timeout_sec=config.budget.max_command_timeout_sec,
        )
        if self.with_subagents:
            self.specs.append(SPAWN_SPEC)
        self.specs.append(FINISH_SPEC)
        self.messages: list[Message] = []
        self.system = orchestrator_system_for(config, self.workdir)

    def _make_clients(self) -> None:
        cfg = self.config
        self.rt.orchestrator_client = self._client(cfg.orchestrator)
        if self.with_subagents:
            self.rt.subagent_client = self._client(cfg.resolved_subagent_model())
            if cfg.subagent.strategy is Strategy.ARBITER:
                self.rt.arbiter_client = self._client(cfg.resolved_arbiter_model())

    def _client(self, cfg: Any) -> Any:
        key = json.dumps(dataclasses.asdict(cfg), sort_keys=True, default=str)
        if key not in self._clients:
            self._clients[key] = self._factory(cfg)
        return self._clients[key]

    @property
    def dispatches_left(self) -> int:
        return max(self.max_dispatches - self.used_dispatches, 0)

    # ---- the episode ------------------------------------------------------------

    def run(self, instruction: str, *, raise_on_error: bool = True) -> EpisodeResult:
        rec = self.recorder
        menv = self.rt.measure_env
        marker = measure(workspace.make_marker, menv, f"episode-{rec.episode_id}")
        exit_reason: str | None = None
        error: str | None = None
        final_summary: str | None = None
        pending: BaseException | None = None
        cause: LLMError | None = None
        last_text = ""
        try:
            self._make_clients()
            self.log.transcript(Message(role="system", content=self.system))
            first = Message(role="user", content=instruction)
            self.messages.append(first)
            self.log.transcript(first)
            for step in range(1, self.config.budget.orchestrator_steps + 1):
                cap = self.rt.cap_reason()
                if cap is not None:
                    exit_reason = cap
                    break
                outcome, text = self._step(step)
                if text and text.strip():
                    last_text = text
                if outcome == "finished":
                    exit_reason, final_summary = "finished", text
                    break
                if outcome == "no_tool_call":
                    exit_reason, final_summary = "no_tool_call", text
                    break
            else:
                exit_reason = "max_steps"
        except AgentStopped:
            exit_reason = "cancelled"
        except BillingError as e:
            exit_reason, error, pending = "billing_error", f"{type(e).__name__}: {e}", e
        except LLMError as e:
            error = llm_error_text(e)
            if is_provider_error(error):
                # The provider could not serve the orchestrator after every
                # retry: not an outcome of the condition. Not raised here (the
                # record is complete either way); the harbor agent raises
                # ProviderUnavailableError from `cause` after finalizing.
                exit_reason, cause = "provider_error", e
            else:
                # ContextLengthError (the orchestrator's context outgrew the
                # model: an outcome of the condition, graded like any other)
                # or another LLMError subclass.
                exit_reason = "error"
        except BaseException as e:  # noqa: BLE001 - recorded, then re-raised below
            exit_reason, error = "error", traceback.format_exc()
            if isinstance(e, KeyboardInterrupt):
                exit_reason = "cancelled"
            if raise_on_error or not isinstance(e, Exception):
                pending = e
        if final_summary is None and last_text:
            final_summary = last_text
        files = git_stat = None
        if marker is not None and not self.stop_event.is_set():
            files = measure(workspace.files_modified_since, menv, marker, self.workdir, EPISODE_FILES_CAP)
            git_stat = measure(workspace.git_diff_stat, menv, self.workdir)
        record = rec.finalize(
            exit_reason or "error",
            error=error,
            final_summary=final_summary,
            files_modified=files[0] if files else None,
            n_files_modified=files[1] if files else None,
            git_diff_stat=git_stat,
            **self.rt.harness_exec_fields(),
        )
        self._close_clients()
        if pending is not None:
            raise pending
        return EpisodeResult(
            exit_reason=record.get("exit_reason") or "error",
            final_summary=record.get("final_summary"),
            error=record.get("error"),
            episode_id=rec.episode_id,
            episode_dir=rec.dir,
            record=record,
            recorder=rec,
            cause=cause if record.get("exit_reason") == "provider_error" else None,
        )

    def _close_clients(self) -> None:
        for c in self._clients.values():
            close = getattr(c, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    log.debug("client close failed", exc_info=True)

    def _step(self, step: int) -> tuple[str, str]:
        """One step -> (outcome, text); outcome: continue | finished | no_tool_call
        (``continue`` also after a reply cut off at the output limit).

        The StepRecord is written even when the step raises.
        """
        t_step = time.time()
        t_llm_end: float | None = None
        call_id: str | None = None
        tool_names: list[str] = []
        groups: list[str] = []
        finished = False
        try:
            try:
                resp, call_id = self.log.llm(self.rt.orchestrator_client, self.system, self.messages, self.specs, step=step)
            finally:
                t_llm_end = time.time()
            content = resp.text or (LENGTH_CUTOFF_REPLY if cut_off(resp) else "")
            asst = Message(role="assistant", content=content, tool_calls=resp.tool_calls, reasoning=resp.reasoning or None)
            self.messages.append(asst)
            self.log.transcript(asst, call_id)
            tool_names = [tc.name for tc in resp.tool_calls]
            if not resp.tool_calls:
                if not cut_off(resp):
                    return "no_tool_call", resp.text
                note = step_note(
                    step, self.config.budget.orchestrator_steps, self.dispatches_left if self.with_subagents else None
                )
                cut = Message(role="user", content=f"{LENGTH_CUTOFF_NOTE}\n\n{note}")
                self.messages.append(cut)
                self.log.transcript(cut)
                return "continue", resp.text
            summary = resp.text
            results: list[ToolResult] = []
            for tc in resp.tool_calls:
                if finished:
                    results.append(ToolResult(tc.id, "[not run: finish was called earlier in this step]", True))
                    continue
                ev = ToolEvent()
                if tc.name == "finish":
                    out = self.log.tool(tc, lambda e: text_output("Episode finished."), step=step, call_id=call_id, event=ev)
                    s = tc.arguments.get("summary") if isinstance(tc.arguments, dict) else None
                    summary = s if isinstance(s, str) and s.strip() else (resp.text or "")
                    finished = True
                elif tc.name == "spawn_subagents" and self.with_subagents:
                    try:
                        out = self.log.tool(
                            tc, lambda e, tc=tc: self._spawn(tc, step, call_id, e), step=step, call_id=call_id, event=ev
                        )
                    finally:
                        if ev.group_id:
                            groups.append(ev.group_id)
                elif tc.name in self.own_tools:
                    out = self.log.tool(
                        tc, lambda e, tc=tc: run_tool(self.ctx, tc.name, tc.arguments), step=step, call_id=call_id, event=ev
                    )
                else:
                    avail = ", ".join(s.name for s in self.specs)
                    msg = f"unknown tool {tc.name!r}; available: {avail}"
                    out = self.log.tool(
                        tc, lambda e, m=msg: text_output(f"[error] {m}", is_error=True, error=m),
                        step=step, call_id=call_id, event=ev,
                    )
                results.append(ToolResult(call_id=tc.id, content=out.text, is_error=out.is_error))
                if self.stop_event.is_set():
                    raise AgentStopped("episode stopped")
            note = step_note(
                step, self.config.budget.orchestrator_steps, self.dispatches_left if self.with_subagents else None
            )
            last = results[-1]
            results[-1] = ToolResult(last.call_id, f"{last.content}\n\n{note}", last.is_error)
            tool_msg = Message(role="tool", tool_results=results)
            self.messages.append(tool_msg)
            self.log.transcript(tool_msg)
            return ("finished" if finished else "continue"), (summary if finished else resp.text)
        finally:
            t_end = time.time()
            llm_end = t_llm_end if t_llm_end is not None else t_end
            self.recorder.record_step(
                StepRecord(
                    episode_id=self.recorder.episode_id,
                    step=step,
                    call_id=call_id,
                    t_start=t_step,
                    t_end=t_end,
                    duration_sec=t_end - t_step,
                    llm_sec=llm_end - t_step,
                    tools_sec=t_end - llm_end,
                    n_tool_calls=len(tool_names),
                    tool_names=tool_names,
                    spawned_group_ids=groups,
                    finished=finished,
                )
            )

    # ---- spawn_subagents ---------------------------------------------------------

    def _grant(self, requested: int) -> tuple[int, int, str | None, int]:
        """(granted instructions, dispatches per instruction, cap_reason, concurrency)."""
        s = self.config.subagent
        remaining = self.dispatches_left
        if s.strategy is Strategy.ARBITER:
            n = s.arbiter.n_candidates
            by_conc = max(1, s.max_concurrent // n)
            by_budget = remaining // n
            granted = min(requested, by_conc, by_budget)
            reason = None
            if granted < requested:
                reason = "max_dispatches" if by_budget < min(requested, by_conc) else "max_concurrent"
            return granted, n, reason, granted * n
        granted = min(requested, s.max_concurrent, remaining)
        reason = None
        if granted < requested:
            reason = "max_dispatches" if remaining < min(requested, s.max_concurrent) else "max_concurrent"
        conc = 1 if s.strategy is Strategy.SEQUENTIAL else granted
        return granted, 1, reason, conc

    def _spawn(self, tc: ToolCall, step: int, call_id: str, ev: ToolEvent) -> ToolOutput:
        raw = tc.arguments.get("instructions") if isinstance(tc.arguments, dict) else None
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list) or not raw or not all(isinstance(x, str) and x.strip() for x in raw):
            msg = "'instructions' must be a non-empty list of non-empty strings"
            return text_output(f"[error] {msg}", is_error=True, error=msg)
        requested = list(raw)
        s = self.config.subagent
        strategy = s.strategy
        granted, per_instr, cap_reason, concurrency = self._grant(len(requested))
        group_id = self.recorder.new_id("group")
        ev.group_id = group_id
        menv = MeteredEnv(self.env)
        # Group measurement commands count on the spawn tool (menv) and as harness measurements.
        mmenv = MeteredEnv(menv, also=self.rt.harness)
        t_start = time.time()
        marker = measure(workspace.make_marker, mmenv, f"group-{group_id}")
        self.used_dispatches += granted * per_instr
        dispatch_ids: list[str] = []
        arbitration_ids: list[str] = []
        ids_lock = threading.Lock()
        results: list[DispatchResult] = []
        outcomes: list[ArbitrationOutcome | None] = []
        files = None
        try:
            if granted == 0:
                pass
            elif strategy is Strategy.ARBITER:
                outcomes = self._run_arbitrations(requested[:granted], group_id, step, menv, dispatch_ids, arbitration_ids, ids_lock, t_start)
            else:
                subs = [
                    Subagent(
                        self.rt,
                        DispatchSpec(
                            instruction=instr,
                            instruction_index=i,
                            group_id=group_id,
                            orchestrator_step=step,
                            strategy=strategy.value,
                            t_queued=t_start,
                            initial_cwd=self.workdir,
                        ),
                    )
                    for i, instr in enumerate(requested[:granted])
                ]
                dispatch_ids.extend(sa.dispatch_id for sa in subs)
                if strategy is Strategy.PARALLEL:
                    with ThreadPoolExecutor(max_workers=len(subs)) as pool:
                        futs = [pool.submit(sa.run) for sa in subs]
                        errors: list[BaseException | None] = []
                        for f in futs:
                            try:
                                results.append(f.result())
                                errors.append(None)
                            except BaseException as e:  # noqa: BLE001 - re-raised below
                                errors.append(e)
                    raise_first(errors)
                else:
                    for k, sa in enumerate(subs):
                        try:
                            results.append(sa.run())
                        except BillingError:
                            for rest in subs[k + 1 :]:
                                rest.skip("cancelled", "not run: an earlier subagent in this group hit a billing error")
                            raise
        finally:
            t_members_end = time.time()
            if marker is not None:
                files = measure(workspace.files_modified_since, mmenv, marker, self.workdir, GROUP_FILES_CAP)
            t_end = time.time()
            with self.recorder._lock:  # noqa: SLF001 - snapshot of this group's dispatches
                member_recs = [d for d in self.recorder.dispatches if d["group_id"] == group_id]
            self.recorder.record_spawn(
                SpawnRecord(
                    episode_id=self.recorder.episode_id,
                    group_id=group_id,
                    orchestrator_step=step,
                    call_id=call_id,
                    tool_call_id=tc.id,
                    strategy=strategy.value,
                    scope=s.scope.value,
                    requested=len(requested),
                    granted=granted,
                    cap_reason=cap_reason,
                    dispatches_remaining_after=self.dispatches_left,
                    instructions=requested,
                    dispatch_ids=list(dispatch_ids),
                    arbitration_ids=list(arbitration_ids),
                    concurrency=concurrency,
                    t_start=t_start,
                    t_end=t_end,
                    duration_sec=t_end - t_start,
                    sum_dispatch_sec=float(sum(d["duration_sec"] for d in member_recs)),
                    files_modified=files[0] if files else None,
                    n_files_modified=files[1] if files else None,
                )
            )
            log.debug("group %s members done in %.2fs", group_id, t_members_end - t_start)
        text, raw_chars, truncated = self._spawn_text(requested, granted, cap_reason, strategy, results, outcomes)
        return ToolOutput(
            text=text,
            output_chars_raw=raw_chars,
            truncated=truncated,
            n_exec=menv.n,
            exec_sec=menv.sec,
        )

    def _run_arbitrations(
        self,
        instructions: list[str],
        group_id: str,
        step: int,
        menv: MeteredEnv,
        dispatch_ids: list[str],
        arbitration_ids: list[str],
        ids_lock: threading.Lock,
        t_queued: float,
    ) -> list[ArbitrationOutcome | None]:
        def register(did: str) -> None:
            with ids_lock:
                dispatch_ids.append(did)

        def one(i: int, instr: str) -> ArbitrationOutcome:
            out = run_arbitration(
                self.rt,
                menv=menv,
                group_id=group_id,
                instruction=instr,
                instruction_index=i,
                orchestrator_step=step,
                register_dispatch=register,
                t_queued=t_queued,
            )
            return out

        outcomes: list[ArbitrationOutcome | None] = []
        errors: list[BaseException | None] = []
        with ThreadPoolExecutor(max_workers=len(instructions)) as pool:
            futs = [pool.submit(one, i, instr) for i, instr in enumerate(instructions)]
            for f in futs:
                try:
                    outcomes.append(f.result())
                    errors.append(None)
                except BaseException as e:  # noqa: BLE001 - re-raised below
                    outcomes.append(None)
                    errors.append(e)
        # Arbitration ids in instruction order (from the records written).
        with self.recorder._lock:  # noqa: SLF001
            arbs = sorted(
                (a for a in self.recorder.arbitrations if a["group_id"] == group_id),
                key=lambda a: a["instruction_index"],
            )
        arbitration_ids.extend(a["arbitration_id"] for a in arbs)
        with ids_lock:
            # Candidates register concurrently; list them by instruction, then candidate.
            order = {}
            for a in arbs:
                for k, did in enumerate(a["candidate_dispatch_ids"]):
                    order[did] = (a["instruction_index"], k)
            dispatch_ids.sort(key=lambda d: order.get(d, (1 << 30, 0)))
        raise_first(errors)
        return outcomes

    def _spawn_text(
        self,
        requested: list[str],
        granted: int,
        cap_reason: str | None,
        strategy: Strategy,
        results: list[DispatchResult],
        outcomes: list[ArbitrationOutcome | None],
    ) -> tuple[str, int, bool]:
        """The spawn_subagents result shown to the orchestrator -> (text, raw chars, truncated)."""
        s = self.config.subagent
        header = f"[spawn_subagents: strategy {strategy.value}, scope {s.scope.value}; ran {granted} of {len(requested)} instruction(s)"
        if strategy is Strategy.ARBITER:
            header += f" with {s.arbiter.n_candidates} candidates each"
        header += f"; dispatch budget left: {self.dispatches_left}]"
        parts = [header]
        if granted < len(requested):
            limit = {
                "max_concurrent": "the per-call limit",
                "max_dispatches": "the episode's remaining dispatch budget",
            }.get(cap_reason or "", "a limit")
            skipped = ", ".join(str(i + 1) for i in range(granted, len(requested)))
            parts.append(
                f"[note: you requested {len(requested)} instruction(s) but only {granted} could run because of "
                f"{limit}; instruction(s) {skipped} were NOT run]"
            )
        per_report = max(MIN_REPORT_CHARS, self.config.tools.max_output_chars // max(granted, 1))
        raw_total, shown_total, truncated_any = 0, 0, False

        def cap(text: str) -> str:
            nonlocal raw_total, shown_total, truncated_any
            out, t = truncate_output(text, per_report)
            raw_total += len(text)
            shown_total += len(out)
            truncated_any = truncated_any or t
            return out

        if strategy is Strategy.ARBITER:
            for i, o in enumerate(outcomes):
                if o is None:
                    parts.append(f"### Instruction {i + 1}\n(arbitration did not complete)")
                    continue
                conf = f"{o.confidence:.2f}" if o.confidence is not None else "unknown (judge output unparseable)"
                agreed = {True: "yes", False: "no", None: "unknown"}[o.agreement]
                head = (
                    f"### Instruction {i + 1} (arbiter: candidate {o.winner + 1} of {len(o.candidates) or s.arbiter.n_candidates}, "
                    f"confidence {conf}, candidates agreed: {agreed})"
                )
                body = cap(o.report)
                tail = ""
                if o.isolation == "copy":
                    if o.applied_files:
                        tail = "\napplied files: " + ", ".join(o.applied_files[:50])
                        if len(o.applied_files) > 50:
                            tail += f" (+{len(o.applied_files) - 50} more)"
                    else:
                        tail = "\napplied files: (none: the winner changed no files)"
                    if o.apply_error:
                        tail += f"\n[apply error: {o.apply_error}]"
                elif o.isolation == "shared_fallback":
                    tail = "\n(candidates shared the working directory: it was too large to copy)"
                parts.append(f"{head}\n{body}{tail}")
        else:
            for i, r in enumerate(results):
                parts.append(f"### Instruction {i + 1} ({r.agent_id}: {r.status}, {r.steps_used} steps)\n{cap(r.report)}")
        text = "\n\n".join(parts)
        # Raw size = the text as it would be without per-report truncation.
        return text, len(text) - shown_total + raw_total, truncated_any
