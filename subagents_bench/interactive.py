"""Interactive episodes: a timed session in which the task's subtasks arrive as user messages.

The first user message is the task's instruction plus hint 0; hint k arrives at ``k *
interval_sec``. There is no step budget to plan around (``budget.orchestrator_steps`` is only a
safety cap) and no ``finish``: the session ends at ``wall_clock_sec`` (exit ``deadline``), or,
with ``end_when_idle``, once every hint was sent, no subagent runs and the orchestrator waits
(exit ``idle``). Messages are delivered between steps, never during one: a hint that arrives
while the orchestrator's LLM call or its own tools run (a blocking ``spawn_subagents``
included) waits for the step to end. That queueing is what the interactivity analysis measures.

With ``subagent.spawn_mode: background``, ``spawn_subagents`` starts its subagents on a pool and
returns at once; each report is delivered like a user message once its subagent ends. ``wait``
blocks until a hint is due, a report is ready, or its timeout; a reply without a tool call does
the same. ``inbox.jsonl`` records every delivery (when it was ready, when the orchestrator got
it); the deadline stops every agent through the recorder's stop event, like harbor's cancel.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .config import ExperimentConfig
from .environment import AgentStopped
from .hints import Hint
from .llm import BillingError, LLMError, Message, SailClient, ToolCall, ToolResult, ToolSpec
from .orchestrator import SPAWN_SPEC, Orchestrator
from .prompts import (FIRST_HINT, NEXT_HINT, REPORT_EVENT, USER_EVENT, WAIT_RESULT, clock, interactive_note,
                      interactive_prompt)
from .records import EpisodeRecorder
from .subagent import Dispatch, DispatchResult, Runtime, new_dispatch, run_dispatch
from .tools import ToolOutput, truncate

WAIT_SPEC = ToolSpec(
    "wait",
    "Pause until something new arrives for you (a new user message, or a subagent report when subagents run "
    "in the background), or until timeout_sec passes.",
    {"type": "object", "properties": {"timeout_sec": {"type": "integer", "description": "Longest wait in seconds "
                                                                                       "(default: until something arrives)."}},
     "required": []},
)
BACKGROUND_SPAWN_SPEC = ToolSpec(
    "spawn_subagents",
    "Start subagents in the background in the same container; each instruction must be self-contained. Returns at "
    "once; each subagent's report arrives as a message when it finishes. The system prompt describes the limits.",
    SPAWN_SPEC.parameters,
)
POLL_SEC = 0.5  # a wait also wakes this often to notice the stop event
JOIN_SEC = 30.0  # for the background subagents to unwind after a stop
ELIDED = "[elided to keep the conversation within its size limit: {n} characters of older tool output]"
ELIDE_MIN_CHARS = 200
ELIDE_TO = 0.75  # elide down to this fraction of max_context_chars, so the prompt prefix changes rarely


class InteractiveOrchestrator(Orchestrator):
    def __init__(self, config: ExperimentConfig, env: Any, rec: EpisodeRecorder, workdir: str, hints: list[Hint],
                 client_factory: Callable[[Any], Any] = SailClient):
        super().__init__(config, env, rec, workdir, client_factory)
        self.icfg = config.interactive
        self.hints = hints
        self.has_finish = False
        self.background = self.subagents and config.subagent.spawn_mode == "background"
        specs = [s for s in self.specs if s.name != "finish"]
        if self.background:
            specs = [BACKGROUND_SPAWN_SPEC if s.name == "spawn_subagents" else s for s in specs]
        self.specs = specs + [WAIT_SPEC]
        self.system = interactive_prompt(config, workdir)
        self.cv = threading.Condition()
        self.live: dict[str, tuple[Dispatch, float]] = {}  # running background dispatches -> (dispatch, t_start)
        self.done: list[tuple[Dispatch, DispatchResult, float]] = []  # ended, report not delivered yet
        self.futures: list[concurrent.futures.Future] = []
        self.pool: ThreadPoolExecutor | None = None
        self.next_hint = 0  # the first hint not delivered yet
        self.t0 = 0.0
        self.deadline_hit = self.idle = False
        self.n_elided = 0

    # -- the session -------------------------------------------------------------------------

    def run(self, instruction: str) -> str:
        clients: list[Any] = []
        exit_reason, error = "max_steps", None
        self.t0 = time.time()
        timer = threading.Timer(self.icfg.wall_clock_sec, self._on_deadline)
        timer.daemon = True
        try:
            clients.append(self.client_factory(self.cfg.orchestrator))
            if self.subagents:
                clients.append(self.client_factory(self.cfg.subagent_model))
            self.rt = Runtime(self.cfg, self.env, self.rec, self.workdir, *clients)
            if self.background:
                self.pool = ThreadPoolExecutor(self.cfg.subagent.max_concurrent, thread_name_prefix="bg-subagent")
            self.rec.update(session_t0=self.t0)
            timer.start()
            self.rec.transcript(self.who, Message("system", self.system))
            first = FIRST_HINT.format(instruction=instruction, hint=self.hints[0].text) if self.hints else instruction
            if self.hints:
                self._hint_record(self.hints[0], self.t0, 1)
                self.next_hint = 1
            self._add(Message("user", first))
            for step in range(1, self.cfg.budget.orchestrator_steps + 1):
                self._elide()
                outcome = self._step(step)
                if outcome == "no_tool_call" and self._wait(None) == "idle":
                    self.idle = True
                if self.idle:
                    exit_reason = "idle"
                    break
                self._deliver(step + 1)
        except (AgentStopped, LLMError) as e:
            failure = self.rec.failure
            if isinstance(failure, BillingError):
                exit_reason = "billing_error"
            elif failure is not None:
                exit_reason = "provider_error"
            elif self.deadline_hit:
                exit_reason = "deadline"
            else:
                exit_reason = "cancelled" if isinstance(e, AgentStopped) else "error"
            cause = failure or e
            error = None if exit_reason in ("cancelled", "deadline") else f"{type(cause).__name__}: {cause}"
        except BaseException:
            timer.cancel()
            self._shutdown()
            self._close(clients)
            self._finish_records()
            self.rec.finalize("error", traceback.format_exc()[-4000:])
            raise
        timer.cancel()
        self._shutdown()
        self._close(clients)
        self._finish_records()
        self.rec.finalize(exit_reason, error)
        return exit_reason

    def _on_deadline(self) -> None:
        self.deadline_hit = True
        self.rec.stop_event.set()
        with self.cv:
            self.cv.notify_all()

    @staticmethod
    def _close(clients: list[Any]) -> None:
        for c in clients:
            c.close()

    def _shutdown(self) -> None:
        """Stop the background subagents still running and wait for their records."""
        if self.pool is None:
            return
        if any(not f.done() for f in self.futures):
            self.rec.stop_event.set()
        self.pool.shutdown(wait=False, cancel_futures=True)
        concurrent.futures.wait(self.futures, timeout=JOIN_SEC)

    def _finish_records(self) -> None:
        """Hints never delivered (the session ended first) get their inbox record too."""
        for h in self.hints[self.next_hint:]:
            self._hint_record(h, None, None)
        self.next_hint = len(self.hints)
        self.rec.update(n_elided_results=self.n_elided)

    # -- the inbox ---------------------------------------------------------------------------

    def _due(self, now: float) -> list[Hint]:
        return [h for h in self.hints[self.next_hint:] if self.t0 + h.t_offset <= now]

    def _deliver(self, step: int) -> None:
        """Everything that arrived (due hints, ended subagents) as one user message before ``step``."""
        now = time.time()
        with self.cv:
            hints = self._due(now)
            self.next_hint += len(hints)
            done, self.done = self.done, []
        if not hints and not done:
            return
        parts: list[tuple[float, str]] = []
        for h in hints:
            t_ready = self.t0 + h.t_offset
            self._hint_record(h, now, step)
            parts.append((t_ready, USER_EVENT.format(at=clock(t_ready - self.t0), text=NEXT_HINT.format(hint=h.text))))
        cap = max(2000, self.cfg.tools.max_output_chars // max(len(done), 1))
        for d, r, t_end in done:
            self.rec.write("inbox", {
                "kind": "report", "t_ready": t_end, "t_delivered": now, "step": step, "hint_index": None,
                "subtask_id": None, "target_files": None, "f2p": None, "dispatch_id": d.dispatch_id,
                "agent_id": d.agent_id, "status": r.status, "chars": len(r.report),
            })
            parts.append((t_end, REPORT_EVENT.format(at=clock(t_end - self.t0), agent=d.agent_id, status=r.status,
                                                     steps=r.steps_used, instruction=truncate(d.instruction, 300),
                                                     report=truncate(r.report, cap))))
        text = "\n\n".join(p for _, p in sorted(parts, key=lambda x: x[0]))
        if self.messages and self.messages[-1].role == "user":  # keep user/assistant turns alternating
            prev = self.messages[-1]
            self.messages[-1] = Message("user", f"{prev.content}\n\n{text}")
            self.rec.transcript(self.who, Message("user", text))
        else:
            self._add(Message("user", text))

    def _hint_record(self, h: Hint, t_delivered: float | None, step: int | None) -> None:
        self.rec.write("inbox", {
            "kind": "hint", "t_ready": self.t0 + h.t_offset, "t_delivered": t_delivered, "step": step,
            "hint_index": h.index, "subtask_id": h.subtask.id, "target_files": list(h.subtask.target_files),
            "f2p": list(h.subtask.f2p), "dispatch_id": None, "agent_id": None, "status": None, "chars": len(h.text),
        })

    def _wait(self, timeout: float | None) -> str:
        """Block until something arrives (``event``), ``timeout`` passes, or, with ``end_when_idle``,
        the session has nothing left (``idle``); raises AgentStopped on a stop (the deadline)."""
        t_end = None if timeout is None else time.time() + max(0.0, timeout)
        with self.cv:
            while True:
                if self.rec.stop_event.is_set():
                    raise AgentStopped("episode stopped during wait")
                now = time.time()
                if self.done or self._due(now):
                    return "event"
                left = self.hints[self.next_hint:]
                if not left and not self.live and self.icfg.end_when_idle:
                    return "idle"
                if t_end is not None and now >= t_end:
                    return "timeout"
                waits = [POLL_SEC]
                if left:
                    waits.append(self.t0 + left[0].t_offset - now)
                if t_end is not None:
                    waits.append(t_end - now)
                self.cv.wait(max(0.0, min(waits)))

    # -- steps and tools ---------------------------------------------------------------------

    def _note(self, step: int) -> str:
        with self.cv:
            now = time.time()
            running = [(d.agent_id, now - t) for d, t in self.live.values()] if self.background else None
        return interactive_note(time.time() - self.t0, self.icfg.wall_clock_sec, running,
                                self.dispatches_left if self.subagents else None)

    def _tool(self, tc: ToolCall, step: int, call_id: str, groups: list[str]) -> ToolOutput:
        if tc.name != "wait":
            return super()._tool(tc, step, call_id, groups)

        def wait() -> ToolOutput:
            raw = tc.arguments.get("timeout_sec")
            timeout = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0 else None
            t = time.time()
            if self._wait(timeout) == "idle":
                self.idle = True
            return ToolOutput(WAIT_RESULT.format(sec=time.time() - t))

        return self.rec.tool(self.who, step, call_id, tc, wait)

    def _spawn(self, instructions: list[str], step: int, group_id: str) -> ToolOutput:
        if not self.background:
            return super()._spawn(instructions, step, group_id)
        s = self.cfg.subagent
        t_start = time.time()
        with self.cv:
            free = s.max_concurrent - len(self.live)
        granted = max(0, min(len(instructions), free, self.dispatches_left))
        self.dispatches_left -= granted
        ds = [new_dispatch(self.rt, text, i, group_id, step) for i, text in enumerate(instructions[:granted])]
        with self.cv:
            for d in ds:
                self.live[d.dispatch_id] = (d, time.time())
        try:
            for d in ds:
                self.futures.append(self.pool.submit(self._run_background, d))
        finally:
            self.rec.write("spawns", {
                "group_id": group_id, "step": step, "strategy": s.strategy, "scope": s.scope,
                "spawn_mode": "background", "requested": len(instructions), "granted": granted,
                "dispatch_ids": [d.dispatch_id for d in ds], "arbitration_ids": [], "t_start": t_start,
                "t_end": time.time(),
            })
        with self.cv:
            running = len(self.live)
        started = ", ".join(d.agent_id for d in ds) or "none"
        parts = [f"[spawn_subagents: background, scope {s.scope}; started {granted} of {len(instructions)} "
                 f"instruction(s): {started}; running now: {running} of at most {s.max_concurrent}; dispatch budget "
                 f"left: {self.dispatches_left}. Each report arrives as a message when its subagent finishes.]"]
        if granted < len(instructions):
            parts.append(f"[note: instruction(s) {granted + 1}-{len(instructions)} were NOT run: no free subagent "
                         "slot or the episode's dispatch budget is used up; wait for a report and spawn them again]")
        text = "\n".join(parts)
        return ToolOutput(text, output_chars=len(text))

    def _run_background(self, d: Dispatch) -> None:
        try:
            r = run_dispatch(self.rt, d)
        except BaseException as e:  # run_dispatch catches everything an agent raises; this is a harness bug
            r = DispatchResult("error", f"(no report: harness error {type(e).__name__}: {e})", 0)
        with self.cv:
            self.live.pop(d.dispatch_id, None)
            if not self.rec.stop_event.is_set():
                self.done.append((d, r, time.time()))
            self.cv.notify_all()

    # -- context size ------------------------------------------------------------------------

    def _elide(self) -> None:
        """Replace the oldest tool outputs with a marker while the conversation is over its limit."""
        limit = self.icfg.max_context_chars
        if limit is None:
            return
        size = sum(_chars(m) for m in self.messages)
        if size <= limit:
            return
        for m in self.messages[1:-1]:  # never the task message or the newest results
            if size <= ELIDE_TO * limit:
                break
            if m.role != "tool":
                continue
            results = []
            for r in m.tool_results:
                if len(r.content) > ELIDE_MIN_CHARS and not r.content.startswith("[elided"):
                    marker = ELIDED.format(n=len(r.content))
                    size -= len(r.content) - len(marker)
                    r = ToolResult(r.call_id, marker, r.is_error)
                    self.n_elided += 1
                results.append(r)
            m.tool_results = results


def _chars(m: Message) -> int:
    return (len(m.content or "") + len(m.reasoning or "") + sum(len(r.content) for r in m.tool_results)
            + sum(len(str(c.arguments)) for c in m.tool_calls))
