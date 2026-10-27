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

With ``interactive.replies`` the schedule also holds questions and status checks, every message
has an id, and the orchestrator answers with ``reply`` (``replies.jsonl``: when, to which message,
the text, an ETA in minutes, done). The user sees nothing else, so a reply's time is when the user
got an answer, an estimate or a done. With ``require_eta``, a reply to a bug without a usable
``eta_min`` and without ``done: true`` is refused (an error result; nothing is sent or recorded).

With ``subagent.stop_tool``, ``stop_subagents(agent_ids)`` asks running background subagents to
stop: each ends at its next step (before its next LLM call or tool call) with status ``stopped``,
and its report is delivered as usual. With ``interactive.change_of_plan``, ``cancel`` and ``undo``
messages take back an earlier bug (their inbox records name it: ``target_msg_id``, ``subtask_id``).

With ``interactive.snapshots``, a :class:`~subagents_bench.snapshots.Snapshotter` records the
workspace's diff against ``baseline`` in the background from the session's start until it ends.
"""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .config import ExperimentConfig
from .environment import AgentStopped
from .hints import Hint
from .llm import BillingError, CallCancel, ChatClient, LLMError, Message, ToolCall, ToolResult, ToolSpec
from .orchestrator import SPAWN_SPEC, Orchestrator
from .prompts import (FIRST_HINT, REPORT_EVENT, REPORTED_SCOPE, USER_EVENT, WAIT_RESULT, clock, interactive_note,
                      interactive_prompt, request_text)
from .records import EpisodeRecorder
from .snapshots import Snapshotter
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
REPLY_SPEC = ToolSpec(
    "reply",
    "Send a message to the user about one of their requests: an answer, an estimate of when a bug will be "
    "fixed, or that it is done. The user sees nothing else you write.",
    {"type": "object", "properties": {
        "to": {"type": "integer", "description": "The id of the user's message (the 3 of #3)."},
        "text": {"type": "string", "description": "The message."},
        "eta_min": {"type": "number", "description": "For a bug: minutes from now until it is fixed."},
        "done": {"type": "boolean", "description": "For a bug: true once it is fixed and checked."}},
     "required": ["to", "text"]},
)
STOP_SPEC = ToolSpec(
    "stop_subagents",
    "Stop background subagents that are still running. Each stops at its next step (a command or model call "
    "in progress finishes first); its report so far arrives as a message.",
    {"type": "object", "properties": {"agent_ids": {"type": "array", "items": {"type": "string"},
                                                    "description": "Ids of running subagents (sub-3)."}},
     "required": ["agent_ids"]},
)
# interactive.bundle > 1: a reply can be about one bug of a message that lists several
REPLY_ITEM = {"type": "integer", "description": "For a message that lists several bugs: the number of the bug this "
                                                "reply is about (omit: every bug of the message)."}
BUNDLE_REPLY_SPEC = ToolSpec(REPLY_SPEC.name, REPLY_SPEC.description,
                             {**REPLY_SPEC.parameters, "properties": {**REPLY_SPEC.parameters["properties"],
                                                                       "item": REPLY_ITEM}})
POLL_SEC = 0.5  # a wait also wakes this often to notice the stop event
JOIN_SEC = 30.0  # for the background subagents to unwind after a stop
ELIDED = "[elided to keep the conversation within its size limit: {n} characters of older tool output]"
ELIDED_REPORT = "[elided to keep the conversation within its size limit: {n} characters of an older report]"
ELIDE_MIN_CHARS = 200
ELIDE_TO = 0.75  # elide down to this fraction of max_context_chars, so the prompt prefix changes rarely


class InteractiveOrchestrator(Orchestrator):
    def __init__(self, config: ExperimentConfig, env: Any, rec: EpisodeRecorder, workdir: str, hints: list[Hint],
                 client_factory: Callable[[Any], Any] = ChatClient):
        super().__init__(config, env, rec, workdir, client_factory)
        self.icfg = config.interactive
        self.hints = hints
        self.has_finish = False
        self.background = self.subagents and config.subagent.spawn_mode == "background"
        specs = [s for s in self.specs if s.name != "finish"]
        if self.background:
            specs = [BACKGROUND_SPAWN_SPEC if s.name == "spawn_subagents" else s for s in specs]
        reply = BUNDLE_REPLY_SPEC if self.icfg.bundle > 1 else REPLY_SPEC
        self.specs = specs + ([STOP_SPEC] if self.background and config.subagent.stop_tool else []) + [WAIT_SPEC] + \
            ([reply] if self.icfg.replies else [])
        self.sent: dict[int, Hint] = {}  # delivered messages by id, for `reply`
        self.answered: set[int] = set()  # ids replied to
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
        self.report_bodies: list[str] = []  # delivered report texts, for elision
        self.snapshotter: Snapshotter | None = None
        self.in_row = 0  # interrupted calls in a row
        self._interrupt = self._interrupt_due if self.icfg.interrupt else None

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
            if self.icfg.snapshots:
                self.snapshotter = Snapshotter(self.env, self.rec, self.workdir, self.icfg.snapshot_every_sec)
                self.snapshotter.start()
            timer.start()
            self.rec.transcript(self.who, Message("system", self.system))
            template = REPORTED_SCOPE if self.icfg.scope == "reported" else FIRST_HINT
            note = f"\n\n{self.icfg.session_note.strip()}" if self.icfg.session_note else ""
            if self.hints:
                h = self.hints[0]
                first = template.format(instruction=instruction, note=note,
                                        request=request_text(h.kind, h.msg_id, h.text, h.urgent, first=True,
                                                             n_items=max(1, len(h.items))))
                self._hint_record(h, self.t0, 1)
                self.next_hint = 1
            else:
                first = instruction + note
            self._add(Message("user", first))
            for step in range(1, self.cfg.budget.orchestrator_steps + 1):
                if step > 1:  # never after the last step: what arrives then stays undelivered
                    self._deliver(step)
                self._elide()
                if self._step(step) == "no_tool_call":
                    self._implicit_wait(step)
                self.in_row = self.in_row + 1 if self.last_interrupted else 0
                if self.idle and self._nothing_left():
                    exit_reason = "idle"
                    break
                self.idle = False  # a later tool of the same step started work after `wait` found none
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
            self._stop_snapshots()
            self._close(clients)
            self._finish_records()
            self.rec.finalize("error", traceback.format_exc()[-4000:])
            raise
        timer.cancel()
        self._shutdown()
        self._stop_snapshots()
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

    def _stop_snapshots(self) -> None:
        """After the agents stopped: no more snapshots, and none is written once the records finish
        (the final state is the trial's ``model.patch``)."""
        if self.snapshotter is not None:
            self.snapshotter.stop()

    def _finish_records(self) -> None:
        """Hints and reports never delivered (the session ended first) get their inbox record too."""
        for h in self.hints[self.next_hint:]:
            self._hint_record(h, None, None)
        self.next_hint = len(self.hints)
        with self.cv:
            done, self.done = self.done, []
        for d, r, t_end in done:
            self._report_record(d, r, t_end, None, None)
        self.rec.update(n_elided_results=self.n_elided)

    def _nothing_left(self) -> bool:
        with self.cv:
            return self.next_hint >= len(self.hints) and not self.live and not self.done

    def _implicit_wait(self, step: int) -> None:
        """A reply without a tool call ends the turn: wait for the next event, recorded as a
        ``wait`` tool of that step (so the critical path and busy-at-arrival count it as waiting)."""
        t = time.time()
        steps = self.rec.records["steps"]
        try:
            if self._wait(None) == "idle":
                self.idle = True
        finally:
            self.rec.write("tools", {
                **self.who.fields(), "step": step, "call_id": steps[-1]["call_id"] if steps else None, "name": "wait",
                "t_start": t, "t_end": time.time(), "exit_code": None, "timed_out": False, "is_error": False,
                "output_chars": 0, "tool_call_id": None, "arguments": {"implicit": True},
            })

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
            parts.append((t_ready, USER_EVENT.format(at=clock(t_ready - self.t0),
                                                     text=request_text(h.kind, h.msg_id, h.text, h.urgent,
                                                                       n_items=max(1, len(h.items))))))
        cap = max(2000, self.cfg.tools.max_output_chars // max(len(done), 1))
        for d, r, t_end in done:
            self._report_record(d, r, t_end, now, step)
            body = truncate(r.report, cap)
            self.report_bodies.append(body)
            parts.append((t_end, REPORT_EVENT.format(at=clock(t_end - self.t0), agent=d.agent_id, status=r.status,
                                                     steps=r.steps_used, instruction=truncate(d.instruction, 300),
                                                     report=body)))
        text = "\n\n".join(p for _, p in sorted(parts, key=lambda x: x[0]))
        if self.messages and self.messages[-1].role == "user":  # keep user/assistant turns alternating
            prev = self.messages[-1]
            self.messages[-1] = Message("user", f"{prev.content}\n\n{text}")
            self.rec.transcript(self.who, Message("user", text))
        else:
            self._add(Message("user", text))

    def _report_record(self, d: Dispatch, r: DispatchResult, t_end: float, t_delivered: float | None,
                       step: int | None) -> None:
        self.rec.write("inbox", {
            "kind": "report", "t_ready": t_end, "t_delivered": t_delivered, "step": step, "hint_index": None,
            "subtask_id": None, "target_files": None, "f2p": None, "dispatch_id": d.dispatch_id,
            "agent_id": d.agent_id, "status": r.status, "chars": len(r.report),
        })

    def _hint_record(self, h: Hint, t_delivered: float | None, step: int | None) -> None:
        """A bug's record keeps the kind ``hint``; questions and status checks have their own kinds.
        A bug message gets one record per bug it lists (``interactive.bundle``): each with its own
        subtask, ``hint_index`` (its position among every bug sent) and ``item`` (1, 2, ... in the
        list), and the message's ``msg_id``, ``msg_index`` (its position among the bug messages),
        ``subtask_ids`` (every item's, in order) and ``chars`` (the whole message's text)."""
        if t_delivered is not None and h.msg_id is not None:
            self.sent[h.msg_id] = h
        q = h.question
        items = list(zip(h.items, h.positions)) if h.items else [(h.subtask, h.index)]
        for k, (s, pos) in enumerate(items, 1):
            bug = h.kind == "bug" and s is not None
            self.rec.write("inbox", {
                "kind": "hint" if h.kind == "bug" else h.kind, "t_ready": self.t0 + h.t_offset,
                "t_delivered": t_delivered, "step": step, "hint_index": pos, "subtask_id": s.id if s else None,
                "target_files": list(s.target_files) if s else None, "f2p": list(s.f2p) if s else None,
                "dispatch_id": None, "agent_id": None, "status": None, "chars": len(h.text), "msg_id": h.msg_id,
                "urgent": h.urgent, "question_id": q.id if q else None, "question": q.question if q else None,
                "reference": q.answer if q else None, "target_msg_id": h.target, "msg_index": h.index,
                "item": k if bug else None, "subtask_ids": [x.id for x, _ in items] if bug else None,
            })

    def _interrupt_due(self, call: CallCancel) -> bool:
        """Abort the orchestrator's call in progress: a user message (any kind; subagent reports never
        interrupt) is due, the call has run ``interrupt_min_sec``, it is not streaming a tool call yet
        (its reasoning is over: let it act), and fewer than ``interrupt_max_in_row`` calls in a row were
        interrupted. Checked every ``records.POLL_SEC``; the next step delivers all that is due."""
        now = time.time()
        if (now - call.t_start < self.icfg.interrupt_min_sec or self.in_row >= self.icfg.interrupt_max_in_row
                or call.in_tool_call):
            return False
        with self.cv:
            return bool(self._due(now))

    def _wait(self, timeout: float | None) -> str:
        """Block until something arrives (``event``) or ``timeout`` passes; an untimed wait (the end of a
        turn) also returns ``idle`` when, with ``end_when_idle``, the session has nothing left. A timed
        wait runs its course: the agent may be waiting on its own background commands. Raises
        AgentStopped on a stop (the deadline)."""
        t_end = None if timeout is None else time.time() + max(0.0, timeout)
        with self.cv:
            while True:
                if self.rec.stop_event.is_set():
                    raise AgentStopped("episode stopped during wait")
                now = time.time()
                if self.done or self._due(now):
                    return "event"
                left = self.hints[self.next_hint:]
                if timeout is None and not left and not self.live and self.icfg.end_when_idle:
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
        if tc.name == "reply" and self.icfg.replies:
            return self.rec.tool(self.who, step, call_id, tc, lambda: self._reply(tc.arguments, step))
        if tc.name == "stop_subagents" and self.background and self.cfg.subagent.stop_tool:
            return self.rec.tool(self.who, step, call_id, tc, lambda: self._stop_subagents(tc.arguments))
        if tc.name != "wait":
            return super()._tool(tc, step, call_id, groups)

        def wait() -> ToolOutput:
            timeout = wait_timeout(tc.arguments.get("timeout_sec"), self.icfg.wall_clock_sec)
            t = time.time()
            if self._wait(timeout) == "idle":
                self.idle = True
            return ToolOutput(WAIT_RESULT.format(sec=time.time() - t))

        return self.rec.tool(self.who, step, call_id, tc, wait)

    def _reply(self, args: dict, step: int) -> ToolOutput:
        to = _message_id(args.get("to"))
        text = args.get("text")
        if to is None or to not in self.sent:
            known = ", ".join(f"#{i}" for i in sorted(self.sent)) or "none yet"
            return ToolOutput(f"[error] `to` must be the id of a message you received ({known})", True)
        if not isinstance(text, str) or not text.strip():
            return ToolOutput("[error] `text` must be a non-empty message", True)
        eta, eta_ok = _minutes(args.get("eta_min"))
        done, done_ok = _flag(args.get("done"))
        h = self.sent[to]
        items = list(h.items) or ([h.subtask] if h.subtask else [])
        item = _message_id(args.get("item")) if len(items) > 1 else None  # one bug of a list, or every one
        if len(items) > 1 and args.get("item") is not None and not (item is not None and 1 <= item <= len(items)):
            return ToolOutput(f"[error] `item` must be the number of one of the {len(items)} bugs of #{to} "
                              f"(1-{len(items)}), or left out for all of them; nothing was sent", True)
        covered = [items[item - 1]] if item else items
        pos = h.positions[item - 1] if item and h.positions else h.index  # the bug's hint_index (one bug)
        if self.icfg.require_eta and h.kind == "bug" and eta is None and not done:
            return ToolOutput(f"[error] a reply to a bug (#{to}) needs eta_min (your estimate, in minutes from now, of "
                              "when it will be fixed) or done: true; nothing was sent", True)
        first_answer = h.kind == "question" and to not in self.answered
        self.answered.add(to)
        self.rec.write("replies", {
            "t": time.time(), "step": step, "to": to, "kind": h.kind, "hint_index": pos,
            "subtask_id": covered[0].id if len(covered) == 1 else None,
            "question_id": h.question.id if h.question else None, "text": text, "eta_min": eta, "done": done,
            "item": item, "subtask_ids": [x.id for x in covered] if h.kind == "bug" else None,
        })
        notes = (["taken as your answer to this question"] if first_answer else []) + \
                ([] if eta_ok else ["eta_min ignored: expected a number of minutes >= 0"]) + \
                ([] if done_ok else ["done ignored: expected true or false"])
        about = f" item {item}" if item else ""
        return ToolOutput(f"[sent to the user, re #{to}{about}" + "".join(f"; {n}" for n in notes) + "]")

    def _stop_subagents(self, args: dict) -> ToolOutput:
        ids = args.get("agent_ids")
        if isinstance(ids, str):
            ids = [ids]
        if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids):
            return ToolOutput("[error] `agent_ids` must be a non-empty list of subagent ids (sub-3)", True)
        now = time.time()
        with self.cv:
            live = {d.agent_id: d for d, _ in self.live.values()}
        stopping, unknown = [], []
        for a in dict.fromkeys(x.strip() for x in ids):
            d = live.get(a)
            if d is None:
                unknown.append(a)
                continue
            if not d.stop.is_set():
                d.t_stop = now
                d.stop.set()
            stopping.append(a)
        parts = [f"[stop_subagents: {', '.join(stopping)} will stop at its next step; each report arrives as a "
                 "message]"] if stopping else []
        if unknown:
            parts.append(f"[not running: {', '.join(unknown)}]")
        return ToolOutput("\n".join(parts), not stopping)

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
        """While the conversation is over its limit, replace, oldest message first, tool outputs and
        subagent reports with a marker and drop old reasoning (never the task message or the newest
        message, and never the newest assistant turn's reasoning: Anthropic and Gemini reject a tool
        loop whose current turn lost its signed thinking or thought signature)."""
        limit = self.icfg.max_context_chars
        if limit is None:
            return
        size = sum(_chars(m) for m in self.messages)
        if size <= limit:
            return
        newest_turn = next((m for m in reversed(self.messages) if m.role == "assistant"), None)
        for m in self.messages[1:-1]:
            if size <= ELIDE_TO * limit:
                break
            if m.role == "tool":
                results = []
                for r in m.tool_results:
                    if len(r.content) > ELIDE_MIN_CHARS and not r.content.startswith("[elided"):
                        marker = ELIDED.format(n=len(r.content))
                        size -= len(r.content) - len(marker)
                        r = ToolResult(r.call_id, marker, r.is_error)
                        self.n_elided += 1
                    results.append(r)
                m.tool_results = results
            elif m.role == "user":
                for body in self.report_bodies:
                    if len(body) > ELIDE_MIN_CHARS and body in (m.content or ""):
                        marker = ELIDED_REPORT.format(n=len(body))
                        m.content = m.content.replace(body, marker, 1)
                        size -= len(body) - len(marker)
                        self.n_elided += 1
            elif m.role == "assistant" and m is not newest_turn and (m.reasoning or m.reasoning_details):
                size -= _reasoning_chars(m)
                m.reasoning = m.reasoning_details = None
                self.n_elided += 1


def _message_id(raw: Any) -> int | None:
    """``reply``'s ``to``: 3, 3.0, "3" or "#3"."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    if isinstance(raw, str):
        raw = raw.strip().lstrip("#")
        return int(raw) if raw.isascii() and raw.isdigit() and len(raw) <= 9 else None
    return raw if isinstance(raw, int) else None


def _minutes(raw: Any) -> tuple[float | None, bool]:
    """``reply``'s ``eta_min`` -> (minutes or None, whether it was absent or usable)."""
    if raw is None:
        return None, True
    if isinstance(raw, bool):
        return None, False
    try:
        x = float(raw)
    except (TypeError, ValueError, OverflowError):
        return None, False
    return (x, True) if 0 <= x < 1e6 else (None, False)


def _flag(raw: Any) -> tuple[bool, bool]:
    """``reply``'s ``done`` -> (done, whether it was absent or readable)."""
    if raw is None or isinstance(raw, bool):
        return raw is True, True
    if isinstance(raw, (int, float)) and raw in (0, 1):
        return raw == 1, True
    if isinstance(raw, str) and raw.strip().lower() in ("true", "false", "yes", "no"):
        return raw.strip().lower() in ("true", "yes"), True
    return False, False


def wait_timeout(raw: Any, cap: float) -> float | None:
    """``wait``'s timeout_sec: None (absent or unreadable) waits for an event; 0 or less checks
    once; anything else is capped at the session length."""
    if raw is None or isinstance(raw, bool):
        return None
    try:
        t = float(raw)
    except OverflowError:  # an integer beyond a float: as long as possible
        return cap
    except (TypeError, ValueError):
        return None
    if t != t:  # NaN
        return None
    return min(max(t, 0.0), cap)


def _chars(m: Message) -> int:
    return (len(m.content or "") + _reasoning_chars(m) + sum(len(r.content) for r in m.tool_results)
            + sum(len(str(c.arguments)) for c in m.tool_calls))


def _reasoning_chars(m: Message) -> int:
    """The reasoning sent back with ``m``: its reasoning_details if any (llm._wire), else its reasoning."""
    return len(json.dumps(m.reasoning_details)) if m.reasoning_details else len(m.reasoning or "")
