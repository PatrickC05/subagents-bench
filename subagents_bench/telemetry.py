"""Episode recording: every LLM call, tool call, step, spawn, dispatch,
arbitration and message of one episode, written as it happens.

:class:`EpisodeRecorder` owns the episode directory and writes the eight files
defined in :mod:`subagents_bench.schema` (``episode.json`` plus seven JSONL
files). Each JSONL line is appended and flushed under one lock, so a crashed or
cancelled episode still leaves everything up to the failure on disk. The
recorder allocates the global ``seq`` counter (start order of calls and tools)
and one ``agent_seq`` counter per agent; both are taken when an event STARTS
and the record is written when it ends, so every allocation is followed by
exactly one record (failed and cancelled events included) and ``agent_seq`` is
contiguous per agent.

``episode.json`` is written at construction (``status="running"``) and again
by :meth:`EpisodeRecorder.finalize` (``status="done"``) with every aggregate
computed from the records written, never from separate counters, so the totals
always equal the sums over the JSONL files. ``finalize`` is idempotent: the
first call wins, and records arriving afterwards (a worker thread that had not
yet unwound when harbor cancelled the agent) are dropped and counted in
:attr:`EpisodeRecorder.dropped`.

:class:`AgentLog` is the per-agent facade the runtime uses: it wraps an LLM
call or a tool invocation, allocates the counters, measures it, builds the
:class:`~subagents_bench.schema.CallRecord`/``ToolRecord`` and appends
transcript messages. LLM calls are made stop-aware: when the episode's stop
event is set while a provider call is in flight, the call is abandoned (its
HTTP request finishes in a daemon thread and is discarded) and
:class:`~subagents_bench.types.AgentStopped` is raised, with a CallRecord whose
``error`` says it was cancelled and whose usage is 0 (unknown).

``Telemetry`` remains importable as an alias of :class:`EpisodeRecorder`.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import functools
import json
import logging
import os
import platform
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .schema import (
    RECORD_TYPES,
    SCHEMA_VERSION,
    ArbitrationRecord,
    CallRecord,
    DispatchRecord,
    EpisodeRecord,
    SpawnRecord,
    StepRecord,
    ToolRecord,
    TranscriptRecord,
    validate_record,
)
from .types import (
    AgentStopped,
    Attempt,
    LLMError,
    LLMResponse,
    Message,
    RequestInfo,
    ToolCall,
    ToolOutput,
    ToolSpec,
    Usage,  # noqa: F401 - re-exported for older imports
)

log = logging.getLogger(__name__)

ROLES = ("orchestrator", "subagent", "arbiter", "debate")
# String tool arguments longer than this are truncated in ToolRecord.args.
ARG_STR_CAP = 4000
# How often a stop-aware LLM wait checks the stop event.
_POLL_SEC = 0.2

_ID_PREFIX = {"call": "c", "tool": "t", "group": "g", "dispatch": "d", "arbitration": "a"}


def new_episode_id() -> str:
    """Sortable, unique: ``20261011T153000Z-1a2b3c4d``."""
    now = _dt.datetime.now(_dt.timezone.utc)
    return f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def provenance() -> dict[str, Any]:
    """Harness/host provenance for ``episode.json`` (None where unknown).

    Computed once per process (each harbor job is a fresh process that
    imports the code on disk). ``source_sha256`` fingerprints the source
    that ran, committed or not (:func:`subagents_bench.provenance.source_fingerprint`);
    ``git_commit``/``git_dirty`` describe the repository holding the package
    and are None outside a git work tree (:func:`subagents_bench.provenance.git_info`:
    ``git_dirty`` looks at tracked source files only).
    """
    from . import __version__  # noqa: PLC0415 - avoid an import cycle at module load
    from .provenance import git_info, repo_root, source_fingerprint  # noqa: PLC0415

    root = repo_root()
    git = git_info(root)
    try:
        from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

        try:
            harbor_version: str | None = version("harbor")
        except PackageNotFoundError:
            harbor_version = None
    except ImportError:  # pragma: no cover
        harbor_version = None
    return {
        "harness_version": str(__version__),
        "git_commit": git["git_commit"],
        "git_dirty": git["git_dirty"],
        "source_sha256": source_fingerprint(root),
        "harbor_version": harbor_version,
        "python_version": platform.python_version(),
        "host_platform": platform.platform(),
        "host_machine": platform.machine(),
    }


# ---------------------------------------------------------------------------
# record builders
# ---------------------------------------------------------------------------


def _opt_float(v: Any) -> float | None:
    return None if v is None else float(v)


def _price_fields(cfg: Any) -> dict[str, Any]:
    return {
        "price_in": _opt_float(getattr(cfg, "price_in", None)),
        "price_out": _opt_float(getattr(cfg, "price_out", None)),
        "price_cache_read": _opt_float(getattr(cfg, "price_cache_read", None)),
    }


def sum_or_none(values: Any) -> float | None:
    """Sum of costs; None if any is None (one unpriced call makes the total unknown)."""
    total = 0.0
    for v in values:
        if v is None:
            return None
        total += float(v)
    return total


# ``error`` prefix (CallRecord, DispatchRecord) of a provider failure: a plain
# LLMError, i.e. retries exhausted, a non-retryable 4xx (401/403, ...) or a
# missing API key. BillingError and ContextLengthError have their own names
# and outcomes.
PROVIDER_ERROR_PREFIX = "LLMError:"


def is_provider_error(error: Any) -> bool:
    """Whether a recorded ``error`` text is a provider failure (counted in
    ``EpisodeRecord.n_provider_errors``); works on records of any schema version."""
    return isinstance(error, str) and error.startswith(PROVIDER_ERROR_PREFIX)


def _request_fields(req: RequestInfo) -> dict[str, Any]:
    return {
        "model": req.model,
        "provider": req.provider,
        "base_url": req.base_url,
        "completion_window": req.completion_window,
        "max_tokens": int(req.max_tokens),
        "temperature": float(req.temperature),
        "extra_body": dict(req.extra_body or {}),
        "n_messages": int(req.n_messages),
        "n_tools": int(req.n_tools),
        "prompt_chars": int(req.prompt_chars),
        "system_hash": req.system_hash,
        "tools_hash": req.tools_hash,
        "msg_hashes": list(req.msg_hashes),
        "msg_chars": list(req.msg_chars),
        "reasoning_passback": bool(req.reasoning_passback),
    }


def _attempts(attempts: list[Attempt]) -> list[dict[str, Any]]:
    return [asdict(a) for a in attempts]


def _diff(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else b - a


def call_record_from_response(
    resp: LLMResponse, *, ids: dict[str, Any], cfg: Any, context_trimmed_chars: int = 0
) -> CallRecord:
    """CallRecord for a successful call. ``ids``: episode_id, call_id, seq,
    role, agent_id, dispatch_id, group_id, arbitration_id, step, agent_seq."""
    t = resp.timing
    u = resp.usage
    attempts = list(resp.attempts)
    return CallRecord(
        **ids,
        **_request_fields(resp.request),
        context_trimmed_chars=int(context_trimmed_chars),
        t_start=t.t_start,
        t_end=t.t_end,
        latency_sec=t.t_end - t.t_start,
        t_request_sent=t.t_request_sent,
        t_headers=t.t_headers,
        t_first_chunk=t.t_first_chunk,
        t_first_reasoning=t.t_first_reasoning,
        t_first_content=t.t_first_content,
        t_first_tool_call=t.t_first_tool_call,
        t_last_chunk=t.t_last_chunk,
        ttft_sec=_diff(t.t_request_sent, t.t_first_chunk),
        decode_sec=_diff(t.t_first_chunk, t.t_last_chunk),
        n_chunks=int(t.n_chunks),
        chunk_offsets_ms=list(t.chunk_offsets_ms),
        streamed=bool(t.streamed),
        n_attempts=max(len(attempts), 1),
        backoff_sec=float(sum(a.backoff_sec for a in attempts)),
        attempts=_attempts(attempts),
        prompt_tokens=int(u.prompt_tokens),
        completion_tokens=int(u.completion_tokens),
        cached_tokens=int(u.cached_tokens),
        uncached_prompt_tokens=int(u.uncached_prompt_tokens),
        reasoning_tokens=u.reasoning_tokens,
        usage_source=str(resp.usage_source),
        usage_raw=resp.usage_raw,
        finish_reason=str(resp.finish_reason),
        n_tool_calls=len(resp.tool_calls),
        tool_call_names=[tc.name for tc in resp.tool_calls],
        stream_repairs=int(getattr(resp, "stream_repairs", 0) or 0),
        content_chars=len(resp.text or ""),
        reasoning_chars=len(resp.reasoning or ""),
        response_id=resp.response_id,
        response_headers=dict(resp.response_headers or {}),
        cost_usd=_opt_float(resp.cost_usd),
        **_price_fields(cfg),
        provider_reported_cost=resp.provider_reported_cost,
        error=None,
    )


def failed_call_record(
    *,
    ids: dict[str, Any],
    cfg: Any,
    request: RequestInfo,
    t_start: float,
    t_end: float,
    attempts: list[Attempt],
    error: str,
    context_trimmed_chars: int = 0,
) -> CallRecord:
    """CallRecord for a call that raised (provider error, or abandoned on
    stop). Usage is 0 and ``usage_source="none"``: nothing was reported."""
    return CallRecord(
        **ids,
        **_request_fields(request),
        context_trimmed_chars=int(context_trimmed_chars),
        t_start=t_start,
        t_end=t_end,
        latency_sec=t_end - t_start,
        t_request_sent=None,
        t_headers=None,
        t_first_chunk=None,
        t_first_reasoning=None,
        t_first_content=None,
        t_first_tool_call=None,
        t_last_chunk=None,
        ttft_sec=None,
        decode_sec=None,
        n_chunks=0,
        chunk_offsets_ms=[],
        streamed=bool(getattr(cfg, "stream", True)),
        n_attempts=len(attempts),
        backoff_sec=float(sum(a.backoff_sec for a in attempts)),
        attempts=_attempts(attempts),
        prompt_tokens=0,
        completion_tokens=0,
        cached_tokens=0,
        uncached_prompt_tokens=0,
        reasoning_tokens=None,
        usage_source="none",
        usage_raw=None,
        finish_reason="error",
        n_tool_calls=0,
        tool_call_names=[],
        content_chars=0,
        reasoning_chars=0,
        response_id=None,
        response_headers={},
        # Usage 0 by convention: 0 USD when the model is priced, unknown otherwise.
        cost_usd=0.0 if (getattr(cfg, "price_in", None) is not None and getattr(cfg, "price_out", None) is not None) else None,
        **_price_fields(cfg),
        provider_reported_cost=None,
        error=error,
    )


def _generic_wire(messages: list[Message]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            d["tool_calls"] = [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in m.tool_calls]
        if m.tool_results:
            d["tool_results"] = [asdict(tr) for tr in m.tool_results]
        if m.reasoning:
            d["reasoning"] = m.reasoning
        out.append(d)
    return out


def request_info_for(client: Any, system: str, messages: list[Message], tools: list[ToolSpec]) -> RequestInfo:
    """RequestInfo for a call that failed before the client returned one.

    OpenAI-compatible clients are serialized exactly as they would be sent;
    other clients fall back to a canonical serialization of the runtime
    messages (same field meanings, different hashes).
    """
    from .llm import build_request_info  # noqa: PLC0415 - llm imports config; keep telemetry light

    cfg = client.cfg
    passback = bool(getattr(cfg, "reasoning_passback", True))
    to_wire = getattr(client, "_to_openai", None)
    tool_defs_fn = getattr(client, "_tool_defs", None)
    try:
        if callable(to_wire) and callable(tool_defs_fn):
            wire = to_wire(messages)
            defs = tool_defs_fn(tools)
            system_wire: Any = {"role": "system", "content": system} if system else None
        else:
            raise TypeError
    except Exception:  # noqa: BLE001 - fall back to the generic form
        wire = _generic_wire(messages)
        defs = [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in tools]
        system_wire = system or None
    return build_request_info(
        cfg, system=system, system_wire=system_wire, wire_messages=wire, tool_defs=defs, reasoning_passback=passback
    )


def _cap_args(args: Any) -> dict[str, Any]:
    if not isinstance(args, dict):
        return {"_raw": str(args)[:ARG_STR_CAP]}

    def cap(v: Any) -> Any:
        if isinstance(v, str) and len(v) > ARG_STR_CAP:
            return v[:ARG_STR_CAP] + f"...[truncated: {len(v)} chars total]"
        if isinstance(v, dict):
            return {k: cap(x) for k, x in v.items()}
        if isinstance(v, list):
            return [cap(x) for x in v]
        return v

    return {str(k): cap(v) for k, v in args.items()}


def _message_fields(m: Message) -> dict[str, Any]:
    return {
        "message_role": m.role,
        "content": m.content,
        "reasoning": m.reasoning or None,
        "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in m.tool_calls],
        "tool_results": [asdict(tr) for tr in m.tool_results],
    }


# ---------------------------------------------------------------------------
# the recorder
# ---------------------------------------------------------------------------


class EpisodeRecorder:
    """Writes one episode's record files; thread-safe.

    ``static`` fields of :class:`~subagents_bench.schema.EpisodeRecord` that
    are not aggregates (task identity, provenance, sandbox facts) are given
    to the constructor; unknown names raise. ``on_event(kind, record_dict)``
    is called (outside the lock) after every record written, e.g. to update
    harbor's ``AgentContext`` progressively.
    """

    def __init__(
        self,
        episode_dir: str | os.PathLike,
        *,
        config: Any,
        task_name: str,
        instruction: str,
        episode_id: str | None = None,
        sandbox_kind: str | None = None,
        workdir: str = "",
        trial_name: str | None = None,
        harbor_session_id: str | None = None,
        harbor_trial_id: str | None = None,
        container_arch: str | None = None,
        has_timeout_binary: bool | None = None,
        provenance_fields: dict[str, Any] | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        from .config import Strategy, Topology  # noqa: PLC0415

        self.dir = Path(episode_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.episode_id = episode_id or new_episode_id()
        self.config = config
        self.on_event = on_event
        # Called (by AgentLog.llm) when a call of any role fails with a
        # provider error after its retries; the orchestrator stops the episode.
        self.on_provider_error: Callable[[BaseException], None] | None = None
        self._lock = threading.RLock()
        self._seq = 0
        self._agent_seq: dict[str, int] = defaultdict(int)
        self._ids: dict[str, int] = defaultdict(int)
        self.calls: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []
        self.steps: list[dict[str, Any]] = []
        self.spawns: list[dict[str, Any]] = []
        self.dispatches: list[dict[str, Any]] = []
        self.arbitrations: list[dict[str, Any]] = []
        self.n_transcript = 0
        self.dropped = 0
        self._tokens_total = 0
        self._finalized = False
        self.record: dict[str, Any] = {}
        self._fh: dict[str, Any] = {}
        self.t_start = time.time()

        subagents = config.topology is Topology.SUBAGENTS
        prov = dict(provenance_fields) if provenance_fields is not None else provenance()
        self._static: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "experiment": config.name,
            "config": config.to_dict(),
            "config_hash": config.config_hash(),
            "strategy": config.subagent.strategy.value if subagents else None,
            "scope": config.subagent.scope.value if subagents else None,
            "topology": config.topology.value,
            "orchestrator_model": config.orchestrator.model,
            "subagent_model": config.resolved_subagent_model().model if subagents else None,
            "arbiter_model": (
                config.resolved_arbiter_model().model
                if subagents and config.subagent.strategy is Strategy.ARBITER
                else None
            ),
            "task_name": task_name,
            "trial_name": trial_name,
            "harbor_session_id": harbor_session_id,
            "harbor_trial_id": harbor_trial_id,
            "logs_dir": str(self.dir),
            "instruction": instruction,
            "harness_version": prov.get("harness_version", ""),
            "git_commit": prov.get("git_commit"),
            "git_dirty": prov.get("git_dirty"),
            "source_sha256": prov.get("source_sha256"),
            "harbor_version": prov.get("harbor_version"),
            "python_version": prov.get("python_version", platform.python_version()),
            "host_platform": prov.get("host_platform", platform.platform()),
            "host_machine": prov.get("host_machine", platform.machine()),
            "container_arch": container_arch,
            "sandbox_kind": sandbox_kind or config.sandbox.kind,
            "workdir": workdir or "",
            "has_timeout_binary": has_timeout_binary,
            "t_start": self.t_start,
        }
        for fname in RECORD_TYPES:
            if fname != "episode.json":
                (self.dir / fname).touch()
        self._write_episode(self._running_record())

    # ---- ids and counters -----------------------------------------------------

    def new_id(self, kind: str) -> str:
        with self._lock:
            self._ids[kind] += 1
            n = self._ids[kind]
        return f"{_ID_PREFIX.get(kind, kind)}{n:04d}"

    def next_number(self, kind: str) -> int:
        """1-based counter per kind (e.g. ``sub`` for agent ids ``sub-<n>``)."""
        with self._lock:
            self._ids["#" + kind] += 1
            return self._ids["#" + kind]

    def alloc(self, agent_id: str) -> tuple[int, int]:
        """(global seq, agent_seq) for an event that is starting now."""
        with self._lock:
            seq = self._seq
            self._seq += 1
            aseq = self._agent_seq[agent_id]
            self._agent_seq[agent_id] += 1
            return seq, aseq

    # ---- writing ----------------------------------------------------------------

    @property
    def finalized(self) -> bool:
        return self._finalized

    def _append(self, fname: str, rec: Any, store: list | None) -> dict[str, Any] | None:
        d = asdict(rec) if dataclasses.is_dataclass(rec) else dict(rec)
        line = json.dumps(d, default=str, ensure_ascii=False)
        with self._lock:
            if self._finalized:
                self.dropped += 1
                log.warning("episode %s already finalized; dropping a %s record", self.episode_id, fname)
                return None
            fh = self._fh.get(fname)
            if fh is None:
                fh = self._fh[fname] = open(self.dir / fname, "a", encoding="utf-8")
            fh.write(line + "\n")
            fh.flush()
            if store is not None:
                store.append(d)
            if fname == "calls.jsonl":
                self._tokens_total += int(d.get("prompt_tokens") or 0) + int(d.get("completion_tokens") or 0)
            elif fname == "transcript.jsonl":
                self.n_transcript += 1
        if self.on_event is not None:
            try:
                self.on_event(fname.split(".")[0], d)
            except Exception:  # noqa: BLE001 - a progress hook must never break recording
                log.exception("on_event hook failed")
        return d

    def record_call(self, rec: CallRecord) -> dict[str, Any] | None:
        return self._append("calls.jsonl", rec, self.calls)

    def record_tool(self, rec: ToolRecord) -> dict[str, Any] | None:
        return self._append("tools.jsonl", rec, self.tools)

    def record_step(self, rec: StepRecord) -> dict[str, Any] | None:
        return self._append("steps.jsonl", rec, self.steps)

    def record_spawn(self, rec: SpawnRecord) -> dict[str, Any] | None:
        return self._append("spawns.jsonl", rec, self.spawns)

    def record_dispatch(self, rec: DispatchRecord) -> dict[str, Any] | None:
        return self._append("dispatches.jsonl", rec, self.dispatches)

    def record_arbitration(self, rec: ArbitrationRecord) -> dict[str, Any] | None:
        return self._append("arbitrations.jsonl", rec, self.arbitrations)

    def record_transcript(self, rec: TranscriptRecord) -> dict[str, Any] | None:
        return self._append("transcript.jsonl", rec, None)

    def _write_episode(self, rec: EpisodeRecord) -> None:
        d = asdict(rec)
        tmp = self.dir / f".episode.json.{uuid.uuid4().hex[:6]}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, default=str, ensure_ascii=False)
        os.replace(tmp, self.dir / "episode.json")
        self.record = d

    # ---- aggregates ---------------------------------------------------------------

    def total_tokens(self) -> int:
        """prompt + completion tokens over every call recorded so far."""
        with self._lock:
            return self._tokens_total

    def _running_record(self) -> EpisodeRecord:
        return EpisodeRecord(
            **self._static,
            status="running",
            t_end=None,
            wall_clock_sec=None,
            t_first_call=None,
            exit_reason=None,
            error=None,
            cancel_kill=None,
            final_summary=None,
            orchestrator_steps=0,
            spawn_calls=0,
            dispatches=0,
            redispatches=0,
            arbitrations=0,
            requested_total=0,
            granted_total=0,
            instruction_following=None,
            n_calls={},
            n_tool_calls={},
            tokens={},
            cost_usd={},
            llm_sec={},
            tool_sec={},
            n_retries=0,
            n_stream_repairs=0,
            backoff_sec=0.0,
            n_provider_errors=0,
            n_command_timeouts=0,
            n_context_trims=0,
            files_modified=None,
            n_files_modified=None,
            git_diff_stat=None,
            harness_exec_n=None,
            harness_exec_sec=None,
            harness_exec_sum_sec=None,
            reward=None,
            resolved=None,
        )

    def aggregates(self) -> dict[str, Any]:
        """Every EpisodeRecord aggregate, computed from the records written."""
        with self._lock:
            calls, tools = list(self.calls), list(self.tools)
            steps, spawns = list(self.steps), list(self.spawns)
            dispatches, arbs = list(self.dispatches), list(self.arbitrations)
        by_role = {r: [c for c in calls if c["role"] == r] for r in ROLES}
        tools_by_role = {r: [t for t in tools if t["role"] == r] for r in ROLES}
        tokens = {}
        for r, cs in by_role.items():
            reasoning = [c["reasoning_tokens"] for c in cs if c["reasoning_tokens"] is not None]
            tokens[r] = {
                "prompt": sum(c["prompt_tokens"] for c in cs),
                "completion": sum(c["completion_tokens"] for c in cs),
                "cached": sum(c["cached_tokens"] for c in cs),
                "uncached_prompt": sum(c["uncached_prompt_tokens"] for c in cs),
                "reasoning": sum(reasoning) if reasoning else None,
            }
        cost = {r: sum_or_none(c["cost_usd"] for c in cs) for r, cs in by_role.items()}
        cost["total"] = sum_or_none(c["cost_usd"] for c in calls)
        first_group = min(spawns, key=lambda s: s["t_start"])["group_id"] if spawns else None
        ratios = [s["granted"] / s["requested"] for s in spawns if s["requested"] > 0]
        return {
            "t_first_call": min((c["t_start"] for c in calls), default=None),
            "orchestrator_steps": len(steps),
            "spawn_calls": len(spawns),
            "dispatches": len(dispatches),
            "redispatches": sum(1 for d in dispatches if d["group_id"] != first_group),
            "arbitrations": len(arbs),
            "requested_total": sum(s["requested"] for s in spawns),
            "granted_total": sum(s["granted"] for s in spawns),
            "instruction_following": (sum(ratios) / len(ratios)) if ratios else None,
            "n_calls": {r: len(cs) for r, cs in by_role.items()},
            "n_tool_calls": {r: len(ts) for r, ts in tools_by_role.items()},
            "tokens": tokens,
            "cost_usd": cost,
            "llm_sec": {r: sum(c["latency_sec"] for c in cs) for r, cs in by_role.items()},
            "tool_sec": {r: sum(t["duration_sec"] for t in ts) for r, ts in tools_by_role.items()},
            "n_retries": sum(max(c["n_attempts"] - 1, 0) for c in calls),
            "n_stream_repairs": sum(c.get("stream_repairs") or 0 for c in calls),
            "backoff_sec": float(sum(c["backoff_sec"] for c in calls)),
            "n_provider_errors": sum(1 for c in calls if is_provider_error(c.get("error"))),
            "n_command_timeouts": sum(1 for t in tools if t["timed_out"]),
            "n_context_trims": sum(d["context_trims"] for d in dispatches),
        }

    def role_totals(self) -> dict[str, Any]:
        """Token/cost totals for harbor's AgentContext.

        ``{"roles": {role: {calls, prompt, completion, cached, cost_usd}},
        "models": {model: {n_input_tokens, n_cache_tokens, n_output_tokens,
        cost_usd}}, "total": {...same keys as a model...}}``. Input tokens
        include cached tokens (harbor's convention, same as ours).
        """
        with self._lock:
            calls = list(self.calls)
        roles: dict[str, dict[str, Any]] = {
            r: {"calls": 0, "prompt": 0, "completion": 0, "cached": 0, "cost_usd": 0.0} for r in ROLES
        }
        models: dict[str, dict[str, Any]] = {}
        for c in calls:
            r = roles.setdefault(c["role"], {"calls": 0, "prompt": 0, "completion": 0, "cached": 0, "cost_usd": 0.0})
            r["calls"] += 1
            r["prompt"] += c["prompt_tokens"]
            r["completion"] += c["completion_tokens"]
            r["cached"] += c["cached_tokens"]
            r["cost_usd"] = sum_or_none([r["cost_usd"], c["cost_usd"]])
            m = models.setdefault(
                c["model"], {"n_input_tokens": 0, "n_cache_tokens": 0, "n_output_tokens": 0, "cost_usd": 0.0}
            )
            m["n_input_tokens"] += c["prompt_tokens"]
            m["n_cache_tokens"] += c["cached_tokens"]
            m["n_output_tokens"] += c["completion_tokens"]
            m["cost_usd"] = sum_or_none([m["cost_usd"], c["cost_usd"]])
        total = {
            "n_input_tokens": sum(m["n_input_tokens"] for m in models.values()),
            "n_cache_tokens": sum(m["n_cache_tokens"] for m in models.values()),
            "n_output_tokens": sum(m["n_output_tokens"] for m in models.values()),
            "cost_usd": sum_or_none(m["cost_usd"] for m in models.values()),
        }
        return {"roles": roles, "models": models, "total": total}

    def summary(self) -> dict[str, Any]:
        """Compact episode summary (harbor ``AgentContext.metadata``)."""
        agg = self.aggregates()
        rec = self.record
        keys = (
            "episode_id", "status", "experiment", "topology", "strategy", "scope", "orchestrator_model",
            "subagent_model", "arbiter_model", "task_name", "trial_name", "logs_dir", "exit_reason", "error",
            "wall_clock_sec", "container_arch", "has_timeout_binary", "workdir", "git_commit", "git_dirty",
            "source_sha256", "harbor_version", "harness_version", "config_hash", "n_files_modified",
            "harness_exec_sec", "harness_exec_sum_sec", "cancel_kill",
        )
        out = {k: rec.get(k) for k in keys}
        if out.get("error"):
            out["error"] = str(out["error"])[-2000:]
        out.update({k: agg[k] for k in (
            "orchestrator_steps", "spawn_calls", "dispatches", "redispatches", "arbitrations",
            "requested_total", "granted_total", "instruction_following", "n_calls", "n_tool_calls",
            "tokens", "cost_usd", "llm_sec", "tool_sec", "n_retries", "n_stream_repairs", "backoff_sec", "n_provider_errors",
            "n_command_timeouts", "n_context_trims",
        )})
        return out

    def update_static(self, **fields: Any) -> None:
        """Change static fields (e.g. workdir) and rewrite the running record."""
        unknown = set(fields) - set(self._static)
        if unknown:
            raise KeyError(f"not static EpisodeRecord fields: {sorted(unknown)}")
        with self._lock:
            self._static.update(fields)
            if not self._finalized:
                self._write_episode(self._running_record())

    def finalize(
        self,
        exit_reason: str,
        *,
        error: str | None = None,
        final_summary: str | None = None,
        files_modified: list[str] | None = None,
        n_files_modified: int | None = None,
        git_diff_stat: str | None = None,
        harness_exec_n: int | None = None,
        harness_exec_sec: float | None = None,
        harness_exec_sum_sec: float | None = None,
        cancel_kill: dict[str, Any] | None = None,
        reward: float | None = None,
        resolved: bool | None = None,
    ) -> dict[str, Any]:
        """Write the final ``episode.json`` (first call wins; later calls
        return the existing record unchanged) and close the JSONL files.
        Measurements the caller did not take stay None. ``harness_exec_sec``
        is the union of the measurement execs, ``harness_exec_sum_sec`` their
        plain sum (:meth:`subagents_bench.subagent.MeteredEnv.union_sec`)."""
        with self._lock:
            if self._finalized:
                return self.record
            t_end = time.time()
            agg = self.aggregates()
            rec = EpisodeRecord(
                **self._static,
                status="done",
                t_end=t_end,
                wall_clock_sec=t_end - self.t_start,
                exit_reason=exit_reason,
                error=error,
                cancel_kill=cancel_kill,
                final_summary=final_summary,
                files_modified=files_modified,
                n_files_modified=n_files_modified,
                git_diff_stat=git_diff_stat,
                harness_exec_n=harness_exec_n,
                harness_exec_sec=harness_exec_sec,
                harness_exec_sum_sec=harness_exec_sum_sec,
                reward=reward,
                resolved=resolved,
                **agg,
            )
            self._write_episode(rec)
            self._finalized = True
            for fh in self._fh.values():
                try:
                    fh.close()
                except OSError:
                    pass
            self._fh.clear()
            return self.record

    def set_outcome(self, reward: float | None, resolved: bool | None) -> None:
        """Record the verifier outcome in the final episode.json (local runs)."""
        self.update_record(reward=reward, resolved=resolved)

    def update_record(self, **fields: Any) -> None:
        """Rewrite ``episode.json`` with ``fields`` changed (e.g. what happened
        after the episode was finalized: the verifier outcome, harbor's
        cancel-time kill). Unknown names raise."""
        unknown = set(fields) - {f.name for f in dataclasses.fields(EpisodeRecord)}
        if unknown:
            raise KeyError(f"not EpisodeRecord fields: {sorted(unknown)}")
        with self._lock:
            if not self.record:
                return
            self._write_episode(EpisodeRecord(**{**self.record, **fields}))


# Back-compat name for older callers.
Telemetry = EpisodeRecorder


# ---------------------------------------------------------------------------
# per-agent facade
# ---------------------------------------------------------------------------


def complete_stoppable(
    client: Any,
    system: str,
    messages: list[Message],
    tools: list[ToolSpec],
    stop_event: threading.Event | None,
) -> LLMResponse:
    """``client.complete`` that gives up when ``stop_event`` is set.

    The call runs in a daemon thread; on stop the request is abandoned (it
    finishes in the background and its result is discarded) and
    :class:`AgentStopped` is raised.
    """
    if stop_event is None:
        return client.complete(system, messages, tools)
    if stop_event.is_set():
        raise AgentStopped("episode stopped before an LLM call")
    box: dict[str, Any] = {}
    done = threading.Event()

    def target() -> None:
        try:
            box["result"] = client.complete(system, messages, tools)
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = e
        finally:
            done.set()

    threading.Thread(target=target, name="llm-call", daemon=True).start()
    while not done.wait(_POLL_SEC):
        if stop_event.is_set():
            raise AgentStopped("episode stopped during an LLM call")
    if "error" in box:
        raise box["error"]
    return box["result"]


@dataclass
class ToolEvent:
    """Mutable bits a tool runner may set while it runs (spawn: its group)."""

    group_id: str | None = None


@dataclass
class AgentLog:
    """Recording facade for one agent context (one transcript)."""

    recorder: EpisodeRecorder
    agent_id: str
    role: str
    dispatch_id: str | None = None
    group_id: str | None = None
    arbitration_id: str | None = None
    stop_event: threading.Event | None = None
    # call_id of the most recent llm() call, set before the call is made (so
    # it is known even when the call raises).
    last_call_id: str | None = None
    _index: int = field(default=0, repr=False)

    def transcript(self, message: Message, call_id: str | None = None) -> None:
        self.recorder.record_transcript(
            TranscriptRecord(
                episode_id=self.recorder.episode_id,
                agent_id=self.agent_id,
                role=self.role,
                dispatch_id=self.dispatch_id,
                index=self._index,
                t=time.time(),
                call_id=call_id,
                **_message_fields(message),
            )
        )
        self._index += 1

    def _ids(self, call_id: str, seq: int, aseq: int, step: int) -> dict[str, Any]:
        return {
            "episode_id": self.recorder.episode_id,
            "call_id": call_id,
            "seq": seq,
            "role": self.role,
            "agent_id": self.agent_id,
            "dispatch_id": self.dispatch_id,
            "group_id": self.group_id,
            "arbitration_id": self.arbitration_id,
            "step": step,
            "agent_seq": aseq,
        }

    def llm(
        self,
        client: Any,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        step: int,
        context_trimmed_chars: int = 0,
    ) -> tuple[LLMResponse, str]:
        """One recorded LLM call -> (response, call_id).

        On any exception a CallRecord with ``error`` set is written and the
        exception re-raised (AgentStopped when the stop event fired).
        """
        seq, aseq = self.recorder.alloc(self.agent_id)
        call_id = self.recorder.new_id("call")
        self.last_call_id = call_id
        ids = self._ids(call_id, seq, aseq, step)
        snapshot = list(messages)
        t0 = time.time()
        try:
            resp = complete_stoppable(client, system, snapshot, tools, self.stop_event)
        except BaseException as e:
            t1 = time.time()
            attempts = list(getattr(e, "attempts", []) or []) if isinstance(e, LLMError) else []
            if isinstance(e, AgentStopped):
                msg = "cancelled: the episode was stopped while this call was in flight; usage unknown"
            else:
                msg = f"{type(e).__name__}: {e}"
            try:
                # The client's own RequestInfo when it attached one (exact
                # hashes); otherwise rebuilt from the messages.
                req = getattr(e, "request", None) if isinstance(e, LLMError) else None
                if req is None:
                    req = request_info_for(client, system, snapshot, tools)
                self.recorder.record_call(
                    failed_call_record(
                        ids=ids, cfg=client.cfg, request=req, t_start=t0, t_end=t1, attempts=attempts,
                        error=msg[:4000], context_trimmed_chars=context_trimmed_chars,
                    )
                )
            except Exception:  # noqa: BLE001 - never mask the original error
                log.exception("could not record a failed call")
            if isinstance(e, LLMError) and is_provider_error(msg) and self.recorder.on_provider_error is not None:
                try:
                    self.recorder.on_provider_error(e)
                except Exception:  # noqa: BLE001 - never mask the original error
                    log.exception("on_provider_error failed")
            raise
        self.recorder.record_call(
            call_record_from_response(resp, ids=ids, cfg=client.cfg, context_trimmed_chars=context_trimmed_chars)
        )
        return resp, call_id

    def tool(
        self,
        tc: ToolCall,
        run: Callable[[ToolEvent], ToolOutput],
        *,
        step: int,
        call_id: str,
        event: ToolEvent | None = None,
    ) -> ToolOutput:
        """One recorded tool invocation. ``run(event)`` does the work; pass
        your own ``event`` to read what the runner set on it afterwards."""
        seq, aseq = self.recorder.alloc(self.agent_id)
        tool_event_id = self.recorder.new_id("tool")
        ev = event if event is not None else ToolEvent()
        if ev.group_id is None:
            ev.group_id = self.group_id
        t0 = time.time()
        raised: BaseException | None = None
        try:
            out = run(ev)
        except BaseException as e:
            # Recorded, then re-raised: AgentStopped (cancellation),
            # BillingError (aborts the episode) or a harness bug. Tool
            # failures proper never get here (run_tool returns them).
            raised = e
            if isinstance(e, AgentStopped):
                out = ToolOutput(text="[cancelled: the episode was stopped]", is_error=True, error="cancelled")
            else:
                msg = f"{type(e).__name__}: {e}"
                out = ToolOutput(text=f"[error] {tc.name} failed: {msg}", is_error=True, error=msg[:4000])
        t1 = time.time()
        self.recorder.record_tool(
            ToolRecord(
                episode_id=self.recorder.episode_id,
                tool_event_id=tool_event_id,
                seq=seq,
                role=self.role,
                agent_id=self.agent_id,
                dispatch_id=self.dispatch_id,
                group_id=ev.group_id,
                arbitration_id=self.arbitration_id,
                step=step,
                agent_seq=aseq,
                call_id=call_id,
                tool_call_id=tc.id,
                name=tc.name,
                args=_cap_args(tc.arguments),
                t_start=t0,
                t_end=t1,
                duration_sec=t1 - t0,
                exit_code=out.exit_code,
                timed_out=bool(out.timed_out),
                timeout_sec=out.timeout_sec,
                output_chars_raw=int(out.output_chars_raw),
                output_chars_returned=len(out.text),
                truncated=bool(out.truncated),
                is_error=bool(out.is_error),
                error=out.error,
                cwd_before=out.cwd_before,
                cwd_after=out.cwd_after,
                n_exec=int(out.n_exec),
                exec_sec=float(out.exec_sec),
            )
        )
        if raised is not None:
            raise raised
        return out


# ---------------------------------------------------------------------------
# validation helper (tests, check scripts)
# ---------------------------------------------------------------------------


def validate_episode_dir(episode_dir: str | os.PathLike) -> list[str]:
    """Schema problems in every line of every record file (empty = valid)."""
    d = Path(episode_dir)
    problems: list[str] = []
    for fname, cls in RECORD_TYPES.items():
        path = d / fname
        if not path.exists():
            problems.append(f"{fname}: missing")
            continue
        if fname == "episode.json":
            try:
                recs = [json.loads(path.read_text())]
            except ValueError as e:
                problems.append(f"{fname}: invalid JSON ({e})")
                continue
        else:
            recs = []
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    recs.append(json.loads(line))
                except ValueError as e:
                    problems.append(f"{fname}:{i}: invalid JSON ({e})")
        for i, rec in enumerate(recs, 1):
            for p in validate_record(cls, rec):
                problems.append(f"{fname}:{i}: {p}")
    return problems


def read_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


__all__ = [
    "AgentLog",
    "EpisodeRecorder",
    "Telemetry",
    "ToolEvent",
    "call_record_from_response",
    "complete_stoppable",
    "failed_call_record",
    "is_provider_error",
    "new_episode_id",
    "provenance",
    "read_jsonl",
    "request_info_for",
    "validate_episode_dir",
]
