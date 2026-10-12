"""Synthetic harbor jobs dirs, built from the ``schema.py`` dataclasses.

The analysis tests need episodes whose structure (and so whose critical path)
is known exactly. :class:`EpisodeBuilder` lays an episode out on one clock:
orchestrator steps (an LLM call, then each requested tool in order), spawn
groups (parallel members list-scheduled on ``concurrency`` slots, sequential
members one after another, arbiter candidates + judge rounds + debate rounds +
apply time), with optional harness gaps between events. Every record is
constructed through its dataclass (``CallRecord(**fields)`` etc.), so a field
missing from the synthetic data fails loudly, and ``episode.json`` aggregates
are computed from the records exactly as the contract states.

:func:`write_trial` wraps an episode into a harbor-shaped trial dir
(``result.json`` modeled on a real harbor 0.24.0 result), and
:func:`make_sweep` writes a whole sweep: ``<root>/<sweep>__r<round>__<cond>/``
job dirs, ``sweep_manifest.jsonl``, ``tasks_meta.json`` and resolved condition
configs, plus edge cases (environment failure without an agent dir, a
``running`` episode whose agent raised so harbor skipped the verifier, a
missing ``result.json``, a zero-dispatch episode and a billing error).

EVERY NUMBER HERE IS SYNTHETIC: latencies, token counts, rewards and model
parameter counts are test fixtures, not measurements.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml

from subagents_bench.schema import (
    SCHEMA_VERSION,
    ArbitrationRecord,
    CallRecord,
    DispatchRecord,
    EpisodeRecord,
    SpawnRecord,
    StepRecord,
    ToolRecord,
    TranscriptRecord,
)

ORCH = "deepseek-ai/DeepSeek-V4.1-Flash"
GEMMA = "nvidia/Gemma-4-31B-IT-NVFP4"
# Runtime price table (in, cached, out) USD/1M = the sail_balanced list prices,
# so cost_runtime_total equals cost_sail_balanced_total on synthetic data.
RUNTIME_PRICES = {ORCH: (0.07, 0.005, 0.22), GEMMA: (0.11, 0.06, 0.32)}
ROLES = ("orchestrator", "subagent", "arbiter", "debate")
SCOPE_TOOL = {"easy": "read_file", "medium": "bash", "hard": "edit_file"}
SHELL_TOOLS = {"bash", "read_file", "list_dir", "search"}
T0 = 1_791_720_000.0  # 2026-10-11T12:00:00Z


# ---------------------------------------------------------------------------
# Event specs
# ---------------------------------------------------------------------------


@dataclass
class C:
    """An LLM call: ``latency`` is the whole logical call (backoff included)."""

    latency: float
    prompt: int = 1000
    completion: int = 100
    cached: int = 0
    reasoning: int | None = None
    ttft: float | None = None
    attempts: int = 1
    backoff: float = 0.0
    error: str | None = None


@dataclass
class T:
    """A tool invocation."""

    name: str
    duration: float
    timed_out: bool = False
    is_error: bool = False


@dataclass
class Judge:
    """One judge round; ``debates`` (one per candidate) run concurrently before the judge call.

    ``judge=None`` is a round aborted before its judge call (a cancel during the
    debate): the round is recorded with ``judge_call_id`` None, as the arbiter does."""

    judge: C | None
    debates: list[C] = field(default_factory=list)
    winner: int = 0
    confidence: float | None = 0.9
    agreement: bool | None = True
    parse_ok: bool = True


@dataclass
class Arb:
    candidates: list[list]  # one chain (list of C/T) per candidate
    rounds: list[Judge]
    apply_sec: float = 0.0
    isolation: str = "copy"
    applied_files: list[str] = field(default_factory=lambda: ["src/main.py"])
    isolation_sec: float = 0.0  # private copies before the candidates start
    diff_sec: float = 0.0  # diff summaries between the candidates' end and the first judge round
    leaked_files: list[str] = field(default_factory=list)  # copy isolation: real-workdir files a candidate wrote
    n_path_remaps: int = 0


@dataclass
class Spawn:
    """A spawn_subagents tool call. ``members`` for parallel/sequential, ``arbitrations`` for arbiter."""

    members: list[list] = field(default_factory=list)
    arbitrations: list[Arb] = field(default_factory=list)
    strategy: str | None = None  # default: the builder's strategy
    requested: int | None = None  # default: granted
    concurrency: int | None = None  # default: max_concurrent (parallel), 1 (sequential), all (arbiter)
    start_overhead: float = 0.0  # spawn bookkeeping before members start (marker file, ...)
    end_overhead: float = 0.0  # after members end (files_modified scan)
    member_gap: float = 0.0  # harness gap between events inside each dispatch chain
    statuses: list[str] | None = None


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _chain_sum(chain: list) -> float:
    return sum(e.latency if isinstance(e, C) else e.duration for e in chain)


# ---------------------------------------------------------------------------
# Episode builder
# ---------------------------------------------------------------------------


class EpisodeBuilder:
    def __init__(
        self,
        *,
        episode_id: str = "ep0001",
        condition: str = "cond",
        topology: str = "subagents",
        strategy: str | None = "parallel",
        scope: str | None = "hard",
        orch_model: str = ORCH,
        sub_model: str | None = GEMMA,
        arbiter_model: str | None = None,
        task_name: str = "fix-git",
        trial_name: str | None = None,
        t0: float = T0,
        gap: float = 0.0,
        max_concurrent: int = 3,
        n_candidates: int = 3,
        steps_budget: int = 5,
        provider: str = "sail",
        window: str | None = "balanced",
    ):
        self.episode_id = episode_id
        self.condition = condition
        self.topology = topology
        self.strategy = strategy if topology == "subagents" else None
        self.scope = scope if topology == "subagents" else None
        self.orch_model = orch_model
        self.sub_model = (sub_model or orch_model) if topology == "subagents" else None
        self.arbiter_model = (arbiter_model or orch_model) if self.strategy == "arbiter" else None
        self.task_name = task_name
        self.trial_name = trial_name or f"{task_name}__{episode_id[-7:]}"
        self.t0 = t0
        self.t = t0
        self.gap = gap
        self.max_concurrent = max_concurrent
        self.n_candidates = n_candidates
        self.steps_budget = steps_budget
        self.provider = provider
        self.window = window
        self.max_dispatches = max_concurrent * steps_budget
        self.calls: list[dict] = []
        self.tools: list[dict] = []
        self.steps: list[dict] = []
        self.spawns: list[dict] = []
        self.dispatches: list[dict] = []
        self.arbitrations: list[dict] = []
        self.transcript: list[dict] = []
        self._seq = 0
        self._agent_seq: dict[str, int] = defaultdict(int)
        self._n = defaultdict(int)
        self.step_no = 0
        self.episode: dict | None = None

    # -- ids / counters -------------------------------------------------------
    def _id(self, kind: str) -> str:
        self._n[kind] += 1
        return f"{kind}{self._n[kind]:03d}"

    def _next_seq(self) -> int:
        s = self._seq
        self._seq += 1
        return s

    def _next_agent_seq(self, agent: str) -> int:
        s = self._agent_seq[agent]
        self._agent_seq[agent] += 1
        return s

    def config_dict(self) -> dict:
        return config_dict(self.condition, self.topology, self.strategy, self.scope, self.orch_model,
                           self.sub_model, self.max_concurrent, self.n_candidates, self.steps_budget, self.window)

    # -- low-level records ----------------------------------------------------
    def _call(self, spec: C, *, t_start: float, role: str, agent_id: str, model: str, step: int,
              tool_names: list[str] = (), dispatch_id: str | None = None, group_id: str | None = None,
              arbitration_id: str | None = None) -> dict:
        cid = self._id("call")
        t_end = t_start + spec.latency
        n_att = max(spec.attempts, 1)
        b = spec.backoff / (n_att - 1) if n_att > 1 else 0.0
        attempts = [
            {"attempt": i, "t_start": t_start + (i - 1) * b, "t_end": t_start + (i - 1) * b, "status": 429,
             "error_type": "RateLimited", "error": "rate limited", "retry_after_sec": None, "backoff_sec": b}
            for i in range(1, n_att)
        ]
        t_req = t_start + spec.backoff
        attempts.append({"attempt": n_att, "t_start": t_req, "t_end": t_end, "status": 200, "error_type": None,
                         "error": None, "retry_after_sec": None, "backoff_sec": 0.0})
        ttft = spec.ttft if spec.ttft is not None else min(0.5, (spec.latency - spec.backoff) / 4)
        t_first = t_req + ttft
        decode = t_end - t_first
        failed = spec.error is not None
        prompt, completion, cached = (0, 0, 0) if failed else (spec.prompt, spec.completion, spec.cached)
        reasoning = None if failed else spec.reasoning
        p_in, p_cache, p_out = RUNTIME_PRICES.get(model, (0.0, None, 0.0))
        cost = ((prompt - cached) * p_in + cached * (p_cache if p_cache is not None else p_in) + completion * p_out) / 1e6
        offsets = [round(ttft * 1000 + k * decode * 1000 / 5, 6) for k in range(6)]
        visible = completion - (reasoning or 0)
        rec = CallRecord(
            episode_id=self.episode_id, call_id=cid, seq=self._next_seq(), role=role, agent_id=agent_id,
            dispatch_id=dispatch_id, group_id=group_id, arbitration_id=arbitration_id, step=step,
            agent_seq=self._next_agent_seq(agent_id), model=model, provider=self.provider,
            base_url="https://api.sailresearch.com/v1", completion_window=self.window, max_tokens=8192,
            temperature=0.7, extra_body={}, n_messages=2 + 2 * step, n_tools=0 if role == "debate" else 6,
            prompt_chars=prompt * 4, system_hash="a1b2c3d4e5f6", tools_hash="0f9e8d7c6b5a",
            msg_hashes=["0123456789ab"] * (2 + 2 * step), msg_chars=[400] * (2 + 2 * step),
            reasoning_passback=True, context_trimmed_chars=0, t_start=t_start, t_end=t_end,
            latency_sec=spec.latency, t_request_sent=t_req, t_headers=t_req + ttft / 2,
            t_first_chunk=None if failed else t_first,
            t_first_reasoning=t_first if reasoning else None,
            t_first_content=t_first if visible > 0 else None,
            t_first_tool_call=t_first if tool_names else None,
            t_last_chunk=None if failed else t_end, ttft_sec=None if failed else ttft,
            decode_sec=None if failed else decode, n_chunks=0 if failed else 6,
            chunk_offsets_ms=[] if failed else offsets, streamed=True, n_attempts=n_att,
            backoff_sec=spec.backoff, attempts=attempts, prompt_tokens=prompt, completion_tokens=completion,
            cached_tokens=cached, uncached_prompt_tokens=prompt - cached, reasoning_tokens=reasoning,
            usage_source="provider",
            usage_raw=None if failed else {"prompt_tokens": prompt, "completion_tokens": completion},
            finish_reason="error" if failed else ("tool_calls" if tool_names else "stop"),
            n_tool_calls=len(tool_names), tool_call_names=list(tool_names), content_chars=max(visible, 0) * 4,
            reasoning_chars=(reasoning or 0) * 4, response_id=None if failed else f"resp-{cid}",
            response_headers={"content-type": "text/event-stream"}, cost_usd=cost, price_in=p_in,
            price_out=p_out, price_cache_read=p_cache, provider_reported_cost=None, error=spec.error,
        )
        d = asdict(rec)
        self.calls.append(d)
        return d

    def _tool(self, spec: T, *, t_start: float, role: str, agent_id: str, step: int, call_id: str,
              dispatch_id: str | None = None, group_id: str | None = None, arbitration_id: str | None = None,
              duration: float | None = None) -> dict:
        dur = spec.duration if duration is None else duration
        shell = spec.name in SHELL_TOOLS
        rec = ToolRecord(
            episode_id=self.episode_id, tool_event_id=self._id("tool"), seq=self._next_seq(), role=role,
            agent_id=agent_id, dispatch_id=dispatch_id, group_id=group_id, arbitration_id=arbitration_id,
            step=step, agent_seq=self._next_agent_seq(agent_id), call_id=call_id,
            tool_call_id=f"tc-{self._n['tool']}", name=spec.name, args={"command": "ls"} if shell else {},
            t_start=t_start, t_end=t_start + dur, duration_sec=dur,
            exit_code=(124 if spec.timed_out else 0) if shell else None, timed_out=spec.timed_out,
            timeout_sec=180 if spec.name == "bash" else None, output_chars_raw=120, output_chars_returned=120,
            truncated=False, is_error=spec.is_error, error="failed" if spec.is_error else None,
            cwd_before="/app" if spec.name == "bash" else None, cwd_after="/app" if spec.name == "bash" else None,
            n_exec=1 if shell else 0, exec_sec=dur * 0.9 if shell else 0.0,
        )
        d = asdict(rec)
        self.tools.append(d)
        return d

    # -- dispatches -----------------------------------------------------------
    def _dispatch(self, chain: list, *, t_queued: float, t_start: float, group_id: str, instr_idx: int,
                  strategy: str, member_gap: float, status: str = "submitted", arbitration_id: str | None = None,
                  candidate_index: int | None = None, isolation: str = "shared") -> dict:
        n = self._n["sub"] = self._n["sub"] + 1
        agent_id, dispatch_id = f"sub-{n}", f"d{n:03d}"
        t, step, last_call = t_start, 0, None
        first_call_idx, first_tool_idx = len(self.calls), len(self.tools)
        for k, ev in enumerate(chain):
            if k:
                t += member_gap
            if isinstance(ev, C):
                step += 1
                nxt = chain[k + 1] if k + 1 < len(chain) and isinstance(chain[k + 1], T) else None
                last_call = self._call(ev, t_start=t, role="subagent", agent_id=agent_id, model=self.sub_model,
                                       step=step, tool_names=[nxt.name] if nxt else [], dispatch_id=dispatch_id,
                                       group_id=group_id, arbitration_id=arbitration_id)
                t = last_call["t_end"]
            else:
                rec = self._tool(ev, t_start=t, role="subagent", agent_id=agent_id, step=step,
                                 call_id=last_call["call_id"] if last_call else "none", dispatch_id=dispatch_id,
                                 group_id=group_id, arbitration_id=arbitration_id)
                t = rec["t_end"]
        calls, tools = self.calls[first_call_idx:], self.tools[first_tool_idx:]
        calls = [c for c in calls if c["dispatch_id"] == dispatch_id]
        tools = [x for x in tools if x["dispatch_id"] == dispatch_id]
        counts: dict[str, int] = defaultdict(int)
        for x in tools:
            counts[x["name"]] += 1
        reasoning = [c["reasoning_tokens"] for c in calls if c["reasoning_tokens"] is not None]
        rec = DispatchRecord(
            episode_id=self.episode_id, dispatch_id=dispatch_id, agent_id=agent_id, group_id=group_id,
            arbitration_id=arbitration_id, candidate_index=candidate_index, instruction_index=instr_idx,
            orchestrator_step=self.step_no, strategy=strategy, scope=self.scope or "hard", model=self.sub_model,
            tools=["read_file", "list_dir", "search", "submit"], instruction=f"instruction {instr_idx}",
            report=f"report from {agent_id}", status=status, steps_used=step, max_steps=8, t_queued=t_queued,
            t_start=t_start, t_end=t, duration_sec=t - t_start, queue_sec=t_start - t_queued,
            n_llm_calls=len(calls), prompt_tokens=sum(c["prompt_tokens"] for c in calls),
            completion_tokens=sum(c["completion_tokens"] for c in calls),
            cached_tokens=sum(c["cached_tokens"] for c in calls), reasoning_tokens=sum(reasoning) if reasoning else None,
            cost_usd=sum(c["cost_usd"] for c in calls), llm_sec=sum(c["latency_sec"] for c in calls),
            tool_sec=sum(x["duration_sec"] for x in tools), n_tool_calls=len(tools), tool_counts=dict(counts),
            n_tool_errors=sum(1 for x in tools if x["is_error"]), n_timeouts=sum(1 for x in tools if x["timed_out"]),
            context_trims=0, workdir="/app", isolation=isolation, files_modified=["src/main.py"],
            n_files_modified=1, error=None,
        )
        d = asdict(rec)
        self.dispatches.append(d)
        return d

    def _group(self, spec: Spawn, *, t_tool: float, step: int, call_id: str) -> tuple[dict, float]:
        """Lay out a spawn group starting at ``t_tool``; returns (SpawnRecord dict, tool end time)."""
        strategy = spec.strategy or self.strategy
        gid = self._id("g")
        g0 = t_tool + spec.start_overhead
        members: list[dict] = []
        arb_ids: list[str] = []
        if strategy == "arbiter":
            ends = []
            for i, a in enumerate(spec.arbitrations):
                arb_rec = self._arbitration(a, group_id=gid, instr_idx=i, t_start=g0)
                arb_ids.append(arb_rec["arbitration_id"])
                members += [d for d in self.dispatches if d["arbitration_id"] == arb_rec["arbitration_id"]]
                ends.append(arb_rec["t_end"])
            granted = len(spec.arbitrations)
            concurrency = spec.concurrency or sum(len(a.candidates) for a in spec.arbitrations)
            t_end_members = max(ends) if ends else g0
        else:
            granted = len(spec.members)
            if strategy == "sequential":
                concurrency = 1
            else:
                concurrency = spec.concurrency or self.max_concurrent
            free = [g0] * max(1, min(concurrency, max(granted, 1)))
            statuses = spec.statuses or ["submitted"] * granted
            for i, chain in enumerate(spec.members):
                lane = min(range(len(free)), key=lambda k: free[k])
                d = self._dispatch(chain, t_queued=g0, t_start=free[lane], group_id=gid, instr_idx=i,
                                   strategy=strategy, member_gap=spec.member_gap, status=statuses[i])
                free[lane] = d["t_end"]
                members.append(d)
            t_end_members = max([d["t_end"] for d in members], default=g0)
        t_end = t_end_members + spec.end_overhead
        requested = spec.requested if spec.requested is not None else granted
        used = sum(1 for x in self.dispatches)
        rec = SpawnRecord(
            episode_id=self.episode_id, group_id=gid, orchestrator_step=step, call_id=call_id,
            tool_call_id=f"tc-spawn-{gid}", strategy=strategy, scope=self.scope or "hard", requested=requested,
            granted=granted, cap_reason=None if requested == granted else "max_concurrent",
            dispatches_remaining_after=max(self.max_dispatches - used, 0),
            instructions=[f"instruction {i}" for i in range(requested)], dispatch_ids=[d["dispatch_id"] for d in members],
            arbitration_ids=arb_ids, concurrency=concurrency, t_start=t_tool, t_end=t_end, duration_sec=t_end - t_tool,
            sum_dispatch_sec=sum(d["duration_sec"] for d in members), files_modified=["src/main.py"], n_files_modified=1,
        )
        d = asdict(rec)
        self.spawns.append(d)
        return d, t_end

    def _arbitration(self, a: Arb, *, group_id: str, instr_idx: int, t_start: float) -> dict:
        aid = self._id("arb")
        t_start = t_start + a.isolation_sec  # candidates start after the isolation setup
        cands = [
            self._dispatch(chain, t_queued=t_start, t_start=t_start, group_id=group_id, instr_idx=instr_idx,
                           strategy="arbiter", member_gap=0.0, arbitration_id=aid, candidate_index=k,
                           isolation=a.isolation)
            for k, chain in enumerate(a.candidates)
        ]
        t_cand_end = max(d["t_end"] for d in cands)
        t = t_cand_end + a.diff_sec
        rounds = []
        judge_agent = f"arbiter-{aid}"
        for r, j in enumerate(a.rounds, 1):
            r_start = t
            debate_ids = []
            if j.debates:
                ends = []
                for k, spec in enumerate(j.debates):
                    cand = cands[k]
                    c = self._call(spec, t_start=t, role="debate", agent_id=cand["agent_id"], model=self.sub_model,
                                   step=r, dispatch_id=cand["dispatch_id"], group_id=group_id, arbitration_id=aid)
                    debate_ids.append(c["call_id"])
                    ends.append(c["t_end"])
                t = max(ends)
            if j.judge is None:  # aborted before the judge call: a fallback judgement, as the arbiter records
                rounds.append({"round": r, "judge_call_id": None, "debate_call_ids": debate_ids, "winner": 0,
                               "confidence": None, "agreement": None, "rationale": "aborted", "parse_ok": False,
                               "t_start": r_start, "t_end": t})
                continue
            jc = self._call(j.judge, t_start=t, role="arbiter", agent_id=judge_agent, model=self.arbiter_model,
                            step=r, tool_names=["submit_judgement"], group_id=group_id, arbitration_id=aid)
            t = jc["t_end"]
            rounds.append({"round": r, "judge_call_id": jc["call_id"], "debate_call_ids": debate_ids,
                           "winner": j.winner, "confidence": j.confidence, "agreement": j.agreement,
                           "rationale": "synthetic", "parse_ok": j.parse_ok, "t_start": r_start, "t_end": t})
        t_end = t + a.apply_sec
        judged = [(k, j) for k, j in enumerate(a.rounds, 1) if j.judge is not None]
        final_round, last = judged[-1] if judged else (None, Judge(C(0.0)))
        copy = a.isolation == "copy"
        rec = ArbitrationRecord(
            episode_id=self.episode_id, arbitration_id=aid, group_id=group_id, instruction_index=instr_idx,
            instruction=f"instruction {instr_idx}", n_candidates=len(cands),
            candidate_dispatch_ids=[d["dispatch_id"] for d in cands], isolation=a.isolation, rounds=rounds,
            debate_rounds_used=sum(1 for j in a.rounds if j.debates), final_winner=last.winner,
            final_confidence=last.confidence, agreement=last.agreement, applied=copy,
            applied_files=list(a.applied_files) if copy else [], apply_error=None,
            isolation_sec=a.isolation_sec, t_start=t_start, t_candidates_end=t_cand_end, t_end=t_end,
            duration_sec=t_end - t_start,
            isolation_violated=bool(a.leaked_files) if copy else None,
            leaked_files=[f"/app/{p}" for p in a.leaked_files] if copy else [],
            n_path_remaps=a.n_path_remaps if copy else 0, final_from_round=final_round,
        )
        d = asdict(rec)
        self.arbitrations.append(d)
        return d

    # -- orchestrator API ------------------------------------------------------
    def step(self, call: C, tools: list = (), finish: bool = False) -> "EpisodeBuilder":
        """One orchestrator step: the call, then each tool (T or Spawn) in order, with ``gap`` between events."""
        self.step_no += 1
        k = self.step_no
        if k > 1:
            self.t += self.gap
        t_step = self.t
        names = [("spawn_subagents" if isinstance(x, Spawn) else x.name) for x in tools] + (["finish"] if finish else [])
        c = self._call(call, t_start=self.t, role="orchestrator", agent_id="orchestrator", model=self.orch_model,
                       step=k, tool_names=names)
        self.t = c["t_end"]
        groups = []
        for x in list(tools) + ([T("finish", 0.0)] if finish else []):
            self.t += self.gap
            if isinstance(x, Spawn):
                # agent_seq / seq of the spawn tool are allocated when it starts (before its members).
                seq, aseq = self._next_seq(), self._next_agent_seq("orchestrator")
                g, t_end = self._group(x, t_tool=self.t, step=k, call_id=c["call_id"])
                rec = asdict(ToolRecord(
                    episode_id=self.episode_id, tool_event_id=self._id("tool"), seq=seq, role="orchestrator",
                    agent_id="orchestrator", dispatch_id=None, group_id=g["group_id"], arbitration_id=None, step=k,
                    agent_seq=aseq, call_id=c["call_id"], tool_call_id=g["tool_call_id"], name="spawn_subagents",
                    args={"instructions": g["instructions"]}, t_start=self.t, t_end=t_end, duration_sec=t_end - self.t,
                    exit_code=None, timed_out=False, timeout_sec=None, output_chars_raw=500, output_chars_returned=500,
                    truncated=False, is_error=False, error=None, cwd_before=None, cwd_after=None, n_exec=2,
                    exec_sec=0.0,
                ))
                self.tools.append(rec)
                groups.append(g["group_id"])
                self.t = t_end
            else:
                rec = self._tool(x, t_start=self.t, role="orchestrator", agent_id="orchestrator", step=k,
                                 call_id=c["call_id"])
                self.t = rec["t_end"]
        self.steps.append(asdict(StepRecord(
            episode_id=self.episode_id, step=k, call_id=c["call_id"], t_start=t_step, t_end=self.t,
            duration_sec=self.t - t_step, llm_sec=c["latency_sec"], tools_sec=self.t - c["t_end"],
            n_tool_calls=len(names), tool_names=names, spawned_group_ids=groups, finished=finish,
        )))
        return self

    # -- episode.json ------------------------------------------------------------
    def finish(self, *, exit_reason: str | None = "finished", status: str = "done", end_gap: float = 0.0,
               n_files_modified: int | None = 1, error: str | None = None) -> dict:
        t_end = self.t + end_gap
        base = dict(
            schema_version=SCHEMA_VERSION, status=status, episode_id=self.episode_id, experiment=self.condition,
            config=self.config_dict(), config_hash="cfg000000000", strategy=self.strategy, scope=self.scope,
            topology=self.topology, orchestrator_model=self.orch_model, subagent_model=self.sub_model,
            arbiter_model=self.arbiter_model, task_name=self.task_name, trial_name=self.trial_name,
            harbor_session_id="sess-1", harbor_trial_id="trial-uuid", logs_dir="/logs/agent",
            instruction="Fix the repository.", harness_version="0.1.0", git_commit="abc123", git_dirty=False,
            source_sha256="0" * 64,
            harbor_version="0.24.0", python_version="3.13.0", host_platform="macOS-27.0-arm64",
            host_machine="arm64", container_arch="x86_64", sandbox_kind="harbor", workdir="/app",
            has_timeout_binary=True, t_start=self.t0,
            t_first_call=self.calls[0]["t_start"] if self.calls else None,
        )
        if status != "done":  # what the recorder writes at start: no aggregates yet
            ep = EpisodeRecord(
                **base, t_end=None, wall_clock_sec=None, exit_reason=None, error=None, final_summary=None,
                orchestrator_steps=0, spawn_calls=0, dispatches=0, redispatches=0, arbitrations=0, requested_total=0,
                granted_total=0, instruction_following=None, n_calls={}, n_tool_calls={}, tokens={}, cost_usd={},
                llm_sec={}, tool_sec={}, n_retries=0, backoff_sec=0.0, n_provider_errors=0, n_command_timeouts=0,
                n_context_trims=0, cancel_kill=None,
                files_modified=None, n_files_modified=None, git_diff_stat=None,
                harness_exec_n=None, harness_exec_sec=None, harness_exec_sum_sec=None,
            )
            self.episode = asdict(ep)
            return self.episode
        by_role = {r: [c for c in self.calls if c["role"] == r] for r in ROLES}
        tools_by_role = {r: [x for x in self.tools if x["role"] == r] for r in ROLES}
        tokens = {}
        for r, cs in by_role.items():
            reasoning = [c["reasoning_tokens"] for c in cs if c["reasoning_tokens"] is not None]
            tokens[r] = {
                "prompt": sum(c["prompt_tokens"] for c in cs), "completion": sum(c["completion_tokens"] for c in cs),
                "cached": sum(c["cached_tokens"] for c in cs),
                "uncached_prompt": sum(c["uncached_prompt_tokens"] for c in cs),
                "reasoning": sum(reasoning) if reasoning else None,
            }
        cost = {r: sum(c["cost_usd"] for c in cs) for r, cs in by_role.items()}
        cost["total"] = sum(c["cost_usd"] for c in self.calls)
        first = self.spawns[0]["group_id"] if self.spawns else None
        ratios = [s["granted"] / s["requested"] for s in self.spawns if s["requested"]]
        ep = EpisodeRecord(
            **base, t_end=t_end, wall_clock_sec=t_end - self.t0, exit_reason=exit_reason,
            error=error or ("billing" if exit_reason in ("error", "billing_error") else None), final_summary="done",
            cancel_kill=None,
            orchestrator_steps=len(self.steps), spawn_calls=len(self.spawns), dispatches=len(self.dispatches),
            redispatches=sum(1 for d in self.dispatches if d["group_id"] != first),
            arbitrations=len(self.arbitrations), requested_total=sum(s["requested"] for s in self.spawns),
            granted_total=sum(s["granted"] for s in self.spawns),
            instruction_following=sum(ratios) / len(ratios) if ratios else None,
            n_calls={r: len(cs) for r, cs in by_role.items()},
            n_tool_calls={r: len(ts) for r, ts in tools_by_role.items()}, tokens=tokens, cost_usd=cost,
            llm_sec={r: sum(c["latency_sec"] for c in cs) for r, cs in by_role.items()},
            tool_sec={r: sum(x["duration_sec"] for x in ts) for r, ts in tools_by_role.items()},
            n_retries=sum(c["n_attempts"] - 1 for c in self.calls), backoff_sec=sum(c["backoff_sec"] for c in self.calls),
            n_provider_errors=sum(1 for c in self.calls if (c["error"] or "").startswith("LLMError:")),
            n_command_timeouts=sum(1 for x in self.tools if x["timed_out"]),
            n_context_trims=sum(d["context_trims"] for d in self.dispatches),
            files_modified=["src/main.py"] if n_files_modified else None, n_files_modified=n_files_modified,
            git_diff_stat=None,
            # Synthetic: the builder runs no measurement commands (the gaps model all overhead).
            harness_exec_n=0, harness_exec_sec=0.0, harness_exec_sum_sec=0.0,
        )
        self.episode = asdict(ep)
        return self.episode

    def write(self, episode_dir: Path) -> Path:
        """Write episode.json and every JSONL file (finishing with defaults if not finished yet)."""
        if self.episode is None:
            self.finish()
        d = Path(episode_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "episode.json").write_text(json.dumps(self.episode, indent=2))
        transcript = [asdict(TranscriptRecord(
            episode_id=self.episode_id, agent_id="orchestrator", role="orchestrator", dispatch_id=None, index=i,
            t=self.t0 + i, message_role=mr, content=txt, reasoning=None, tool_calls=[], tool_results=[], call_id=None,
        )) for i, (mr, txt) in enumerate([("system", "You are the orchestrator."), ("user", "Fix the repository.")])]
        for name, recs in (("calls", self.calls), ("tools", self.tools), ("steps", self.steps),
                           ("spawns", self.spawns), ("dispatches", self.dispatches),
                           ("arbitrations", self.arbitrations), ("transcript", transcript)):
            (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        return d


def config_dict(name: str, topology: str, strategy: str | None, scope: str | None, orch_model: str,
                sub_model: str | None, max_concurrent: int, n_candidates: int, steps: int,
                window: str | None = "balanced") -> dict:
    """A resolved ExperimentConfig dict shaped per docs/DESIGN.md section 2."""
    orch = {"model": orch_model, "provider": "sail", "completion_window": window, "temperature": 0.7}
    return {
        "name": name, "benchmark": "terminal_bench", "topology": topology, "subtask_type": "none",
        "orchestrator": orch,
        "subagent": {
            "model": None if sub_model in (None, orch_model) else {**orch, "model": sub_model},
            "strategy": strategy or "parallel", "scope": scope or "hard", "max_concurrent": max_concurrent,
            "max_dispatches": None, "max_steps": 8, "token_budget": None, "max_output_chars": 12000,
            "arbiter": {"n_candidates": n_candidates, "debate_rounds": 1, "confidence_threshold": 0.8,
                        "model": None, "max_copy_mb": 500},
        },
        "budget": {"orchestrator_steps": steps, "total_token_budget": 5000000, "timeout_sec": 3600,
                   "command_timeout_sec": 180, "max_command_timeout_sec": 900},
        "tools": {"max_output_chars": 12000},
        "sandbox": {"kind": "harbor", "image": None, "workdir": "auto", "network": False},
        "task_ids": [], "difficulties": [], "n_trials": 1, "seed": 1, "output_dir": "runs",
    }


# ---------------------------------------------------------------------------
# harbor trial / job dirs
# ---------------------------------------------------------------------------


def harbor_result(task_name: str, trial_name: str, trial_dir: Path, *, t_agent_start: float, t_agent_end: float,
                  reward: float | None, exception: str | None = None, exception_phase: str = "agent",
                  agent_meta: dict | None = None) -> dict:
    """A trial result.json shaped like harbor 0.24.0's (see jobs/oracle-smoke for a real one).

    ``exception_phase``: environment (setup failed; no agent ran), agent (raised by
    the agent; harbor skips the verifier unless ``reward`` is given, as after an
    AgentTimeoutError) or verifier (the verifier started, then failed: no reward)."""
    env0, env1 = t_agent_start - 8.0, t_agent_start - 1.0
    ver0, ver1 = t_agent_end + 0.4, t_agent_end + 6.0
    env_failed = exception_phase == "environment"
    ver_ran = not env_failed and (reward is not None or exception_phase == "verifier")
    exc = None
    if exception:
        occurred = env0 + 5.0 if env_failed else (ver0 + 1.0 if exception_phase == "verifier" else t_agent_end)
        exc = {"exception_type": exception, "exception_message": f"synthetic {exception}",
               "exception_traceback": "Traceback (synthetic)", "occurred_at": _iso(occurred)}
    end = (env0 + 5.5) if env_failed else ver1 + 0.5
    return {
        "id": "00000000-0000-0000-0000-000000000000", "task_name": task_name, "trial_name": trial_name,
        "trial_uri": trial_dir.resolve().as_uri(),
        "task_id": {"git_url": "https://github.com/laude-institute/terminal-bench-2.git",
                    "git_commit_id": "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c", "path": task_name},
        "source": "terminal-bench", "task_checksum": "0" * 64,
        "config": {"task": {"path": task_name}, "trial_name": trial_name, "trials_dir": str(trial_dir.parent),
                   "agent": {"name": None, "import_path": "subagents_bench.harness.harbor_agent:OrchestratorAgent"}},
        "agent_info": {"name": "subagents-bench", "version": "0.1.0", "model_info": None},
        "agent_result": None if env_failed else {
            "n_input_tokens": None, "n_cache_tokens": None, "n_output_tokens": None, "cost_usd": None,
            "model_usage": None, "rollout_details": None, "metadata": agent_meta},
        "verifier_result": {"rewards": {"reward": reward}} if reward is not None else None,
        "verifier_environment_mode": "shared",
        "exception_info": exc,
        "started_at": _iso(env0 - 0.05), "finished_at": _iso(end),
        "environment_setup": {"started_at": _iso(env0), "finished_at": None if env_failed else _iso(env1)},
        "agent_setup": None if env_failed else {"started_at": _iso(env1), "finished_at": _iso(env1 + 0.5)},
        "agent_execution": None if env_failed else {"started_at": _iso(t_agent_start), "finished_at": _iso(t_agent_end)},
        "verifier": {"started_at": _iso(ver0), "finished_at": _iso(ver1)} if ver_ran else None,
        "step_results": None,
    }


def write_trial(job_dir: Path, task_name: str, trial_name: str, builder: EpisodeBuilder | None, *,
                reward: float | None, exception: str | None = None, exception_phase: str = "agent",
                write_result: bool = True, t0: float = T0) -> Path:
    trial_dir = Path(job_dir) / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "config.json").write_text(json.dumps({"trial_name": trial_name, "task": {"path": task_name}}))
    if builder is not None:
        builder.write(trial_dir / "agent")
        t_a0 = builder.t0 - 0.2
        ep = builder.episode or {}
        t_a1 = (ep.get("t_end") or builder.t) + 0.2
    else:
        t_a0 = t_a1 = t0
    if write_result:
        res = harbor_result(task_name, trial_name, trial_dir, t_agent_start=t_a0, t_agent_end=t_a1, reward=reward,
                            exception=exception, exception_phase=exception_phase)
        (trial_dir / "result.json").write_text(json.dumps(res, indent=4))
    return trial_dir


def write_job(job_dir: Path, n_trials: int) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "config.json").write_text(json.dumps({"job_name": job_dir.name, "n_concurrent_trials": 4}))
    (job_dir / "result.json").write_text(json.dumps({"id": "job", "n_total_trials": n_trials, "stats": {}}))
    (job_dir / "job.log").write_text("synthetic job\n")


# ---------------------------------------------------------------------------
# Whole sweeps
# ---------------------------------------------------------------------------

TASKS = {
    "fix-git": {"difficulty": "easy", "category": "software-engineering", "expert_time_min": 5.0,
                "junior_time_min": 20.0},
    "crack-7z-hash": {"difficulty": "medium", "category": "security", "expert_time_min": 15.0,
                      "junior_time_min": 60.0},
}
EDGE_TASKS = {
    "overfull-hbox": {"difficulty": "easy", "category": "debugging", "expert_time_min": 10.0, "junior_time_min": 30.0},
    "cobol-modernization": {"difficulty": "easy", "category": "software-engineering", "expert_time_min": 20.0,
                            "junior_time_min": 60.0},
}


def pilot_conditions() -> dict[str, dict]:
    conds = {
        "single_5step": {"topology": "single", "steps_budget": 5},
        "single_matched": {"topology": "single", "steps_budget": 40},
    }
    for st in ("parallel", "sequential", "arbiter"):
        for sc in ("easy", "medium", "hard"):
            conds[f"gemma31b_{st}_{sc}"] = {"topology": "subagents", "strategy": st, "scope": sc, "sub_model": GEMMA}
    conds["samemodel_parallel_hard"] = {"topology": "subagents", "strategy": "parallel", "scope": "hard",
                                        "sub_model": ORCH}
    return conds


def default_outcome(condition: str, task: str, rnd: int) -> float:
    """Deterministic synthetic rewards: baselines solve less, so paired tables have discordant pairs."""
    if condition == "single_5step":
        return 1.0 if (task == "fix-git" and rnd == 1) else 0.0
    if condition == "single_matched":
        return 1.0 if task == "fix-git" else 0.0
    return 1.0 if (len(condition) + len(task) + rnd) % 3 != 0 else 0.0


def standard_episode(b: EpisodeBuilder, task_idx: int = 0, rnd: int = 1) -> EpisodeBuilder:
    """A plausible episode for the builder's condition; structure varies with task and round."""
    v = 0.1 * task_idx + 0.05 * rnd  # small variation so episodes differ
    big = C(2.0 + v, prompt=8000, completion=600, cached=2560, reasoning=300)
    if b.topology == "single":
        b.step(big, [T("bash", 1.0 + v)])
        b.step(C(1.5, prompt=9000, completion=400, cached=7936, reasoning=200), [T("edit_file", 0.2)])
        b.step(C(1.0, prompt=9500, completion=200, cached=8960, reasoning=100), finish=True)
        return b
    tool = SCOPE_TOOL[b.scope or "hard"]
    sub = lambda lat: [C(lat, prompt=3000, completion=300, cached=0), T(tool, 0.4), C(0.8, prompt=3500, completion=150)]
    if b.strategy == "arbiter":
        rounds = [Judge(C(0.9, prompt=4000, completion=200, reasoning=100), confidence=0.6, agreement=False),
                  Judge(C(0.7, prompt=5000, completion=150, reasoning=80),
                        debates=[C(0.5, prompt=2000, completion=200)] * 3, confidence=0.9, agreement=True)]
        if rnd % 2 == 0:
            rounds = rounds[:1]
            rounds[0] = Judge(rounds[0].judge, confidence=0.95, agreement=True)
        # Copy isolation; on the second task a candidate also wrote the real workdir (a restored leak).
        arb = Arb(candidates=[sub(1.0 + v), sub(1.4), sub(0.9)], rounds=rounds,
                  apply_sec=0.3 if b.scope == "hard" else 0.0, isolation="copy" if b.scope == "hard" else "shared",
                  leaked_files=["notes.txt", "src/main.py"] if task_idx == 1 else [], n_path_remaps=2)
        b.step(big, [Spawn(arbitrations=[arb], requested=2, start_overhead=0.05, end_overhead=0.1)])
    else:
        b.step(big, [Spawn(members=[sub(1.0 + v), sub(1.6), sub(1.2)], start_overhead=0.05, end_overhead=0.1)])
    b.step(C(1.2, prompt=9000, completion=300, cached=7936, reasoning=150), [T("bash", 0.5)])
    b.step(C(0.8, prompt=9500, completion=150, cached=8960, reasoning=60), finish=True)
    return b


@dataclass
class SynthTrial:
    job_name: str
    condition: str
    round: int
    task_name: str
    trial_name: str
    reward: float | None
    builder: EpisodeBuilder | None
    edge: str | None = None
    expected_valid: bool = True
    expected_resolved: int | None = 0


@dataclass
class SynthSweep:
    root: Path
    sweep: str
    trials: list[SynthTrial]
    conditions: dict[str, dict]

    def get(self, condition: str, task: str, rnd: int) -> SynthTrial:
        for t in self.trials:
            if (t.condition, t.task_name, t.round) == (condition, task, rnd):
                return t
        raise KeyError((condition, task, rnd))


def make_sweep(
    root: Path,
    *,
    sweep: str = "synth",
    tasks: dict[str, dict] | None = None,
    rounds: tuple[int, ...] = (1, 2),
    conditions: dict[str, dict] | None = None,
    outcome: Callable[[str, str, int], float] = default_outcome,
    episode: Callable[[EpisodeBuilder, int, int], EpisodeBuilder] = standard_episode,
    edge_cases: bool = True,
    with_manifest: bool = True,
    with_tasks_meta: bool = True,
    gap: float = 0.01,
) -> SynthSweep:
    """Write a synthetic sweep under ``root`` (= the jobs dir) and return what was written."""
    root = Path(root)
    tasks = dict(TASKS if tasks is None else tasks)
    conditions = pilot_conditions() if conditions is None else conditions
    trials: list[SynthTrial] = []
    manifest = []
    (root / "configs").mkdir(parents=True, exist_ok=True)
    n_ep = 0
    t_clock = T0
    for rnd in rounds:
        for cond, spec in conditions.items():
            job = f"{sweep}__r{rnd}__{cond}"
            job_dir = root / job
            cfg_path = root / "configs" / f"{cond}.yaml"
            kw = {k: v for k, v in spec.items() if k != "steps_budget"}
            cfg_path.write_text(yaml.safe_dump(config_dict(
                cond, spec["topology"], spec.get("strategy"), spec.get("scope"), ORCH, spec.get("sub_model"), 3, 3,
                spec.get("steps_budget", 5))))
            job_trials: list[SynthTrial] = []
            for ti, task in enumerate(tasks):
                n_ep += 1
                t_clock += 100.0
                b = EpisodeBuilder(episode_id=f"ep{n_ep:04d}", condition=cond, task_name=task,
                                   trial_name=f"{task}__s{n_ep:04d}", t0=t_clock, gap=gap,
                                   steps_budget=spec.get("steps_budget", 5), **kw)
                episode(b, ti, rnd)
                b.finish(end_gap=0.02)
                reward = outcome(cond, task, rnd)
                write_trial(job_dir, task, b.trial_name, b, reward=reward)
                job_trials.append(SynthTrial(job, cond, rnd, task, b.trial_name, reward, b,
                                             expected_resolved=int(reward >= 1.0)))
            if edge_cases and rnd == rounds[0]:
                job_trials += _edge_trials(job_dir, job, cond, rnd, spec, t_clock, gap)
            write_job(job_dir, len(job_trials))
            trials += job_trials
            # The sweep launcher's manifest line shape (subagents_bench.sweep._manifest_record).
            manifest.append({"sweep": sweep, "job": job, "round": rnd, "condition": cond, "event": "run",
                             "config_path": str(cfg_path), "cmd": ["harbor", "run", "--job-name", job],
                             "t_start": t_clock - 50, "t_end": t_clock, "duration_sec": 50.0, "return_code": 0,
                             "n_trials": len(job_trials), "n_completed": len(job_trials),
                             "n_errors": sum(1 for t in job_trials if t.edge), "complete": True})
    if with_manifest:
        (root / "sweep_manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    if with_tasks_meta:
        meta = {k: {**v, "agent_timeout_sec": 900.0, "verifier_timeout_sec": 600.0,
                    "docker_image": f"tb2/{k}:latest", "tags": ["synthetic"]}
                for k, v in {**tasks, **(EDGE_TASKS if edge_cases else {})}.items()}
        (root / "tasks_meta.json").write_text(json.dumps({"dataset": "terminal-bench@2.0", "tasks": meta}, indent=2))
    return SynthSweep(root=root, sweep=sweep, trials=trials, conditions=conditions)


def _edge_trials(job_dir: Path, job: str, cond: str, rnd: int, spec: dict, t_clock: float, gap: float) -> list[SynthTrial]:
    kw = {k: v for k, v in spec.items() if k != "steps_budget"}
    out = []

    def builder(task: str, suffix: str) -> EpisodeBuilder:
        return EpisodeBuilder(episode_id=f"edge-{cond}-{suffix}", condition=cond, task_name=task,
                              trial_name=f"{task}__edge{suffix}", t0=t_clock + 50.0, gap=gap, **kw)

    if cond == "gemma31b_parallel_easy":  # environment never came up: no agent dir, no reward
        name = "overfull-hbox__edgeenv"
        write_trial(job_dir, "overfull-hbox", name, None, reward=None, exception="EnvironmentStartTimeoutError",
                    exception_phase="environment", t0=t_clock + 50.0)
        out.append(SynthTrial(job, cond, rnd, "overfull-hbox", name, None, None, edge="env_failure",
                              expected_valid=False, expected_resolved=None))
    if cond == "gemma31b_sequential_medium":  # the agent raised mid-episode: status stays running, no verifier
        b = builder("cobol-modernization", "crash")
        b.step(C(2.0, prompt=8000, completion=500, cached=2560, reasoning=200),
               [Spawn(members=[[C(1.0), T("bash", 0.3)], [C(1.5)]])])
        b.finish(status="running")
        write_trial(job_dir, "cobol-modernization", b.trial_name, b, reward=None, exception="RuntimeError")
        out.append(SynthTrial(job, cond, rnd, "cobol-modernization", b.trial_name, None, b, edge="running",
                              expected_valid=False, expected_resolved=None))
    if cond == "gemma31b_arbiter_hard":  # harbor never wrote result.json
        b = builder("overfull-hbox", "nores")
        standard_episode(b)
        b.finish()
        write_trial(job_dir, "overfull-hbox", b.trial_name, b, reward=None, write_result=False)
        out.append(SynthTrial(job, cond, rnd, "overfull-hbox", b.trial_name, None, b, edge="no_result",
                              expected_valid=False, expected_resolved=None))
    if cond == "gemma31b_parallel_hard":  # subagent condition, orchestrator never spawned
        b = builder("cobol-modernization", "nodisp")
        b.step(C(2.0, prompt=8000, completion=500, cached=2560, reasoning=200), [T("bash", 1.0)])
        b.step(C(1.0, prompt=9000, completion=200, cached=7936, reasoning=50), finish=True)
        b.finish()
        write_trial(job_dir, "cobol-modernization", b.trial_name, b, reward=1.0)
        out.append(SynthTrial(job, cond, rnd, "cobol-modernization", b.trial_name, 1.0, b, edge="zero_dispatch",
                              expected_valid=True, expected_resolved=1))
    if cond == "single_matched":  # provider refused for billing
        b = builder("cobol-modernization", "bill")
        b.step(C(1.0, prompt=8000, completion=300, cached=0, reasoning=100), [T("bash", 0.5)])
        b.step(C(0.3, error="HTTP 402: insufficient balance"))
        b.finish(exit_reason="billing_error")
        write_trial(job_dir, "cobol-modernization", b.trial_name, b, reward=None, exception="BillingError")
        out.append(SynthTrial(job, cond, rnd, "cobol-modernization", b.trial_name, None, b, edge="billing",
                              expected_valid=False, expected_resolved=None))
    return out
