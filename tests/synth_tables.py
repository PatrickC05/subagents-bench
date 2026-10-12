"""Synthetic analysis directory for the plot tests.

Writes the CSV tables of ``docs/DESIGN.md`` section 6 (``episodes.csv``,
``calls.csv``, ``tools.csv``, ``dispatches.csv``, ``spawns.csv``,
``arbitrations.csv``, ``steps.csv``, ``conditions.csv``,
``conditions_by_difficulty.csv``, ``paired.csv``, ``calibration.csv``) plus
``summary.json``, as the aggregator would.

The data come from a small event simulator: every episode lays out its
orchestrator steps, LLM calls (with retry/TTFT/decode instants), tools, spawn
groups, subagent dispatches and arbitrations on one timeline, and the
per-episode aggregates (tokens, costs, critical path, simulated wall-clock)
are computed from those events. So the tables are internally consistent and
the timeline figures have real structure to draw. The default conditions
include partial treatment uptake (one grid condition never delegates), a
manager control (restricted orchestrator tools) and one re-run trial flagged
``duplicate``, so the figures' handling of each is exercised.

EVERY NUMBER HERE IS SYNTHETIC. Prices, parameter counts, speeds and outcomes
are placeholders chosen to look plausible; they are not measurements and must
never be quoted as results.

Usage from tests::

    from synth_tables import write_synthetic_analysis
    write_synthetic_analysis(tmp_path / "analysis")

or from a shell (to eyeball the figures)::

    python tests/synth_tables.py /tmp/synth_analysis
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ORCH = "deepseek-ai/DeepSeek-V4.1-Flash"
GEMMA = "nvidia/Gemma-4-31B-IT-NVFP4"

STRATEGIES = ["parallel", "sequential", "arbiter"]
SCOPES = ["easy", "medium", "hard"]
SCHEMES = ["sail_balanced", "sail_asap", "prime", "reference"]
PROFILES = ["measured", "fitted", "roofline_h100"]
ROLES = {"orchestrator": "orch", "subagent": "sub", "arbiter": "arb", "debate": "deb"}
SCOPE_TOOLS = {
    "easy": ["read_file", "list_dir", "search"],
    "medium": ["read_file", "list_dir", "search", "bash"],
    "hard": ["read_file", "list_dir", "search", "bash", "edit_file"],
}

# Synthetic serving behaviour: base TTFT (s), extra TTFT per 1k prompt tokens,
# decode tokens/s. Placeholders, not measurements.
SPEED = {
    ORCH: {"ttft": 0.55, "ttft_per_1k": 0.020, "tok_s": 150.0},
    GEMMA: {"ttft": 0.90, "ttft_per_1k": 0.035, "tok_s": 70.0},
}
# Synthetic active parameter counts (billions) for the FLOPs columns. Placeholders.
ACTIVE_PARAMS_B = {ORCH: 13.0, GEMMA: 31.0}
# USD per 1M tokens (in, cached-in or None -> in, out); None = model unpriced
# in that scheme, which makes the cost NaN (never 0). Synthetic placeholders.
PRICES: dict[str, dict[str, tuple[float, float | None, float] | None]] = {
    "sail_balanced": {ORCH: (0.07, 0.005, 0.22), GEMMA: (0.05, None, 0.10)},
    "sail_asap": {ORCH: (0.14, 0.010, 0.44), GEMMA: (0.10, None, 0.20)},
    "prime": {ORCH: None, GEMMA: None},
    "reference": {ORCH: (0.09, 0.020, 0.30), GEMMA: None},
}

TASKS = [
    {"name": "fix-git", "difficulty": "easy", "category": "software-engineering", "expert": 5.0, "junior": 20.0, "p": 0.75},
    {"name": "crack-7z-hash", "difficulty": "medium", "category": "security", "expert": 15.0, "junior": 60.0, "p": 0.45},
    {"name": "raman-fitting", "difficulty": "medium", "category": "scientific-computing", "expert": 30.0, "junior": 120.0, "p": 0.25},
]
ROUNDS = [1, 2, 3]
SWEEP_T0 = 1791720000.0  # 2026-10-11T12:00Z


@dataclass
class Condition:
    name: str
    topology: str
    strategy: str | None
    scope: str | None
    sub_model: str | None
    steps: int
    p_mult: float  # multiplier on the task's base solve probability
    uptake: float = 1.0  # probability that the orchestrator spawns subagents at all in an episode
    orchestrator_tools: str = "all"  # tools.orchestrator_tools label; restricted = manager control


MANAGER_TOOLS = "read_file+list_dir+search"


def default_conditions() -> list[Condition]:
    conds = [
        Condition("single_5step", "single", None, None, None, 5, 0.75),
        Condition("single_matched", "single", None, None, None, 40, 1.05),
    ]
    mult = {"parallel": 1.0, "sequential": 0.95, "arbiter": 1.1}
    smult = {"easy": 0.85, "medium": 1.0, "hard": 1.15}
    for st in STRATEGIES:
        for sc in SCOPES:
            # Partial treatment uptake; sequential/easy never delegates (an untreated condition).
            uptake = 0.0 if (st, sc) == ("sequential", "easy") else 0.8
            conds.append(Condition(f"gemma31b_{st}_{sc}", "subagents", st, sc, GEMMA, 5, mult[st] * smult[sc], uptake))
    conds.append(Condition("samemodel_parallel_hard", "subagents", "parallel", "hard", ORCH, 5, 1.25))
    conds.append(Condition("manager_arbiter_hard", "subagents", "arbiter", "hard", GEMMA, 5, 1.1,
                           orchestrator_tools=MANAGER_TOOLS))
    return conds


# ---------------------------------------------------------------------------
# event simulator
# ---------------------------------------------------------------------------


@dataclass
class _Chain:
    """Critical-path bookkeeping for one sequential agent chain."""

    llm: float = 0.0
    tool: float = 0.0

    @property
    def dur(self) -> float:
        return self.llm + self.tool


@dataclass
class _Episode:
    rng: np.random.Generator
    cond: Condition
    task: dict[str, Any]
    round: int
    episode_id: str
    t0: float
    drift: float  # multiplicative TTFT drift (provider load over the sweep)
    calls: list[dict] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    spawns: list[dict] = field(default_factory=list)
    dispatches: list[dict] = field(default_factory=list)
    arbitrations: list[dict] = field(default_factory=list)
    seq: int = 0
    agent_seq: dict[str, int] = field(default_factory=dict)
    n_sub: int = 0
    n_ids: int = 0
    dispatch_budget: int = 15
    cp_sub_llm: float = 0.0
    cp_arb_llm: float = 0.0
    cp_group_tool: float = 0.0
    cp_in_groups: float = 0.0

    def _id(self, prefix: str) -> str:
        self.n_ids += 1
        return f"{prefix}{self.n_ids:04d}"

    def _next_seq(self, agent_id: str) -> tuple[int, int]:
        s = self.seq
        self.seq += 1
        a = self.agent_seq.get(agent_id, 0)
        self.agent_seq[agent_id] = a + 1
        return s, a

    def llm_call(
        self,
        t: float,
        *,
        role: str,
        agent_id: str,
        model: str,
        step: int,
        prompt: int,
        completion: int,
        cached: int = 0,
        dispatch_id: str | None = None,
        group_id: str | None = None,
        arbitration_id: str | None = None,
        n_tool_calls: int = 1,
    ) -> dict:
        rng = self.rng
        sp = SPEED[model]
        seq, aseq = self._next_seq(agent_id)
        n_attempts, backoff = 1, 0.0
        t_req = t + 0.003
        if rng.random() < 0.04:  # one retried 429
            first = float(rng.uniform(0.2, 1.5))
            backoff = float(rng.uniform(2.0, 4.0))
            n_attempts = 2
            t_req = t + first + backoff
        ttft = (sp["ttft"] + sp["ttft_per_1k"] * prompt / 1000.0) * self.drift * float(rng.lognormal(0, 0.25))
        decode = completion / (sp["tok_s"] * float(rng.lognormal(0, 0.12)))
        t_first = t_req + ttft
        t_last = t_first + decode
        t_end = t_last + 0.004
        reasoning = int(round(completion * float(rng.uniform(0.4, 0.75)))) if model == ORCH else None
        price = PRICES["sail_balanced"][model]
        p_in, p_cache, p_out = price  # type: ignore[misc]
        uncached = prompt - cached
        cost = (uncached * p_in + cached * (p_cache if p_cache is not None else p_in) + completion * p_out) / 1e6
        rec = {
            "call_id": self._id("c"),
            "seq": seq,
            "role": role,
            "agent_id": agent_id,
            "dispatch_id": dispatch_id,
            "group_id": group_id,
            "arbitration_id": arbitration_id,
            "step": step,
            "agent_seq": aseq,
            "model": model,
            "provider": "sail",
            "base_url": "https://api.sailresearch.com/v1",
            "completion_window": "balanced",
            "max_tokens": 8192,
            "temperature": 0.7,
            "n_messages": 2 + 2 * step,
            "n_tools": 6,
            "prompt_chars": prompt * 4,
            "system_hash": "a1b2c3d4e5f6",
            "tools_hash": "0f9e8d7c6b5a",
            "reasoning_passback": True,
            "context_trimmed_chars": 0,
            "t_start": t,
            "t_end": t_end,
            "latency_sec": t_end - t,
            "t_request_sent": t_req,
            "t_headers": t_req + 0.9 * ttft,
            "t_first_chunk": t_first,
            "t_first_reasoning": t_first if reasoning else None,
            "t_first_content": t_first + (0.6 * decode if reasoning else 0.0),
            "t_first_tool_call": t_first + 0.8 * decode if n_tool_calls else None,
            "t_last_chunk": t_last,
            "ttft_sec": ttft,
            "decode_sec": decode,
            "n_chunks": int(rng.integers(5, 9)),
            "chunk_offsets_n": int(rng.integers(5, 9)),
            "chunk_gap_ms_max": float(rng.uniform(200, 2000)),
            "streamed": True,
            "n_attempts": n_attempts,
            "backoff_sec": backoff,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cached_tokens": cached,
            "uncached_prompt_tokens": uncached,
            "reasoning_tokens": reasoning,
            "usage_source": "provider",
            "finish_reason": "tool_calls" if n_tool_calls else "stop",
            "n_tool_calls": n_tool_calls,
            "content_chars": int(completion * 1.5),
            "reasoning_chars": int((reasoning or 0) * 4),
            "response_id": None,
            "cost_usd": cost,
            "price_in": p_in,
            "price_out": p_out,
            "price_cache_read": p_cache,
            "provider_reported_cost": None,
            "error": None,
        }
        self.calls.append(rec)
        return rec

    def tool(
        self,
        t: float,
        *,
        role: str,
        agent_id: str,
        name: str,
        dur: float,
        step: int,
        call_id: str,
        dispatch_id: str | None = None,
        group_id: str | None = None,
        arbitration_id: str | None = None,
        timed_out: bool = False,
    ) -> dict:
        seq, aseq = self._next_seq(agent_id)
        shell = name in ("bash", "search", "list_dir", "read_file", "edit_file")
        raw = int(self.rng.integers(50, 30000))
        rec = {
            "tool_event_id": self._id("t"),
            "seq": seq,
            "role": role,
            "agent_id": agent_id,
            "dispatch_id": dispatch_id,
            "group_id": group_id,
            "arbitration_id": arbitration_id,
            "step": step,
            "agent_seq": aseq,
            "call_id": call_id,
            "tool_call_id": f"tc_{self.n_ids}",
            "name": name,
            "t_start": t,
            "t_end": t + dur,
            "duration_sec": dur,
            "exit_code": (124 if timed_out else 0) if shell else None,
            "timed_out": timed_out,
            "timeout_sec": 180 if name == "bash" else None,
            "output_chars_raw": raw,
            "output_chars_returned": min(raw, 12000),
            "truncated": raw > 12000,
            "is_error": timed_out,
            "error": None,
            "cwd_before": "/app" if name == "bash" else None,
            "cwd_after": "/app" if name == "bash" else None,
            "n_exec": 2 if shell else 0,
            "exec_sec": dur * 0.9 if shell else 0.0,
        }
        self.tools.append(rec)
        return rec

    def _tool_duration(self, name: str) -> tuple[float, bool]:
        rng = self.rng
        if name == "bash":
            if rng.random() < 0.03:
                return 180.0 + float(rng.uniform(0.5, 2.0)), True
            return float(rng.lognormal(math.log(2.5), 0.9)), False
        if name == "edit_file":
            return float(rng.uniform(0.2, 0.6)), False
        if name in ("submit", "finish"):
            return 0.001, False
        return float(rng.uniform(0.15, 0.9)), False

    # -- subagents -------------------------------------------------------------

    def run_dispatch(
        self,
        t_queued: float,
        t_start: float,
        *,
        group_id: str,
        step: int,
        instruction_index: int,
        arbitration_id: str | None = None,
        candidate_index: int | None = None,
    ) -> tuple[dict, _Chain]:
        rng, cond = self.rng, self.cond
        model = cond.sub_model or ORCH
        self.n_sub += 1
        agent_id = f"sub-{self.n_sub}"
        dispatch_id = self._id("d")
        max_steps = 8
        steps_used = int(rng.integers(2, max_steps + 1))
        status = "submitted" if (steps_used < max_steps or rng.random() < 0.5) else "max_steps"
        if rng.random() < 0.04:
            status = "error"
        chain = _Chain()
        t = t_start
        prompt = 1800
        counts: dict[str, int] = {}
        tok = {"p": 0, "c": 0, "cached": 0}
        n_err = n_to = 0
        for k in range(1, steps_used + 1):
            completion = int(rng.integers(150, 900))
            cached = (int(prompt * 0.85) // 256) * 256 if (k > 1 and model == ORCH) else 0
            c = self.llm_call(
                t, role="subagent", agent_id=agent_id, model=model, step=k, prompt=prompt, completion=completion,
                cached=cached, dispatch_id=dispatch_id, group_id=group_id, arbitration_id=arbitration_id,
            )
            chain.llm += c["latency_sec"]
            tok["p"] += prompt
            tok["c"] += completion
            tok["cached"] += cached
            t = c["t_end"] + float(rng.uniform(0.005, 0.03))
            last = k == steps_used and status == "submitted"
            name = "submit" if last else str(rng.choice(SCOPE_TOOLS[cond.scope or "hard"]))
            dur, timed_out = self._tool_duration(name)
            tr = self.tool(
                t, role="subagent", agent_id=agent_id, name=name, dur=dur, step=k, call_id=c["call_id"],
                dispatch_id=dispatch_id, group_id=group_id, arbitration_id=arbitration_id, timed_out=timed_out,
            )
            chain.tool += dur
            if name != "submit":
                counts[name] = counts.get(name, 0) + 1
                n_err += int(tr["is_error"])
                n_to += int(timed_out)
            t = tr["t_end"] + float(rng.uniform(0.005, 0.03))
            prompt += int(rng.integers(400, 2500))
        sub_calls = [c for c in self.calls if c["dispatch_id"] == dispatch_id]
        rec = {
            "dispatch_id": dispatch_id,
            "agent_id": agent_id,
            "group_id": group_id,
            "arbitration_id": arbitration_id,
            "candidate_index": candidate_index,
            "instruction_index": instruction_index,
            "orchestrator_step": step,
            "strategy": cond.strategy,
            "scope": cond.scope,
            "model": model,
            "instruction": f"synthetic instruction {instruction_index}",
            "status": status,
            "steps_used": steps_used,
            "max_steps": max_steps,
            "t_queued": t_queued,
            "t_start": t_start,
            "t_end": t,
            "duration_sec": t - t_start,
            "queue_sec": t_start - t_queued,
            "n_llm_calls": len(sub_calls),
            "prompt_tokens": tok["p"],
            "completion_tokens": tok["c"],
            "cached_tokens": tok["cached"],
            "reasoning_tokens": (sum(c["reasoning_tokens"] for c in sub_calls) if model == ORCH else None),
            "cost_usd": sum(c["cost_usd"] for c in sub_calls),
            "llm_sec": chain.llm,
            "tool_sec": chain.tool,
            "n_tool_calls": sum(counts.values()),
            "n_tool_errors": n_err,
            "n_timeouts": n_to,
            "context_trims": 0,
            "workdir": "/app",
            "isolation": "copy" if (arbitration_id and cond.scope == "hard") else "shared",
            "n_files_modified": int(rng.integers(0, 4)) if cond.scope == "hard" else 0,
            "error": "synthetic failure" if status == "error" else None,
        }
        self.dispatches.append(rec)
        return rec, chain

    def run_arbitration(self, t: float, *, group_id: str, step: int, instruction_index: int) -> float:
        rng, cond = self.rng, self.cond
        arb_id = self._id("a")
        cands = [
            self.run_dispatch(t, t + 0.01 * k, group_id=group_id, step=step, instruction_index=instruction_index,
                              arbitration_id=arb_id, candidate_index=k)
            for k in range(3)
        ]
        t_cand_end = max(d["t_end"] for d, _ in cands)
        cp = max(ch.dur for _, ch in cands)
        best = max(cands, key=lambda dc: dc[1].dur)[1]
        self.cp_sub_llm += best.llm
        self.cp_group_tool += best.tool
        judge_agent = f"arbiter-{arb_id}"
        rounds = []
        t_cur = t_cand_end + 0.02
        debate_used = 0
        for rnd in (1, 2):
            j = self.llm_call(
                t_cur, role="arbiter", agent_id=judge_agent, model=ORCH, step=rnd, prompt=int(rng.integers(5000, 12000)),
                completion=int(rng.integers(200, 700)), group_id=group_id, arbitration_id=arb_id,
            )
            self.cp_arb_llm += j["latency_sec"]
            cp += j["latency_sec"]
            conf = float(np.clip(rng.beta(5, 2), 0, 1))
            agree = bool(rng.random() < 0.55)
            winner = int(rng.integers(0, 3))
            t_cur = j["t_end"] + 0.01
            debate_ids: list[str] = []
            if rnd == 1 and not (agree and conf >= 0.8):
                debate_used = 1
                ends, durs = [], []
                for d, _ in cands:
                    dc = self.llm_call(
                        t_cur, role="debate", agent_id=d["agent_id"], model=d["model"], step=1,
                        prompt=int(rng.integers(3000, 8000)), completion=int(rng.integers(200, 600)),
                        dispatch_id=d["dispatch_id"], group_id=group_id, arbitration_id=arb_id, n_tool_calls=0,
                    )
                    debate_ids.append(dc["call_id"])
                    ends.append(dc["t_end"])
                    durs.append(dc["latency_sec"])
                self.cp_arb_llm += max(durs)
                cp += max(durs)
                t_cur = max(ends) + 0.01
            rounds.append({"winner": winner, "confidence": conf, "agreement": agree, "judge": j["call_id"], "debate": debate_ids})
            if not debate_ids:
                break
        apply_sec = float(rng.uniform(0.4, 2.0)) if cond.scope == "hard" else 0.0
        t_end = t_cur + apply_sec
        self.cp_group_tool += apply_sec
        cp += apply_sec
        final = rounds[-1]
        self.arbitrations.append({
            "arbitration_id": arb_id,
            "group_id": group_id,
            "instruction_index": instruction_index,
            "instruction": f"synthetic instruction {instruction_index}",
            "n_candidates": 3,
            "isolation": "copy" if cond.scope == "hard" else "shared",
            "n_rounds": len(rounds),
            "debate_rounds_used": debate_used,
            "final_winner": final["winner"],
            "final_confidence": final["confidence"],
            "agreement": final["agreement"],
            "first_confidence": rounds[0]["confidence"],
            "applied": cond.scope == "hard",
            "n_applied_files": int(rng.integers(0, 4)) if cond.scope == "hard" else 0,
            "apply_error": None,
            "t_start": t,
            "t_candidates_end": t_cand_end,
            "t_end": t_end,
            "duration_sec": t_end - t,
        })
        self._last_arb_cp = cp
        return t_end

    def spawn(self, t: float, *, step: int, call_id: str) -> tuple[dict, float]:
        rng, cond = self.rng, self.cond
        group_id = self._id("g")
        requested = int(rng.integers(1, 6))
        if cond.strategy == "arbiter":
            granted = min(requested, 1, self.dispatch_budget // 3)
            used = granted * 3
        else:
            granted = min(requested, 3, self.dispatch_budget)
            used = granted
        cap_reason = None
        if granted < requested:
            cap_reason = "max_dispatches" if (self.dispatch_budget < 3 and cond.strategy != "arbiter") or \
                (cond.strategy == "arbiter" and self.dispatch_budget // 3 < 1) else "max_concurrent"
        self.dispatch_budget -= used
        gt0 = t + 0.01
        group_cp = 0.0
        t_end = gt0
        n_before = len(self.dispatches)
        if cond.strategy == "parallel":
            chains = []
            for i in range(granted):
                d, ch = self.run_dispatch(gt0, gt0 + 0.01 * i, group_id=group_id, step=step, instruction_index=i)
                chains.append(ch)
                t_end = max(t_end, d["t_end"])
            if chains:
                best = max(chains, key=lambda c: c.dur)
                group_cp = best.dur
                self.cp_sub_llm += best.llm
                self.cp_group_tool += best.tool
        elif cond.strategy == "sequential":
            cur = gt0
            for i in range(granted):
                d, ch = self.run_dispatch(gt0, cur, group_id=group_id, step=step, instruction_index=i)
                group_cp += ch.dur
                self.cp_sub_llm += ch.llm
                self.cp_group_tool += ch.tool
                cur = d["t_end"] + 0.02
            t_end = cur
        else:
            arb_cps = []
            for i in range(granted):
                t_end = max(t_end, self.run_arbitration(gt0, group_id=group_id, step=step, instruction_index=i))
                arb_cps.append(self._last_arb_cp)
            group_cp = max(arb_cps) if arb_cps else 0.0
        t_end += 0.05
        members = self.dispatches[n_before:]
        self.cp_in_groups += group_cp
        rec = {
            "group_id": group_id,
            "orchestrator_step": step,
            "call_id": call_id,
            "tool_call_id": f"tc_{group_id}",
            "strategy": cond.strategy,
            "scope": cond.scope,
            "requested": requested,
            "granted": granted,
            "cap_reason": cap_reason,
            "dispatches_remaining_after": self.dispatch_budget,
            "n_dispatches": len(members),
            "n_arbitrations": granted if cond.strategy == "arbiter" else 0,
            "concurrency": 3,
            "t_start": gt0,
            "t_end": t_end,
            "duration_sec": t_end - gt0,
            "sum_dispatch_sec": sum(d["duration_sec"] for d in members),
            "n_files_modified": int(rng.integers(0, 5)) if cond.scope == "hard" else 0,
        }
        self.spawns.append(rec)
        return rec, group_cp

    # -- orchestrator ------------------------------------------------------------

    def run(self) -> dict:
        rng, cond = self.rng, self.cond
        cp = _Chain()
        t = self.t0 + 0.2
        prompt = 2600
        n_steps = cond.steps if cond.steps <= 5 else int(rng.integers(12, 30))
        finish_at = n_steps if rng.random() < 0.5 else max(2, n_steps - 1)
        spawn_steps: set[int] = set()
        if cond.topology == "subagents" and rng.random() < cond.uptake:
            spawn_steps = {1} | {s for s in (2, 3) if rng.random() < 0.6}
        own_tools = ["read_file", "search", "list_dir"] if cond.orchestrator_tools != "all" else \
            ["bash", "read_file", "bash", "search"]
        prev_prompt = 0
        exit_reason = "max_steps"
        for step in range(1, finish_at + 1):
            completion = int(rng.integers(300, 1600))
            cached = (int(prev_prompt * 0.9) // 256) * 256
            c = self.llm_call(t, role="orchestrator", agent_id="orchestrator", model=ORCH, step=step, prompt=prompt,
                              completion=completion, cached=cached)
            cp.llm += c["latency_sec"]
            step_t0 = t
            t = c["t_end"] + float(rng.uniform(0.01, 0.05))
            names: list[str] = []
            groups: list[str] = []
            if step == finish_at and finish_at < n_steps:
                names = ["finish"]
                exit_reason = "finished"
            elif step in spawn_steps and self.dispatch_budget > 0:
                names = ["spawn_subagents"]
            else:
                names = [str(rng.choice(own_tools)) for _ in range(int(rng.integers(1, 3)))]
            tools_t0 = t
            for name in names:
                if name == "spawn_subagents":
                    sp, group_cp = self.spawn(t, step=step, call_id=c["call_id"])
                    groups.append(sp["group_id"])
                    tr = self.tool(t, role="orchestrator", agent_id="orchestrator", name=name,
                                   dur=sp["t_end"] + 0.01 - t, step=step, call_id=c["call_id"], group_id=sp["group_id"])
                else:
                    dur, timed_out = self._tool_duration(name)
                    tr = self.tool(t, role="orchestrator", agent_id="orchestrator", name=name, dur=dur, step=step,
                                   call_id=c["call_id"], timed_out=timed_out)
                    cp.tool += dur
                t = tr["t_end"] + float(rng.uniform(0.01, 0.06))
            self.steps.append({
                "step": step,
                "call_id": c["call_id"],
                "t_start": step_t0,
                "t_end": t,
                "duration_sec": t - step_t0,
                "llm_sec": c["latency_sec"],
                "tools_sec": t - tools_t0,
                "n_tool_calls": len(names),
                "finished": names == ["finish"],
                "n_spawned_groups": len(groups),
            })
            prev_prompt = prompt
            prompt += int(rng.integers(1500, 6000))
        t_end = t + 0.3
        cp_total = cp.dur + self.cp_in_groups
        return {"t_end": t_end, "exit_reason": exit_reason, "cp_orch_llm": cp.llm, "cp_orch_tool": cp.tool,
                "cp_total": cp_total}


# ---------------------------------------------------------------------------
# aggregation (mirrors DESIGN section 6)
# ---------------------------------------------------------------------------


def _cost(model: str | None, scheme: str, prompt: float, cached: float, completion: float) -> float:
    if model is None:
        return float("nan")
    price = PRICES[scheme].get(model)
    if price is None:
        return float("nan")
    p_in, p_cache, p_out = price
    return ((prompt - cached) * p_in + cached * (p_cache if p_cache is not None else p_in) + completion * p_out) / 1e6


def _episode_row(ep: _Episode, out: dict, resolved: bool) -> dict:
    cond, rng = ep.cond, ep.rng
    calls = pd.DataFrame(ep.calls)
    tools = pd.DataFrame(ep.tools)
    row: dict[str, Any] = {}
    total_tokens = 0
    cost_role: dict[str, dict[str, float]] = {s: {} for s in SCHEMES}
    for role, r in ROLES.items():
        sub = calls[calls["role"] == role] if len(calls) else calls
        n = len(sub)
        p = int(sub["prompt_tokens"].sum()) if n else 0
        c = int(sub["completion_tokens"].sum()) if n else 0
        ca = int(sub["cached_tokens"].sum()) if n else 0
        reas = sub["reasoning_tokens"].dropna() if n else pd.Series(dtype=float)
        row[f"{r}_calls"] = n
        row[f"{r}_prompt_tokens"] = p
        row[f"{r}_cached_tokens"] = ca
        row[f"{r}_uncached_prompt_tokens"] = p - ca
        row[f"{r}_completion_tokens"] = c
        row[f"{r}_reasoning_tokens"] = int(reas.sum()) if len(reas) else (0 if n == 0 else float("nan"))
        total_tokens += p + c
        model = {"orchestrator": ORCH, "arbiter": ORCH}.get(role, cond.sub_model or ORCH)
        for s in SCHEMES:
            cost_role[s][r] = 0.0 if n == 0 else _cost(model, s, p, ca, c)
        row[f"flops_{r}"] = 2 * ACTIVE_PARAMS_B[model] * 1e9 * (p + c) if n else 0.0
        row[f"llm_sec_{r}"] = float(sub["latency_sec"].sum()) if n else 0.0
        tsub = tools[tools["role"] == role] if len(tools) else tools
        # spawn_subagents blocks on the group; its time is not "tool work".
        tsub = tsub[tsub["name"] != "spawn_subagents"] if len(tsub) else tsub
        row[f"tool_sec_{r}"] = float(tsub["duration_sec"].sum()) if len(tsub) else 0.0
    row["total_tokens"] = total_tokens
    for s in SCHEMES:
        for r, v in cost_role[s].items():
            row[f"cost_{s}_{r}"] = v
        row[f"cost_{s}_total"] = sum(cost_role[s].values())  # NaN propagates: never 0 for unpriced
    row["cost_runtime_total"] = float(calls["cost_usd"].sum())
    row["flops_total"] = sum(row[f"flops_{r}"] for r in ROLES.values())
    row["backoff_sec"] = float(calls["backoff_sec"].sum())
    row["n_retries"] = int((calls["n_attempts"] - 1).sum())
    row["n_command_timeouts"] = int(tools["timed_out"].sum())
    wall = out["t_end"] - ep.t0
    cp_tool = out["cp_orch_tool"] + ep.cp_group_tool
    cp_total = out["cp_total"]
    llm_cp = out["cp_orch_llm"] + ep.cp_sub_llm + ep.cp_arb_llm
    sub_busy = float(sum(d["duration_sec"] for d in ep.dispatches))
    row.update({
        "wall_clock_sec": wall,
        "orchestrator_steps": len(ep.steps),
        "spawn_calls": len(ep.spawns),
        "delegated": int(len(ep.spawns) > 0),
        "dispatches": len(ep.dispatches),
        "redispatches": sum(1 for d in ep.dispatches if d["group_id"] != (ep.spawns[0]["group_id"] if ep.spawns else None)),
        "arbitrations": len(ep.arbitrations),
        "instruction_following": (float(np.mean([s["granted"] / s["requested"] for s in ep.spawns])) if ep.spawns else float("nan")),
        "dispatch_submitted_frac": (float(np.mean([d["status"] == "submitted" for d in ep.dispatches])) if ep.dispatches else float("nan")),
        "n_files_modified": int(rng.integers(0, 6)),
        "cp_total_sec": cp_total,
        "cp_orch_llm_sec": out["cp_orch_llm"],
        "cp_sub_llm_sec": ep.cp_sub_llm,
        "cp_arb_llm_sec": ep.cp_arb_llm,
        "cp_tool_sec": cp_tool,
        "cp_group_sec": ep.cp_in_groups,
        "overhead_sec": wall - cp_total,
        "sub_busy_sec": sub_busy,
        "parallel_speedup": (sub_busy / max(ep.cp_in_groups, 1e-6)) if ep.spawns and sub_busy > 0 else float("nan"),
        "sim_measured_wall_sec": cp_total,
        "sim_fitted_wall_sec": cp_tool + 0.93 * llm_cp,
        "sim_roofline_h100_wall_sec": cp_tool + 0.22 * llm_cp,
    })
    env_setup = float(rng.uniform(8, 60))
    agent_setup = float(rng.uniform(0.5, 3))
    verifier = float(rng.uniform(4, 30))
    row.update({
        "env_setup_sec": env_setup,
        "agent_setup_sec": agent_setup,
        "agent_exec_sec": wall + 0.4,
        "verifier_sec": verifier,
        "trial_total_sec": env_setup + agent_setup + wall + 0.4 + verifier + 1.0,
        "reward": 1.0 if resolved else 0.0,
        "resolved": int(resolved),
        "harbor_exception": None,
        "exit_reason": out["exit_reason"],
        "valid": True,
        "duplicate": False,
    })
    return row


def _keys(ep: _Episode) -> dict:
    return {"condition": ep.cond.name, "task_name": ep.task["name"], "round": ep.round, "episode_id": ep.episode_id}


def _wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _boot(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    if len(v) == 0 or np.isnan(v).any():
        return float("nan"), float("nan"), float("nan")
    mean = float(v.mean())
    if len(v) < 2:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return mean, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _condition_table(eps: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    metrics = (["wall_clock_sec"] + [f"sim_{p}_wall_sec" for p in PROFILES] + [f"cost_{s}_total" for s in SCHEMES]
               + ["cost_runtime_total", "total_tokens", "dispatches"]
               + [f"{r}_{k}" for r in ("orch", "sub") for k in
                  ("prompt_tokens", "cached_tokens", "uncached_prompt_tokens", "completion_tokens", "reasoning_tokens")])
    rows = []
    for key, g in eps[~eps["duplicate"].astype(bool)].groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row: dict[str, Any] = dict(zip(by, key))
        first = g.iloc[0]
        for col in ("topology", "strategy", "scope", "orchestrator_model", "subagent_model", "orchestrator_tools"):
            row[col] = first[col]
        v = g[g["valid"].astype(bool)]
        k = int(v["resolved"].sum())
        p, lo, hi = _wilson(k, len(v))
        row.update({"n": len(g), "n_valid": len(v), "n_resolved": k, "resolve_rate": p, "resolve_ci_lo": lo, "resolve_ci_hi": hi})
        per_task = v.groupby("task_name")["resolved"].mean()
        took = v[v["delegated"] == 1]
        kd = int(took["resolved"].sum())
        dlo, dhi = _wilson(kd, len(took))[1:]
        row.update({"n_tasks_valid": len(per_task),
                    "resolve_rate_task_balanced": float(per_task.mean()) if len(per_task) else float("nan"),
                    "n_delegated": len(took), "delegated_frac": len(took) / len(v) if len(v) else float("nan"),
                    "resolve_rate_delegated": kd / len(took) if len(took) else float("nan"),
                    "resolve_delegated_ci_lo": dlo, "resolve_delegated_ci_hi": dhi})
        for m in metrics:
            row[f"{m}_mean"], row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = _boot(v[m].to_numpy(dtype=float))
        for s in SCHEMES:
            tot = v[f"cost_{s}_total"].to_numpy(dtype=float)
            row[f"cost_per_solve_{s}"] = (float(tot.sum()) / k) if (k and not np.isnan(tot).any()) else float("nan")
        row["time_per_solve_sec"] = float(v["wall_clock_sec"].sum()) / k if k else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _paired(eps: pd.DataFrame) -> pd.DataFrame:
    v = eps[eps["valid"].astype(bool) & ~eps["duplicate"].astype(bool)]
    out = []
    for base in ("single_5step", "single_matched"):
        b = v[v["condition"] == base].set_index(["task_name", "round"])["resolved"]
        for cond, g in v.groupby("condition", sort=False):
            if cond == base:
                continue
            c = g.set_index(["task_name", "round"])["resolved"]
            j = pd.concat([c.rename("c"), b.rename("b")], axis=1, join="inner")
            both = int(((j["c"] == 1) & (j["b"] == 1)).sum())
            oc = int(((j["c"] == 1) & (j["b"] == 0)).sum())
            ob = int(((j["c"] == 0) & (j["b"] == 1)).sum())
            n = oc + ob
            # exact two-sided binomial McNemar p-value
            p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(oc, ob) + 1)) / 2 ** n)
            out.append({"condition": cond, "baseline": base, "n_pairs": len(j), "both": both, "only_cond": oc,
                        "only_base": ob, "delta_resolve": (oc - ob) / len(j) if len(j) else float("nan"), "mcnemar_p": p})
    return pd.DataFrame(out)


def _calibration(calls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, provider, window), g in calls.groupby(["model", "provider", "completion_window"]):
        x = g["prompt_tokens"].to_numpy(float) / 1000.0
        y = g["ttft_sec"].to_numpy(float)
        A = np.vstack([np.ones_like(x), x]).T
        (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        dec = g[g["decode_sec"] > 0]
        rows.append({
            "model": model, "provider": provider, "completion_window": window, "n_calls": len(g),
            "ttft_median": float(np.median(y)), "ttft_p90": float(np.percentile(y, 90)),
            "decode_tok_s_median": float(np.median(dec["completion_tokens"] / dec["decode_sec"])),
            "ttft_fit_a": float(a), "ttft_fit_b_per_1k_prompt_tokens": float(b),
            "latency_median": float(g["latency_sec"].median()),
        })
    return pd.DataFrame(rows)


def write_synthetic_analysis(
    out_dir: str | Path,
    *,
    seed: int = 0,
    conditions: list[Condition] | None = None,
    tasks: list[dict[str, Any]] | None = None,
    rounds: list[int] | None = None,
    invalid: tuple | list[tuple] | None = ("gemma31b_sequential_medium", "raman-fitting", 2),
    duplicate: tuple[str, str, int] | None = ("gemma31b_parallel_hard", "fix-git", 1),
) -> Path:
    """Simulate a sweep and write its analysis tables to ``out_dir``.

    ``invalid`` names (condition, task, round) trials that fail in the
    environment build (``valid=False``, no events), to exercise filtering;
    one tuple or a list, and a task or round of None matches every one.
    ``duplicate`` names one trial that is also re-run: the copy, with the
    opposite outcome, is flagged ``duplicate`` in episodes.csv and left out
    of the condition tables, as the aggregator does.
    """
    if invalid and isinstance(invalid[0], str):
        invalid = [invalid]

    def is_invalid(cond: str, task: str, rnd: int) -> bool:
        return any(c == cond and t in (None, task) and r in (None, rnd) for c, t, r in (invalid or []))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    conds = conditions or default_conditions()
    tasks = tasks or TASKS
    rounds = rounds or ROUNDS
    sweep_hours = 0.0
    t_job = SWEEP_T0
    ep_rows: list[dict] = []
    tables: dict[str, list[dict]] = {k: [] for k in ("calls", "tools", "steps", "spawns", "dispatches", "arbitrations")}
    n_jobs = len(rounds) * len(conds)
    job_i = 0
    for rnd in rounds:
        order = list(conds)
        rng.shuffle(order)
        for cond in order:
            job_name = f"pilot_synth__r{rnd}__{cond.name}"
            job_end = t_job
            for task in tasks:
                eid = f"ep{len(ep_rows):04d}{int(rng.integers(0, 16**4)):04x}"
                base = {
                    "sweep": "pilot_synth", "job_name": job_name, "round": rnd, "condition": cond.name,
                    "trial_name": f"{task['name']}__{eid[-6:]}", "task_name": task["name"], "episode_id": eid,
                    "topology": cond.topology, "strategy": cond.strategy, "scope": cond.scope,
                    "orchestrator_model": ORCH, "subagent_model": cond.sub_model,
                    "orchestrator_tools": cond.orchestrator_tools,
                    "arbiter_model": ORCH if cond.strategy == "arbiter" else None,
                    "max_concurrent": 3 if cond.topology == "subagents" else None,
                    "n_candidates": 3 if cond.strategy == "arbiter" else None,
                    "orchestrator_steps_budget": cond.steps,
                    "difficulty": task["difficulty"], "category": task["category"],
                    "expert_time_min": task["expert"], "junior_time_min": task["junior"],
                }
                if is_invalid(cond.name, task["name"], rnd):
                    base.update({"valid": False, "harbor_exception": "EnvironmentBuildError", "exit_reason": None,
                                 "reward": None, "resolved": None, "env_setup_sec": 41.0})
                    ep_rows.append(base)
                    continue
                t0 = t_job + float(rng.uniform(5, 40))
                drift = 1.0 + 0.25 * (job_i / max(n_jobs - 1, 1))
                ep = _Episode(rng=rng, cond=cond, task=task, round=rnd, episode_id=eid, t0=t0, drift=drift)
                res = ep.run()
                resolved = bool(rng.random() < min(0.95, task["p"] * cond.p_mult))
                row = {**base, **_episode_row(ep, res, resolved)}
                ep_rows.append(row)
                if duplicate and (cond.name, task["name"], rnd) == tuple(duplicate):
                    ep_rows.append({**row, "trial_name": row["trial_name"] + "-rerun", "episode_id": eid + "d",
                                    "resolved": int(not resolved), "reward": float(not resolved), "duplicate": True})
                job_end = max(job_end, t0 + row["trial_total_sec"])
                keys = _keys(ep)
                for name in tables:
                    tables[name].extend({**keys, **r} for r in getattr(ep, name))
            t_job = job_end + 5.0
            job_i += 1
    sweep_hours = (t_job - SWEEP_T0) / 3600.0
    eps = pd.DataFrame(ep_rows)
    eps["valid"] = eps["valid"].fillna(True).astype(bool)
    eps["duplicate"] = eps["duplicate"].fillna(False).astype(bool)
    eps.to_csv(out / "episodes.csv", index=False)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    _condition_table(eps, ["condition"]).to_csv(out / "conditions.csv", index=False)
    _condition_table(eps, ["condition", "difficulty"]).to_csv(out / "conditions_by_difficulty.csv", index=False)
    _paired(eps).to_csv(out / "paired.csv", index=False)
    _calibration(pd.DataFrame(tables["calls"])).to_csv(out / "calibration.csv", index=False)
    (out / "summary.json").write_text(json.dumps({
        "synthetic": True, "sweep": "pilot_synth", "n_episodes": len(eps), "n_valid": int(eps["valid"].sum()),
        "conditions": [c.name for c in conds], "schemes": SCHEMES, "profiles": PROFILES,
        "sweep_hours": round(sweep_hours, 3),
    }, indent=2))
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python tests/synth_tables.py OUT_DIR", file=sys.stderr)
        raise SystemExit(2)
    print(write_synthetic_analysis(sys.argv[1]))
