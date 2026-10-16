"""Measured critical path of an episode, split into orchestrator LLM, subagent LLM,
judge/debate LLM and tool time (overhead = wall clock - critical path).

Every agent is a chain of its calls and tools in series. A blocking ``spawn_subagents`` tool on
the orchestrator chain counts as its group: parallel = max over its dispatches, sequential = sum,
arbiter = its arbitrations in series, each = copy setup + max over candidates + per round
(max over the concurrent debate calls + the judge calls) + the rest (diffs, leak restore,
apply, cleanup). Arbiter shell steps count as tool time. A background spawn (interactive
episodes) returns at once, so it counts as its own short tool time: its subagents run off the
orchestrator's chain, and the orchestrator's ``wait`` calls are tool time.
"""

from __future__ import annotations

from collections import defaultdict

PARTS = ("orch_llm", "sub_llm", "judge_llm", "tool")
CATEGORY = {"orchestrator": "orch_llm", "subagent": "sub_llm", "arbiter": "judge_llm", "debate": "judge_llm"}
Parts = dict[str, float]


def _zero() -> Parts:
    return dict.fromkeys(PARTS, 0.0)


def _sum(paths: list[Parts]) -> Parts:
    return {p: sum(x[p] for x in paths) for p in PARTS}


def _max(paths: list[Parts]) -> Parts:
    return max(paths, key=lambda x: sum(x.values()), default=_zero())


def _dur(rec: dict) -> float:
    return rec["t_end"] - rec["t_start"]


class _Episode:
    def __init__(self, records: dict[str, list[dict]]):
        self.calls = {c["call_id"]: c for c in records["calls"]}
        self.spawns = {s["group_id"]: s for s in records["spawns"]}
        self.arbs = {a["arbitration_id"]: a for a in records["arbitrations"]}
        self.dispatch_start = {d["dispatch_id"]: d["t_start"] for d in records["dispatches"]}
        self.chains: dict[str, list[dict]] = defaultdict(list)  # "orchestrator" or a dispatch id
        for r in records["calls"] + records["tools"]:
            if r["role"] in ("orchestrator", "subagent"):
                self.chains[r["dispatch_id"] or "orchestrator"].append(r)

    def chain(self, key: str) -> Parts:
        out = _zero()
        for r in sorted(self.chains[key], key=lambda r: r["t_start"]):
            if "name" not in r:  # an LLM call
                out[CATEGORY[r["role"]]] += _dur(r)
            elif (r["name"] == "spawn_subagents" and r["group_id"] in self.spawns
                  and self.spawns[r["group_id"]].get("spawn_mode", "blocking") == "blocking"):
                out = _sum([out, self.group(self.spawns[r["group_id"]])])
            else:
                out["tool"] += _dur(r)
        return out

    def group(self, spawn: dict) -> Parts:
        if spawn["strategy"] == "arbiter":
            return _sum([self.arbitration(self.arbs[a]) for a in spawn["arbitration_ids"]])
        paths = [self.chain(d) for d in spawn["dispatch_ids"]]
        return _sum(paths) if spawn["strategy"] == "sequential" else _max(paths)

    def arbitration(self, a: dict) -> Parts:
        cands = a["candidate_dispatch_ids"]
        out = dict(_max([self.chain(d) for d in cands]))
        starts = [self.dispatch_start[d] for d in cands if d in self.dispatch_start]
        out["tool"] += max(0.0, min(starts) - a["t_start"]) if starts else 0.0
        judging = 0.0
        for rnd in a["rounds"]:
            judging += max((_dur(self.calls[c]) for c in rnd["debate_call_ids"]), default=0.0)
            judging += sum(_dur(self.calls[c]) for c in rnd["judge_call_ids"])
        out["judge_llm"] += judging
        if a["t_candidates_end"] is not None:
            out["tool"] += max(0.0, a["t_end"] - a["t_candidates_end"] - judging)
        return out


def critical_path(records: dict[str, list[dict]]) -> Parts:
    """Seconds per part (``PARTS``) on the episode's critical path."""
    return _Episode(records).chain("orchestrator")
