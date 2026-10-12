"""Critical path of an episode and offline latency simulation under serving profiles.

Model (``docs/DESIGN.md`` section 6, "Latency simulation and critical path"):
every agent is a sequential chain of its calls and tools, ordered by
``agent_seq``. A chain's duration is the sum of its event durations, except
that a ``spawn_subagents`` tool on the orchestrator chain is replaced by the
duration of the group it created:

=============  ==============================================================
parallel       max over member dispatch chains (list-scheduled on
               ``concurrency`` slots, which is the max whenever
               granted <= concurrency, as the runtime guarantees)
sequential     sum over member dispatch chains
arbiter        max over the group's arbitrations of: isolation setup
               (``isolation_sec``: private copies) + max(candidate chains)
               + diff summaries (first round's ``t_start`` minus
               ``t_candidates_end``) + sum over judge rounds of (judge call
               + max of that round's debate calls) + apply and cleanup
               (arbitration ``t_end`` minus the latest end of any judge or
               debate call counted, so a round aborted mid-debate is not
               counted twice); the setup, diff and apply parts are always
               measured
=============  ==============================================================

The episode duration is the orchestrator chain. Only LLM call durations
change between serving profiles; tool durations (and arbiter apply time) are
always the measured ones. With the ``measured`` profile the simulation
replays logged call latencies and therefore equals ``cp_total_sec`` by
construction; ``wall_clock_sec - cp_total_sec`` is the harness overhead the
model does not capture: the harness's own measurement commands (markers,
modified-file scans, ``git diff --stat``; ``EpisodeRecord.harness_exec_sec``),
thread start-up and record writing. Each chain element is attributed to ``orch_llm``, ``sub_llm``,
``arb_llm`` (judge and debate calls) or ``tool``.

Serving profiles (``configs/serving.yaml``):

* ``measured`` -- the call's logged ``latency_sec`` (retries and backoff included);
* ``fitted`` -- ``lat_fit_a + lat_fit_b * prompt/1000 + lat_fit_c * completion``
  per (model, provider, completion_window): one least-squares fit of the
  whole call time (:func:`fit_target_sec`) with every coefficient >= 0, from
  ``calibration.csv`` or a file written by ``subagents-bench calibrate``. One
  fit of the total (rather than a TTFT fit plus a median decode rate) keeps
  the per-call intercept that short replies pay, so in-sample its total
  matches the measured total and its error does not depend on reply length
  (``summary.json`` ``latency_checks.fitted_vs_measured`` checks both). Params
  files from before the joint fit (``ttft_a_sec``, ``decode_tok_s``) still
  load, as ``c = 1 / decode_tok_s``;
* ``roofline`` -- theoretical dedicated batch-1 serving:
  ``prefill_tok_s = mfu * peak_flops / (2 * active_params)``,
  ``decode_tok_s = mbu * peak_bw / (active_params * bytes_per_param)``,
  ``ttft = overhead_sec + prompt / prefill_tok_s``. Prefill and decode use
  models.yaml's ``active_params_prefill_b`` / ``active_params_decode_b`` when
  given; bytes_per_param comes from a per-model profile override, else
  models.yaml's effective value, else the profile default;
* ``table`` -- measured local-GPU numbers per model:
  ``ttft_base_sec + prompt / prefill_tok_s + completion / decode_tok_s``.

A call whose model has no parameters in a profile simulates to NaN, and NaN
propagates to the episode: unknown, never a made-up number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from .load import EpisodeData
from .pricing import ModelCatalog, _num

NAN = float("nan")
CATEGORIES = ("orch_llm", "sub_llm", "arb_llm", "tool")
ROLE_CATEGORY = {"orchestrator": "orch_llm", "subagent": "sub_llm", "arbiter": "arb_llm", "debate": "arb_llm"}
SPAWN_TOOL = "spawn_subagents"


# ---------------------------------------------------------------------------
# Path costs
# ---------------------------------------------------------------------------


@dataclass
class PathCost:
    """Duration of a (sub)path split by category; ``group_sec`` = time inside spawn groups."""

    parts: dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in CATEGORIES})
    group_sec: float = 0.0

    @property
    def total(self) -> float:
        return sum(self.parts.values())

    @classmethod
    def of(cls, category: str, sec: float) -> "PathCost":
        pc = cls()
        pc.parts[category] = float(sec)
        return pc

    @classmethod
    def nan(cls) -> "PathCost":
        return cls(parts={c: NAN for c in CATEGORIES}, group_sec=NAN)

    def __add__(self, other: "PathCost") -> "PathCost":
        return PathCost(
            parts={c: self.parts[c] + other.parts[c] for c in CATEGORIES},
            group_sec=self.group_sec + other.group_sec,
        )


def _max_cost(costs: list[PathCost]) -> PathCost:
    if not costs:
        return PathCost()
    if any(math.isnan(c.total) for c in costs):
        return PathCost.nan()
    return max(costs, key=lambda c: c.total)


def measured_call_sec(call: dict) -> float:
    lat = call.get("latency_sec")
    if lat is None:
        t0, t1 = call.get("t_start"), call.get("t_end")
        return float(t1 - t0) if t0 is not None and t1 is not None else NAN
    return float(lat)


def fit_target_sec(call: dict) -> float:
    """What the ``fitted`` profile models: the serving time of a successful call.

    ``ttft_sec + decode_sec`` (the final attempt, request to last chunk) when
    both were measured, else ``latency_sec - backoff_sec``; NaN if neither is
    known. Retry backoff is left out, so the fit describes one served request.
    A single-chunk reply has ``decode_sec`` 0 and its whole time in TTFT, which
    this total handles like any other call.
    """
    ttft, dec = call.get("ttft_sec"), call.get("decode_sec")
    if ttft is not None and dec is not None:
        return float(ttft) + float(dec)
    lat = call.get("latency_sec")
    if lat is None:
        return NAN
    return float(lat) - float(call.get("backoff_sec") or 0.0)


def _tool_sec(tool: dict) -> float:
    d = tool.get("duration_sec")
    if d is None:
        t0, t1 = tool.get("t_start"), tool.get("t_end")
        return float(t1 - t0) if t0 is not None and t1 is not None else NAN
    return float(d)


def _order(rec: dict) -> tuple:
    return (rec.get("agent_seq") if rec.get("agent_seq") is not None else 1 << 30, rec.get("seq") or 0)


class EpisodeGraph:
    """Index of an episode's records by agent chain, group, dispatch and arbitration."""

    def __init__(self, data: EpisodeData):
        self.data = data
        self.calls = {c.get("call_id"): c for c in data.calls}
        self.spawns = {s.get("group_id"): s for s in data.spawns}
        self.dispatches = {d.get("dispatch_id"): d for d in data.dispatches}
        self.arbs_by_group: dict[str, list[dict]] = {}
        for a in data.arbitrations:
            self.arbs_by_group.setdefault(a.get("group_id"), []).append(a)
        for arbs in self.arbs_by_group.values():
            arbs.sort(key=lambda a: (a.get("instruction_index") or 0))
        self.orch_chain = self._chain(
            [("call", c) for c in data.calls if c.get("role") == "orchestrator"]
            + [("tool", t) for t in data.tools if t.get("role") == "orchestrator"]
        )
        self.dispatch_chains: dict[str, list[tuple[str, dict]]] = {}
        for c in data.calls:
            if c.get("role") == "subagent" and c.get("dispatch_id"):
                self.dispatch_chains.setdefault(c["dispatch_id"], []).append(("call", c))
        for t in data.tools:
            if t.get("role") == "subagent" and t.get("dispatch_id"):
                self.dispatch_chains.setdefault(t["dispatch_id"], []).append(("tool", t))
        for k, v in self.dispatch_chains.items():
            self.dispatch_chains[k] = self._chain(v)

    @staticmethod
    def _chain(events: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
        return sorted(events, key=lambda e: _order(e[1]))

    def group_members(self, group_id: str) -> list[str]:
        """Member dispatch ids of a group, in spawn order (spawn record, else dispatch records)."""
        spawn = self.spawns.get(group_id)
        if spawn and spawn.get("dispatch_ids"):
            return list(spawn["dispatch_ids"])
        members = [d for d in self.data.dispatches if d.get("group_id") == group_id]
        members.sort(key=lambda d: (d.get("instruction_index") or 0, d.get("candidate_index") or 0, d.get("t_start") or 0))
        ids = [d["dispatch_id"] for d in members]
        # Dispatches still running at a crash have calls but no record.
        for c in self.data.calls:
            if c.get("group_id") == group_id and c.get("role") == "subagent" and c.get("dispatch_id") not in ids:
                ids.append(c["dispatch_id"])
        return ids

    def group_strategy(self, group_id: str) -> str | None:
        spawn = self.spawns.get(group_id)
        if spawn and spawn.get("strategy"):
            return spawn["strategy"]
        for d in self.data.dispatches:
            if d.get("group_id") == group_id:
                return d.get("strategy")
        return None

    def has_group(self, group_id: str | None) -> bool:
        return bool(group_id) and (group_id in self.spawns or bool(self.group_members(group_id)))


class CriticalPath:
    """Evaluate the chain model of one episode with a pluggable LLM-call duration."""

    def __init__(self, graph: EpisodeGraph, call_sec: Callable[[dict], float] = measured_call_sec):
        self.g = graph
        self.call_sec = call_sec

    def call_cost(self, call: dict) -> PathCost:
        return PathCost.of(ROLE_CATEGORY.get(call.get("role"), "orch_llm"), self.call_sec(call))

    def chain(self, events: Iterable[tuple[str, dict]]) -> PathCost:
        total = PathCost()
        for kind, rec in events:
            if kind == "call":
                total = total + self.call_cost(rec)
            elif rec.get("name") == SPAWN_TOOL and self.g.has_group(rec.get("group_id")):
                total = total + self.group(rec["group_id"])
            else:
                total = total + PathCost.of("tool", _tool_sec(rec))
        return total

    def dispatch(self, dispatch_id: str) -> PathCost:
        return self.chain(self.g.dispatch_chains.get(dispatch_id, []))

    def group(self, group_id: str) -> PathCost:
        if self.g.group_strategy(group_id) == "arbiter" and self.g.arbs_by_group.get(group_id):
            cost = _max_cost([self.arbitration(a) for a in self.g.arbs_by_group[group_id]])
        else:
            cost = self._scheduled_group(group_id)
        cost.group_sec = cost.total
        return cost

    def _scheduled_group(self, group_id: str) -> PathCost:
        members = [self.dispatch(d) for d in self.g.group_members(group_id)]
        if not members:
            return PathCost()
        if any(math.isnan(m.total) for m in members):
            return PathCost.nan()
        strategy = self.g.group_strategy(group_id)
        spawn = self.g.spawns.get(group_id) or {}
        if strategy == "sequential":
            slots = 1
        elif strategy == "arbiter":  # arbitration records missing (crash): candidates ran concurrently
            slots = len(members)
        else:
            slots = int(spawn.get("concurrency") or 0) or len(members)
        # Greedy list scheduling in spawn order: each member takes the earliest-free slot.
        lanes = [PathCost() for _ in range(min(slots, len(members)))]
        for m in members:
            i = min(range(len(lanes)), key=lambda k: lanes[k].total)
            lanes[i] = lanes[i] + m
        return _max_cost(lanes)

    def arbitration(self, arb: dict) -> PathCost:
        # Isolation setup (private copies) before the candidates: measured.
        total = PathCost.of("tool", max(0.0, float(arb.get("isolation_sec") or 0.0)))
        total = total + _max_cost([self.dispatch(d) for d in arb.get("candidate_dispatch_ids") or []])
        rounds = sorted(arb.get("rounds") or [], key=lambda r: r.get("round") or 0)
        # Diff summaries for the judge, between the candidates' end and the first round: measured.
        if rounds and rounds[0].get("t_start") is not None and arb.get("t_candidates_end") is not None:
            total = total + PathCost.of("tool", max(0.0, float(rounds[0]["t_start"]) - float(arb["t_candidates_end"])))
        # Apply and cleanup run from the end of the last judge OR debate call counted: a round
        # aborted mid-debate (cancel) has debate calls and no judge, and measuring from the
        # previous judge would count that debate a second time.
        ends: list[float] = []
        for rnd in rounds:
            dcalls = [self.g.calls[c] for c in rnd.get("debate_call_ids") or [] if c in self.g.calls]
            total = total + _max_cost([self.call_cost(c) for c in dcalls])
            ends += [float(c["t_end"]) for c in dcalls if c.get("t_end") is not None]
            judge = self.g.calls.get(rnd.get("judge_call_id"))
            if judge is not None:
                total = total + self.call_cost(judge)
                if judge.get("t_end") is not None:
                    ends.append(float(judge["t_end"]))
        apply_sec = 0.0
        if ends and arb.get("t_end") is not None:
            apply_sec = max(0.0, float(arb["t_end"]) - max(ends))
        return total + PathCost.of("tool", apply_sec)

    def episode(self) -> PathCost:
        return self.chain(self.g.orch_chain)


def critical_path(data: EpisodeData, call_sec: Callable[[dict], float] = measured_call_sec) -> PathCost:
    return CriticalPath(EpisodeGraph(data), call_sec).episode()


# ---------------------------------------------------------------------------
# Serving profiles
# ---------------------------------------------------------------------------


def _tok(call: dict, key: str) -> float:
    v = call.get(key)
    return float(v) if v is not None else 0.0


def calib_key(model: Any, provider: Any, window: Any) -> tuple[str, str, str]:
    """Normalized (model, provider, completion_window) key; missing parts become ''."""

    def norm(v: Any) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        return str(v)

    return (norm(model), norm(provider), norm(window))


@dataclass
class MeasuredProfile:
    name: str = "measured"
    kind: str = "measured"
    description: str = "replay of logged call latencies"

    def call_sec(self, call: dict) -> float:
        return measured_call_sec(call)

    def describe(self) -> dict:
        return {"kind": self.kind, "description": self.description}


@dataclass
class FitParams:
    """``call_sec = a_sec + b * prompt_tokens/1000 + c * completion_tokens``."""

    a_sec: float
    b_sec_per_1k_prompt_tokens: float
    c_sec_per_completion_token: float
    n_calls: int | None = None
    # lat_fit: the joint fit of the whole call; ttft_fit+decode_rate: an older params file
    # (TTFT fit plus a median decode rate, c = 1 / decode_tok_s).
    form: str = "lat_fit"


@dataclass
class FittedProfile:
    name: str
    params: dict[tuple[str, str, str], FitParams]
    source: str
    kind: str = "fitted"
    description: str = ""

    def call_sec(self, call: dict) -> float:
        p = self.params.get(calib_key(call.get("model"), call.get("provider"), call.get("completion_window")))
        if p is None:
            return NAN
        sec = p.a_sec + p.b_sec_per_1k_prompt_tokens * _tok(call, "prompt_tokens") / 1000.0
        completion = _tok(call, "completion_tokens")
        if completion > 0:
            if not (p.c_sec_per_completion_token >= 0 and math.isfinite(p.c_sec_per_completion_token)):
                return NAN
            sec += p.c_sec_per_completion_token * completion
        return sec

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "description": self.description,
            "source": self.source,
            "params": [
                {"model": k[0], "provider": k[1], "completion_window": k[2] or None, **vars(v)}
                for k, v in sorted(self.params.items())
            ],
        }


@dataclass
class RooflineProfile:
    name: str
    catalog: ModelCatalog
    mfu: float
    mbu: float
    bytes_per_param: float
    overhead_sec: float
    peak_flops: float
    peak_bw: float
    prefill_tokens: str = "prompt"  # prompt | uncached
    model_overrides: dict[str, dict] = field(default_factory=dict)
    # Use models.yaml's effective bytes_per_param (exact, from the served checkpoint) when known.
    use_model_bytes_per_param: bool = True
    theoretical: bool = True
    kind: str = "roofline"
    description: str = ""

    def bytes_per_param_for(self, model: str | None) -> tuple[float, str]:
        """(bytes_per_param, where it came from): profile override > models.yaml > profile default."""
        ov = self.catalog.lookup(self.model_overrides, model) or {}
        if _num(ov.get("bytes_per_param")):
            return (float(_num(ov["bytes_per_param"])), "profile override")
        if self.use_model_bytes_per_param:
            b = self.catalog.bytes_per_param(model)
            if b:
                return (b, "models.yaml")
        return (self.bytes_per_param, "profile default")

    def rates(self, model: str | None) -> tuple[float, float, float] | None:
        """(prefill_tok_s, decode_tok_s, overhead_sec) for ``model``; None if params unknown.

        Prefill uses the prefill active-parameter count and decode the decode
        count when models.yaml distinguishes them.
        """
        ov = self.catalog.lookup(self.model_overrides, model) or {}
        if _num(ov.get("active_params_b")):
            pre_b = dec_b = float(_num(ov["active_params_b"]))
        else:
            split = self.catalog.active_params_split_b(model)
            if split is None:
                return None
            pre_b, dec_b = split
        if pre_b <= 0 or dec_b <= 0:
            return None
        mfu = _num(ov.get("mfu")) or self.mfu
        mbu = _num(ov.get("mbu")) or self.mbu
        bpp = self.bytes_per_param_for(model)[0]
        overhead = _num(ov.get("overhead_sec"))
        overhead = self.overhead_sec if overhead is None else overhead
        prefill = mfu * self.peak_flops / (2.0 * pre_b * 1e9)
        decode = mbu * self.peak_bw / (dec_b * 1e9 * bpp)
        return (prefill, decode, overhead)

    def call_sec(self, call: dict) -> float:
        r = self.rates(call.get("model"))
        if r is None:
            return NAN
        prefill, decode, overhead = r
        if self.prefill_tokens == "uncached":
            prompt = call.get("uncached_prompt_tokens")
            prompt = float(prompt) if prompt is not None else max(_tok(call, "prompt_tokens") - _tok(call, "cached_tokens"), 0.0)
        else:
            prompt = _tok(call, "prompt_tokens")
        return overhead + prompt / prefill + _tok(call, "completion_tokens") / decode

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "theoretical": self.theoretical,
            "description": self.description,
            "mfu": self.mfu,
            "mbu": self.mbu,
            "bytes_per_param": self.bytes_per_param,
            "overhead_sec": self.overhead_sec,
            "peak_flops": self.peak_flops,
            "peak_bw": self.peak_bw,
            "prefill_tokens": self.prefill_tokens,
            "use_model_bytes_per_param": self.use_model_bytes_per_param,
            "model_overrides": self.model_overrides,
        }


@dataclass
class TableProfile:
    name: str
    catalog: ModelCatalog
    models: dict[str, dict]
    kind: str = "table"
    description: str = ""

    def call_sec(self, call: dict) -> float:
        m = self.catalog.lookup(self.models, call.get("model"))
        if not isinstance(m, dict):
            return NAN
        base, pre, dec = _num(m.get("ttft_base_sec")), _num(m.get("prefill_tok_s")), _num(m.get("decode_tok_s"))
        if base is None or not pre or not dec:
            return NAN
        return base + _tok(call, "prompt_tokens") / pre + _tok(call, "completion_tokens") / dec

    def describe(self) -> dict:
        return {"kind": self.kind, "description": self.description, "models": self.models}


COMPLETION_BINS = ((0, 100), (100, 400), (400, None))


def fitted_vs_measured(profile: FittedProfile, calls: Iterable[dict]) -> dict[str, dict]:
    """Per model: sum of the profile's call times / sum of :func:`fit_target_sec`.

    Over the successful calls with a measured target, overall and per
    completion-length bin. For a profile fitted in-sample the overall ratio is
    1 whenever the fitted intercept is positive (residuals of a fit with an
    intercept sum to 0), and the bins show any length-dependent bias; NaN
    when the profile has no parameters for some of the model's calls.
    """
    labels = [f"{lo}-{hi - 1}" if hi else f"{lo}+" for lo, hi in COMPLETION_BINS]
    sums: dict[str, dict[str, list[float]]] = {}
    for c in calls:
        meas = fit_target_sec(c)
        if c.get("error") or not math.isfinite(meas):
            continue
        comp = _tok(c, "completion_tokens")
        b = next(lab for lab, (lo, hi) in zip(labels, COMPLETION_BINS) if comp >= lo and (hi is None or comp < hi))
        fit = profile.call_sec(c)
        for key in ("all", b):
            s = sums.setdefault(str(c.get("model")), {}).setdefault(key, [0.0, 0.0, 0])
            s[0], s[1], s[2] = s[0] + fit, s[1] + meas, s[2] + 1

    def ratio(s: list[float]) -> float:
        return s[0] / s[1] if s[1] > 0 else NAN

    return {
        model: {
            "n_calls": by["all"][2],
            "ratio": ratio(by["all"]),
            "by_completion_tokens": {k: {"n_calls": by[k][2], "ratio": ratio(by[k])} for k in labels if k in by},
        }
        for model, by in sorted(sums.items())
    }


LAT_FIT_KEYS = ("lat_fit_a", "lat_fit_b_per_1k_prompt", "lat_fit_c_per_completion_token")


def fit_params_from_rows(rows: Iterable[dict]) -> dict[tuple[str, str, str], FitParams]:
    """FitParams from calibration rows (calibration.csv columns, or calibrate's YAML entries).

    Rows with the joint fit (``lat_fit_*``) use it. Rows without those keys
    (params files written before the joint fit) fall back to the TTFT fit plus
    decode rate, ``c = 1 / decode_tok_s``.
    """
    out = {}
    for r in rows:
        if any(k in r for k in LAT_FIT_KEYS):
            a, b, c = (_num(r.get(k)) for k in LAT_FIT_KEYS)
            form = "lat_fit"
        else:
            a = _num(r.get("ttft_fit_a", r.get("ttft_a_sec")))
            b = _num(r.get("ttft_fit_b_per_1k_prompt_tokens", r.get("ttft_b_sec_per_1k_prompt_tokens")))
            dec = _num(r.get("decode_tok_s_median", r.get("decode_tok_s")))
            c = 1.0 / dec if dec is not None and dec > 0 else None
            form = "ttft_fit+decode_rate"
        if a is None or b is None or c is None:
            continue  # this model's fit is not identifiable -> its calls simulate to NaN
        n = r.get("lat_fit_n", r.get("n_calls")) if form == "lat_fit" else r.get("n_calls")
        out[calib_key(r.get("model"), r.get("provider"), r.get("completion_window"))] = FitParams(
            a, b, c, int(n) if n is not None and not (isinstance(n, float) and math.isnan(n)) else None, form
        )
    return out


def load_serving(
    path: str | Path | None,
    catalog: ModelCatalog,
    calibration_rows: list[dict] | None = None,
) -> tuple[list[Any], list[str]]:
    """(profiles, warnings). ``measured`` is always present (first)."""
    warnings: list[str] = []
    profiles: list[Any] = [MeasuredProfile()]
    if path is None or not Path(path).is_file():
        warnings.append(f"serving profiles file {path} not found: only the 'measured' replay is simulated")
        return profiles, warnings
    p = Path(path)
    raw = yaml.safe_load(p.read_text()) or {}
    for name, body in (raw.get("profiles") or {}).items():
        name = str(name)
        body = body or {}
        kind = str(body.get("kind") or name)
        desc = str(body.get("description") or "").strip()
        if kind == "measured":
            if name != "measured":
                profiles.append(MeasuredProfile(name=name, description=desc or "replay"))
            elif desc:
                profiles[0].description = desc
        elif kind == "fitted":
            pf = body.get("params_file")
            pf_path = (p.parent / pf) if pf and not Path(pf).is_absolute() else (Path(pf) if pf else None)
            if pf_path is not None and pf_path.is_file():
                fraw = yaml.safe_load(pf_path.read_text()) or {}
                params = fit_params_from_rows(fraw.get("params") or [])
                source = str(pf_path)
            else:
                if pf_path is not None:
                    warnings.append(f"fitted params_file {pf_path} not found: fitting in-sample from calibration.csv")
                params = fit_params_from_rows(calibration_rows or [])
                source = "calibration.csv (in-sample)"
            profiles.append(FittedProfile(name=name, params=params, source=source, description=desc))
        elif kind == "roofline":
            try:
                prof = RooflineProfile(
                    name=name,
                    catalog=catalog,
                    mfu=float(_num(body["mfu"])),
                    mbu=float(_num(body["mbu"])),
                    bytes_per_param=float(_num(body["bytes_per_param"])),
                    overhead_sec=float(_num(body["overhead_sec"])),
                    peak_flops=float(_num(body["peak_flops"])),
                    peak_bw=float(_num(body["peak_bw"])),
                    prefill_tokens=str(body.get("prefill_tokens") or "prompt"),
                    model_overrides={str(k): (v or {}) for k, v in (body.get("models") or {}).items()},
                    use_model_bytes_per_param=bool(body.get("use_model_bytes_per_param", True)),
                    theoretical=bool(body.get("theoretical", True)),
                    description=desc,
                )
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"{p}: roofline profile {name!r} needs numeric mfu, mbu, bytes_per_param, "
                                 f"overhead_sec, peak_flops, peak_bw ({e})") from e
            if prof.prefill_tokens not in ("prompt", "uncached"):
                raise ValueError(f"{p}: profile {name!r}: prefill_tokens must be 'prompt' or 'uncached'")
            profiles.append(prof)
        elif kind == "table":
            profiles.append(TableProfile(name=name, catalog=catalog, description=desc,
                                         models={str(k): (v or {}) for k, v in (body.get("models") or {}).items()}))
        else:
            warnings.append(f"{p}: profile {name!r} has unknown kind {kind!r}; skipped")
    return profiles, warnings
