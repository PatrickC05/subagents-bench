"""``check-run`` (schema + internal consistency) and ``completeness`` (field coverage).

``check-run`` validates every record file of every episode against
:mod:`subagents_bench.schema` and cross-checks the records
(``docs/DESIGN.md`` section 6):

* episode totals (``n_calls``, ``tokens``, ``cost_usd``, ``llm_sec``,
  ``tool_sec``, ``n_tool_calls``, retries, timeouts, counts) equal the sums
  over ``calls.jsonl`` / ``tools.jsonl`` / the other record files;
* every ``dispatch_id`` used by calls and tools exists in ``dispatches.jsonl``;
* ``agent_seq`` is contiguous per agent and global ``seq`` is unique;
* timestamps are monotonic along each agent's chain;
* each spawn's ``dispatch_ids`` match the dispatches of its group;
* arbitration judge/debate call ids and candidate dispatch ids exist.

An episode that never finalized (``status == "running"``) is itself an error;
its totals are not checked and dangling references are warnings, since a
crash legitimately leaves them. So does a cancel (``exit_reason ==
"cancelled"``: harbor stopped the agent, e.g. at its timeout, while a spawn
group was still running), so its dangling references are warnings too.
Trials without an episode dir are warnings.

``completeness`` reports, per record file, role and schema field, the
fraction of records where the field is None, empty or zero, and flags fields
that are never populated.

Both also report :func:`treatment_coverage`: per subagents condition, the
share of episodes that delegated, and a problem when none did (or an arbiter
condition ran no arbitration). Field coverage cannot catch that, because a
record file nobody wrote has no fields to flag. ``check-run`` lists it as a
warning; ``completeness --strict`` fails on it.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable

from ..schema import RECORD_TYPES, SCHEMA_VERSION, validate_record
from ..telemetry import is_provider_error
from .load import ROLES, discover, load_episode, load_trials, read_json

ABS_TOL = 1e-6
REL_TOL = 1e-6
TIME_TOL = 1e-3  # seconds of clock jitter tolerated between consecutive events


@dataclass
class Problem:
    level: str  # error | warning
    where: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level.upper()}] {self.where}: {self.message}"


@dataclass
class CheckReport:
    n_episodes: int = 0
    n_trials_without_episode: int = 0
    problems: list[Problem] = field(default_factory=list)
    coverage: list[dict[str, Any]] = field(default_factory=list)  # treatment_coverage rows

    @property
    def errors(self) -> list[Problem]:
        return [p for p in self.problems if p.level == "error"]

    @property
    def warnings(self) -> list[Problem]:
        return [p for p in self.problems if p.level == "warning"]

    @property
    def ok(self) -> bool:
        return self.n_episodes > 0 and not self.errors


def _num(v: Any) -> float | int:
    """A record value as a number for summing; wrong types count 0 (schema checks report them)."""
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def _cost_sum(calls: list[dict]) -> float | None:
    """Sum of run-time call costs; None if any call is unpriced (cost_usd None)."""
    if any(c.get("cost_usd") is None for c in calls):
        return None
    return sum(_num(c.get("cost_usd")) for c in calls)


def _close(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)


class _Checker:
    def __init__(self, episode_dir: Path):
        self.dir = episode_dir
        self.data = load_episode(episode_dir, with_transcript=True)
        self.problems: list[Problem] = []
        ep = self.data.episode or {}
        self.running = ep.get("status") != "done"
        # Cut off mid-run: records of work still in flight (a dispatch, its group) may be missing.
        self.partial = self.running or ep.get("exit_reason") == "cancelled"

    def err(self, msg: str, where: str | None = None) -> None:
        self.problems.append(Problem("error", where or str(self.dir), msg))

    def warn(self, msg: str, where: str | None = None) -> None:
        self.problems.append(Problem("warning", where or str(self.dir), msg))

    def ref(self, msg: str) -> None:
        """A dangling reference: an error for finished episodes, a warning after a crash or cancel."""
        (self.warn if self.partial else self.err)(msg)

    def run(self) -> list[Problem]:
        d = self.data
        for p in d.problems:
            self.err(p)
        if d.episode is None:
            self.err("episode.json missing or unreadable")
            return self.problems
        self.check_schema()
        if d.episode.get("schema_version") != SCHEMA_VERSION:
            self.warn(f"schema_version {d.episode.get('schema_version')} != {SCHEMA_VERSION}")
        stages = [self.check_ids_and_order, self.check_spawns, self.check_arbitrations]
        if self.running:
            self.err(f"episode never finalized (status={d.episode.get('status')!r}); totals not checked")
        else:
            stages.insert(0, self.check_totals)
        for stage in stages:
            try:
                stage()
            except Exception as e:  # malformed records must be reported, never crash the checker
                self.err(f"{stage.__name__} could not run on malformed records ({type(e).__name__}: {e})")
        return self.problems

    # -- schema ------------------------------------------------------------------
    def check_schema(self) -> None:
        for fname, cls in RECORD_TYPES.items():
            if fname == "episode.json":
                recs = [self.data.episode]
            else:
                attr = fname.split(".")[0]
                recs = getattr(self.data, attr)
            for i, rec in enumerate(recs, 1):
                for prob in validate_record(cls, rec):
                    self.err(prob, f"{self.dir / fname}" + (f":{i}" if fname != "episode.json" else ""))

    # -- totals --------------------------------------------------------------------
    def check_totals(self) -> None:
        d, ep = self.data, self.data.episode
        by_role_calls = {r: [c for c in d.calls if c.get("role") == r] for r in ROLES}
        by_role_tools = {r: [t for t in d.tools if t.get("role") == r] for r in ROLES}

        def get(dct: Any, role: str, default: Any = 0) -> Any:
            return (dct or {}).get(role, default) if isinstance(dct, dict) else default

        for r in ROLES:
            rc, rt = by_role_calls[r], by_role_tools[r]
            if get(ep.get("n_calls"), r) != len(rc):
                self.err(f"n_calls[{r}]={get(ep.get('n_calls'), r)} but calls.jsonl has {len(rc)}")
            if get(ep.get("n_tool_calls"), r) != len(rt):
                self.err(f"n_tool_calls[{r}]={get(ep.get('n_tool_calls'), r)} but tools.jsonl has {len(rt)}")
            tok = get(ep.get("tokens"), r, {}) or {}
            sums = {
                "prompt": sum(_num(c.get("prompt_tokens")) for c in rc),
                "completion": sum(_num(c.get("completion_tokens")) for c in rc),
                "cached": sum(_num(c.get("cached_tokens")) for c in rc),
                "uncached_prompt": sum(_num(c.get("uncached_prompt_tokens")) for c in rc),
            }
            for k, v in sums.items():
                if (tok.get(k) or 0) != v:
                    self.err(f"tokens[{r}][{k}]={tok.get(k)} but calls sum to {v}")
            reasoning = [_num(c["reasoning_tokens"]) for c in rc if c.get("reasoning_tokens") is not None]
            exp_reasoning = sum(reasoning) if reasoning else None
            got_reasoning = tok.get("reasoning")
            if not (got_reasoning == exp_reasoning or (not reasoning and got_reasoning in (0, None))):
                self.err(f"tokens[{r}][reasoning]={got_reasoning} but calls sum to {exp_reasoning}")
            cost = _cost_sum(rc)
            if not _close(get(ep.get("cost_usd"), r, 0.0), cost):
                self.err(f"cost_usd[{r}]={get(ep.get('cost_usd'), r)} but calls sum to {cost}")
            llm = sum(_num(c.get("latency_sec")) for c in rc)
            if not _close(get(ep.get("llm_sec"), r, 0.0), llm):
                self.err(f"llm_sec[{r}]={get(ep.get('llm_sec'), r)} but calls sum to {llm}")
            # tool_sec includes the time the orchestrator blocked in spawn_subagents (schema doc).
            tool_all = sum(_num(t.get("duration_sec")) for t in rt)
            got_tool = get(ep.get("tool_sec"), r, 0.0)
            if not _close(got_tool, tool_all):
                self.err(f"tool_sec[{r}]={got_tool} but tools sum to {tool_all}")
        total_cost = _cost_sum(d.calls)
        if not _close(get(ep.get("cost_usd"), "total", 0.0), total_cost):
            self.err(f"cost_usd[total]={get(ep.get('cost_usd'), 'total')} but calls sum to {total_cost}")

        scalar_checks = {
            "n_retries": sum(max((_num(c.get("n_attempts")) or 1) - 1, 0) for c in d.calls),
            "n_command_timeouts": sum(1 for t in d.tools if t.get("timed_out")),
            "dispatches": len(d.dispatches),
            "spawn_calls": len(d.spawns),
            "arbitrations": len(d.arbitrations),
            "orchestrator_steps": len(d.steps),
            "requested_total": sum(_num(s.get("requested")) for s in d.spawns),
            "granted_total": sum(_num(s.get("granted")) for s in d.spawns),
        }
        if "n_provider_errors" in ep:  # schema_version >= 3
            scalar_checks["n_provider_errors"] = sum(1 for c in d.calls if is_provider_error(c.get("error")))
        if "n_stream_repairs" in ep:
            scalar_checks["n_stream_repairs"] = sum(_num(c.get("stream_repairs")) for c in d.calls)
        for k, v in scalar_checks.items():
            if ep.get(k) != v:
                self.err(f"{k}={ep.get(k)} but the records give {v}")
        backoff = sum(_num(c.get("backoff_sec")) for c in d.calls)
        if not _close(ep.get("backoff_sec") or 0.0, backoff):
            self.err(f"backoff_sec={ep.get('backoff_sec')} but calls sum to {backoff}")
        trims = sum(_num(x.get("context_trims")) for x in d.dispatches)
        if (ep.get("n_context_trims") or 0) < trims:
            self.err(f"n_context_trims={ep.get('n_context_trims')} < sum over dispatches {trims}")
        if d.spawns:
            first = min(d.spawns, key=lambda s: (s.get("t_start") or 0))["group_id"]
            redis = sum(1 for x in d.dispatches if x.get("group_id") != first)
            if ep.get("redispatches") != redis:
                self.err(f"redispatches={ep.get('redispatches')} but dispatches outside the first group: {redis}")
            ratios = [s["granted"] / s["requested"] for s in d.spawns if (s.get("requested") or 0) > 0]
            if len(ratios) == len(d.spawns) and not _close(ep.get("instruction_following"), sum(ratios) / len(ratios)):
                self.err(f"instruction_following={ep.get('instruction_following')} but spawns give "
                         f"{sum(ratios) / len(ratios)}")
        elif ep.get("instruction_following") is not None:
            self.err("instruction_following set but there are no spawns")
        if ep.get("t_end") is not None and ep.get("t_start") is not None:
            if not _close(ep.get("wall_clock_sec"), ep["t_end"] - ep["t_start"]):
                self.err(f"wall_clock_sec={ep.get('wall_clock_sec')} != t_end - t_start")

    # -- ids, seq, ordering ----------------------------------------------------------
    def check_ids_and_order(self) -> None:
        d = self.data
        dispatch_ids = {x.get("dispatch_id") for x in d.dispatches}
        call_ids = {c.get("call_id") for c in d.calls}
        for kind, recs in (("calls", d.calls), ("tools", d.tools)):
            for rec in recs:
                did = rec.get("dispatch_id")
                if did is not None and did not in dispatch_ids:
                    self.ref(f"{kind}: dispatch_id {did!r} not in dispatches.jsonl")
        for t in d.tools:
            if t.get("call_id") not in call_ids:
                self.ref(f"tools: tool {t.get('tool_event_id')!r} requested by unknown call {t.get('call_id')!r}")
        for s in d.steps:
            if s.get("call_id") is not None and s["call_id"] not in call_ids:
                self.err(f"steps: step {s.get('step')} call_id {s['call_id']!r} not in calls.jsonl")
        for c in d.calls:
            if isinstance(c.get("t_start"), (int, float)) and isinstance(c.get("t_end"), (int, float)):
                if c["t_end"] < c["t_start"] - TIME_TOL:
                    self.err(f"call {c.get('call_id')}: t_end < t_start")
                if c.get("latency_sec") is not None and not math.isclose(
                    c["latency_sec"], c["t_end"] - c["t_start"], abs_tol=TIME_TOL
                ):
                    self.err(f"call {c.get('call_id')}: latency_sec != t_end - t_start")
            p, k, u = c.get("prompt_tokens"), c.get("cached_tokens"), c.get("uncached_prompt_tokens")
            if all(isinstance(x, int) for x in (p, k, u)) and u != max(p - k, 0):
                self.err(f"call {c.get('call_id')}: uncached_prompt_tokens {u} != prompt - cached ({p} - {k})")
        for t in d.tools:
            if None not in (t.get("t_start"), t.get("t_end"), t.get("duration_sec")):
                if not math.isclose(t["duration_sec"], t["t_end"] - t["t_start"], abs_tol=TIME_TOL):
                    self.err(f"tool {t.get('tool_event_id')}: duration_sec != t_end - t_start")

        seqs = Counter(r.get("seq") for r in d.calls + d.tools)
        dup = sorted(s for s, n in seqs.items() if n > 1 and s is not None)
        if dup:
            self.err(f"global seq not unique: {dup[:10]}")
        if seqs and not dup:
            vals = sorted(s for s in seqs if s is not None)
            if vals and vals != list(range(vals[0], vals[0] + len(vals))):
                self.warn(f"global seq has gaps ({vals[0]}..{vals[-1]} over {len(vals)} events)")

        chains: dict[str, list[dict]] = defaultdict(list)
        for rec in d.calls + d.tools:
            chains[rec.get("agent_id")].append(rec)
        for agent, recs in chains.items():
            seq = sorted(r.get("agent_seq") for r in recs if r.get("agent_seq") is not None)
            if len(set(seq)) != len(seq):
                self.err(f"agent {agent}: duplicate agent_seq values")
            elif seq and seq != list(range(len(seq))):  # 0-based and contiguous
                self.err(f"agent {agent}: agent_seq not contiguous ({seq[:12]}{'...' if len(seq) > 12 else ''})")
            ordered = sorted(recs, key=lambda r: (r.get("agent_seq") or 0))
            for prev, cur in zip(ordered, ordered[1:]):
                if None in (prev.get("t_end"), cur.get("t_start")):
                    continue
                if cur["t_start"] < prev["t_end"] - TIME_TOL:
                    self.err(f"agent {agent}: event agent_seq={cur.get('agent_seq')} starts "
                             f"{prev['t_end'] - cur['t_start']:.3f}s before the previous event ended")

    # -- spawns / dispatches ---------------------------------------------------------
    def check_spawns(self) -> None:
        d = self.data
        groups = {s.get("group_id") for s in d.spawns}
        by_group: dict[str, set[str]] = defaultdict(set)
        for x in d.dispatches:
            by_group[x.get("group_id")].add(x.get("dispatch_id"))
            if x.get("group_id") not in groups:
                self.ref(f"dispatch {x.get('dispatch_id')}: group {x.get('group_id')!r} not in spawns.jsonl")
        for s in d.spawns:
            ids = set(s.get("dispatch_ids") or [])
            have = by_group.get(s.get("group_id"), set())
            if ids != have:
                missing, extra = sorted(ids - have), sorted(have - ids)
                self.ref(f"spawn {s.get('group_id')}: dispatch_ids mismatch (no record: {missing}; not listed: {extra})")
            if (s.get("granted") or 0) > (s.get("requested") or 0):
                self.err(f"spawn {s.get('group_id')}: granted > requested")
        spawn_tools = {t.get("group_id") for t in d.tools if t.get("name") == "spawn_subagents"}
        for g in groups - spawn_tools:
            self.ref(f"spawn {g}: no spawn_subagents tool record carries this group_id")
        for st in d.steps:
            for g in st.get("spawned_group_ids") or []:
                if g not in groups:
                    self.ref(f"step {st.get('step')}: spawned group {g!r} not in spawns.jsonl")

    def check_arbitrations(self) -> None:
        d = self.data
        calls = {c.get("call_id"): c for c in d.calls}
        dispatch_ids = {x.get("dispatch_id") for x in d.dispatches}
        groups = {s.get("group_id") for s in d.spawns}
        for a in d.arbitrations:
            aid = a.get("arbitration_id")
            if a.get("group_id") not in groups:
                self.ref(f"arbitration {aid}: group {a.get('group_id')!r} not in spawns.jsonl")
            for did in a.get("candidate_dispatch_ids") or []:
                if did not in dispatch_ids:
                    self.ref(f"arbitration {aid}: candidate dispatch {did!r} not in dispatches.jsonl")
            for rnd in a.get("rounds") or []:
                j = rnd.get("judge_call_id")
                if j is not None and (j not in calls or calls[j].get("role") != "arbiter"):
                    self.ref(f"arbitration {aid} round {rnd.get('round')}: judge call {j!r} missing or not role arbiter")
                for dc in rnd.get("debate_call_ids") or []:
                    if dc not in calls or calls[dc].get("role") != "debate":
                        self.ref(f"arbitration {aid} round {rnd.get('round')}: debate call {dc!r} missing or not role debate")


def check_episode(episode_dir: str | Path) -> list[Problem]:
    return _Checker(Path(episode_dir)).run()


def check_run(paths: Iterable[str | Path]) -> CheckReport:
    report = CheckReport()
    for ref in discover(paths):
        if ref.episode_dir is None:
            report.n_trials_without_episode += 1
            res, _ = read_json(ref.trial_dir / "result.json") if ref.trial_dir else (None, None)
            exc = (res or {}).get("exception_info") if isinstance(res, dict) else None
            why = f"harbor exception {exc.get('exception_type')}" if isinstance(exc, dict) else "no harbor exception"
            report.problems.append(Problem("warning", str(ref.trial_dir), f"no agent/episode.json ({why})"))
            continue
        report.n_episodes += 1
        report.problems.extend(check_episode(ref.episode_dir))
    report.coverage = treatment_coverage(paths)
    for row in report.coverage:
        if row["problem"]:
            report.problems.append(Problem("warning", f"condition {row['condition']}",
                                           f"treatment not exercised: {row['problem']}"))
    return report


# ---------------------------------------------------------------------------
# treatment coverage
# ---------------------------------------------------------------------------


def treatment_coverage_rows(episodes: Iterable[dict]) -> list[dict[str, Any]]:
    """Per subagents condition: whether its treatment happened at all.

    ``episodes``: one dict per episode that has records (condition, topology,
    strategy, spawn_calls, dispatches, arbitrations). A subagents condition in
    which no orchestrator ever spawned (or, for the arbiter strategy, no
    arbitration ran) measured only its prompt text; ``problem`` says so.
    """
    by: dict[str, list[dict]] = defaultdict(list)
    for e in episodes:
        if e.get("condition") is not None:
            by[str(e["condition"])].append(e)
    out = []
    for cond in sorted(by):
        eps = by[cond]
        if next((e["topology"] for e in eps if e.get("topology")), None) != "subagents":
            continue
        strategy = next((e["strategy"] for e in eps if e.get("strategy")), None)
        n = len(eps)
        n_deleg = sum(1 for e in eps if (e.get("spawn_calls") or 0) > 0)
        n_disp = sum(int(e.get("dispatches") or 0) for e in eps)
        n_arb = sum(int(e.get("arbitrations") or 0) for e in eps)
        problems = []
        if n_deleg == 0:
            problems.append(f"no episode delegated (spawn_calls = 0 in all {n})")
        elif n_disp == 0:
            problems.append("spawn calls but no dispatch ran")
        if strategy == "arbiter" and n_arb == 0:
            problems.append("arbiter strategy but no arbitration ran")
        out.append({
            "condition": cond, "strategy": strategy, "n_episodes": n, "n_delegated": n_deleg,
            "delegated_frac": n_deleg / n if n else None, "n_dispatches": n_disp,
            "n_arbitrations": n_arb if strategy == "arbiter" else None,
            "ok": not problems, "problem": "; ".join(problems) or None,
        })
    return out


def treatment_coverage(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """:func:`treatment_coverage_rows` over every episode under ``paths``, counted from the records
    (so an episode that crashed before finalizing counts what it did)."""
    from .tables import design_fields

    rows = []
    for t in load_trials(paths):
        if t.data is None:
            continue
        d = design_fields(t.episode, t.config)
        rows.append({"condition": t.condition, "topology": d["topology"], "strategy": d["strategy"],
                     "spawn_calls": len(t.data.spawns), "dispatches": len(t.data.dispatches),
                     "arbitrations": len(t.data.arbitrations)})
    return treatment_coverage_rows(rows)


def coverage_lines(rows: list[dict[str, Any]]) -> list[str]:
    """Printable treatment-coverage lines (one per subagents condition)."""
    out = []
    for r in rows:
        arb = f", {r['n_arbitrations']} arbitrations" if r["n_arbitrations"] is not None else ""
        line = (f"treatment {r['condition']}: delegated in {r['n_delegated']}/{r['n_episodes']} episodes, "
                f"{r['n_dispatches']} dispatches{arb}")
        out.append(line + (f"  ** {r['problem']} **" if r["problem"] else ""))
    return out


# ---------------------------------------------------------------------------
# completeness
# ---------------------------------------------------------------------------

# Fields that are legitimately None/empty/zero across a whole healthy run: failure
# fields, retry/timeout counters, provider extras some providers never send, and
# fields that only apply to some topologies/strategies. Flags on these are notes.
OFTEN_EMPTY = {
    # failures, retries, timeouts, context guard
    "error", "apply_error", "backoff_sec", "n_retries", "n_command_timeouts", "n_timeouts",
    "n_tool_errors", "context_trimmed_chars", "context_trims", "n_context_trims", "cap_reason",
    "n_provider_errors", "cancel_kill", "stream_repairs", "n_stream_repairs",
    # provider extras (Sail sends no request id; Prime reports cost 0.0 and no reasoning tokens)
    "provider_reported_cost", "response_id", "reasoning_tokens", "reasoning_chars", "t_first_reasoning",
    "reasoning", "completion_window", "extra_body", "cached_tokens",
    # topology/strategy dependent
    "redispatches", "arbitration_ids", "spawned_group_ids", "queue_sec", "git_diff_stat",
    "applied_files", "debate_rounds_used", "instruction_index",
}
# Role-specific fields that are expected to be empty for that role.
ROLE_OFTEN_EMPTY = {
    "orchestrator": {"dispatch_id", "group_id", "arbitration_id"},
    "subagent": {"arbitration_id", "candidate_index"},
    # Debate calls are tool-less; their transcript messages carry no tool calls/results.
    "debate": {"n_tool_calls", "tool_call_names", "t_first_tool_call", "n_tools", "tools_hash",
               "tool_calls", "tool_results"},
    # A judge sees one user message per round and answers once: no tool results in its context.
    # Arbitrations: no leak and no redirected path is the normal case.
    "arbiter": {"dispatch_id", "tool_results", "leaked_files", "n_path_remaps"},
    "episode": {"subagent_model", "arbiter_model", "strategy", "scope", "instruction_following",
                "spawn_calls", "dispatches", "arbitrations", "requested_total", "granted_total",
                "reward", "resolved", "trial_name", "harbor_session_id", "harbor_trial_id",
                # None outside a git work tree (ZIP/tarball copy); source_sha256 identifies the code there.
                "git_commit", "git_dirty"},
}

# Fields where 0 is a real value (exit code 0, candidate/winner index 0): only
# None/empty counts as unpopulated for the never-populated flag.
ZERO_IS_VALUE = {
    "exit_code", "final_winner", "winner", "instruction_index", "candidate_index", "index", "seq",
    "agent_seq", "step", "orchestrator_step", "dispatches_remaining_after",
}

FILE_ROLE = {
    "steps.jsonl": "orchestrator",
    "spawns.jsonl": "orchestrator",
    "dispatches.jsonl": "subagent",
    "arbitrations.jsonl": "arbiter",
    "episode.json": "episode",
}


def _classify(v: Any) -> str:
    if v is None:
        return "none"
    if isinstance(v, (str, list, dict)) and len(v) == 0:
        return "empty"
    if isinstance(v, bool):
        return "zero" if v is False else "set"
    if isinstance(v, (int, float)) and v == 0:
        return "zero"
    return "set"


def completeness(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Rows: file, role, field, n_records, frac_none, frac_empty, frac_zero, frac_unpopulated, never_populated.

    A field added in a later schema version (``since``) is not counted for
    records of an episode written by older code, which cannot have it; a
    field no counted record could have gets no row."""
    counts: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    types = {}
    for ref in discover(paths):
        if ref.episode_dir is None:
            continue
        data = load_episode(ref.episode_dir, with_transcript=True)
        version = (data.episode or {}).get("schema_version")
        for fname, cls in RECORD_TYPES.items():
            recs = [data.episode] if fname == "episode.json" else getattr(data, fname.split(".")[0])
            for rec in recs:
                if not isinstance(rec, dict):
                    continue
                role = FILE_ROLE.get(fname) or str(rec.get("role") or "unknown")
                for f in fields(cls):
                    types[(fname, f.name)] = f.type
                    since = f.metadata.get("since")
                    if f.name not in rec and since is not None and isinstance(version, int) and version < since:
                        continue  # written before the field existed
                    counts[(fname, role, f.name)][_classify(rec.get(f.name))] += 1
    rows = []
    for (fname, role, name), c in sorted(counts.items()):
        n = sum(c.values())
        # schema.py uses postponed annotations, so field types are strings like "bool | None".
        is_bool = str(types.get((fname, name), "")).startswith("bool")
        unpop = c["none"] + c["empty"] + c["zero"]
        # A boolean is "populated" whenever it is not None (False is a value); likewise 0 for index-like fields.
        if is_bool or name in ZERO_IS_VALUE:
            never = c["none"] + c["empty"] == n
        else:
            never = unpop == n
        rows.append({
            "file": fname, "role": role, "field": name, "n_records": n,
            "frac_none": c["none"] / n, "frac_empty": c["empty"] / n, "frac_zero": c["zero"] / n,
            "frac_unpopulated": unpop / n, "never_populated": never,
            "often_empty": name in OFTEN_EMPTY or name in ROLE_OFTEN_EMPTY.get(role, ()),
        })
    return rows
