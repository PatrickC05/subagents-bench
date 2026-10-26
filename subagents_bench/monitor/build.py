"""Render an interactive sweep's monitor page from its jobs dir (read-only; any state of the sweep).

The sweep's design (tasks, rounds, conditions, session length, schedule) comes from
``<jobs>/sweep.json`` (the spec and the resolved config of every condition, written when the sweep
starts) or from a spec file. Controls are conditions without ``interactive.replies``; a control's
twin is the condition with the same overrides apart from ``interactive``. Two sweeps have their own
text: the requests sweep (``swe_interactive_requests``: its condition labels, header and "what
changed" table) and the levels sweep (names starting with ``swe_interactive_levels``, so its smoke
too: conditions ``<setting>_<level>``, a header and a "what each level adds" table built from the
design, a "by level" ladder per setting, and the plan of its runs: rounds 3 and up only for L2 and
L3). Other sweeps get a short header built from the design, and their conditions their name as
label (the levels names wherever they appear, e.g. the bugs-only baseline's).

Pages of several sweeps can share one out dir (``page``: the file name; ``nav``: a tab bar linking
them) and its ``transcripts/``. The host wraps only ``index.html`` in a document, so any other page
starts with its own doctype, charset and viewport.

A trial may be partial: no episode yet (skipped), a live session (no ``t_end``, no result.json, a
half-written last jsonl line), grading (ended, no result.json) or graded. Per-message metrics come
from ``analysis.requests``, per-bug pickup (delivery, first edit) from ``analysis.interactivity``;
ungraded sessions count for latency and estimates, not for fixes, and a question's verdict counts
only once ``agent/answers.json`` exists. Controls have bugs only: no ids, replies, questions or
estimates.

The comparison with an earlier sweep (``compare``, e.g. the arrival-rate sweep: the same repositories
and, per round, the same first bugs) matches bugs by subtask on the (task, round) pairs a condition
has graded; each condition is shown against the earlier conditions of the same design (overrides
apart from ``interactive``), or all of them when none has it. When both sweeps send the same bugs in
every round (``vary_order_by_round: false`` and one ``order_seed`` everywhere, as the levels sweep
and PR #13's baseline), rounds are pooled: pairs are tasks, so the levels' rounds 3-4 still meet the
baseline's two. When the earlier sweep sent more bugs per session, its first edits leave out work
that may belong to those later bugs (``_own_edits``).
"""

from __future__ import annotations

import html
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..analysis.interactivity import _median_censored, _tokens, hint_rows, tool_touches
from ..analysis.interactivity import condition_table as hint_table
from ..analysis.load import JOB_RE, RECORD_KINDS, Trial, load_trials
from ..analysis.pricing import DEFAULT_PRICING, call_cost, load_prices
from ..analysis.requests import request_rows, request_table
from ..analysis.validity import invalid_reason

HERE = Path(__file__).parent
TEMPLATE = (HERE / "template.html").read_text()  # str.format fields; literal braces doubled
VIEWER = (HERE / "transcript_viewer.html").read_text()  # inserted verbatim (not formatted)
SCHEME = "default"  # configs/pricing.yaml scheme, as analysis' default
REQUESTS_SWEEP = "swe_interactive_requests"  # the sweep the "what changed" table describes
LEVELS_SWEEP = "swe_interactive_levels"  # the levels sweep; its smoke and other variants start with this name
LEVELS_ALL_ROUNDS = 2  # swe_interactive_levels.yaml's commands: rounds 1-2 run every level, later ones L2 and L3
SETTINGS = {  # an orchestration: (label, description); the levels sweep's conditions are <setting>_<level suffix>
    "single": ("Single agent", "No subagents. It reads, edits, tests and replies itself; a message waits until its "
               "current step (thinking and commands) ends."),
    "manager_background": ("Background manager", "Read-only orchestrator (read, list, search, reply): every edit and "
                           "test goes to subagents that run in the background and report as messages."),
    "background_hinted": ("All tools + background, hinted", "All tools and background subagents, and told that "
                          "delegating keeps it free to answer the user."),
    "manager_available": ("Manager, stay available", "The background manager, told to keep each step short and hand "
                          "each bug at once to a subagent with a short instruction (the subagent works out the problem)."),
    "manager_owner": ("Manager, owner + helpers", "Stay available, and each bug's subagent is in charge of it and may "
                      "start up to 10 helper subagents of its own."),
    "dispatcher_hinted": ("Dispatcher, no tools, hinted", "No tools of its own (not even reading): only spawn_subagents, "
                          "wait and reply, and told that delegating keeps it free to answer the user."),
}
LEVELS = {  # condition suffix -> (level, short name, what the user sends), each level the one above plus one kind
    "baseline": ("L0", "bugs only", "Bugs only, with the reply tool: PR #13's baseline."),
    "eta": ("L1", "status checks", "Bugs, and status checks asking how long each bug so far will take."),
    "questions": ("L2", "questions", "Bugs, status checks, and questions about the code after each bug."),
    "urgent": ("L3", "urgent bug", "Bugs, status checks and questions, and one bug is urgent."),
}
LABELS = {  # condition -> (card label, description)
    "single_requests": SETTINGS["single"],
    "manager_background_requests": SETTINGS["manager_background"],
    "background_hinted_requests": SETTINGS["background_hinted"],
    "single_bugsonly": ("Single agent, bugs only", "Control: the same bugs at the same times, nothing else. No "
                        "ids, no reply tool, no questions, status checks or urgent bug."),
    "manager_background_bugsonly": ("Background manager, bugs only", "Control: the background manager with the "
                                    "same bugs at the same times and none of the requests."),
    **{f"{s}_{x}": (f"{sl} · {lv} {ln}", f"{sd} {ld}") for s, (sl, sd) in SETTINGS.items()
       for x, (lv, ln, ld) in LEVELS.items()},
}
SWEEP_LABEL = {"swe_interactive_rate": "arrival-rate",  # an earlier sweep, as the comparison names it
               "swe_interactive_baseline_modal": "PR #13 baseline"}
SHORT = {"manager_background": "manager"}  # an earlier design's short name (its conditions minus _every<s>)
REQUESTS_LEDE = (  # the requests sweep's header (str.format fields)
    "A simulated user works with the agent on one repository for {minutes} minutes. A bug arrives every {interval} "
    "minutes ({n_bugs} in all), two questions about the code come between bugs, a status check asks how long the open "
    "bugs will take, and bug 3 is urgent. The user sees only what the agent sends with <code>reply</code>, so this "
    "measures how fast and how well questions are answered, whether bugs get fixed, and whether the agent's estimates "
    "of its own work hold. Two bugs-only controls get the same bugs at the same times and nothing else, so the gap to "
    "them is what the requests cost. Orchestrator: DeepSeek-V4.1-Flash. Subagents: Gemma-4-31B, 20 steps each, up to "
    "3 at once.")
CONTROLS_NOTE = (" Controls (dashed cards) have no ids or replies: their bugs are numbered by position and show only "
                 "pickup (waited to be read, first edit after) and fixes.")
PICKUP_NOTE = " {what}: per-bug pickup from <code>interactivity.csv</code>, over the bugs after the first."
LEVELS_NOTE = ("<p>By level: each setting's levels, one row each, so a step down adds one kind of message; a cell is "
               "\"–\" when the level sends no such message (or nothing has been measured yet), and a median is "
               "\"never\" when at least half of its messages were never read, replied to or edited (not yet, in "
               "a live session; they count as infinitely late). Sessions: started, of those planned. Status check "
               "replied: the checks with any reply after them, and the median time to the first. Bugs fixed is the "
               "end-of-session grade; each bug's real fix time comes from the workspace snapshots after the run "
               "(<code>subagents-bench fix-times</code>) and is not on this page.</p>")
DOC_HEAD = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'  # a page other than index.html: the host
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n')  # wraps only index.html
KIND_LABEL = {"bug": "Bug", "question": "Question", "status": "Status check"}
esc = html.escape


# --- the sweep's design -------------------------------------------------------------------------

def level_of(cond: str | None) -> tuple[str, int] | None:
    """A levels condition's (setting, level number): ``manager_background_questions`` ->
    ("manager_background", 2); None for any other name."""
    setting, _, suffix = (cond or "").rpartition("_")
    return (setting, list(LEVELS).index(suffix)) if setting and suffix in LEVELS else None


def level_key(cond: str) -> tuple:
    """Sorts levels conditions by setting (``SETTINGS`` order), then level; other names last."""
    lv = level_of(cond)
    return (list(SETTINGS).index(lv[0]) if lv[0] in SETTINGS else len(SETTINGS), lv[0], lv[1]) if lv else (99, "", 0)


def planned_rounds(name: str, conditions, rounds: int) -> dict[str, tuple[int, ...]]:
    """Per condition, the rounds it will run in: all of them, except in the levels sweep, whose rounds
    after ``LEVELS_ALL_ROUNDS`` run only L2 and L3 (the urgency comparison needs more pairs)."""
    def last(c):
        lv = level_of(c)
        return min(rounds, LEVELS_ALL_ROUNDS) if name.startswith(LEVELS_SWEEP) and lv and lv[1] < 2 else rounds
    return {c: tuple(range(1, last(c) + 1)) for c in conditions}


@dataclass
class Sweep:
    name: str
    tasks: tuple[str, ...]
    rounds: int
    conditions: dict[str, dict]  # condition -> its spec overrides, in spec order
    interactive: dict[str, dict]  # condition -> resolved ``interactive`` config (interactive conditions only)
    configs: dict[str, dict] = field(default_factory=dict)  # condition -> its whole resolved config (models)
    raw: dict = field(default_factory=dict)  # the spec after ``extends`` (``harbor.environment``)
    plan: dict[str, tuple[int, ...]] | None = None  # condition -> the rounds it runs in (``planned_rounds``)

    def __post_init__(self) -> None:
        if self.plan is None:
            self.plan = planned_rounds(self.name, self.conditions, self.rounds)

    @property
    def levels(self) -> bool:
        """The levels sweep or a variant of it (its smoke)."""
        return self.name.startswith(LEVELS_SWEEP)

    def sends(self, cond: str, key: str) -> bool:
        """Whether a condition's schedule has this kind of message (``questions_per_interval``,
        ``status_every``, ``urgent_every``)."""
        return bool((self.interactive.get(cond) or {}).get(key))

    @property
    def icfg(self) -> dict:
        """The schedule shown in the header: the first condition with replies, else the first."""
        withr = [c for c in self.interactive if self.interactive[c].get("replies")]
        return self.interactive[(withr or list(self.interactive))[0]]

    @property
    def wall(self) -> float:
        return float(self.icfg["wall_clock_sec"])

    @property
    def interval(self) -> float:
        return float(self.icfg["interval_sec"])

    @property
    def n_bugs(self) -> int:
        """The bugs a session is sent: ``max_hints``, or as many as fit in it (``hints.n_slots``: one
        every ``interval_sec`` below the wall clock), e.g. 4 of 6 in the levels smoke's 2 minutes."""
        return min(int(self.icfg["max_hints"]), max(1, math.ceil(self.wall / self.interval - 1e-9)))

    @property
    def n_sessions(self) -> int:
        """The sessions the sweep will run: per condition, a session per task in each planned round."""
        return len(self.tasks) * sum(len(self.plan.get(c, ())) for c in self.conditions)

    def control(self, cond: str) -> bool:
        return not (self.interactive.get(cond) or {}).get("replies")

    def twins(self) -> dict[str, str]:
        """control -> the condition with replies and the same design."""
        return {c: r for c in self.conditions if self.control(c)
                for r in [next((r for r in self.conditions if not self.control(r)
                                and design(self.conditions[r]) == design(self.conditions[c])), None)] if r}


def design(over: dict | None) -> dict:
    """A condition's overrides apart from ``interactive``: what an orchestration is."""
    return {k: v for k, v in (over or {}).items() if k != "interactive"}


def load_sweep(jobs: Path | None = None, spec: Path | None = None) -> Sweep:
    """From a spec file when given, else ``<jobs>/sweep.json``; FileNotFoundError without either,
    ValueError for a malformed sweep.json or a sweep with no interactive condition (its message names
    no path: the page shows it)."""
    if spec is not None:
        from ..sweep import load_spec
        s = load_spec(spec)
        raw, configs = s.raw, {c: cfg.to_dict() for c, cfg in s.configs.items()}
    else:
        p = Path(jobs) / "sweep.json"
        if not p.is_file():
            raise FileNotFoundError(f"{p} not found: pass the sweep's spec (--spec)")
        try:
            doc = json.loads(p.read_text())
            raw, configs = doc["spec"], doc.get("configs") or {}
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise ValueError(f"sweep.json is malformed ({type(e).__name__}: {e})") from e
    try:
        inter = {c: (configs.get(c) or {}).get("interactive") for c in raw["conditions"]}
        inter = {c: v for c, v in inter.items() if v}
        if not inter:
            raise ValueError(f"sweep {raw['name']} has no interactive condition")
        return Sweep(raw["name"], tuple(raw["tasks"]), int(raw["rounds"]),
                     {c: v or {} for c, v in raw["conditions"].items()}, inter,
                     {c: v for c, v in configs.items() if isinstance(v, dict)}, raw)
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"sweep.json is malformed: its spec has no valid {e}") from e


# --- formatting ---------------------------------------------------------------------------------

def isnum(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v)


def clock(sec) -> str:
    if not isnum(sec) or math.isinf(sec):
        return "–"
    sec = max(0, int(round(sec)))
    return f"{sec} s" if sec < 100 else f"{sec // 60}:{sec % 60:02d}"


def censored(sec, n) -> str:
    """A censored median over ``n`` values (``_median_censored``: NaN when it is infinite, i.e. at
    least half were never answered or edited): "never" when there are values but no finite median,
    else ``clock``."""
    return "never" if n and not (isnum(sec) and math.isfinite(sec)) else clock(sec)


def pct(v) -> str:
    return f"{100 * v:.0f}%" if isnum(v) and math.isfinite(v) else "–"


def log2err(v, signed: bool = False) -> str:
    """An ETA's log2 error; +inf is a bug estimated and never said done."""
    if not isnum(v):
        return "–"
    if math.isinf(v):
        return "never done"
    return f"{v:+.1f}" if signed else f"{v:.2f}"


def mins(m) -> str:
    return f"{m:g} min" if isnum(m) else "?"


def trunc(t, n: int) -> str:
    t = " ".join(str(t or "").split())
    return t if len(t) <= n else t[:n - 1] + "…"


def _m(sec: float) -> str:
    return f"{int(sec) // 60}:{int(sec) % 60:02d}"


def _or(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " or " + xs[-1]


def _and(xs: list[str]) -> str:
    return "" if not xs else xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def dur(sec: float) -> str:
    """A length in whole minutes ("4 min"), else in seconds ("30 s")."""
    return f"{sec / 60:g} min" if sec % 60 == 0 else f"{sec:g} s"


def _rounds(rs: tuple[int, ...]) -> str:
    return "no round" if not rs else f"round {rs[0]}" if len(rs) == 1 else f"rounds {rs[0]}–{rs[-1]}"


def _nth(k: int) -> str:
    return f"{k}{'th' if 10 <= k % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th')}"


def schedule(icfg: dict) -> dict:
    """A condition's messages as ``hints.schedule`` sends them (seconds into the session): the bugs
    (``max_hints``, every ``interval_sec``, while the session lasts; a task with fewer bugs sends
    fewer), the status checks (halfway after every ``status_every``-th bug), the questions' offsets
    after each bug and how many fit, and the urgent bugs (every ``urgent_every``-th, 0-based, not the
    first)."""
    iv, wall = float(icfg.get("interval_sec") or 0), float(icfg.get("wall_clock_sec") or 0)
    b = max(1, int(icfg.get("bundle") or 1))  # bugs per message: the times are the messages'
    bugs = [k * iv for k in range(-(-int(icfg.get("max_hints") or 0) // b)) if k * iv < wall] or [0.0]
    on = bool(icfg.get("replies"))
    se, q, u = (int(icfg.get(k) or 0) * on for k in ("status_every", "questions_per_interval", "urgent_every"))
    q_at = [j * iv / (q + 1) for j in range(1, q + 1)]
    return {"iv": iv, "bugs": bugs, "q": q, "q_at": q_at, "n_q": sum(b + t < wall for b in bugs for t in q_at),
            "checks": [(k + 0.5) * iv for k in range(len(bugs)) if se and (k + 1) % se == 0 and (k + 0.5) * iv < wall],
            "urgent": [k * iv for k in range(1, len(bugs)) if u and k % u == 0]}


def _median_inf(xs):
    """Median with missing as infinitely late (None when that is infinite)."""
    if not xs:
        return None
    v = sorted(math.inf if x is None or not isnum(x) else x for x in xs)
    n = len(v)
    m = v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
    return m if math.isfinite(m) else None


def _num(x, key):
    v = x.get(key) if x is not None else None
    return v if isnum(v) else None


# --- loading (tolerant of a live session's half-written files) ----------------------------------

def _json(p: Path):
    try:
        return json.loads(p.read_text()) if p.is_file() else None
    except (OSError, ValueError):
        return None


def _jsonl(p: Path) -> list[dict]:
    try:
        lines = p.read_text().splitlines() if p.is_file() else []
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line)) if line.strip() else None
        except ValueError:
            pass  # the line being written
    return out


def load(trial_dir: Path, jobs: Path) -> Trial:
    """``analysis.load.load_trial`` that skips a half-written jsonl line."""
    m = JOB_RE.match(trial_dir.parent.name)
    ep = _json(trial_dir / "agent" / "episode.json")
    t = Trial(dir=trial_dir, sweep=m["sweep"] if m else jobs.name, round=int(m["round"]) if m else None,
              condition=m["condition"] if m else (ep or {}).get("condition"), result=_json(trial_dir / "result.json"),
              episode=ep, records={k: _jsonl(trial_dir / "agent" / f"{k}.jsonl") for k in RECORD_KINDS})
    t.n_subtasks = ((_json(jobs / "tasks.json") or {}).get("n_subtasks") or {}).get(t.task_name)
    return t


def trial_dirs(jobs: Path) -> list[Path]:
    """Trials with an episode, in job then trial order (``_``/``.`` dirs skipped)."""
    if not jobs.is_dir():
        return []
    return [tr for job in sorted(jobs.iterdir())
            if job.is_dir() and "__r" in job.name and not job.name.startswith(("_", "."))
            for tr in sorted(job.iterdir()) if tr.is_dir() and (tr / "agent" / "episode.json").is_file()]


def rate_rows(jobs: Path, n_bugs: int) -> list[dict]:
    """An earlier (finished) sweep's per-bug rows, valid graded trials only, first edits by ``_own_edits``."""
    out = []
    for t in load_trials(jobs):
        if t.episode is None or invalid_reason(t) is not None:
            continue
        keys = {"condition": t.condition, "task": t.task_name.replace("swe-sweep__", ""), "round": t.round,
                "trial_name": f"{t.dir.parent.name}/{t.dir.name}"}
        out += _own_edits(t, hint_rows(t, keys, True), n_bugs)
    return out


def _own_edits(t: Trial, rows: list[dict], n_bugs: int) -> list[dict]:
    """The first edit of each of the earlier sweep's first ``n_bugs`` bugs, leaving out edits that may
    belong to its later bugs (``n_bugs`` + 1 on, which this sweep never sends): an edit made once such
    a bug had arrived, to files of this bug that are all also files of an arrived later bug. Rescans
    the tools as hint_rows does (same file matching); ``edit_sec_any`` keeps hint_rows' value. A no-op
    when the earlier sweep sent no more bugs than this one."""
    hints = sorted((x for x in t.records["inbox"] if x["kind"] == "hint"), key=lambda x: x["hint_index"])
    all_files = [f for h in hints for f in dict.fromkeys(h["target_files"] or [])]
    n_base: dict[str, int] = {}
    for f in dict.fromkeys(all_files):
        n_base[_tokens(f)[1]] = n_base.get(_tokens(f)[1], 0) + 1
    basename_ok = {f for f in all_files if n_base[_tokens(f)[1]] == 1}
    later = [(h["t_ready"], set(h["target_files"] or [])) for h in hints if h["hint_index"] >= n_bugs]
    tools = sorted(t.records["tools"], key=lambda x: x["t_start"])
    by_idx = {h["hint_index"]: h for h in hints}
    for r in rows:
        r["edit_sec_any"] = r["edit_sec"]
        h = by_idx.get(r["hint_index"])
        if r["hint_index"] >= n_bugs or h is None:
            continue
        t0, files, first = h["t_ready"], h["target_files"] or [], None
        for x in tools:
            if x["t_start"] < t0:
                continue
            edited = [f for f in files if tool_touches(x, [f], basename_ok)[1]]
            if not edited:
                continue
            theirs = set().union(*(fs for tr, fs in later if tr <= x["t_start"]))
            if all(f in theirs for f in edited):
                continue  # may be work on a later bug that shares the file
            first = x["t_start"] - t0
            break
        r["edit_sec"] = first
    return rows


# --- the page -----------------------------------------------------------------------------------

class Monitor:
    """One rendering: ``sweep`` (this run), ``tx`` (trial dir -> its exported transcript entry),
    ``compare``: (earlier jobs dir, its Sweep, or why it has none: load_sweep's error) or None;
    ``page`` (its file name) and ``nav`` ([(label, href)], the tab bar; none without)."""

    def __init__(self, jobs: Path, sweep: Sweep, compare: tuple[Path, Sweep | str] | None = None,
                 tx: dict[str, dict] | None = None, prices: dict | None = None, page: str = "index.html",
                 nav: list[tuple[str, str]] | None = None):
        self.jobs, self.sw, self.tx = Path(jobs), sweep, tx or {}
        self.page, self.nav = page, list(nav or [])
        self.prices = prices if prices is not None else load_prices(DEFAULT_PRICING, SCHEME)
        conds = sorted(sweep.conditions, key=level_key) if sweep.levels else list(sweep.conditions)
        self.cond = {c: LABELS.get(c, (c.replace("_", " "), "")) for c in conds}  # in page order
        self.twin = sweep.twins()
        self.compare, self.pooled = compare, False
        if compare and isinstance(compare[1], Sweep):
            self._compare_design(compare[1])

    def label(self, cond: str | None) -> tuple[str, str]:
        return self.cond.get(cond or "", LABELS.get(cond or "", (cond or "?", "")))

    def group_label(self, cond: str) -> str:
        """What a condition's design is called: its setting in the levels sweep, else its label."""
        lv = level_of(cond) if self.sw.levels else None
        return SETTINGS.get(lv[0], (lv[0].replace("_", " "),))[0] if lv else self.label(cond)[0]

    def col_label(self, cond: str) -> str:
        """A condition of this run as a column of the comparison grid: its level, else "with requests"."""
        lv = level_of(cond) if self.sw.levels else None
        return list(LEVELS.values())[lv[1]][0] if lv else "with requests"

    # --- sessions --------------------------------------------------------------------------------

    def session(self, trial_dir: Path) -> dict | None:
        t = load(trial_dir, self.jobs)
        ep, rec = t.episode, t.records
        if ep is None or not rec["inbox"]:
            return None
        graded = t.result is not None
        try:  # graded and valid: the comparison with the earlier sweep counts only these
            why = invalid_reason(t) if graded else None
        except (OSError, KeyError, TypeError, ValueError):
            why = "unreadable"
        valid = graded and why is None
        state = "graded" if graded else ("grading" if ep.get("exit_reason") else "live")
        if ep.get("exit_reason") in ("provider_error", "billing_error"):
            state, why = "invalid", ep.get("exit_reason")
        elif why:  # graded, but invalid (e.g. a provider error it survived): excluded, and its card says so
            state = "invalid"
        icfg = (ep.get("config") or {}).get("interactive") or {}
        control = not icfg.get("replies")
        t0 = ep.get("session_t0") or ep.get("t_start") or min(x["t_ready"] for x in rec["inbox"])
        wall = float(icfg.get("wall_clock_sec") or self.sw.wall)
        has_answers = (trial_dir / "agent" / "answers.json").is_file()
        keys = {"condition": t.condition, "trial_name": f"{trial_dir.parent.name}/{trial_dir.name}"}
        try:
            hrows = hint_rows(t, keys, state != "invalid")
        except (KeyError, TypeError, ValueError):  # a record cut mid-write
            hrows = []
        for h in hrows:  # live/grading: no verdict yet
            if not graded:
                h["resolved"] = h["counted"] = h["partial"] = None
                h["graded"] = False
        rows = [] if control else request_rows(t, keys, state != "invalid")
        for r in rows:  # live/grading: no verdict yet; a question counts only once answers.json exists
            if not graded and r["kind"] == "bug":
                r["fixed"] = r["claim_ok"] = None
            if not has_answers and r["kind"] == "question":
                r["correct"] = None
        edit_of = {h["subtask_id"]: h.get("edit_sec") for h in hrows}
        for r in rows:
            if r["kind"] == "bug":
                r["edit_sec"] = edit_of.get(r.get("subtask_id"))
        if control:  # bugs only: the same row shape, numbered by position (the user sent no ids)
            rows = [{"msg_id": h["hint_index"] + 1, "kind": "bug", "index": h["hint_index"], "urgent": False,
                     "t_ready_sec": h["t_ready_sec"], "delivery_sec": h["delivery_sec"], "reply_sec": None,
                     "subtask_id": h["subtask_id"], "edit_sec": h.get("edit_sec"), "fixed": h["counted"],
                     "done_sec": None, "eta_sec": None, "eta_min": None, "on_time": None, "eta_log2_err": None,
                     "urgent_first": None, "claim_ok": None} for h in hrows]
        elapsed = min(wall, (ep.get("t_end") or time.time()) - t0)
        last_msg: Any = ""
        tr = trial_dir / "agent" / "transcript.jsonl"
        try:
            lines = tr.read_text().splitlines() if tr.is_file() else []
        except OSError:
            lines = []
        for line in reversed(lines):
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if m.get("role") == "orchestrator" and m.get("msg_role") == "assistant":
                text, kind = (m.get("content") or "").strip(), "message"
                if not text and (m.get("reasoning") or "").strip():
                    text, kind = m["reasoning"].strip()[-700:], "reasoning"
                if text:
                    last_msg = (m["t"] - t0, text, kind)
                    break
        otools = [x for x in rec["tools"] if x.get("role") == "orchestrator"]
        arg = lambda x: (x.get("arguments") or {})  # noqa: E731
        recent = [(x["t_start"] - t0, x["name"],
                   arg(x).get("command") or arg(x).get("path") or arg(x).get("text") or "")
                  for x in otools if x["name"] != "wait"][-4:]
        costs = [call_cost(c, self.prices) for c in rec["calls"]]
        return {
            "dir": trial_dir, "round": t.round, "valid": valid and state == "graded", "control": control,
            "hrows": hrows, "condition": t.condition, "task": t.task_name.replace("swe-sweep__", ""),
            "state": state, "graded": graded, "exit": ep.get("exit_reason"), "why": why, "wall": wall, "t0": t0,
            "elapsed": elapsed, "rows": rows, "inbox": [x for x in rec["inbox"] if x.get("msg_id") is not None],
            "replies": [] if control else sorted(rec["replies"], key=lambda x: x["t"]), "has_answers": has_answers,
            "cost": sum(c for c in costs if not math.isnan(c)), "dispatches": rec["dispatches"],
            "calls": [c for c in rec["calls"] if c.get("role") == "orchestrator"], "tools": otools,
            "last_msg": last_msg, "recent": recent,
        }

    def sessions(self) -> list[dict]:
        return [s for s in map(self.session, trial_dirs(self.jobs)) if s]

    def questions_ready(self) -> list[tuple[str, int | None]]:
        """Per task, the questions in ``<jobs>/hints/<task>.questions.json`` (None: not written yet)."""
        out = []
        for task in self.sw.tasks:
            d = _json(self.jobs / "hints" / f"{task}.questions.json")
            qs = d.get("questions") if isinstance(d, dict) else None
            out.append((task.replace("swe-sweep__", ""), len(qs) if isinstance(qs, list) else None))
        return out

    # --- timeline --------------------------------------------------------------------------------

    def timeline(self, s) -> str:
        W, L, R = 960, 92, 24
        plot, wall, t0 = W - L - R, s["wall"], s["t0"]
        x = lambda sec: L + plot * max(0.0, min(sec, wall)) / wall  # noqa: E731
        disp = sorted(s["dispatches"], key=lambda d: d["t_start"])
        lanes_end: list[float] = []
        lane_of = []
        for d in disp:
            for i, e in enumerate(lanes_end):
                if d["t_start"] >= e:
                    lanes_end[i] = d["t_end"]
                    lane_of.append(i)
                    break
            else:
                lanes_end.append(d["t_end"])
                lane_of.append(len(lanes_end) - 1)
        sent = sent_rows(s)
        bugs = [r for r in sent if r["kind"] == "bug"]
        ctl = s["control"]  # bugs only: no reply lane, no estimates
        Y = {"user": 34, "reply": 74, "orch": 74 if ctl else 112}
        eta_y0 = Y["orch"] + 30
        sub_y0 = (Y["orch"] + 24) if ctl else eta_y0 + 14 * max(1, len(bugs)) + 14
        H = sub_y0 + (22 * len(lanes_end) if disp else 0) + 30
        p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Session timeline" class="tl">']
        step = self.sw.interval if self.sw.interval >= 60 else 60  # a tick per bug interval, and one at the end
        ticks = [k * step for k in range(int(wall // step) + 1)]
        if wall - ticks[-1] > step / 4:
            ticks.append(wall)
        elif ticks[-1] < wall:
            ticks[-1] = wall
        for k in ticks:
            p.append(f'<line x1="{x(k):.1f}" x2="{x(k):.1f}" y1="14" y2="{H - 22}" class="grid"/>'
                     f'<text x="{x(k):.1f}" y="{H - 6}" class="tick" text-anchor="middle">{int(k) // 60}:'
                     f'{int(k) % 60:02d}</text>')
        labels = [("User", Y["user"]), ("Orchestrator", Y["orch"])]
        if not ctl:
            labels += [("Replies", Y["reply"]), ("Estimates", eta_y0 + 5)]
        if disp:
            labels.append(("Subagents", sub_y0 + 11))
        for lab, y in labels:
            p.append(f'<text x="4" y="{y + 4}" class="lane">{lab}</text>')
        for c in s["calls"]:  # orchestrator
            a, b = c["t_start"] - t0, c["t_end"] - t0
            p.append(f'<rect x="{x(a):.1f}" y="{Y["orch"] - 9}" width="{max(1.2, x(b) - x(a)):.1f}" height="18" '
                     f'class="think"><title>thinking {clock(b - a)}</title></rect>')
        for tl in s["tools"]:
            a, b = tl["t_start"] - t0, tl["t_end"] - t0
            if tl["name"] == "reply":
                continue  # on the Replies lane
            w = tl["name"] == "wait"
            p.append(f'<rect x="{x(a):.1f}" y="{Y["orch"] - (4 if w else 9)}" width="{max(1.2, x(b) - x(a)):.1f}" '
                     f'height="{8 if w else 18}" class="{"waitbar" if w else "tool"}"><title>{esc(tl["name"])} '
                     f'{clock(b - a)}</title></rect>')
        for d, lane in zip(disp, lane_of):  # subagents
            a, b = d["t_start"] - t0, d["t_end"] - t0
            p.append(f'<rect x="{x(a):.1f}" y="{sub_y0 + 22 * lane}" width="{max(1.5, x(b) - x(a)):.1f}" height="16" '
                     f'rx="2" class="sub sub-{esc(str(d.get("status")))}"><title>{esc(str(d.get("agent_id")))}: '
                     f'{esc(str(d.get("status")))}, {d.get("steps_used", "?")} steps, {clock(b - a)}</title></rect>')
        first = {}  # replies, and the line from each question to its first reply
        for rp in s["replies"]:
            first.setdefault(rp["to"], rp)
        for r in sent:
            if r["kind"] == "question" and r["msg_id"] in first:
                xa, xb = x(r["t_ready_sec"]), x(first[r["msg_id"]]["t"] - t0)
                p.append(f'<line x1="{xa:.1f}" y1="{Y["user"] + 7}" '
                         f'x2="{xb:.1f}" y2="{Y["reply"] - 9}" class="qlink"/>')
        for rp in s["replies"]:
            xr = x(rp["t"] - t0)
            kind = rp.get("kind") or "bug"
            extra = f", eta {mins(rp['eta_min'])}" if rp.get("eta_min") is not None else ""
            extra += ", done" if rp.get("done") else ""
            p.append(f'<rect x="{xr - 1.2:.1f}" y="{Y["reply"] - 8}" width="2.4" height="16" class="r-{esc(kind)}">'
                     f'<title>{clock(rp["t"] - t0)} re #{rp["to"]}{extra}: {esc(trunc(rp.get("text"), 200))}'
                     f'</title></rect>')
            if rp.get("done"):
                p.append(f'<circle cx="{xr:.1f}" cy="{Y["reply"] - 11}" r="2.6" class="r-{esc(kind)}"/>')
        order = {"question": 0, "status": 1, "bug": 2}  # user messages, the bugs on top
        for r in sorted(sent, key=lambda r: order[r["kind"]]):
            xa, y, mid = x(r["t_ready_sec"]), Y["user"], r["msg_id"]
            tip = f'#{mid} {KIND_LABEL[r["kind"]].lower()} at {clock(r["t_ready_sec"])}'
            if r["kind"] == "question":
                cls = "q" if r["reply_sec"] is not None else "q open"
                p.append(f'<path d="M{xa:.1f} {y - 6} l6 6 l-6 6 l-6 -6 z" class="{cls}"><title>{tip}: '
                         f'{esc(trunc(r.get("question"), 200))}</title></path>')
            elif r["kind"] == "status":
                cls = "st" if r["reply_sec"] is not None else "st open"
                p.append(f'<rect x="{xa - 5:.1f}" y="{y - 5}" width="10" height="10" class="{cls}"><title>{tip}'
                         f'</title></rect>')
            else:
                if r["delivery_sec"] is not None and r["delivery_sec"] > 1:
                    p.append(f'<line x1="{xa:.1f}" x2="{x(r["t_ready_sec"] + r["delivery_sec"]):.1f}" y1="{y}" '
                             f'y2="{y}" class="waitline"/>')
                cls = "bug urgent" if r["urgent"] else "bug"
                p.append(f'<circle cx="{xa:.1f}" cy="{y}" r="8.5" class="{cls}"><title>{tip}'
                         f'{" (urgent)" if r["urgent"] else ""}</title></circle>'
                         f'<text x="{xa:.1f}" y="{y + 3.5}" class="bugn" text-anchor="middle">{mid}</text>')
                if r["urgent"]:
                    p.append(f'<text x="{xa:.1f}" y="{y - 12}" class="urgentlab" text-anchor="middle">urgent</text>')
        # estimates: per bug, from its first estimate to the predicted done time; a mark at the done claim
        by_msg: dict[int, list] = {}
        for rp in s["replies"]:
            by_msg.setdefault(rp["to"], []).append(rp)
        for i, r in enumerate([] if ctl else bugs):
            y = eta_y0 + 14 * i
            t_bug = r["t_ready_sec"]
            p.append(f'<text x="{L - 6}" y="{y + 3.5}" class="tick" text-anchor="end">#{r["msg_id"]}</text>')
            ests = [rp for rp in by_msg.get(r["msg_id"], []) if rp.get("eta_min") is not None]
            if ests:
                a = ests[0]["t"] - t0
                b = a + 60 * ests[0]["eta_min"]
                never = isnum(r.get("eta_log2_err")) and math.isinf(r["eta_log2_err"])
                p.append(f'<line x1="{x(t_bug):.1f}" x2="{x(a):.1f}" y1="{y}" y2="{y}" class="etawait"/>'
                         f'<rect x="{x(a):.1f}" y="{y - 2.5}" width="{max(1.5, x(b) - x(a)):.1f}" height="5" rx="2.5" '
                         f'class="eta{" late" if r.get("on_time") is False else ""}"><title>#{r["msg_id"]}: estimate '
                         f'{mins(ests[0]["eta_min"])} at {clock(a)}, due {clock(b)}'
                         f'{"; never said done" if never else ""}</title></rect>')
                if never and b < wall:  # missed and never done: a dotted tail to the session's end
                    p.append(f'<line x1="{x(b):.1f}" x2="{x(min(wall, s["elapsed"])):.1f}" y1="{y}" y2="{y}" '
                             f'class="etanever"/>')
                if b > wall:
                    p.append(f'<text x="{x(wall) + 3:.1f}" y="{y + 3.5}" class="tick">›</text>')
                for u in ests[1:]:  # later estimates: where each one put the done time
                    ub = u["t"] - t0 + 60 * u["eta_min"]
                    p.append(f'<circle cx="{x(ub):.1f}" cy="{y}" r="2.6" class="etaupd"><title>update at '
                             f'{clock(u["t"] - t0)}: {mins(u["eta_min"])}</title></circle>')
            if r.get("done_sec") is not None:
                xd = x(t_bug + r["done_sec"])
                cls = {True: "ok", False: "bad"}.get(r.get("claim_ok"), "unk")
                verdict = {True: "fixed", False: "not fixed"}.get(r.get("claim_ok"), "not graded yet")
                p.append(f'<path d="M{xd - 4:.1f} {y} l3 4 l6 -8" class="done {cls}"><title>#{r["msg_id"]} said '
                         f'done at {clock(t_bug + r["done_sec"])}: {verdict}</title></path>')
        if s["state"] == "live":
            xn = x(s["elapsed"])
            p.append(f'<line x1="{xn:.1f}" x2="{xn:.1f}" y1="14" y2="{H - 22}" class="now"/>')
        p.append("</svg>")
        return "".join(p)

    # --- cards and tables ------------------------------------------------------------------------

    def session_card(self, s) -> str:
        label, _ = self.label(s["condition"])
        st = stats(s)
        trs = []
        for r in sent_rows(s):
            if s["control"]:
                trs.append(f'<tr class="k-bug"><td class="num">{r["msg_id"]}</td>'
                           f'<td class="num">{clock(r["t_ready_sec"])}</td><td class="num">{clock(r["delivery_sec"])}'
                           f'</td><td class="num">{clock(r.get("edit_sec"))}</td><td>{message_cell(r)}</td>'
                           f'<td>{verdict_cell(r, s)}</td></tr>')
                continue
            kind = KIND_LABEL[r["kind"]] + (' <span class="tag">urgent</span>' if r["urgent"] else "")
            trs.append(f'<tr class="k-{r["kind"]}"><td class="num">#{r["msg_id"]}</td><td>{kind}</td>'
                       f'<td class="num">{clock(r["t_ready_sec"])}</td><td class="num">{clock(r["delivery_sec"])}</td>'
                       f'<td class="num">{clock(r["reply_sec"])}</td>'
                       f'<td>{message_cell(r, s if self.sw.levels else None)}</td>'
                       f'<td>{reply_cell(r, s)}</td>'
                       f'<td>{verdict_cell(r, s)}</td></tr>')
        if s["control"]:
            head = ('<th>Bug</th><th>Sent</th><th>Waited to be read</th><th>First edit after</th><th>Subtask</th>'
                    '<th>Verdict</th>')
            meta = (f'bugs only · edited {st["edited"]}/{st["bugs"]}, first edit median {clock(st["edit_med"])} · '
                    f'{len(s["dispatches"])} subagent runs · ${s["cost"]:.3f}')
            tclass = "msgs ctl"
        else:
            head = ('<th>Id</th><th>Kind</th><th>Sent</th><th>Waited to be read</th><th>Replied after</th>'
                    '<th>Message</th><th>Reply: answer or estimate</th><th>Verdict</th>')
            asked = (f'answered {st["answered"]}/{st["q"]} questions, median {clock(st["answer_med"])} · '
                     if st["q"] or self.sw.sends(s["condition"], "questions_per_interval") else "")
            meta = (f'{asked}estimates '
                    f'{st["eta"]}/{st["bugs"]} · said done {st["done"]}/{st["bugs"]} · {len(s["dispatches"])} '
                    f'subagent runs · ${s["cost"]:.3f}')
            tclass = "msgs"
        msg = ""
        if s["last_msg"]:
            at, text, kind = s["last_msg"]
            text = text if len(text) < 700 else text[:700] + " …"
            if kind == "reasoning" and not text.startswith("…"):
                text = "… " + text
            msg = (f'<div class="said"><div class="eyebrow">Orchestrator\'s latest {kind} · {clock(at)}</div>'
                   f'<p>{esc(text)}</p></div>')
        recent = "".join(f'<li><span class="num">{clock(a)}</span> <b>{esc(n)}</b> '
                         f'<code>{esc(str(c)[:160])}</code></li>' for a, n, c in s["recent"])
        open_attr = " open" if s["state"] in ("live", "grading") else ""
        body = (f'<div class="scroll"><table class="{tclass}"><thead><tr>{head}</tr></thead><tbody>{"".join(trs)}'
                f'</tbody></table></div>') if trs else '<p class="empty">No bug sent yet.</p>'
        tx = self.tx.get(str(s["dir"]))
        txh = ""
        if tx:
            tid = "tx-" + re.sub(r"[^A-Za-z0-9_-]", "-", tx["file"][:-5])
            txh = (f'<div class="tx" data-src="transcripts/{esc(tx["file"])}?v={esc(tx["sig"])}">'
                   f'<button type="button" class="txbtn" aria-expanded="false" aria-controls="{tid}">Show transcript'
                   f'</button><span class="meta">{tx["bytes"] / 1e6:.1f} MB · loads on request</span>'
                   f'<div class="txbody" id="{tid}" hidden></div></div>')
        recent_h = (f'<div class="eyebrow">Recent orchestrator tool calls</div><ul class="recent">{recent}</ul>'
                    if recent else "")
        repo = f'{esc(s["task"])} · round {s["round"]}'
        return f'''<details class="card{" control" if s["control"] else ""}"{open_attr}>
<summary><span class="who">{esc(label)}</span><span class="repo">{repo}</span>{state_pill(s)}
<span class="meta">{meta}</span></summary>
<div class="scroll">{self.timeline(s)}</div>
{body}
{msg}{recent_h}
{txh}
</details>'''

    @staticmethod
    def request_tab(ss) -> pd.DataFrame:
        """``analysis.requests.request_table`` over the sent messages of the sessions with replies."""
        rows = pd.DataFrame([r for s in ss if not s["control"] for r in sent_rows(s)])
        return request_table(rows) if not rows.empty else pd.DataFrame()

    @staticmethod
    def hint_tab(ss) -> pd.DataFrame:
        """``analysis.interactivity.condition_table`` (per-bug pickup) over every session's sent bugs."""
        hr = pd.DataFrame([h for s in ss for h in s["hrows"] if h["t_ready_sec"] <= s["elapsed"] + 1])
        try:
            return hint_table(hr) if not hr.empty else pd.DataFrame()
        except (KeyError, TypeError, ValueError):
            return pd.DataFrame()

    def cells(self, cond: str, ss, tab: pd.DataFrame, never: bool = False) -> dict[str, str] | None:
        """A request condition's metrics as table cells, over its sessions that are not invalid (None
        before ``tab``, the request table, has it). Question cells are "–" when the condition asks
        none; the status cell when it sends no status check. ``never``: a median that is infinite
        (at least half never replied to) reads "never" (the levels ladder, where "–" means no such message),
        not "–"."""
        med = (lambda v, n: censored(v, n)) if never else (lambda v, n: clock(v))  # noqa: E731
        mine = [s for s in ss if s["condition"] == cond]
        g = tab[tab["condition"] == cond] if not tab.empty else tab
        if not mine or g.empty:
            return None
        x = g.iloc[0]
        rr = [r for s in mine if s["state"] != "invalid" for r in sent_rows(s)]
        q = [r for r in rr if r["kind"] == "question"]
        b = [r for r in rr if r["kind"] == "bug"]
        st = [r for r in rr if r["kind"] == "status"]
        graded_b = [r for r in b if r.get("fixed") is not None]
        n_ans = sum(r["reply_sec"] is not None for r in q)
        judged = [r for r in q if r.get("correct") is not None]
        if any(s["has_answers"] for s in mine) and judged:
            asked = (f' <span class="muted">· {pct(x["correct_of_asked"])} of asked</span>'
                     if isnum(x["correct_of_asked"]) else "")
            correct = f'{pct(x["correct_of_answered"])} <span class="muted">of {len(judged)}</span>{asked}'
        else:
            correct = "–"
        n_eta = sum(r.get("eta_sec") is not None for r in b)
        never = int(x.get("n_eta_never_done") or 0)
        claims = [r for r in b if r.get("claim_ok") is not None]
        asks = bool(q) or self.sw.sends(cond, "questions_per_interval")
        n_st = sum(r["reply_sec"] is not None for r in st)
        return {
            "answered": f"{n_ans}/{len(q)}" if asks else "–",
            "answer_med": med(x["answer_median_sec"], len(q)) if asks else "–",
            "answered_60s": pct(x["answered_60s"]) if asks else "–",
            "correct": correct if asks else "–",
            "fixed": f'{sum(bool(r["fixed"]) for r in graded_b)}/{len(graded_b)}' if graded_b else "–",
            "eta": f'{n_eta}/{len(b)}{f" <span class=muted>{never} never done</span>" if never else ""}',
            "eta_err": log2err(x["eta_abs_log2_median"]),
            "on_time": pct(x["on_time_frac"]),
            "claims": (f'{pct(x["claim_precision"]) if claims else "–"}'
                       f'{f" <span class=muted>of {len(claims)}</span>" if claims else ""}'),
            "status": (f'{n_st}/{len(st)} <span class="muted">· {med(x["status_reply_median_sec"], len(st))}</span>'
                       if st else "–"),
            "urgent": pct(x["urgent_first_frac"]),
            "cost": f'${sum(s["cost"] for s in mine):.2f}',
        }

    def condition_table(self, ss) -> str:
        """The request conditions: answers, fixes, estimates, done claims, urgency."""
        tab = self.request_tab(ss)
        trs = []
        for cond in self.cond:
            if self.sw.control(cond):
                continue
            mine = [s for s in ss if s["condition"] == cond]
            label, _ = self.label(cond)
            c = self.cells(cond, ss, tab)
            if c is None:
                trs.append(f'<tr><td>{esc(label)}</td><td class="num">{len(mine)}</td>'
                           + '<td class="num">–</td>' * 11 + "</tr>")
                continue
            trs.append(f'<tr><td>{esc(label)}</td><td class="num">{len(mine)}</td>' + "".join(
                f'<td class="num">{c[k]}</td>' for k in ("answered", "answer_med", "answered_60s", "correct", "fixed",
                                                         "eta", "eta_err", "on_time", "claims", "urgent", "cost"))
                       + "</tr>")
        return ('<div class="scroll"><table class="all"><thead><tr><th>Condition</th><th>Sessions</th>'
                '<th>Questions answered</th><th>Answer time, median</th><th>Answered within 60 s</th>'
                '<th>Correct (of answered)</th><th>Bugs fixed (graded)</th><th>ETA given</th>'
                '<th>|log2 ETA error|, median</th><th>Done by the ETA</th><th>Done claims true</th>'
                '<th>Urgent bug first</th><th>Cost</th></tr></thead><tbody>' + "".join(trs) + "</tbody></table></div>")

    LADDER = (("Sessions", "sessions"), ("Bugs fixed (graded)", "fixed"), ("Bug waited to be read, median", "wait"),
              ("First edit after, median", "edit"), ("Questions answered", "answered"),
              ("Answered within 60 s", "answered_60s"), ("Correct (of answered)", "correct"), ("ETA given", "eta"),
              ("Done by the ETA", "on_time"), ("Status check replied · median", "status"),
              ("Urgent bug first", "urgent"), ("Cost", "cost"))

    def ladder(self, ss) -> str:
        """The levels sweep, by setting: one row per level (L0 to L3), so each step down adds one kind of
        message; the per-condition metrics and per-bug pickup, "–" where a level sends no such message."""
        tab, htab = self.request_tab(ss), self.hint_tab(ss)
        by_setting: dict[str, list[str]] = {}
        for c in self.cond:
            lv = level_of(c)
            if lv and not self.sw.control(c):
                by_setting.setdefault(lv[0], []).append(c)
        n_cols = len(self.LADDER) + 1
        trs = []
        for setting, conds in by_setting.items():
            mine = [s for s in ss if s["condition"] in conds]
            planned = sum(len(self.sw.tasks) * len(self.sw.plan.get(c, ())) for c in conds)
            name = SETTINGS.get(setting, (setting.replace("_", " "),))[0]
            trs.append(f'<tr class="grp-head"><td colspan="{n_cols}"><div class="glab"><b>{esc(name)}</b> '
                       f'<span class="muted">· {len(mine)} of {planned} sessions</span></div></td></tr>')
            for c in conds:
                lv, short, _ = list(LEVELS.values())[level_of(c)[1]]
                n = [s for s in ss if s["condition"] == c]
                cell = self.cells(c, ss, tab, never=True) or {k: "–" for _, k in self.LADDER}
                g = htab[htab["condition"] == c] if not htab.empty else htab
                hx = g.iloc[0] if len(g) else None
                plan = len(self.sw.tasks) * len(self.sw.plan.get(c, ()))
                cell.update(sessions=f'{len(n)} <span class="muted">of {plan}</span>',
                            wait=censored(_num(hx, "delivery_median_sec"), _num(hx, "delivery_frac") is not None),
                            edit=censored(_num(hx, "edit_median_sec"), _num(hx, "edit_frac") is not None))
                if not n:
                    cell["cost"] = "–"
                trs.append(f'<tr><td>{lv} <span class="muted">{esc(short)}</span></td>'
                           + "".join(f'<td class="num">{cell[k]}</td>' for _, k in self.LADDER) + "</tr>")
        return ('<div class="scroll"><table class="twins cmp ladder"><thead><tr><th>Level</th>'
                + "".join(f"<th>{h}</th>" for h, _ in self.LADDER) + "</tr></thead><tbody>" + "".join(trs)
                + "</tbody></table></div>")

    def twins_table(self, ss) -> str:
        """Each bugs-only control next to its request twin, on the bug metrics both have."""
        tab = self.hint_tab(ss)
        trs = []
        for ctl, req in self.twin.items():
            for i, cond in enumerate((req, ctl)):
                mine = [s for s in ss if s["condition"] == cond]
                g = tab[tab["condition"] == cond] if not tab.empty else tab
                x = g.iloc[0] if len(g) else None
                label = "with requests" if cond == req else "bugs only"
                graded = [h for s in mine if s["state"] != "invalid" for h in s["hrows"]
                          if h.get("counted") is not None]
                fixed = (f'{pct(_num(x, "hinted_fixed_frac"))} <span class="muted">'
                         f'{sum(bool(h["counted"]) for h in graded)}/{len(graded)}</span>') if graded else "–"
                first = f'<td rowspan="2">{esc(self.label(req)[0])}</td>' if i == 0 else ""
                cost = f"${sum(s['cost'] for s in mine):.2f}" if mine else "–"
                trs.append(f'<tr class="{"twin-top" if i == 0 else ""}">{first}<td>{label}</td>'
                           f'<td class="num">{len(mine)}</td>'
                           f'<td class="num">{clock(_num(x, "delivery_median_sec"))}</td>'
                           f'<td class="num">{clock(_num(x, "edit_median_sec"))}</td><td class="num">{fixed}</td>'
                           f'<td class="num">{cost}</td></tr>')
        return ('<div class="scroll"><table class="twins"><thead><tr><th>Orchestration</th><th>Messages</th>'
                '<th>Sessions</th><th>Bug waited to be read, median</th><th>First edit after, median</th>'
                '<th>Bugs fixed (graded)</th><th>Cost</th></tr></thead><tbody>'
                + "".join(trs) + "</tbody></table></div>")

    # --- compared with an earlier sweep ----------------------------------------------------------

    def _compare_design(self, other: Sweep) -> None:
        """The comparison's groups (this run's condition -> the earlier conditions it is compared with,
        how that is labelled), the grid's columns, and each earlier condition's interval and length."""
        sw = self.sw
        self.c_iv = {c: float(v["interval_sec"]) for c, v in other.interactive.items()}
        self.c_wall = {c: float(v["wall_clock_sec"]) for c, v in other.interactive.items()}
        self.c_bugs = other.n_bugs
        # both sweeps send the same bugs in every round (one order everywhere): pool the rounds
        orders = {(bool(v.get("vary_order_by_round", True)), v.get("order_seed"))
                  for x in (sw, other) for v in x.interactive.values()}
        self.pooled = len(orders) == 1 and not next(iter(orders))[0]
        reqs = [c for c in self.cond if not sw.control(c)]
        key = {c: json.dumps(design(sw.conditions[c]), sort_keys=True) for c in sw.conditions}
        same = {c: [lc for lc in other.interactive if design(other.conditions[lc]) == design(sw.conditions[c])]
                for c in sw.conditions}
        # an earlier design, named after this run's request condition of that design (else its short name)
        designs: dict[str, str] = {}
        for lc in other.interactive:
            mine = next((self.group_label(c).lower() for c in reqs if lc in same[c]), None)
            designs.setdefault(json.dumps(design(other.conditions[lc]), sort_keys=True), mine or self._short(lc))
        self.groups: dict[str, tuple[list[str], str, bool]] = {}  # cond -> (earlier conds, how, same design)
        self.grid_cols: list[tuple[str, str, str, bool]] = []  # (group, condition, label, earlier?)
        for r in reqs:
            for c in [c for c, t in self.twin.items() if t == r] + [r]:
                if same[c]:
                    who = self.group_label(r).lower()
                    how = f"the last run's {who}: the same design, " + (
                        "bugs only" if sw.control(c) else "which had no requests")
                    self.groups[c] = (same[c], how, True)
                else:
                    self.groups[c] = (list(other.interactive), "no last-run condition of the same design: shown "
                                      f"against the last run's {' and '.join(designs.values())}", False)
        # the grid: per design, its earlier conditions, then this run's (controls first, then by level)
        for r in dict.fromkeys(next(x for x in reqs if key[x] == key[c]) for c in reqs):
            grp = self.group_label(r)
            self.grid_cols += [(grp, lc, f"every {self.c_iv[lc]:g} s", True) for lc in same[r]]
            for x in (x for x in reqs if key[x] == key[r]):
                self.grid_cols += [(grp, c, "bugs only", False) for c, t in self.twin.items() if t == x]
                self.grid_cols.append((grp, x, self.col_label(x), False))

    def rk(self, rnd):
        """A round as the comparison keys it: 0 (every round) when the rounds are pooled."""
        return 0 if self.pooled else rnd

    @staticmethod
    def _short(lc: str) -> str:
        base = re.sub(r"_every\d+$", "", lc)
        return SHORT.get(base, base.replace("_", " "))

    def shared_bugs(self, ss, rrows) -> tuple[dict, set]:
        """Per (task, round), this run's bugs as [(position, subtask_id)]: from its sessions when one has
        started, else the earlier sweep's first n_bugs (the same order); and the (task, round) pairs this
        run has graded (a valid graded session in any condition): the grid's rows. Pooled, the round is 0."""
        bugs: dict[tuple, dict[int, str]] = {}
        for r in rrows:
            if r["hint_index"] < self.sw.n_bugs:
                bugs.setdefault((r["task"], self.rk(r["round"])), {})[r["hint_index"]] = r["subtask_id"]
        for s in ss:  # what this run actually sent wins
            for h in s["hrows"]:
                bugs.setdefault((s["task"], self.rk(s["round"])), {})[h["hint_index"]] = h["subtask_id"]
        return ({k: sorted(v.items()) for k, v in bugs.items()},
                {(s["task"], self.rk(s["round"])) for s in ss if s["valid"]})

    def _cmp_head(self) -> str:
        return ('<th>Condition</th><th>Fixed<sup>c</sup></th><th>Edited<sup>a</sup></th><th>First edit, median'
                '<sup>a</sup></th><th>Waited to be read, median<sup>b</sup></th><th>Gap between bugs</th><th>Session, '
                f'as set<sup>d</sup></th><th>Bug {self.sw.n_bugs} at · time left after it</th><th>Sessions</th>'
                '<th>Shared bugs sent</th>')

    def _cmp_cells(self, mine, wall, gap, n_shared) -> str:
        """Fixed (and resolved when they differ), edited k/N, first edit (never-edited as never), waited
        to be read, gap, session as set, time left after the last shared bug, sessions, bugs."""
        n_sess = len({r["trial_name"] for r in mine})
        last = gap * (self.sw.n_bugs - 1)
        mid = (f'<td class="num">{gap:g} s</td><td class="num">{wall / 60:g} min</td>'
               f'<td class="num">{_m(last)} · <b>{(wall - last) / 60:g} min</b></td>')
        if not mine:
            return '<td class="num">–</td>' * 4 + mid + '<td class="num">0</td><td class="num">–</td>'
        n = len(mine)
        fixed = sum(bool(r["counted"]) for r in mine)
        resolved = sum(bool(r["resolved"]) for r in mine)
        edited = sum(isnum(r["edit_sec"]) for r in mine)

        def med(k):  # never-edited count as never: an infinite median reads "never", not "–" (no data)
            v = _median_censored(pd.Series([r[k] for r in mine], dtype=float))
            return clock(v) if isnum(v) and math.isfinite(v) else "never"
        res = f' <span class="muted">· {resolved} resolved</span>' if resolved != fixed else ""
        short = f' <span class="muted">of {n_shared}</span>' if n != n_shared else ""
        return (f'<td class="num">{fixed}/{n} · {pct(fixed / n)}{res}</td>'
                f'<td class="num">{edited}/{n}</td><td class="num">{med("edit_sec")}</td>'
                f'<td class="num">{med("delivery_sec")}</td>' + mid
                + f'<td class="num">{n_sess}</td><td class="num">{n}{short}</td>')

    def _grp_head(self, inner: str) -> str:
        """A group's label row: the label sits in a sticky box so it stays on screen when the table scrolls."""
        n = self._cmp_head().count("<th>")
        return f'<tr class="grp-head"><td colspan="{n}"><div class="glab">{inner}</div></td></tr>'

    def compare_table(self, ss, rrows, shared) -> tuple[str, int]:
        """Grouped by this run's condition: its row, then the matching earlier rows restricted to exactly
        the (task, round) pairs it has graded, so both sides of a group cover the same bugs. Pooled, the
        pairs are tasks and a side's bug sends are its sessions' (one per session and shared bug)."""
        trs, n_groups = [], 0
        for cond, (lasts, how, same) in self.groups.items():
            pairs = {(s["task"], self.rk(s["round"])) for s in ss if s["valid"] and s["condition"] == cond}
            label = self.label(cond)[0]
            if not pairs:
                trs.append(self._grp_head(f'<b>{esc(label)}</b> <span class="muted">· no graded session yet</span>'))
                continue
            n_groups += 1
            n_shared = sum(len(shared.get(k, ())) for k in pairs)
            rows = compare_rows([s for s in ss if s["condition"] == cond], rrows, shared, pairs, self.pooled)

            def sends(mine):  # the shared bugs a side was sent: per session when pooled
                if not self.pooled:
                    return n_shared
                return sum(len(shared.get((t, 0), ())) for t in {r["trial_name"]: r["task"] for r in mine}.values())
            trs.append(self._grp_head(f'<b>{esc(label)}</b> <span class="muted">· on {esc(_pairs_text(pairs))} '
                                      f'({n_shared} shared bugs) · vs {esc(how)}</span>'))
            mine = [r for r in rows if r["condition"] == cond and not r["last"]]
            trs.append(f'<tr class="this"><td>{esc(label)} <span class="pill live">this run</span></td>'
                       + self._cmp_cells(mine, self.sw.wall, self.sw.interval, sends(mine)) + "</tr>")
            for lc in lasts:
                who = "" if same else f"{self._short(lc)}, "
                theirs = [r for r in rows if r["condition"] == lc and r["last"]]
                trs.append(f'<tr><td>{esc(who)}every {self.c_iv[lc]:g} s <span class="pill past">last run</span></td>'
                           + self._cmp_cells(theirs, self.c_wall[lc], self.c_iv[lc], sends(theirs)) + "</tr>")
        return ('<div class="scroll"><table class="twins cmp"><thead><tr>' + self._cmp_head() + '</tr></thead><tbody>'
                + "".join(trs) + "</tbody></table></div>", n_groups)

    def compare_grid(self, rows, shared, pairs) -> tuple[str, int, int]:
        """One row per shared bug, one column per condition: ✓ fixed, ✗ not, – no graded session (pooled:
        a mark per session, every round in one row). A row is marked when, for some condition of this
        run, every session of its own earlier counterpart (the table's group) that has the bug agrees,
        and a graded session of that condition disagrees."""
        cell: dict[tuple, list[dict]] = {}  # keyed by side too: an earlier condition may share a name with one here
        for r in rows:
            cell.setdefault((r["task"], self.rk(r["round"]), r["subtask_id"], r["condition"], r["last"]), []).append(r)
        cols = self.grid_cols
        groups = [(g, sum(c[0] == g for c in cols)) for g in dict.fromkeys(c[0] for c in cols)]
        head1 = "".join(f'<th colspan="{n}" class="grp">{esc(g)}</th>' for g, n in groups)
        head2 = "".join(f'<th class="{"this" if not last else ""}">{esc(label.replace("every ", ""))}'
                        f'<br><span class="muted">{"last" if last else "this"}</span></th>'
                        for _, _, label, last in cols)
        trs, n_marked = [], 0
        for task, rnd in sorted(k for k in shared if k in pairs):
            for i, sid in shared[(task, rnd)]:
                vals, tds = {}, []
                for _, cond, _, last in cols:
                    rs = cell.get((task, rnd, sid, cond, last), [])
                    v = vals[(cond, last)] = [bool(r["counted"]) for r in rs]
                    if not rs:
                        tds.append(f'<td class="g none{" this" if not last else ""}" '
                                   f'title="no graded session">–</td>')
                        continue
                    sym = "".join("✓" if x else "✗" for x in v)
                    tip = "; ".join(f'{"fixed" if r["counted"] else "not fixed"}, first edit '
                                    f'{clock(r["edit_sec"]) if isnum(r["edit_sec"]) else "never"} after it was sent'
                                    for r in rs)
                    ok = "ok" if all(v) else "bad" if not any(v) else ""
                    tds.append(f'<td class="g {ok}{" this" if not last else ""}" title="{esc(tip)}">{sym}</td>')
                why = []  # per this-run condition, against its own earlier counterpart (the table's groups)
                for c, (lasts, _, _) in self.groups.items():
                    past = [x for lc in lasts for x in vals.get((lc, True), [])]
                    mine = vals.get((c, False))
                    if mine and past and len(set(past)) == 1 and any(x != past[0] for x in mine):
                        why.append(f'{self.label(c)[0]}: {"fixed" if not past[0] else "not fixed"} here, '
                                   f'{"fixed" if past[0] else "not fixed"} in every last-run '
                                   f'session of its counterpart')
                n_marked += bool(why)
                title = f' title="{esc("; ".join(why))}"' if why else ""
                trs.append(f'<tr class="{"diffrow" if why else ""}"{title}><td>{esc(task)}</td>'
                           f'<td class="num">{rnd or "all"}</td>'
                           f'<td class="num">{i + 1}</td><td class="mono">{esc(sid.split("-", 1)[-1][:8])}</td>'
                           f'{"".join(tds)}</tr>')
        html_ = ('<div class="scroll"><table class="grid"><thead><tr><th colspan="4"></th>' + head1 + '</tr><tr>'
                 '<th>Repo</th><th>Round</th><th>Bug</th><th>Subtask</th>' + head2 + '</tr></thead><tbody>'
                 + "".join(trs) + "</tbody></table></div>")
        return html_, len(trs), n_marked

    def compare_html(self, ss) -> str:
        if self.compare is None:
            return ""
        cdir, other = self.compare
        h2 = '<h2>Compared with the last run</h2>'
        name = SWEEP_LABEL.get(cdir.name, cdir.name)
        ok = cdir.is_dir() and isinstance(other, Sweep)
        rrows = rate_rows(cdir, self.sw.n_bugs) if ok else None
        if not rrows:
            why = ("its jobs folder is missing" if not cdir.is_dir() else esc(other) if not ok
                   else "it has no valid graded session")
            return (f'<section><div class="sechead">{h2}</div><p class="empty">The comparison with the last run '
                    f'(the {esc(name)} sweep) is unavailable: {why}, so no shared bugs can be matched.</p></section>')
        nb, cb, wall, iv = self.sw.n_bugs, self.c_bugs, self.sw.wall, self.sw.interval
        later = cb > nb  # the earlier sweep sent bugs this one never sends
        shared, pairs = self.shared_bugs(ss, rrows)
        n_all = sum(len(v) for v in shared.values())
        # does the earlier sweep send this run's bugs in the same positions? (it should: same seed, its first n_bugs)
        pos = {(k, sid): i for k, v in shared.items() for i, sid in v}
        moved = sum(1 for r in rrows if ((r["task"], self.rk(r["round"])), r["subtask_id"]) in pos
                    and pos[((r["task"], self.rk(r["round"])), r["subtask_id"])] != r["hint_index"])
        order = ("in the same order" if not moved
                 else f"<b>but {moved} last-run bug sends are at a different position</b>")
        repos = esc(", ".join(sorted({t for t, _ in shared})))
        if self.pooled:
            intro = (f'<p class="lede">The same repositories ({repos}) and the same first {nb} bugs {order} as the '
                     f'{esc(name)} run, which sent {cb} per session: <b>{n_all} shared bugs</b>. Both runs send the '
                     f'same bugs in every round, so rounds are pooled: each group below compares one condition of this '
                     f'run with the last run on exactly the repositories that condition has graded, matched by '
                     f'subtask, so both sides of a group cover the same bugs. Groups differ in repositories, so '
                     f'compare within a group, not across.</p>')
        else:
            intro = (f'<p class="lede">The same repositories ({repos}) '
                     f'and, in each round, the same first {nb} bugs {order} as the {esc(name)} run, which sent {cb} '
                     f'per session: <b>{n_all} shared bugs</b>. Each group below compares one condition of this run '
                     f'with the last run on exactly the repository and round pairs that condition has graded, matched '
                     f'by subtask, so both sides of a group cover the same bugs. Groups differ in pairs, so compare '
                     f'within a group, not across.</p>')
        table, n_groups = self.compare_table(ss, rrows, shared)
        first = [r for r in rrows if r["hint_index"] < nb]
        n_any = sum(isnum(r.get("edit_sec_any")) for r in first)
        n_own = sum(isnum(r["edit_sec"]) for r in first)
        n_moved = sum(isnum(r["edit_sec"]) and r["edit_sec"] != r.get("edit_sec_any") for r in first)
        own = (f'In the last run, an edit made once one of its bugs {nb + 1} to {cb} had arrived, to files that bug '
               f'also targets, may be work on that bug, so it is left out (this run has no later bugs). Over all its '
               f'rounds and pairs, this leaves {n_own} of its {len(first)} sends of bugs 1 to {nb} edited instead '
               f'of {n_any}, and moves {n_moved} first edits later. ') if later else ""
        gate = (f', so in the last run a break made while working on bugs {nb + 1} to {cb} zeroes its fixes of bugs 1 '
                f'to {nb} too; "resolved" shows the hidden-test count where it differs') if later else \
            '; "resolved" shows the hidden-test count where it differs'
        notes = ('<p class="note"><sup>a</sup> Edited: the agent wrote to one of the bug\'s files after it was sent. '
                 f'{own}A scratch script saved with '
                 '<code>cat &lt;&lt;EOF &gt; file</code> is not an edit of the paths written inside it. A bug never '
                 'edited counts as infinitely late and is censored at the end of its session; the time left after it '
                 'differs between conditions, so a condition with more time can edit late bugs another never reaches. '
                 'Read the edited count first; "never" when the median is never, "–" when there is no data. '
                 '<sup>b</sup> Includes the time the bug queued behind the agent\'s current step, '
                 'which grows when bugs arrive faster, so it depends on the gap between bugs. <sup>c</sup> Resolved by '
                 'the hidden tests with the session\'s visible tests passing; only valid graded sessions count. The '
                 f'visible-tests gate is per session{gate}. <sup>d</sup> The configured length; a '
                 'session ends early when the agent goes idle, so the time left is nominal.</p>')
        runs = sorted({(self.c_iv[c], self.c_wall[c]) for c in self.c_iv})
        walls = list(dict.fromkeys(f"{w / 60:g}" for _, w in runs))
        there = (f'there, bugs {nb + 1} to {cb} arrived after these, competed for the same time, and could fail the '
                 f'visible tests that gate every fix of the session (c); ') if later else ""
        if runs == [(iv, wall)]:  # the same schedule: only what the sessions were asked and the day differ
            caveat = (f'<p class="note caveat">What still differs inside a group: not the schedule (sessions of '
                      f'{wall / 60:g} min, a bug every {iv:g} s, in both runs); {there}{self._cmp_here()}and it ran '
                      f'on a different day under different provider load. Per-bug fixed on the same bugs is the most '
                      f'direct comparison.</p>')
        else:
            caveat = (f'<p class="note caveat">What still differs inside a group: the last run\'s sessions lasted '
                      f'{_or(walls)} min (every {", ".join(f"{i:g}" for i in dict.fromkeys(i for i, _ in runs))} s) '
                      f'against {wall / 60:g} min here; bug {nb} came at '
                      f'{_or([f"{_m(i * (nb - 1))} with {(w - i * (nb - 1)) / 60:g} min left" for i, w in runs])}, '
                      f'against {_m(iv * (nb - 1))} with {(wall - iv * (nb - 1)) / 60:g} min left here; {there}'
                      f'{self._cmp_here()}and it ran on a different day under different provider load. Per-bug fixed '
                      f'on the same bugs is the most direct comparison, but it still favours conditions with more time '
                      f'per bug.</p>')
        if not n_groups:
            return (f'<section><div class="sechead">{h2}</div>{intro}{table}'
                    f'<p class="empty">No graded session yet: the comparison starts with the first.</p></section>')
        grid, n_grid, n_marked = self.compare_grid(compare_rows(ss, rrows, shared, pairs, self.pooled), shared, pairs)
        n_cmp = sum(len(shared[k]) for k in pairs if k in shared)
        tail = (f' Last-run first edits leave out possible work on bugs {nb + 1} to {cb} (a).' if later else "")
        odd = [c for c, (_, _, same) in self.groups.items() if not same]
        alt = ("" if not odd else "; for the hinted condition, both designs" if self.sw.name == REQUESTS_SWEEP
               else f"; for {_and([self.label(c)[0] for c in odd])}, every last-run condition")
        pooled = " (pooled: a mark per session)" if self.pooled else ""
        return (f'<section><div class="sechead">{h2}<span class="prog">{n_cmp} of {n_all} shared bugs have a graded '
                f'session here</span></div>{intro}{table}{notes}{caveat}'
                f'<div class="sechead"><h3>Bug by bug</h3><span class="prog">{n_grid} bugs · ✓ fixed · ✗ not fixed · '
                f'– no graded session · hover for the first edit</span></div>{grid}'
                f'<p class="note">Each row is one bug, so its cells compare the same bug{pooled}. Highlighted '
                f'({n_marked}): for a condition of this run, every session of its own last-run counterpart (the '
                f'table\'s group: the same design{alt}) that has the bug agrees, and a '
                f'graded session of that condition disagrees; "–" cells are left out.{tail}</p></section>')

    def _cmp_here(self) -> str:
        """The caveat's clause on what this run's sessions do that the earlier run's did not."""
        if self.sw.name == REQUESTS_SWEEP:
            return "here the request conditions also answer questions and status checks; "
        if self.sw.levels:
            snaps, eta = self._rules()[:2]
            rules = _and([x for x, on in (("records workspace snapshots", snaps),
                                          ("must give an estimate or done with each bug reply", eta)) if on])
            return ("here L1 to L3 also get status checks, L2 and L3 questions and L3 an urgent bug; "
                    + (f"every session here {rules}; " if rules else ""))
        kinds = [k for key, k in (("questions_per_interval", "questions"), ("status_every", "status checks"),
                                  ("urgent_every", "an urgent bug")) if any(self.sw.sends(c, key) for c in self.cond)]
        return f"here some conditions also get {_and(kinds)}; " if kinds else ""

    # --- the header ------------------------------------------------------------------------------

    def _rules(self) -> tuple[bool, bool, float | None]:
        """Whether every condition records workspace snapshots, whether every one refuses a bug reply
        without eta_min or done, and the snapshots' heartbeat (``interactive.snapshot_every_sec``)."""
        v = [self.sw.interactive[c] for c in self.cond if c in self.sw.interactive]
        every = {x.get("snapshot_every_sec") for x in v if x.get("snapshots")}
        return (bool(v) and all(x.get("snapshots") for x in v), bool(v) and all(x.get("require_eta") for x in v),
                next(iter(every)) if len(every) == 1 and isnum(next(iter(every))) else None)

    def models_text(self) -> str:
        """Who runs, from the resolved configs: the models (and the subagents' caps), and where the task
        containers run (``harbor.environment``); "" without configs."""
        cfgs = [self.sw.configs[c] for c in self.cond if c in self.sw.configs]
        short = lambda m: str(m).rsplit("/", 1)[-1]  # noqa: E731
        orch = {(c.get("orchestrator") or {}).get("model") for c in cfgs} - {None}
        if not orch:
            return ""
        subs = [c for c in cfgs if c.get("topology") == "subagents"]
        sub_models = {(c.get("subagent") or {}).get("model") or (c.get("orchestrator") or {}).get("model")
                      for c in subs}  # a subagent without its own model runs the orchestrator's
        caps = {((c.get("subagent") or {}).get("max_concurrent"), (c.get("subagent") or {}).get("max_steps"))
                for c in subs}
        cap = (f" (up to {next(iter(caps))[0]} at once, {next(iter(caps))[1]} steps each)"
               if len(caps) == 1 and None not in next(iter(caps)) else "")
        if len(orch | sub_models) == 1:
            text = f"Every model is {short(next(iter(orch)))}" + (f", subagents too{cap}" if subs else "")
        else:
            text = f"Orchestrator: {_and(sorted(map(short, orch)))}" + (
                f"; subagents: {_and(sorted(map(short, sub_models - {None})))}{cap}" if subs else "")
        modal = (self.sw.raw.get("harbor") or {}).get("environment") == "modal"
        return text + ("; the task containers run on Modal." if modal else ".")

    def level_conds(self) -> dict[int, list[str]]:
        """Level number -> this sweep's conditions at that level, in page order (levels conditions only)."""
        out: dict[int, list[str]] = {}
        for c in self.cond:
            lv = level_of(c)
            if lv:
                out.setdefault(lv[1], []).append(c)
        return dict(sorted(out.items()))

    def header(self) -> dict[str, str]:
        """The page's title, eyebrow, heading and lede (HTML): the requests sweep's own text, the levels
        sweep's built from its design, else a short lede built from the design."""
        sw = self.sw
        if sw.name == REQUESTS_SWEEP:
            return {"title": "Requests Sweep Monitor", "eyebrow": "subagents-bench · requests sweep",
                    "h1": "Is the agent there when you ask?",
                    "lede": REQUESTS_LEDE.format(minutes=f"{sw.wall / 60:g}", interval=f"{sw.interval / 60:g}",
                                                 n_bugs=sw.n_bugs)}
        if sw.levels:
            smoke = "smoke" in sw.name
            who = "levels smoke" if smoke else "levels sweep" if sw.name == LEVELS_SWEEP else sw.name
            return {"title": "Levels Smoke Monitor" if smoke else "Levels Sweep Monitor",
                    "eyebrow": f"subagents-bench · {esc(who)}",
                    "h1": f"Levels smoke: every kind of request in {dur(sw.wall)}" if smoke
                    else "One kind of request at a time", "lede": self.levels_lede(smoke)}
        return {"title": f"{esc(sw.name)} Monitor", "eyebrow": "subagents-bench · interactive sweep",
                "h1": esc(sw.name), "lede": self.generic_lede()}

    def levels_lede(self, smoke: bool) -> str:
        sw, by = self.sw, self.level_conds()
        s = schedule(sw.icfg)
        steps = []
        for k, conds in by.items():
            x = schedule(sw.interactive.get(conds[0]) or {})
            steps.append("L0 sends only the bugs (PR #13's baseline)" if k == 0 else
                         f"L1 adds a status check at {_and([_m(t) for t in x['checks']])} asking how long each bug "
                         f"sent so far will take" if k == 1 else
                         f"L2 adds {x['q']} questions about the code after each bug" if k == 2 else
                         f"L3 makes the bug sent at {_and([_m(t) for t in x['urgent']])} urgent")
        snaps, eta, _ = self._rules()
        checks = ["every kind of message"] + ["the reply rule"] * eta + ["the snapshots"] * snaps
        pre = (f"A smoke of the levels sweep ({dur(sw.wall)} sessions), too short for results: it checks that "
               f"{_and(checks)} work{'s' * (len(checks) == 1)}. ") if smoke else ""
        return (f"{pre}A simulated user works with the agent on one repository for {dur(sw.wall)}, sending the same "
                f"{len(s['bugs'])} bugs at the same times at every level, one every {dur(sw.interval)}. Each level "
                f"adds one kind of message to the level below and changes nothing else: {'; '.join(steps)}. The user "
                f"sees only what the agent sends with <code>reply</code>"
                + (", which refuses a reply to a bug without <code>eta_min</code> or <code>done</code>" if eta else "")
                + ". " + ("Snapshots of the workspace give each bug's real fix time, graded after the run. "
                          if snaps else "") + self.models_text()).strip()

    def generic_lede(self) -> str:
        sw = self.sw
        s, conds = schedule(sw.icfg), list(self.cond)
        extras = []
        for key, what in (("questions_per_interval", f"{s['q']} questions about the code after each bug"),
                          ("status_every", "status checks"), ("urgent_every", "an urgent bug")):
            has = [c for c in conds if sw.sends(c, key)]
            if has:
                extras.append(what + ("" if len(has) == len(conds) else " in some conditions"))
        n_reply = sum(not sw.control(c) for c in conds)
        replies = ("The user sees only what the agent sends with <code>reply</code>. " if n_reply == len(conds) else
                   "Conditions without <code>reply</code> get bugs only. " if n_reply else "")
        tasks = _and([t.replace("swe-sweep__", "") for t in sw.tasks])
        return (f"A simulated user works with the agent on one repository for {dur(sw.wall)}: {len(s['bugs'])} bugs, "
                f"one every {dur(sw.interval)}{', ' if extras else ''}{_and(extras) if extras else ''}. {replies}"
                f"{len(conds)} conditions on {esc(tasks)}, {sw.rounds} round{'s' if sw.rounds > 1 else ''}: "
                f"{sw.n_sessions} sessions. {self.models_text()}").strip()

    def levels_html(self) -> str:
        """The levels sweep: what each level adds to the one below (its messages and config), and how
        the levels are run."""
        sw, by = self.sw, self.level_conds()
        trs, below = [], None
        for k, conds in by.items():
            lv, short, _ = list(LEVELS.values())[k]
            icfg = sw.interactive.get(conds[0]) or {}
            x = schedule(icfg)
            if below is None:
                keys = ["replies", "questions_per_interval", "status_every", "urgent_every"]
                adds = (f"{len(x['bugs'])} bugs, one every {dur(x['iv'])} from 0:00 to {_m(x['bugs'][-1])}, "
                        f"answered with <code>reply</code>")
            else:
                keys = [key for key in icfg if icfg.get(key) != below.get(key)]
                parts = []
                if "status_every" in keys:
                    parts.append(f"a status check at {_and([_m(t) for t in x['checks']])} ({dur(x['iv'] / 2)} after "
                                 f"the {_and([_nth(int(t // x['iv']) + 1) for t in x['checks']])} bugs) naming the "
                                 f"bugs sent so far and asking for an estimate of each")
                if "questions_per_interval" in keys:
                    parts.append(f"{x['q']} questions about the code after each bug (at "
                                 f"{_and(['+' + _m(t) for t in x['q_at']])}): {x['n_q']} per session")
                if "urgent_every" in keys:
                    parts.append(f"the bug sent at {_and([_m(t) for t in x['urgent']])} is urgent: \"fix it before "
                                 f"the earlier ones\"")
                adds = "+ " + "; + ".join(parts) if parts else "–"
            cfg = ", ".join(f"<code>{esc(key)}: {esc(json.dumps(icfg.get(key)))}</code>" for key in keys) or "–"
            n = sum(len(sw.tasks) * len(sw.plan.get(c, ())) for c in conds)
            rounds = _and(list(dict.fromkeys(_rounds(sw.plan.get(c, ())) for c in conds)))
            trs.append(f'<tr><td>{lv} <span class="muted">{esc(short)}</span><br><code class="muted">'
                       f'*_{esc(list(LEVELS)[k])}</code></td><td>{adds}</td><td>{cfg}</td><td class="num">{n} '
                       f'<span class="muted">· {rounds}</span></td></tr>')
            below = icfg
        same = all(not v.get("vary_order_by_round", True) for v in sw.interactive.values())
        snaps, eta, every = self._rules()
        settings = [SETTINGS.get(s, (s.replace("_", " "),))[0].lower()
                    for s in dict.fromkeys(level_of(c)[0] for cs in by.values() for c in cs)]
        tasks = _and([t.replace("swe-sweep__", "") for t in sw.tasks])
        plans = {}
        for k, conds in by.items():
            plans.setdefault(_rounds(sw.plan.get(conds[0], ())), []).append(list(LEVELS.values())[k][0])
        runs = ("; ".join(f"{r} run {_and(ls)}" for r, ls in plans.items()) if len(plans) > 1
                else f"{next(iter(plans), 'every round')} for every level")
        notes = [f"Every level differs from the one below only in the messages it sends: the same system prompt "
                 f"(<code>replies: true</code> puts the whole reply section in it at every level) and the same bugs at "
                 f"the same times{', in the same order in every round and setting' if same else ''}.",
                 f"Each level runs in {len(settings)} setting{'s' * (len(settings) > 1)} ({esc(_and(settings))}), on "
                 f"{esc(tasks)}: {runs}, {sw.n_sessions} sessions in all."]
        if snaps:
            notes.append("Every session records snapshots of the workspace (after each edit or command of any agent"
                         + (f", and every {every:g} s" if every else "") + "); after the run, "
                         "<code>subagents-bench fix-times</code> grades each one with the hidden tests, so each bug's "
                         "real fix time is known, not only whether it ended fixed. Bugs fixed on this page is the "
                         "end-of-session grade.")
        if eta:
            notes.append("<code>reply</code> refuses a reply to a bug that has neither <code>eta_min</code> nor "
                         "<code>done: true</code>, so every bug reply carries an estimate or a done claim; replies to "
                         "questions and status checks are unaffected.")
        if sw.name == LEVELS_SWEEP:
            notes.append("The run's main numbers, computed after it: the share of L1's estimates met (against the "
                         "real fix times), the share of L2's questions answered correctly within 60 s (per setting), "
                         "and how often L3's urgent bug is fixed sooner than the same bug in L2.")
        return ('<section><h2>What each level adds</h2><div class="scroll"><table class="diff"><thead><tr><th>Level, '
                'conditions</th><th>Messages added to the level below</th><th>Config</th><th>Sessions</th></tr>'
                f'</thead><tbody>{"".join(trs)}</tbody></table></div><ul class="notes">'
                + "".join(f"<li>{n}</li>" for n in notes) + "</ul></section>")

    def conds_html(self) -> str:
        """A card per condition; the levels sweep's, a card per setting (the levels table has the rest)."""
        if self.sw.levels:
            settings = dict.fromkeys(level_of(c)[0] for c in self.cond if level_of(c))
            rest = [c for c in self.cond if not level_of(c)]
            cards = [SETTINGS.get(s, (s.replace("_", " "), "")) for s in settings] + [self.cond[c] for c in rest]
            return "".join(f'<div class="cond"><div class="who">{esc(n)}</div><p>{esc(d)}</p></div>' for n, d in cards)
        return "".join(f'<div class="cond{" control" if c in self.twin else ""}"><div class="who">{esc(n)}</div>'
                       f'<p>{esc(d)}</p></div>' for c, (n, d) in self.cond.items())

    def nav_html(self) -> str:
        """The tab bar: a link per page, the current one (``href`` equal to this page's file) marked."""
        if not self.nav:
            return ""
        cur = ' aria-current="page"'
        tabs = "".join(f'<a href="{esc(href)}"{cur if href == self.page else ""}>{esc(label)}</a>'
                       for label, href in self.nav)
        return f'<nav class="tabs" aria-label="Sweeps">{tabs}</nav>\n'

    # --- the whole page --------------------------------------------------------------------------

    def differences_html(self) -> str:
        """The requests sweep against the earlier interactive runs (that sweep only); the levels sweep's
        levels."""
        if self.sw.levels:
            return self.levels_html()
        if self.sw.name != REQUESTS_SWEEP:
            return ""
        icfg, nb, iv, wall = self.sw.icfg, self.sw.n_bugs, self.sw.interval, self.sw.wall
        last = iv * (nb - 1)
        diffs = [
            ("What the user sends", "Bugs only", "Bugs, questions about the code, status checks; bug 3 is urgent"),
            ("How the agent answers", "Nothing to answer: only edits counted", "A <code>reply(to, text, eta_min, done)"
             "</code> tool; the user sees nothing else"),
            ("Prompt rules", "Fix the bugs", "The first reply to a question is the answer; estimate a bug "
             "(<code>eta_min</code>) in the first response after it arrives, update it when it changes, "
             "<code>done</code> once fixed"),
            ("Schedule", "10 bugs every 60 to 300 s",
             f"{nb} bugs every {iv / 60:g} min; {icfg.get('questions_per_interval', 0)} questions between bugs "
             f"(80 s and 160 s after each); a status check 2 min after bugs 2, 4 and 6"),
            ("Session", "30 min (slow commands); 20, 40 or 60 min (arrival rate)",
             f"{wall / 60:g} min: the last bug at "
             f"{_m(last)}, then {(wall - last) / 60:g} min more (ends early when idle)"),
            ("Conditions", "Single agent vs background manager", "Single agent, background manager, all tools + "
             "background subagents with a delegation hint; and two bugs-only controls (single, background manager): "
             "the "
             "same bugs at the same times, no replies, questions, status checks or urgency"),
            ("Repositories", "pint, isort, click",
             f"{' and '.join(t.replace('swe-sweep__', '') for t in self.sw.tasks)}, "
             f"{self.sw.rounds} rounds: {self.sw.n_sessions} sessions"),
            ("Measured", "Wait to be read, first touch and edit, fixed", "Answer time and correctness, fixes, ETA "
             "accuracy (log2 error, on time), done claims, urgent bug first; against the controls, what the requests "
             "cost in fixes and in pickup time"),
        ]
        trs = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in diffs)
        return (f'<section><h2>What changed from the earlier runs</h2><div class="scroll"><table class="diff">'
                f'<thead><tr>'
                f'<th></th><th>Earlier runs (slow commands, arrival rate)</th><th>This sweep</th></tr></thead>'
                f'<tbody>{trs}</tbody></table></div><ul class="notes"><li>Before, an agent could serve the bugs it '
                f'already knew and still look responsive. Now the user also asks things that need an answer within '
                f'minutes, so the agent has to stay available while it works.</li><li>Every bug reply can carry an '
                f'estimate, so the sweep measures how well the agent predicts its own work, and whether "done" is '
                f'true.</li><li>The bugs-only controls run on the same day and provider, so the gap to their twins is '
                f'the price of the requests.</li></ul></section>')

    def render(self, ss: list[dict] | None = None) -> str:
        ss = self.sessions() if ss is None else ss
        sw = self.sw
        count = lambda st: sum(s["state"] == st for s in ss)  # noqa: E731
        # the planned sessions, and any session of a round the plan left out (run with --rounds anyway)
        total = sw.n_sessions + sum(s["round"] not in sw.plan.get(s["condition"], ()) for s in ss)
        want = max((schedule(v)["n_q"] for v in sw.interactive.values()), default=0)  # questions per task
        qs = " · ".join(f"{t} {min(n, want)}/{want}" if n is not None else f"{t} not written yet"
                        for t, n in self.questions_ready())
        order = list(self.cond)
        if ss:
            cards = "".join(self.session_card(s) for s in sorted(
                ss, key=lambda s: (s["state"] == "graded", s["round"] or 0, order.index(s["condition"])
                                   if s["condition"] in self.cond else 99, s["task"])))
            twins = (f'<div class="sechead"><h3>Bugs only vs with requests</h3><span class="prog">the same bugs at the '
                     f'same times; the gap is what the requests cost</span></div>{self.twins_table(ss)}'
                     if self.twin else "")
            ladder = (f'<section><div class="sechead"><h2>By level</h2><span class="prog">per setting: each row adds '
                      f'one kind of message to the row above</span></div>{self.ladder(ss)}</section>'
                      if sw.levels and self.level_conds() else "")
            body = (f'{ladder}<section><div class="sechead"><h2>By condition</h2><span class="prog">'
                    f'{count("graded")} of {total} sessions graded</span></div>{self.condition_table(ss)}{twins}'
                    f'</section>'
                    f'{self.compare_html(ss)}'
                    f'<section><div class="sechead"><h2>Sessions</h2><span class="prog">live and grading first</span>'
                    f'</div>{cards}</section>')
        else:
            body = ('<section><h2>Sessions</h2><p class="empty">Waiting for the first session. Nothing under '
                    f'<code>{esc(self.jobs.name)}</code> has started yet.</p></section>')
        pickup = ("Bugs only vs with requests" if self.twin else
                  "By level, bug waited to be read and first edit after" if sw.levels else "")
        page = TEMPLATE.format(
            snap=esc(time.strftime("%a %d %b, %H:%M %Z")), live=count("live"), grading=count("grading"),
            graded=count("graded"), total=total, cost=sum(s["cost"] for s in ss),
            questions=f"<span>Questions ready: {esc(qs)}</span>" if want else "", conds=self.conds_html(),
            differences=self.differences_html(), body=body, nav=self.nav_html(), **self.header(),
            controls_note=CONTROLS_NOTE if any(sw.control(c) for c in self.cond) else "",
            pickup_note=PICKUP_NOTE.format(what=pickup) if pickup else "",
            footer_extra=LEVELS_NOTE if sw.levels else "", viewer=VIEWER)
        return page if self.page == "index.html" else DOC_HEAD + page


# --- per-session helpers ------------------------------------------------------------------------

def sent_rows(s) -> list[dict]:
    return [r for r in s["rows"] if r["t_ready_sec"] <= s["elapsed"] + 1]


def stats(s) -> dict:
    sent = sent_rows(s)
    q = [r for r in sent if r["kind"] == "question"]
    b = [r for r in sent if r["kind"] == "bug"]
    return {"q": len(q), "answered": sum(r["reply_sec"] is not None for r in q),
            "answer_med": _median_inf([r["reply_sec"] for r in q]), "bugs": len(b),
            "eta": sum(r.get("eta_sec") is not None for r in b), "done": sum(r.get("done_sec") is not None for r in b),
            "fixed": sum(bool(r.get("fixed")) for r in b), "edited": sum(r.get("edit_sec") is not None for r in b),
            "edit_med": _median_inf([r.get("edit_sec") for r in b])}


def state_pill(s) -> str:
    st = stats(s)
    if s["state"] == "invalid":
        why = esc(str(s.get("why") or s["exit"]).replace("_", " "))
        return f'<span class="pill warn">Invalid · {why}, rerun later</span>'
    if s["state"] == "graded":
        return f'<span class="pill ok">Fixed {st["fixed"]} of {st["bugs"]} bugs</span>'
    if s["state"] == "grading":
        return '<span class="pill warn">Grading</span>'
    return f'<span class="pill live">Live · {clock(s["elapsed"])}</span>'


def message_cell(r, s=None) -> str:
    """A message's text; with its session ``s``, a status check names the bugs sent before it (the
    levels sweep's check text) instead of asking about the open ones (the requests sweep's)."""
    if r["kind"] == "question":
        return f'<span title="{esc(r.get("question") or "")}">{esc(trunc(r.get("question"), 110))}</span>'
    if r["kind"] == "bug":
        return f'<span class="mono">{esc(r.get("subtask_id") or "")}</span>'
    if s is not None:
        ids = ", ".join(f'#{b["msg_id"]}' for b in s["rows"]
                        if b["kind"] == "bug" and b["t_ready_sec"] < r["t_ready_sec"])
        return f"How long until each bug so far ({ids}) is fixed?"
    return "How long will each open bug take?"


def reply_cell(r, s) -> str:
    mine = [x for x in s["replies"] if x["to"] == r["msg_id"]]
    if r["kind"] == "question":
        a = r.get("answer")
        return (f'<span title="{esc(a or "")}">{esc(trunc(a, 140))}</span>' if a
                else '<span class="muted">no answer</span>')
    if r["kind"] == "status":
        after = [x for x in s["replies"] if x["t"] - s["t0"] >= r["t_ready_sec"]][:1]
        return (f'<span class="muted">first reply re #{after[0]["to"]}:</span> {esc(trunc(after[0]["text"], 110))}'
                if after else '<span class="muted">no reply</span>')
    bits = []
    if r.get("eta_min") is not None:
        due = r["t_ready_sec"] + r["eta_sec"] + 60 * r["eta_min"]
        bits.append(f'estimate {mins(r["eta_min"])} at {clock(r["t_ready_sec"] + r["eta_sec"])} (due {clock(due)})')
        n = r.get("n_estimates") or 0
        if n > 1:
            bits.append(f'{n - 1} update{"s" if n > 2 else ""}')
    if r.get("done_sec") is not None:
        bits.append(f'said done at {clock(r["t_ready_sec"] + r["done_sec"])}')
    err = r.get("eta_log2_err")
    if isnum(err):
        bits.append("never said done" if math.isinf(err) else f'log2 error {log2err(err, True)}')
    txt = "; ".join(bits) or ("no estimate" if mine else "no reply")
    return f'<span title="{esc(mine[-1]["text"] if mine else "")}">{esc(txt)}</span>'


def verdict_cell(r, s) -> str:
    if r["kind"] == "question":
        if r["reply_sec"] is None:
            return '<span class="v bad">unanswered</span>' if s["state"] != "live" else "–"
        ok = {True: '<span class="v ok">correct</span>', False: '<span class="v bad">wrong</span>'}
        return ok.get(r.get("correct"), "–")
    if r["kind"] == "bug":
        ok = {True: '<span class="v ok">fixed</span>', False: '<span class="v bad">not fixed</span>'}
        v = ok.get(r.get("fixed"), "–")
        if r.get("on_time") is not None:
            v += f' · {"on time" if r["on_time"] else "late"}'
        if r.get("urgent_first") is not None:
            v += f' · before {pct(r["urgent_first"])} of earlier'
        return v
    return "–"


def compare_rows(ss, rrows, shared, pairs, pooled: bool = False) -> list[dict]:
    """Per-bug rows of both sweeps on the shared bugs of the given pairs: this run's valid graded
    sessions, the earlier rows whose subtask is among this run's bugs for that task and round (pooled:
    the pairs' round is 0, every round); ``last`` tells the earlier rows apart (PR #13's baseline
    conditions have the same names as the levels sweep's L0)."""
    ids = {k: {sid for _, sid in v} for k, v in shared.items()}
    pos = {(k, sid): i for k, v in shared.items() for i, sid in v}
    key = (lambda r: (r["task"], 0)) if pooled else (lambda r: (r["task"], r["round"]))  # noqa: E731
    rows = []
    for s in ss:
        if s["valid"] and key(s) in pairs:
            rows += [{**h, "task": s["task"], "round": s["round"], "last": False} for h in s["hrows"]]
    rows += [{**r, "last": True} for r in rrows if key(r) in pairs]
    return [{**r, "pos": pos[(key(r), r["subtask_id"])]} for r in rows if r["subtask_id"] in ids.get(key(r), ())]


def _pairs_text(pairs) -> str:
    """The (task, round) pairs; pooled ones (round 0) as the tasks, every round."""
    if all(r == 0 for _, r in pairs):
        return ", ".join(t for t, _ in sorted(pairs)) + ", every round"
    return ", ".join(f"{t} r{r}" for t, r in sorted(pairs))


PAGE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.html")


def build(jobs: Path, out: Path, compare: Path | None = None, spec: Path | None = None,
          transcripts: bool = True, page: str = "index.html",
          nav: list[tuple[str, str]] | None = None) -> tuple[Path, dict[str, dict], int]:
    """Write ``out/<page>`` (and ``out/transcripts/``, shared by every page written to ``out``) with
    the tab bar ``nav`` ([(label, href)]); return the page, the exported transcripts and how many were
    rewritten. ValueError for a page name that is not a plain ``*.html`` file name."""
    from .transcripts import export_all
    if not PAGE_RE.fullmatch(page):
        raise ValueError(f"--page {page!r}: a file name like levels.html (letters, digits, '_', '.', '-')")
    jobs, out = Path(jobs), Path(out)
    sw = load_sweep(jobs, spec)
    cmp_ = None
    if compare is not None:
        compare = Path(compare)
        try:
            cmp_ = (compare, load_sweep(compare))
        except FileNotFoundError:
            cmp_ = (compare, "its sweep.json is missing")
        except ValueError as e:
            cmp_ = (compare, f"its {e}")
    out.mkdir(parents=True, exist_ok=True)
    tx, wrote = export_all(trial_dirs(jobs), out / "transcripts") if transcripts else ({}, 0)
    path = out / page
    path.write_text(Monitor(jobs, sw, cmp_, tx, page=page, nav=nav).render())
    return path, tx, wrote
