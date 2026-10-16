"""When each bug was really fixed: the grader's replayed verdicts on every recorded workspace state
of a session (``interactive.snapshots``) turned into per-bug fix times (``fixes.csv``).

``subagents-bench fix-times`` writes ``agent/fix_states.json``: the states, in order, are the
snapshots that changed the workspace (``agent/snapshots.jsonl``, ``changed``) and last the final
``model.patch`` (at the episode's end), each with every sent bug's status: ``pass``, ``fail`` or
``unknown`` (the targeted run errored or timed out, the build failed, the tests did not import),
and for the first ``fail`` after a ``pass`` the status of one rerun. A state holds from its time
until the next one; it is *observed* once, plus once per later successful snapshot that found the
workspace unchanged (from ``snapshots.jsonl``; without it each state counts once).

Per bug, in state order:

- ``unknown`` carries no information: it never starts, breaks or ends anything, and neither does
  a ``fail`` whose rerun passed (a flaky test);
- a *failing stretch* runs from a ``fail`` through any ``fail``/unknown states to the next
  ``pass``. It is *confirmed* when its fails were observed at least ``CONFIRM_OBS`` (2) times and
  it lasted at least ``CONFIRM_SEC`` (30 s) until that pass, or when no pass follows (the end cuts
  it short, and the final state is what the user gets). Only a confirmed stretch breaks a passing
  streak; ``n_breaks`` counts those after the bug first passed;
- ``fixed_sec``: the first ``pass`` from which the bug stays passing (no confirmed break) for at
  least ``HOLD_SEC`` (120 s), or to the end; ``stable_sec``: the start of the passing streak that
  lasts to the end (None when the bug ends failing); ``first_pass_sec``: the first ``pass``;
  ``visible_ok_at_fix``: whether the visible suite passed in the ``fixed_sec`` state.

Times are seconds after the bug was sent (``t_ready``); ``fixed_session_sec`` is the same fix
from the start of the session. A pass before the bug was sent (``pass_before_sent``; the agent can
fix a bug it was not told about) is flagged and the times are clipped to 0; ``state0_pass`` flags
a bug passing in snapshot 0, the untouched workspace, which should fail every bug.

``timing_ok``: the trial has ``fix_states.json`` and its final state reproduces the verifier's
per-bug verdicts (``final_matches_verifier``); otherwise the trial is left out of the timing
metrics and every time is None. ``fixed_end`` is the end-of-session verdict as before (resolved,
with the visible suite passing; ``verifier/eval.json``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from .interactivity import eval_subtasks
from .load import Trial

HOLD_SEC = 120.0  # a fix must hold this long (or to the end)
CONFIRM_OBS = 2  # a failing stretch breaks a fix only if observed this often ...
CONFIRM_SEC = 30.0  # ... and lasting this long (or reaching the end)
KNOWN = ("pass", "fail")


def keys(t: Trial) -> dict[str, Any]:
    """The trial's identifying columns (as ``aggregate``'s rows)."""
    return {"sweep": t.sweep, "job": t.dir.parent.name, "round": t.round, "condition": t.condition,
            "task_name": t.task_name, "trial_name": t.dir.name}


def load_fix_states(t: Trial) -> dict | None:
    """``agent/fix_states.json``, or None when it is missing or unreadable."""
    try:
        d = json.loads((t.dir / "agent" / "fix_states.json").read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and isinstance(d.get("states"), list) else None


def snapshot_records(t: Trial) -> list[dict]:
    """The trial's ``snapshots`` records in ``seq`` order ([] without any)."""
    recs = t.records.get("snapshots")
    if recs is None:  # a loader that does not know the kind: read the file
        try:
            path = t.dir / "agent" / "snapshots.jsonl"
            recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.is_file() else []
        except (OSError, ValueError):
            recs = []
    return sorted((r for r in recs if isinstance(r, dict) and isinstance(r.get("seq"), int)), key=lambda r: r["seq"])


def final_matches(fs: dict | None) -> bool:
    """The replay's final state reproduces ``verifier/eval.json`` (True only when the host said so;
    a per-bug mapping must match on every bug)."""
    m = (fs or {}).get("final_matches_verifier")
    if isinstance(m, dict):
        return bool(m) and all(v is True for v in m.values())
    return m is True


@dataclass
class State:
    t: float
    seq: int | None
    final: bool
    visible_ok: bool | None
    obs: int  # times observed: itself plus the unchanged snapshots after it
    raw: dict[str, str] = field(default_factory=dict)  # sid -> replayed status
    status: dict[str, str] = field(default_factory=dict)  # sid -> pass | fail | unknown (a flaky fail: unknown)


def _status(b: Any) -> tuple[str, str]:
    """(raw, effective) status of one bug in one state."""
    if not isinstance(b, dict) or b.get("status") not in ("pass", "fail", "unknown"):
        return "unknown", "unknown"
    s = b["status"]
    return s, "unknown" if s == "fail" and b.get("rerun") == "pass" else s


def states(fs: dict | None, snaps: list[dict]) -> list[State]:
    """The replayed states in order (snapshots by ``seq``, the final state last)."""
    if not fs:
        return []
    by_seq = {r["seq"]: r for r in snaps}
    ok = [r for r in snaps if r.get("error") is None and r.get("tree")]
    changed = sorted(r["seq"] for r in ok if r.get("changed"))
    raw = [s for s in fs["states"] if isinstance(s, dict)]
    seqs = sorted(s["seq"] for s in raw if isinstance(s.get("seq"), int))
    out = []
    for s in raw:
        seq = s.get("seq") if isinstance(s.get("seq"), int) else None
        t = s.get("t")
        if not isinstance(t, (int, float)) and seq in by_seq:
            t = by_seq[seq].get("t_start")
        if not isinstance(t, (int, float)):
            continue
        obs = 1
        if seq is not None:  # unchanged snapshots until the workspace changed again
            nxt = min([q for q in seqs + changed if q > seq], default=math.inf)
            obs += sum(1 for r in ok if seq < r["seq"] < nxt and not r.get("changed"))
        st = State(float(t), seq, s.get("source") == "final" or seq is None, s.get("visible_ok"), obs)
        for sid, b in (s.get("bugs") or {}).items():
            st.raw[sid], st.status[sid] = _status(b)
        out.append(st)
    return sorted(out, key=lambda x: (x.final, math.inf if x.seq is None else x.seq, x.t))


def _breaks(sts: list[State], sid: str) -> list[int]:
    """Indexes of the first fail of each confirmed failing stretch of ``sid``."""
    seq = [s.status.get(sid, "unknown") for s in sts]
    out, i, n = [], 0, len(sts)
    while i < n:
        if seq[i] != "fail":
            i += 1
            continue
        j, obs = i, 0
        while j < n and seq[j] != "pass":
            obs += sts[j].obs if seq[j] == "fail" else 0
            j += 1
        if j == n or (obs >= CONFIRM_OBS and sts[j].t - sts[i].t >= CONFIRM_SEC):
            out.append(i)
        i = j
    return out


def bug_times(sts: list[State], sid: str) -> dict[str, Any]:
    """Absolute times (epoch seconds) of one bug: ``fixed_t``, ``stable_t``, ``first_pass_t``, the
    state index of the fix, and ``n_breaks``."""
    seq = [s.status.get(sid, "unknown") for s in sts]
    breaks = _breaks(sts, sid)
    first = next((k for k, s in enumerate(seq) if s == "pass"), None)
    fixed = stable = None
    k = first
    while k is not None:
        brk = next((b for b in breaks if b > k), None)
        if brk is None:  # passing to the end
            fixed = k if fixed is None else fixed
            stable = k
            break
        if fixed is None and sts[brk].t - sts[k].t >= HOLD_SEC:
            fixed = k
        k = next((m for m in range(brk, len(seq)) if seq[m] == "pass"), None)
    return {"fixed_i": fixed, "fixed_t": None if fixed is None else sts[fixed].t,
            "stable_t": None if stable is None else sts[stable].t,
            "first_pass_t": None if first is None else sts[first].t,
            "n_breaks": sum(1 for b in breaks if first is not None and b > first)}


def _sent_bugs(t: Trial) -> list[dict]:
    return sorted((m for m in t.records.get("inbox", []) if m.get("kind") == "hint" and m.get("subtask_id")),
                  key=lambda m: (m.get("hint_index", 0), m["t_ready"]))


def _session_t0(t: Trial, bugs: list[dict]) -> float:
    t0 = (t.episode or {}).get("session_t0")
    return t0 if t0 is not None else bugs[0]["t_ready"] if bugs else 0.0


def fix_times(t: Trial) -> dict[str, dict[str, Any]]:
    """subtask_id -> its ``fixes.csv`` row, for every bug sent in the session."""
    bugs = _sent_bugs(t)
    if not bugs:
        return {}
    t0 = _session_t0(t, bugs)
    fs = load_fix_states(t)
    ok = fs is not None and final_matches(fs)
    snaps = snapshot_records(t)
    every = states(fs, snaps)
    sts = every if ok else []
    verdicts, visible_ok, graded = eval_subtasks(t.dir)
    base = {**keys(t), "has_fix_states": fs is not None, "timing_ok": ok,
            "final_matches_verifier": None if fs is None else final_matches(fs),
            "n_states": len(every) if fs is not None else None,
            "n_snapshots": len(snaps), "n_snapshot_errors": sum(1 for r in snaps if r.get("error") is not None)}
    out = {}
    for m in bugs:
        sid, ready = m["subtask_id"], m["t_ready"]
        v = verdicts.get(sid)
        row = {**base, "msg_id": m.get("msg_id"), "hint_index": m.get("hint_index"), "subtask_id": sid,
               "urgent": bool(m.get("urgent")), "ready_session_sec": ready - t0,
               # no verdict for a bug in a graded trial (a patch that broke the build): unfixed, as the reward
               "fixed_end": (bool(v.get("resolved")) and bool(visible_ok)) if v else (False if graded else None),
               **dict.fromkeys(("fixed_sec", "fixed_session_sec", "stable_sec", "first_pass_sec", "n_breaks",
                                "visible_ok_at_fix", "pass_before_sent", "state0_pass", "final_status",
                                "n_unknown"))}
        if ok and sts:
            bt = bug_times(sts, sid)
            rel = {k: None if bt[f"{k}_t"] is None else max(0.0, bt[f"{k}_t"] - ready)
                   for k in ("fixed", "stable", "first_pass")}
            row.update(fixed_sec=rel["fixed"], stable_sec=rel["stable"], first_pass_sec=rel["first_pass"],
                       fixed_session_sec=None if rel["fixed"] is None else ready + rel["fixed"] - t0,
                       n_breaks=bt["n_breaks"],
                       visible_ok_at_fix=None if bt["fixed_i"] is None else sts[bt["fixed_i"]].visible_ok,
                       pass_before_sent=any(s.raw.get(sid) == "pass" and s.t < ready for s in sts),
                       state0_pass=any(s.seq == 0 and s.raw.get(sid) == "pass" for s in sts),
                       final_status=sts[-1].raw.get(sid, "unknown"),
                       n_unknown=sum(1 for s in sts if s.raw.get(sid, "unknown") == "unknown"))
        out[sid] = row
    return out


def bug_fix_rows(t: Trial) -> list[dict[str, Any]]:
    """One ``fixes.csv`` row per bug sent in the session (none for a session without bugs)."""
    return list(fix_times(t).values())


def state_at(t: Trial, t_session: float) -> dict[str, str]:
    """subtask_id -> the bug's status (``pass``, ``fail`` or ``unknown``) at ``t_session`` seconds
    into the session: in the last state at or before then whose status for the bug is known (a
    state that did not build carries no information, and a flaky fail stands for the pass before
    it); ``unknown`` when there is none. {} without usable fix times (``timing_ok`` false)."""
    fs = load_fix_states(t)
    if fs is None or not final_matches(fs):
        return {}
    bugs = _sent_bugs(t)
    cut = _session_t0(t, bugs) + t_session
    sts = states(fs, snapshot_records(t))
    out = {sid: "unknown" for sid in [m["subtask_id"] for m in bugs] + [s for st in sts for s in st.status]}
    for st in sorted((st for st in sts if st.t <= cut), key=lambda st: st.t):
        out.update({sid: s for sid, s in st.status.items() if s in KNOWN})
    return out
