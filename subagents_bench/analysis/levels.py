"""The levels experiment: four cumulative levels of user messages on the same bugs, and its three
main numbers (ETA on time, questions answered, urgency).

**Levels.** Parsed from the condition name: ``<setting>_baseline`` is L0 (bugs only), ``_eta`` L1
(+ status checks), ``_questions`` L2 (+ questions about the code), ``_urgent`` L3 (+ an urgent
bug); the setting is the rest (``single``, ``manager_background``, ``background_hinted``). Other
conditions have neither and are left out of ``urgency.csv`` and ``levels.csv``. Everything groups
by sweep too, so two runs of a level in different sweeps (the two baselines) never pool. Times
are seconds after the message was sent unless named ``*_session_sec``.

**Questions** (columns added to the question rows of ``messages.csv``, from ``answers.json``
version 2, ``subagents-bench grade-answers``): ``correct_sec``, until the first reply the judge
calls correct (replies are judged in time order up to the first correct one); ``correct_known``,
whether every reply up to it has a verdict (an unread verdict, or a reply not graded yet, leaves
it unknown); ``first_verdict`` and ``first_reply_correct``, the first reply's (today's
``correct``); ``wrong_before_correct``, a reply judged wrong before the first correct one (None
when never correct); ``n_judged``, ``n_wrong``; ``response_sec``, from delivery to the first
correct reply (``delivery_sec`` + ``response_sec`` = ``correct_sec``); ``n_elsewhere``,
``elsewhere_correct`` and ``elsewhere_correct_sec``: replies to another id citing the question,
judged apart and never counted. An older ``answers.json`` (no version) judged only the first reply:
correct there is ``correct_sec`` = its time; a first reply not correct with later replies leaves it
unknown. Shares within 30/60/120/300 s are over every question asked: one never answered
correctly, also when the session ended idle with it open, counts as never (every question is sent
at least 440 s before the deadline, so none is censored).

**Urgency** (``urgency.csv``: one row per urgent bug of a valid L3 session and twin level). Its
twins are the same subtask sent at the same time (within 1 s) in the valid sessions of the same
sweep, task and setting at L2 (the comparison) and L0 (descriptive: all three kinds of message
differ). Fix times come from the replayed snapshots (``fixes.fix_times``); a session without them,
or whose final state does not reproduce the verifier (``timing_ok`` false), has no timing and pairs
with nothing. Per urgent bug, against its twins:

- ``order``: per pair of sessions, over the earlier bugs failing at the urgent bug's send time in
  both (``fixes.state_at``; ``unknown`` counts as failing unless the bug's fix time is before),
  1 if the urgent bug was fixed before the earlier bug, 0 after, 0.5 if neither was fixed (or at
  the same time), the earlier bug by its first fix after the send time (``fixes.fixed_after``
  when its ``fixed_sec`` fix broke before then); the mean over those bugs, then over pairs;
  ``twin_order`` the same in the twins;
- ``fixed_240``/``fixed_480``: fixed within 240 s (before the next bug) and 480 s
  (``twin_fixed_*``: the share of the twins);
- ``win_rate`` over the urgent x twin session pairs: 1 if the urgent bug was fixed more than 30 s
  sooner or only it was fixed, 0 for the reverse, 0.5 within 30 s or neither; ``n_decisive``
  the pairs that are not 0.5;
- ``start_sec``: the earlier of a dispatch naming the bug's files or hidden tests
  (``dispatch_sec``, as in ``hints.csv``) and the first snapshot after it was sent whose diff of
  its target files differs from the last snapshot before it (``change_sec``, at the snapshot's
  ``t_end``); reading a file is not a start;
- preemption: of the tool calls (any agent) in the 240 s after delivery that touch the bug's
  files or an earlier bug's, the share touching the bug's (``preempt_touch_share``), and the same
  for edits (``preempt_edit_share``); the manager has no edit tools, so it can only dispatch;
- cost: the other bugs' fixed share and censored median ``fixed_sec`` (never fixed: infinitely
  late; NaN when more than half never were), against the same bugs in the twins.

**Main numbers** (``levels.csv``: one row per sweep, setting and level, valid sessions), each with
a 95% interval from a bootstrap over sessions stratified by task (``stats.cluster_bootstrap``;
answers and bugs cluster by session):

- ETA, ``eta_on_time``: the on-time share of the structured estimates scored against the fix
  times (``scored`` in ``estimates.csv``: eligible, sent before the fix), each bug weighted once
  (its estimates share its weight); with the text estimates (the regex fallback) too
  ``eta_on_time_any_via``, arrival estimates only ``eta_on_time_arrival``;
- questions, ``q_correct_60s``: the share of all questions asked answered correctly within 60 s
  (NaN while an answered question is ungraded);
- urgency (L3), ``urgent_win_rate``: the win rate over every urgent x L2-twin session pair of the
  sweep and setting (L3 and L2 sessions resampled within each task).

Descriptive, without intervals: sessions (``n_timed`` with fix times), bugs fixed (by
``fixed_sec``, its censored median, and at the end), estimates and bugs estimated, status-check
pairs answered within 120 s and answered right, the other question thresholds, the first reply
correct within 60 s, wrong before correct, the delivery and response medians, and at L3 the
pairs, order and shares fixed within 240/480 s of the urgent bugs and of their L2 twins (each
twin once). Every row has every column (NaN where a level has no such message).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from . import fixes as fixes_mod
from .interactivity import _median_censored, _tokens, eval_subtasks, hint_rows, tool_touches
from .load import Trial
from .stats import cluster_bootstrap

LEVELS = {"baseline": "L0", "eta": "L1", "questions": "L2", "urgent": "L3"}
LEVEL_RE = re.compile(rf"^(?P<setting>.+)_(?P<suffix>{'|'.join(LEVELS)})$")
SETTINGS = ("single", "manager_background", "background_hinted")
THRESHOLDS = (30, 60, 120, 300)
URGENT_WITHIN = (240, 480)
TIE_SEC = 30.0  # fix times closer than this are a tie in the win rate
PREEMPT_SEC = 240.0  # the window after delivery for preemption
SAME_SEND_SEC = 1.0  # a twin is sent at the same time, within this
TWIN_LEVELS = {"L2": "the comparison", "L0": "descriptive"}
QUESTION_COLUMNS = ("correct_sec", "correct_known", "first_verdict", "first_reply_correct", "wrong_before_correct",
                    "n_judged", "n_wrong", "response_sec", "n_elsewhere", "elsewhere_correct", "elsewhere_correct_sec")
MAIN = ("eta_on_time", "q_correct_60s", "urgent_win_rate")  # each with _lo and _hi in levels.csv
Q_NUMBERS = (*(f"q_correct_{s}s" for s in THRESHOLDS), "q_correct_60s_lo", "q_correct_60s_hi", "q_first_correct_60s",
             "q_wrong_before_correct_frac", "q_delivery_median_sec", "q_response_median_sec")
U_NUMBERS = ("urgent_win_rate", "urgent_win_rate_lo", "urgent_win_rate_hi", "urgent_n_pairs", "urgent_n_decisive",
             "urgent_order", "twin_order", *(f"{w}_fixed_{s}" for s in URGENT_WITHIN for w in ("urgent", "twin")))


def setting_level(condition: str | None) -> tuple[str | None, str | None]:
    """(setting, level) of a levels condition name, else (None, None)."""
    m = LEVEL_RE.match(condition or "")
    return (m["setting"], LEVELS[m["suffix"]]) if m else (None, None)


# --- questions ---------------------------------------------------------------------------------------


def _answers_doc(t: Trial) -> dict:
    try:
        doc = json.loads((t.dir / "agent" / "answers.json").read_text())
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def verdict(entry: Any) -> str | None:
    """``correct``, ``wrong``, ``none`` (not an answer) or None (unread or absent); an entry of an
    older ``answers.json`` has only ``correct`` (false there covered both wrong and not an answer:
    read as wrong)."""
    if not isinstance(entry, dict):
        return None
    v = entry.get("verdict")
    if v in ("correct", "wrong", "none"):
        return v
    c = entry.get("correct")
    return None if c is None else "correct" if c is True else "wrong"


def question_columns(t: Trial) -> dict[int, dict[str, Any]]:
    """msg_id -> the ``QUESTION_COLUMNS`` of each question of an episode with replies (empty otherwise)."""
    ep, r = t.episode or {}, t.records
    if not ((ep.get("config") or {}).get("interactive") or {}).get("replies"):
        return {}
    questions = [m for m in r.get("inbox", []) if m.get("kind") == "question" and m.get("msg_id") is not None]
    if not questions:
        return {}
    doc = _answers_doc(t)
    v2 = doc.get("version") == 2
    first, judged, elsewhere = (doc.get(k) or {} for k in ("answers", "replies", "elsewhere"))
    by_msg: dict[int, list[dict]] = {}
    for x in sorted(r.get("replies", []), key=lambda x: x["t"]):
        by_msg.setdefault(x["to"], []).append(x)
    out = {}
    for m in questions:
        key, t0, mine = str(m["msg_id"]), m["t_ready"], by_msg.get(m["msg_id"], [])
        if v2:  # judged in time order up to the first correct one; a missing time is its reply's
            seq = [(e.get("t", mine[i]["t"] if i < len(mine) else None), verdict(e))
                   for i, e in enumerate(judged.get(key) or []) if isinstance(e, dict)]
        else:  # only the first reply was judged
            seq = [(mine[0]["t"], verdict(first.get(key)))] if mine else []
        k = next((i for i, (_, v) in enumerate(seq) if v == "correct"), None)
        if k is not None:
            known = all(v is not None for _, v in seq[:k]) and seq[k][0] is not None
        else:
            known = len(seq) >= len(mine) and all(v is not None for _, v in seq)
        t_ok = seq[k][0] if known and k is not None else None
        fv = verdict(first.get(key)) if mine else None
        el = [e for e in elsewhere.get(key) or [] if isinstance(e, dict)]
        el_ok = [e["t"] for e in el if verdict(e) == "correct" and e.get("t") is not None]
        el_known = bool(el_ok) or all(verdict(e) is not None for e in el)
        out[m["msg_id"]] = {
            "correct_sec": None if t_ok is None else t_ok - t0, "correct_known": known,
            "first_verdict": fv, "first_reply_correct": None if fv is None else fv == "correct",
            "wrong_before_correct": any(v == "wrong" for _, v in seq[:k]) if t_ok is not None else None,
            "n_judged": len(seq), "n_wrong": sum(v == "wrong" for _, v in seq),
            "response_sec": t_ok - m["t_delivered"] if t_ok is not None and m.get("t_delivered") is not None else None,
            "n_elsewhere": len(el), "elsewhere_correct": bool(el_ok) if el and el_known else None,
            "elsewhere_correct_sec": min(el_ok) - t0 if el_ok else None}
    return out


# --- bugs: fix times, start, preemption ------------------------------------------------------------


def file_diffs(patch: str, files: set[str] | None = None) -> dict[str, str]:
    """path -> its section of a ``git diff`` (its ``diff --git`` line up to the next); only
    ``files`` when given."""
    out: dict[str, str] = {}
    cur, buf = None, []
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if cur is not None and (files is None or cur in files):
                out[cur] = "".join(buf)
            m = re.match(r"diff --git a/(.*?) b/(.*)$", line.rstrip("\n"))
            cur, buf = (m.group(2) if m else line.strip()), [line]
        elif cur is not None:
            buf.append(line)
    if cur is not None and (files is None or cur in files):
        out[cur] = "".join(buf)
    return out


def _snapshots(t: Trial) -> list[dict]:
    """The snapshot records (``agent/snapshots.jsonl``; loaded with the records once they are a kind)."""
    if "snapshots" in t.records:
        return t.records["snapshots"]
    path = t.dir / "agent" / "snapshots.jsonl"
    try:
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except (OSError, ValueError):
        return []


def snapshot_states(t: Trial, files: set[str]) -> list[tuple[float, dict[str, str]]]:
    """(``t_end``, path -> diff section of ``files``) of every snapshot that changed the workspace
    and wrote its patch, in seq order; a missing patch file is left out."""
    out = []
    for s in sorted(_snapshots(t), key=lambda s: s.get("seq", 0)):
        if not s.get("changed") or not s.get("patch") or s.get("error") or s.get("t_end") is None:
            continue
        try:
            patch = (t.dir / "agent" / s["patch"]).read_text(errors="replace")
        except OSError:
            continue
        out.append((s["t_end"], file_diffs(patch, files)))
    return out


def change_after(states: list[tuple[float, dict[str, str]]], t0: float, files: list[str]) -> float | None:
    """Seconds after ``t0`` until the first snapshot whose diff of ``files`` differs from the last
    snapshot at or before ``t0`` (none before it: the clean tree)."""
    before = [d for t, d in states if t <= t0]
    ref = before[-1] if before else {}
    for t, d in states:
        if t > t0 and any(d.get(f) != ref.get(f) for f in files):
            return t - t0
    return None


def _basename_ok(hints: list[dict]) -> set[str]:
    """Hinted target files whose bare basename names no other hinted target file (as hints.csv)."""
    files = list(dict.fromkeys(f for h in hints for f in h.get("target_files") or []))
    count: dict[str, int] = {}
    for f in files:
        count[_tokens(f)[1]] = count.get(_tokens(f)[1], 0) + 1
    return {f for f in files if count[_tokens(f)[1]] == 1}


def preemption(tools: list[dict], t_from: float, own: list[str], earlier: list[str],
               basename_ok: set[str]) -> dict[str, int]:
    """Tool calls starting in ``PREEMPT_SEC`` from ``t_from`` that touch (edit) ``own`` or
    ``earlier`` files, and of them those on ``own`` (a call touching both counts as on ``own``)."""
    n = {"preempt_n_touch": 0, "preempt_n_touch_own": 0, "preempt_n_edit": 0, "preempt_n_edit_own": 0}
    for x in tools:
        if not t_from <= x["t_start"] < t_from + PREEMPT_SEC:
            continue
        t_own, e_own = tool_touches(x, own, basename_ok)
        t_er, e_er = tool_touches(x, earlier, basename_ok) if earlier else (False, False)
        n["preempt_n_touch"] += t_own or t_er
        n["preempt_n_touch_own"] += t_own
        n["preempt_n_edit"] += e_own or e_er
        n["preempt_n_edit_own"] += e_own
    return n


def _failing(status: str | None, fix_at: float | None, t: float) -> bool:
    """A bug fails at ``t``: its state there fails, or is unknown (e.g. a half-done edit broke the
    import) while its fix time is not before ``t``."""
    return status == "fail" or (status != "pass" and (fix_at is None or fix_at > t))


def bug_rows(t: Trial, keys: dict[str, Any], valid: bool, fixes: dict[str, dict] | None = None,
             hints: list[dict] | None = None) -> list[dict[str, Any]]:
    """One row per bug sent in an interactive episode (none otherwise): its fix times
    (``fixes``: ``fixes.fix_times(t)`` unless given), start, preemption, the earlier bugs
    failing when it was sent (``open_before``, a list) and their first fix after then
    (``open_fixed_session_sec``: subtask id -> seconds into the session or None; ``fixed_sec``'s
    fix unless it was at or before the send time). ``hints``: the trial's ``hint_rows``."""
    ep, r = t.episode or {}, t.records
    inbox = sorted((x for x in r.get("inbox", []) if x.get("kind") == "hint"), key=lambda x: x["hint_index"])
    if (ep.get("config") or {}).get("interactive") is None or not inbox:
        return []
    session_t0 = ep.get("session_t0", inbox[0]["t_ready"])
    t_end = ep.get("t_end") or math.inf
    sent = [h for h in inbox if h["t_ready"] <= t_end]
    fixes = fixes_mod.fix_times(t) if fixes is None else fixes
    timed = bool(fixes) and all((f or {}).get("timing_ok", True) is not False for f in fixes.values())
    if hints is None:
        hints = hint_rows(t, keys, valid)
    dispatch = {h["subtask_id"]: h.get("dispatch_sec") for h in hints}
    setting, level = setting_level(keys.get("condition"))
    verdicts, visible_ok, _ = eval_subtasks(t.dir)
    all_files = {f for h in sent for f in h.get("target_files") or []}
    states = snapshot_states(t, all_files)
    basename_ok = _basename_ok(sent)
    tools = sorted(r.get("tools", []), key=lambda x: x["t_start"])
    fix_at: dict[str, float | None] = {}
    rows = []
    for h in sent:
        sid, t0 = h["subtask_id"], h["t_ready"]
        f = fixes.get(sid) or {}
        fixed_sec = f.get("fixed_sec") if timed else None
        v = verdicts.get(sid)
        fixed_end = f.get("fixed_end")
        if fixed_end is None:
            fixed_end = (bool(v.get("resolved")) and bool(visible_ok)) if v else (False if valid else None)
        files = list(dict.fromkeys(h.get("target_files") or []))
        change = change_after(states, t0, files) if states and files else None
        starts = [x for x in (dispatch.get(sid), change) if x is not None and not pd.isna(x)]
        row = {**keys, "valid": valid, "setting": setting, "level": level, "msg_id": h.get("msg_id"),
               "hint_index": h["hint_index"], "subtask_id": sid, "urgent": bool(h.get("urgent")),
               "target_files": files, "t_ready_sec": t0 - session_t0,
               "delivery_sec": None if h.get("t_delivered") is None else h["t_delivered"] - t0,
               "timed": timed and sid in fixes, "fixed_sec": fixed_sec,
               "pass_before_sent": f.get("pass_before_sent") if timed else None, "fixed_end": fixed_end,
               "dispatch_sec": dispatch.get(sid), "change_sec": change, "start_sec": min(starts) if starts else None}
        earlier = [e for e in sent if e["hint_index"] < h["hint_index"]]
        own = set(files)
        other = list(dict.fromkeys(x for e in earlier for x in e.get("target_files") or [] if x not in own))
        if h.get("t_delivered") is not None and earlier:
            row.update(preemption(tools, h["t_delivered"], files, other, basename_ok))
        fix_at[sid] = None if fixed_sec is None else row["t_ready_sec"] + fixed_sec
        rows.append(row)
    for row in rows:
        earlier = [x["subtask_id"] for x in rows if x["hint_index"] < row["hint_index"]]
        row["open_before"] = row["open_fixed_session_sec"] = None
        if row["timed"] and earlier:
            sent_t = row["t_ready_sec"]
            at = fixes_mod.state_at(t, sent_t) or {}
            row["open_before"] = [s for s in earlier if _failing(at.get(s), fix_at.get(s), sent_t)]
            # their first fix after it was sent: fixed_sec's, unless that fix broke before then (the next one)
            old = [s for s in row["open_before"] if fix_at.get(s) is not None and fix_at[s] <= sent_t]
            after = fixes_mod.fixed_after(t, sent_t) if old else {}
            row["open_fixed_session_sec"] = {s: after.get(s) if s in old else fix_at.get(s)
                                             for s in row["open_before"]}
    return rows


# --- urgency -----------------------------------------------------------------------------------------


def win(fu: float | None, fw: float | None) -> float:
    """The urgent bug's fix time ``fu`` against its twin's ``fw`` (None: never fixed)."""
    if fu is None or fw is None:
        return 0.5 if fu is None and fw is None else 1.0 if fw is None else 0.0
    return 1.0 if fw - fu > TIE_SEC else 0.0 if fu - fw > TIE_SEC else 0.5


def _fix_at(b: dict) -> float:
    return math.inf if b["fixed_sec"] is None else b["t_ready_sec"] + b["fixed_sec"]


def order_score(session: dict[str, dict], sid: str, earlier: list[str]) -> float | None:
    """Mean over ``earlier`` of 1 if ``sid`` was fixed before it, 0 after, 0.5 if neither was fixed
    (or at the same time); None without earlier bugs. An earlier bug failing when ``sid`` was sent
    counts by its first fix after then (``open_fixed_session_sec``), not by a fix that broke before."""
    if not earlier:
        return None
    mine, after = _fix_at(session[sid]), session[sid].get("open_fixed_session_sec")
    other = [_fix_at(session[e]) if after is None or e not in after else math.inf if after[e] is None else after[e]
             for e in earlier]
    return float(np.mean([1.0 if mine < o else 0.0 if mine > o else 0.5 for o in other]))


def _session_key(b: dict) -> tuple:
    return b.get("sweep"), b.get("job"), b.get("trial_name")


def _sessions(bugs: list[dict]) -> dict[tuple, dict[str, dict]]:
    """Valid levels sessions: (sweep, job, trial) -> subtask id -> bug row."""
    out: dict[tuple, dict[str, dict]] = {}
    for b in bugs:
        if b.get("valid") and b.get("level") is not None:
            out.setdefault(_session_key(b), {})[b["subtask_id"]] = b
    return out


def twins(sessions: dict[tuple, dict[str, dict]], b: dict, level: str) -> list[tuple[tuple, dict]]:
    """(session, bug) of ``b``'s twins at ``level``: same sweep, task, setting and subtask, sent at
    the same time."""
    out = []
    for key, s in sessions.items():
        w = s.get(b["subtask_id"])
        if (w is not None and w["level"] == level and w["setting"] == b["setting"] and w["sweep"] == b["sweep"]
                and w["task_name"] == b["task_name"] and abs(w["t_ready_sec"] - b["t_ready_sec"]) <= SAME_SEND_SEC):
            out.append((key, w))
    return out


def _share(xs: list) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else math.nan


def _ratio(num: int, den: int) -> float:
    return num / den if den else math.nan


def urgency_rows(bugs: list[dict]) -> list[dict[str, Any]]:
    """One row per urgent bug of a valid L3 session and twin level (L2, L0); see the module doc."""
    sessions = _sessions(bugs)
    return [_urgency_row(sessions, s, u, level) for s in sessions.values() for u in s.values()
            if u["urgent"] and u["level"] == "L3" for level in TWIN_LEVELS]


def _urgency_row(sessions: dict, s: dict[str, dict], u: dict, level: str) -> dict[str, Any]:
    tw = twins(sessions, u, level)
    timed = [(k, w) for k, w in tw if w["timed"]] if u["timed"] else []
    pairs = [win(u["fixed_sec"], w["fixed_sec"]) for _, w in timed]
    orders, twin_orders, n_bugs = [], [], 0
    for k, w in timed:
        both = [e for e in u["open_before"] or [] if e in set(w["open_before"] or []) and e in sessions[k]]
        if both:
            orders.append(order_score(s, u["subtask_id"], both))
            twin_orders.append(order_score(sessions[k], w["subtask_id"], both))
            n_bugs += len(both)
    others = [b for sid, b in s.items() if sid != u["subtask_id"] and b["timed"]] if u["timed"] else []
    twin_others = [sessions[k][b["subtask_id"]] for k, _ in timed for b in others if b["subtask_id"] in sessions[k]]
    keep = ("sweep", "job", "round", "condition", "task_name", "trial_name", "setting", "level", "msg_id",
            "subtask_id", "t_ready_sec", "timed", "fixed_sec", "dispatch_sec", "change_sec", "start_sec")
    row = {k: u.get(k) for k in keep}
    row.update(twin_level=level, n_twins=len(tw), n_twins_timed=len(timed),
               twin_fixed_sec_median=_median_censored(pd.Series([w["fixed_sec"] for _, w in timed], dtype=float))
               if timed else math.nan)
    for sec in URGENT_WITHIN:
        row[f"fixed_{sec}"] = None if not u["timed"] else u["fixed_sec"] is not None and u["fixed_sec"] <= sec
        row[f"twin_fixed_{sec}"] = _share([w["fixed_sec"] is not None and w["fixed_sec"] <= sec for _, w in timed])
    row.update(win_rate=_share(pairs), n_pairs=len(pairs), n_decisive=sum(p != 0.5 for p in pairs),
               order=_share(orders), twin_order=_share(twin_orders), n_order_pairs=len(orders), n_order_bugs=n_bugs,
               twin_start_sec_median=_median_censored(pd.Series([w["start_sec"] for _, w in tw], dtype=float))
               if tw else math.nan)
    for kind in ("touch", "edit"):
        n, own = u.get(f"preempt_n_{kind}"), u.get(f"preempt_n_{kind}_own")
        row[f"preempt_n_{kind}"] = n
        row[f"preempt_{kind}_share"] = _ratio(own, n) if n is not None else math.nan
        tn = sum(w.get(f"preempt_n_{kind}") or 0 for _, w in tw)
        row[f"twin_preempt_{kind}_share"] = _ratio(sum(w.get(f"preempt_n_{kind}_own") or 0 for _, w in tw), tn)
    row.update(others_fixed_frac=_share([b["fixed_sec"] is not None for b in others]),
               twin_others_fixed_frac=_share([b["fixed_sec"] is not None for b in twin_others]),
               others_fixed_sec_median=_median_censored(pd.Series([b["fixed_sec"] for b in others], dtype=float))
               if others else math.nan,
               twin_others_fixed_sec_median=_median_censored(pd.Series([b["fixed_sec"] for b in twin_others],
                                                                       dtype=float)) if twin_others else math.nan)
    return row


def pooled_win_rate(sessions: list[tuple[str, str, dict[str, dict]]]) -> float:
    """Win rate over every (urgent L3 bug, timed L2 twin) pair among ``sessions`` ((level, task,
    subtask id -> bug row), a session drawn twice counting twice)."""
    l2 = [s for lv, _, s in sessions if lv == "L2"]
    scores = []
    for lv, task, s in sessions:
        if lv != "L3":
            continue
        for u in s.values():
            if not (u["urgent"] and u["timed"]):
                continue
            for w2 in l2:
                w = w2.get(u["subtask_id"])
                if (w is not None and w["timed"] and w["task_name"] == task
                        and abs(w["t_ready_sec"] - u["t_ready_sec"]) <= SAME_SEND_SEC):
                    scores.append(win(u["fixed_sec"], w["fixed_sec"]))
    return float(np.mean(scores)) if scores else math.nan


# --- the levels table --------------------------------------------------------------------------------


def _with_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Valid rows of levels conditions, with setting, level and a session id (job/trial)."""
    if df.empty or "condition" not in df:
        return pd.DataFrame()
    d = df[df["valid"].fillna(False).astype(bool)].copy() if "valid" in df else df.copy()
    parsed = [setting_level(c) for c in d["condition"]]
    d["setting"], d["level"] = [p[0] for p in parsed], [p[1] for p in parsed]
    d["_session"] = d["job"].astype(str) + "/" + d["trial_name"].astype(str)
    return d[d["level"].notna()]


def _bool(x: Any) -> bool | None:
    """A flag read back from a table: None when missing (None, NaN)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return x is True or x == 1 or (isinstance(x, str) and x.lower() == "true")


def _flag(s: pd.Series) -> pd.Series:
    return s.map(lambda x: _bool(x) is True)


def scored_estimates(est: pd.DataFrame) -> pd.DataFrame:
    """Estimates scored against the fix times (``scored``; without that column: an on-time
    verdict, eligible, not stale)."""
    if est.empty or "on_time" not in est:
        return est.iloc[0:0]
    m = est["on_time"].notna()
    if "scored" in est:
        return est[m & _flag(est["scored"])]
    if "eligible" in est:
        m &= _flag(est["eligible"])
    if "stale" in est:
        m &= ~_flag(est["stale"])
    return est[m]


def _bug_col(est: pd.DataFrame) -> str | None:
    return next((c for c in ("subtask_id", "msg_id", "to") if c in est), None)


def eta_clusters(est: pd.DataFrame, sessions: list[str]) -> list[tuple[float, int]]:
    """Per session: (sum over its bugs of the bug's on-time share, number of bugs), so a ratio of
    sums weights each bug once."""
    if est.empty:
        return [(0.0, 0) for _ in sessions]
    col = _bug_col(est)
    out = []
    for s in sessions:
        g = est[est["_session"] == s]
        if col is None or g.empty:
            out.append((float(_flag(g["on_time"]).sum()) if len(g) else 0.0, len(g)))
            continue
        shares = _flag(g["on_time"]).astype(float).groupby(g[col].astype(str)).mean()
        out.append((float(shares.sum()), len(shares)))
    return out


def _sum_ratio(cl: list[tuple[float, int]]) -> float:
    den = sum(c[1] for c in cl)
    return sum(c[0] for c in cl) / den if den else math.nan


def _boot(clusters: list, stat: Callable[[list], float], strata: list) -> tuple[float, float, float]:
    return cluster_bootstrap(clusters, stat, strata) if clusters else (math.nan, math.nan, math.nan)


def level_table(episodes: pd.DataFrame, messages: pd.DataFrame, estimates: pd.DataFrame,
                status_pairs: pd.DataFrame, bugs: list[dict], urgency: pd.DataFrame) -> pd.DataFrame:
    """One row per (sweep, setting, level) of valid levels sessions: the three main numbers with
    95% intervals and descriptive numbers (see the module doc)."""
    eps = _with_levels(episodes)
    if eps.empty:
        return pd.DataFrame()
    msgs, est, pairs = _with_levels(messages), _with_levels(estimates), _with_levels(status_pairs)
    q_all = msgs[msgs["kind"] == "question"] if len(msgs) else msgs
    sessions = _sessions(bugs)
    bug_df = pd.DataFrame([b for s in sessions.values() for b in s.values()])
    if len(bug_df):
        bug_df["_session"] = bug_df["job"].astype(str) + "/" + bug_df["trial_name"].astype(str)
    rows = []
    for (sweep, setting, level), g in eps.groupby(["sweep", "setting", "level"], sort=False):
        ids = list(dict.fromkeys(g["_session"]))
        task = dict(zip(g["_session"], g["task_name"]))
        strata = [task[s] for s in ids]
        row: dict[str, Any] = {"sweep": sweep, "setting": setting, "level": level,
                               "condition": ";".join(dict.fromkeys(g["condition"])),
                               "n_sessions": len(ids), "n_tasks": g["task_name"].nunique()}

        def mine(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            return df[(df["sweep"] == sweep) & (df["setting"] == setting) & (df["level"] == level)]

        # bugs
        b = mine(bug_df)
        bt = b[b["timed"].astype(bool)] if len(b) else b
        row.update(n_timed=int(bt["_session"].nunique()) if len(bt) else 0, n_bugs=len(b),
                   bugs_fixed_frac=float(bt["fixed_sec"].notna().mean()) if len(bt) else math.nan,
                   bugs_fixed_sec_median=_median_censored(pd.to_numeric(bt["fixed_sec"], errors="coerce"))
                   if len(bt) else math.nan,
                   bugs_fixed_end_frac=_share(list(b["fixed_end"])) if len(b) else math.nan)
        # ETA
        e = mine(est)
        se = scored_estimates(e)
        structured = se[se["via"] == "structured"] if "via" in se else se
        row["eta_on_time"], row["eta_on_time_lo"], row["eta_on_time_hi"] = _boot(
            eta_clusters(structured, ids), _sum_ratio, strata)
        col = _bug_col(structured)
        row.update(eta_n_estimates=len(structured),
                   eta_n_bugs=int(structured.groupby(["_session", col]).ngroups) if col and len(structured) else 0,
                   eta_on_time_any_via=_sum_ratio(eta_clusters(se, ids)),
                   eta_n_text=int((se["via"] != "structured").sum()) if "via" in se else 0,
                   eta_on_time_arrival=_sum_ratio(eta_clusters(structured[structured["source"] == "arrival"], ids))
                   if "source" in structured else math.nan,
                   eta_bugs_estimated_frac=_ratio(int(e.groupby(["_session", _bug_col(e)]).ngroups), len(b))
                   if len(e) and _bug_col(e) and len(b) else math.nan)
        p = mine(pairs)
        row.update(status_n_pairs=len(p),
                   status_answered_frac=float(_flag(p["answered"]).mean()) if len(p) and "answered" in p else math.nan,
                   status_right_frac=_share([_bool(x) for x in p["right"]]) if len(p) and "right" in p else math.nan)
        # questions
        q = mine(q_all)
        row.update(_question_numbers(q, ids, strata))
        # urgency: L3 against its L2 twins
        row.update(_urgency_numbers(sessions, urgency, sweep, setting, level))
        rows.append(row)
    out = pd.DataFrame(rows)
    rank = {s: i for i, s in enumerate(SETTINGS)}
    order = sorted(range(len(out)), key=lambda i: (str(out.iloc[i]["sweep"]),
                                                   rank.get(out.iloc[i]["setting"], len(rank)),
                                                   str(out.iloc[i]["setting"]), out.iloc[i]["level"]))
    return out.iloc[order].reset_index(drop=True)


def _question_numbers(q: pd.DataFrame, ids: list[str], strata: list) -> dict[str, Any]:
    """Shares of the questions asked answered correctly within each threshold (NaN while one is
    unknown), the main one with its interval, and the descriptive ones."""
    out: dict[str, Any] = {"q_n": len(q), **dict.fromkeys(Q_NUMBERS, math.nan), "q_n_elsewhere_correct": 0}
    known = bool(len(q)) and "correct_known" in q and bool(_flag(q["correct_known"]).all())
    secs = pd.to_numeric(q["correct_sec"], errors="coerce").fillna(math.inf) if known else None
    for sec in THRESHOLDS:
        out[f"q_correct_{sec}s"] = float((secs <= sec).mean()) if known else math.nan
    if known:
        clusters = [(int((secs[q["_session"] == s] <= 60).sum()), int((q["_session"] == s).sum())) for s in ids]
        out["q_correct_60s"], out["q_correct_60s_lo"], out["q_correct_60s_hi"] = _boot(clusters, _sum_ratio, strata)
    if not len(q):
        return out
    answered = q[q["reply_sec"].notna()]
    first_known = "first_reply_correct" in q and bool(answered["first_reply_correct"].notna().all())
    out["q_first_correct_60s"] = float((_flag(q["first_reply_correct"])
                                        & (pd.to_numeric(q["reply_sec"], errors="coerce") <= 60)).mean()) \
        if first_known else math.nan
    ok = q[q["correct_sec"].notna()] if "correct_sec" in q else q.iloc[0:0]
    out.update(q_wrong_before_correct_frac=float(_flag(ok["wrong_before_correct"]).mean()) if len(ok) else math.nan,
               q_delivery_median_sec=_median_censored(q["delivery_sec"]),
               q_response_median_sec=_median_censored(q["response_sec"]) if "response_sec" in q else math.nan,
               q_n_elsewhere_correct=int(_flag(q["elsewhere_correct"]).sum()) if "elsewhere_correct" in q else 0)
    return out


def _urgency_numbers(sessions: dict[tuple, dict[str, dict]], urgency: pd.DataFrame, sweep: str, setting: str,
                     level: str) -> dict[str, Any]:
    """L3: the win rate with its interval (L3 and L2 sessions resampled within each task), pairs,
    order and the shares fixed within 240/480 s of the urgent bugs and of their L2 twins (each once)."""
    keys = ("urgent_win_rate", "urgent_win_rate_lo", "urgent_win_rate_hi")
    out: dict[str, Any] = dict.fromkeys(U_NUMBERS, math.nan)
    if level != "L3":
        return out
    pool, urgent, twin = [], [], {}
    for s in sessions.values():
        b = next(iter(s.values()))  # every bug row of a session has its keys
        if b["sweep"] == sweep and b["setting"] == setting and b["level"] in ("L2", "L3"):
            pool.append((b["level"], b["task_name"], s))
            if b["level"] == "L3":
                urgent += [u for u in s.values() if u["urgent"] and u["timed"]]
    for u in urgent:
        twin.update({(k, w["subtask_id"]): w for k, w in twins(sessions, u, "L2") if w["timed"]})
    out.update(zip(keys, _boot(pool, pooled_win_rate, [(lv, task) for lv, task, _ in pool])))
    for sec in URGENT_WITHIN:
        out[f"urgent_fixed_{sec}"] = _share([u["fixed_sec"] is not None and u["fixed_sec"] <= sec for u in urgent])
        out[f"twin_fixed_{sec}"] = _share([w["fixed_sec"] is not None and w["fixed_sec"] <= sec
                                           for w in twin.values()])
    u = urgency[(urgency["sweep"] == sweep) & (urgency["setting"] == setting) & (urgency["twin_level"] == "L2")] \
        if len(urgency) else urgency
    w = u["n_order_pairs"].astype(float) if len(u) else pd.Series(dtype=float)
    out.update(urgent_n_pairs=int(u["n_pairs"].sum()) if len(u) else 0,
               urgent_n_decisive=int(u["n_decisive"].sum()) if len(u) else 0,
               urgent_order=float((u["order"].fillna(0) * w).sum() / w.sum()) if w.sum() else math.nan,
               twin_order=float((u["twin_order"].fillna(0) * w).sum() / w.sum()) if w.sum() else math.nan)
    return out


# --- printing ----------------------------------------------------------------------------------------


def _ci(row: pd.Series, col: str) -> str:
    x, lo, hi = row.get(col), row.get(f"{col}_lo"), row.get(f"{col}_hi")
    if x is None or pd.isna(x):
        return "-"
    return f"{x:.2f}" if pd.isna(lo) else f"{x:.2f} [{lo:.2f}, {hi:.2f}]"


def summary_text(levels: pd.DataFrame) -> str:
    """The main numbers per sweep, setting and level, with their intervals."""
    show = levels.copy()
    for col in MAIN:
        show[col] = [_ci(r, col) for _, r in levels.iterrows()]
    cols = (["sweep"] if levels["sweep"].nunique() > 1 else []) + ["setting", "level", "n_sessions", "n_timed", *MAIN]
    return show[cols].to_string(index=False)
