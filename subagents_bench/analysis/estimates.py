"""How well the agent's estimates predict when a bug is fixed (``estimates.csv``) and whether a
status check got an answer for each bug it lists (``status_pairs.csv``), scored against the
replayed fix times (``fixes.py``: ``fixed_sec``) and against the agent's own done claims.

**Estimates.** Every ``reply`` to a bug's id with ``eta_min`` and without ``done: true`` is an
estimate (``via`` ``structured``). Estimates written only in text are read by regex, no LLM
(``parse_estimates``): a bug reply without ``eta_min`` (``text_bug``: ``ETA ~10 min``), and a reply
to a status check's id, once per bug it gives a time for (``text_status``: ``#4 ~8 min; #5 ~6
min``, ``#8 and #18 ... ETA ~10 min each``). Each estimate has a ``source``: ``arrival`` (the
bug's first estimate, sent before the next status check after the bug; in a session without
checks (L0), before the time L1's next check after it would be sent (``ARRIVAL_EVERY``: L1's
``status_every``, a check half an interval after every second bug, before the deadline), so that
L0 and L1 arrival estimates share one window; with no such check after it, whenever it came),
``check`` (sent within ``CHECK_SEC`` = 120 s after a status check that lists the bug: every bug
sent before the check, ``check_id``), else ``update``.

**Scoring.** An estimate of m minutes sent at t predicts the fix at ``pred = t + 60 m``, with a
grace of max(60 s, 25% of 60 m). It is ``eligible`` iff pred + grace falls within the session
(``wall_clock_sec``), a rule on the prediction alone; ineligible estimates are reported only as
the share whose bug was fixed by the deadline (``fixed_by_deadline``).

- Against the fix (``timing_ok`` trials): an estimate sent at or after the bug's fix is
  ``stale`` and not scored; an eligible one sent before it is ``scored``: ``on_time`` iff fixed by
  pred + grace (``on_time_strict``: by pred; ``within_1m``/``2m``/``5m``: by pred + 1/2/5 min),
  ``err_min`` = (fix - pred) / 60, ``log2_err`` = log2(max(fix - t, 30) / max(60 m, 30)) (> 0:
  late; a 30 s floor on both). A scored estimate whose bug is never fixed is late whatever the
  exit: ``log2_err`` +inf and ``err_min`` a lower bound (``err_lower_bound``) from pred to the
  later of the episode's end and the deadline.
- Against the done claim (the bug's first reply with ``done: true``, ``claim_sec``): the same
  rules, for eligible estimates sent before the claim (``claim_scored``); a bug never said done
  is late (+inf).
- ``weight`` (``claim_weight``) = 1 / the bug's scored (claim-scored) estimates, so each bug
  counts once; ``estimate_table`` recomputes it within each subset it reports.

**Status pairs.** For each status check and each bug it lists that was not said done before the
check was sent: the first estimate (any via) or done for the bug after the check was sent
(``reply_sec``, ``kind`` ``eta``/``done``, ``via``); said done is a reply to the bug with ``done:
true`` (``structured``) or a done written in a reply to a status check's id (``text_status``,
``parse_done``: ``#1 is fixed and verified``), whichever came first. ``answered`` iff within
``CHECK_SEC`` (a fixed window: in L2/L3 a question follows each check after 40 s). ``right``: a
done on a bug fixed by then, an estimate on one not fixed (by ``fixed_sec``); ``done_owed``: fixed
by the check and never said done in the session.

All times are seconds after the message was sent (the bug for estimates, the check for pairs),
except ``*_session_sec`` (from the session's start).
"""

from __future__ import annotations

import math
import re
from typing import Any, Collection

import numpy as np
import pandas as pd

from .fixes import fix_times, keys
from .interactivity import _median_censored
from .load import Trial

CHECK_SEC = 120.0  # a check reply counts within this long after the check was sent
GRACE_MIN_SEC = 60.0
GRACE_FRAC = 0.25
FLOOR_SEC = 30.0  # log2 floor on the predicted and the actual horizon
WITHIN_MIN = (1, 2, 5)
SOURCES = ("arrival", "check", "update")
ARRIVAL_EVERY = 2  # L1's status_every: where a session without checks (L0) ends a bug's arrival window
VIAS = ("structured", "text_bug", "text_status")

# -- text estimates ----------------------------------------------------------------------------

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
ID_RE = re.compile(r"#(\d+)\b")
DURATION_RE = re.compile(
    r"(?P<approx>~\s*|≈\s*|\babout\s+|\bapprox(?:\.|imately)?\s+|\baround\s+|\broughly\s+|\b(?:with)?in\s+)?"
    r"(?<![\w.#/:-])(?P<n>\d+(?:\.\d+)?)(?:\s*(?:-|–|to)\s*(?P<n2>\d+(?:\.\d+)?))?"
    r"(?:\s*(?P<unit>min(?:ute)?s?|h(?:ou)?rs?|sec(?:ond)?s?)\b|(?P<short>(?-i:[mh]))\b)", re.IGNORECASE)
CUE_RE = re.compile(r"\b(?:ETAs?|estimated?|estimates|expect(?:ed|s)?)\b", re.IGNORECASE)
FUTURE_RE = re.compile(r"\b(?:will|should|shall)\b|'ll\b", re.IGNORECASE)
DONE_RE = re.compile(r"\b(?:fixed|done|resolved)\b", re.IGNORECASE)
FOR_RE = re.compile(r"\s*(?:for|on)\s+(?:bugs?\s+|requests?\s+)?#", re.IGNORECASE)  # "~5 min for #2"
NOT_DONE_RE = re.compile(r"(?:\b(?:be|being|get|getting|not|once|when|until|if|before)|n't)\s+(?:\S+\s+)?$",
                         re.IGNORECASE)


def _minutes(m: re.Match) -> float:
    n = float(m["n2"] or m["n"])  # a range: its end
    unit = (m["unit"] or m["short"]).lower()
    return n * 60 if unit.startswith("h") else n / 60 if unit.startswith("s") else n


def _cued(sentence: str, m: re.Match) -> bool:
    """A time that states an estimate: approximate (``~10 min``, ``about 10 min``), after an ETA
    word in its sentence (``ETA: 10 min``), or ``in 10 min`` after a future (``should be done in 6 min``)."""
    before, approx = sentence[:m.start()], (m["approx"] or "").strip().lower()
    if CUE_RE.search(before):
        return True
    return bool(approx) and (approx not in ("in", "within") or FUTURE_RE.search(before) is not None)


def parse_estimates(text: str, ids: Collection[int], default: int | None = None) -> dict[int, float]:
    """Minutes per message id that ``text`` gives an estimate for (the first one per id).

    Only ``#N`` with N in ``ids`` names a bug. Within a sentence a time goes to the ids named since
    the previous time (``#1 ~5 min; #4 ~8 min``, ``#8 and #18 ... ETA ~10 min each``); a sentence whose
    only time comes before its ids gives it to all of them (``ETA ~10 min for #8 and #18``), and with
    more times, a time followed by ``for #`` goes to the ids after it (``ETA ~10 min for #8, ~5 min
    for #2``); a done word (``#1 is fixed``) drops the ids named before it. Ids of a sentence without
    a time carry to the next one if that starts with a time (``#8 and #18 are open. ETA ~10 min.``).
    A time left over goes to ``default`` (the bug a reply is addressed to) unless it got one, the
    first one after an ETA word if any (``Test suite takes ~2 min; ETA 10 min``: 10); with a
    ``default``, a time is given to other ids before or after it in this way only when ``default``
    is among them (a reply's ETA is for its own bug)."""
    ids = set(ids)
    out: dict[int, float] = {}
    leftover: float | None = None  # the first time no id took
    leftover_cue: float | None = None  # ... of those after an ETA word
    carried: list[int] = []
    for sentence in SENTENCE_RE.split(text or ""):
        times = [m for m in DURATION_RE.finditer(sentence) if _cued(sentence, m)]
        forward = {m.start() for m in times if FOR_RE.match(sentence, m.end())}  # for the ids after it
        cue = {m.start() for m in times if CUE_RE.search(sentence[:m.start()])}
        events = sorted([(m.start(), "id", int(m[1])) for m in ID_RE.finditer(sentence) if int(m[1]) in ids]
                        + [(m.start(), "time", _minutes(m)) for m in times]
                        + [(m.start(), "done", None) for m in DONE_RE.finditer(sentence)
                           if not NOT_DONE_RE.search(sentence[:m.start()])], key=lambda e: e[0])
        had_time = any(kind == "time" for _, kind, _ in events)
        if carried and events and events[0][1] == "time":
            for i in carried:
                out.setdefault(i, events[0][2])
            events = events[1:]
        carried = []
        named = {x for _, kind, x in events if kind == "id"}
        time_first = (events and events[0][1] == "time" and sum(kind == "time" for _, kind, _ in events) == 1
                      and (default is None or default in named))  # else the reply's own bug's time
        pending: list[int] = []
        free: list[tuple[float, bool]] = []  # times no id took (yet), and whether an ETA word cued them
        fwd: float | None = None  # a time written before the ids it is for (not time_first)
        for pos, kind, x in events:
            if kind == "id":
                if fwd is not None:
                    out.setdefault(x, fwd)
                else:
                    pending += [] if x in pending else [x]
            elif kind == "done":
                pending, fwd = [], None
            elif pending:
                for i in pending:
                    out.setdefault(i, x)
                pending, fwd = [], None
            elif pos in forward and not time_first and (default is None or default in named):
                fwd = x
            else:
                free.append((x, pos in cue))
                fwd = None
        if pending and time_first:
            x = free.pop(0)[0]
            for i in pending:
                out.setdefault(i, x)
        elif pending and not had_time and (default is None or default in pending):
            carried = pending
        leftover = free[0][0] if leftover is None and free else leftover
        leftover_cue = next((x for x, c in free if c), None) if leftover_cue is None else leftover_cue
    if default is not None and default not in out and leftover is not None:
        out[default] = leftover if leftover_cue is None else leftover_cue
    return out


def parse_done(text: str, ids: Collection[int]) -> set[int]:
    """Message ids that ``text`` says are done: the ids named before an un-negated done word in
    the same sentence, since the previous time or done word (``#1 is fixed and verified``, ``#1,
    #4 are fixed``; not ``#4 will be fixed in ~5 min`` or ``#2 is not yet fixed``)."""
    ids, out = set(ids), set()
    for sentence in SENTENCE_RE.split(text or ""):
        events = sorted([(m.start(), "id", int(m[1])) for m in ID_RE.finditer(sentence) if int(m[1]) in ids]
                        + [(m.start(), "time", None) for m in DURATION_RE.finditer(sentence) if _cued(sentence, m)]
                        + [(m.start(), "done", None) for m in DONE_RE.finditer(sentence)
                           if not NOT_DONE_RE.search(sentence[:m.start()])], key=lambda e: e[0])
        pending: list[int] = []
        for _, kind, x in events:
            if kind == "id":
                pending.append(x)
            else:
                out.update(pending if kind == "done" else ())
                pending = []
    return out


# -- rows --------------------------------------------------------------------------------------


def _session(t: Trial) -> tuple[list[dict], list[dict], list[dict], float, float, float]:
    """Bugs, status checks and replies (by time), the session's start, its deadline and the
    episode's end (epoch seconds)."""
    ep, r = t.episode or {}, t.records
    inbox = [m for m in r.get("inbox", []) if m.get("msg_id") is not None]
    bugs = sorted((m for m in inbox if m.get("kind") == "hint"), key=lambda m: m["t_ready"])
    checks = sorted((m for m in inbox if m.get("kind") == "status"), key=lambda m: m["t_ready"])
    replies = sorted(r.get("replies", []), key=lambda x: x["t"])
    t0 = ep.get("session_t0")
    t0 = t0 if t0 is not None else min((m["t_ready"] for m in inbox), default=0.0)
    wall = ((ep.get("config") or {}).get("interactive") or {}).get("wall_clock_sec")
    last = max([x["t"] for x in replies] + [m["t_ready"] for m in inbox], default=t0)
    t_end = ep.get("t_end") or last
    deadline = t0 + wall if wall is not None else t_end
    return bugs, checks, replies, t0, deadline, max(t_end, deadline)


def _raw_estimates(bugs: list[dict], checks: list[dict], replies: list[dict]) -> list[dict]:
    """(t, bug msg id, minutes, via, the id replied to, text) of every estimate, by time."""
    kind = {m["msg_id"]: "bug" for m in bugs} | {m["msg_id"]: "status" for m in checks}
    out = []
    for x in replies:
        to = x.get("to")
        if kind.get(to) == "bug" and not x.get("done"):
            if x.get("eta_min") is not None:
                out.append({"t": x["t"], "bug": to, "eta_min": float(x["eta_min"]), "via": "structured", "to": to,
                            "text": None})
                continue
            sent = {m["msg_id"] for m in bugs if m["t_ready"] <= x["t"]}
            m = parse_estimates(x.get("text") or "", sent, default=to).get(to)
            if m is not None:
                out.append({"t": x["t"], "bug": to, "eta_min": m, "via": "text_bug", "to": to, "text": x.get("text")})
        elif kind.get(to) == "status":
            sent = {m["msg_id"] for m in bugs if m["t_ready"] <= x["t"]}
            for bug, m in parse_estimates(x.get("text") or "", sent).items():
                out.append({"t": x["t"], "bug": bug, "eta_min": m, "via": "text_status", "to": to,
                            "text": x.get("text")})
    return sorted(out, key=lambda e: (e["t"], e["bug"]))


def _arrival_end(m: dict, checks: list[dict], n_bugs: int, every: float | None, t0: float,
                 deadline: float) -> float:
    """When bug ``m``'s arrival window ends (epoch seconds; inf: never): the first status check
    sent after it; in a session without checks (L0), when L1's next check would be sent, as
    ``hints.schedule`` places it: at (k + 0.5) ``interval_sec`` for the first bug index k >= the
    bug's with (k + 1) % ``ARRIVAL_EVERY`` == 0, if that bug exists and the time is before the deadline."""
    if checks or not every or m.get("hint_index") is None:
        return next((c["t_ready"] for c in checks if c["t_ready"] > m["t_ready"]), math.inf)
    k = -(-(m["hint_index"] + 1) // ARRIVAL_EVERY) * ARRIVAL_EVERY - 1
    at = t0 + (k + 0.5) * every
    return at if k < n_bugs and at < deadline else math.inf


def _claims(replies: list[dict]) -> dict[int, float]:
    """Message id -> its first reply with ``done: true``."""
    out: dict[int, float] = {}
    for x in replies:
        if x.get("done"):
            out.setdefault(x.get("to"), x["t"])
    return out


def _fix_at(m: dict, fixes: dict[str, dict]) -> tuple[bool, float | None]:
    """(timing_ok, absolute fix time or None) of a bug."""
    f = fixes.get(m.get("subtask_id")) or {}
    if not f.get("timing_ok"):
        return False, None
    return True, None if f.get("fixed_sec") is None else m["t_ready"] + f["fixed_sec"]


def _fixes(t: Trial, fixes: Any) -> dict[str, dict]:
    if fixes is None:
        return fix_times(t)
    if isinstance(fixes, dict):
        return fixes
    return {f["subtask_id"]: f for f in fixes}


def _log2(actual: float, predicted: float) -> float:
    return math.log2(max(actual, FLOOR_SEC) / max(predicted, FLOOR_SEC))


def _score(t: float, eta_min: float, at: float | None, end: float) -> dict[str, Any]:
    """An estimate sent at ``t`` against an event at ``at`` (None: never)."""
    pred, grace = t + 60 * eta_min, max(GRACE_MIN_SEC, GRACE_FRAC * 60 * eta_min)
    if at is None:
        return {"on_time": False, "on_time_strict": False, **{f"within_{k}m": False for k in WITHIN_MIN},
                "err_min": (end - pred) / 60, "err_lower_bound": True, "log2_err": math.inf}
    return {"on_time": at <= pred + grace, "on_time_strict": at <= pred,
            **{f"within_{k}m": at <= pred + 60 * k for k in WITHIN_MIN},
            "err_min": (at - pred) / 60, "err_lower_bound": False, "log2_err": _log2(at - t, 60 * eta_min)}


SCORE_COLS = ("on_time", "on_time_strict", *(f"within_{k}m" for k in WITHIN_MIN), "err_min", "err_lower_bound",
              "log2_err")


def estimate_rows(t: Trial, fixes: Any = None) -> list[dict[str, Any]]:
    """One ``estimates.csv`` row per estimate of a bug (``fixes``: ``fix_times(t)``, computed when
    None; a list of ``bug_fix_rows`` works too)."""
    bugs, checks, replies, t0, deadline, end = _session(t)
    if not bugs:
        return []
    fixes = _fixes(t, fixes)
    claims = _claims(replies)
    by_id = {m["msg_id"]: m for m in bugs}
    every = (((t.episode or {}).get("config") or {}).get("interactive") or {}).get("interval_sec")
    seen: set[int] = set()
    rows = []
    for e in _raw_estimates(bugs, checks, replies):
        m, eta = by_id[e["bug"]], e["eta_min"]
        ready = m["t_ready"]
        nxt = _arrival_end(m, checks, len(bugs), every, t0, deadline)
        check = next((c for c in reversed(checks) if ready < c["t_ready"] <= e["t"] <= c["t_ready"] + CHECK_SEC), None)
        source = "arrival" if e["bug"] not in seen and e["t"] < nxt else "check" if check else "update"
        seen.add(e["bug"])
        pred, grace = e["t"] + 60 * eta, max(GRACE_MIN_SEC, GRACE_FRAC * 60 * eta)
        eligible = pred + grace <= deadline
        timing_ok, fix = _fix_at(m, fixes)
        stale = None if not timing_ok else fix is not None and e["t"] >= fix
        claim = claims.get(e["bug"])
        row = {**keys(t), "msg_id": e["bug"], "subtask_id": m.get("subtask_id"), "hint_index": m.get("hint_index"),
               "urgent": bool(m.get("urgent")), "ready_session_sec": ready - t0, "via": e["via"], "source": source,
               "reply_to": e["to"], "check_id": check["msg_id"] if check else None,
               "est_session_sec": e["t"] - t0, "est_sec": e["t"] - ready, "eta_min": eta,
               "pred_session_sec": pred - t0, "grace_sec": grace, "eligible": eligible, "timing_ok": timing_ok,
               "fixed_sec": None if fix is None else fix - ready, "stale": stale,
               "fixed_by_deadline": None if not timing_ok else fix is not None and fix <= deadline,
               "scored": bool(timing_ok and eligible and not stale),
               **dict.fromkeys(SCORE_COLS), "weight": None,
               "claim_sec": None if claim is None else claim - ready,
               "claim_scored": bool(eligible and (claim is None or e["t"] < claim)),
               **{f"claim_{k}": None for k in SCORE_COLS}, "claim_weight": None,
               "text": None if e["text"] is None else e["text"][:500]}
        if row["scored"]:
            row.update(_score(e["t"], eta, fix, end))
        if row["claim_scored"]:
            row.update({f"claim_{k}": v for k, v in _score(e["t"], eta, claim, end).items()})
        rows.append(row)
    for scored, weight in (("scored", "weight"), ("claim_scored", "claim_weight")):
        n: dict[int, int] = {}
        for r in rows:
            n[r["msg_id"]] = n.get(r["msg_id"], 0) + r[scored]
        for r in rows:
            r[weight] = 1 / n[r["msg_id"]] if r[scored] else None
    return rows


def status_pair_rows(t: Trial, fixes: Any = None) -> list[dict[str, Any]]:
    """One ``status_pairs.csv`` row per (status check, bug it lists not said done before it)."""
    bugs, checks, replies, t0, _, _ = _session(t)
    if not bugs or not checks:
        return []
    fixes = _fixes(t, fixes)
    claims = _claims(replies)
    answers: dict[int, list[tuple[float, str, str]]] = {}  # bug -> (t, kind, via), by time
    for e in _raw_estimates(bugs, checks, replies):
        answers.setdefault(e["bug"], []).append((e["t"], "eta", e["via"]))
    for bug, tc in claims.items():
        answers.setdefault(bug, []).append((tc, "done", "structured"))
    check_ids = {c["msg_id"] for c in checks}
    for x in replies:  # a done written in a reply to a status check's id (the user's view: said done)
        if x.get("to") in check_ids:
            for bug in parse_done(x.get("text") or "", {m["msg_id"] for m in bugs if m["t_ready"] <= x["t"]}):
                answers.setdefault(bug, []).append((x["t"], "done", "text_status"))
                claims[bug] = min(claims.get(bug, math.inf), x["t"])
    rows = []
    for c in checks:
        tc = c["t_ready"]
        for m in bugs:
            claim = claims.get(m["msg_id"])
            if m["t_ready"] >= tc or (claim is not None and claim < tc):
                continue
            first = min((a for a in answers.get(m["msg_id"], []) if a[0] >= tc), default=None)
            answered = first is not None and first[0] - tc <= CHECK_SEC
            timing_ok, fix = _fix_at(m, fixes)
            fixed_then = None if not (timing_ok and answered) else fix is not None and fix <= first[0]
            rows.append({**keys(t), "check_id": c["msg_id"], "check_session_sec": tc - t0, "msg_id": m["msg_id"],
                         "subtask_id": m.get("subtask_id"), "hint_index": m.get("hint_index"),
                         "urgent": bool(m.get("urgent")), "reply_sec": None if first is None else first[0] - tc,
                         "answered": answered, "kind": first[1] if answered else None,
                         "via": first[2] if answered else None, "timing_ok": timing_ok,
                         "right": None if fixed_then is None else fixed_then == (first[1] == "done"),
                         "fixed_at_check": None if not timing_ok else fix is not None and fix <= tc,
                         "claimed": claim is not None,
                         "done_owed": None if not timing_ok else fix is not None and fix <= tc and claim is None})
    return rows


# -- the table ---------------------------------------------------------------------------------


def _bug_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in ("job", "trial_name", "msg_id") if c in df]


def _weighted(df: pd.DataFrame, col: str) -> float:
    """Mean of ``col`` over the rows that have it, each bug weighing 1 (its rows share it)."""
    d = df[df[col].notna()] if col in df else df.iloc[0:0]
    if not len(d):
        return math.nan
    w = 1.0 / d.groupby(_bug_cols(d))[col].transform("size").astype(float)
    return float((d[col].astype(float) * w).sum() / w.sum())


def _flag(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].fillna(False).astype(bool) if col in df else pd.Series(False, index=df.index)


def _frac(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.astype(float).mean()) if len(x) else math.nan


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").astype(float) if len(df) and col in df else pd.Series(dtype=float)


def _median(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    return float(np.median(x)) if len(x) else math.nan


def _valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["valid"].fillna(False).astype(bool)] if "valid" in df else df


def _groups(df: pd.DataFrame, group: list[str]) -> dict[tuple, pd.DataFrame]:
    if not len(df):
        return {}
    if not group:
        return {(): df}
    return {k if isinstance(k, tuple) else (k,): g for k, g in df.groupby(group, sort=False, dropna=False)}


def estimate_table(rows: pd.DataFrame, pairs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (sweep, condition), over valid trials: the on-time share of scored estimates (each bug
    weighing 1), overall, strict, within 1/2/5 min and by source and via; the median |log2| error
    over fixed bugs (``abs_log2_median_fixed``) and over every scored estimate (+inf for a bug
    never fixed); the stale rate; ineligible estimates' share fixed by the deadline; the on-time
    share against done claims; and from ``pairs`` (``status_pair_rows``) the share of (check, bug)
    pairs answered within 120 s, answered right, and done owed (fixed by the check, never said
    done)."""
    rows = _valid(rows) if rows is not None and len(rows) else pd.DataFrame()
    pairs = _valid(pairs) if pairs is not None and len(pairs) else pd.DataFrame()
    if rows.empty and pairs.empty:
        return pd.DataFrame()
    group = [c for c in ("sweep", "condition") if c in (rows if len(rows) else pairs)]
    by_e, by_p = _groups(rows, group), _groups(pairs, group)
    out = []
    for name in dict.fromkeys([*by_e, *by_p]):
        e, p = by_e.get(name, pd.DataFrame()), by_p.get(name, pd.DataFrame())
        row = {**dict(zip(group, name)), "n_episodes_estimated": e["trial_name"].nunique() if len(e) else 0,
               "n_estimates": len(e),
               **{f"n_{v}": int((e["via"] == v).sum()) if len(e) else 0 for v in VIAS}}
        timed = e[_flag(e, "timing_ok")] if len(e) else e
        s = e[_flag(e, "scored")] if len(e) else e
        row.update(n_bugs_estimated=len(e.drop_duplicates(_bug_cols(e))) if len(e) else 0, n_timed=len(timed),
                   n_scored=len(s), n_scored_bugs=len(s.drop_duplicates(_bug_cols(s))) if len(s) else 0)
        for col in ("on_time", "on_time_strict", *(f"within_{k}m" for k in WITHIN_MIN)):
            row[f"{col}_frac"] = _weighted(s, col)
        for col, values in (("source", SOURCES), ("via", VIAS)):
            for v in values:
                sub = s[s[col] == v] if len(s) else s
                row[f"n_scored_{v}"], row[f"on_time_frac_{v}"] = len(sub), _weighted(sub, "on_time")
        log2 = _num(s, "log2_err")
        row.update(abs_log2_median_fixed=_median(log2[np.isfinite(log2)].abs()), abs_log2_median=_median(log2.abs()),
                   log2_median=_median(log2),
                   err_min_median_fixed=_median(s.loc[~_flag(s, "err_lower_bound"), "err_min"]) if len(s) else math.nan,
                   stale_frac=_frac(timed["stale"]) if len(timed) else math.nan)
        late = timed[~_flag(timed, "eligible") & ~_flag(timed, "stale")] if len(timed) else timed
        row.update(n_ineligible=len(late),
                   ineligible_fixed_frac=_frac(late["fixed_by_deadline"]) if len(late) else math.nan)
        cs = e[_flag(e, "claim_scored")] if len(e) else e
        row.update(n_claim_scored=len(cs), claim_on_time_frac=_weighted(cs, "claim_on_time"),
                   claim_abs_log2_median=_median(_num(cs, "claim_log2_err").abs()))
        pt = p[_flag(p, "timing_ok")] if len(p) else p
        owed = pt[_flag(pt, "fixed_at_check")] if len(pt) else pt
        row.update(n_pairs=len(p), coverage_120s=_frac(p["answered"]) if len(p) else math.nan,
                   pair_reply_median_sec=_median_censored(p["reply_sec"]) if len(p) else math.nan,
                   right_frac=_frac(pt["right"]) if len(pt) else math.nan,
                   done_owed_frac=_frac(owed["done_owed"]) if len(owed) else math.nan,
                   n_done_owed=len(owed[_flag(owed, "done_owed")].drop_duplicates(_bug_cols(owed))) if len(owed) else 0)
        out.append(row)
    return pd.DataFrame(out)
