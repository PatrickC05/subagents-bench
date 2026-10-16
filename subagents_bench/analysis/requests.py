"""Requests of interactive episodes with ``interactive.replies``: per user message (a bug, a
question about the code, a status check), when the user got a reply, what it said, and, for a
bug, how the agent's estimates compare with when it said the bug was done.

All times are seconds after the message was sent (``t_ready``). The user reads only ``reply``
calls, so:

- ``reply_sec``: a question: until its first reply (the answer). A bug: until the agent first
  acknowledged it, the first orchestrator turn after the bug reached it whose text, reasoning or
  tool calls name it (``#N``, request/bug N, or one of its tests), or a reply to it if that came
  first; ``user_reply_sec`` is until the first reply the user saw;
- a question's ``correct``: the judge's verdict on that first reply against the reference answer
  (``subagents-bench grade-answers`` writes ``agent/answers.json``; None until it ran and for a
  question never answered: the table reports accuracy over answered and over asked questions);
- a bug's ``eta_sec`` and ``eta_min``: its first estimate sent before it was said done, and the
  estimate; ``done_sec``: its first reply with ``done``. The estimate predicts done at ``eta_sec +
  60 * eta_min``: ``eta_log2_err`` is log2(actual / predicted time from the estimate to done) (> 0:
  late; a 30 s floor on both), +inf for a bug never said done although the session ran past the
  predicted time or ended for another reason than its deadline (the agent gave up), and None when
  the deadline came first (censored); ``on_time`` is done by the predicted time, with the same
  censoring; ``eta_total_log2_err`` scores the same prediction against the user's whole wait,
  log2(done_sec / (eta_sec + 60 * eta_min)) from when the bug was sent (an estimate given late
  is still scored on its own short horizon by ``eta_log2_err``; how late it came is ``eta_sec``);
  ``claim_ok`` is a done claim the grader confirms;
- an urgent bug's ``urgent_first``: of the earlier bugs still open (not said done) when it was
  sent, the share it was said done before (never-done bugs count as later; None without one);
- a status check's ``reply_sec``: until the first reply of any kind after it was delivered.

A message that lists several bugs (``interactive.bundle``) has a row per bug, with the message's
``msg_id`` and ``msg_index`` and the bug's ``item`` (1, 2, ...) and ``subtask_id``: a bug's replies
are those to its message with its ``item`` or with none (about every bug of the message), and its
acknowledgement is the first turn naming the message (or one of the bug's tests). Summaries leave
out the bugs of the first message (``msg_index`` 0, sent with the task).
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from .interactivity import _group_keys, _median_censored, eval_subtasks
from .load import Trial


def _answers(t: Trial) -> dict[str, dict]:
    try:
        return json.loads((t.dir / "agent" / "answers.json").read_text()).get("answers") or {}
    except (OSError, ValueError, AttributeError):
        return {}


def _acks(t: Trial, bugs: list[dict]) -> dict[tuple[int, Any], float]:
    """(msg_id, item) -> time of the first orchestrator turn after the bug's delivery that names its
    message or one of its tests."""
    path = t.dir / "agent" / "transcript.jsonl"
    if not bugs or not path.is_file():
        return {}
    turns = []
    for line in path.read_text().splitlines():
        x = json.loads(line)
        if x.get("agent_id") == "orchestrator" and x.get("msg_role") == "assistant":
            turns.append((x["t"], json.dumps([x.get("content"), x.get("reasoning"), x.get("tool_calls")])))
    out = {}
    for m in bugs:
        n, tests = m["msg_id"], [f.split("::")[-1] for f in m.get("f2p") or []]
        pat = re.compile(rf"#\s?{n}\b|\b(?:request|bug)\s+#?{n}\b", re.I)
        out[n, m.get("item")] = next((tt for tt, blob in turns if m["t_delivered"] is not None
                                      and tt > m["t_delivered"] and (pat.search(blob) or any(x in blob for x in tests))),
                                     None)
    return out


def request_rows(t: Trial, keys: dict[str, Any], valid: bool) -> list[dict[str, Any]]:
    """One row per message of an episode with replies (none otherwise)."""
    ep, r = t.episode or {}, t.records
    icfg = (ep.get("config") or {}).get("interactive") or {}
    if not icfg.get("replies"):
        return []
    inbox = sorted((x for x in r.get("inbox", []) if x.get("msg_id") is not None), key=lambda x: x["msg_id"])
    if not inbox:
        return []
    replies = sorted(r.get("replies", []), key=lambda x: x["t"])
    session_t0 = ep.get("session_t0", inbox[0]["t_ready"])
    t_end = ep.get("t_end") or max([x["t"] for x in replies] + [x["t_ready"] for x in inbox])
    cut = ep.get("exit_reason") in ("deadline", "cancelled", None)  # observation cut short, not given up
    verdicts, visible_ok, _ = eval_subtasks(t.dir)
    answers = _answers(t)
    by_msg: dict[int, list[dict]] = {}
    for x in replies:
        by_msg.setdefault(x["to"], []).append(x)

    def about(m: dict) -> list[dict]:  # the replies about this message (a listed bug: its item, or none)
        return [x for x in by_msg.get(m["msg_id"], []) if x.get("item") is None or x["item"] == m.get("item")]

    def bug_key(m: dict) -> tuple[int, Any]:
        return m["msg_id"], m.get("item")

    done_at = {bug_key(m): next((x["t"] for x in about(m) if x["done"]), None) for m in inbox if m["kind"] == "hint"}
    acks = _acks(t, [m for m in inbox if m["kind"] == "hint"])
    rows = []
    for m in inbox:
        t0, mine = m["t_ready"], about(m)
        kind = "bug" if m["kind"] == "hint" else m["kind"]
        row = {**keys, "valid": valid, "msg_id": m["msg_id"], "kind": kind, "index": m["hint_index"],
               "msg_index": m.get("msg_index", m["hint_index"]), "item": m.get("item"),
               "urgent": bool(m.get("urgent")), "t_ready_sec": t0 - session_t0,
               "delivery_sec": None if m["t_delivered"] is None else m["t_delivered"] - t0,
               "n_replies": len(mine)}
        if kind == "status":
            after = [x for x in replies if m["t_delivered"] is not None and x["t"] >= m["t_delivered"]]
            row["reply_sec"] = after[0]["t"] - t0 if after else None
        else:
            row["reply_sec"] = mine[0]["t"] - t0 if mine else None
        if kind == "question":
            a = answers.get(str(m["msg_id"])) or {}
            row.update(question_id=m.get("question_id"), question=m.get("question"), reference=m.get("reference"),
                       answer=mine[0]["text"] if mine else None,
                       correct=a.get("correct") if mine else None)
        if kind == "bug":
            done = done_at.get(bug_key(m))
            ahead = [x for x in mine if x["eta_min"] is not None and not x["done"] and (done is None or x["t"] < done)]
            eta = ahead[0] if ahead else None
            v = verdicts.get(m["subtask_id"])
            fixed = (bool(v.get("resolved")) and bool(visible_ok)) if v else (False if valid else None)
            seen = [x for x in (acks.get(bug_key(m)), mine[0]["t"] if mine else None) if x is not None]
            row["user_reply_sec"] = row["reply_sec"]
            row["reply_sec"] = min(seen) - t0 if seen else None
            row.update(subtask_id=m["subtask_id"], subtask_ids=m.get("subtask_ids"), eta_sec=eta["t"] - t0 if eta else None,
                       eta_min=eta["eta_min"] if eta else None, done_sec=done - t0 if done else None,
                       n_estimates=len(ahead), fixed=fixed,
                       claim_ok=None if done is None or fixed is None else fixed)
            row["eta_log2_err"] = row["eta_total_log2_err"] = row["on_time"] = None
            if eta is not None:
                due = eta["t"] + 60 * eta["eta_min"]
                if done is not None:
                    pred, actual = max(60 * eta["eta_min"], 30.0), max(done - eta["t"], 30.0)  # 30 s floor: no log of 0
                    row["eta_log2_err"] = math.log2(actual / pred)
                    row["eta_total_log2_err"] = math.log2(max(done - t0, 30.0) / max(due - t0, 30.0))
                    row["on_time"] = done <= due
                elif t_end > due or not cut:
                    row["eta_log2_err"] = row["eta_total_log2_err"] = math.inf
                    row["on_time"] = False
            row["urgent_first"] = None
            if m.get("urgent"):
                earlier = [o for o in inbox if o["kind"] == "hint" and o["msg_id"] < m["msg_id"]
                           and (done_at.get(bug_key(o)) is None or done_at[bug_key(o)] > t0)]
                if earlier:
                    mine_done = done if done is not None else math.inf
                    row["urgent_first"] = float(np.mean([mine_done < (done_at.get(bug_key(o)) or math.inf)
                                                         for o in earlier]))
        rows.append(row)
    return rows


def _frac(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.astype(float).mean()) if len(x) else math.nan


def _within(x: pd.Series, sec: float) -> float:
    return float((pd.to_numeric(x, errors="coerce").fillna(math.inf) <= sec).mean()) if len(x) else math.nan


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").astype(float) if col in df else pd.Series(dtype=float)


def _median(x: pd.Series) -> float:
    """Median of the known values; +inf ones (never done) count, so it can be +inf."""
    x = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    return float(np.median(x)) if len(x) else math.nan


def _correct_of_asked(q: pd.DataFrame) -> float:
    """Correct answers over every question asked; NaN while an answered question is ungraded."""
    if not len(q) or "correct" not in q or (q["reply_sec"].notna() & q["correct"].isna()).any():
        return math.nan
    return float((q["correct"] == True).sum() / len(q))  # noqa: E712 (object column)


def request_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Per (sweep, condition), over valid episodes: questions, bug estimates and done claims, status
    checks, urgency (per condition for rows without a sweep)."""
    if rows.empty or not rows["valid"].astype(bool).any():
        return pd.DataFrame()
    out = []
    v = rows[rows["valid"].astype(bool)]
    keys = _group_keys(v)
    for key, g in v.groupby(keys, sort=False):
        q, b, s = g[g["kind"] == "question"], g[g["kind"] == "bug"], g[g["kind"] == "status"]
        later = b[b["msg_index"] > 0] if "msg_index" in b else b[b["index"] > 0]  # not the first message's
        row = {**dict(zip(keys, key)), "n_episodes": g["trial_name"].nunique(), "n_questions": len(q),
               "answered_frac": _frac(q["reply_sec"].notna()), "answer_median_sec": _median_censored(q["reply_sec"]),
               "answered_60s": _within(q["reply_sec"], 60), "answered_180s": _within(q["reply_sec"], 180),
               "n_graded_answers": int(q["correct"].notna().sum()) if "correct" in q else 0,
               "correct_of_answered": _frac(q["correct"]) if "correct" in q else math.nan,
               "correct_of_asked": _correct_of_asked(q),
               "n_bugs": len(b), "fixed_frac": _frac(b["fixed"]) if "fixed" in b else math.nan,
               "first_reply_median_sec": _median_censored(later["reply_sec"]),
               "user_reply_median_sec": _median_censored(later["user_reply_sec"]) if "user_reply_sec" in b else math.nan,
               "eta_given_frac": _frac(b["eta_sec"].notna()) if "eta_sec" in b else math.nan,
               "eta_median_sec": _median_censored(later["eta_sec"]) if "eta_sec" in b else math.nan,
               "eta_min_median": float(b["eta_min"].median()) if "eta_min" in b and b["eta_min"].notna().any() else math.nan,
               "done_claimed_frac": _frac(b["done_sec"].notna()) if "done_sec" in b else math.nan,
               "claim_precision": _frac(b["claim_ok"]) if "claim_ok" in b else math.nan,
               "eta_abs_log2_median": _median(_num(b, "eta_log2_err").abs()),
               "eta_log2_median": _median(_num(b, "eta_log2_err")),
               "eta_total_abs_log2_median": _median(_num(b, "eta_total_log2_err").abs()),
               "n_eta_never_done": int(np.isinf(pd.to_numeric(b["eta_log2_err"], errors="coerce")).sum())
               if "eta_log2_err" in b else 0,
               "on_time_frac": _frac(b["on_time"]) if "on_time" in b else math.nan,
               "n_status": len(s), "status_reply_median_sec": _median_censored(s["reply_sec"]),
               "n_urgent": int(b["urgent"].sum()) if len(b) else 0,
               "urgent_first_frac": _frac(b["urgent_first"]) if "urgent_first" in b else math.nan}
        out.append(row)
    return pd.DataFrame(out)
