"""Interactivity of interactive episodes: per hint (a SWE-sweep subtask sent as a user message
at ``t_ready``), how long until the orchestrator received it, answered it, and until any agent
dispatched on, touched and edited the subtask's files, and whether the grader counts the
subtask fixed.

All latencies are seconds after ``t_ready`` (when the hint was sent):

- ``delivery``: until it entered the orchestrator's conversation (it waits for the step in progress);
- ``response``: until the orchestrator's first LLM call that saw it returned ("it says something");
- ``dispatch``: until a subagent started whose instruction names one of the target files or tests;
- ``touch``: until any agent's tool call named one of the target files (a path argument or a bash
  command; ``search`` over the whole repository does not count);
- ``edit``: until any agent edited one: ``edit_file`` on it, or a bash command naming it that
  writes (``sed -i``, a redirection to it, ``patch``/``git apply``, a Python ``open(..., 'w')``).

Matching is by path text, so it is a heuristic: a target file shared with an earlier hint's
subtask (``shared_files``) can be touched for that subtask, which makes this hint look started
early; the summary reports touch and edit for unshared hints separately. A latency is missing
when the event never happened before the session ended; the summary's medians count those as
infinitely late (NaN when more than half are missing) and ``*_before_next`` is the fraction of
hints with the event before the next hint was sent (one ``interval_sec``).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .load import Trial
from .stats import bootstrap_mean

LATENCIES = ("delivery", "response", "dispatch", "touch", "edit")
PATH_ARGS = {"read_file": "path", "edit_file": "path", "list_dir": "path", "search": "path"}
WRITE_RE = re.compile(r"sed\s+(?:-[a-zA-Z]*\s+)*-[a-zA-Z]*i|perl\s+-[a-zA-Z]*i|\bpatch\b|git\s+apply|write_text\(|"
                      r"\.write\(|open\([^)]*['\"][wa]b?['\"]")
BUSY = ("llm", "tool", "spawn")


def _tokens(path: str) -> tuple[list[str], str]:
    """Substrings that name ``path`` (it and its last two components), and its basename."""
    parts = path.strip("/").split("/")
    return list(dict.fromkeys(["/".join(parts), "/".join(parts[-2:])])) if len(parts) > 1 else [path], parts[-1]


def names_file(text: str, path: str) -> bool:
    """``text`` names ``path``: by its path or last two components (as a path, e.g. after the
    repository's absolute root), or by its bare basename (not as part of another path)."""
    subs, base = _tokens(path)
    end = r"(?![\w\-]|\.\w)"
    if any(re.search(rf"(?<![\w.\-]){re.escape(x)}{end}", text) for x in subs):
        return True
    return re.search(rf"(?<![\w.\-/]){re.escape(base)}{end}", text) is not None


def _test_names(f2p: list[str]) -> list[str]:
    """Distinctive names of the hidden tests (the last component of each id, at least 8 chars)."""
    out = []
    for t in f2p:
        name = re.split(r"::|\.|\s", re.sub(r"\[.*\]$", "", t).strip())[-1]
        if len(name) >= 8:
            out.append(name)
    return list(dict.fromkeys(out))


def tool_touches(tool: dict, files: list[str]) -> tuple[bool, bool]:
    """(touches, edits) one of ``files``."""
    args = tool.get("arguments") or {}
    if not isinstance(args, dict):
        return False, False
    name = tool["name"]
    if name in PATH_ARGS:
        p = args.get(PATH_ARGS[name])
        hit = isinstance(p, str) and any(names_file(p, f) for f in files)
        return hit, hit and name == "edit_file"
    if name == "bash":
        cmd = args.get("command")
        if not isinstance(cmd, str):
            return False, False
        named = [f for f in files if names_file(cmd, f)]
        if not named:
            return False, False
        redirect = any(re.search(rf">>?\s*['\"]?[^\s'\"&|;<>]*{re.escape(_tokens(f)[1])}(?![\w\-])", cmd) for f in named)
        tee = any(re.search(rf"\btee\s+(?:-a\s+)?['\"]?[^\s'\"]*{re.escape(_tokens(f)[1])}", cmd) for f in named)
        return True, bool(redirect or tee or WRITE_RE.search(cmd))
    return False, False


def _busy(t: float, calls: list[dict], tools: list[dict], spawns: dict[str, dict]) -> str:
    """What the orchestrator was doing at ``t``: llm, tool, spawn (a blocking spawn), wait or idle."""
    for c in calls:
        if c["role"] == "orchestrator" and c["t_start"] <= t < c["t_end"]:
            return "llm"
    for x in tools:
        if x["role"] == "orchestrator" and x["t_start"] <= t < x["t_end"]:
            if x["name"] == "wait":
                return "wait"
            if x["name"] == "spawn_subagents" and (spawns.get(x["group_id"]) or {}).get("spawn_mode", "blocking") == "blocking":
                return "spawn"
            return "tool"
    return "idle"


def eval_subtasks(trial_dir: Path) -> tuple[dict[str, dict], bool | None]:
    """The grader's per-subtask verdicts (``verifier/eval.json``) and its visible_ok."""
    try:
        ev = json.loads((trial_dir / "verifier" / "eval.json").read_text())
    except (OSError, ValueError):
        return {}, None
    subs = {s["subtask_id"]: s for s in ev.get("subtasks") or [] if isinstance(s, dict) and "subtask_id" in s}
    return subs, ev.get("visible_ok")


def hint_rows(t: Trial, keys: dict[str, Any], valid: bool) -> list[dict[str, Any]]:
    """One row per hint of an interactive episode (none for a batch episode)."""
    ep, r = t.episode or {}, t.records
    icfg = (ep.get("config") or {}).get("interactive")
    hints = sorted((x for x in r.get("inbox", []) if x["kind"] == "hint"), key=lambda x: x["hint_index"])
    if icfg is None or not hints:
        return []
    interval = icfg["interval_sec"]
    steps = {s["step"]: s for s in r["steps"]}
    spawns = {s["group_id"]: s for s in r["spawns"]}
    tools = sorted(r["tools"], key=lambda x: x["t_start"])
    dispatches = sorted(r["dispatches"], key=lambda d: d["t_start"])
    verdicts, visible_ok = eval_subtasks(t.dir)
    sub = ep.get("config", {}).get("subagent") or {}
    rows, earlier_files = [], set()
    for k, h in enumerate(hints):
        t0, files = h["t_ready"], h["target_files"] or []
        t_next = hints[k + 1]["t_ready"] if k + 1 < len(hints) else None
        row = {**keys, "valid": valid, "topology": ep.get("topology"),
               "spawn_mode": sub.get("spawn_mode", "blocking") if ep.get("topology") == "subagents" else None,
               "hint_detail": icfg.get("hint_detail"), "interval_sec": interval, "hint_index": h["hint_index"],
               "subtask_id": h["subtask_id"], "target_files": ";".join(files), "n_target_files": len(files),
               "shared_files": bool(set(files) & earlier_files), "t_ready_sec": t0 - ep.get("session_t0", t0),
               "delivered": h["t_delivered"] is not None,
               "busy_at_arrival": _busy(t0, r["calls"], r["tools"], spawns) if k else "start"}
        earlier_files |= set(files)
        lat: dict[str, float | None] = dict.fromkeys(LATENCIES)
        if h["t_delivered"] is not None:
            lat["delivery"] = h["t_delivered"] - t0
            s = steps.get(h["step"])
            if s is not None and s.get("t_llm_end") is not None and s.get("call_id") is not None:
                lat["response"] = s["t_llm_end"] - t0
        names = _test_names(h["f2p"] or [])
        for d in dispatches:
            if d["t_start"] >= t0 and (any(names_file(d.get("instruction") or "", f) for f in files)
                                       or any(n in (d.get("instruction") or "") for n in names)):
                lat["dispatch"] = d["t_start"] - t0
                break
        touch_by = None
        for x in tools:
            if x["t_start"] < t0:
                continue
            touched, edited = tool_touches(x, files)
            if touched and lat["touch"] is None:
                lat["touch"], touch_by = x["t_start"] - t0, x["role"]
            if edited:
                lat["edit"] = x["t_start"] - t0
                break
        later = [x for x in tools if t_next is not None and x["t_start"] >= t_next]
        v = verdicts.get(h["subtask_id"])
        row.update({f"{m}_sec": lat[m] for m in LATENCIES}, touch_by=touch_by,
                   continued_after_next=None if t_next is None else any(tool_touches(x, files)[0] for x in later),
                   resolved=None if v is None else bool(v.get("resolved")),
                   partial=None if v is None else bool(v.get("partial")),
                   counted=None if v is None else bool(v.get("resolved")) and bool(visible_ok))
        rows.append(row)
    return rows


def _median_censored(x: pd.Series) -> float:
    """Median with missing values as infinitely late; NaN when that median is infinite or no rows."""
    if not len(x):
        return math.nan
    m = float(np.median(x.astype(float).fillna(math.inf)))
    return m if math.isfinite(m) else math.nan


def condition_table(hints: pd.DataFrame) -> pd.DataFrame:
    """Per condition, over the valid episodes' hints."""
    if hints.empty:
        return pd.DataFrame()
    rows = []
    for cond, g in hints[hints["valid"].astype(bool)].groupby("condition", sort=False):
        row = {"condition": cond, "topology": g["topology"].iloc[0], "spawn_mode": g["spawn_mode"].iloc[0],
               "n_episodes": g["trial_name"].nunique(), "n_hints": len(g),
               "delivered_frac": float(g["delivered"].mean()),
               "busy_at_arrival_frac": float(g.loc[g["hint_index"] > 0, "busy_at_arrival"].isin(BUSY).mean())
               if (g["hint_index"] > 0).any() else math.nan}
        for m in LATENCIES:
            x = g[f"{m}_sec"]
            row[f"{m}_frac"] = float(x.notna().mean())
            row[f"{m}_median_sec"] = _median_censored(x)
            row[f"{m}_mean_sec"], row[f"{m}_mean_ci_lo"], row[f"{m}_mean_ci_hi"] = bootstrap_mean(x.dropna())
            row[f"{m}_before_next"] = float((x.astype(float) < g["interval_sec"]).mean())
        own = g[~g["shared_files"].astype(bool)]
        for m in ("touch", "edit"):
            row[f"{m}_median_sec_unshared"] = _median_censored(own[f"{m}_sec"])
            row[f"{m}_before_next_unshared"] = (float((own[f"{m}_sec"].astype(float) < own["interval_sec"]).mean())
                                                if len(own) else math.nan)
        cont = g["continued_after_next"].dropna()
        row["continued_after_next_frac"] = float(cont.astype(bool).mean()) if len(cont) else math.nan
        graded = g["counted"].dropna()
        row["n_graded"] = len(graded)
        row["hinted_fixed_frac"] = float(graded.astype(bool).mean()) if len(graded) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def position_table(hints: pd.DataFrame) -> pd.DataFrame:
    """Per condition and hint position: does the reaction slow down as requests pile up?"""
    if hints.empty:
        return pd.DataFrame()
    rows = []
    v = hints[hints["valid"].astype(bool)]
    for (cond, k), g in v.groupby(["condition", "hint_index"], sort=False):
        graded = g["counted"].dropna()
        rows.append({"condition": cond, "hint_index": k, "n": len(g),
                     **{f"{m}_median_sec": _median_censored(g[f"{m}_sec"]) for m in LATENCIES},
                     "touch_before_next": float((g["touch_sec"].astype(float) < g["interval_sec"]).mean()),
                     "hinted_fixed_frac": float(graded.astype(bool).mean()) if len(graded) else math.nan})
    return pd.DataFrame(rows).sort_values(["condition", "hint_index"], kind="stable").reset_index(drop=True)
