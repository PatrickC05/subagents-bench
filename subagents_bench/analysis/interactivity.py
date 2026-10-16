"""Interactivity of interactive episodes: per hint (a SWE-sweep subtask sent as a user message
at ``t_ready``), how long until the orchestrator received it, answered it, and until any agent
dispatched on, touched and edited the subtask's files; what happened to it when the next hint
arrived; and whether the grader counts the subtask fixed.

All latencies are seconds after ``t_ready`` (when the hint was sent):

- ``delivery``: until it entered the orchestrator's conversation (it waits for the step in progress);
- ``response``: until the orchestrator's first LLM call that saw it returned ("it says something");
- ``dispatch``: until a subagent started whose instruction names one of the target files or tests;
- ``touch``: until any agent's tool call named one of the target files (a path argument or a bash
  command; ``search`` over the whole repository does not count);
- ``edit``: until any agent edited one: ``edit_file`` on it, or a bash command segment naming it
  that writes (``sed -i``, a redirection or ``tee`` to it, ``patch``/``git apply``, ``gofmt -w``,
  a ``cp``/``mv`` onto it, a Python write).

At the next hint's arrival a hint is ``not_started`` (no touch yet), ``continued`` (touched before
and again after) or ``stopped`` (touched before, never after); with the final verdict, a stopped
hint that ends fixed was finished, one that does not was dropped. Work on a hint's files before
it was sent (``pre_touched``/``pre_edited``) is recorded rather than credited: with
``interactive.scope: all`` the agent may fix bugs before they are sent.

Matching is by path text, so it is a heuristic: a target file shared with another hint's subtask
(``shared_files``: an earlier one; ``shares_any``: any) can be touched for that subtask. The
summary reports touch and edit for unshared hints separately and the status at the next hint
only for hints that share no file. A bare basename (after a ``cd``) counts only when no other
hinted target file has that basename. Summaries leave out hint 0 (sent with the task, while the
agent starts up; it has its own ``first_*`` columns). A latency is missing when the event never
happened before the session ended: medians count those as infinitely late (NaN when more than
half are missing), and ``*_before_next`` is the share of hints with the event within one
``interval_sec``, before the next hint was sent.
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
STATUSES = ("not_started", "continued", "stopped")
PATH_ARGS = {"read_file": "path", "edit_file": "path", "list_dir": "path", "search": "path"}
WRITE_RE = re.compile(r"\bsed\b[^\n]*(?:\s-[a-zA-Z]*i|--in-place)|\bperl\s+-[a-zA-Z]*i|\bpatch\b|\bgit\s+apply\b|"
                      r"\b(?:gofmt|goimports)\b[^\n]*\s-w\b|write_text\(|(?<!stdout)(?<!stderr)\.write\(|"
                      r"open\([^)]*['\"][wa]b?\+?['\"]")
COPY_RE = re.compile(r"\s*(?:sudo\s+)?(?:cp|mv|install)\s(.*)")
BUSY = ("llm", "tool", "spawn")


def _tokens(path: str) -> tuple[list[str], str]:
    """Paths that name ``path`` (it and its last two components), and its basename."""
    parts = path.strip("/").split("/")
    return list(dict.fromkeys(["/".join(parts), "/".join(parts[-2:])])) if len(parts) > 1 else [path], parts[-1]


def names_file(text: str, path: str, basename_ok: bool = True) -> bool:
    """``text`` names ``path``: by its path or last two components (as a path, e.g. after the
    repository's absolute root), or (``basename_ok``) by its bare basename, not inside another path."""
    subs, base = _tokens(path)
    end = r"(?![\w\-]|\.\w)"
    if any(re.search(rf"(?<![\w.\-]){re.escape(x)}{end}", text) for x in subs):
        return True
    return basename_ok and re.search(rf"(?<![\w.\-/]){re.escape(base)}{end}", text) is not None


def segments(cmd: str) -> list[str]:
    """``cmd`` split at unquoted ``;``, ``&&``, ``||``, ``|`` and newlines; a command with a heredoc
    stays whole (its body belongs to it)."""
    if "<<" in cmd:
        return [cmd]
    out, cur, quote, i = [], [], None, 0
    while i < len(cmd):
        c = cmd[i]
        if quote:
            if c == "\\" and quote == '"' and i + 1 < len(cmd):
                cur.append(cmd[i:i + 2])
                i += 2
                continue
            quote = None if c == quote else quote
        elif c in "'\"":
            quote = c
        elif c in ";|\n" or cmd.startswith("&&", i):
            out.append("".join(cur))
            cur = []
            i += 2 if cmd.startswith(("&&", "||"), i) else 1
            continue
        elif c == "\\" and i + 1 < len(cmd):
            cur.append(cmd[i:i + 2])
            i += 2
            continue
        cur.append(c)
        i += 1
    return [x for x in out + ["".join(cur)] if x.strip()]


def _writes(segment: str, path: str, basename_ok: bool) -> bool:
    base = re.escape(_tokens(path)[1])
    if re.search(rf">>?\s*['\"]?[^\s'\"&|;<>]*{base}(?![\w\-]|\.\w)", segment):
        return True
    if re.search(rf"\btee\s+(?:-a\s+)?['\"]?[^\s'\"]*{base}(?![\w\-]|\.\w)", segment):
        return True
    m = COPY_RE.match(segment)
    if m:  # only the destination is written
        dest = m.group(1).split()[-1:] or [""]
        return names_file(dest[0], path, basename_ok)
    return WRITE_RE.search(segment) is not None


HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_]\w*)\1")


def strip_data_heredocs(cmd: str) -> str:
    """``cmd`` without the bodies of heredocs that ``cat``/``tee`` save to a file: data, not commands
    (a scratch script ``cat <<'EOF' > /tmp/s.py`` listing repository paths edits none of them). The
    header line stays, so ``cat <<EOF > pint/x.py`` still edits pint/x.py."""
    lines, out, i = cmd.split("\n"), [], 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        m = HEREDOC_RE.search(line)
        if not m:
            continue
        before, after = re.split(r"&&|\|\||;|\||\(", line[:m.start()])[-1], line[m.end():]
        if (re.match(r"\s*(?:cat|tee)\b", before) and "|" not in after
                and (">" in before + after or before.strip().startswith("tee"))):
            while i < len(lines) and lines[i].strip() != m.group(2):
                i += 1  # the body; the terminator line is kept
    return "\n".join(out)


def tool_touches(tool: dict, files: list[str], basename_ok: set[str] | None = None) -> tuple[bool, bool]:
    """(touches, edits) one of ``files``; ``basename_ok``: the files a bare basename may stand for
    (default: all)."""
    args = tool.get("arguments") or {}
    if not isinstance(args, dict):
        return False, False
    ok = (lambda f: True) if basename_ok is None else (lambda f: f in basename_ok)  # noqa: E731
    name = tool["name"]
    if name in PATH_ARGS:
        p = args.get(PATH_ARGS[name])
        hit = isinstance(p, str) and any(names_file(p, f, ok(f)) for f in files)
        return hit, hit and name == "edit_file"
    if name == "bash":
        cmd = args.get("command")
        if not isinstance(cmd, str):
            return False, False
        cmd = strip_data_heredocs(cmd) if "<<" in cmd else cmd
        touched = edited = False
        for seg in segments(cmd):
            named = [f for f in files if names_file(seg, f, ok(f))]
            if named:
                touched = True
                edited = edited or any(_writes(seg, f, ok(f)) for f in named)
        return touched, edited
    return False, False


def _busy(t: float, calls: list[dict], tools: list[dict], spawns: dict[str, dict]) -> tuple[str, str | None, float | None]:
    """What the orchestrator was doing at ``t`` (llm, tool, spawn = a blocking spawn, wait or idle),
    the call or tool's name (``llm`` for a call), and how long it still ran after ``t``."""
    for c in calls:
        if c["role"] == "orchestrator" and c["t_start"] <= t < c["t_end"]:
            return "llm", "llm", c["t_end"] - t
    for x in tools:
        if x["role"] == "orchestrator" and x["t_start"] <= t < x["t_end"]:
            if x["name"] == "wait":
                return "wait", "wait", x["t_end"] - t
            if x["name"] == "spawn_subagents" and (spawns.get(x["group_id"]) or {}).get("spawn_mode", "blocking") == "blocking":
                return "spawn", x["name"], x["t_end"] - t
            return "tool", x["name"], x["t_end"] - t
    return "idle", None, None


def eval_subtasks(trial_dir: Path) -> tuple[dict[str, dict], bool | None, bool]:
    """The grader's per-subtask verdicts (``verifier/eval.json``), its visible_ok, and whether the
    file exists."""
    try:
        ev = json.loads((trial_dir / "verifier" / "eval.json").read_text())
    except (OSError, ValueError):
        return {}, None, False
    subs = {s["subtask_id"]: s for s in ev.get("subtasks") or [] if isinstance(s, dict) and "subtask_id" in s}
    return subs, ev.get("visible_ok"), True


def _test_names(f2p: list[str]) -> list[str]:
    """Distinctive names of the hidden tests (the last component of each id, at least 8 chars)."""
    out = []
    for t in f2p:
        name = re.split(r"::|\.|\s", re.sub(r"\[.*\]$", "", t).strip())[-1]
        if len(name) >= 8:
            out.append(name)
    return list(dict.fromkeys(out))


def hint_rows(t: Trial, keys: dict[str, Any], valid: bool) -> list[dict[str, Any]]:
    """One row per hint of an interactive episode (none for a batch episode)."""
    ep, r = t.episode or {}, t.records
    icfg = (ep.get("config") or {}).get("interactive")
    hints = sorted((x for x in r.get("inbox", []) if x["kind"] == "hint"), key=lambda x: x["hint_index"])
    if icfg is None or not hints:
        return []
    interval = icfg["interval_sec"]
    session_t0 = ep.get("session_t0", hints[0]["t_ready"])
    steps = {s["step"]: s for s in r["steps"]}
    spawns = {s["group_id"]: s for s in r["spawns"]}
    tools = sorted(r["tools"], key=lambda x: x["t_start"])
    dispatches = sorted(r["dispatches"], key=lambda d: d["t_start"])
    verdicts, visible_ok, graded = eval_subtasks(t.dir)
    sub = ep.get("config", {}).get("subagent") or {}
    all_files = [f for h in hints for f in dict.fromkeys(h["target_files"] or [])]
    base_count: dict[str, int] = {}
    for f in dict.fromkeys(all_files):
        base_count[_tokens(f)[1]] = base_count.get(_tokens(f)[1], 0) + 1
    basename_ok = {f for f in all_files if base_count[_tokens(f)[1]] == 1}
    file_hints: dict[str, int] = {}
    for h in hints:
        for f in set(h["target_files"] or []):
            file_hints[f] = file_hints.get(f, 0) + 1
    rows, earlier_files = [], set()
    for k, h in enumerate(hints):
        t0, files = h["t_ready"], h["target_files"] or []
        t_next = hints[k + 1]["t_ready"] if k + 1 < len(hints) else None
        row = {**keys, "valid": valid, "topology": ep.get("topology"),
               "spawn_mode": sub.get("spawn_mode", "blocking") if ep.get("topology") == "subagents" else None,
               "hint_detail": icfg.get("hint_detail"), "scope": icfg.get("scope", "all"), "interval_sec": interval,
               "hint_index": h["hint_index"], "urgent": bool(h.get("urgent")), "subtask_id": h["subtask_id"], "target_files": ";".join(files),
               "n_target_files": len(files), "shared_files": bool(set(files) & earlier_files),
               "shares_any": any(file_hints[f] > 1 for f in files), "t_ready_sec": t0 - session_t0,
               "delivered": h["t_delivered"] is not None}
        busy, blocker, left = _busy(t0, r["calls"], r["tools"], spawns) if k else ("start", None, None)
        # what the hint waited behind: the orchestrator's call or tool in progress, and how much of it was left
        row.update(busy_at_arrival=busy, blocked_by=blocker if busy in BUSY else None,
                   blocked_sec=left if busy in BUSY else None)
        earlier_files |= set(files)
        lat: dict[str, float | None] = dict.fromkeys(LATENCIES)
        if h["t_delivered"] is not None:
            lat["delivery"] = h["t_delivered"] - t0
            s = steps.get(h["step"])
            if s is not None and s.get("t_llm_end") is not None and s.get("call_id") is not None:
                lat["response"] = s["t_llm_end"] - t0
        names = _test_names(h["f2p"] or [])
        for d in dispatches:
            text = d.get("instruction") or ""
            if d["t_start"] >= t0 and (any(names_file(text, f, f in basename_ok) for f in files)
                                       or any(re.search(rf"\b{re.escape(n)}\b", text) for n in names)):
                lat["dispatch"] = d["t_start"] - t0
                break
        touch_by, pre_touched, pre_edited, before_next, after_next = None, False, False, False, False
        for x in tools:
            touched, edited = tool_touches(x, files, basename_ok)
            if not touched:
                continue
            if x["t_start"] < t0:
                pre_touched, pre_edited = True, pre_edited or edited
                continue
            if lat["touch"] is None:
                lat["touch"], touch_by = x["t_start"] - t0, x["role"]
            if edited and lat["edit"] is None:
                lat["edit"] = x["t_start"] - t0
            if t_next is not None:
                before_next = before_next or x["t_start"] < t_next
                after_next = after_next or x["t_start"] >= t_next
        status = None if t_next is None else "not_started" if not before_next else "continued" if after_next else "stopped"
        v = verdicts.get(h["subtask_id"])
        # A valid trial whose grader reported no verdict for the subtask (no eval.json, or a patch that
        # broke the build) counts it unfixed, as the reward does; an invalid trial's stay unknown.
        resolved = bool(v.get("resolved")) if v else (False if valid else None)
        row.update({f"{m}_sec": lat[m] for m in LATENCIES}, touch_by=touch_by, pre_touched=pre_touched,
                   pre_edited=pre_edited, status_at_next=status,
                   continued_after_next=None if status in (None, "not_started") else status == "continued",
                   graded=graded and v is not None, resolved=resolved,
                   partial=bool(v.get("partial")) if v else (False if valid else None),
                   counted=None if resolved is None else resolved and bool(visible_ok))
        rows.append(row)
    return rows


def _median_censored(x: pd.Series) -> float:
    """Median with missing values as infinitely late; NaN when that median is infinite or no rows."""
    if not len(x):
        return math.nan
    m = float(np.median(pd.to_numeric(x, errors="coerce").astype(float).fillna(math.inf)))
    return m if math.isfinite(m) else math.nan


def _frac(mask: pd.Series) -> float:
    return float(mask.astype(bool).mean()) if len(mask) else math.nan


def _before(x: pd.Series, interval: pd.Series) -> float:
    return float((pd.to_numeric(x, errors="coerce").astype(float) < interval.astype(float)).mean()) if len(x) else math.nan


def _flag(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].fillna(False).astype(bool)


def condition_table(hints: pd.DataFrame) -> pd.DataFrame:
    """Per condition, over the valid episodes' hints after the first (hint 0 has ``first_*``)."""
    if hints.empty or not hints["valid"].astype(bool).any():
        return pd.DataFrame()
    rows = []
    for cond, all_g in hints[hints["valid"].astype(bool)].groupby("condition", sort=False):
        first, g = all_g[all_g["hint_index"] == 0], all_g[all_g["hint_index"] > 0]
        row = {"condition": cond, "topology": all_g["topology"].iloc[0], "spawn_mode": all_g["spawn_mode"].iloc[0],
               "n_episodes": all_g["trial_name"].nunique(), "n_hints": len(all_g), "n_later_hints": len(g),
               "delivered_frac": _frac(g["delivered"]), "busy_at_arrival_frac": _frac(g["busy_at_arrival"].isin(BUSY)),
               "pre_edited_frac": _frac(_flag(g, "pre_edited")),
               "blocked_median_sec": _median_censored(g["blocked_sec"].dropna()) if "blocked_sec" in g else math.nan}
        for m in ("response", "touch", "edit"):
            row[f"first_{m}_median_sec"] = _median_censored(first[f"{m}_sec"])
        fresh = g[~_flag(g, "pre_edited")]  # touch and edit of bugs not already worked on before they were sent
        for m in LATENCIES:
            x = (fresh if m in ("touch", "edit") else g)[f"{m}_sec"]
            interval = (fresh if m in ("touch", "edit") else g)["interval_sec"]
            row[f"{m}_frac"] = float(x.notna().mean()) if len(x) else math.nan
            row[f"{m}_median_sec"] = _median_censored(x)
            row[f"{m}_mean_sec"], row[f"{m}_mean_ci_lo"], row[f"{m}_mean_ci_hi"] = bootstrap_mean(
                pd.to_numeric(x, errors="coerce").dropna())
            row[f"{m}_before_next"] = _before(x, interval)
        own = fresh[~_flag(fresh, "shared_files")]
        for m in ("touch", "edit"):
            row[f"{m}_median_sec_unshared"] = _median_censored(own[f"{m}_sec"])
            row[f"{m}_before_next_unshared"] = _before(own[f"{m}_sec"], own["interval_sec"])
        alone = all_g[~_flag(all_g, "shares_any") & all_g["status_at_next"].notna()]
        row["n_status_unshared"] = len(alone)
        for st in STATUSES:
            row[f"{st}_at_next_frac"] = _frac(alone["status_at_next"] == st)
        stopped = alone[alone["status_at_next"] == "stopped"]
        counted = stopped["counted"].dropna()
        row["stopped_fixed_frac"] = _frac(counted) if len(counted) else math.nan
        counted = all_g["counted"].dropna()  # every hint of a valid trial: no verdict counts as unfixed
        row["n_graded"] = int(_flag(all_g, "graded").sum())
        row["hinted_fixed_frac"] = _frac(counted) if len(counted) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def position_table(hints: pd.DataFrame) -> pd.DataFrame:
    """Per condition and hint position: does the reaction slow down as requests pile up?"""
    if hints.empty or not hints["valid"].astype(bool).any():
        return pd.DataFrame()
    rows = []
    v = hints[hints["valid"].astype(bool)]
    for (cond, k), g in v.groupby(["condition", "hint_index"], sort=False):
        graded = g["counted"].dropna()
        rows.append({"condition": cond, "hint_index": k, "n": len(g),
                     **{f"{m}_median_sec": _median_censored(g[f"{m}_sec"]) for m in LATENCIES},
                     "touch_before_next": _before(g["touch_sec"], g["interval_sec"]),
                     "hinted_fixed_frac": _frac(graded) if len(graded) else math.nan})
    return pd.DataFrame(rows).sort_values(["condition", "hint_index"], kind="stable").reset_index(drop=True)
