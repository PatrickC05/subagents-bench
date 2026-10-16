"""Each session's transcript as compact JSON for the monitor's transcript viewer (``transcript_viewer.html``).

One file per trial: ``<out>/<job>__<trial>.json`` (``file_name``: job names start with the sweep's
name, so the pages of several sweeps can share one ``<out>``). A file is rewritten only when its
sources (transcript.jsonl, dispatches.jsonl, spawns.jsonl, episode.json, inbox.jsonl) changed since
the last write (``manifest.json`` lists every file with its source signature, the other pages' files
included), so a republish sends little.

Format (short keys keep it small)::

  {"v": 2, "job", "trial", "t0", "live", "sys": <orchestrator system prompt>,
   "msgs": [msg...],                       # the orchestrator thread, in order, system prompt excluded
   "subs": [{"agent", "dispatch", "instruction", "status", "steps", "t_start", "t_end",
             "files", "noise", "after": <index in msgs or null>, "msgs": [msg...]}],
   "urgent": [msg_id...]}                  # inbox messages marked urgent
  msg = {"t": sec since session_t0, "r": "user"|"assistant"|"tool"|"system",
         "c": content, "k": reasoning, "tc": [[call id, name, args]], "tr": [[call id, content, is_error]]}

``files`` leaves out git/test side effects (.git, .pytest_cache, __pycache__, *.pyc); ``noise`` counts
them. ``after``: the spawn_subagents call that made the run (the k-th call made the k-th group in
spawns.jsonl), else the last spawn result at or before the run started. No message is dropped; a
text field longer than ``CAP`` characters is cut with an explicit marker.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

CAP = 30000
FORMAT = 2  # bump when the export format changes, so every file is rewritten
SOURCES = ("transcript.jsonl", "dispatches.jsonl", "spawns.jsonl", "episode.json", "inbox.jsonl")
# paths a subagent "changes" only as a side effect of running git or tests: left out of its changed files
NOISE = re.compile(r"(^|/)(\.git|\.pytest_cache|__pycache__|\.mypy_cache|[^/]*\.egg-info)(/|$)|\.py[co]$")


def cap(s):
    """``s`` (a non-string as JSON) cut to ``CAP`` characters with a marker saying how much is left out."""
    if s is None:
        return None
    if not isinstance(s, str):
        s = json.dumps(s, ensure_ascii=False)
    if len(s) <= CAP:
        return s
    return s[:CAP] + f"\n[... {len(s) - CAP} more characters not shown in the viewer; full text in transcript.jsonl]"


def _cap_args(a):
    if isinstance(a, dict):
        return {k: (cap(v) if isinstance(v, str) else _cap_args(v)) for k, v in a.items()}
    if isinstance(a, list):
        return [cap(v) if isinstance(v, str) else _cap_args(v) for v in a]
    return a


def _jsonl(p: Path) -> list[dict]:
    try:
        lines = p.read_text().splitlines() if p.is_file() else []
    except OSError:
        return []
    out = []
    for line in lines:
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass  # the line being written
    return out


def _sig(agent: Path) -> str:
    parts = [f"format:{FORMAT}"]
    for n in SOURCES:
        try:
            st = (agent / n).stat()
            parts.append(f"{n}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            parts.append(f"{n}:-")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def _msg(m: dict, t0: float) -> dict:
    out = {"t": round(float(m.get("t") or t0) - t0, 2), "r": m.get("msg_role")}
    if m.get("content") is not None:
        out["c"] = cap(m["content"])
    if m.get("reasoning"):
        out["k"] = cap(m["reasoning"])
    if m.get("tool_calls"):
        out["tc"] = [[tc.get("id"), tc.get("name"), _cap_args(tc.get("arguments"))] for tc in m["tool_calls"]]
    if m.get("tool_results"):
        out["tr"] = [[r.get("call_id"), cap(r.get("content")), bool(r.get("is_error"))] for r in m["tool_results"]]
    return out


def export_trial(trial_dir: Path) -> dict:
    """One trial's viewer JSON (the format above); tolerant of a live session's half-written files."""
    agent = trial_dir / "agent"
    try:
        ep = json.loads((agent / "episode.json").read_text())
    except (OSError, ValueError):
        ep = {}
    rows = _jsonl(agent / "transcript.jsonl")
    t0 = ep.get("session_t0") or ep.get("t_start") or (rows[0]["t"] if rows else 0.0)
    orch = [m for m in rows if m.get("agent_id") == "orchestrator"]
    sys_prompt = None
    if orch and orch[0].get("msg_role") == "system":
        sys_prompt = cap(orch[0].get("content"))
        orch = orch[1:]
    msgs = [_msg(m, t0) for m in orch]
    # where each spawn call's result sits in the orchestrator thread (its subagents show right after it)
    spawn_at: dict[str, int] = {}
    spawn_calls: list[str] = []  # spawn_subagents call ids, in call order
    for i, m in enumerate(orch):
        for tc in m.get("tool_calls") or []:
            if tc.get("name") == "spawn_subagents":
                spawn_at.setdefault(tc.get("id"), i)
                spawn_calls.append(tc.get("id"))
        for r in m.get("tool_results") or []:
            if r.get("call_id") in spawn_at:
                spawn_at[r["call_id"]] = i
    spawn_idx = sorted(spawn_at.values())
    disp = {d.get("dispatch_id"): d for d in _jsonl(agent / "dispatches.jsonl")}
    placed: dict[str, int] = {}
    for call_id, sp in zip(spawn_calls, _jsonl(agent / "spawns.jsonl")):
        for did in sp.get("dispatch_ids") or []:
            placed[did] = spawn_at[call_id]
    urgent = sorted({e["msg_id"] for e in _jsonl(agent / "inbox.jsonl")
                     if e.get("urgent") and e.get("msg_id") is not None})
    groups: dict[tuple, list] = {}
    for m in rows:
        if m.get("agent_id") != "orchestrator":
            groups.setdefault((m.get("agent_id"), m.get("dispatch_id")), []).append(m)
    for d in disp.values():  # a dispatch with no message yet still shows
        groups.setdefault((d.get("agent_id"), d.get("dispatch_id")), [])
    subs = []
    for (aid, did), ms in groups.items():
        d = disp.get(did) or {}
        ts = d.get("t_start") or (ms[0].get("t") if ms else None)
        after = placed.get(did)
        if after is None and ts is not None:
            cands = [i for i in spawn_idx if orch[i].get("t", 0) <= ts + 1.0]
            after = cands[-1] if cands else None
        files = d.get("files_modified") or []
        kept = [f for f in files if not NOISE.search(str(f))]
        subs.append({
            "agent": aid, "dispatch": did, "instruction": cap(d.get("instruction")),
            "status": d.get("status") or ("running" if not d.get("t_end") else None),
            "steps": d.get("steps_used"), "files": kept, "noise": len(files) - len(kept),
            "t_start": round(ts - t0, 2) if ts else None,
            "t_end": round(d["t_end"] - t0, 2) if d.get("t_end") else None,
            "after": after, "msgs": [_msg(m, t0) for m in ms],
        })
    subs.sort(key=lambda s: (s["t_start"] is None, s["t_start"] or 0))
    return {"v": FORMAT, "job": trial_dir.parent.name, "trial": trial_dir.name, "t0": t0, "live": not ep.get("t_end"),
            "sys": sys_prompt, "urgent": urgent, "msgs": msgs, "subs": subs}


def file_name(trial_dir: Path) -> str:
    """``<job>__<trial>.json``, with the jobs dir's name in front unless the job's name starts with it.
    A sweep's jobs are ``<sweep>__r<k>__<condition>`` in ``jobs/<sweep>`` (unless its spec sets
    ``jobs_dir``), so the pages of several sweeps written to one out dir never share a file."""
    job, jobs = trial_dir.parent.name, trial_dir.parent.parent.name
    return f"{'' if job.startswith(f'{jobs}__') else f'{jobs}__'}{job}__{trial_dir.name}.json"


def export_all(trial_dirs: list[Path], out_dir: Path) -> tuple[dict[str, dict], int]:
    """Write the changed transcripts (trials with a transcript.jsonl) and the manifest; return
    ``{str(trial_dir): {"file", "job", "trial", "sig", "bytes"}}`` and how many files were rewritten.
    The manifest keeps the entries of other jobs whose file is still there (another sweep's page in
    the same ``out_dir``), so writing one page never makes the next rewrite every file of another."""
    out_dir.mkdir(parents=True, exist_ok=True)
    man_p = out_dir / "manifest.json"
    try:
        old = {e["file"]: e for e in json.loads(man_p.read_text()).get("files", [])}
    except (OSError, ValueError, AttributeError, KeyError, TypeError):
        old = {}
    entries, by_dir, wrote = [], {}, 0
    for td in trial_dirs:
        if not (td / "agent" / "transcript.jsonl").is_file():
            continue
        name, sig = file_name(td), _sig(td / "agent")
        p = out_dir / name
        if not (p.is_file() and old.get(name, {}).get("sig") == sig):
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(export_trial(td), ensure_ascii=False, separators=(",", ":")))
            tmp.replace(p)
            wrote += 1
        e = {"file": name, "job": td.parent.name, "trial": td.name, "sig": sig, "bytes": p.stat().st_size}
        entries.append(e)
        by_dir[str(td)] = e
    jobs = {td.parent.name for td in trial_dirs}
    names = {e["file"] for e in entries}
    others = [e for f, e in old.items() if f not in names and isinstance(e, dict) and e.get("job") not in jobs
              and (out_dir / f).is_file()]
    man_p.write_text(json.dumps({"files": entries + others}, indent=1))
    return by_dir, wrote
