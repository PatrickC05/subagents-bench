"""Write a compact, grader-readable timeline of each trial: python timeline.py JOBS_DIR OUT_DIR

Times are seconds since the session start. Per trial: the user's messages, the replayed ground
truth per bug (from fix_states.json, if fix-times ran), and the orchestrator's turns, tool calls
and results, the replies, and the subagent dispatches and reports (subagents' own steps omitted).
"""
import json
import re
import sys
from pathlib import Path

JOBS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
LV = {"baseline": "L0", "eta": "L1", "questions": "L2", "urgent": "L3"}


def cut(s, n):
    s = "" if s is None else str(s)
    s = re.sub(r"/(?:U" r"sers|home)/[^\s\"']+", "<local path>", s)  # local paths
    return s if len(s) <= n else s[: n * 2 // 3] + f" […{len(s) - n} chars cut…] " + s[-n // 3:]


def ld(a, n):
    p = a / f"{n}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


index = []
for T in sorted(JOBS.glob("*__r*__*/swe-sweep__*")):
    a = T / "agent"
    if not (a / "episode.json").exists():
        continue
    ep = json.loads((a / "episode.json").read_text())
    if not ep.get("exit_reason"):
        continue  # still running
    t0 = ep.get("session_t0") or ep["t_start"]
    rel = lambda t: None if t is None else round(t - t0, 1)
    cond = ep["condition"]
    m = re.match(r"^(.+)_(baseline|eta|questions|urgent)$", cond)
    setting, level = (m[1], LV[m[2]]) if m else (cond, "-")
    task = ep["task_name"].replace("swe-sweep__", "")
    inbox = ld(a, "inbox")
    qref = {}
    hints_dir = JOBS / "hints"
    qf = hints_dir / f"{ep['task_name']}.questions.json"
    if qf.exists():
        for q in json.loads(qf.read_text())["questions"]:
            qref[q["question"]] = q
    lines = [f"# Trial {T.parent.name}/{T.name}", f"setting={setting} level={level} task={task} "
             f"session_length_sec={ep.get('config', {}).get('interactive', {}).get('wall_clock_sec')} exit={ep.get('exit_reason')}",
             f"Repository source (unmodified) for checking answers: {JOBS / 'src' / ep['task_name']}", "", "## User messages"]
    msgs = []
    for h in inbox:
        if h.get("msg_id") is None:
            continue
        kind = "bug" if h["kind"] == "hint" else h["kind"]
        d = {"msg_id": h["msg_id"], "kind": kind, "sent_t": rel(h["t_ready"]), "delivered_t": rel(h["t_delivered"]),
             "urgent": bool(h.get("urgent"))}
        if kind in ("cancel", "undo"):
            d.update(target_msg_id=h.get("target_msg_id"), subtask_id=h.get("subtask_id"))
        if kind == "bug" and len(h.get("subtask_ids") or []) > 1:
            d["item"] = h.get("item")
        if kind == "bug":
            d["subtask_id"] = h.get("subtask_id")
            d["test"] = (h.get("f2p") or [None])[0]
            d["files"] = h.get("target_files")
        if kind == "question":
            q = qref.get(h.get("question")) or {}
            d.update(question=h.get("question"), reference_answer=h.get("reference") or q.get("answer"), file=q.get("file"))
        msgs.append(d)
        lines.append(json.dumps(d))
    fs = a / "fix_states.json"
    lines += ["", "## Ground truth: replayed status of each bug's hidden test over the workspace states"]
    if fs.exists():
        st = json.loads(fs.read_text())
        last = {}
        for s in st["states"]:
            for sid, b in (s.get("bugs") or {}).items():
                if last.get(sid) != b["status"]:
                    lines.append(f"t={rel(s['t'])} {sid}: {b['status']}")
                    last[sid] = b["status"]
        lines.append(f"(final verdict matches grader: {st.get('final_matches_verifier')})")
    else:
        lines.append("(not available: no snapshots for this trial; use the final grade only)")
    res = json.loads((T / "result.json").read_text()) if (T / "result.json").exists() else {}
    lines.append(f"final grader: {json.dumps((res.get('verifier_result') or {}).get('rewards'))}")
    vf = T / "verifier"
    lines += ["", "## Timeline (orchestrator turns; subagent work summarised as dispatch start/report)"]
    ev = []
    for r in ld(a, "transcript"):
        if r["agent_id"] != "orchestrator" or r["msg_role"] == "system":
            continue
        t = rel(r["t"])
        if r["msg_role"] == "assistant":
            calls = "; ".join(f"{c['name']}({cut(json.dumps(c['arguments'], ensure_ascii=False), 500)})" for c in r.get("tool_calls") or [])
            ev.append((t, f"[{t}] ORCHESTRATOR TURN\n  visible text: {cut(r.get('content'), 600)}\n  hidden reasoning: {cut(r.get('reasoning'), 1500)}\n  tool calls: {calls}"))
        elif r["msg_role"] == "tool":
            res_s = " | ".join(cut(x.get("content"), 300) for x in r.get("tool_results") or [])
            ev.append((t, f"[{t}] tool results: {res_s}"))
        elif r["msg_role"] == "user":
            ev.append((t, f"[{t}] DELIVERED TO ORCHESTRATOR: {cut(r.get('content'), 900)}"))
    for d in ld(a, "dispatches"):
        ev.append((rel(d["t_start"]), f"[{rel(d['t_start'])}] subagent {d['agent_id']} started"
                   + (f" (helper of {d['parent_agent_id']})" if d.get("parent_agent_id") else "")
                   + f"; ended at {rel(d['t_end'])} ({d['status']}, {d['steps_used']} steps); instruction: {cut(d['instruction'], 400)}"))
    for r in ld(a, "replies"):
        it = f" item {r['item']}" if r.get("item") else ""
        ev.append((rel(r["t"]), f"[{rel(r['t'])}] >>> REPLY SENT TO USER re #{r['to']}{it} (eta_min={r.get('eta_min')}, done={r.get('done')}): {cut(r['text'], 600)}"))
    for c in ld(a, "calls"):
        if c.get("agent_id") == "orchestrator" and c.get("error") == "interrupted":
            ev.append((rel(c["t_end"]), f"[{rel(c['t_end'])}] (the orchestrator's model call was INTERRUPTED by a due user message; its partial output was discarded)"))
    for d in ld(a, "dispatches"):
        if d.get("t_stop_requested"):
            ev.append((rel(d["t_stop_requested"]), f"[{rel(d['t_stop_requested'])}] (stop requested for subagent {d['agent_id']}; it ended at {rel(d['t_end'])} with status {d['status']})"))
    for x in ld(a, "tools"):
        if x["name"] == "reply" and x.get("is_error") and x["agent_id"] == "orchestrator":
            ev.append((rel(x["t_start"]), f"[{rel(x['t_start'])}] (a reply call was REFUSED by the harness, the user did not see it)"))
    ev.sort(key=lambda e: e[0] if e[0] is not None else 1e9)
    lines += [e[1] for e in ev]
    name = f"{T.parent.name.split('__r')[0].replace('swe_interactive_', '')}__{cond}__{task}"
    (OUT / f"{name}.txt").write_text("\n".join(lines))
    index.append({"name": name, "trial": f"{T.parent.name}/{T.name}", "trial_dir": str(T.resolve()), "setting": setting, "level": level, "task": task,
                  "messages": msgs, "chars": sum(len(x) for x in lines)})
(OUT / "index.json").write_text(json.dumps(index, indent=1))
print(len(index), "timelines;", "max chars", max([i["chars"] for i in index] or [0]))
