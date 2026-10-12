"""The arbiter strategy: n candidates per instruction, an LLM judge, confidence-aware debate.

Hard scope: each candidate works in a private ``cp -a`` copy of the workdir (its tool paths
are remapped there). Afterwards, workdir files that changed anyway (a route the remap missed)
are restored from a pristine ``base`` copy and recorded; in each copy, the copy's own path
(written into files by remapped commands) becomes the workdir path again; and the winner's
changes relative to the pristine workdir are applied. Easy and medium scopes share the workdir.
"""

from __future__ import annotations

import json
import re
import shlex
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from .environment import AgentStopped
from .llm import ContextLengthError, LLMResponse, Message, ToolSpec
from .prompts import ARBITER, DEBATE, JUDGE_RETRY_NOTE, debate_prompt, judge_prompt, report_with_revision
from .records import Who
from .subagent import Dispatch, DispatchResult, Runtime, new_dispatch, run_dispatch
from .tools import truncate

JUDGE_SPEC = ToolSpec(
    "submit_judgement",
    "Submit your judgement: the best candidate, your confidence, whether the candidates agreed, and why.",
    {"type": "object", "properties": {
        "winner": {"type": "integer", "description": "0-based index of the best candidate."},
        "confidence": {"type": "number", "description": "Probability (0..1) that the winner is correct and best."},
        "agreement": {"type": "boolean", "description": "True if the candidates reached substantially the same result."},
        "rationale": {"type": "string", "description": "One to three sentences."},
    }, "required": ["winner", "confidence", "agreement", "rationale"]},
)
DIFF_CAP_CHARS = 4000
NO_JUDGEMENT = {"winner": None, "confidence": None, "agreement": None, "parse_ok": False}


def _sh(env: Any, script: str) -> str:
    res = env.exec(script, 900)
    if res.return_code != 0:
        raise RuntimeError(f"arbiter shell step failed (exit {res.return_code}): {(res.stdout + res.stderr)[-500:]}")
    return res.stdout


def manifest(env: Any, root: str) -> dict[str, str]:
    """relpath -> md5 (+ " x" if executable) for every regular file under ``root``."""
    out = _sh(env, f"cd -- {shlex.quote(root)} && find . -type f -exec md5sum {{}} + && echo __SB_X && find . -type f -perm -100")
    sums, _, execs = out.partition("__SB_X\n")
    files = {line[36:]: line[:32] for line in sums.splitlines()}  # "<md5>  ./<path>"
    for path in execs.splitlines():
        files[path[2:]] = files.get(path[2:], "") + " x"
    return files


def _changed(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))


def _copy_files(env: Any, src: str, dst: str, rels: list[str]) -> None:
    """Make ``rels`` under ``dst`` match ``src`` (copied, or deleted where ``src`` has none)."""
    lines = []
    for rel in rels:
        s, d = shlex.quote(f"{src}/{rel}"), shlex.quote(f"{dst}/{rel}")
        lines.append(f'if [ -e {s} ]; then mkdir -p -- "$(dirname -- {d})" && rm -f -- {d} && cp -a -- {s} {d}; '
                     f"else rm -f -- {d}; fi || exit 1")
    if lines:
        _sh(env, "\n".join(lines))


def _map_back(env: Any, copy: str, workdir: str) -> None:
    """Replace ``copy`` by ``workdir`` in the text files under ``copy``."""
    expr = "s|" + re.sub(r"([\\.*\[\]^$|])", r"\\\1", copy) + "|" + re.sub(r"([\\&|])", r"\\\1", workdir) + "|g"
    _sh(env, f"cd -- {shlex.quote(copy)} && grep -rlIF -e {shlex.quote(copy)} . | while IFS= read -r f; do "
             f'sed {shlex.quote(expr)} "$f" > "$f.sb_map" && cat "$f.sb_map" > "$f"; rm -f -- "$f.sb_map"; done')


def _json_objects(text: str) -> list[Any]:
    decoder, out = json.JSONDecoder(), []
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                out.append(decoder.raw_decode(text[i:])[0])
            except ValueError:
                pass
    return out


def parse_judgement(r: LLMResponse, n: int) -> dict[str, Any]:
    """From the submit_judgement call, else a JSON object in the text; winner None if neither."""
    for obj in [c.arguments for c in r.tool_calls if c.name == "submit_judgement"] + _json_objects(r.text):
        w = obj.get("winner") if isinstance(obj, dict) else None
        if isinstance(w, int) and not isinstance(w, bool) and 0 <= w < n:
            conf, agree = obj.get("confidence"), obj.get("agreement")
            conf = float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) and 0 <= conf <= 1 else None
            agree = agree if isinstance(agree, bool) else None
            return {"winner": w, "confidence": conf, "agreement": agree, "parse_ok": conf is not None and agree is not None}
    return dict(NO_JUDGEMENT)


def run_arbitration(rt: Runtime, instruction: str, index: int, group_id: str, step: int,
                    dispatch_ids: list[str], arbitration_ids: list[str]) -> tuple[str, str]:
    """One arbitrated instruction -> (heading, report) for the spawn result; always recorded."""
    a = rt.config.subagent.arbiter
    n, copy, env, q = a.n_candidates, rt.config.subagent.scope == "hard", rt.env, shlex.quote
    aid = rt.rec.new_id("arb")
    arbitration_ids.append(aid)
    root = f"{env.state_root}/arb/{aid}"
    record: dict[str, Any] = {
        "arbitration_id": aid, "group_id": group_id, "instruction_index": index, "candidate_dispatch_ids": [],
        "isolation": "copy" if copy else "shared", "rounds": [], "final_winner": None, "final_confidence": None,
        "debate_rounds": 0, "applied": False, "n_applied_files": 0, "isolation_violated": None, "n_leaked_files": 0,
        "t_start": time.time(), "t_candidates_end": None,
    }
    try:
        if copy:
            pristine = manifest(env, rt.workdir)
            _sh(env, "\n".join(f"mkdir -p {q(f'{root}/{x}')} && cp -a {q(rt.workdir)}/. {q(f'{root}/{x}')}/ || exit 1"
                               for x in ["base"] + [f"cand{k}" for k in range(n)]))
        cands = [new_dispatch(rt, instruction, index, group_id, step, cwd=f"{root}/cand{k}" if copy else None,
                              arbitration_id=aid, candidate_index=k, private_copy=copy) for k in range(n)]
        dispatch_ids += [d.dispatch_id for d in cands]
        record["candidate_dispatch_ids"] = [d.dispatch_id for d in cands]
        with ThreadPoolExecutor(n) as pool:
            results = list(pool.map(lambda d: run_dispatch(rt, d), cands))
        record["t_candidates_end"] = time.time()
        if rt.stopped:
            raise AgentStopped("episode stopped")
        diffs: list[str | None] = [None] * n
        if copy:
            leaked = _changed(pristine, manifest(env, rt.workdir))
            _copy_files(env, f"{root}/base", rt.workdir, leaked)
            record.update(isolation_violated=bool(leaked), n_leaked_files=len(leaked))
            for k in range(n):
                _map_back(env, f"{root}/cand{k}", rt.workdir)
            diffs = [_sh(env, f"cd -- {q(root)} && diff -ruN -x .git base cand{k} | head -c {DIFF_CAP_CHARS}")
                     or "(no file changes)" for k in range(n)]
        reports = [r.report for r in results]
        revised, best = list(reports), None
        for rnd in range(1, a.debate_rounds + 2):
            debate_ids: list[str] = []
            if rnd > 1:
                revised = _debate(rt, cands, instruction, revised, rnd, debate_ids)
                record["debate_rounds"] = rnd - 1
            judge_ids: list[str] = []
            j = _judge(rt, aid, group_id, instruction, results, revised, diffs, rnd, judge_ids)
            record["rounds"].append({"round": rnd, "judge_call_ids": judge_ids, "debate_call_ids": debate_ids, **j})
            # A failed or unparsed later round never replaces an earlier parsed judgement.
            if j["winner"] is not None and (best is None or j["parse_ok"] or not best["parse_ok"]):
                best = j
            if j["agreement"] and (j["confidence"] or 0) >= a.confidence_threshold:
                break
        w = best["winner"] if best else 0
        record.update(final_winner=w, final_confidence=best["confidence"] if best else None)
        applied: list[str] = []
        if copy:
            applied = _changed(pristine, manifest(env, f"{root}/cand{w}"))
            _copy_files(env, f"{root}/cand{w}", rt.workdir, applied)
            record.update(applied=True, n_applied_files=len(applied))
    finally:
        if copy and not rt.stopped:
            env.exec(f"rm -rf -- {q(root)}", 300)
        record["t_end"] = time.time()
        rt.rec.write("arbitrations", record)
    report = reports[w]
    if record["debate_rounds"] and revised[w].strip() != reports[w].strip():
        report = report_with_revision(reports[w], revised[w])
    if copy:
        report += "\n[applied files: " + (", ".join(applied[:50]) or "none, the winner changed no files") + "]"
    conf = "unknown" if record["final_confidence"] is None else f"{record['final_confidence']:.2f}"
    agreed = {True: "yes", False: "no", None: "unknown"}[best["agreement"] if best else None]
    return f"arbiter: candidate {w + 1} of {n}, confidence {conf}, candidates agreed: {agreed}", report


def _judge(rt: Runtime, aid: str, group_id: str, instruction: str, results: list[DispatchResult], reports: list[str],
           diffs: list[str | None], rnd: int, call_ids: list[str]) -> dict[str, Any]:
    """One judgement; a reply cut off at the output limit without one gets one retry
    (2 of 5 judge calls hit the old 8192-token limit, all reasoning and no verdict)."""
    who = Who("arbiter", f"arbiter-{aid}", group_id=group_id, arbitration_id=aid)
    cap = rt.config.subagent.max_output_chars
    prompt = judge_prompt(instruction, [{"status": r.status, "steps": r.steps_used, "report": truncate(rep, cap),
                                         "diff": d} for r, rep, d in zip(results, reports, diffs)], rnd)
    best = dict(NO_JUDGEMENT)
    for attempt in range(2):
        user = Message("user", prompt if attempt == 0 else f"{prompt}\n\n{JUDGE_RETRY_NOTE}")
        rt.rec.transcript(who, user)
        try:
            r, call_id = rt.rec.llm(who, rt.orch_client, ARBITER, [user], [JUDGE_SPEC], rnd)
        except ContextLengthError:
            break
        call_ids.append(call_id)
        rt.rec.transcript(who, Message("assistant", r.text, r.tool_calls, reasoning=r.reasoning or None), call_id)
        j = parse_judgement(r, len(results))
        if j["parse_ok"] or (j["winner"] is not None and best["winner"] is None):
            best = j
        if best["parse_ok"] or r.finish_reason != "length":
            break
    return best


def _debate(rt: Runtime, cands: list[Dispatch], instruction: str, reports: list[str], rnd: int,
            call_ids: list[str]) -> list[str]:
    """Every candidate revises its report after reading the others' (tool-less calls, concurrently)."""
    n, cap = len(cands), rt.config.subagent.max_output_chars

    def one(k: int) -> tuple[str, str | None]:
        who = replace(cands[k].who, role="debate")
        user = Message("user", debate_prompt(instruction, truncate(reports[k], cap),
                                             [(j, truncate(reports[j], cap)) for j in range(n) if j != k]))
        rt.rec.transcript(who, user)
        try:
            r, call_id = rt.rec.llm(who, rt.sub_client, DEBATE.format(k=k, n=n), [user], [], rnd)
        except ContextLengthError:
            return reports[k], None
        rt.rec.transcript(who, Message("assistant", r.text, reasoning=r.reasoning or None), call_id)
        return r.text.strip() or reports[k], call_id

    with ThreadPoolExecutor(n) as pool:
        outs = list(pool.map(one, range(n)))
    call_ids += [c for _, c in outs if c]
    return [r for r, _ in outs]
