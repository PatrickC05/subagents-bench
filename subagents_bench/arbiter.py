"""The ``arbiter`` strategy: N candidates per instruction, an LLM judge, and
confidence-aware debate (arXiv 2602.12276).

For one instruction (:func:`run_arbitration`):

1. **Isolation.** Hard scope: if ``du`` of the workdir is at most
   ``arbiter.max_copy_mb``, each candidate gets a private ``cp -a`` copy under
   ``<sb_root>/arb/<arbitration_id>/cand<k>`` and starts its shell there
   (``copy``); otherwise all share the workdir (``shared_fallback``). Easy and
   medium scopes always share (``shared``).
2. **Candidates** run concurrently; each is a normal :class:`Subagent` with
   ``candidate_index`` set.
3. **Judge.** One arbiter-model call (role ``arbiter``, agent
   ``arbiter-<arbitration_id>``, step = round) sees the instruction, every
   report (each capped at ``subagent.max_output_chars``, head and tail kept)
   and, with copy isolation, a diff summary per candidate (at most 4000
   chars). It must call
   ``submit_judgement(winner, confidence, agreement, rationale)``; without a
   tool call the JSON is parsed from the text; if that fails too the result
   is winner 0, confidence None, ``parse_ok=False``.
4. **Debate.** Unless the judge reported agreement with confidence at least
   ``confidence_threshold``, and while debate rounds remain, every
   candidate's model makes one tool-less call (role ``debate``, the
   candidate's agent id) with its own and the others' reports and returns a
   revised report; the calls run concurrently; then the judge decides again.
5. **Apply.** With copy isolation the winner's copy is applied to the real
   workdir (:func:`subagents_bench.workspace.apply_copy`, restricted to what
   the winner changed since its copy was made), then the copies are removed.
6. An :class:`~subagents_bench.schema.ArbitrationRecord` is written, also when
   the arbitration is cancelled part-way.

Timing (what the offline latency model replays): ``isolation_sec`` is step 1;
``t_start``..``t_candidates_end`` step 2; ``t_candidates_end`` to the first
round's ``t_start`` the diff summaries; each round's judge (and debate) calls
are recorded as calls; the last judge call's end to ``t_end`` is the apply
plus copy cleanup. All of these are part of the strategy's cost, so all of
them lie on the episode's critical path.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from . import workspace
from .config import Scope
from .prompts import ARBITER_SYSTEM, DEBATE_SYSTEM, arbiter_user, debate_user
from .schema import ArbitrationRecord
from .subagent import DispatchResult, DispatchSpec, Runtime, Subagent, measure
from .telemetry import AgentLog
from .tools import truncate_output
from .types import AgentStopped, BillingError, LLMError, LLMResponse, Message, ToolSpec

log = logging.getLogger(__name__)

DIFF_CAP_CHARS = 4000


def judgement_spec(n_candidates: int) -> ToolSpec:
    return ToolSpec(
        name="submit_judgement",
        description="Submit your judgement: the best candidate, your confidence, whether the candidates agreed, and why.",
        parameters={
            "type": "object",
            "properties": {
                "winner": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": max(n_candidates - 1, 0),
                    "description": "0-based index of the best candidate.",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Probability (0..1) that the winner is correct and the best choice.",
                },
                "agreement": {
                    "type": "boolean",
                    "description": "True if the candidates reached substantially the same result.",
                },
                "rationale": {"type": "string", "description": "One to three sentences."},
            },
            "required": ["winner", "confidence", "agreement", "rationale"],
        },
    )


@dataclass
class Judgement:
    winner: int
    confidence: float | None
    agreement: bool | None
    rationale: str
    parse_ok: bool


def _as_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and re.fullmatch(r"\s*-?\d+\s*", v):
        return int(v)
    return None


def _as_conf(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, str):
        try:
            v = float(v.strip())
        except ValueError:
            return None
    if isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0:
        return float(v)
    return None


def _as_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "yes"):
        return True
    if isinstance(v, str) and v.strip().lower() in ("false", "no"):
        return False
    return None


def judgement_from_obj(obj: Any, n: int) -> Judgement | None:
    """Judgement from a parsed object; None if it has no usable winner."""
    if not isinstance(obj, dict):
        return None
    w = _as_int(obj.get("winner"))
    if w is None or not 0 <= w < n:
        return None
    conf = _as_conf(obj.get("confidence"))
    agree = _as_bool(obj.get("agreement"))
    rationale = obj.get("rationale")
    return Judgement(
        winner=w,
        confidence=conf,
        agreement=agree,
        rationale=rationale if isinstance(rationale, str) else ("" if rationale is None else json.dumps(rationale)),
        parse_ok=conf is not None and agree is not None,
    )


def _json_objects(text: str) -> list[Any]:
    """Every JSON object that can be decoded from ``text`` (fenced or inline)."""
    out: list[Any] = []
    dec = json.JSONDecoder()
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.S):
        try:
            out.append(json.loads(m.group(1)))
        except ValueError:
            pass
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[i:])
                out.append(obj)
            except ValueError:
                continue
    return out


def parse_judgement(resp: LLMResponse, n: int) -> Judgement:
    """From the submit_judgement tool call, else JSON in the text, else the
    fallback (winner 0, confidence None, parse_ok False)."""
    for tc in resp.tool_calls:
        if tc.name == "submit_judgement":
            j = judgement_from_obj(tc.arguments, n)
            if j is not None:
                return j
    for obj in _json_objects(resp.text or ""):
        j = judgement_from_obj(obj, n)
        if j is not None:
            return j
    return Judgement(winner=0, confidence=None, agreement=None, rationale="", parse_ok=False)


@dataclass
class ArbitrationOutcome:
    arbitration_id: str
    instruction_index: int
    instruction: str
    isolation: str
    candidates: list[DispatchResult]
    winner: int
    confidence: float | None
    agreement: bool | None
    report: str  # the winner's final (possibly revised) report
    debate_rounds_used: int
    applied: bool
    applied_files: list[str]
    apply_error: str | None
    parse_ok: bool
    record: dict[str, Any] = field(default_factory=dict)


def _run_concurrently(fns: list[Callable[[], Any]]) -> list[tuple[Any, BaseException | None]]:
    """Run every fn in its own thread; (result, exception) per fn, in order."""
    if not fns:
        return []
    out: list[tuple[Any, BaseException | None]] = []
    with ThreadPoolExecutor(max_workers=len(fns)) as pool:
        futs = [pool.submit(fn) for fn in fns]
        for f in futs:
            try:
                out.append((f.result(), None))
            except BaseException as e:  # noqa: BLE001 - surfaced by the caller
                out.append((None, e))
    return out


def raise_first(errors: list[BaseException | None]) -> None:
    """BillingError wins over AgentStopped, which wins over anything else."""
    errs = [e for e in errors if e is not None]
    for kind in (BillingError, AgentStopped):
        for e in errs:
            if isinstance(e, kind):
                raise e
    if errs:
        raise errs[0]


def run_arbitration(
    rt: Runtime,
    *,
    menv: Any,
    group_id: str,
    instruction: str,
    instruction_index: int,
    orchestrator_step: int,
    register_dispatch: Callable[[str], None],
    t_queued: float,
) -> ArbitrationOutcome:
    """Run one arbitrated instruction (see the module docstring).

    ``menv`` is the (metered) environment for the runtime's own helper
    commands; ``register_dispatch(dispatch_id)`` is called as soon as each
    candidate exists, so the spawn record lists it even if this raises.
    Raises AgentStopped (after recording) when cancelled, BillingError on a
    billing failure.
    """
    cfg = rt.config
    acfg = cfg.subagent.arbiter
    n = acfg.n_candidates
    scope = cfg.subagent.scope
    rec = rt.recorder
    aid = rec.new_id("arbitration")
    arb_dir = f"{workspace.sb_root(rt.env)}/arb/{aid}"

    # 1. isolation
    t_setup = time.time()
    isolation = "shared"
    copies: list[str | None] = [None] * n
    base_manifest: dict[str, str] | None = None
    if scope is Scope.HARD:
        isolation = "shared_fallback"
        size = measure(workspace.dir_size_mb, menv, rt.workdir)
        if size is not None and size <= acfg.max_copy_mb:
            paths = [f"{arb_dir}/cand{k}" for k in range(n)]
            if all(measure(workspace.copy_tree, menv, rt.workdir, p) for p in paths):
                isolation = "copy"
                copies = list(paths)
                base_manifest = measure(workspace.manifest, menv, paths[0], include_mode=True)
            else:
                measure(menv.exec, f"rm -rf -- {shlex.quote(arb_dir)}", 300)

    # 2. candidates
    candidates: list[Subagent] = []
    for k in range(n):
        sa = Subagent(
            rt,
            DispatchSpec(
                instruction=instruction,
                instruction_index=instruction_index,
                group_id=group_id,
                orchestrator_step=orchestrator_step,
                strategy="arbiter",
                t_queued=t_queued,
                initial_cwd=copies[k] or rt.workdir,
                isolation=isolation,
                private_copy=copies[k],
                arbitration_id=aid,
                candidate_index=k,
            ),
        )
        register_dispatch(sa.dispatch_id)
        candidates.append(sa)

    t_start = time.time()
    t_candidates_end = t_start
    rounds: list[dict[str, Any]] = []
    results: list[DispatchResult | None] = [None] * n
    judgement = Judgement(winner=0, confidence=None, agreement=None, rationale="", parse_ok=False)
    reports: list[str] = []
    applied, applied_files, apply_error = False, [], None
    debate_rounds_used = 0
    try:
        outs = _run_concurrently([sa.run for sa in candidates])
        t_candidates_end = time.time()
        for k, (res, _) in enumerate(outs):
            results[k] = res
        raise_first([e for _, e in outs])
        if rt.stopped():
            raise AgentStopped("episode stopped")
        reports = [r.report for r in results]  # type: ignore[union-attr]

        diffs: list[str | None] = [None] * n
        if isolation == "copy":
            for k in range(n):
                d = measure(workspace.diff_summary, menv, rt.workdir, copies[k], DIFF_CAP_CHARS)
                diffs[k] = d if d is not None else "(diff summary unavailable)"

        judge_log = AgentLog(
            rec, f"arbiter-{aid}", "arbiter", group_id=group_id, arbitration_id=aid, stop_event=rt.stop_event
        )
        judge_log.transcript(Message(role="system", content=ARBITER_SYSTEM))
        spec = judgement_spec(n)
        for rnd in range(1, acfg.debate_rounds + 2):
            r_start = time.time()
            debate_ids: list[str] = []
            if rnd > 1:
                try:
                    reports = _debate_round(rt, candidates, instruction, reports, rnd, group_id, aid, debate_ids)
                except BaseException:
                    rounds.append(_round(rnd, None, debate_ids, judgement, r_start, ok=False))
                    raise
                debate_rounds_used += 1
            payload = [
                {
                    "index": k,
                    "status": results[k].status,  # type: ignore[union-attr]
                    "steps": results[k].steps_used,  # type: ignore[union-attr]
                    "report": _cap(rt, reports[k]),
                    "diff": diffs[k],
                }
                for k in range(n)
            ]
            user = Message(role="user", content=arbiter_user(instruction, payload, rnd))
            judge_log.transcript(user)
            judge_call_id: str | None = None
            try:
                resp, judge_call_id = judge_log.llm(rt.arbiter_client, ARBITER_SYSTEM, [user], [spec], step=rnd)
            except (AgentStopped, BillingError):
                rounds.append(_round(rnd, judge_log.last_call_id, debate_ids, judgement, r_start, ok=False))
                raise
            except LLMError as e:
                log.warning("arbiter %s judge call failed: %s", aid, e)
                judgement = Judgement(winner=0, confidence=None, agreement=None, rationale=f"judge call failed: {e}"[:500], parse_ok=False)
                rounds.append(_round(rnd, judge_log.last_call_id, debate_ids, judgement, r_start))
                break
            judge_log.transcript(
                Message(role="assistant", content=resp.text, tool_calls=resp.tool_calls, reasoning=resp.reasoning or None),
                judge_call_id,
            )
            judgement = parse_judgement(resp, n)
            rounds.append(_round(rnd, judge_call_id, debate_ids, judgement, r_start))
            confident = (
                judgement.agreement is True
                and judgement.confidence is not None
                and judgement.confidence >= acfg.confidence_threshold
            )
            if confident:
                break

        # 5. apply
        if isolation == "copy":
            winner_copy = copies[judgement.winner]
            res = measure(workspace.apply_copy, menv, winner_copy, rt.workdir, 200, base=base_manifest)
            if res is None:
                apply_error = "apply failed (stopped or crashed)"
            else:
                applied_files, apply_error = res
                applied = apply_error is None
    finally:
        if isolation == "copy":
            measure(menv.exec, f"rm -rf -- {shlex.quote(arb_dir)}", 300)
        t_end = time.time()
        record = rec.record_arbitration(
            ArbitrationRecord(
                episode_id=rec.episode_id,
                arbitration_id=aid,
                group_id=group_id,
                instruction_index=instruction_index,
                instruction=instruction,
                n_candidates=n,
                candidate_dispatch_ids=[sa.dispatch_id for sa in candidates],
                isolation=isolation,
                rounds=rounds,
                debate_rounds_used=debate_rounds_used,
                final_winner=judgement.winner,
                final_confidence=judgement.confidence,
                agreement=judgement.agreement,
                applied=applied,
                applied_files=list(applied_files)[:200],
                apply_error=apply_error,
                isolation_sec=t_start - t_setup,
                t_start=t_start,
                t_candidates_end=t_candidates_end,
                t_end=t_end,
                duration_sec=t_end - t_start,
            )
        )
    return ArbitrationOutcome(
        arbitration_id=aid,
        instruction_index=instruction_index,
        instruction=instruction,
        isolation=isolation,
        candidates=[r for r in results if r is not None],
        winner=judgement.winner,
        confidence=judgement.confidence,
        agreement=judgement.agreement,
        report=reports[judgement.winner] if reports else "",
        debate_rounds_used=debate_rounds_used,
        applied=applied,
        applied_files=list(applied_files),
        apply_error=apply_error,
        parse_ok=judgement.parse_ok,
        record=record or {},
    )


def _cap(rt: Runtime, report: str) -> str:
    """A report as shown to the judge/debaters: at most subagent.max_output_chars."""
    return truncate_output(report or "", rt.config.subagent.max_output_chars)[0]


def _round(rnd: int, judge_call_id: str | None, debate_ids: list[str], j: Judgement, t0: float, ok: bool = True) -> dict[str, Any]:
    return {
        "round": rnd,
        "judge_call_id": judge_call_id,
        "debate_call_ids": list(debate_ids),
        "winner": j.winner,
        "confidence": j.confidence,
        "agreement": j.agreement,
        "rationale": j.rationale,
        "parse_ok": j.parse_ok and ok,
        "t_start": t0,
        "t_end": time.time(),
    }


def _debate_round(
    rt: Runtime,
    candidates: list[Subagent],
    instruction: str,
    reports: list[str],
    rnd: int,
    group_id: str,
    aid: str,
    ids_out: list[str],
) -> list[str]:
    """One concurrent debate round -> revised reports; the debate call ids
    (failed ones included) are appended to ``ids_out``, also on error.

    A candidate whose debate call fails (non-billing provider error) keeps
    its previous report.
    """
    n = len(candidates)
    call_ids: list[str | None] = [None] * n
    lock = threading.Lock()

    def one(k: int) -> str:
        sa = candidates[k]
        dlog = AgentLog(
            rt.recorder, sa.agent_id, "debate", dispatch_id=sa.dispatch_id, group_id=group_id,
            arbitration_id=aid, stop_event=rt.stop_event,
        )
        system = DEBATE_SYSTEM.format(index=k, n=n)
        user = Message(
            role="user",
            content=debate_user(instruction, _cap(rt, reports[k]), [(j, _cap(rt, reports[j])) for j in range(n) if j != k]),
        )
        dlog.transcript(Message(role="system", content=system))
        dlog.transcript(user)
        try:
            resp, cid = dlog.llm(sa.client, system, [user], [], step=rnd)
        except LLMError as e:
            with lock:
                call_ids[k] = dlog.last_call_id
            if isinstance(e, BillingError):
                raise
            return reports[k]
        except AgentStopped:
            with lock:
                call_ids[k] = dlog.last_call_id
            raise
        with lock:
            call_ids[k] = cid
        dlog.transcript(Message(role="assistant", content=resp.text, reasoning=resp.reasoning or None), cid)
        return resp.text.strip() or reports[k]

    outs = _run_concurrently([lambda k=k: one(k) for k in range(n)])
    ids_out.extend(c for c in call_ids if c is not None)
    raise_first([e for _, e in outs])
    return [r for r, _ in outs]
