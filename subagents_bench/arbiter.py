"""The ``arbiter`` strategy: N candidates per instruction, an LLM judge, and
confidence-aware debate (arXiv 2602.12276).

For one instruction (:func:`run_arbitration`):

1. **Isolation.** Hard scope: if ``du`` of the workdir is at most
   ``arbiter.max_copy_mb``, a manifest of the pristine workdir is taken, a
   ``base`` copy and one private copy per candidate are made with ``cp -a``
   under ``<sb_root>/arb/<arbitration_id>/``, and each candidate starts its
   shell in its copy (``copy``); otherwise all share the workdir
   (``shared_fallback``). Easy and medium scopes always share (``shared``).
2. **Candidates** run concurrently; each is a normal :class:`Subagent` with
   ``candidate_index`` set. With copy isolation its tool arguments go through
   a :class:`~subagents_bench.tools.PathRemap`: a path (file tools) or a
   whole-path occurrence in a bash command naming the real workdir is
   redirected to its copy (``n_path_remaps``).
3. **Contain** (copy isolation). Once the candidates are done, files of the
   real workdir that changed meanwhile (``leaked_files``; a candidate wrote
   it by a route the remap missed) are restored from the base copy, so the
   real workdir is pristine again before anything is applied
   (``isolation_violated``). Files a concurrent arbitration of the same
   episode applied meanwhile are not leaks (a per-episode journal of applied
   files, under a lock shared by setup, containment and apply). Then, inside
   each copy, mentions of the copy's own path (in text files and symlink
   targets, e.g. written by a remapped command) are mapped back to the
   workdir path. Writes OUTSIDE the workdir (``/etc``, packages, services)
   are not isolated: every candidate makes them live in the shared
   container.
4. **Judge.** One arbiter-model call (role ``arbiter``, agent
   ``arbiter-<arbitration_id>``, step = round) sees the instruction, every
   report (each capped at ``subagent.max_output_chars``, head and tail kept)
   and, with copy isolation, a diff summary per candidate against the base
   copy (at most 4000 chars). It must call
   ``submit_judgement(winner, confidence, agreement, rationale)``; without a
   tool call the JSON is parsed from the text. A reply cut off at the output
   limit (``finish_reason`` ``length``) without a complete judgement gets one
   retry: the same input plus :data:`~subagents_bench.prompts.JUDGE_RETRY_NOTE`
   (the round's ``judge_retry_call_id``); the better-ranked of the two
   judgements counts. If nothing parses, the round's result is winner 0,
   confidence None, ``parse_ok=False``.
5. **Debate.** Unless the judge reported agreement with confidence at least
   ``confidence_threshold``, and while debate rounds remain, every
   candidate's model makes one tool-less call (role ``debate``, the
   candidate's agent id) with its own and the others' reports and returns a
   revised report; the calls run concurrently; then the judge decides again.
   The judgement used is the latest complete one (else the latest with a
   parseable winner): a later round whose judge call fails or does not parse
   never replaces it (``final_from_round``).
6. **Apply.** With copy isolation the winner's copy is applied to the real
   workdir (:func:`subagents_bench.workspace.apply_copy`, restricted to what
   the winner changed relative to the pristine manifest), then the copies
   are removed. The orchestrator then gets the winner's ORIGINAL report (what
   its files reflect) plus, after a debate, its revision under a note that
   it was written without tools; with shared isolation it gets the final
   (possibly revised) report.
7. An :class:`~subagents_bench.schema.ArbitrationRecord` is written, also when
   the arbitration is cancelled part-way.

Timing (what the offline latency model replays): ``isolation_sec`` is step 1;
``t_start``..``t_candidates_end`` step 2; ``t_candidates_end`` to the first
round's ``t_start`` the containment, the copy-path mapping and the diff
summaries; each round's judge (and debate) calls are recorded as calls; the
last judge call's end to ``t_end`` is the apply plus copy cleanup. All of
these are part of the strategy's cost, so all of them lie on the episode's
critical path.
"""

from __future__ import annotations

import json
import logging
import posixpath
import re
import shlex
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from . import workspace
from .config import Scope
from .prompts import ARBITER_SYSTEM, JUDGE_RETRY_NOTE, arbiter_user, debate_system, debate_user, report_with_revision
from .schema import ArbitrationRecord
from .subagent import DispatchResult, DispatchSpec, Runtime, Subagent, measure
from .telemetry import AgentLog
from .tools import PathRemap, copy_context, truncate_output
from .types import AgentStopped, BillingError, LLMError, LLMResponse, Message, ToolSpec

log = logging.getLogger(__name__)

DIFF_CAP_CHARS = 4000
# Max entries of ArbitrationRecord.applied_files / leaked_files.
FILES_CAP = 200


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
    # False when ``winner`` is the fallback 0 (nothing parseable, or the call failed).
    winner_parsed: bool = True

    @property
    def rank(self) -> int:
        """2 complete, 1 winner only, 0 fallback. A round's judgement replaces
        the arbitration's current one only at an equal or higher rank > 0."""
        return 2 if self.parse_ok else 1 if self.winner_parsed else 0


def no_judgement(rationale: str = "") -> Judgement:
    """The fallback: winner 0, nothing parsed."""
    return Judgement(winner=0, confidence=None, agreement=None, rationale=rationale, parse_ok=False, winner_parsed=False)


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
    return no_judgement()


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
    # The winner's report for the orchestrator: with copy isolation its
    # original report (plus, after a debate, its revision under a note); with
    # shared isolation its final (possibly revised) report.
    report: str
    debate_rounds_used: int
    applied: bool
    applied_files: list[str]
    apply_error: str | None
    parse_ok: bool
    record: dict[str, Any] = field(default_factory=dict)
    # Copy isolation: real-workdir files candidates changed (then restored).
    isolation_violated: bool | None = None
    leaked_files: list[str] = field(default_factory=list)
    final_from_round: int | None = None


@dataclass
class _ApplyJournal:
    """Per-episode: the workdir files arbitrations applied (relpaths, in
    order) and the lock that serializes isolation setup, containment and
    apply, so one arbitration's leak check never mistakes (and undoes) a
    concurrent arbitration's apply."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    applied: list[str] = field(default_factory=list)


_journals: "weakref.WeakKeyDictionary[Any, _ApplyJournal]" = weakref.WeakKeyDictionary()
_journals_lock = threading.Lock()


def _journal(rt: Runtime) -> _ApplyJournal:
    with _journals_lock:
        j = _journals.get(rt.recorder)
        if j is None:
            j = _journals[rt.recorder] = _ApplyJournal()
        return j


def _contain(
    menv: Any, workdir: str, base_dir: str, pristine: dict[str, str] | None, journal: _ApplyJournal, mark: int
) -> tuple[bool | None, list[str], str | None]:
    """Restore real-workdir files changed since ``pristine`` (other than those
    applied by concurrent arbitrations since ``mark``) from the base copy.
    Returns (isolation_violated, leaked relpaths, restore error); None when
    the check cannot run."""
    if pristine is None:
        return None, [], None
    with journal.lock:
        post = measure(workspace.manifest, menv, workdir, include_mode=True)
        if post is None:
            return None, [], None
        others = set(journal.applied[mark:])
        leaked = [p for p in workspace.changed_paths(pristine, post) if p not in others]
        if not leaked:
            return False, [], None
        log.warning("arbitration: %d file(s) of %s changed while candidates ran; restoring", len(leaked), workdir)
        res = measure(workspace.apply_copy, menv, base_dir, workdir, None, only=leaked, relative=True)
    return True, leaked, "restore failed (stopped or crashed)" if res is None else res[1]


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
    base_dir = f"{arb_dir}/base"
    journal = _journal(rt)

    # 1. isolation
    t_setup = time.time()
    isolation = "shared"
    copies: list[str | None] = [None] * n
    pristine: dict[str, str] | None = None  # manifest of the workdir before the candidates
    mark = 0  # journal entries before ours: applied by arbitrations that finished earlier
    if scope is Scope.HARD:
        isolation = "shared_fallback"
        size = measure(workspace.dir_size_mb, menv, rt.workdir)
        if size is not None and size <= acfg.max_copy_mb:
            paths = [f"{arb_dir}/cand{k}" for k in range(n)]
            # Under the lock no concurrent apply lands between the manifest and the copies.
            with journal.lock:
                mark = len(journal.applied)
                pristine = measure(workspace.manifest, menv, rt.workdir, include_mode=True)
                ok = all(measure(workspace.copy_tree, menv, rt.workdir, p) for p in [base_dir, *paths])
            if ok:
                isolation = "copy"
                copies = list(paths)
            else:
                pristine = None
                measure(menv.exec, f"rm -rf -- {shlex.quote(arb_dir)}", 300)

    # 2. candidates
    candidates: list[Subagent] = []
    remaps: list[PathRemap] = []
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
        if copies[k] is not None:
            remap = PathRemap(rt.workdir, copies[k])
            sa.ctx = copy_context(sa.ctx, remap)
            remaps.append(remap)
        register_dispatch(sa.dispatch_id)
        candidates.append(sa)

    t_start = time.time()
    t_candidates_end = t_start
    rounds: list[dict[str, Any]] = []
    results: list[DispatchResult | None] = [None] * n
    judgement = no_judgement()
    final_from_round: int | None = None
    orig_reports: list[str] = []
    reports: list[str] = []
    applied, applied_files, apply_error = False, [], None
    violated: bool | None = None
    leaked: list[str] = []
    restore_error: str | None = None
    debate_rounds_used = 0
    try:
        outs = _run_concurrently([sa.run for sa in candidates])
        t_candidates_end = time.time()
        for k, (res, _) in enumerate(outs):
            results[k] = res
        raise_first([e for _, e in outs])
        if rt.stopped():
            raise AgentStopped("episode stopped")
        orig_reports = [r.report for r in results]  # type: ignore[union-attr]
        reports = list(orig_reports)

        diffs: list[str | None] = [None] * n
        if isolation == "copy":
            # 3. contain: the real workdir back to pristine, copies in workdir terms.
            violated, leaked, restore_error = _contain(menv, rt.workdir, base_dir, pristine, journal, mark)
            for k in range(n):
                measure(workspace.replace_in_tree, menv, copies[k], copies[k], rt.workdir)
                d = measure(workspace.diff_summary, menv, base_dir, copies[k], DIFF_CAP_CHARS)
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
                    reports = _debate_round(
                        rt, candidates, instruction, reports, rnd, group_id, aid, debate_ids,
                        private_copy=isolation == "copy",
                    )
                except BaseException:
                    rounds.append(_round(rnd, None, debate_ids, no_judgement(), r_start, ok=False))
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
                rounds.append(_round(rnd, judge_log.last_call_id, debate_ids, no_judgement(), r_start, ok=False))
                raise
            except LLMError as e:
                # The earlier judgement (if any) stands; judging ends here.
                log.warning("arbiter %s judge call failed: %s", aid, e)
                failed = no_judgement(f"judge call failed: {e}"[:500])
                rounds.append(_round(rnd, judge_log.last_call_id, debate_ids, failed, r_start))
                break
            judge_log.transcript(
                Message(role="assistant", content=resp.text, tool_calls=resp.tool_calls, reasoning=resp.reasoning or None),
                judge_call_id,
            )
            j = parse_judgement(resp, n)
            retry_call_id: str | None = None
            if not j.parse_ok and resp.finish_reason == "length":  # cut off, with or without a partial call
                # Cut off (typically while reasoning) before a complete
                # judgement: one fresh retry, told to be brief.
                retry_user = Message(role="user", content=f"{user.content}\n\n{JUDGE_RETRY_NOTE}")
                judge_log.transcript(retry_user)
                try:
                    resp2, retry_call_id = judge_log.llm(
                        rt.arbiter_client, ARBITER_SYSTEM, [retry_user], [spec], step=rnd
                    )
                except (AgentStopped, BillingError):
                    rounds.append(
                        _round(rnd, judge_call_id, debate_ids, no_judgement(), r_start, ok=False,
                               retry_call_id=judge_log.last_call_id)
                    )
                    raise
                except LLMError as e:
                    log.warning("arbiter %s judge retry failed: %s", aid, e)
                    retry_call_id = judge_log.last_call_id
                else:
                    judge_log.transcript(
                        Message(role="assistant", content=resp2.text, tool_calls=resp2.tool_calls, reasoning=resp2.reasoning or None),
                        retry_call_id,
                    )
                    j2 = parse_judgement(resp2, n)
                    if j2.rank >= j.rank:
                        j = j2
            rounds.append(_round(rnd, judge_call_id, debate_ids, j, r_start, retry_call_id=retry_call_id))
            # An unparseable later round never replaces a parsed earlier one.
            if j.rank > 0 and j.rank >= judgement.rank:
                judgement, final_from_round = j, rnd
            confident = j.agreement is True and j.confidence is not None and j.confidence >= acfg.confidence_threshold
            if confident:
                break

        # 6. apply
        if isolation == "copy":
            winner_copy = copies[judgement.winner]
            with journal.lock:
                res = measure(workspace.apply_copy, menv, winner_copy, rt.workdir, None, base=pristine, relative=True)
                if res is not None:
                    journal.applied.extend(res[0])
            if res is None:
                apply_error = "apply failed (stopped or crashed)"
            else:
                applied_files = [posixpath.join(rt.workdir, p) for p in res[0]]
                apply_error = res[1]
                applied = apply_error is None
    finally:
        if isolation == "copy":
            measure(menv.exec, f"rm -rf -- {shlex.quote(arb_dir)}", 300)
        if restore_error:
            apply_error = "; ".join(e for e in (f"restoring leaked files: {restore_error}", apply_error) if e)
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
                applied_files=list(applied_files)[:FILES_CAP],
                apply_error=apply_error,
                isolation_sec=t_start - t_setup,
                t_start=t_start,
                t_candidates_end=t_candidates_end,
                t_end=t_end,
                duration_sec=t_end - t_start,
                isolation_violated=violated,
                leaked_files=[posixpath.join(rt.workdir, p) for p in leaked[:FILES_CAP]],
                n_path_remaps=sum(r.n for r in remaps),
                final_from_round=final_from_round,
            )
        )
    w = judgement.winner
    report = reports[w] if reports else ""
    if isolation == "copy" and orig_reports:
        # The applied files are what the candidate did before the debate.
        report = orig_reports[w]
        if debate_rounds_used and reports[w].strip() != orig_reports[w].strip():
            report = report_with_revision(orig_reports[w], reports[w])
    return ArbitrationOutcome(
        arbitration_id=aid,
        instruction_index=instruction_index,
        instruction=instruction,
        isolation=isolation,
        candidates=[r for r in results if r is not None],
        winner=w,
        confidence=judgement.confidence,
        agreement=judgement.agreement,
        report=report,
        debate_rounds_used=debate_rounds_used,
        applied=applied,
        applied_files=list(applied_files),
        apply_error=apply_error,
        parse_ok=judgement.parse_ok,
        record=record or {},
        isolation_violated=violated,
        leaked_files=[posixpath.join(rt.workdir, p) for p in leaked],
        final_from_round=final_from_round,
    )


def _cap(rt: Runtime, report: str) -> str:
    """A report as shown to the judge/debaters: at most subagent.max_output_chars."""
    return truncate_output(report or "", rt.config.subagent.max_output_chars)[0]


def _round(
    rnd: int,
    judge_call_id: str | None,
    debate_ids: list[str],
    j: Judgement,
    t0: float,
    ok: bool = True,
    *,
    retry_call_id: str | None = None,
) -> dict[str, Any]:
    return {
        "round": rnd,
        "judge_call_id": judge_call_id,
        "judge_retry_call_id": retry_call_id,
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
    *,
    private_copy: bool = False,
) -> list[str]:
    """One concurrent debate round -> revised reports; the debate call ids
    (failed ones included) are appended to ``ids_out``, also on error.

    A candidate whose debate call fails (non-billing provider error) keeps
    its previous report. ``private_copy``: the candidates worked in private
    copies (their prompt says the copies are frozen).
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
        system = debate_system(k, n, private_copy=private_copy)
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
