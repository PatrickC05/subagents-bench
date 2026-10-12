"""``check-run`` and ``completeness`` on synthetic runs, clean and with injected faults."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth import GEMMA, Arb, C, EpisodeBuilder, Judge, Spawn, T, make_sweep  # noqa: E402

from subagents_bench import analysis  # noqa: E402
from subagents_bench.analysis.validate import check_episode, check_run, completeness  # noqa: E402

CONDS = {
    "single_5step": {"topology": "single", "steps_budget": 5},
    "gemma31b_parallel_medium": {"topology": "subagents", "strategy": "parallel", "scope": "medium", "sub_model": GEMMA},
    "gemma31b_sequential_easy": {"topology": "subagents", "strategy": "sequential", "scope": "easy", "sub_model": GEMMA},
    "gemma31b_arbiter_hard": {"topology": "subagents", "strategy": "arbiter", "scope": "hard", "sub_model": GEMMA},
}


def _cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    analysis.register_cli(sub)
    args = parser.parse_args(argv)
    return args.func(args)


def _episode(tmp_path: Path, retries: bool = False) -> Path:
    """An arbiter episode with every record type, written to tmp_path/ep."""
    b = EpisodeBuilder(strategy="arbiter", t0=1000.0, gap=0.05)
    arb = Arb(candidates=[[C(1.0), T("edit_file", 0.5)], [C(2.0)], [C(1.2)]],
              rounds=[Judge(C(0.8), confidence=0.5, agreement=False), Judge(C(0.6), debates=[C(0.4)] * 3)],
              apply_sec=0.2)
    b.step(C(2.0, attempts=3 if retries else 1, backoff=1.0 if retries else 0.0), [Spawn(arbitrations=[arb])])
    b.step(C(1.0), [T("bash", 0.3, timed_out=True)])
    b.step(C(0.5), finish=True)
    b.finish()
    return b.write(tmp_path / "ep")


def _rewrite_jsonl(path: Path, fn) -> None:
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    out = fn(recs)
    path.write_text("".join(json.dumps(r) + "\n" for r in out))


def _messages(problems, level="error") -> list[str]:
    return [p.message for p in problems if p.level == level]


def test_clean_sweep_passes(tmp_path, capsys):
    make_sweep(tmp_path / "jobs", conditions=CONDS, edge_cases=False)
    report = check_run([tmp_path / "jobs"])
    assert report.n_episodes == 4 * 2 * 2
    assert report.errors == [] and report.ok
    assert _cli(["check-run", str(tmp_path / "jobs")]) == 0
    assert "check-run OK: 16 episodes" in capsys.readouterr().out


def test_clean_episode_with_retries_and_timeouts(tmp_path):
    ep = _episode(tmp_path, retries=True)
    assert _messages(check_episode(ep)) == []


def test_token_total_mismatch(tmp_path):
    ep = _episode(tmp_path)
    doc = json.loads((ep / "episode.json").read_text())
    doc["tokens"]["orchestrator"]["prompt"] += 1
    doc["cost_usd"]["total"] += 0.01
    doc["n_command_timeouts"] = 0
    (ep / "episode.json").write_text(json.dumps(doc))
    msgs = _messages(check_episode(ep))
    assert any(m.startswith("tokens[orchestrator][prompt]=3001") for m in msgs)
    assert any(m.startswith("cost_usd[total]=") for m in msgs)
    assert any(m.startswith("n_command_timeouts=0 but the records give 1") for m in msgs)
    assert _cli(["check-run", str(ep)]) == 1


def test_stream_repair_count_is_cross_checked(tmp_path):
    ep = _episode(tmp_path)
    doc = json.loads((ep / "episode.json").read_text())
    assert doc["n_stream_repairs"] == 0
    (ep / "episode.json").write_text(json.dumps({**doc, "n_stream_repairs": 3}))
    assert "n_stream_repairs=3 but the records give 0" in _messages(check_episode(ep))
    _rewrite_jsonl(ep / "calls.jsonl", lambda recs: [{**r, "stream_repairs": 3} if i == 0 else r for i, r in enumerate(recs)])
    assert not any(m.startswith("n_stream_repairs") for m in _messages(check_episode(ep)))
    # Records written before the field existed are not checked for it.
    doc = {k: v for k, v in doc.items() if k != "n_stream_repairs"} | {"schema_version": 2}
    (ep / "episode.json").write_text(json.dumps(doc))
    assert not any(m.startswith("n_stream_repairs") for m in _messages(check_episode(ep)))


def test_judge_retry_call_must_exist(tmp_path):
    ep = _episode(tmp_path)
    _rewrite_jsonl(ep / "arbitrations.jsonl", lambda recs: [
        {**recs[0], "rounds": [{**recs[0]["rounds"][0], "judge_retry_call_id": "gone"}, *recs[0]["rounds"][1:]]},
        *recs[1:],
    ])
    assert any("judge retry call 'gone' missing or not role arbiter" in m for m in _messages(check_episode(ep)))


def test_provider_error_count_is_cross_checked(tmp_path):
    ep = _episode(tmp_path)
    doc = json.loads((ep / "episode.json").read_text())
    (ep / "episode.json").write_text(json.dumps({**doc, "n_provider_errors": 2}))
    assert "n_provider_errors=2 but the records give 0" in _messages(check_episode(ep))
    _rewrite_jsonl(ep / "calls.jsonl", lambda recs: [{**r, "error": "LLMError: sail: giving up after 9 attempts"}
                                                     if i < 2 else r for i, r in enumerate(recs)])
    assert not any(m.startswith("n_provider_errors") for m in _messages(check_episode(ep)))
    # A record written before the field existed is not checked for it.
    doc = {k: v for k, v in doc.items() if k != "n_provider_errors"} | {"schema_version": 2}
    (ep / "episode.json").write_text(json.dumps(doc))
    assert not any(m.startswith("n_provider_errors") for m in _messages(check_episode(ep)))


def test_missing_dispatch_record(tmp_path):
    ep = _episode(tmp_path)
    _rewrite_jsonl(ep / "dispatches.jsonl", lambda recs: recs[1:])
    msgs = _messages(check_episode(ep))
    assert any("dispatch_id 'd001' not in dispatches.jsonl" in m for m in msgs)
    assert any("dispatch_ids mismatch" in m for m in msgs)
    assert any("candidate dispatch 'd001'" in m for m in msgs)
    assert any(m.startswith("dispatches=3 but the records give 2") for m in msgs)


def test_agent_seq_gap_and_duplicate_seq(tmp_path):
    ep = _episode(tmp_path)

    def bump(recs):
        for r in recs:
            if r["agent_id"] == "orchestrator" and r["step"] == 3:
                r["agent_seq"] += 5
        recs[0]["seq"] = recs[1]["seq"]
        return recs

    _rewrite_jsonl(ep / "calls.jsonl", bump)
    msgs = _messages(check_episode(ep))
    assert any(m.startswith("agent orchestrator: agent_seq not contiguous") for m in msgs)
    assert any(m.startswith("global seq not unique") for m in msgs)


def test_non_monotonic_timestamps(tmp_path):
    ep = _episode(tmp_path)

    def shift(recs):
        for r in recs:
            if r["name"] == "bash":  # move the orchestrator's bash before its requesting call ended
                r["t_start"] -= 0.5
                r["t_end"] -= 0.5
        return recs

    _rewrite_jsonl(ep / "tools.jsonl", shift)
    msgs = _messages(check_episode(ep))
    assert any("agent_seq=3 starts 0.450s before the previous event ended" in m for m in msgs)  # 0.5 - 0.05 gap


def test_arbitration_ids_must_exist(tmp_path):
    ep = _episode(tmp_path)

    def corrupt(recs):
        recs[0]["rounds"][0]["judge_call_id"] = "nope"
        recs[0]["rounds"][1]["debate_call_ids"][0] = recs[0]["rounds"][1]["judge_call_id"]  # wrong role
        return recs

    _rewrite_jsonl(ep / "arbitrations.jsonl", corrupt)
    msgs = _messages(check_episode(ep))
    assert any("judge call 'nope' missing or not role arbiter" in m for m in msgs)
    assert any("missing or not role debate" in m for m in msgs)


def test_schema_violations_and_truncated_line(tmp_path):
    ep = _episode(tmp_path)

    def damage(recs):
        del recs[0]["ttft_sec"]
        recs[1]["prompt_tokens"] = "lots"
        recs[2]["surprise"] = 1
        return recs

    _rewrite_jsonl(ep / "calls.jsonl", damage)
    with open(ep / "tools.jsonl", "a") as f:
        f.write('{"episode_id": "ep0001", "tool_ev')  # a writer killed mid-line
    msgs = _messages(check_episode(ep))
    assert "missing field 'ttft_sec'" in msgs
    assert "field 'prompt_tokens': str not accepted by int" in msgs
    assert "unknown field 'surprise'" in msgs
    assert "tools.jsonl: invalid JSON on last line (truncated write?)" in msgs


def test_running_episode_is_an_error_but_refs_only_warn(tmp_path):
    b = EpisodeBuilder(strategy="parallel", t0=1000.0)
    b.step(C(2.0), [Spawn(members=[[C(1.0)], [C(1.5)]])])
    b.finish(status="running")
    ep = b.write(tmp_path / "ep")
    _rewrite_jsonl(ep / "dispatches.jsonl", lambda recs: recs[:1])  # second dispatch still running at the crash
    problems = check_episode(ep)
    assert _messages(problems) == ["episode never finalized (status='running'); totals not checked"]
    assert any("not in dispatches.jsonl" in m for m in _messages(problems, "warning"))


def test_cancelled_episode_refs_only_warn(tmp_path):
    # harbor cancelled the agent mid-group (agent timeout): the recorder finalized the episode
    # from what was written, but the dispatch still running never wrote its record.
    b = EpisodeBuilder(strategy="parallel", t0=1000.0)
    b.step(C(2.0), [Spawn(members=[[C(1.0)], [C(1.5)]])])
    b.finish(exit_reason="cancelled")
    ep = b.write(tmp_path / "ep")
    _rewrite_jsonl(ep / "dispatches.jsonl", lambda recs: recs[:1])
    doc = json.loads((ep / "episode.json").read_text())
    (ep / "episode.json").write_text(json.dumps({**doc, "dispatches": 1}))
    problems = check_episode(ep)
    assert _messages(problems) == []
    assert any("not in dispatches.jsonl" in m for m in _messages(problems, "warning"))
    # The same dangling references in an episode that ended normally are errors.
    (ep / "episode.json").write_text(json.dumps({**doc, "dispatches": 1, "exit_reason": "finished"}))
    assert any("not in dispatches.jsonl" in m for m in _messages(check_episode(ep)))


def test_jobs_dir_without_episodes_fails(tmp_path):
    make_sweep(tmp_path / "jobs", conditions={"gemma31b_parallel_easy": CONDS["gemma31b_parallel_medium"]},
               tasks={}, edge_cases=True)  # only the environment-failure trial
    report = check_run([tmp_path / "jobs"])
    assert report.n_episodes == 0 and report.n_trials_without_episode == 1 and not report.ok
    assert "EnvironmentStartTimeoutError" in report.warnings[0].message


def test_completeness_flags_all_none_field(tmp_path, capsys):
    make_sweep(tmp_path / "jobs", conditions=CONDS, edge_cases=False, rounds=(1,))
    for calls in (tmp_path / "jobs").rglob("calls.jsonl"):
        _rewrite_jsonl(calls, lambda recs: [{**r, "ttft_sec": None} for r in recs])
    rows = {(r["file"], r["role"], r["field"]): r for r in completeness([tmp_path / "jobs"])}
    for role in ("orchestrator", "subagent", "arbiter", "debate"):
        r = rows[("calls.jsonl", role, "ttft_sec")]
        assert r["never_populated"] and not r["often_empty"] and r["frac_none"] == 1.0
    ok = rows[("calls.jsonl", "orchestrator", "prompt_tokens")]
    assert not ok["never_populated"] and ok["frac_unpopulated"] == 0.0
    # Booleans count as populated when False; zero-valued index fields too.
    assert not rows[("tools.jsonl", "orchestrator", "timed_out")]["never_populated"]
    assert not rows[("arbitrations.jsonl", "arbiter", "final_winner")]["never_populated"]
    # Role-specific expectations: orchestrator calls never carry a dispatch_id.
    od = rows[("calls.jsonl", "orchestrator", "dispatch_id")]
    assert od["never_populated"] and od["often_empty"]
    assert rows[("calls.jsonl", "subagent", "dispatch_id")]["frac_none"] == 0.0
    # CLI: report + CSV; --strict fails on the unexpected never-populated field.
    assert _cli(["completeness", str(tmp_path / "jobs"), "--out", str(tmp_path / "c.csv")]) == 0
    out = capsys.readouterr().out
    assert "ttft_sec" in out and "NEVER POPULATED" in out
    assert (tmp_path / "c.csv").is_file()
    assert _cli(["completeness", str(tmp_path / "jobs"), "--strict"]) == 1


def test_completeness_skips_fields_newer_than_the_record(tmp_path):
    # Records written by older code (e.g. a sweep still running schema-2 code)
    # lack the fields added since; that is not a never-populated field.
    from dataclasses import fields

    from subagents_bench.schema import RECORD_TYPES

    make_sweep(tmp_path / "jobs", conditions=CONDS, edge_cases=False, rounds=(1,))
    newer = {fname: {f.name for f in fields(cls) if (f.metadata.get("since") or 0) >= 3}
             for fname, cls in RECORD_TYPES.items()}
    assert "source_sha256" in newer["episode.json"]
    for ep_path in (tmp_path / "jobs").rglob("episode.json"):
        ep = json.loads(ep_path.read_text())
        ep = {k: v for k, v in ep.items() if k not in newer["episode.json"]}
        ep_path.write_text(json.dumps({**ep, "schema_version": 2}))
        for fname, names in newer.items():
            if fname != "episode.json" and names and (ep_path.parent / fname).is_file():
                _rewrite_jsonl(ep_path.parent / fname, lambda recs: [{k: v for k, v in r.items() if k not in names}
                                                                    for r in recs])
    rows = {(r["file"], r["role"], r["field"]): r for r in completeness([tmp_path / "jobs"])}
    assert not [k for k in rows if k[2] in newer[k[0]]]
    assert not rows[("episode.json", "episode", "exit_reason")]["never_populated"]
    assert rows[("episode.json", "episode", "exit_reason")]["n_records"] == 8
    # A current-version record that lacks one is still flagged.
    for ep_path in (tmp_path / "jobs").rglob("episode.json"):
        ep = json.loads(ep_path.read_text())
        ep_path.write_text(json.dumps({**ep, "schema_version": 3}))
    rows = {(r["file"], r["role"], r["field"]): r for r in completeness([tmp_path / "jobs"])}
    r = rows[("episode.json", "episode", "source_sha256")]
    assert r["never_populated"] and not r["often_empty"] and r["n_records"] == 8


def _no_delegation_in(condition: str):
    """make_sweep episode builder: ``condition``'s orchestrator never spawns; the others delegate."""
    from synth import standard_episode

    def build(b: EpisodeBuilder, task_idx: int, rnd: int) -> EpisodeBuilder:
        if b.condition != condition:
            return standard_episode(b, task_idx, rnd)
        return b.step(C(2.0), [T("bash", 1.0)]).step(C(1.0), finish=True)

    return build


def test_treatment_coverage_gates_completeness_and_warns_in_check_run(tmp_path, capsys, monkeypatch):
    from subagents_bench.analysis import validate

    # Synthetic records leave a few fields unpopulated (no transcripts of tool turns, no measurement
    # execs); keep only the fields that are populated so --strict depends on the treatment check alone.
    real = validate.completeness
    monkeypatch.setattr(validate, "completeness", lambda paths: [
        r for r in real(paths) if not (r["never_populated"] and not r["often_empty"])])
    # Every subagents condition delegated: reported, and nothing fails.
    make_sweep(tmp_path / "ok", conditions=CONDS, edge_cases=False, rounds=(1,))
    assert _cli(["completeness", str(tmp_path / "ok"), "--strict"]) == 0
    out = capsys.readouterr().out
    assert "treatment gemma31b_parallel_medium: delegated in 2/2 episodes, 6 dispatches" in out
    assert "treatment gemma31b_arbiter_hard: delegated in 2/2 episodes, 6 dispatches, 2 arbitrations" in out
    assert "single_5step" not in out and "not exercised" not in out
    assert all(r["ok"] for r in check_run([tmp_path / "ok"]).coverage)

    # A subagents condition whose orchestrator never spawned: every field check still passes (no
    # spawn/dispatch record exists to flag), but the treatment check catches it.
    make_sweep(tmp_path / "jobs", conditions=CONDS, edge_cases=False, rounds=(1,),
               episode=_no_delegation_in("gemma31b_parallel_medium"))
    assert _cli(["completeness", str(tmp_path / "jobs")]) == 0
    assert _cli(["completeness", str(tmp_path / "jobs"), "--strict"]) == 1
    out = capsys.readouterr().out
    assert "treatment not exercised in 1 subagents condition(s): gemma31b_parallel_medium" in out
    assert "no episode delegated (spawn_calls = 0 in all 2)" in out
    report = check_run([tmp_path / "jobs"])
    assert report.ok  # check-run is about record consistency: a warning, not an error
    assert [str(p) for p in report.warnings if "treatment" in p.message] == [
        "[WARNING] condition gemma31b_parallel_medium: treatment not exercised: "
        "no episode delegated (spawn_calls = 0 in all 2)"]
    cov = {r["condition"]: r for r in report.coverage}
    assert (cov["gemma31b_parallel_medium"]["n_delegated"], cov["gemma31b_parallel_medium"]["delegated_frac"]) == (0, 0.0)
    assert _cli(["check-run", str(tmp_path / "jobs")]) == 0
    assert "** no episode delegated" in capsys.readouterr().out


def test_treatment_coverage_rows():
    from subagents_bench.analysis.validate import treatment_coverage_rows

    ep = lambda c, t, s, sp, d, a: {"condition": c, "topology": t, "strategy": s, "spawn_calls": sp,  # noqa: E731
                                    "dispatches": d, "arbitrations": a}
    rows = treatment_coverage_rows([
        ep("single_5step", "single", None, 0, 0, 0),
        ep("arb", "subagents", "arbiter", 1, 3, 0), ep("arb", "subagents", "arbiter", 0, 0, 0),
        ep("par", "subagents", "parallel", 1, 0, 0),
        ep("seq", "subagents", "sequential", 2, 4, 0), ep("seq", "subagents", "sequential", None, None, None),
    ])
    by = {r["condition"]: r for r in rows}
    assert set(by) == {"arb", "par", "seq"}  # single topology has no treatment to check
    assert by["arb"]["problem"] == "arbiter strategy but no arbitration ran" and by["arb"]["delegated_frac"] == 0.5
    assert by["par"]["problem"] == "spawn calls but no dispatch ran"
    assert by["seq"]["ok"] and by["seq"]["n_arbitrations"] is None and by["seq"]["n_delegated"] == 1
