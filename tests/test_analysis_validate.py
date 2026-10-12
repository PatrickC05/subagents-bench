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
