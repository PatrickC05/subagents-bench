"""The arbiter strategy: copy isolation, remap, leak restore, judge (retry, JSON fallback),
debate, and applying the winner."""

from __future__ import annotations

import os
import re

import pytest

from fakes import ORCH, SUB, FakeFactory, Reply, assert_consistent, make_config, run_local, tc
from subagents_bench import arbiter
from subagents_bench.arbiter import parse_judgement
from subagents_bench.llm import LLMResponse
from subagents_bench.prompts import JUDGE_RETRY_NOTE


def judgement(winner, confidence=0.9, agreement=True):
    return Reply("", [tc("submit_judgement", winner=winner, confidence=confidence, agreement=agreement, rationale="r")])


def arbitrate(tmp_path, cand, judge, debate=None, scope="hard", files=None):
    """One spawn with one instruction; ``cand(k, messages, work)`` scripts candidate k."""
    work = tmp_path / "work"
    work.mkdir()
    for name, text in (files or {"app.py": "v0\n", "old.txt": "old\n"}).items():
        (work / name).write_text(text)
    work = os.path.realpath(work)
    judges = list(judge)

    def orch(system, messages, tools):
        if system.startswith("You are an impartial arbiter"):
            return judges.pop(0)
        if len(messages) == 1:
            return Reply("", [tc("spawn_subagents", instructions=["fix app.py"])])
        return Reply("", [tc("finish")])

    def sub(system, messages, tools):
        m = re.match(r"You are candidate (\d)", system)
        if m:
            return debate(int(m.group(1)), messages)
        k = int(re.search(r"cand(\d)", system).group(1)) if "Isolation:" in system else None
        return cand(k, messages, work)

    f = FakeFactory({ORCH: orch, SUB: sub})
    cfg = make_config(subagent={"strategy": "arbiter", "scope": scope})
    exit_reason, rec = run_local(cfg, f, tmp_path / "work", tmp_path / "ep")
    r = assert_consistent(tmp_path / "ep")
    result = [m for m in f[ORCH].requests[-1]["messages"] if m.role == "tool"][0].tool_results[0].content
    return exit_reason, r, f, result, work, rec


def editing(k, messages, work):
    """Candidate k edits app.py by its absolute workdir path (remapped to its copy)."""
    if len(messages) == 1:
        cmd = f"echo v{k} > {work}/app.py" + (f" && rm {work}/old.txt && echo new > {work}/new.txt" if k == 1 else "")
        return Reply("", [tc("bash", command=cmd)])
    return Reply("", [tc("submit", report=f"candidate {k} wrote v{k}")])


def test_copy_isolation_judge_and_apply_the_winner(tmp_path):
    exit_reason, r, f, result, work, rec = arbitrate(tmp_path, editing, [judgement(1)])
    assert exit_reason == "finished"
    assert open(f"{work}/app.py").read() == "v1\n" and not os.path.exists(f"{work}/old.txt")
    assert open(f"{work}/new.txt").read() == "new\n"
    (a,) = r["arbitrations"]
    assert (a["isolation"], a["final_winner"], a["final_confidence"], a["debate_rounds"]) == ("copy", 1, 0.9, 0)
    assert (a["applied"], a["n_applied_files"], a["isolation_violated"], a["n_leaked_files"]) == (True, 3, False, 0)
    assert len(a["rounds"]) == 1 and len(a["rounds"][0]["judge_call_ids"]) == 1 and a["rounds"][0]["parse_ok"]
    (spawn,) = r["spawns"]
    assert spawn["arbitration_ids"] == [a["arbitration_id"]] and spawn["dispatch_ids"] == a["candidate_dispatch_ids"]
    assert sorted(d["candidate_index"] for d in r["dispatches"]) == [0, 1, 2] and spawn["granted"] == 1
    judge_prompt = [m for m in f[ORCH].requests if m["system"].startswith("You are an impartial")][0]["messages"][0].content
    assert "File changes in its private copy" in judge_prompt and "+v2" in judge_prompt
    assert "candidate 2 of 3, confidence 0.90, candidates agreed: yes" in result and "candidate 1 wrote v1" in result
    assert "applied files: app.py, new.txt, old.txt" in result and "dispatch budget left: 12" in result
    assert sorted(os.listdir(work)) == ["app.py", "new.txt"]
    arb_calls = [c for c in r["calls"] if c["role"] == "arbiter"]
    assert len(arb_calls) == 1 and arb_calls[0]["arbitration_id"] == a["arbitration_id"]


def test_leaks_into_the_real_workdir_are_restored_and_recorded(tmp_path):
    def leaking(k, messages, work):
        rel = work.lstrip("/")  # a relative path from / is not remapped
        if len(messages) == 1 and k == 0:
            return Reply("", [tc("bash", command=f"cd / && echo leaked > {rel}/app.py && echo x > {rel}/leak.txt")])
        return Reply("", [tc("submit", report="done")])

    _, r, _, _, work, _ = arbitrate(tmp_path, leaking, [judgement(2)])
    (a,) = r["arbitrations"]
    assert (a["isolation_violated"], a["n_leaked_files"]) == (True, 2)
    assert open(f"{work}/app.py").read() == "v0\n" and not os.path.exists(f"{work}/leak.txt")


def test_copy_paths_written_into_files_name_the_workdir_after_apply(tmp_path):
    def writing(k, messages, work):
        if len(messages) == 1:
            return Reply("", [tc("bash", command=f"echo 'cat {work}/app.py' > {work}/run.sh")])
        return Reply("", [tc("submit", report="wrote run.sh")])

    _, r, f, _, work, _ = arbitrate(tmp_path, writing, [judgement(0)])
    assert open(f"{work}/run.sh").read() == f"cat {work}/app.py\n"
    judge = [q for q in f[ORCH].requests if q["system"].startswith("You are an impartial")][0]["messages"][0].content
    assert "/arb/" not in judge


def test_apply_of_many_files_stays_under_the_argv_limit(tmp_path, monkeypatch):
    sizes, sh = [], arbiter._sh
    monkeypatch.setattr(arbiter, "SCRIPT_MAX_BYTES", 3000)
    monkeypatch.setattr(arbiter, "_sh", lambda env, script: sizes.append(len(script.encode())) or sh(env, script))

    def many(k, messages, work):
        if len(messages) == 1:
            return Reply("", [tc("bash", command=f"mkdir -p {work}/lib && for i in $(seq 60); do echo $i > {work}/lib/m$i.py; done")])
        return Reply("", [tc("submit", report="ok")])

    _, r, _, _, work, _ = arbitrate(tmp_path, many, [judgement(0)])
    assert r["arbitrations"][0]["n_applied_files"] == 60 and len(os.listdir(f"{work}/lib")) == 60
    assert max(sizes) <= 3000


def test_debate_when_not_confident_then_original_report_plus_revision(tmp_path):
    def debate(k, messages):
        assert "Other candidates' reports" in messages[0].content
        return Reply(f"revised {k}")

    _, r, f, result, work, _ = arbitrate(tmp_path, editing, [judgement(0, 0.6, False), judgement(2, 0.95)], debate)
    (a,) = r["arbitrations"]
    assert a["debate_rounds"] == 1 and [x["winner"] for x in a["rounds"]] == [0, 2] and a["final_winner"] == 2
    assert len(a["rounds"][1]["debate_call_ids"]) == 3
    debates = [c for c in r["calls"] if c["role"] == "debate"]
    assert len(debates) == 3 and {c["dispatch_id"] for c in debates} == set(a["candidate_dispatch_ids"])
    assert "candidate 2 wrote v2" in result and "post-debate revision" in result and "revised 2" in result
    assert open(f"{work}/app.py").read() == "v2\n"


def test_judge_cut_off_gets_one_retry(tmp_path):
    cut = Reply("", finish_reason="length", reasoning="...")
    _, r, f, result, _, _ = arbitrate(tmp_path, editing, [cut, judgement(1)])
    (a,) = r["arbitrations"]
    assert len(a["rounds"]) == 1 and len(a["rounds"][0]["judge_call_ids"]) == 2 and a["final_winner"] == 1
    judge_reqs = [q for q in f[ORCH].requests if q["system"].startswith("You are an impartial")]
    assert JUDGE_RETRY_NOTE in judge_reqs[1]["messages"][0].content


def test_failed_later_round_keeps_the_earlier_judgement(tmp_path):
    _, r, _, result, work, _ = arbitrate(tmp_path, editing, [judgement(2, 0.5, False), Reply("no idea")],
                                         lambda k, m: Reply("same"))
    (a,) = r["arbitrations"]
    assert [x["parse_ok"] for x in a["rounds"]] == [True, False] and (a["final_winner"], a["final_confidence"]) == (2, 0.5)
    assert open(f"{work}/app.py").read() == "v2\n"


def test_easy_scope_shares_the_workdir_and_json_text_fallback(tmp_path):
    def reading(k, messages, work):
        return Reply("", [tc("list_dir")]) if len(messages) == 1 else Reply("", [tc("submit", report="saw app.py")])

    text = Reply('Verdict: {"winner": 1, "confidence": 0.85, "agreement": true, "rationale": "all same"}')
    _, r, f, result, work, _ = arbitrate(tmp_path, reading, [text], scope="easy")
    (a,) = r["arbitrations"]
    assert (a["isolation"], a["final_winner"], a["isolation_violated"], a["applied"]) == ("shared", 1, None, False)
    assert "private copy" not in [q for q in f[ORCH].requests if q["system"].startswith("You are an impartial")][0][
        "messages"][0].content
    assert "applied files" not in result


def test_a_failed_copy_ends_the_episode_before_any_candidate_runs(tmp_path):
    from subagents_bench.environment import LocalShellEnvironment
    from subagents_bench.orchestrator import Orchestrator
    from subagents_bench.records import EpisodeRecorder

    work = tmp_path / "work"
    work.mkdir()
    (work / "app.py").write_text("v0\n")
    env = LocalShellEnvironment(os.path.realpath(work))
    os.makedirs(f"{env.state_root}/arb/arb-1")
    open(f"{env.state_root}/arb/arb-1/base", "w").close()  # the base copy fails (as on a full disk)
    f = FakeFactory({ORCH: [Reply("", [tc("spawn_subagents", instructions=["x"])])]})
    cfg = make_config(subagent={"strategy": "arbiter", "scope": "hard"})
    rec = EpisodeRecorder(tmp_path / "ep", cfg, task_name="t")
    with pytest.raises(RuntimeError, match="arbiter shell step failed"):
        Orchestrator(cfg, env, rec, env.workdir, f).run("x")
    env.close()
    assert rec.episode["exit_reason"] == "error" and f[SUB].requests == []
    assert os.listdir(work) == ["app.py"] and (work / "app.py").read_text() == "v0\n"


def test_a_stop_during_the_cleanup_still_records_the_arbitration(tmp_path, monkeypatch):
    import fakes
    from subagents_bench.environment import LocalShellEnvironment

    class StopOnCleanup(LocalShellEnvironment):
        def exec(self, command, timeout):
            if command.startswith("rm -rf") and "/arb/" in command:
                self.stop_event.set()  # harbor's agent timeout fires while the copies are removed
            return super().exec(command, timeout)

    monkeypatch.setattr(fakes, "LocalShellEnvironment", StopOnCleanup)
    (tmp_path / "work").mkdir()
    f = FakeFactory({ORCH: [Reply("", [tc("spawn_subagents", instructions=["x"])]), judgement(0)],
                     SUB: lambda *_: Reply("", [tc("submit", report="ok")])})
    exit_reason, _ = run_local(make_config(subagent={"strategy": "arbiter"}), f, tmp_path / "work", tmp_path / "ep")
    r = assert_consistent(tmp_path / "ep")  # the spawn's arbitration id has its record
    assert exit_reason == "cancelled" and r["arbitrations"][0]["final_winner"] == 0


def test_parse_judgement_rules():
    def resp(text="", calls=()):
        return LLMResponse(text, list(calls), "stop")

    assert parse_judgement(resp(calls=[tc("submit_judgement", winner=True, confidence=1, agreement=True)]), 3)["winner"] is None
    assert parse_judgement(resp('{"winner": 5}'), 3)["winner"] is None
    partial = parse_judgement(resp('```json\n{"winner": 2, "confidence": "high"}\n```'), 3)
    assert (partial["winner"], partial["confidence"], partial["parse_ok"]) == (2, None, False)


@pytest.mark.docker
def test_copy_isolation_in_a_linux_container(tmp_path):
    from subagents_bench.orchestrator import Orchestrator
    from subagents_bench.records import EpisodeRecorder
    from test_tools import DockerEnv

    d = DockerEnv("ubuntu:24.04")
    try:
        d.exec("mkdir -p /app && echo v0 > /app/app.py && echo old > /app/old.txt", 30)
        judges = [judgement(1)]

        def orch(system, messages, tools):
            if system.startswith("You are an impartial arbiter"):
                return judges.pop(0)
            return Reply("", [tc("spawn_subagents", instructions=["fix"])]) if len(messages) == 1 else Reply("", [tc("finish")])

        f = FakeFactory({ORCH: orch, SUB: lambda s, m, t: editing(int(re.search(r"cand(\d)", s).group(1)), m, "/app")})
        cfg = make_config(subagent={"strategy": "arbiter", "scope": "hard"})
        rec = EpisodeRecorder(tmp_path / "ep", cfg, task_name="t")
        assert Orchestrator(cfg, d, rec, "/app", f).run("x") == "finished"
        (a,) = assert_consistent(tmp_path / "ep")["arbitrations"]
        assert (a["isolation_violated"], a["n_applied_files"]) == (False, 3)
        assert d.exec("cat /app/app.py /app/new.txt; ls /app; ls /tmp/.sb/arb", 30).stdout.split() == ["v1", "new", "app.py", "new.txt"]
    finally:
        d.close()
