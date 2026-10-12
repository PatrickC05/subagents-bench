"""The arbiter strategy: judgement parsing, copy isolation on LocalShell,
winner application (and losers not applied), confidence-aware debate, the
JSON-text fallback, parse failures, shared/shared_fallback isolation, and
concurrent arbitrations not clobbering each other's applied files."""

from __future__ import annotations

import re

import pytest

from fakes import ORCH, SUB, FakeFactory, Reply, assert_valid, make_config, run, tc
from subagents_bench.arbiter import judgement_from_obj, parse_judgement
from subagents_bench.types import CallTiming, LLMError, LLMResponse, RequestInfo, Usage


def resp(text="", tool_calls=()):
    req = RequestInfo(model="m", provider="p", base_url=None, completion_window=None, max_tokens=1, temperature=0.0,
                      extra_body={}, n_messages=1, n_tools=1, prompt_chars=1, system_hash="", tools_hash="",
                      msg_hashes=[], msg_chars=[], reasoning_passback=True)
    return LLMResponse(text=text, tool_calls=list(tool_calls), usage=Usage(), finish_reason="stop", cost_usd=0.0,
                       timing=CallTiming(t_start=0.0, t_end=0.0), request=req)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_parse_tool_call():
    j = parse_judgement(resp(tool_calls=[tc("submit_judgement", winner=2, confidence=0.7, agreement=False, rationale="r")]), 3)
    assert (j.winner, j.confidence, j.agreement, j.rationale, j.parse_ok) == (2, 0.7, False, "r", True)


def test_parse_json_text_fallbacks():
    fenced = 'My verdict:\n```json\n{"winner": 1, "confidence": "0.9", "agreement": "yes", "rationale": "ok"}\n```'
    j = parse_judgement(resp(fenced), 3)
    assert (j.winner, j.confidence, j.agreement, j.parse_ok) == (1, 0.9, True, True)
    inline = 'I pick {"winner": 0, "confidence": 0.4, "agreement": false, "rationale": "close"} overall.'
    j = parse_judgement(resp(inline), 2)
    assert (j.winner, j.confidence, j.agreement, j.parse_ok) == (0, 0.4, False, True)


@pytest.mark.parametrize(
    "text, calls",
    [
        ("Candidate 1 is best.", []),
        ('{"winner": 5, "confidence": 0.9, "agreement": true}', []),  # out of range
        ("", [tc("submit_judgement", winner="first", confidence=0.9, agreement=True)]),
        ("{not json", []),
    ],
)
def test_parse_failure_falls_back_to_candidate_zero(text, calls):
    j = parse_judgement(resp(text, calls), 3)
    assert (j.winner, j.confidence, j.agreement, j.parse_ok) == (0, None, None, False)


def test_partial_judgement_keeps_winner_but_is_not_ok():
    j = judgement_from_obj({"winner": 1, "confidence": 3, "agreement": "maybe", "rationale": None}, 2)
    assert (j.winner, j.confidence, j.agreement, j.rationale, j.parse_ok) == (1, None, None, "", False)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


def arb_config(tmp_path, **sub):
    base = {"strategy": "arbiter", "scope": "hard", "max_concurrent": 3, "arbiter": {"n_candidates": 3}}
    for k, v in sub.items():
        base[k] = {**base[k], **v} if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return make_config(tmp_path, subagent=base)


def orchestrator(judgements, instructions=("write result.txt",)):
    """Orchestrator script: spawn, then finish; judge calls pop ``judgements``."""
    js = list(judgements)

    def script(system, messages, tools, client):
        if "submit_judgement" in [t.name for t in tools]:
            j = js.pop(0)
            return j if isinstance(j, (Reply, Exception)) else Reply("", [tc("submit_judgement", **j)])
        if not any(m.role == "assistant" for m in messages):
            return Reply("", [tc("spawn_subagents", instructions=list(instructions))])
        return Reply("", [tc("finish")])

    return script


def candidate(system, messages, tools, client):
    """Each candidate writes result.txt (relative: lands in its own copy)."""
    if not tools:  # debate call
        own = re.search(r"Your report:\n(.*?)\n", messages[0].content).group(1)
        return Reply(f"revised {own}")
    k = re.search(r"cand(\d)", system).group(1)
    if len(messages) == 1:
        return Reply("", [tc("edit_file", path="result.txt", old="", new=f"answer-from-cand{k}\n")])
    return Reply("", [tc("submit", report=f"report {k}")])


def test_copy_isolation_judge_and_apply_winner(tmp_path):
    cfg = arb_config(tmp_path)
    work = tmp_path / "work"
    (work / "keep.txt").write_text("untouched\n")
    f = FakeFactory({ORCH: orchestrator([{"winner": 2, "confidence": 0.95, "agreement": True, "rationale": "c2"}]),
                     SUB: candidate})
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    # Only the winner's file landed in the real workdir.
    assert (work / "result.txt").read_text() == "answer-from-cand2\n"
    assert (work / "keep.txt").read_text() == "untouched\n"
    a = r["arbitrations"][0]
    assert (a["isolation"], a["final_winner"], a["final_confidence"], a["agreement"]) == ("copy", 2, 0.95, True)
    assert a["applied"] and a["applied_files"] == [str(work / "result.txt")] and a["apply_error"] is None
    assert a["debate_rounds_used"] == 0 and len(a["rounds"]) == 1 and a["rounds"][0]["parse_ok"]
    assert a["t_start"] <= a["t_candidates_end"] <= a["t_end"]
    # Each candidate worked in its own copy under the private state root, not the workdir.
    ds = sorted(r["dispatches"], key=lambda d: d["candidate_index"])
    assert [d["candidate_index"] for d in ds] == [0, 1, 2] and a["candidate_dispatch_ids"] == [d["dispatch_id"] for d in ds]
    for k, d in enumerate(ds):
        assert d["isolation"] == "copy" and d["workdir"].endswith(f"/arb/{a['arbitration_id']}/cand{k}")
        assert not d["workdir"].startswith(str(work))
        assert d["files_modified"] == [d["workdir"] + "/result.txt"]
    # The judge saw every report and a diff per candidate.
    judge_req = f[ORCH].requests[1]
    assert judge_req["tools"] == ["submit_judgement"]
    prompt = judge_req["messages"][0].content
    for k in range(3):
        assert f"report {k}" in prompt and f"+answer-from-cand{k}" in prompt
    judge_call = [c for c in r["calls"] if c["role"] == "arbiter"][0]
    assert judge_call["agent_id"] == f"arbiter-{a['arbitration_id']}" and judge_call["step"] == 1
    assert judge_call["call_id"] == a["rounds"][0]["judge_call_id"]
    # The orchestrator got the winner's report with the arbiter summary and applied files.
    out = f[ORCH].requests[2]["messages"][-1].tool_results[0].content
    assert "candidate 3 of 3, confidence 0.95, candidates agreed: yes" in out
    assert "report 2" in out and f"applied files: {work / 'result.txt'}" in out
    assert r["spawns"][0]["files_modified"] == [str(work / "result.txt")]
    # Copies are removed after the apply.
    assert not any("cand" in str(p) for p in work.rglob("*"))


def test_debate_runs_only_when_not_confident(tmp_path):
    cfg = arb_config(tmp_path, scope="medium")
    f = FakeFactory({ORCH: orchestrator([
        {"winner": 0, "confidence": 0.6, "agreement": True, "rationale": "unsure"},
        {"winner": 1, "confidence": 0.9, "agreement": True, "rationale": "after debate"},
    ]), SUB: lambda s, m, t, c: Reply("revised take") if not t else Reply("", [tc("submit", report="plain")])})
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    a = r["arbitrations"][0]
    assert a["isolation"] == "shared" and not a["applied"] and a["applied_files"] == []
    assert a["debate_rounds_used"] == 1 and [x["round"] for x in a["rounds"]] == [1, 2]
    assert a["rounds"][0]["debate_call_ids"] == [] and len(a["rounds"][1]["debate_call_ids"]) == 3
    assert (a["final_winner"], a["final_confidence"]) == (1, 0.9)
    debates = [c for c in r["calls"] if c["role"] == "debate"]
    cand_agents = {d["agent_id"]: d["dispatch_id"] for d in r["dispatches"]}
    assert len(debates) == 3 and all(c["step"] == 2 and c["n_tools"] == 0 for c in debates)
    assert {c["agent_id"] for c in debates} == set(cand_agents)
    assert all(cand_agents[c["agent_id"]] == c["dispatch_id"] for c in debates)
    assert sorted(c["call_id"] for c in debates) == sorted(a["rounds"][1]["debate_call_ids"])
    # Round 2's judge saw the revised reports.
    assert f[ORCH].requests[2]["messages"][0].content.count("revised take") == 3
    assert "debate round 1" in f[ORCH].requests[2]["messages"][0].content
    # Debate calls continue the candidate's own chain (validated above) and end after it.
    for c in debates:
        cand = [d for d in r["dispatches"] if d["agent_id"] == c["agent_id"]][0]
        assert c["t_start"] >= cand["t_end"]


def test_confident_but_disagreeing_debates_and_rounds_are_capped(tmp_path):
    cfg = arb_config(tmp_path, scope="easy", arbiter={"debate_rounds": 1})
    f = FakeFactory({ORCH: orchestrator([
        {"winner": 1, "confidence": 0.99, "agreement": False, "rationale": "they differ"},
        {"winner": 1, "confidence": 0.5, "agreement": False, "rationale": "still differ"},
    ]), SUB: lambda s, m, t, c: Reply("rev") if not t else Reply("", [tc("submit", report="x")])})
    res = run(cfg, f)
    a = assert_valid(res.episode_dir)["arbitrations"][0]
    # Not confident after round 2 either, but only one debate round is allowed.
    assert a["debate_rounds_used"] == 1 and len(a["rounds"]) == 2 and a["final_confidence"] == 0.5


def test_json_text_fallback_and_parse_failure(tmp_path):
    cfg = arb_config(tmp_path, scope="easy", arbiter={"debate_rounds": 1})
    f = FakeFactory({ORCH: orchestrator([
        Reply("no tool, sorry"),  # unparseable -> winner 0, confidence None -> debate
        Reply('{"winner": 2, "confidence": 0.85, "agreement": true, "rationale": "text json"}'),
    ]), SUB: lambda s, m, t, c: Reply("rev") if not t else Reply("", [tc("submit", report="x")])})
    res = run(cfg, f)
    a = assert_valid(res.episode_dir)["arbitrations"][0]
    r1, r2 = a["rounds"]
    assert (r1["winner"], r1["confidence"], r1["agreement"], r1["parse_ok"]) == (0, None, None, False)
    assert (r2["winner"], r2["confidence"], r2["agreement"], r2["parse_ok"]) == (2, 0.85, True, True)
    assert a["final_winner"] == 2


def test_judge_failure_keeps_episode_going(tmp_path):
    cfg = arb_config(tmp_path, scope="easy")
    f = FakeFactory({ORCH: orchestrator([LLMError("judge provider down")]),
                     SUB: lambda s, m, t, c: Reply("", [tc("submit", report="x")])})
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    assert res.exit_reason == "finished"
    a = r["arbitrations"][0]
    assert (a["final_winner"], a["final_confidence"], len(a["rounds"]), a["rounds"][0]["parse_ok"]) == (0, None, 1, False)
    judge = [c for c in r["calls"] if c["role"] == "arbiter"][0]
    assert judge["error"].startswith("LLMError") and a["rounds"][0]["judge_call_id"] == judge["call_id"]


def test_shared_fallback_when_workdir_too_big(tmp_path):
    cfg = arb_config(tmp_path, arbiter={"max_copy_mb": 0})
    (tmp_path / "work" / "data.bin").write_bytes(b"x" * 20000)
    f = FakeFactory({ORCH: orchestrator([{"winner": 0, "confidence": 1.0, "agreement": True, "rationale": "-"}]),
                     SUB: lambda s, m, t, c: Reply("", [tc("submit", report="x")])})
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    a = r["arbitrations"][0]
    assert a["isolation"] == "shared_fallback" and not a["applied"]
    assert {d["isolation"] for d in r["dispatches"]} == {"shared_fallback"}
    assert {d["workdir"] for d in r["dispatches"]} == {str(tmp_path / "work")}
    assert "Isolation:" not in f[SUB].requests[0]["system"]


def test_concurrent_arbitrations_apply_both_winners(tmp_path):
    cfg = arb_config(tmp_path, max_concurrent=4, arbiter={"n_candidates": 2})

    def cand(system, messages, tools, client):
        k = re.search(r"cand(\d)", system).group(1)
        name = "alpha.txt" if "alpha" in messages[0].content else "beta.txt"
        if len(messages) == 1:
            return Reply("", [tc("edit_file", path=name, old="", new=f"{name} by cand{k}\n")], latency=0.1)
        return Reply("", [tc("submit", report=f"{name} {k}")])

    def orch(system, messages, tools, client):
        if "submit_judgement" in [t.name for t in tools]:
            # The beta judge answers later, so the two applies happen one after
            # the other: beta's copy (made before alpha was applied) lacks
            # alpha.txt, which a plain two-way sync would delete.
            late = "make beta" in messages[0].content
            return Reply("", [tc("submit_judgement", winner=1, confidence=0.9, agreement=True, rationale="-")],
                         latency=0.6 if late else 0.0)
        if not any(m.role == "assistant" for m in messages):
            return Reply("", [tc("spawn_subagents", instructions=["make alpha", "make beta"])])
        return Reply("", [tc("finish")])

    f = FakeFactory({ORCH: orch, SUB: cand})
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    work = tmp_path / "work"
    # Both winners' files survive: applying one arbitration never deletes the other's file.
    assert (work / "alpha.txt").read_text() == "alpha.txt by cand1\n"
    assert (work / "beta.txt").read_text() == "beta.txt by cand1\n"
    sp = r["spawns"][0]
    assert (sp["granted"], sp["concurrency"], len(sp["dispatch_ids"]), len(sp["arbitration_ids"])) == (2, 4, 4, 2)
    assert f[SUB].max_in_flight == 4
    by_instr = {a["instruction_index"]: a for a in r["arbitrations"]}
    assert sp["arbitration_ids"] == [by_instr[0]["arbitration_id"], by_instr[1]["arbitration_id"]]
    assert sp["dispatch_ids"] == by_instr[0]["candidate_dispatch_ids"] + by_instr[1]["candidate_dispatch_ids"]
    assert by_instr[0]["t_end"] < by_instr[1]["t_end"] - 0.3  # applies were sequential
    assert by_instr[1]["applied_files"] == [str(work / "beta.txt")]


def test_long_reports_are_capped_for_judge_and_debate(tmp_path):
    cfg = arb_config(tmp_path, scope="easy", max_output_chars=1000, arbiter={"n_candidates": 2})
    long = "A" * 3000 + "MIDDLE" + "Z" * 3000
    f = FakeFactory({ORCH: orchestrator([
        {"winner": 0, "confidence": 0.1, "agreement": False, "rationale": "-"},
        {"winner": 1, "confidence": 0.9, "agreement": True, "rationale": "-"},
    ]), SUB: lambda s, m, t, c: Reply("rev") if not t else Reply("", [tc("submit", report=long)])})
    res = run(cfg, f)
    assert_valid(res.episode_dir)
    judge_prompt = f[ORCH].requests[1]["messages"][0].content
    assert "MIDDLE" not in judge_prompt and "characters elided" in judge_prompt
    assert len(judge_prompt) < 2 * 1100 + 1000
    debate_prompts = [r["messages"][0].content for r in f[SUB].requests if not r["tools"]]
    assert len(debate_prompts) == 2 and all("MIDDLE" not in p and len(p) < 2 * 1100 + 500 for p in debate_prompts)


TB_IMAGE = "alexgshaw/fix-git:20251031"


@pytest.mark.docker
def test_arbiter_copy_isolation_in_terminal_bench_container(tmp_path):
    """Real Linux coreutils (cp -a, md5sum, diff, timeout) in a Terminal-Bench
    image: candidates edit private copies of the git repo, only the winner's
    change reaches the WORKDIR. Runs only with RUN_DOCKER=1."""
    import subprocess

    from subagents_bench.environment import LocalDockerEnvironment
    from subagents_bench.runner import run_task

    have = subprocess.run(["docker", "image", "inspect", TB_IMAGE], capture_output=True).returncode == 0
    if not have:
        pytest.skip(f"{TB_IMAGE} not pulled")
    cfg = arb_config(tmp_path)
    cfg.sandbox.kind, cfg.sandbox.image, cfg.sandbox.workdir = "local_docker", TB_IMAGE, "auto"

    def cand(system, messages, tools, client):
        if not tools:
            return Reply("revised")
        k = re.search(r"cand(\d)", system).group(1)
        if len(messages) == 1:
            return Reply("", [tc("edit_file", path="arbiter_note.txt", old="", new=f"from cand{k}\n"),
                              tc("bash", command="git status --short | head -5; pwd")])
        return Reply("", [tc("submit", report=f"wrote note {k}")])

    f = FakeFactory({ORCH: orchestrator([{"winner": 1, "confidence": 0.9, "agreement": True, "rationale": "-"}]),
                     SUB: cand})
    with LocalDockerEnvironment(TB_IMAGE, workdir=None, platform="linux/amd64") as env:
        res = run_task(cfg, "fix-git", "write a note", env, client_factory=f)
        wd = res.record["workdir"]
        note = env.exec(f"cat {wd}/arbiter_note.txt; git -C {wd} status --short", timeout_sec=60).stdout
    r = assert_valid(res.episode_dir)
    ep, a = r["episode"], r["arbitrations"][0]
    assert wd == "/app/personal-site" and ep["container_arch"] == "x86_64" and ep["has_timeout_binary"] is True
    assert a["isolation"] == "copy" and a["applied"] and f"{wd}/arbiter_note.txt" in a["applied_files"]
    # The winner's `git status` refreshed its index's stat cache; that file is applied too.
    assert set(a["applied_files"]) <= {f"{wd}/arbiter_note.txt", f"{wd}/.git/index"}
    assert note.startswith("from cand1\n") and "?? arbiter_note.txt" in note
    assert all(d["workdir"].startswith("/tmp/.sb/arb/") for d in r["dispatches"])
    judge_prompt = f[ORCH].requests[1]["messages"][0].content
    assert "+from cand0" in judge_prompt and "+from cand2" in judge_prompt
