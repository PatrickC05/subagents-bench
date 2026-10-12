"""The arbiter strategy: judgement parsing, copy isolation on LocalShell,
winner application (and losers not applied), path remapping into the
candidates' copies, leak containment (restore + record), copy paths mapped
back on apply, confidence-aware debate (and what the orchestrator is told
after one), the JSON-text fallback, parse failures and failed later rounds
keeping the earlier judgement, shared/shared_fallback isolation, and
concurrent arbitrations not clobbering each other's applied files."""

from __future__ import annotations

import os
import re

import pytest

from fakes import ORCH, SUB, FakeFactory, Reply, assert_valid, make_config, run, tc
from subagents_bench.arbiter import judgement_from_obj, parse_judgement
from subagents_bench.analysis.latency import critical_path
from subagents_bench.analysis.load import load_episode
from subagents_bench.prompts import DEBATE_FROZEN_NOTE, JUDGE_RETRY_NOTE, POST_DEBATE_HEADER, POST_DEBATE_NOTE
from subagents_bench.schema import ArbitrationRecord, validate_record
from subagents_bench.tools import PathRemap
from subagents_bench.types import CallTiming, LLMError, LLMResponse, RequestInfo, Usage

# ArbitrationRecord fields added in schema_version 3 (absent from older records).
NEW_FIELDS = ("isolation_violated", "leaked_files", "n_path_remaps", "final_from_round")


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
    # Ranked between a complete judgement and the fallback.
    assert j.rank == 1 and parse_judgement(resp("no verdict"), 2).rank == 0
    assert judgement_from_obj({"winner": 1, "confidence": 0.5, "agreement": True}, 2).rank == 2


# ---------------------------------------------------------------------------
# path remapping (candidates in private copies)
# ---------------------------------------------------------------------------

COPY = "/tmp/.sb/arb/a1/cand0"


@pytest.mark.parametrize(
    "command, expected",
    [
        ("cd /app && ls", f"cd {COPY} && ls"),
        ("echo x > /app/out.txt; cat '/app/a b' \"/app\"", f"echo x > {COPY}/out.txt; cat '{COPY}/a b' \"{COPY}\""),
        ("python3 -c \"open('/app/s.txt', 'w')\"", f"python3 -c \"open('{COPY}/s.txt', 'w')\""),
        ("cat <<EOF > /app/run.sh\npython /app/p.py\nEOF", f"cat <<EOF > {COPY}/run.sh\npython {COPY}/p.py\nEOF"),
        ("PATH=/app/bin:$PATH x=/app (cd /app)", f"PATH={COPY}/bin:$PATH x={COPY} (cd {COPY})"),
        # Not the workdir: longer names, other roots, home-relative, already in the copy.
        ("ls /apple /app_data /app.bak /srv/app ~/app $HOME/app", None),
        ("cat ${HOME}/app/x; cd $(pwd)/app", None),
        (f"ls {COPY}/x", None),
    ],
)
def test_path_remap_rewrites_whole_path_prefixes_in_commands(command, expected):
    assert PathRemap("/app", COPY).command(command) == (command if expected is None else expected)


def test_path_remap_paths_and_tool_args():
    r = PathRemap("/app/", COPY)
    assert r.src == "/app" and r.path("/app") == COPY and r.path("/app/src/x.py") == f"{COPY}/src/x.py"
    for p in ["src/x.py", "/apple/x", "/app/../etc/passwd", f"{COPY}/x", "/etc/hosts"]:
        assert r.path(p) == p
    args = {"path": "/app/a.txt", "old": "/app", "new": "/app/x"}
    assert r.args("edit_file", args) == {"path": f"{COPY}/a.txt", "old": "/app", "new": "/app/x"}  # content untouched
    assert args["path"] == "/app/a.txt" and r.n == 1  # the model's arguments are not mutated
    rel = {"path": "rel.txt"}
    assert r.args("read_file", rel) is rel and r.n == 1
    assert r.args("bash", {"command": "cat /app/a"}) == {"command": f"cat {COPY}/a"} and r.n == 2
    for name, a in [("search", {"pattern": "/app", "path": "."}), ("list_dir", {}), ("submit", {"path": "/app"})]:
        assert r.args(name, a) is a
    assert r.n == 2
    # Never for "/" (everything would move); a copy under the workdir is not remapped again.
    assert PathRemap("/", "/c").command("ls /") == "ls /" and PathRemap("/", "/c").path("/x") == "/x"
    nested = PathRemap("/tmp", COPY)
    assert nested.command(f"ls /tmp/x {COPY}/y") == f"ls {COPY}/x {COPY}/y" and nested.path(f"{COPY}/q") == f"{COPY}/q"


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
    assert (a["isolation_violated"], a["leaked_files"], a["n_path_remaps"], a["final_from_round"]) == (False, [], 0, 1)
    # A record written before the schema_version 3 fields still validates.
    assert validate_record(ArbitrationRecord, {k: v for k, v in a.items() if k not in NEW_FIELDS}) == []
    assert validate_record(ArbitrationRecord, {**a, "n_path_remaps": "2"}) != []
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


def test_absolute_workdir_paths_land_in_each_candidates_copy(tmp_path):
    # Candidates that name the REAL workdir (as the orchestrator's instruction
    # does) still write only their own copy: file-tool paths and bash
    # commands are redirected, so the judge's pick is what gets applied.
    cfg = arb_config(tmp_path)
    work = tmp_path / "work"
    (work / "keep.txt").write_text("untouched\n")

    def cand(system, messages, tools, client):
        k = int(re.search(r"cand(\d)", system).group(1))
        if len(messages) == 1:
            if k == 0:
                return Reply("", [tc("edit_file", path=f"{work}/solution.txt", old="", new="from-0\n")])
            return Reply("", [tc("bash", command=f"sleep 0.{3 - k}; echo from-{k} > {work}/solution.txt && cd {work}"),
                              tc("read_file", path=f"{work}/solution.txt")])
        return Reply("", [tc("submit", report=f"wrote {k}")])

    f = FakeFactory({ORCH: orchestrator([{"winner": 1, "confidence": 0.9, "agreement": True, "rationale": "-"}],
                                        instructions=[f"write the answer to {work}/solution.txt"]), SUB: cand})
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    a = r["arbitrations"][0]
    assert (work / "solution.txt").read_text() == "from-1\n" and (work / "keep.txt").read_text() == "untouched\n"
    assert a["isolation"] == "copy" and a["applied_files"] == [str(work / "solution.txt")]
    assert (a["isolation_violated"], a["leaked_files"], a["n_path_remaps"]) == (False, [], 5)
    for d in r["dispatches"]:
        assert d["files_modified"] == [d["workdir"] + "/solution.txt"]
    # The record keeps the model's own arguments; the model is told about the redirect.
    reads = [t for t in r["tools"] if t["role"] == "subagent" and t["name"] == "read_file"]
    assert len(reads) == 2 and all(t["args"]["path"] == f"{work}/solution.txt" for t in reads)
    results = [tr.content for req in f[SUB].requests for m in req["messages"] for tr in m.tool_results]
    redirected = [c for c in results if f"[note: {work} was redirected to your private copy " in c]
    assert len(redirected) == 5 and any(c.startswith("     1\tfrom-2") for c in results)
    # Each judge diff shows only that candidate's own file.
    judge_prompt = f[ORCH].requests[1]["messages"][0].content
    assert judge_prompt.count("A solution.txt") == 3 and "keep.txt" not in judge_prompt


def test_leaks_into_the_real_workdir_are_restored_and_recorded(tmp_path):
    # A write the remap cannot see (the path is assembled at run time) reaches
    # the real workdir; it is undone before the winner is applied, recorded,
    # and does not show up as a change in anyone's diff.
    cfg = arb_config(tmp_path)
    work = tmp_path / "work"
    (work / "keep.txt").write_text("untouched\n")
    (work / "gone.txt").write_text("still here\n")
    head, tail = str(work)[:-2], str(work)[-2:]

    def cand(system, messages, tools, client):
        k = int(re.search(r"cand(\d)", system).group(1))
        if len(messages) == 1:
            if k == 2:
                cmd = (f"w='{head}'\"{tail}\"; echo leaked > \"$w/keep.txt\"; mkdir \"$w/new dir\"; "
                       "echo x > \"$w/new dir/n.txt\"; rm \"$w/gone.txt\"")
                return Reply("", [tc("bash", command=cmd)])
            return Reply("", [tc("edit_file", path="result.txt", old="", new=f"answer-from-cand{k}\n")])
        return Reply("", [tc("submit", report=f"report {k}")])

    f = FakeFactory({ORCH: orchestrator([{"winner": 0, "confidence": 0.9, "agreement": True, "rationale": "-"}]),
                     SUB: cand})
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    a = r["arbitrations"][0]
    assert (work / "keep.txt").read_text() == "untouched\n" and (work / "gone.txt").read_text() == "still here\n"
    assert not (work / "new dir").exists()
    assert (work / "result.txt").read_text() == "answer-from-cand0\n"
    assert a["isolation_violated"] is True and a["n_path_remaps"] == 0
    assert a["leaked_files"] == [str(work / p) for p in ("gone.txt", "keep.txt", "new dir/n.txt")]
    assert a["applied"] and a["applied_files"] == [str(work / "result.txt")] and a["apply_error"] is None
    # Diffs are taken against the pristine base copy, so the leak is in nobody's diff.
    judge_prompt = f[ORCH].requests[1]["messages"][0].content
    assert "keep.txt" not in judge_prompt and "gone.txt" not in judge_prompt and "n.txt" not in judge_prompt
    assert judge_prompt.count("A result.txt") == 2 and "no file changes" in judge_prompt


def test_copy_paths_written_into_files_are_mapped_back_on_apply(tmp_path):
    # A remapped command writes the copy's path into file contents and a
    # symlink; the candidate's own run uses its copy consistently, and the
    # applied files name the real workdir.
    cfg = arb_config(tmp_path, arbiter={"n_candidates": 2})
    work = tmp_path / "work"
    (work / "p.txt").write_text("payload\n")

    def cand(system, messages, tools, client):
        if len(messages) == 1:
            cmd = f"printf 'cat {work}/p.txt\\n' > {work}/run.sh && ln -s {work}/p.txt {work}/link.txt && bash {work}/run.sh"
            return Reply("", [tc("bash", command=cmd), tc("read_file", path="run.sh")])
        return Reply("", [tc("submit", report="made run.sh")])

    f = FakeFactory({ORCH: orchestrator([{"winner": 1, "confidence": 0.9, "agreement": True, "rationale": "-"}]),
                     SUB: cand})
    res = run(cfg, f)
    a = assert_valid(res.episode_dir)["arbitrations"][0]
    assert (work / "run.sh").read_text() == f"cat {work}/p.txt\n"
    assert os.readlink(work / "link.txt") == str(work / "p.txt")
    assert sorted(a["applied_files"]) == [str(work / "link.txt"), str(work / "run.sh")]
    # During the run each candidate saw (and ran) its own copy's paths.
    second_steps = [q for q in f[SUB].requests if len(q["messages"]) == 3]
    assert len(second_steps) == 2
    for q in second_steps:
        k = re.search(r"cand(\d)", q["system"]).group(1)
        seen = [tr.content for m in q["messages"] for tr in m.tool_results]
        assert seen[0].startswith("payload") and f"/cand{k}/p.txt" in seen[1]
    assert a["isolation_violated"] is False and a["n_path_remaps"] == 2


def test_copy_isolation_debate_reports_original_plus_revision(tmp_path):
    # After a debate the orchestrator gets the winner's ORIGINAL report (what
    # its applied files reflect) and the tool-less revision under a note.
    cfg = arb_config(tmp_path, arbiter={"debate_rounds": 1})
    work = tmp_path / "work"
    f = FakeFactory({ORCH: orchestrator([
        {"winner": 0, "confidence": 0.4, "agreement": False, "rationale": "split"},
        {"winner": 1, "confidence": 0.9, "agreement": True, "rationale": "1 revised"},
    ]), SUB: candidate})
    res = run(cfg, f)
    a = assert_valid(res.episode_dir)["arbitrations"][0]
    assert (work / "result.txt").read_text() == "answer-from-cand1\n"
    assert (a["final_winner"], a["final_from_round"], a["debate_rounds_used"]) == (1, 2, 1)
    orch_reqs = [q for q in f[ORCH].requests if q["tools"] != ["submit_judgement"]]
    out = orch_reqs[1]["messages"][-1].tool_results[0].content
    body = out.split("candidates agreed: yes)\n", 1)[1]
    assert body.startswith(f"{POST_DEBATE_HEADER}\nreport 1\n\n{POST_DEBATE_NOTE}\nrevised report 1\napplied files: ")
    # Debaters are told their copies are frozen; the round-2 judge to trust the diffs.
    debates = [q["system"] for q in f[SUB].requests if not q["tools"]]
    assert len(debates) == 3 and all(DEBATE_FROZEN_NOTE in s and "do not claim" in s for s in debates)
    judge2 = f[ORCH].requests[2]["messages"][0].content
    assert "weigh them over claims in the revised reports" in judge2 and "+answer-from-cand1" in judge2


@pytest.mark.parametrize("round2", [Reply("I cannot decide.")], ids=["unparseable"])
def test_failed_later_round_keeps_the_earlier_judgement(tmp_path, round2):
    cfg = arb_config(tmp_path, arbiter={"debate_rounds": 1})
    work = tmp_path / "work"
    f = FakeFactory({ORCH: orchestrator([{"winner": 2, "confidence": 0.7, "agreement": False, "rationale": "c2"}, round2]),
                     SUB: candidate})
    res = run(cfg, f)
    a = assert_valid(res.episode_dir)["arbitrations"][0]
    assert res.exit_reason == "finished"
    r1, r2 = a["rounds"]
    assert (r1["winner"], r1["parse_ok"]) == (2, True) and (r2["winner"], r2["parse_ok"]) == (0, False)
    assert (a["final_winner"], a["final_confidence"], a["agreement"], a["final_from_round"]) == (2, 0.7, False, 1)
    assert (work / "result.txt").read_text() == "answer-from-cand2\n"
    out = [q for q in f[ORCH].requests if q["tools"] != ["submit_judgement"]][1]["messages"][-1].tool_results[0].content
    assert "candidate 3 of 3, confidence 0.70, candidates agreed: no" in out


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
    assert "written without tools" in f[ORCH].requests[2]["messages"][0].content
    assert "weigh them over claims" not in f[ORCH].requests[2]["messages"][0].content  # no diffs shown
    # Shared isolation: no frozen-copy note, and the orchestrator gets the revised report.
    assert all(DEBATE_FROZEN_NOTE not in q["system"] for q in f[SUB].requests if not q["tools"])
    out = f[ORCH].requests[3]["messages"][-1].tool_results[0].content
    assert "revised take" in out and POST_DEBATE_NOTE not in out
    assert a["final_from_round"] == 2 and a["isolation_violated"] is None and a["n_path_remaps"] == 0
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
    assert a["final_winner"] == 2 and a["final_from_round"] == 2
    # An unparseable reply that was not cut off gets no retry.
    assert r1["judge_retry_call_id"] is None and r2["judge_retry_call_id"] is None


def judge_requests(f):
    return [q for q in f[ORCH].requests if q["tools"] == ["submit_judgement"]]


def test_judge_cut_off_at_the_output_limit_gets_one_retry(tmp_path):
    cfg = arb_config(tmp_path, scope="easy")
    f = FakeFactory({ORCH: orchestrator([
        Reply("", reasoning="weighing candidate after candidate", finish_reason="length", latency=0.05),
        {"winner": 2, "confidence": 0.9, "agreement": True, "rationale": "brief"},
    ]), SUB: lambda s, m, t, c: Reply("", [tc("submit", report="x")])})
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    a = r["arbitrations"][0]
    judges = [c for c in r["calls"] if c["role"] == "arbiter"]
    assert [c["finish_reason"] for c in judges] == ["length", "tool_calls"]
    (rnd,) = a["rounds"]
    assert (rnd["judge_call_id"], rnd["judge_retry_call_id"]) == (judges[0]["call_id"], judges[1]["call_id"])
    assert (rnd["winner"], rnd["parse_ok"], a["final_winner"], a["final_from_round"]) == (2, True, 2, 1)
    first, retry = (q["messages"][0].content for q in judge_requests(f))
    assert retry == f"{first}\n\n{JUDGE_RETRY_NOTE}"
    # Both judge calls lie on the arbitration's critical path, as judge time.
    cp = critical_path(load_episode(res.episode_dir))
    assert cp.parts["arb_llm"] == pytest.approx(sum(c["latency_sec"] for c in judges))
    assert cp.parts["arb_llm"] >= 0.05


@pytest.mark.parametrize("retry", [Reply("", finish_reason="length")], ids=["cut-again"])
def test_judge_retry_is_made_once_and_its_failure_keeps_the_fallback(tmp_path, retry):
    cfg = arb_config(tmp_path, scope="easy", arbiter={"debate_rounds": 0})
    f = FakeFactory({ORCH: orchestrator([Reply("", finish_reason="length"), retry]),
                     SUB: lambda s, m, t, c: Reply("", [tc("submit", report="x")])})
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    assert res.exit_reason == "finished"
    a = r["arbitrations"][0]
    (rnd,) = a["rounds"]
    judges = [c for c in r["calls"] if c["role"] == "arbiter"]
    assert len(judges) == 2 and rnd["judge_retry_call_id"] == judges[1]["call_id"]
    assert (rnd["winner"], rnd["parse_ok"], a["final_winner"], a["final_from_round"]) == (0, False, 0, None)


def test_debate_provider_failure_stops_the_episode(tmp_path):
    cfg = arb_config(tmp_path, scope="easy", arbiter={"debate_rounds": 1})
    f = FakeFactory({ORCH: orchestrator([{"winner": 1, "confidence": 0.5, "agreement": False, "rationale": "-"}]),
                     SUB: lambda s, m, t, c: LLMError("debate provider down") if not t
                     else Reply("", [tc("submit", report="x")])})
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    assert res.exit_reason == "provider_error" and r["episode"]["n_provider_errors"] >= 1
    assert r["episode"]["orchestrator_steps"] == 1


def test_judge_provider_failure_stops_the_episode(tmp_path):
    cfg = arb_config(tmp_path, scope="easy")
    f = FakeFactory({ORCH: orchestrator([LLMError("judge provider down")]),
                     SUB: lambda s, m, t, c: Reply("", [tc("submit", report="x")])})
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    # A provider failure makes the trial invalid: the episode ends instead of finishing.
    assert res.exit_reason == "provider_error" and str(res.cause) == "judge provider down"
    a = r["arbitrations"][0]
    assert (a["final_winner"], a["final_confidence"], len(a["rounds"]), a["rounds"][0]["parse_ok"]) == (0, None, 1, False)
    assert a["final_from_round"] is None and a["rounds"][0]["rationale"].startswith("judge call failed: ")
    judge = [c for c in r["calls"] if c["role"] == "arbiter"][0]
    assert judge["error"].startswith("LLMError") and a["rounds"][0]["judge_call_id"] == judge["call_id"]
    assert r["episode"]["n_provider_errors"] == 1 and not a["applied"]
    assert r["episode"]["orchestrator_steps"] == 1  # no step after the spawn


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
    # alpha.txt, applied while beta ran, is not a leak of beta's candidates.
    assert [a["isolation_violated"] for a in r["arbitrations"]] == [False, False]


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
    """Real Linux coreutils (cp -a, md5sum, diff, timeout, GNU grep/sed) in a
    Terminal-Bench image: candidates edit private copies of the git repo, one
    through the real WORKDIR path (redirected to its copy, the path written
    into a file is mapped back), one leaking into the real WORKDIR (restored);
    only the winner's change reaches the WORKDIR. Runs only with RUN_DOCKER=1."""
    import subprocess

    from subagents_bench.environment import LocalDockerEnvironment
    from subagents_bench.runner import run_task

    have = subprocess.run(["docker", "image", "inspect", TB_IMAGE], capture_output=True).returncode == 0
    if not have:
        pytest.skip(f"{TB_IMAGE} not pulled")
    cfg = arb_config(tmp_path)
    cfg.sandbox.kind, cfg.sandbox.image, cfg.sandbox.workdir = "local_docker", TB_IMAGE, "auto"
    wd = "/app/personal-site"

    def cand(system, messages, tools, client):
        if not tools:
            return Reply("revised")
        k = re.search(r"cand(\d)", system).group(1)
        if len(messages) == 1:
            if k == "0":  # names the real workdir: lands in its copy
                return Reply("", [tc("bash", command=f"printf 'from cand0 in {wd}\\n' > {wd}/arbiter_note.txt; cd {wd} && pwd")])
            calls = [tc("edit_file", path="arbiter_note.txt", old="", new=f"from cand{k}\n"),
                     tc("bash", command="git status --short | head -5; pwd")]
            if k == "2":  # a path assembled at run time escapes the remap
                calls.append(tc("bash", command="w=/app/personal'-site'; echo leak > $w/LEAK.txt"))
            return Reply("", calls)
        return Reply("", [tc("submit", report=f"wrote note {k}")])

    f = FakeFactory({ORCH: orchestrator([{"winner": 1, "confidence": 0.9, "agreement": True, "rationale": "-"}]),
                     SUB: cand})
    with LocalDockerEnvironment(TB_IMAGE, workdir=None, platform="linux/amd64") as env:
        res = run_task(cfg, "fix-git", "write a note", env, client_factory=f)
        assert res.record["workdir"] == wd
        note = env.exec(f"cat {wd}/arbiter_note.txt; git -C {wd} status --short; ls {wd}/LEAK.txt", timeout_sec=60).stdout
    r = assert_valid(res.episode_dir)
    ep, a = r["episode"], r["arbitrations"][0]
    assert ep["container_arch"] == "x86_64" and ep["has_timeout_binary"] is True
    assert a["isolation"] == "copy" and a["applied"] and f"{wd}/arbiter_note.txt" in a["applied_files"]
    # The winner's `git status` refreshed its index's stat cache; that file is applied too.
    assert set(a["applied_files"]) <= {f"{wd}/arbiter_note.txt", f"{wd}/.git/index"}
    assert note.startswith("from cand1\n") and "?? arbiter_note.txt" in note and "LEAK.txt" not in note
    assert (a["isolation_violated"], a["leaked_files"], a["n_path_remaps"]) == (True, [f"{wd}/LEAK.txt"], 1)
    assert all(d["workdir"].startswith("/tmp/.sb/arb/") for d in r["dispatches"])
    judge_prompt = f[ORCH].requests[1]["messages"][0].content
    # cand0's copy path, written into its file by the remapped command, reads as the workdir again.
    assert f"+from cand0 in {wd}\n" in judge_prompt and "+from cand2" in judge_prompt and "LEAK" not in judge_prompt
