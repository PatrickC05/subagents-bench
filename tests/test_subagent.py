"""Subagents end to end (spawned by a scripted orchestrator on LocalShell):
scope tool sets, statuses and reports, the context guard, the
ContextLengthError trim-and-retry, budgets and the DispatchRecord contents."""

from __future__ import annotations

import time

import pytest

from fakes import ORCH, SUB, FakeFactory, Reply, assert_valid, make_config, run, tc
from subagents_bench.prompts import LAST_STEP_NOTE, LENGTH_CUTOFF_NOTE
from subagents_bench.subagent import ELIDED_PREFIX
from subagents_bench.types import ContextLengthError, LLMError


def one_spawn(instructions=("do the thing",)):
    return [Reply("", [tc("spawn_subagents", instructions=list(instructions))]), Reply("", [tc("finish")])]


def test_easy_scope_cannot_call_bash(tmp_path):
    cfg = make_config(tmp_path, subagent={"scope": "easy"})
    f = FakeFactory({ORCH: one_spawn(), SUB: [
        Reply("", [tc("bash", command="touch hacked.txt")]),
        Reply("", [tc("submit", report="could not run bash")]),
    ]})
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    assert not (tmp_path / "work" / "hacked.txt").exists()
    assert f[SUB].requests[0]["tools"] == ["read_file", "list_dir", "search", "submit"]
    bash = [t for t in r["tools"] if t["name"] == "bash"][0]
    assert bash["is_error"] and "not available to you (scope easy)" in bash["error"] and bash["n_exec"] == 0
    second = f[SUB].requests[1]["messages"][-1].tool_results[0]
    assert second.is_error and "available: read_file, list_dir, search, submit" in second.content
    d = r["dispatches"][0]
    assert d["tools"] == ["read_file", "list_dir", "search", "submit"]
    assert (d["status"], d["report"], d["n_tool_calls"], d["n_tool_errors"], d["tool_counts"]) == (
        "submitted", "could not run bash", 1, 1, {"bash": 1},
    )


def test_hard_scope_edits_and_dispatch_record(tmp_path):
    cfg = make_config(tmp_path)
    f = FakeFactory({ORCH: one_spawn(), SUB: [
        Reply("", [tc("bash", command="mkdir -p src && cd src"), tc("edit_file", path="m.py", old="", new="x = 1\n")],
              prompt_tokens=100, completion_tokens=10, reasoning_tokens=3),
        Reply("", [tc("submit", report="created src/m.py")], prompt_tokens=150, completion_tokens=5),
    ]})
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    work = tmp_path / "work"
    assert (work / "src" / "m.py").read_text() == "x = 1\n"  # relative to the persisted cwd
    d = r["dispatches"][0]
    assert d["status"] == "submitted" and d["steps_used"] == 2 and d["n_llm_calls"] == 2
    assert (d["prompt_tokens"], d["completion_tokens"], d["reasoning_tokens"]) == (250, 15, 3)
    assert d["tool_counts"] == {"bash": 1, "edit_file": 1} and d["n_tool_calls"] == 2
    assert d["files_modified"] == [str(work / "src" / "m.py")] and d["n_files_modified"] == 1
    assert d["isolation"] == "shared" and d["workdir"] == str(work) and d["arbitration_id"] is None
    assert d["llm_sec"] == pytest.approx(sum(c["latency_sec"] for c in r["calls"] if c["role"] == "subagent"))
    assert r["spawns"][0]["files_modified"] == [str(work / "src" / "m.py")]
    sub_tools = [t for t in r["tools"] if t["role"] == "subagent"]
    assert d["tool_sec"] == pytest.approx(sum(t["duration_sec"] for t in sub_tools))
    assert all(t["dispatch_id"] == d["dispatch_id"] and t["group_id"] == d["group_id"] for t in sub_tools)


def test_no_tool_call_uses_text_as_report(tmp_path):
    cfg = make_config(tmp_path)
    f = FakeFactory({ORCH: one_spawn(), SUB: [Reply("The answer is 42.")]})
    res = run(cfg, f)
    d = assert_valid(res.episode_dir)["dispatches"][0]
    assert (d["status"], d["report"]) == ("no_tool_call", "The answer is 42.")
    assert "The answer is 42." in f[ORCH].requests[1]["messages"][-1].tool_results[0].content


def test_reply_cut_off_at_the_output_limit_does_not_end_the_dispatch(tmp_path):
    cfg = make_config(tmp_path, subagent={"max_steps": 3})
    f = FakeFactory({ORCH: one_spawn(), SUB: [
        Reply("I will now carefully", finish_reason="length"),
        Reply("", [tc("submit", report="brief report")]),
    ]})
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    d = r["dispatches"][0]
    assert (d["status"], d["report"], d["steps_used"]) == ("submitted", "brief report", 2)
    msgs = f[SUB].requests[1]["messages"]
    assert [m.role for m in msgs] == ["user", "assistant", "user"] and msgs[2].content == LENGTH_CUTOFF_NOTE
    assert msgs[1].content == "I will now carefully"  # visible text is kept as is
    assert [c["finish_reason"] for c in r["calls"] if c["role"] == "subagent"] == ["length", "tool_calls"]


def test_cut_off_before_the_last_step_also_warns_about_it(tmp_path):
    cfg = make_config(tmp_path, subagent={"max_steps": 2})
    f = FakeFactory({ORCH: one_spawn(), SUB: [Reply("cut", finish_reason="length"), Reply("cut again", finish_reason="length")]})
    res = run(cfg, f)
    d = assert_valid(res.episode_dir)["dispatches"][0]
    assert (d["status"], d["report"], d["steps_used"]) == ("max_steps", "cut again", 2)
    assert f[SUB].requests[1]["messages"][-1].content == f"{LENGTH_CUTOFF_NOTE}\n\n{LAST_STEP_NOTE}"


def test_max_steps_uses_last_text_and_warns_before_last_step(tmp_path):
    cfg = make_config(tmp_path, subagent={"max_steps": 2})
    f = FakeFactory({ORCH: one_spawn(), SUB: [
        Reply("still looking", [tc("list_dir")]),
        Reply("found nothing yet", [tc("list_dir")]),
    ]})
    res = run(cfg, f)
    d = assert_valid(res.episode_dir)["dispatches"][0]
    assert (d["status"], d["report"], d["steps_used"]) == ("max_steps", "found nothing yet", 2)
    assert f[SUB].requests[1]["messages"][-1].tool_results[-1].content.endswith(LAST_STEP_NOTE)


def test_tools_after_submit_are_not_run(tmp_path):
    cfg = make_config(tmp_path)
    f = FakeFactory({ORCH: one_spawn(), SUB: [Reply("", [tc("submit", report="r"), tc("bash", command="touch late.txt")])]})
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    assert not (tmp_path / "work" / "late.txt").exists()
    assert [t["name"] for t in r["tools"] if t["role"] == "subagent"] == ["submit"]


def test_context_guard_elides_old_tool_output(tmp_path):
    big = "\n".join(f"line {i:04d} " + "x" * 50 for i in range(110))  # ~6.6K chars
    cfg = make_config(tmp_path, subagent={"scope": "easy", "max_steps": 6,
                                          "model": {"model": SUB, "context_window": 4000, "max_output_tokens": 200}})
    (tmp_path / "work" / "big.txt").write_text(big)
    f = FakeFactory({ORCH: one_spawn(), SUB: [Reply("", [tc("read_file", path="big.txt")])] * 4 + [
        Reply("", [tc("submit", report="read it")])]})
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    d = r["dispatches"][0]
    assert d["status"] == "submitted" and d["context_trims"] >= 1
    calls = [c for c in r["calls"] if c["role"] == "subagent"]
    trimmed = [c["context_trimmed_chars"] for c in calls]
    assert trimmed[0] == trimmed[1] == 0 and sum(trimmed) > 6000
    first_trim = next(i for i, t in enumerate(trimmed) if t)
    results = [tr for m in f[SUB].requests[first_trim]["messages"] for tr in m.tool_results]
    assert results[0].content.startswith(ELIDED_PREFIX) and results[0].content.endswith("chars]")
    assert not results[-1].content.startswith(ELIDED_PREFIX) and not results[-2].content.startswith(ELIDED_PREFIX)
    assert r["episode"]["n_context_trims"] == d["context_trims"]
    # The transcript keeps the original output.
    sub_tool_msgs = [t for t in r["transcript"] if t["agent_id"] == d["agent_id"] and t["message_role"] == "tool"]
    assert sub_tool_msgs[0]["tool_results"][0]["content"].startswith("     1\tline 0000")


def test_context_length_error_trims_and_retries_once(tmp_path):
    cfg = make_config(tmp_path, subagent={"scope": "easy"})
    (tmp_path / "work" / "f.txt").write_text("hello\n" * 50)
    f = FakeFactory({ORCH: one_spawn(), SUB: [
        Reply("", [tc("read_file", path="f.txt")]),
        Reply("", [tc("read_file", path="f.txt")]),
        ContextLengthError("maximum context length exceeded", status=400),
        Reply("", [tc("submit", report="ok after trim")]),
    ]})
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    d = r["dispatches"][0]
    assert (d["status"], d["report"], d["context_trims"], d["steps_used"], d["n_llm_calls"]) == ("submitted", "ok after trim", 1, 3, 4)
    calls = [c for c in r["calls"] if c["role"] == "subagent"]
    assert calls[2]["error"].startswith("ContextLengthError") and calls[2]["step"] == calls[3]["step"] == 3
    assert calls[3]["context_trimmed_chars"] > 0
    retry = [tr for m in f[SUB].requests[3]["messages"] for tr in m.tool_results]
    assert retry[0].content.startswith(ELIDED_PREFIX) and "hello" in retry[1].content


def test_context_overflow_after_second_failure(tmp_path):
    cfg = make_config(tmp_path, subagent={"scope": "easy"})
    f = FakeFactory({ORCH: one_spawn(), SUB: [
        Reply("partial findings", [tc("list_dir")]),
        Reply("", [tc("list_dir")]),
        ContextLengthError("too long", status=400),
        ContextLengthError("too long", status=400),
    ]})
    res = run(cfg, f)
    d = assert_valid(res.episode_dir)["dispatches"][0]
    assert d["status"] == "context_overflow" and d["report"] == "partial findings"
    assert "ContextLengthError" in d["error"] and res.exit_reason == "finished"


def test_a_client_crash_fails_only_the_dispatch(tmp_path):
    cfg = make_config(tmp_path, subagent={"strategy": "sequential"})
    f = FakeFactory({ORCH: one_spawn(["a", "b"]), SUB: [RuntimeError("bug in the client"), Reply("", [tc("submit", report="ok")])]})
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    d1, d2 = r["dispatches"]
    assert d1["status"] == "error" and "RuntimeError: bug in the client" in d1["error"] and "Traceback" in d1["error"]
    assert d2["status"] == "submitted"
    # A harness error, not a provider failure: the episode carries on.
    assert r["episode"]["n_provider_errors"] == 0 and res.exit_reason == "finished"


def test_a_provider_failure_stops_the_episode_at_once(tmp_path):
    cfg = make_config(tmp_path, subagent={"strategy": "sequential"})
    f = FakeFactory({ORCH: one_spawn(["a", "b"]), SUB: [LLMError("server error after retries", status=503)]})
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    # The trial cannot measure the condition: no second dispatch, no second orchestrator step.
    assert res.exit_reason == "provider_error" and res.cause is not None and res.cause.status == 503
    assert r["episode"]["error"].startswith("LLMError: server error") and r["episode"]["n_provider_errors"] == 1
    d1, d2 = r["dispatches"]
    assert d1["status"] == "error" and d1["error"].startswith("LLMError: server error")
    assert d2["status"] == "cancelled" and d2["steps_used"] == 0
    assert r["episode"]["orchestrator_steps"] == 1 and len(f[ORCH].requests) == 1


def test_a_provider_failure_stops_parallel_siblings_promptly(tmp_path):
    cfg = make_config(tmp_path, subagent={"strategy": "parallel"})

    def sub(system, messages, tools, client):
        if "fail" in messages[0].content:
            return LLMError("giving up after 5 attempts", status=None)
        return Reply("", [tc("submit", report="slow")], latency=30.0)

    f = FakeFactory({ORCH: one_spawn(["fail now", "slow one", "slow two"]), SUB: sub})
    t0 = time.time()
    res = run(cfg, f)
    assert time.time() - t0 < 10  # the 30 s siblings were abandoned
    r = assert_valid(res.episode_dir)
    assert res.exit_reason == "provider_error"
    assert sorted(d["status"] for d in r["dispatches"]) == ["cancelled", "cancelled", "error"]


def test_metered_env_union_counts_overlapping_execs_once():
    import threading
    import time

    from subagents_bench.subagent import MeteredEnv

    class SlowEnv:
        def exec(self, command, timeout_sec=None):
            time.sleep(0.2)
            return command

    episode = MeteredEnv(SlowEnv())
    group = MeteredEnv(SlowEnv(), also=episode)
    threads = [threading.Thread(target=group.exec, args=("x",)) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    episode.exec("y")  # after the others: adds its own 0.2 s to both
    assert (group.n, episode.n) == (3, 4)
    assert group.sec >= 0.6 and 0.2 <= group.union_sec() < 0.35
    assert episode.sec >= 0.8 and 0.4 <= episode.union_sec() < 0.55
    assert MeteredEnv(SlowEnv()).union_sec() == 0.0


def test_per_subagent_token_budget(tmp_path):
    cfg = make_config(tmp_path, subagent={"token_budget": 100})
    f = FakeFactory({ORCH: one_spawn(), SUB: [Reply("", [tc("list_dir")], prompt_tokens=90, completion_tokens=20)]})
    res = run(cfg, f)
    d = assert_valid(res.episode_dir)["dispatches"][0]
    assert d["status"] == "token_budget" and d["steps_used"] == 1


def test_subagents_have_isolated_shell_state(tmp_path):
    cfg = make_config(tmp_path, subagent={"strategy": "sequential"})
    (tmp_path / "work" / "sub").mkdir()
    f = FakeFactory({
        ORCH: one_spawn(["first", "second"]),
        SUB: [Reply("", [tc("bash", command="cd sub && export FOO=1")]), Reply("", [tc("bash", command="pwd; echo ${FOO:-unset}")]),
              Reply("", [tc("submit", report="1")]),
              Reply("", [tc("bash", command="pwd; echo ${FOO:-unset}")]), Reply("", [tc("submit", report="2")])],
    })
    res = run(cfg, f)
    r = assert_valid(res.episode_dir)
    work = tmp_path / "work"
    sub_bash = [t for t in r["tools"] if t["role"] == "subagent" and t["name"] == "bash"]
    assert sub_bash[1]["cwd_before"] == str(work / "sub")  # persisted for sub-1
    assert sub_bash[2]["cwd_before"] == str(work)  # sub-2 starts fresh
    outputs = [m.tool_results[0].content for req in f[SUB].requests for m in req["messages"][-1:] if m.role == "tool"]
    assert any(o.startswith(f"{work / 'sub'}\n1") for o in outputs)
    assert any(o.startswith(f"{work}\nunset") for o in outputs)


@pytest.mark.parametrize("scope, scanned", [("easy", False), ("medium", True)])
def test_per_dispatch_scan_only_for_scopes_that_can_write(tmp_path, monkeypatch, scope, scanned):
    from subagents_bench import workspace

    markers: list[str] = []
    orig = workspace.make_marker
    monkeypatch.setattr(workspace, "make_marker", lambda env, name: markers.append(name) or orig(env, name))
    cfg = make_config(tmp_path, subagent={"scope": scope})
    f = FakeFactory({ORCH: one_spawn(), SUB: [Reply("", [tc("submit", report="r")])]})
    res = run(cfg, f)
    d = assert_valid(res.episode_dir)["dispatches"][0]
    assert any(m.startswith("dispatch-") for m in markers) is scanned
    assert any(m.startswith("group-") for m in markers) and any(m.startswith("episode-") for m in markers)
    assert (d["files_modified"], d["n_files_modified"]) == ([], 0)
