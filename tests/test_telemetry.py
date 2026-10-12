"""EpisodeRecorder and AgentLog: files, counters, record contents, aggregates,
finalization, stop-aware LLM calls and schema validity."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import threading
import time
from collections import Counter

import pytest

from fakes import FakeClient, Reply, tc
from subagents_bench import provenance as prov_mod
from subagents_bench.config import ExperimentConfig, ModelConfig
from subagents_bench.schema import RECORD_TYPES, SCHEMA_VERSION, DispatchRecord, EpisodeRecord, StepRecord, validate_record
from subagents_bench.telemetry import (
    AgentLog,
    EpisodeRecorder,
    Telemetry,
    ToolEvent,
    is_provider_error,
    provenance,
    read_jsonl,
    validate_episode_dir,
)
from subagents_bench.types import (
    AgentStopped,
    Attempt,
    BillingError,
    ContextLengthError,
    LLMError,
    Message,
    ToolOutput,
    ToolSpec,
)

SPEC = ToolSpec(name="bash", description="run", parameters={"type": "object", "properties": {}})


def make_recorder(tmp_path, **kw):
    cfg = ExperimentConfig.from_dict({"name": "cond", "topology": "subagents", "subagent": {"strategy": "arbiter"}})
    return EpisodeRecorder(tmp_path / "ep", config=cfg, task_name="fix-git", instruction="fix it", workdir="/app", **kw)


def test_start_writes_all_files_and_running_record(tmp_path):
    rec = make_recorder(tmp_path, container_arch="x86_64", has_timeout_binary=True)
    for fname in RECORD_TYPES:
        assert (rec.dir / fname).exists(), fname
    ep = json.loads((rec.dir / "episode.json").read_text())
    assert ep["status"] == "running" and ep["schema_version"] == SCHEMA_VERSION
    assert ep["strategy"] == "arbiter" and ep["scope"] == "hard" and ep["topology"] == "subagents"
    assert ep["arbiter_model"] == ep["orchestrator_model"] == "claude-opus-4-8"
    assert ep["task_name"] == "fix-git" and ep["workdir"] == "/app" and ep["container_arch"] == "x86_64"
    assert ep["config_hash"] == rec.config.config_hash()
    assert ep["exit_reason"] is None and ep["t_end"] is None
    assert validate_episode_dir(rec.dir) == []


def test_single_topology_has_no_strategy_fields(tmp_path):
    cfg = ExperimentConfig.from_dict({"name": "s", "topology": "single"})
    rec = EpisodeRecorder(tmp_path / "e", config=cfg, task_name="t", instruction="i")
    assert (rec.record["strategy"], rec.record["scope"], rec.record["subagent_model"], rec.record["arbiter_model"]) == (
        None, None, None, None,
    )


def test_counters_are_unique_and_contiguous_across_threads(tmp_path):
    rec = make_recorder(tmp_path)
    got: list[tuple[str, int, int]] = []
    lock = threading.Lock()

    def worker(agent):
        for _ in range(200):
            s, a = rec.alloc(agent)
            with lock:
                got.append((agent, s, a))

    threads = [threading.Thread(target=worker, args=(f"sub-{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    seqs = sorted(s for _, s, _ in got)
    assert seqs == list(range(800))
    for i in range(4):
        aseqs = sorted(a for ag, _, a in got if ag == f"sub-{i}")
        assert aseqs == list(range(200))
    ids = [rec.new_id("call") for _ in range(3)]
    assert ids == ["c0001", "c0002", "c0003"]
    assert rec.new_id("group") == "g0001" and rec.next_number("sub") == 1 and rec.next_number("sub") == 2


def test_llm_call_record_fields(tmp_path):
    rec = make_recorder(tmp_path)
    cfg = ModelConfig(model="m", provider="sail", price_in=1.0, price_out=2.0, price_cache_read=0.5, completion_window="balanced")
    client = FakeClient(
        cfg,
        [Reply("hi", [tc("bash", command="ls")], prompt_tokens=1000, completion_tokens=50, cached_tokens=400,
               reasoning_tokens=20, reasoning="think", failed_attempts=2, backoff_sec=1.5, latency=0.05)],
    )
    log = AgentLog(rec, "sub-1", "subagent", dispatch_id="d0001", group_id="g0001")
    resp, call_id = log.llm(client, "sys", [Message(role="user", content="go")], [SPEC], step=3, context_trimmed_chars=77)
    assert call_id == "c0001" and log.last_call_id == "c0001"
    c = read_jsonl(rec.dir / "calls.jsonl")[0]
    assert (c["role"], c["agent_id"], c["dispatch_id"], c["group_id"], c["step"], c["seq"], c["agent_seq"]) == (
        "subagent", "sub-1", "d0001", "g0001", 3, 0, 0,
    )
    assert (c["prompt_tokens"], c["cached_tokens"], c["uncached_prompt_tokens"], c["completion_tokens"]) == (1000, 400, 600, 50)
    assert c["reasoning_tokens"] == 20 and c["reasoning_chars"] == 5 and c["content_chars"] == 2
    assert c["n_attempts"] == 3 and c["backoff_sec"] == pytest.approx(3.0) and len(c["attempts"]) == 3
    assert c["attempts"][0]["status"] == 429 and c["attempts"][-1]["status"] == 200
    assert c["context_trimmed_chars"] == 77
    assert c["latency_sec"] == pytest.approx(c["t_end"] - c["t_start"])
    assert c["ttft_sec"] == pytest.approx(c["t_first_chunk"] - c["t_request_sent"])
    assert c["decode_sec"] == pytest.approx(c["t_last_chunk"] - c["t_first_chunk"])
    assert c["latency_sec"] >= 0.05
    assert c["tool_call_names"] == ["bash"] and c["n_tool_calls"] == 1
    assert c["cost_usd"] == pytest.approx((600 * 1.0 + 400 * 0.5 + 50 * 2.0) / 1e6)
    assert (c["price_in"], c["price_out"], c["price_cache_read"]) == (1.0, 2.0, 0.5)
    assert c["completion_window"] == "balanced" and c["n_messages"] == 1 and c["n_tools"] == 1
    assert len(c["msg_hashes"]) == 1 and len(c["system_hash"]) == 12
    assert c["error"] is None and c["usage_source"] == "provider"
    assert validate_episode_dir(rec.dir) == []


def test_failed_call_is_recorded_then_reraised(tmp_path):
    rec = make_recorder(tmp_path)
    att = [Attempt(attempt=1, t_start=1.0, t_end=2.0, status=402, error_type="Billing", error="no balance")]
    client = FakeClient(ModelConfig(model="m", provider="sail"), [BillingError("no balance", attempts=att, status=402)])
    log = AgentLog(rec, "orchestrator", "orchestrator")
    with pytest.raises(BillingError):
        log.llm(client, "sys", [Message(role="user", content="go")], [SPEC], step=1)
    c = read_jsonl(rec.dir / "calls.jsonl")[0]
    assert c["error"].startswith("BillingError: no balance")
    # Unpriced model config: the cost of the (zero-usage) failed call is unknown, not 0.
    assert (c["prompt_tokens"], c["completion_tokens"], c["cost_usd"], c["usage_source"]) == (0, 0, None, "none")
    assert (c["price_in"], c["price_out"]) == (None, None)
    assert c["n_attempts"] == 1 and c["attempts"][0]["status"] == 402 and c["finish_reason"] == "error"
    assert c["n_messages"] == 1 and c["n_tools"] == 1  # request info still describes what was sent
    assert validate_episode_dir(rec.dir) == []


def test_llm_call_is_abandoned_when_stopped(tmp_path):
    rec = make_recorder(tmp_path)
    stop = threading.Event()
    client = FakeClient(ModelConfig(model="m"), [Reply("late", latency=5.0)])
    log = AgentLog(rec, "orchestrator", "orchestrator", stop_event=stop)
    threading.Timer(0.2, stop.set).start()
    t0 = time.time()
    with pytest.raises(AgentStopped):
        log.llm(client, "s", [Message(role="user", content="x")], [], step=1)
    assert time.time() - t0 < 2.0
    c = read_jsonl(rec.dir / "calls.jsonl")[0]
    assert c["error"].startswith("cancelled") and c["prompt_tokens"] == 0 and c["n_attempts"] == 0


def test_tool_records_measurements_and_reraises(tmp_path):
    rec = make_recorder(tmp_path)
    log = AgentLog(rec, "sub-1", "subagent", dispatch_id="d0001", group_id="g0001", arbitration_id="a0001")
    out = ToolOutput(text="hello", exit_code=0, timeout_sec=180, output_chars_raw=9, truncated=True,
                     cwd_before="/app", cwd_after="/tmp", n_exec=2, exec_sec=0.25)
    ev = ToolEvent()
    got = log.tool(tc("bash", command="x" * 5000), lambda e: out, step=2, call_id="c0009", event=ev)
    assert got is out and ev.group_id == "g0001"
    with pytest.raises(AgentStopped):
        log.tool(tc("bash", command="y"), lambda e: (_ for _ in ()).throw(AgentStopped("x")), step=2, call_id="c0009")
    with pytest.raises(RuntimeError):
        log.tool(tc("bash", command="z"), lambda e: (_ for _ in ()).throw(RuntimeError("bug")), step=2, call_id="c0009")
    t0, t1, t2 = read_jsonl(rec.dir / "tools.jsonl")
    assert (t0["exit_code"], t0["timeout_sec"], t0["output_chars_raw"], t0["output_chars_returned"]) == (0, 180, 9, 5)
    assert (t0["cwd_before"], t0["cwd_after"], t0["n_exec"], t0["exec_sec"], t0["truncated"]) == ("/app", "/tmp", 2, 0.25, True)
    assert t0["arbitration_id"] == "a0001" and t0["call_id"] == "c0009" and t0["step"] == 2
    assert t0["args"]["command"].endswith("[truncated: 5000 chars total]") and len(t0["args"]["command"]) < 4100
    assert t1["is_error"] and t1["error"] == "cancelled"
    assert t2["is_error"] and "RuntimeError: bug" in t2["error"]
    assert [t["agent_seq"] for t in (t0, t1, t2)] == [0, 1, 2]
    assert validate_episode_dir(rec.dir) == []


def _dispatch(rec, did, group, t0, trims=0):
    rec.record_dispatch(DispatchRecord(
        episode_id=rec.episode_id, dispatch_id=did, agent_id="sub-1", group_id=group, arbitration_id=None,
        candidate_index=None, instruction_index=0, orchestrator_step=1, strategy="parallel", scope="hard", model="m",
        tools=["submit"], instruction="i", report="r", status="submitted", steps_used=1, max_steps=8, t_queued=t0,
        t_start=t0, t_end=t0 + 1, duration_sec=1.0, queue_sec=0.0, n_llm_calls=0, prompt_tokens=0, completion_tokens=0,
        cached_tokens=0, reasoning_tokens=None, cost_usd=0.0, llm_sec=0.0, tool_sec=0.0, n_tool_calls=0, tool_counts={},
        n_tool_errors=0, n_timeouts=0, context_trims=trims, workdir="/app", isolation="shared", files_modified=None,
        n_files_modified=None, error=None,
    ))


def test_finalize_aggregates_equal_record_sums_and_is_idempotent(tmp_path):
    rec = make_recorder(tmp_path)
    cfg_a = ModelConfig(model="big", price_in=1.0, price_out=1.0)
    cfg_b = ModelConfig(model="small", price_in=0.5, price_out=0.5)
    a = FakeClient(cfg_a, [Reply("x", prompt_tokens=100, completion_tokens=10, reasoning_tokens=4, failed_attempts=1, backoff_sec=2.0)] * 2)
    b = FakeClient(cfg_b, [Reply("y", prompt_tokens=50, completion_tokens=5, cached_tokens=20)] * 4)
    orch = AgentLog(rec, "orchestrator", "orchestrator")
    sub = AgentLog(rec, "sub-1", "subagent", dispatch_id="d0001", group_id="g0001")
    msgs = [Message(role="user", content="q")]
    for _ in range(2):
        orch.llm(a, "s", msgs, [], step=1)
    for _ in range(3):
        sub.llm(b, "s", msgs, [], step=1)
    sub.tool(tc("bash"), lambda e: ToolOutput(text="t", timed_out=True), step=1, call_id="c0003")
    _dispatch(rec, "d0001", "g0001", time.time(), trims=2)
    rec.record_step(StepRecord(episode_id=rec.episode_id, step=1, call_id="c0001", t_start=0.0, t_end=1.0, duration_sec=1.0,
                               llm_sec=1.0, tools_sec=0.0, n_tool_calls=0, tool_names=[], spawned_group_ids=[], finished=False))
    final = rec.finalize("finished", final_summary="ok", files_modified=["/app/x"], n_files_modified=1, git_diff_stat="")
    assert final["status"] == "done" and final["exit_reason"] == "finished"
    assert final["n_calls"] == {"orchestrator": 2, "subagent": 3, "arbiter": 0, "debate": 0}
    assert final["tokens"]["orchestrator"] == {"prompt": 200, "completion": 20, "cached": 0, "uncached_prompt": 200, "reasoning": 8}
    assert final["tokens"]["subagent"] == {"prompt": 150, "completion": 15, "cached": 60, "uncached_prompt": 90, "reasoning": None}
    calls = read_jsonl(rec.dir / "calls.jsonl")
    assert final["cost_usd"]["total"] == pytest.approx(sum(c["cost_usd"] for c in calls))
    assert final["llm_sec"]["subagent"] == pytest.approx(sum(c["latency_sec"] for c in calls if c["role"] == "subagent"))
    assert final["n_retries"] == 2 and final["backoff_sec"] == pytest.approx(4.0)
    assert final["n_command_timeouts"] == 1 and final["n_context_trims"] == 2
    assert final["n_tool_calls"]["subagent"] == 1 and final["orchestrator_steps"] == 1 and final["dispatches"] == 1
    assert final["t_first_call"] == min(c["t_start"] for c in calls)
    assert final["wall_clock_sec"] == pytest.approx(final["t_end"] - final["t_start"])
    assert rec.total_tokens() == 220 + 165
    # Idempotent: a second finalize changes nothing; late records are dropped.
    again = rec.finalize("cancelled")
    assert again["exit_reason"] == "finished"
    sub.llm(b, "s", msgs, [], step=2)
    assert rec.dropped == 1 and len(read_jsonl(rec.dir / "calls.jsonl")) == 5
    assert validate_episode_dir(rec.dir) == []
    totals = rec.role_totals()
    assert totals["models"]["big"] == {"n_input_tokens": 200, "n_cache_tokens": 0, "n_output_tokens": 20,
                                       "cost_usd": pytest.approx(totals["roles"]["orchestrator"]["cost_usd"])}
    assert totals["total"]["n_input_tokens"] == 350 and totals["total"]["n_cache_tokens"] == 60
    s = rec.summary()
    assert s["exit_reason"] == "finished" and s["dispatches"] == 1 and s["n_calls"]["subagent"] == 3
    assert final["n_provider_errors"] == 0 and final["cancel_kill"] is None
    assert (final["harness_exec_n"], final["harness_exec_sec"], final["harness_exec_sum_sec"]) == (None, None, None)


def test_redispatches_and_instruction_following(tmp_path):
    from subagents_bench.schema import SpawnRecord

    rec = make_recorder(tmp_path)
    t = time.time()
    for gi, (req, gr) in enumerate([(2, 2), (4, 1)]):
        rec.record_spawn(SpawnRecord(
            episode_id=rec.episode_id, group_id=f"g{gi}", orchestrator_step=gi + 1, call_id="c", tool_call_id="x",
            strategy="parallel", scope="hard", requested=req, granted=gr, cap_reason=None, dispatches_remaining_after=0,
            instructions=["i"] * req, dispatch_ids=[], arbitration_ids=[], concurrency=gr, t_start=t + gi, t_end=t + gi + 1,
            duration_sec=1.0, sum_dispatch_sec=0.0, files_modified=None, n_files_modified=None,
        ))
    for k, g in enumerate(["g0", "g0", "g1"]):
        _dispatch(rec, f"d{k}", g, t)
    final = rec.finalize("max_steps")
    assert final["redispatches"] == 1
    assert final["instruction_following"] == pytest.approx((1.0 + 0.25) / 2)
    assert (final["requested_total"], final["granted_total"], final["spawn_calls"]) == (6, 3, 2)


def test_update_static_and_set_outcome(tmp_path):
    rec = make_recorder(tmp_path)
    rec.update_static(workdir="/srv", has_timeout_binary=False)
    assert json.loads((rec.dir / "episode.json").read_text())["workdir"] == "/srv"
    with pytest.raises(KeyError):
        rec.update_static(exit_reason="x")
    rec.finalize("finished")
    rec.set_outcome(1.0, True)
    ep = json.loads((rec.dir / "episode.json").read_text())
    assert (ep["reward"], ep["resolved"], ep["status"]) == (1.0, True, "done")


def test_transcript_indexes_per_agent(tmp_path):
    rec = make_recorder(tmp_path)
    a = AgentLog(rec, "orchestrator", "orchestrator")
    b = AgentLog(rec, "sub-1", "subagent", dispatch_id="d0001")
    a.transcript(Message(role="system", content="sys"))
    b.transcript(Message(role="user", content="u"))
    a.transcript(Message(role="assistant", content="", tool_calls=[tc("bash", command="ls")], reasoning="r"), call_id="c0001")
    rows = read_jsonl(rec.dir / "transcript.jsonl")
    assert [(r["agent_id"], r["index"]) for r in rows] == [("orchestrator", 0), ("sub-1", 0), ("orchestrator", 1)]
    assert rows[2]["tool_calls"][0]["name"] == "bash" and rows[2]["reasoning"] == "r" and rows[2]["call_id"] == "c0001"
    assert validate_episode_dir(rec.dir) == []


def test_validate_episode_dir_reports_problems(tmp_path):
    rec = make_recorder(tmp_path)
    with open(rec.dir / "calls.jsonl", "a") as f:
        f.write(json.dumps({"episode_id": "x"}) + "\n")
        f.write("{not json\n")
    (rec.dir / "spawns.jsonl").unlink()
    problems = validate_episode_dir(rec.dir)
    assert any(p.startswith("calls.jsonl:1: missing field 'call_id'") for p in problems)
    assert any("invalid JSON" in p for p in problems)
    assert "spawns.jsonl: missing" in problems


def _in_git_worktree() -> bool:
    """Whether the code under test is a git checkout (not a ZIP/tarball copy)."""
    try:
        res = subprocess.run(["git", "-C", str(prov_mod.repo_root()), "rev-parse", "HEAD"], capture_output=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return False
    return res.returncode == 0


def test_provenance_and_alias():
    p = provenance()
    assert set(p) == {"harness_version", "git_commit", "git_dirty", "source_sha256", "harbor_version",
                      "python_version", "host_platform", "host_machine"}
    # The fingerprint exists with or without git, and is the source tree's.
    assert isinstance(p["source_sha256"], str) and len(p["source_sha256"]) == 64
    assert p["source_sha256"] == prov_mod.source_fingerprint(prov_mod.repo_root())
    if _in_git_worktree():
        assert isinstance(p["git_commit"], str) and len(p["git_commit"]) == 40
        assert isinstance(p["git_dirty"], bool)
    else:  # e.g. a ZIP or tarball copy of the repo
        assert p["git_commit"] is None and p["git_dirty"] is None
    assert Telemetry is EpisodeRecorder


def test_episode_records_source_fingerprint(tmp_path):
    rec = make_recorder(tmp_path, provenance_fields={"harness_version": "x", "source_sha256": "ab" * 32})
    assert rec.record["source_sha256"] == "ab" * 32 and rec.record["git_commit"] is None
    assert rec.finalize("finished")["source_sha256"] == "ab" * 32
    assert make_recorder(tmp_path / "2").record["source_sha256"] == provenance()["source_sha256"]


def test_provider_errors_counted_from_failed_calls(tmp_path):
    rec = make_recorder(tmp_path)
    att = [Attempt(attempt=1, t_start=1.0, t_end=2.0, status=503, error_type="ServerError", error="down")]
    client = FakeClient(ModelConfig(model="m"), [
        LLMError("giving up after 9 attempts; last ServerError (HTTP 503)", attempts=att, status=503),
        LLMError("ServerError", status=401),
        BillingError("no balance", status=402),
        ContextLengthError("prompt is too long", status=400),
        Reply("fine"),
    ])
    orch = AgentLog(rec, "orchestrator", "orchestrator")
    sub = AgentLog(rec, "sub-1", "subagent", dispatch_id="d0001")
    for log in (orch, sub, orch, sub):
        with pytest.raises(LLMError):
            log.llm(client, "s", [Message(role="user", content="q")], [], step=1)
    orch.llm(client, "s", [Message(role="user", content="q")], [], step=2)
    stop = threading.Event()
    stop.set()
    with pytest.raises(AgentStopped):  # abandoned on stop: not a provider error
        AgentLog(rec, "orchestrator", "orchestrator", stop_event=stop).llm(client, "s", [], [], step=3)
    errors = [c["error"] for c in read_jsonl(rec.dir / "calls.jsonl")]
    assert [is_provider_error(e) for e in errors] == [True, True, False, False, False, False]
    assert rec._running_record().n_provider_errors == 0  # noqa: SLF001
    final = rec.finalize("provider_error", error=errors[0])
    # Calls of any role; billing, context-length and cancelled calls have their own outcomes.
    assert final["n_provider_errors"] == 2 and rec.summary()["n_provider_errors"] == 2
    assert validate_episode_dir(rec.dir) == []


def test_update_record_after_finalize(tmp_path):
    rec = make_recorder(tmp_path)
    rec.finalize("finished")
    kill = {"attempted": True, "ok": True, "n_killed": 2, "error": None}
    rec.update_record(cancel_kill=kill)
    ep = json.loads((rec.dir / "episode.json").read_text())
    assert ep["cancel_kill"] == kill and ep["exit_reason"] == "finished" and ep["status"] == "done"
    with pytest.raises(KeyError):
        rec.update_record(no_such_field=1)
    assert validate_episode_dir(rec.dir) == []


def test_validate_accepts_older_schema_without_new_fields(tmp_path):
    rec = make_recorder(tmp_path)
    ep = dict(rec.finalize("finished"))
    added = [f.name for f in dataclasses.fields(EpisodeRecord) if f.metadata.get("since") == 3]
    assert set(added) == {"source_sha256", "cancel_kill", "n_provider_errors", "harness_exec_sum_sec", "n_stream_repairs"}
    old = {k: v for k, v in ep.items() if k not in added}
    assert validate_record(EpisodeRecord, old) == [f"missing field {n!r}" for n in added]
    # episode.json written by schema-2 code (e.g. a sweep already running) lacks them legitimately.
    assert validate_record(EpisodeRecord, {**old, "schema_version": 2}) == []
    assert validate_record(EpisodeRecord, {**old, "schema_version": 2, "harness_version": 1}) != []


def test_concurrent_writes_produce_valid_lines(tmp_path):
    rec = make_recorder(tmp_path)
    client = FakeClient(ModelConfig(model="m"), [Reply("x")] * 80)

    def worker(i):
        log = AgentLog(rec, f"sub-{i}", "subagent", dispatch_id=f"d{i}")
        for _ in range(20):
            log.llm(client, "s", [Message(role="user", content="q")], [], step=1)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    calls = read_jsonl(rec.dir / "calls.jsonl")
    assert len(calls) == 80 and sorted(c["seq"] for c in calls) == list(range(80))
    assert Counter(c["agent_id"] for c in calls) == {f"sub-{i}": 20 for i in range(4)}
    assert validate_episode_dir(rec.dir) == []
