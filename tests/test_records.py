"""The episode recorder: thread safety, finalize semantics, recorded calls and tools."""

from __future__ import annotations

import threading
import time

import pytest

from fakes import ORCH, FakeClient, Reply, make_config, records, tc
from subagents_bench.environment import AgentStopped
from subagents_bench.llm import Message, ProviderError
from subagents_bench.records import EpisodeRecorder, Who
from subagents_bench.tools import ToolOutput

WHO = Who("orchestrator", "orchestrator")


def recorder(tmp_path, **kw) -> EpisodeRecorder:
    return EpisodeRecorder(tmp_path / "ep", make_config(**kw), task_name="fix-git", trial_name="fix-git__abc")


def test_static_fields_and_running_record(tmp_path):
    rec = recorder(tmp_path, topology="single")
    ep = records(rec.dir)["episode"]
    assert (ep["condition"], ep["topology"], ep["strategy"], ep["scope"], ep["subagent_model"]) == ("cond", "single", None, None, None)
    assert ep["exit_reason"] is None and ep["config"]["orchestrator"]["model"] == ORCH
    assert ep["task_name"] == "fix-git" and ep["trial_name"] == "fix-git__abc" and ep["git_commit"]


def test_concurrent_writes_and_ids(tmp_path):
    rec = recorder(tmp_path)
    ids: list[str] = []

    def work():
        for _ in range(50):
            ids.append(rec.new_id("call"))
            rec.write("tools", {"x": 1})

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(ids)) == 400 and len(records(rec.dir)["tools"]) == 400 == len(rec.records["tools"])


def test_finalize_wins_once_and_later_records_are_dropped(tmp_path):
    rec = recorder(tmp_path)
    rec.write("steps", {"step": 1})
    ep = rec.finalize("finished")
    assert ep["exit_reason"] == "finished" and ep["orchestrator_steps"] == 1
    assert rec.finalize("cancelled", "late")["exit_reason"] == "finished"
    rec.write("steps", {"step": 2})
    assert rec.dropped == 1 and len(records(rec.dir)["steps"]) == 1
    rec.update(cancel_kill={"ok": True})
    assert records(rec.dir)["episode"]["cancel_kill"] == {"ok": True}


def test_llm_records_success_failure_and_fails_fast(tmp_path):
    rec = recorder(tmp_path)
    client = FakeClient(make_config().orchestrator, [Reply("hi", prompt_tokens=7, cached_tokens=3, completion_tokens=2),
                                                     ProviderError("down", status=503, n_attempts=5, backoff_sec=9.5)])
    r, call_id = rec.llm(WHO, client, "s", [Message("user", "x")], [], step=1)
    with pytest.raises(ProviderError):
        rec.llm(WHO, client, "s", [], [], step=2)
    ok, failed = records(rec.dir)["calls"]
    assert (ok["call_id"], ok["prompt_tokens"], ok["cached_tokens"], ok["completion_tokens"], ok["error"]) == (call_id, 7, 3, 2, None)
    assert ok["t_start"] <= ok["t_first_chunk"] <= ok["t_end"] and ok["finish_reason"] == "stop"
    assert (failed["n_attempts"], failed["backoff_sec"], failed["error"]) == (5, 9.5, "ProviderError: down")
    assert rec.stop_event.is_set() and isinstance(rec.failure, ProviderError)
    ep = rec.finalize("provider_error")
    assert ep["n_provider_errors"] == 1 and ep["tokens_by_role"]["orchestrator"]["prompt"] == 7


def test_llm_call_is_abandoned_when_stopped(tmp_path):
    rec = recorder(tmp_path)
    client = FakeClient(make_config().orchestrator, [Reply("slow", latency=30)])
    threading.Timer(0.3, rec.stop_event.set).start()
    t0 = time.time()
    with pytest.raises(AgentStopped):
        rec.llm(WHO, client, "s", [], [], step=1)
    assert time.time() - t0 < 2 and records(rec.dir)["calls"][0]["error"] == "cancelled"


def test_tool_exceptions_are_recorded_then_raised(tmp_path):
    rec = recorder(tmp_path)
    out = rec.tool(WHO, 1, "call-1", tc("bash"), lambda: ToolOutput("ok", exit_code=0, output_chars=5000))

    def stopped():
        raise AgentStopped("stop")

    with pytest.raises(AgentStopped):
        rec.tool(WHO, 1, "call-1", tc("bash"), stopped)
    first, second = records(rec.dir)["tools"]
    assert out.text == "ok" and (first["output_chars"], first["is_error"]) == (5000, False)
    assert second["is_error"] and second["name"] == "bash" and second["t_end"] >= second["t_start"]


def test_tool_listeners_see_each_written_record_and_never_change_the_call(tmp_path):
    """A listener runs in the tool's thread after its record is written, also when the tool raised;
    a listener that raises is ignored (the result and the exception are the tool's own); a removed
    listener sees nothing more."""
    rec = recorder(tmp_path)
    seen: list[tuple[str, int, str]] = []

    def listener(record):  # the record is already written when the listener runs
        seen.append((record["name"], len(rec.records["tools"]), threading.current_thread().name))

    def broken(record):
        raise RuntimeError("listener bug")

    rec.add_tool_listener(broken)
    rec.add_tool_listener(listener)
    out = rec.tool(WHO, 1, "call-1", tc("bash", command="ls"), lambda: ToolOutput("ok", exit_code=0))
    assert out.text == "ok" and not out.is_error

    def stopped():
        raise AgentStopped("stop")

    with pytest.raises(AgentStopped):
        rec.tool(WHO, 1, "call-1", tc("edit_file"), stopped)
    me = threading.current_thread().name
    assert seen == [("bash", 1, me), ("edit_file", 2, me)]
    rec.remove_tool_listener(listener)
    rec.remove_tool_listener(listener)  # twice: no harm
    rec.tool(WHO, 2, "call-2", tc("read_file"), lambda: ToolOutput("x"))
    assert len(seen) == 2 and [t["name"] for t in records(rec.dir)["tools"]] == ["bash", "edit_file", "read_file"]


def test_snapshots_is_a_record_kind(tmp_path):
    rec = recorder(tmp_path)
    assert (rec.dir / "snapshots.jsonl").read_text() == "" and rec.records["snapshots"] == []
    rec.write("snapshots", {"seq": 0, "changed": True})
    assert records(rec.dir)["snapshots"][0]["seq"] == 0 and rec.records["snapshots"][0]["changed"] is True
