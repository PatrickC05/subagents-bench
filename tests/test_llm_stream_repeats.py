"""Streamed tool calls that a provider re-sends under another index.

Sail (2026-10-11, nvidia/Gemma-4-31B-IT-NVFP4, balanced tier) streams each
finished tool call twice: first under ``index`` 3, then again in full under
``index`` 0 with the same ``id``. Keyed by index alone that became two
identical calls with one id, so every command ran twice and the duplicated
id in the history later made Sail fail the request ("chat completion
failed"). These chunks are the captured stream, with the long command
shortened.
"""

from __future__ import annotations

import json

import pytest

from subagents_bench.llm import _Accumulator

CMD = "cd /tmp/.sb/arb/a0001/cand1 && \\\ngit status && git reflog --all | head -50"
ARGS = json.dumps({"command": CMD})
CALL_ID = "call_resp_01a12e3f-d477-749a-93fc-7067c70944b1_3"


def _chunk(delta: dict, finish: str | None = None) -> dict:
    return {"id": "resp_x", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def _feed(chunks: list[dict]) -> _Accumulator:
    acc = _Accumulator(t_sent=0.0)
    for k, c in enumerate(chunks):
        acc.feed_chunk(c, now=float(k))
    return acc


def _call(index: int, args: str = ARGS, call_id: str | None = CALL_ID, name: str | None = "bash") -> dict:
    tc: dict = {"index": index, "type": "function", "function": {}}
    if call_id:
        tc["id"] = call_id
    if name:
        tc["function"]["name"] = name
    if args is not None:
        tc["function"]["arguments"] = args
    return {"tool_calls": [tc]}


def test_sail_full_repeat_under_another_index_is_one_call():
    acc = _feed([
        _chunk({"role": "assistant"}),
        _chunk(_call(3)),
        _chunk(_call(0)),
        _chunk({}, "tool_calls"),
    ])
    calls = acc.tool_calls()
    assert len(calls) == 1
    assert calls[0].id == CALL_ID and calls[0].name == "bash" and calls[0].arguments == {"command": CMD}
    assert acc.tool_call_repeats_dropped == 1 and acc.stream_repairs == 1


def test_repeat_with_reformatted_json_is_still_a_repeat():
    acc = _feed([_chunk(_call(3)), _chunk(_call(0, args=json.dumps({"command": CMD}, indent=1)))])
    calls = acc.tool_calls()
    assert len(calls) == 1 and calls[0].arguments == {"command": CMD}


def test_two_distinct_calls_and_trailing_empty_deltas_are_untouched():
    # The shape Sail uses without the repeat (and the usual OpenAI shape).
    other = json.dumps({"path": "/app/README.md"})
    acc = _feed([
        _chunk(_call(0, call_id="call_a")),
        _chunk(_call(1, args=other, call_id="call_b", name="read_file")),
        _chunk({"tool_calls": [{"index": 0, "function": {}}, {"index": 1, "function": {}}]}),
        _chunk({}, "tool_calls"),
    ])
    calls = acc.tool_calls()
    assert [(c.id, c.name) for c in calls] == [("call_a", "bash"), ("call_b", "read_file")]
    assert acc.stream_repairs == 0


def test_fragmented_arguments_without_ids_still_concatenate():
    acc = _feed([
        _chunk(_call(0, args='{"command": "ec', call_id="call_a")),
        _chunk(_call(0, args='ho hi"}', call_id=None, name=None)),
    ])
    calls = acc.tool_calls()
    assert len(calls) == 1 and calls[0].arguments == {"command": "echo hi"}
    assert acc.stream_repairs == 0


def test_same_id_on_different_calls_gets_unique_ids():
    # Two genuinely different calls that share an id (no repeat): both kept,
    # the second re-identified so the history stays valid.
    acc = _feed([
        _chunk({"tool_calls": [{"index": 0, "id": "dup", "function": {"name": "bash", "arguments": '{"command": "a"}'}}]}),
        _chunk({"tool_calls": [{"index": 1, "id": "dup2", "function": {"name": "bash", "arguments": '{"command": "b"}'}}]}),
    ])
    acc.tools[1].id = "dup"  # simulate a provider that reuses one id
    calls = acc.tool_calls()
    assert [c.arguments["command"] for c in calls] == ["a", "b"]
    assert calls[0].id == "dup" and calls[1].id != "dup" and calls[1].id.startswith("dup_dup")
    assert acc.tool_call_ids_renamed == 1


@pytest.mark.live
def test_live_sail_gemma_subagent_turn_has_unique_tool_calls():
    """The real subagent prompt on Sail Gemma: every tool call id is unique
    and no identical call appears twice (the re-send is repaired)."""
    from pathlib import Path

    from subagents_bench.config import ModelConfig
    from subagents_bench.credentials import load_dotenv
    from subagents_bench.llm import make_client
    from subagents_bench.prompts import subagent_system
    from subagents_bench.subagent import SUBMIT_SPEC
    from subagents_bench.tools import SCOPE_TOOLS, specs_for
    from subagents_bench.types import Message

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    cfg = ModelConfig(
        model="nvidia/Gemma-4-31B-IT-NVFP4", provider="sail", completion_window="balanced",
        temperature=0.7, max_output_tokens=2048, max_retries=3,
    )
    system = subagent_system("hard", workdir="/app/personal-site")
    instruction = (
        "In the git repository /app/personal-site, report the current branch, the last 5 commits on all "
        "branches and the reflog. Run one bash command that prints all three."
    )
    specs = specs_for(SCOPE_TOOLS["hard"], command_timeout_sec=180, max_command_timeout_sec=900) + [SUBMIT_SPEC]
    r = make_client(cfg).complete(system, [Message(role="user", content=instruction)], specs)
    assert r.tool_calls, (r.finish_reason, r.text[:300])
    ids = [tc.id for tc in r.tool_calls]
    assert len(ids) == len(set(ids)), ids
    sigs = [(tc.name, json.dumps(tc.arguments, sort_keys=True)) for tc in r.tool_calls]
    assert len(sigs) == len(set(sigs)), sigs
