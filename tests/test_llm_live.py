"""Live provider checks for the LLM clients (opt-in: ``RUN_LIVE=1``).

For each model the study uses, run one tool-calling turn and one follow-up
turn that carries the tool result back, and check what the harness relies on:
the tool call is parsed into a dict, usage comes from the provider, and the
streaming instants are populated and monotonic. Costs a fraction of a cent.

    RUN_LIVE=1 .venv/bin/python -m pytest tests/test_llm_live.py -q -s

Credentials: ``SAIL_API_KEY`` from ``.env``; Prime from ``PRIME_API_KEY`` /
``PRIME_TEAM_ID`` or a logged-in ``prime`` CLI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subagents_bench.config import ModelConfig
from subagents_bench.credentials import load_dotenv
from subagents_bench.llm import LLMResponse, Message, ToolResult, ToolSpec, make_client
from subagents_bench.types import CHUNK_OFFSETS_CAP

pytestmark = pytest.mark.live

REPO = Path(__file__).resolve().parents[1]

MODELS = [
    ("sail", "deepseek-ai/DeepSeek-V4.1-Flash", "balanced"),
    ("sail", "nvidia/Gemma-4-31B-IT-NVFP4", "balanced"),
    ("sail", "google/gemma-4-12B-it", "balanced"),
    ("prime", "Qwen/Qwen3.5-2B", None),
]

ADD = ToolSpec(
    name="add",
    description="Add two integers and return their sum.",
    parameters={
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    },
)
SYSTEM = "You are a calculator assistant. Always use the add tool for additions; never add numbers yourself."


def _check(r: LLMResponse, label: str) -> None:
    t = r.timing
    ttft = f"{t.t_first_chunk - t.t_request_sent:.2f}s" if t.t_first_chunk and t.t_request_sent else None
    print(
        f"\n[{label}] finish={r.finish_reason} attempts={len(r.attempts)} chunks={t.n_chunks} ttft={ttft} "
        f"latency={t.t_end - t.t_start:.2f}s usage=({r.usage.prompt_tokens} in, {r.usage.cached_tokens} cached, "
        f"{r.usage.completion_tokens} out, reasoning={r.usage.reasoning_tokens}) "
        f"reasoning_chars={len(r.reasoning)} provider_cost={r.provider_reported_cost} "
        f"usage_keys={sorted((r.usage_raw or {}).keys())} header_keys={sorted(r.response_headers)[:12]}"
    )
    assert r.usage_source == "provider", f"{label}: provider sent no usage"
    assert r.usage.prompt_tokens > 0 and r.usage.completion_tokens > 0
    assert r.usage.cached_tokens <= r.usage.prompt_tokens
    assert t.streamed is True and t.n_chunks >= 1
    instants = [t.t_start, t.t_request_sent, t.t_headers, t.t_first_chunk, t.t_last_chunk, t.t_end]
    assert None not in instants, instants
    assert instants == sorted(instants), instants
    for first in (t.t_first_reasoning, t.t_first_content, t.t_first_tool_call):
        if first is not None:
            assert t.t_first_chunk <= first <= t.t_last_chunk
    assert len(t.chunk_offsets_ms) == min(t.n_chunks, CHUNK_OFFSETS_CAP)
    assert r.attempts[-1].status == 200
    assert "set-cookie" not in r.response_headers


@pytest.mark.parametrize("provider,model,window", MODELS, ids=[m for _, m, _ in MODELS])
def test_tool_call_round_trip(provider: str, model: str, window: str | None) -> None:
    load_dotenv(REPO / ".env")
    cfg = ModelConfig(
        model=model,
        provider=provider,
        completion_window=window,
        max_output_tokens=2048,
        temperature=0.0,
        max_retries=3,
    )
    client = make_client(cfg)
    msgs = [Message(role="user", content="What is 1234 + 4321? Use the add tool.")]

    r1 = client.complete(SYSTEM, msgs, [ADD])
    _check(r1, f"{model} turn 1")
    assert r1.tool_calls, f"no tool call (finish={r1.finish_reason}, text={r1.text[:300]!r})"
    tc = r1.tool_calls[0]
    assert tc.name == "add"
    assert "_raw" not in tc.arguments, tc.arguments
    assert {int(tc.arguments["a"]), int(tc.arguments["b"])} == {1234, 4321}
    assert tc.id
    assert r1.timing.t_first_tool_call is not None

    msgs += [
        Message(role="assistant", content=r1.text or None, tool_calls=r1.tool_calls, reasoning=r1.reasoning or None),
        Message(role="tool", tool_results=[ToolResult(call_id=tc.id, content="5555")]),
    ]
    r2 = client.complete(SYSTEM, msgs, [ADD])
    _check(r2, f"{model} turn 2")
    assert r2.usage.prompt_tokens > r1.usage.prompt_tokens
    assert "5555" in r2.text.replace(",", "").replace(" ", ""), r2.text[:300]
    assert r2.request.msg_hashes[0] == r1.request.msg_hashes[0]  # shared prefix hashes identically
