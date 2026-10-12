"""One real tool-calling request to each Sail model of configs/sail/base.yaml (RUN_LIVE=1;
SAIL_API_KEY from the environment or .env). Costs a fraction of a cent."""

from __future__ import annotations

from pathlib import Path

import pytest

from subagents_bench.config import load_config, load_dotenv
from subagents_bench.llm import Message, SailClient, ToolSpec

REPO = Path(__file__).resolve().parents[1]
BASE = load_config(REPO / "configs/sail/base.yaml")
ADD = ToolSpec("add", "Add two integers.", {
    "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"],
})


@pytest.mark.live
@pytest.mark.parametrize("cfg", [BASE.orchestrator, BASE.subagent.model], ids=lambda c: c.model)
def test_tool_call(cfg):
    load_dotenv(REPO / ".env")
    client = SailClient(cfg)
    try:
        r = client.complete("Always use the add tool to add numbers.", [Message("user", "What is 1234 + 4321?")], [ADD])
    finally:
        client.close()
    print(f"\n{cfg.model}: {r.finish_reason}, {r.prompt_tokens} in ({r.cached_tokens} cached), "
          f"{r.completion_tokens} out ({r.reasoning_tokens} reasoning), {r.t_end - r.t_start:.1f}s, "
          f"{r.n_attempts} attempt(s), {r.stream_repairs} repair(s)")
    (tc,) = r.tool_calls
    assert tc.name == "add" and {int(tc.arguments["a"]), int(tc.arguments["b"])} == {1234, 4321}
    assert r.prompt_tokens > 0 and r.completion_tokens > 0 and r.t_start < r.t_first_chunk <= r.t_end
