"""Real tool-calling requests (RUN_LIVE=1; PRIME_API_KEY and SAIL_API_KEY from the environment or
.env): one to each Sail model of configs/sail/base.yaml and each Prime model of configs/prime/models/,
and for each Prime model the harness's prompt and tools with a round trip. A few cents in all."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from subagents_bench.config import ModelConfig, from_dict, load_config, load_dotenv
from subagents_bench.llm import ChatClient, Message, ToolResult, ToolSpec

REPO = Path(__file__).resolve().parents[1]
SAIL = load_config(REPO / "configs/sail/base.yaml")
PRIME = sorted((REPO / "configs/prime/models").glob("*.yaml"))
ADD = ToolSpec("add", "Add two integers.", {
    "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"],
})
SYSTEM = "Always use the add tool to add numbers."
ASK = Message("user", "What is 1234 + 4321?")


def prime_model(path: Path) -> ModelConfig:
    return from_dict({"name": "live", "orchestrator": yaml.safe_load(path.read_text())}).orchestrator


MODELS = [SAIL.orchestrator, SAIL.subagent.model] + [prime_model(p) for p in PRIME]


def report(cfg: ModelConfig, r) -> None:
    print(f"\n{cfg.model}: {r.finish_reason}, {r.prompt_tokens} in ({r.cached_tokens} cached), "
          f"{r.completion_tokens} out ({r.reasoning_tokens} reasoning, {len(r.reasoning_details)} detail(s)), "
          f"{r.t_end - r.t_start:.1f}s, {r.n_attempts} attempt(s), {r.stream_repairs} repair(s)")


@pytest.mark.live
@pytest.mark.parametrize("cfg", MODELS, ids=lambda c: c.model)
def test_tool_call(cfg):
    load_dotenv(REPO / ".env")
    client = ChatClient(cfg)
    try:
        r = client.complete(SYSTEM, [ASK], [ADD])
    finally:
        client.close()
    report(cfg, r)
    (tc,) = r.tool_calls
    assert tc.name == "add" and {int(tc.arguments["a"]), int(tc.arguments["b"])} == {1234, 4321}
    assert r.prompt_tokens > 0 and r.completion_tokens > 0 and r.t_start < r.t_first_chunk <= r.t_end


@pytest.mark.live
@pytest.mark.parametrize("path", PRIME, ids=lambda p: p.stem)
def test_harness_prompt_tools_and_round_trip(path):
    """The levels sweep's interactive system prompt and every tool its sessions send (the orchestrator's
    of background_hinted_urgent: all five, background spawn_subagents, wait and reply; the subagents'
    submit), then each call's result sent back with the reply's reasoning (reasoning_details where the
    model sent them): every schema is accepted and the tool loop's second call goes through."""
    from subagents_bench import sweep
    from subagents_bench.interactive import BACKGROUND_SPAWN_SPEC, REPLY_SPEC, WAIT_SPEC
    from subagents_bench.prompts import interactive_prompt
    from subagents_bench.subagent import SUBMIT_SPEC
    from subagents_bench.tools import tool_specs

    load_dotenv(REPO / ".env")
    cfg = sweep.load_spec(REPO / "configs/sweeps/swe_interactive_levels.yaml", model=path).configs[
        "background_hinted_urgent"]
    specs = tool_specs(cfg.orchestrator_tools, cfg.budget) + [BACKGROUND_SPAWN_SPEC, WAIT_SPEC, REPLY_SPEC, SUBMIT_SPEC]
    system = interactive_prompt(cfg, "/app")
    ask = Message("user", "[#1] Quick question before any bugs arrive: what is 1234 + 4321? Answer it with the "
                          "reply tool (to: 1); no need to look at the repository.")
    client = ChatClient(cfg.orchestrator)
    try:
        r = client.complete(system, [ask], specs)
        assert r.tool_calls and all(c.name in {s.name for s in specs} and "_raw" not in c.arguments for c in r.tool_calls)
        history = [ask, Message("assistant", r.text, r.tool_calls, reasoning=r.reasoning or None,
                                reasoning_details=r.reasoning_details or None),
                   Message("tool", tool_results=[ToolResult(c.id, "sent" if c.name == "reply" else "ok")
                                                 for c in r.tool_calls])]
        r2 = client.complete(system, history, specs)
    finally:
        client.close()
    print(f"\n{cfg.orchestrator.model}: called {[(c.name, c.arguments) for c in r.tool_calls]}")
    report(cfg.orchestrator, r)
    report(cfg.orchestrator, r2)
    assert any(c.name == "reply" and str(c.arguments.get("to")).lstrip("#") == "1" for c in r.tool_calls)
    assert r2.finish_reason in ("stop", "tool_calls") and r2.prompt_tokens > r.prompt_tokens
