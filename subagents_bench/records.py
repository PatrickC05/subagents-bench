"""Episode records: one JSONL file per kind plus ``episode.json``, written as events happen.

The recorder is thread-safe. ``finalize`` computes every episode total from the records
written (so totals equal the JSONL sums) and closes the episode: later records (a thread
still unwinding after harbor's cancel) are dropped and counted in ``dropped``.
"""

from __future__ import annotations

import functools
import json
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .config import ExperimentConfig
from .environment import AgentStopped
from .llm import BillingError, LLMError, LLMResponse, Message, ProviderError, ToolCall, ToolSpec
from .tools import ToolOutput

ROLES = ("orchestrator", "subagent", "arbiter", "debate")
KINDS = ("calls", "tools", "steps", "spawns", "dispatches", "arbitrations", "inbox", "transcript")
POLL_SEC = 0.2
ARG_CHARS = 2000  # per string argument of a tool call in its record (the transcript has them whole)


def clip_arguments(arguments: Any) -> Any:
    """A tool call's arguments for its record: long strings cut to ``ARG_CHARS``."""
    if not isinstance(arguments, dict):
        return arguments
    return {k: v[:ARG_CHARS] if isinstance(v, str) else v for k, v in arguments.items()}


@functools.cache
def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True)
    except OSError:
        return None
    return out.stdout.strip() or None


@dataclass(frozen=True)
class Who:
    """The agent an event belongs to."""

    role: str
    agent_id: str
    dispatch_id: str | None = None
    group_id: str | None = None
    arbitration_id: str | None = None

    def fields(self) -> dict[str, Any]:
        return {"role": self.role, "agent_id": self.agent_id, "dispatch_id": self.dispatch_id,
                "group_id": self.group_id, "arbitration_id": self.arbitration_id}

    def with_group(self, group_id: str) -> "Who":
        return replace(self, group_id=group_id)


def complete_stoppable(client: Any, system: str, messages: list[Message], tools: list[ToolSpec],
                       stop: threading.Event) -> LLMResponse:
    """``client.complete`` that is abandoned (AgentStopped) as soon as ``stop`` is set."""
    if stop.is_set():
        raise AgentStopped("episode stopped")
    box: dict[str, Any] = {}
    done = threading.Event()

    def target() -> None:
        try:
            box["r"] = client.complete(system, messages, tools)
        except BaseException as e:  # re-raised in the caller's thread
            box["e"] = e
        finally:
            done.set()

    threading.Thread(target=target, daemon=True).start()
    while not done.wait(POLL_SEC):
        if stop.is_set():
            raise AgentStopped("episode stopped during an LLM call")
    if "e" in box:
        raise box["e"]
    return box["r"]


class EpisodeRecorder:
    def __init__(self, episode_dir: str | Path, config: ExperimentConfig, *, task_name: str,
                 trial_name: str | None = None):
        self.dir = Path(episode_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.episode_id = uuid.uuid4().hex[:12]
        self.stop_event = threading.Event()
        self.failure: LLMError | None = None  # the first provider or billing failure
        self.records: dict[str, list[dict]] = {k: [] for k in KINDS if k != "transcript"}
        self.dropped = 0
        self.finalized = False
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        subagents = config.topology == "subagents"
        self.episode: dict[str, Any] = {
            "episode_id": self.episode_id,
            "condition": config.name,
            "topology": config.topology,
            "strategy": config.subagent.strategy if subagents else None,
            "scope": config.subagent.scope if subagents else None,
            "orchestrator_model": config.orchestrator.model,
            "subagent_model": config.subagent_model.model if subagents else None,
            "task_name": task_name,
            "trial_name": trial_name,
            "git_commit": git_commit(),
            "config": config.to_dict(),
            "t_start": time.time(),
            "exit_reason": None,
        }
        for kind in KINDS:
            (self.dir / f"{kind}.jsonl").touch()
        self._write_episode()

    def new_id(self, kind: str) -> str:
        with self._lock:
            self._counters[kind] = n = self._counters.get(kind, 0) + 1
        return f"{kind}-{n}"

    def write(self, kind: str, record: dict[str, Any]) -> None:
        line = json.dumps({"episode_id": self.episode_id, **record}, default=str, ensure_ascii=False)
        with self._lock:
            if self.finalized:
                self.dropped += 1
                return
            with open(self.dir / f"{kind}.jsonl", "a", encoding="utf-8") as f:
                f.write(line + "\n")
            if kind != "transcript":
                self.records[kind].append(json.loads(line))

    def fail(self, e: LLMError) -> None:
        """Provider fail-fast: remember the first failure and stop the episode."""
        with self._lock:
            self.failure = self.failure or e
        self.stop_event.set()

    def transcript(self, who: Who, message: Message, call_id: str | None = None) -> None:
        self.write("transcript", {
            "agent_id": who.agent_id, "role": who.role, "dispatch_id": who.dispatch_id, "call_id": call_id,
            "t": time.time(), "msg_role": message.role, "content": message.content, "reasoning": message.reasoning,
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in message.tool_calls],
            "tool_results": [{"call_id": r.call_id, "content": r.content, "is_error": r.is_error}
                             for r in message.tool_results],
        })

    def llm(self, who: Who, client: Any, system: str, messages: list[Message], tools: list[ToolSpec],
            step: int) -> tuple[LLMResponse, str]:
        """One recorded, stop-aware LLM call -> (response, call_id). A provider or billing
        failure of any role stops the episode (fail-fast) and is re-raised."""
        call_id = self.new_id("call")
        base = {"call_id": call_id, **who.fields(), "step": step, "model": client.cfg.model}
        t0 = time.time()
        try:
            r = complete_stoppable(client, system, list(messages), tools, self.stop_event)
        except BaseException as e:
            self.write("calls", {
                **base, "t_start": t0, "t_first_chunk": None, "t_end": time.time(), "prompt_tokens": 0,
                "cached_tokens": 0, "completion_tokens": 0, "reasoning_tokens": None, "finish_reason": None,
                "n_tool_calls": 0, "n_attempts": getattr(e, "n_attempts", 0), "backoff_sec": getattr(e, "backoff_sec", 0.0),
                "stream_repairs": 0, "error": "cancelled" if isinstance(e, AgentStopped) else f"{type(e).__name__}: {e}"[:2000],
            })
            if isinstance(e, (ProviderError, BillingError)):
                self.fail(e)
            raise
        self.write("calls", {
            **base, "t_start": r.t_start, "t_first_chunk": r.t_first_chunk, "t_end": r.t_end,
            "prompt_tokens": r.prompt_tokens, "cached_tokens": r.cached_tokens, "completion_tokens": r.completion_tokens,
            "reasoning_tokens": r.reasoning_tokens, "finish_reason": r.finish_reason, "n_tool_calls": len(r.tool_calls),
            "n_attempts": r.n_attempts, "backoff_sec": r.backoff_sec, "stream_repairs": r.stream_repairs, "error": None,
        })
        return r, call_id

    def tool(self, who: Who, step: int, call_id: str, tc: ToolCall, run: Callable[[], ToolOutput]) -> ToolOutput:
        """One recorded tool call; an exception (e.g. AgentStopped) is recorded, then re-raised."""
        t0 = time.time()
        try:
            out = run()
        except BaseException as e:
            out = ToolOutput(f"[{type(e).__name__}: {e}]", is_error=True)
            raise
        finally:
            self.write("tools", {
                **who.fields(), "step": step, "call_id": call_id, "name": tc.name, "t_start": t0, "t_end": time.time(),
                "exit_code": out.exit_code, "timed_out": out.timed_out, "is_error": out.is_error,
                "output_chars": out.output_chars or len(out.text), "tool_call_id": tc.id,
                "arguments": clip_arguments(tc.arguments),
            })
        return out

    def finalize(self, exit_reason: str, error: str | None = None) -> dict[str, Any]:
        """Write the final ``episode.json`` (the first call wins) and close the episode."""
        with self._lock:
            if self.finalized:
                return self.episode
            self.finalized = True
            calls = self.records["calls"]
            tokens = {role: {"prompt": 0, "cached": 0, "completion": 0, "reasoning": 0} for role in ROLES}
            for c in calls:
                t = tokens[c["role"]]
                t["prompt"] += c["prompt_tokens"]
                t["cached"] += c["cached_tokens"]
                t["completion"] += c["completion_tokens"]
                t["reasoning"] += c["reasoning_tokens"] or 0
            t_end = time.time()
            self.episode.update(
                t_end=t_end,
                wall_clock_sec=t_end - self.episode["t_start"],
                exit_reason=exit_reason,
                error=error,
                orchestrator_steps=len(self.records["steps"]),
                spawn_calls=len(self.records["spawns"]),
                dispatches=len(self.records["dispatches"]),
                arbitrations=len(self.records["arbitrations"]),
                tokens_by_role=tokens,
                n_calls=len(calls),
                n_provider_errors=sum(1 for c in calls if (c["error"] or "").startswith("ProviderError")),
                n_stream_repairs=sum(c["stream_repairs"] for c in calls),
                cancel_kill=None,
                reward=None,
                resolved=None,
            )
            if self.episode["config"].get("interactive") is not None:
                hints = [r for r in self.records["inbox"] if r["kind"] == "hint"]
                self.episode.update(n_hints=len(hints),
                                    n_hints_delivered=sum(1 for h in hints if h["t_delivered"] is not None))
            self._write_episode()
            return self.episode

    def update(self, **fields: Any) -> None:
        """Change ``episode.json`` after finalize (harbor's cancel kill, a local verdict)."""
        with self._lock:
            self.episode.update(fields)
            self._write_episode()

    def _write_episode(self) -> None:
        tmp = self.dir / "episode.json.tmp"
        tmp.write_text(json.dumps(self.episode, indent=2, default=str))
        tmp.replace(self.dir / "episode.json")
