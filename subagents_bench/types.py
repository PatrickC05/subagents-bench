"""Shared message, tool and LLM-response types.

These are the interface between the LLM clients (:mod:`subagents_bench.llm`)
and the agent runtime (orchestrator, subagents, arbiter). An
:class:`LLMResponse` carries everything a client observed about one logical
model call -- content, reasoning, tool calls, token usage, streaming
timestamps, retry attempts, response headers -- so the runtime can write a
complete :class:`~subagents_bench.schema.CallRecord` without knowing which
provider served it.

All timestamps are ``time.time()`` epoch seconds (float). Fields a provider or
transport cannot observe are ``None``, never a made-up default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolSpec:
    name: str
    description: str
    # JSON schema for the tool's input object.
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    role: MessageRole
    # Plain assistant/user text.
    content: str | None = None
    # For assistant turns that called tools.
    tool_calls: list[ToolCall] = field(default_factory=list)
    # For tool-result turns (role == "tool").
    tool_results: list[ToolResult] = field(default_factory=list)
    # Assistant reasoning ("thinking") text returned by the provider, kept so it
    # can be passed back on later turns when the model config asks for it.
    reasoning: str | None = None


@dataclass
class Usage:
    """Token usage for one call (or a sum of calls).

    ``prompt_tokens`` INCLUDES ``cached_tokens`` (OpenAI convention).
    ``completion_tokens`` INCLUDES ``reasoning_tokens``.
    ``reasoning_tokens`` is None when the provider does not break it out.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int | None = None

    @property
    def uncached_prompt_tokens(self) -> int:
        return max(self.prompt_tokens - self.cached_tokens, 0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cached_tokens += other.cached_tokens
        if other.reasoning_tokens is not None:
            self.reasoning_tokens = (self.reasoning_tokens or 0) + other.reasoning_tokens


@dataclass
class Attempt:
    """One HTTP attempt within a logical call (retries create several)."""

    attempt: int  # 1-based
    t_start: float
    t_end: float
    # HTTP status received, if any. A transport error after the headers arrived
    # (e.g. the stream broke mid-way) keeps the status already received (200)
    # and names the failure in error_type; None only when no response arrived.
    status: int | None
    error_type: str | None = None  # e.g. "RateLimited", "ServerError", "ReadTimeout"
    error: str | None = None  # truncated error body / message
    retry_after_sec: float | None = None  # server-provided Retry-After, if any
    backoff_sec: float = 0.0  # sleep taken AFTER this attempt before the next


@dataclass
class CallTiming:
    """Wall-clock instants for the SUCCESSFUL attempt of a call.

    ``t_start``/``t_end`` bracket the whole logical call including retries and
    backoff. The other instants refer to the final (successful) attempt.
    """

    t_start: float
    t_end: float
    t_request_sent: float | None = None
    t_headers: float | None = None  # response status/headers received
    t_first_chunk: float | None = None  # first SSE event carrying a non-empty delta
    t_first_reasoning: float | None = None
    t_first_content: float | None = None
    t_first_tool_call: float | None = None
    t_last_chunk: float | None = None
    n_chunks: int = 0
    # Arrival offsets (ms since t_request_sent) of each delta-carrying chunk,
    # capped at CHUNK_OFFSETS_CAP entries.
    chunk_offsets_ms: list[float] = field(default_factory=list)
    streamed: bool = True


CHUNK_OFFSETS_CAP = 4096


@dataclass
class RequestInfo:
    """What was sent, minus the full message bodies (those go to transcripts)."""

    model: str
    provider: str
    base_url: str | None
    completion_window: str | None
    max_tokens: int
    temperature: float
    extra_body: dict[str, Any]
    n_messages: int
    n_tools: int
    prompt_chars: int  # chars of the serialized messages + tool schemas
    system_hash: str  # sha1[:12] of the system prompt
    tools_hash: str  # sha1[:12] of the serialized tool specs
    msg_hashes: list[str]  # sha1[:12] of each serialized message, in order
    msg_chars: list[int]  # serialized length of each message, in order
    reasoning_passback: bool


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    finish_reason: str
    cost_usd: float | None  # from the config price table (billed estimate at run time); None = unpriced
    timing: CallTiming
    request: RequestInfo
    attempts: list[Attempt] = field(default_factory=list)
    reasoning: str = ""
    usage_source: Literal["provider", "estimated"] = "provider"
    usage_raw: dict[str, Any] | None = None
    provider_reported_cost: float | None = None
    response_headers: dict[str, str] = field(default_factory=dict)
    response_id: str | None = None
    raw: Any = None

    # Back-compat conveniences used by older call sites/tests.
    @property
    def t_start(self) -> float:
        return self.timing.t_start

    @property
    def t_end(self) -> float:
        return self.timing.t_end

    @property
    def stop_reason(self) -> str:
        return self.finish_reason


@dataclass
class ToolContext:
    """Everything a tool function needs besides its arguments."""

    env: Any  # subagents_bench.environment.Environment
    agent_id: str  # keys the persisted bash state dir /tmp/.sb/<agent_id>/
    initial_cwd: str  # where the agent's shell starts (workdir, or a private copy)
    max_output_chars: int
    command_timeout_sec: int
    max_command_timeout_sec: int


@dataclass
class ToolOutput:
    """Result of one tool invocation, with the measurements ToolRecord needs."""

    text: str  # what the model sees (already truncated)
    is_error: bool = False
    exit_code: int | None = None
    timed_out: bool = False
    timeout_sec: int | None = None
    output_chars_raw: int = 0
    truncated: bool = False
    cwd_before: str | None = None
    cwd_after: str | None = None
    n_exec: int = 0
    exec_sec: float = 0.0
    error: str | None = None
    # Files this tool wrote (edit_file), for per-dispatch attribution.
    files_written: list[str] = field(default_factory=list)


class LLMError(RuntimeError):
    """A call failed after exhausting retries (or with a non-retryable error).

    ``attempts`` are the HTTP attempts made (empty if none was sent);
    ``request`` is the :class:`RequestInfo` of what was (or would have been)
    sent, when the client built one, so a failed call's record carries the
    same request hashes as a successful one.
    """

    def __init__(
        self,
        message: str,
        attempts: list[Attempt] | None = None,
        status: int | None = None,
        request: RequestInfo | None = None,
    ):
        super().__init__(message)
        self.attempts = attempts or []
        self.status = status
        self.request = request


class BillingError(LLMError):
    """Provider refused for billing/quota reasons. Never retried; aborts the
    episode (exit_reason="billing_error") and the sweep launcher stops."""


class ContextLengthError(LLMError):
    """Prompt exceeded the model's context window (HTTP 400 with a
    context/length message). The runtime may trim and retry once."""


class AgentStopped(RuntimeError):
    """Raised inside the runtime when the episode was cancelled from outside
    (e.g. harbor's agent timeout) so worker threads unwind promptly."""
