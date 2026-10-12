"""On-disk record schemas: the single source of truth for every datum we log.

One episode (= one harbor trial, or one local run) writes these files into its
episode directory (for harbor: ``<job>/<trial>/agent/``):

======================  =====================================================
``episode.json``        :class:`EpisodeRecord` (rewritten at start and end)
``calls.jsonl``         :class:`CallRecord`, one per LLM call (all roles)
``tools.jsonl``         :class:`ToolRecord`, one per tool invocation
``steps.jsonl``         :class:`StepRecord`, one per orchestrator step
``spawns.jsonl``        :class:`SpawnRecord`, one per ``spawn_subagents`` call
``dispatches.jsonl``    :class:`DispatchRecord`, one per subagent run
``arbitrations.jsonl``  :class:`ArbitrationRecord`, one per arbitrated instruction
``transcript.jsonl``    :class:`TranscriptRecord`, every message every agent saw/produced
======================  =====================================================

JSONL files are appended and flushed as events happen, so a crashed or
timed-out episode still leaves everything up to the failure on disk.

Each dataclass field carries ``metadata={"doc": ...}``; :func:`field_docs`
renders them for ``docs/DATA.md`` and :func:`validate_record` checks a parsed
record against its schema (used by ``subagents-bench check-run`` and tests).

Conventions:

* timestamps are ``time.time()`` epoch seconds; durations are seconds;
* token counts follow :class:`~subagents_bench.types.Usage` (prompt includes
  cached, completion includes reasoning);
* "unknown/not observable" is ``None`` -- never a fabricated 0.
"""

from __future__ import annotations

import dataclasses
import types as _pytypes
import typing
from dataclasses import dataclass, field
from typing import Any

# 3: episode.json gained n_provider_errors, cancel_kill, source_sha256 and
# harness_exec_sum_sec (harness_exec_sec became the union of the execs);
# arbitrations.jsonl gained isolation_violated, leaked_files, n_path_remaps
# and final_from_round.
SCHEMA_VERSION = 3

Role = typing.Literal["orchestrator", "subagent", "arbiter", "debate"]


def _f(
    doc: str,
    default: Any = dataclasses.MISSING,
    default_factory: Any = dataclasses.MISSING,
    since: int | None = None,
):
    """A documented field. ``since``: the schema version that added it, so
    :func:`validate_record` accepts its absence from older records."""
    md: dict[str, Any] = {"doc": doc}
    if since is not None:
        md["since"] = since
    if default_factory is not dataclasses.MISSING:
        return field(default_factory=default_factory, metadata=md)
    if default is not dataclasses.MISSING:
        return field(default=default, metadata=md)
    return field(metadata=md)


# ---------------------------------------------------------------------------
# calls.jsonl
# ---------------------------------------------------------------------------


@dataclass
class CallRecord:
    episode_id: str = _f("Episode this call belongs to.")
    call_id: str = _f("Unique id of this LLM call.")
    seq: int = _f("Global start order of this call within the episode (0-based, shared counter with tools).")
    role: Role = _f("orchestrator | subagent | arbiter (judge call) | debate (candidate revision call).")
    agent_id: str = _f("'orchestrator', 'sub-<n>', or 'arbiter-<arbitration_id>'.")
    dispatch_id: str | None = _f("Subagent dispatch this call belongs to (subagent/debate roles).")
    group_id: str | None = _f("spawn_subagents group (subagent/arbiter/debate roles).")
    arbitration_id: str | None = _f("Arbitration this call belongs to (arbiter/debate roles, and candidate subagents).")
    step: int = _f("Agent-local step number (1-based); for arbiter/debate the round number.")
    agent_seq: int = _f("0-based position of this event in its agent's sequential chain (shared with that agent's tool events; a candidate's debate calls continue its counter).")
    # --- request -----------------------------------------------------------
    model: str = _f("Model id sent to the provider.")
    provider: str = _f("Provider key from the config (sail, prime, anthropic, vllm, ...).")
    base_url: str | None = _f("Endpoint base URL.")
    completion_window: str | None = _f("Sail tier sent in X-Sail-Completion-Window (asap/balanced/flex), else None.")
    max_tokens: int = _f("max_tokens requested.")
    temperature: float = _f("Sampling temperature requested.")
    extra_body: dict = _f("Extra request-body params (e.g. reasoning controls).")
    n_messages: int = _f("Wire messages in the request, excluding the system prompt (a runtime tool turn with k results counts as k, as a prefix cache sees them).")
    n_tools: int = _f("Tool schemas offered.")
    prompt_chars: int = _f("Serialized chars of system + messages + tool schemas.")
    system_hash: str = _f("sha1[:12] of the system prompt.")
    tools_hash: str = _f("sha1[:12] of the serialized tool specs.")
    msg_hashes: list = _f("sha1[:12] per serialized wire message (same counting as n_messages), in order, for offline prefix-cache simulation.")
    msg_chars: list = _f("Serialized chars per wire message, in order.")
    reasoning_passback: bool = _f("Whether prior reasoning text was sent back on assistant turns.")
    context_trimmed_chars: int = _f("Chars of older tool output elided by the context guard before this call.")
    # --- timing --------------------------------------------------------------
    t_start: float = _f("Call start (before first attempt).")
    t_end: float = _f("Call end (after the successful attempt's stream closed).")
    latency_sec: float = _f("t_end - t_start, including retries/backoff.")
    t_request_sent: float | None = _f("When the successful attempt's request was sent.")
    t_headers: float | None = _f("When the successful attempt's response headers arrived.")
    t_first_chunk: float | None = _f("First streamed delta of any kind.")
    t_first_reasoning: float | None = _f("First reasoning delta.")
    t_first_content: float | None = _f("First visible content delta.")
    t_first_tool_call: float | None = _f("First tool-call delta.")
    t_last_chunk: float | None = _f("Last streamed delta.")
    ttft_sec: float | None = _f("t_first_chunk - t_request_sent (queue + prefill + first token).")
    decode_sec: float | None = _f("t_last_chunk - t_first_chunk.")
    n_chunks: int = _f("Delta-carrying SSE chunks received.")
    chunk_offsets_ms: list = _f("Arrival offset (ms from t_request_sent) of each delta chunk, capped at 4096.")
    streamed: bool = _f("True if the response was streamed (False for non-streaming clients).")
    n_attempts: int = _f("HTTP attempts made (1 = no retries).")
    backoff_sec: float = _f("Total time slept between attempts.")
    attempts: list = _f("Per-attempt [{attempt,t_start,t_end,status,error_type,error,retry_after_sec,backoff_sec}]. status is the HTTP status received (a stream that broke after the headers keeps it, e.g. 200 with error_type ReadError) and None only when no response arrived; count transport failures by error_type.")
    # --- usage / output -------------------------------------------------------
    prompt_tokens: int = _f("Provider-reported prompt tokens (includes cached).")
    completion_tokens: int = _f("Provider-reported completion tokens (includes reasoning).")
    cached_tokens: int = _f("Prompt tokens served from the provider's prefix cache (0 also when the provider does not report caching; usage_raw tells the two apart).")
    uncached_prompt_tokens: int = _f("prompt_tokens - cached_tokens.")
    reasoning_tokens: int | None = _f("Reasoning tokens, if the provider breaks them out.")
    usage_source: str = _f("'provider'; 'estimated' (ceil(chars/4) when the provider sent no usage); 'none' (the call failed or was abandoned: nothing was reported and the usage fields are 0).")
    usage_raw: dict | None = _f("The provider's usage object verbatim.")
    finish_reason: str = _f("Provider finish reason (stop, tool_calls, length, ...); 'unknown' if the stream ended with [DONE] but no finish_reason; 'error' for a failed or abandoned call.")
    n_tool_calls: int = _f("Tool calls in the response.")
    tool_call_names: list = _f("Names of the tool calls, in order.")
    content_chars: int = _f("Visible content chars.")
    reasoning_chars: int = _f("Reasoning chars.")
    response_id: str | None = _f("Provider response id (SSE chunk 'id').")
    response_headers: dict = _f("Response headers (cookies removed).")
    # --- cost -----------------------------------------------------------------
    cost_usd: float | None = _f("Cost from the config price table at run time (billed estimate); None if the config has no price. The study recomputes cost offline per pricing scheme.")
    price_in: float | None = _f("USD / 1M uncached input tokens used for cost_usd (None = not configured).")
    price_out: float | None = _f("USD / 1M output tokens used for cost_usd (None = not configured).")
    price_cache_read: float | None = _f("USD / 1M cached input tokens used for cost_usd (None -> price_in).")
    provider_reported_cost: float | None = _f("Cost field from the provider's usage object, verbatim (untrusted).")
    error: str | None = _f("Set if the call ultimately failed ('<ExceptionType>: message') or was abandoned when the episode stopped ('cancelled: ...'); usage fields are then 0.", default=None)

    stream_repairs: int | None = _f("Repairs the client made to the streamed tool calls of this response: repeated argument deltas and duplicate calls dropped plus clashing call ids renamed (Sail re-sends a complete tool call under a second index). 0 normally; None for a failed call.", None, since=3)

# ---------------------------------------------------------------------------
# tools.jsonl
# ---------------------------------------------------------------------------


@dataclass
class ToolRecord:
    episode_id: str = _f("Episode id.")
    tool_event_id: str = _f("Unique id of this tool invocation.")
    seq: int = _f("Global start order (shared counter with calls).")
    role: Role = _f("Role of the agent that invoked the tool.")
    agent_id: str = _f("Agent that invoked the tool.")
    dispatch_id: str | None = _f("Subagent dispatch (None for the orchestrator).")
    group_id: str | None = _f("spawn group of the invoking subagent; for spawn_subagents, the group it created.")
    arbitration_id: str | None = _f("Arbitration of the invoking candidate subagent, if any.")
    step: int = _f("Agent-local step in which the tool was requested.")
    agent_seq: int = _f("Position in the agent's sequential chain.")
    call_id: str = _f("LLM call that requested this tool.")
    tool_call_id: str = _f("Provider's tool_call id.")
    name: str = _f("Tool name; the runtime's own tools are recorded too: finish, submit and spawn_subagents (whose duration is the time the orchestrator blocked on its group).")
    args: dict = _f("Tool arguments (string values over 4000 chars truncated with a marker).")
    t_start: float = _f("Tool start.")
    t_end: float = _f("Tool end.")
    duration_sec: float = _f("t_end - t_start.")
    exit_code: int | None = _f("Shell exit code for bash/search/etc.; None for non-shell tools.")
    timed_out: bool = _f("True if the command hit its timeout.")
    timeout_sec: int | None = _f("Timeout applied to the command (None for non-shell tools).")
    output_chars_raw: int = _f("Chars of output before truncation.")
    output_chars_returned: int = _f("Chars returned to the model.")
    truncated: bool = _f("Whether output was truncated.")
    is_error: bool = _f("Whether the tool result was flagged as an error to the model.")
    error: str | None = _f("Error message, if any.")
    cwd_before: str | None = _f("bash: working directory before the command.")
    cwd_after: str | None = _f("bash: working directory after the command.")
    n_exec: int = _f("Number of environment exec round-trips this tool made.")
    exec_sec: float = _f("Seconds spent inside environment exec calls (rest is harness overhead).")


# ---------------------------------------------------------------------------
# steps.jsonl
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    episode_id: str = _f("Episode id.")
    step: int = _f("Orchestrator step (1-based).")
    call_id: str | None = _f("The orchestrator LLM call of this step (None if the call failed).")
    t_start: float = _f("Step start (before the LLM call).")
    t_end: float = _f("Step end (after all of the step's tools finished).")
    duration_sec: float = _f("t_end - t_start.")
    llm_sec: float = _f("Seconds in the orchestrator LLM call.")
    tools_sec: float = _f("Seconds executing the step's tools (incl. blocking on subagents).")
    n_tool_calls: int = _f("Tool calls requested in this step.")
    tool_names: list = _f("Tool names requested, in order.")
    spawned_group_ids: list = _f("spawn groups created in this step.")
    finished: bool = _f("Whether the orchestrator called finish in this step.")


# ---------------------------------------------------------------------------
# spawns.jsonl
# ---------------------------------------------------------------------------


@dataclass
class SpawnRecord:
    episode_id: str = _f("Episode id.")
    group_id: str = _f("Unique id of this spawn_subagents call.")
    orchestrator_step: int = _f("Orchestrator step that issued it.")
    call_id: str = _f("Orchestrator LLM call that requested it.")
    tool_call_id: str = _f("Provider tool_call id.")
    strategy: str = _f("parallel | sequential | arbiter.")
    scope: str = _f("easy | medium | hard (delegation scope -> subagent tool set).")
    requested: int = _f("Instructions the orchestrator asked for.")
    granted: int = _f("Instructions actually run.")
    cap_reason: str | None = _f("Why fewer were granted: max_concurrent | max_dispatches | None.")
    dispatches_remaining_after: int = _f("Dispatch budget left after this call.")
    instructions: list = _f("All requested instruction strings (granted ones first).")
    dispatch_ids: list = _f("Dispatches created (arbiter: all candidates).")
    arbitration_ids: list = _f("Arbitrations created (arbiter strategy only).")
    concurrency: int = _f("Max subagents allowed to run at once in this group.")
    t_start: float = _f("Group start.")
    t_end: float = _f("Group end (all results ready).")
    duration_sec: float = _f("t_end - t_start.")
    sum_dispatch_sec: float = _f("Sum of member dispatch durations (= serial time).")
    files_modified: list | None = _f("Files under the workdir modified during the group (capped at 200), None if not measured.")
    n_files_modified: int | None = _f("Count of modified files (uncapped), None if not measured.")


# ---------------------------------------------------------------------------
# dispatches.jsonl
# ---------------------------------------------------------------------------


@dataclass
class DispatchRecord:
    episode_id: str = _f("Episode id.")
    dispatch_id: str = _f("Unique id of this subagent run.")
    agent_id: str = _f("Subagent id ('sub-<n>').")
    group_id: str = _f("spawn group.")
    arbitration_id: str | None = _f("Arbitration it is a candidate in, if any.")
    candidate_index: int | None = _f("Candidate number within the arbitration (0-based).")
    instruction_index: int = _f("Index of its instruction within the spawn call.")
    orchestrator_step: int = _f("Orchestrator step that spawned it.")
    strategy: str = _f("parallel | sequential | arbiter.")
    scope: str = _f("easy | medium | hard.")
    model: str = _f("Subagent model id.")
    tools: list = _f("Tool names available to the subagent.")
    instruction: str = _f("Full instruction text.")
    report: str = _f("Final report returned (submit text, or last text).")
    status: str = _f("submitted | no_tool_call | max_steps | context_overflow | error | cancelled | token_budget | timeout (the last two: a token or wall-clock safety cap was reached before a step).")
    steps_used: int = _f("LLM steps taken.")
    max_steps: int = _f("Step budget.")
    t_queued: float = _f("When the dispatch was created.")
    t_start: float = _f("When it began running.")
    t_end: float = _f("When it finished.")
    duration_sec: float = _f("t_end - t_start.")
    queue_sec: float = _f("t_start - t_queued (waiting for a concurrency slot).")
    n_llm_calls: int = _f("LLM calls made.")
    prompt_tokens: int = _f("Sum of prompt tokens.")
    completion_tokens: int = _f("Sum of completion tokens.")
    cached_tokens: int = _f("Sum of cached prompt tokens.")
    reasoning_tokens: int | None = _f("Sum of reasoning tokens (None if never reported).")
    cost_usd: float | None = _f("Billed-estimate cost of its calls (None if any call was unpriced).")
    llm_sec: float = _f("Seconds in LLM calls.")
    tool_sec: float = _f("Seconds in tool calls (submit included; it takes ~0 s).")
    n_tool_calls: int = _f("Tool calls made (excluding submit).")
    tool_counts: dict = _f("Tool name -> count.")
    n_tool_errors: int = _f("Tool results flagged as errors.")
    n_timeouts: int = _f("Commands that timed out.")
    context_trims: int = _f("Times the context guard elided old output.")
    workdir: str = _f("Working directory the subagent started in.")
    isolation: str = _f("shared | copy (private copy of the workdir) | shared_fallback (copy skipped: too large).")
    files_modified: list | None = _f("Files it modified (edit_file targets + detected changes; capped), None if not measured.")
    n_files_modified: int | None = _f("Count of modified files, None if not measured.")
    error: str | None = _f("Error text if status == error.")


# ---------------------------------------------------------------------------
# arbitrations.jsonl
# ---------------------------------------------------------------------------


@dataclass
class ArbitrationRecord:
    episode_id: str = _f("Episode id.")
    arbitration_id: str = _f("Unique id.")
    group_id: str = _f("spawn group.")
    instruction_index: int = _f("Which instruction of the spawn call.")
    instruction: str = _f("Instruction text.")
    n_candidates: int = _f("Candidates run.")
    candidate_dispatch_ids: list = _f("Dispatch ids of the candidates, by candidate index.")
    isolation: str = _f("shared | copy | shared_fallback.")
    rounds: list = _f("Per judge round: {round, judge_call_id, debate_call_ids, winner, confidence, agreement, rationale, parse_ok, t_start, t_end}.")
    debate_rounds_used: int = _f("Debate rounds actually run (confidence-aware: 0 if the first judgement was confident).")
    final_winner: int = _f("Chosen candidate index: the winner of the judgement in final_from_round (0 when no round produced one).")
    final_confidence: float | None = _f("Judge confidence in [0,1] of the judgement used (final_from_round).")
    agreement: bool | None = _f("Judge's verdict on whether candidates agreed, from the judgement used (final_from_round).")
    applied: bool = _f("Whether the winner's changes were applied to the real workdir (copy isolation only).")
    applied_files: list = _f("Files written/deleted when applying the winner (capped at 200).")
    apply_error: str | None = _f("Error while applying the winner, or while restoring leaked files (prefixed 'restoring leaked files: '), if any.")
    isolation_sec: float = _f("Seconds spent before the candidates started on isolation: workdir size check, the workdir's pristine manifest, the base copy and the candidates' private copies (~0 for shared).")
    t_start: float = _f("Candidates start (after isolation setup).")
    t_candidates_end: float = _f("All candidates finished. With copy isolation, the leak check and restore, mapping copy paths back to the workdir path inside the copies, and the judge's diff summaries happen between this and rounds[0].t_start.")
    t_end: float = _f("Final judgement, apply and private-copy cleanup finished.")
    duration_sec: float = _f("t_end - t_start (isolation_sec is not included).")
    isolation_violated: bool | None = _f("Copy isolation only: whether files in the real workdir changed while the candidates ran (other than files a concurrent arbitration of the episode applied meanwhile), i.e. a candidate wrote the real workdir instead of its copy (or a process started before the arbitration, e.g. a server logging there, kept writing it). Those changes were undone (restored from the base copy) before the winner was applied, unless the restore failed (apply_error then starts with 'restoring leaked files: '). Changes outside the workdir are not checked. None for shared/shared_fallback isolation or when the check could not run.", default=None, since=3)
    leaked_files: list = _f("The real-workdir files behind isolation_violated (absolute paths, capped at 200; modified, added or deleted while the candidates ran, then restored).", default_factory=list, since=3)
    n_path_remaps: int = _f("Candidate tool calls whose arguments named the real workdir and were redirected to the candidate's private copy (a file tool's path, or occurrences in a bash command); summed over the candidates, 0 without copy isolation.", default=0, since=3)
    final_from_round: int | None = _f("Judge round whose judgement was used: the latest round with a complete judgement (parse_ok), else the latest with a parseable winner. A later round whose judge call failed or whose reply did not parse never replaces it. None when no round produced a winner (final_winner is then the fallback 0).", default=None, since=3)


# ---------------------------------------------------------------------------
# transcript.jsonl
# ---------------------------------------------------------------------------


@dataclass
class TranscriptRecord:
    episode_id: str = _f("Episode id.")
    agent_id: str = _f("Agent whose context this message is in.")
    role: Role = _f("Role of that agent.")
    dispatch_id: str | None = _f("Dispatch id for subagents.")
    index: int = _f("Message index in that agent's context (system prompt = 0). A debate call is a separate context under the candidate's agent_id with role 'debate', and its index restarts at 0 each debate round.")
    t: float = _f("When the message was appended.")
    message_role: str = _f("system | user | assistant | tool.")
    content: str | None = _f("Text content (tool results: the text the model saw).")
    reasoning: str | None = _f("Assistant reasoning text, if any.")
    tool_calls: list = _f("[{id,name,arguments}] for assistant turns.")
    tool_results: list = _f("[{call_id,content,is_error}] for tool turns.")
    call_id: str | None = _f("LLM call that produced this assistant message.")


# ---------------------------------------------------------------------------
# episode.json
# ---------------------------------------------------------------------------


@dataclass
class EpisodeRecord:
    schema_version: int = _f("Record schema version.")
    status: str = _f("running | done (rewritten at the end).")
    episode_id: str = _f("Unique id of this episode.")
    experiment: str = _f("config.name (the condition).")
    config: dict = _f("Full resolved ExperimentConfig.")
    config_hash: str = _f("sha1[:12] of the canonical config JSON.")
    strategy: str | None = _f("parallel | sequential | arbiter; None for topology=single.")
    scope: str | None = _f("easy | medium | hard; None for topology=single.")
    topology: str = _f("single | subagents.")
    orchestrator_model: str = _f("Orchestrator model id.")
    subagent_model: str | None = _f("Subagent model id (None for single).")
    arbiter_model: str | None = _f("Arbiter model id (arbiter strategy only).")
    # --- task / harness identity --------------------------------------------
    task_name: str = _f("Task name (harbor task_name, or the local task id).")
    trial_name: str | None = _f("harbor trial name (task__suffix).")
    harbor_session_id: str | None = _f("harbor agent session id.")
    harbor_trial_id: str | None = _f("harbor trial UUID (agent context_id).")
    logs_dir: str = _f("Directory these files were written to.")
    instruction: str = _f("Full task instruction given to the orchestrator.")
    # --- environment provenance ---------------------------------------------
    harness_version: str = _f("subagents_bench.__version__.")
    git_commit: str | None = _f("Harness git commit (HEAD of the repo holding the package); None outside a git work tree (e.g. a ZIP or tarball copy) or when that work tree tracks none of the package (a copy unpacked inside an unrelated repository).")
    git_dirty: bool | None = _f("Uncommitted changes to tracked harness source (subagents_bench/, configs/, scripts/, pyproject.toml, uv.lock; untracked files and other paths are ignored); None outside a git work tree. Use source_sha256 to tell code versions apart.")
    source_sha256: str | None = _f("Fingerprint of the harness source that ran: sha256 over the sorted (path, sha256 of contents) of subagents_bench/, configs/, scripts/, pyproject.toml and uv.lock, uncommitted and untracked files included (subagents_bench.provenance.source_fingerprint). Equal values = identical code, with or without git; computed once per process, when harbor imports the agent (local runs: at the first episode). Analysis-only files (subagents_bench/analysis/, plots/, sweep.py, cli.py, configs/, scripts/: provenance.NON_RUNTIME_PATHS) are included, though no episode uses them. None if it could not be computed.", since=3)
    harbor_version: str | None = _f("Installed harbor version.")
    python_version: str = _f("Python version.")
    host_platform: str = _f("platform.platform() of the host.")
    host_machine: str = _f("platform.machine() of the host (e.g. arm64).")
    container_arch: str | None = _f("uname -m inside the container (x86_64 on an arm64 host = emulated).")
    sandbox_kind: str = _f("harbor | local_docker | local_shell | exploitbench.")
    workdir: str = _f("Resolved working directory in the container.")
    has_timeout_binary: bool | None = _f("Whether coreutils `timeout` exists in the container (in-container kill of runaway commands).")
    # --- timing ---------------------------------------------------------------
    t_start: float = _f("Episode start.")
    t_end: float | None = _f("Episode end.")
    wall_clock_sec: float | None = _f("t_end - t_start.")
    t_first_call: float | None = _f("First LLM call start.")
    # --- outcome --------------------------------------------------------------
    exit_reason: str | None = _f("finished | max_steps | no_tool_call | token_budget | timeout | cancelled | billing_error | provider_error | error. provider_error: an orchestrator LLM call failed with a provider error after all retries (see n_provider_errors); under harbor the agent then raises ProviderUnavailableError, so the verifier is skipped.")
    error: str | None = _f("Traceback/message for exit_reason in (error, billing_error, provider_error): '<ExceptionType>: message' (the message carries the HTTP status when there was one); harbor's cancel note for cancelled.")
    cancel_kill: dict | None = _f("Set only when harbor cancelled the agent: the best-effort kill of the agent's still-running in-container commands, run after the cancel grace wait and before the verifier. {attempted: a kill exec was sent, ok: it completed, n_killed: commands whose process group/tree was killed, error: why it failed or None}. None when there was no cancel.", since=3)
    final_summary: str | None = _f("Orchestrator's finish summary (or last text).")
    # --- counts ---------------------------------------------------------------
    orchestrator_steps: int = _f("Orchestrator LLM steps taken.")
    spawn_calls: int = _f("spawn_subagents calls.")
    dispatches: int = _f("Subagent runs (arbiter candidates included).")
    redispatches: int = _f("Dispatches from the 2nd and later spawn calls.")
    arbitrations: int = _f("Arbitrations run.")
    requested_total: int = _f("Sum of requested instructions.")
    granted_total: int = _f("Sum of granted instructions.")
    instruction_following: float | None = _f("Mean granted/requested over spawn calls (None if no spawns).")
    n_calls: dict = _f("role -> LLM call count. Every dict keyed by role has the four keys orchestrator, subagent, arbiter, debate.")
    n_tool_calls: dict = _f("role -> tool call count (finish, submit and spawn_subagents included).")
    tokens: dict = _f("role -> {prompt, completion, cached, uncached_prompt, reasoning}; reasoning is None if never reported.")
    cost_usd: dict = _f("role -> billed-estimate USD, plus 'total'; a value is None if any of its calls was unpriced.")
    llm_sec: dict = _f("role -> summed LLM call seconds.")
    tool_sec: dict = _f("role -> summed tool seconds (the orchestrator's includes the time blocked in spawn_subagents).")
    n_retries: int = _f("Extra HTTP attempts across all calls.")
    backoff_sec: float = _f("Total retry backoff sleep.")
    n_stream_repairs: int = _f("Sum of calls.jsonl stream_repairs: streamed tool-call repairs across all calls (0 normally).", since=3)
    n_provider_errors: int = _f("LLM calls of any role that failed with a provider error after all retries: a plain LLMError, not a BillingError or ContextLengthError (retries exhausted, a non-retryable 4xx such as 401/403, a missing API key); their calls.jsonl error starts with 'LLMError:' (telemetry.is_provider_error, which also works on older records). Over 0 means a provider failure touched the episode (a subagent dispatch that failed this way has status error), so the trial does not measure the condition.", since=3)
    n_command_timeouts: int = _f("Commands that hit their timeout.")
    n_context_trims: int = _f("Context-guard trims across all agents.")
    harness_exec_n: int | None = _f("Environment exec round-trips the harness made for its own measurements (workdir markers, modified-file scans, git diff --stat), None if not counted.")
    harness_exec_sec: float | None = _f("Wall-clock seconds during which at least one of those measurement execs ran (the union of their intervals: concurrent ones, e.g. parallel dispatches' markers and scans, count once). They belong to no agent's chain, so they show up in overhead_sec. Before schema_version 3 this was the plain sum (now harness_exec_sum_sec).")
    harness_exec_sum_sec: float | None = _f("Summed seconds of those measurement execs (concurrent ones counted in full); >= harness_exec_sec.", since=3)
    # --- final container state -------------------------------------------------
    files_modified: list | None = _f("Files under the workdir modified during the episode (capped at 500), None if not measured.")
    n_files_modified: int | None = _f("Count (uncapped), None if not measured.")
    git_diff_stat: str | None = _f("`git diff --stat` in the workdir if it is a git repo.")
    # --- scoring (filled at aggregation from harbor's result.json, or by the local runner) ---
    reward: float | None = _f("Verifier reward; None until joined.", default=None)
    resolved: bool | None = _f("reward == 1; None until joined.", default=None)


RECORD_TYPES: dict[str, type] = {
    "calls.jsonl": CallRecord,
    "tools.jsonl": ToolRecord,
    "steps.jsonl": StepRecord,
    "spawns.jsonl": SpawnRecord,
    "dispatches.jsonl": DispatchRecord,
    "arbitrations.jsonl": ArbitrationRecord,
    "transcript.jsonl": TranscriptRecord,
    "episode.json": EpisodeRecord,
}


def field_docs(cls: type) -> list[tuple[str, str, str]]:
    """(name, type, doc) for each field of a record class."""
    out = []
    hints = typing.get_type_hints(cls)
    for f in dataclasses.fields(cls):
        doc = f.metadata.get("doc", "")
        if f.metadata.get("since") is not None:
            doc = f"{doc} (Added in schema_version {f.metadata['since']}; absent from older records.)"
        out.append((f.name, _type_name(hints.get(f.name, Any)), doc))
    return out


def _type_name(tp: Any) -> str:
    origin = typing.get_origin(tp)
    if origin in (typing.Union, _pytypes.UnionType):
        return " | ".join(_type_name(a) for a in typing.get_args(tp))
    if origin is typing.Literal:
        return "str"
    if tp is type(None):
        return "None"
    return getattr(tp, "__name__", str(tp))


def _accepts(tp: Any, value: Any) -> bool:
    origin = typing.get_origin(tp)
    if tp is Any:
        return True
    if origin in (typing.Union, _pytypes.UnionType):
        return any(_accepts(a, value) for a in typing.get_args(tp))
    if origin is typing.Literal:
        return value in typing.get_args(tp)
    if tp is type(None):
        return value is None
    if tp is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if tp is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if tp in (list, dict, str, bool):
        return isinstance(value, tp)
    return True


def validate_record(cls: type, record: dict[str, Any]) -> list[str]:
    """Return a list of problems (missing fields, wrong types, unknown fields).

    A field added in a later schema version (``since``) may be missing from a
    record that carries an older ``schema_version`` (episode.json written by
    older code). Per-event records (the JSONL lines) carry no
    ``schema_version``; there a ``since`` field that has a default may be
    missing (readers use the default)."""
    problems = []
    hints = typing.get_type_hints(cls)
    version = record.get("schema_version")
    names = set()
    for f in dataclasses.fields(cls):
        names.add(f.name)
        if f.name not in record:
            since = f.metadata.get("since")
            if since is not None and isinstance(version, int) and version < since:
                continue
            has_default = f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
            if since is not None and "schema_version" not in record and has_default:
                continue
            problems.append(f"missing field {f.name!r}")
            continue
        if not _accepts(hints[f.name], record[f.name]):
            problems.append(
                f"field {f.name!r}: {type(record[f.name]).__name__} not accepted by {_type_name(hints[f.name])}"
            )
    for k in record:
        if k not in names:
            problems.append(f"unknown field {k!r}")
    return problems
