# subagents-bench design contract

This document is the implementation contract for the harness. The shared types
and on-disk record schemas live in code and are authoritative:

- `subagents_bench/types.py`: messages, tool specs, `LLMResponse`, `Usage`,
  `CallTiming`, `RequestInfo`, `Attempt`, and the error classes.
- `subagents_bench/schema.py`: every per-episode record written to disk (with
  field docs) plus `validate_record`.

Everything below specifies behavior around those types.

## 1. The study

A large **orchestrator** model directs small **subagent** models on
Terminal-Bench 2.0 tasks, run through harbor in Docker. The orchestrator gets a
fixed 5-step budget. We measure:

- resolve rate (harbor's verifier reward);
- tokens, by role and by kind;
- dollars, under several price schemes that are **recomputed offline**;
- wall-clock time, both measured and **simulated** offline under other serving
  assumptions.

Provider prices (Sail) do not track model size, so we **log raw per-call
tokens and timings and recompute cost and latency later**. Budgets are set in
steps and tokens, not seconds. Wall-clock limits are only loose safety caps, so
provider latency never changes a trajectory.

### Axes

| Axis | Values | Where |
|---|---|---|
| topology | `single` (no subagents) / `subagents` | `ExperimentConfig.topology` |
| **strategy** | `parallel` (n subagents concurrently) / `sequential` (one at a time) / `arbiter` (N candidates per instruction, LLM judge, confidence-aware debate) | `subagent.strategy` |
| **scope** (the README's easy/medium/hard) | `easy` = locate/retrieve + summarize: `read_file`, `list_dir`, `search`<br>`medium` = easy + execute/observe + analyze/diagnose: adds `bash`<br>`hard` = medium + plan/decompose + implement/edit + verify/test: adds `edit_file` | `subagent.scope` |
| subagent model | Sail `nvidia/Gemma-4-31B-IT-NVFP4` (default small model), `google/gemma-4-12B-it` (Sail serves a 16384-token context and rejects input + max_tokens above 15872, so it needs `context_window: 16384` and the guard), or the orchestrator itself (same-model control); Prime Qwen3.5-{0.8,2,4,9}B for a size sweep | `subagent.model` |
| task difficulty | Terminal-Bench label (easy/medium/hard), category, expert-time estimate | covariates from `task.toml` |

Naming: "scope" is the delegation difficulty from the README. "difficulty" is
always the Terminal-Bench task label.

### Conditions in the pilot sweep (`configs/sweeps/pilot.yaml`)

The orchestrator is `deepseek-ai/DeepSeek-V4.1-Flash` on Sail's `balanced`
tier. Every Sail model in a sweep uses the same tier.

1. `single_5step`: topology single, 5 steps.
2. `single_matched`: topology single, 40 steps, the **step-matched**
   baseline (the name is kept). 40 = `orchestrator_steps` (5) × subagent
   `max_steps` (8): the orchestrator alone gets as many steps as one serial
   path of 5 orchestrator steps, each followed by one subagent's full 8
   steps. It matches steps, not compute: it does not match FLOPs or tokens
   (DeepSeek steps cost more than Gemma steps), nor the total work of
   parallel subagents, arbiter candidates or several sequential dispatches
   per step.
3. Nine conditions `gemma31b_{parallel,sequential,arbiter}_{easy,medium,hard}`:
   subagents, `max_concurrent=3`, `max_dispatches=15`, `max_steps=8`.
   Arbiter: `n_candidates=3`, `debate_rounds=1`, `confidence_threshold=0.8`,
   arbiter model = orchestrator model.
4. `samemodel_parallel_hard`: subagent model = orchestrator (same-model control).
   The orchestrator's prompt is the same text as in `gemma31b_parallel_hard`
   (see "Prompt policy" below). The subagents do run with the orchestrator's
   model settings (`max_output_tokens` 32768 and no context guard, against
   Gemma's 4096 and 262144); per-call `finish_reason` in `calls.csv` shows
   any truncation.
5. Three **manager** conditions `manager_{parallel,sequential,arbiter}_hard`:
   like `gemma31b_*_hard`, but `tools.orchestrator_tools: [read_file,
   list_dir, search]`, so the orchestrator can read and plan but must delegate
   every command, edit and test to subagents. Added after the 2026-10-11
   smoke sweep, where the orchestrator (with all tools) delegated in 0 of 18
   subagent-condition episodes. Only hard scope is meaningful here: with easy
   or medium scope nobody could edit files. In the follow-up smoke run
   (2026-10-12), `manager_arbiter_hard` delegated in both of its episodes
   (11 dispatches, 3 arbitrations, 2 of them debated); the valid episode
   solved `prove-plus-comm`, and the other is invalid (two Gemma calls
   failed after 9 attempts during a 7-9 minute provider outage). The
   treatment-coverage gate (section 6) now fails a sweep in which a
   subagent condition never delegated.

Tasks:
- 4 TB-2.0 easy tasks: `fix-git`, `prove-plus-comm`, `cobol-modernization`,
  `overfull-hbox`.
- 6 short medium tasks: `crack-7z-hash`, `raman-fitting`,
  `constraints-scheduling`, `nginx-request-logging`,
  `openssl-selfsigned-cert`, `polyglot-c-py`.

That is 3 rounds. The smoke sweep (`configs/sweeps/smoke.yaml`) is `fix-git`
and `prove-plus-comm` × 1 round × all 15 conditions. The optional size sweep
(`configs/sweeps/prime_size.yaml`) runs the four Qwen3.5 arms on Prime with
two anchors in the same shuffled rounds, `single_5step` and
`gemma31b_parallel_hard` (identical to the pilot's), so size effects are
estimated within one sweep rather than against a pilot run hours or days
earlier; the subagent provider (Sail for Gemma, Prime for Qwen) is a
covariate of the Gemma-vs-Qwen gap.

**Prompt policy** (`prompts.py`; snapshot tests in `tests/test_prompts.py`).
The orchestrator's Subagents section is identical in every condition: it
calls subagents "separate model instances running in the SAME container as
you" and never names or characterizes their model (no "smaller, cheaper"),
so the same-model control differs from the Gemma conditions only in the
model actually serving the subagents. It states the step accounting plainly:
one `spawn_subagents` call uses one of the orchestrator's steps, like any
tool call, and the subagents' own steps do not count against its budget.
The default (all-tools) prompt never asks the orchestrator to delegate;
forced delegation is what the manager conditions are for (their prompt adds
the manager note). Medium-scope subagents are told not to modify project
files even if the instruction asks them to, and to report the exact change
(a diff or commands) instead; the orchestrator is told the same. This is a
prompt-level rule: a medium subagent's `bash` can still write files, which
`DispatchRecord.files_modified` shows.

Sampling temperature is 0.7 for every model, so repeated trials are
independent samples. Reasoning is left at the provider default (DeepSeek
thinks; the Gemma models do not) and recorded per call.

## 2. Configuration (`config.py`, owner: runtime)

The existing YAML must keep loading. New fields have defaults:

```yaml
name: str
benchmark: terminal_bench | exploit_bench
topology: single | subagents
subtask_type: none | ...            # legacy prompt prefix, kept
orchestrator: ModelConfig
subagent:
  model: ModelConfig | null         # null -> orchestrator model (same-model control)
  strategy: parallel | sequential | arbiter      # default parallel
  scope: easy | medium | hard                    # default hard
  max_concurrent: 3                 # max subagents LIVE at once (all strategies)
  max_dispatches: null              # null -> max_concurrent * orchestrator_steps; arbiter candidates count individually
  max_steps: 8
  token_budget: null
  max_output_chars: 12000           # tool-output truncation for subagents
  arbiter:
    n_candidates: 3
    debate_rounds: 1                # max debate rounds; run only if not confident/agreeing
    confidence_threshold: 0.8
    model: ModelConfig | null       # null -> orchestrator model
    max_copy_mb: 500                # hard scope: copy workdir per candidate if du <= this, else shared_fallback
budget:
  orchestrator_steps: 5
  total_token_budget: 5000000       # safety cap
  timeout_sec: 3600                 # safety cap only
  command_timeout_sec: 180          # default per-command timeout
  max_command_timeout_sec: 900      # cap on a model-requested timeout
tools:
  max_output_chars: 12000           # orchestrator tool-output truncation
  orchestrator_tools: null          # the orchestrator's own tools (subset of
                                    # bash, edit_file, read_file, list_dir, search;
                                    # finish/spawn_subagents always added); null = all.
                                    # Without bash and edit_file the prompt says
                                    # all commands and edits must be delegated.
sandbox: {kind: harbor|local_docker|local_shell|exploitbench, image, workdir: auto|<path>, network}
task_ids, difficulties, n_trials, seed, output_dir
```

`ModelConfig` adds the following fields:

- `completion_window: str | None` (Sail tier, sent as the `X-Sail-Completion-Window` header)
- `extra_headers: dict`
- `reasoning_passback: bool = True`
- `context_window: int | None` (enables the context guard)
- `max_retries: int = 4`
- `retry_budget_sec: float = 120` (no retry starts later than this after the call's first attempt)
- `request_timeout_sec: float = 180` (connect/read timeout per attempt; read = the longest gap between bytes)
- `stream: bool = True`
- the descriptive `active_params_b: float | None` and `total_params_b: float | None`

`ModelConfig.price_in`/`price_out` default to None ("unknown"): calls of a
model without prices record `cost_usd: null` rather than 0 (write `0.0` for a
model that really is free). Prices only feed the run-time estimate; the study
recomputes cost offline (section 6).

`ExperimentConfig.to_dict()` and `from_dict()` must round-trip every field.
`from_dict` is strict: an unknown key or a bad enum/range value raises
`ConfigError` naming the dotted field (so a typo in a sweep override fails
instead of being ignored), and an explicit `null` for a non-Optional field
means "use the default". `subagents-bench validate` checks experiment configs,
sweep specs (every resolved condition) and the pricing/models/serving YAMLs.

## 3. LLM clients (`llm.py` + `credentials.py`, owner: llm)

`make_client(cfg) -> LLMClient` (provider defaults such as `base_url` are
filled by `resolve_model_config(cfg)` on a copy). A client has `.cfg` and
`.complete(system: str, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse`.
It raises `BillingError`, `ContextLengthError` or `LLMError`. Clients are
thread-safe: one instance is shared by concurrent subagent threads.

`OpenAIClient` (sail, prime, vllm, openai):

- **Transport and streaming.** Use `httpx.Client` directly, without the OpenAI
  SDK, and stream SSE. Send `stream_options.include_usage=true`. Assemble
  content, `reasoning_content` (or `reasoning`) and tool calls, where tool-call
  name and argument fragments are keyed by `index`.
- **Timing.** Record every `CallTiming` instant. Record chunk offsets (capped)
  for delta-carrying chunks only.
- **Headers.** Send `Authorization`, `User-Agent: subagents-bench/<version>`,
  `X-Sail-Completion-Window` when `completion_window` is set,
  `X-Prime-Team-ID` for provider prime, and `extra_headers`. The Prime team id
  comes from `PRIME_TEAM_ID` or `~/.prime/config.json` `team_id`. The Prime API
  key comes from `PRIME_API_KEY` or `~/.prime/config.json` `api_key`. The Sail
  key comes from `SAIL_API_KEY`. `credentials.load_dotenv(path=".env")` loads
  keys without overriding existing environment variables.
- **Message mapping.** Assistant messages carry `tool_calls`. Tool results
  become `role: tool` messages. If `reasoning_passback` is set and a message
  has `reasoning`, add `reasoning_content`.
- **Retries.** Retry on 408, 409, 425, 429, 500, 502, 503, 504, 529, transport
  errors, and streams that end without any finish_reason or `[DONE]`. Use
  exponential backoff from 2 s, doubled per attempt up to 15 s, with full
  jitter. Honor `Retry-After`. Stop after `max_retries`, or as soon as the
  next attempt would start more than `retry_budget_sec` (120 s) after the
  first: **fail fast**. A provider outage then surfaces in about two
  minutes, not after many. The numbers come from the 2026-10-11/12 runs:
  Sail's failed attempts ended after ~28 s (max 60 s, all "chat completion
  failed" stream errors); 4 of 314 calls needed a retry; the two calls that
  never recovered spent 407 s and 533 s on 9 attempts under the old limits
  (8 retries, 120 s backoff cap, 600 s read timeout); one call recovered
  only after 380 s. The 180 s read timeout is 2.5x the longest gap between
  bytes seen (73 s, a Gemma reply sent as one chunk; DeepSeek's longest was
  11 s). Record every `Attempt`, with the backoff stored on the failed
  attempt.
- **Billing errors.** HTTP 402, or a 4xx other than 429 whose body mentions
  balance/billing/credit/quota/payment, raises `BillingError` immediately.
  A 429 is billing only with an explicit billing marker
  (`insufficient_quota`, billing, balance, payment, credit), as OpenAI-style
  quota exhaustion arrives; any other 429, including a per-minute limit
  worded "quota exceeded", is a rate limit and is retried honoring
  `Retry-After`.
- **Context-length errors.** An HTTP 400 or 413 whose body (with `_` read
  as a space) names the overflow explicitly raises `ContextLengthError`:
  "context length", "context window", "maximum context", "too long"
  (e.g. "prompt is too long"), "reduce the length", "maximum model length"
  (vLLM's engine-level check), or a max-tokens limit
  (`max_tokens`, `max_new_tokens`, `max_completion_tokens`) together with
  "input". Generic words such as "maximum" or "tokens" alone are not enough,
  so unrelated validation errors stay plain `LLMError`s.
- **Provider failures.** Any other error is a plain `LLMError`: retries
  exhausted, a non-retryable 4xx such as 401/403, or a missing API key. It
  means the provider did not serve the call. The runtime counts such calls
  in `EpisodeRecord.n_provider_errors` (their `CallRecord.error` starts with
  `LLMError:`), and the first one, in a call of any role, ends the episode
  at once with `exit_reason="provider_error"` (section 4).
- **Errors carry the request.** `LLMError` (and its subclasses) has
  `attempts`, `status` and `request` (the `RequestInfo` of what was sent), so
  a failed call's `CallRecord` has the same request hashes as a successful
  one.
- **Attempt status.** `Attempt.status` is the HTTP status received; a stream
  that breaks after the headers keeps it (e.g. 200 with `error_type`
  `ReadError`). It is None only when no response arrived, so transport
  failures are counted by `error_type`.
- **Finish reason.** A stream that ends with `[DONE]` but no finish_reason
  gets `finish_reason="unknown"`.
- **Streamed tool-call repairs.** Sail re-sends each finished tool call in
  full under a second `index` with the same `id`. The accumulator routes
  deltas by id, drops such re-sends and exact duplicates, and renames
  clashing ids; `CallRecord.stream_repairs` counts the repairs per call
  (`EpisodeRecord.n_stream_repairs` per episode, `episodes.csv`
  `n_stream_repairs`).
- **Usage.** Take it from the final usage chunk: `prompt_tokens`,
  `completion_tokens`, `prompt_tokens_details.cached_tokens` and
  `completion_tokens_details.reasoning_tokens`. If the stream carried no usage,
  set `usage_source="estimated"` with ceil(chars/4) for prompt and completion,
  and `usage_raw=None`. A usage `cost` field is copied to
  `provider_reported_cost` and is never used for `cost_usd`.
- **Cost.** `cost_usd` comes from the config price table: uncached ×
  `price_in` + cached × `price_cache_read` (or `price_in`) + completion ×
  `price_out`.
- **RequestInfo hashes.** The hashes are sha1[:12] over the canonical JSON
  (sort_keys) of each message as sent.
- **Headers kept.** Response headers are stored minus `set-cookie`.

`AnthropicClient` keeps working with the same `LLMResponse`. It uses streaming
(`messages.stream`) to fill timings. If it doesn't stream, it sets
`streamed=False` and leaves the instants None. Usage mapping:
`prompt = input + cache_read + cache_creation`, `cached = cache_read`.

Background measured on 2026-10-11:
- Sail streams only about 1–8 coarse chunks per response; some answers arrive
  in a single chunk (TTFT = total latency). Balanced-tier latencies clustered
  near 1.8 s and 3.0 s, which suggests the tier releases responses on a
  schedule; keep this in mind for the latency simulation.
- Sail ignores passed-back reasoning (prompt tokens are unchanged).
- Sail's DeepSeek cached-token counts were 256, 312 and 380 in live runs: not
  always multiples of 256.
- Sail sends no request-id header; Prime sends `x-request-id` and
  `x-inference-id`.
- Prime reports no reasoning_tokens. Its usage carried no `cost` field in the
  live runs (it may also send `cost: 0.0`), so `provider_reported_cost` is
  often None.
- Sail streams have occasionally omitted usage.
- `chat_template_kwargs: {thinking: false}` disables DeepSeek reasoning on Sail.

## 4. Agent runtime (owner: runtime)

### Episode recorder (`telemetry.py`)

An `EpisodeRecorder` owns the episode directory and writes every file in
`schema.py`. It appends and flushes each JSONL line under one lock. It
allocates the global `seq` counter, plus an `agent_seq` counter per agent.

- It writes `episode.json` at start (`status="running"`) and at the end
  (`status="done"`) with all aggregates computed from the recorded events.
- It exposes `role_totals()` for harbor's `AgentContext`.
- The old `Telemetry` name stays importable (an alias or thin wrapper) so
  `exploitbench_agent` and older callers keep working.
- `seq` and `agent_seq` are 0-based and taken when an event starts; exactly
  one record is written per allocation (failed and cancelled events
  included), so `agent_seq` is contiguous per agent. A candidate's debate
  calls continue that candidate's `agent_seq`.
- Episode dicts keyed by role (`n_calls`, `n_tool_calls`, `tokens`,
  `cost_usd`, `llm_sec`, `tool_sec`) always have the four long role names
  (`orchestrator`, `subagent`, `arbiter`, `debate`); `cost_usd` also has
  `total`. A cost total is None if any of its calls was unpriced.
- The runtime's own tools are recorded as `ToolRecord`s too: `finish`,
  `submit` and `spawn_subagents` (whose duration is the time the
  orchestrator blocked on the group). Episode `n_tool_calls` and `tool_sec`
  include them; `DispatchRecord.n_tool_calls`/`tool_counts`/`n_tool_errors`
  exclude `submit` (its ~0 s is in `DispatchRecord.tool_sec`).
- A call that raised, or was abandoned when the episode stopped, is
  recorded with usage 0, `usage_source="none"`, `finish_reason="error"` and
  `error` set (`"cancelled: ..."` for an abandoned call, whose real usage is
  unknown).
- `harness_exec_n`/`harness_exec_sec` count the environment execs the harness
  makes for its own measurements (workdir markers, modified-file scans,
  `git diff --stat`). They are on no agent's chain, so they explain most of
  `overhead_sec` (section 6). `MeteredEnv` records each exec's interval;
  `harness_exec_sec` is the union of the intervals (wall-clock covered by at
  least one exec, so the overlapping markers and scans of parallel
  dispatches and arbiter candidates count once) and `harness_exec_sum_sec`
  is the plain sum. Records before `schema_version` 3 stored the sum in
  `harness_exec_sec`.
- `n_provider_errors` counts calls of any role whose `error` is a provider
  failure (`LLMError: ...`, section 3); billing, context-length and
  cancelled calls are not counted.
- Provenance (`provenance.py`): `source_sha256` is a sha256 over the sorted
  (relative path, sha256 of contents) of `subagents_bench/`, `configs/`,
  `scripts/`, `pyproject.toml` and `uv.lock`, listed with `git ls-files -co
  --exclude-standard` (or by walking those paths outside git; `__pycache__`
  and `.pyc` skipped either way). Equal fingerprints mean identical code,
  committed or not, with or without git. `git_commit` is HEAD (None outside
  a git work tree, and when the work tree tracks none of the package, as
  for a copy unpacked inside an unrelated repository); `git_dirty` is `git
  status --porcelain --untracked-files=no` restricted to those paths, so
  stray notes at the repo root do not mark every run dirty. Computed once
  per process, when harbor imports `harness/harbor_agent.py` (each harbor
  job is a fresh process importing the code on disk), so an edit between
  that import and the first episode is not attributed to the episodes.
- `validate_record` accepts the absence of a field added in a later schema
  version (`_f(..., since=N)`) from a record whose `schema_version` is older,
  so episodes written by older code still validate.

### Environment (`environment.py`)

```python
exec(command, timeout_sec=None, cwd=None) -> ExecResult(stdout, stderr, return_code, duration_sec, timed_out)
```

`return_code` is None when no exit status was seen (the command was killed on
the host side).

**HarborEnvironment**
- Holds a reference to harbor's running event loop, captured in
  `OrchestratorAgent.run` with `asyncio.get_running_loop()`.
- Each `exec` from a worker thread calls
  `asyncio.run_coroutine_threadsafe(env.exec(...), loop).result()`.
- Once the episode's stop event is set, `exec` raises `AgentStopped`.

**Workdir.** `sandbox.workdir: auto` means: run `pwd` once with `cwd=None`
(harbor uses the task's `WORKDIR`) and use the result
(`runner.probe_environment`, used by both the harbor agent's `setup()` and the
local runner, assigns it to `env.workdir`, probes `uname -m` and caches
`has_timeout_binary`). Tool specs come from `tools.specs_for(names,
command_timeout_sec=..., max_command_timeout_sec=...)` so the bash
description states the real default and maximum timeouts.

**Command wrapper.** Each bash command is wrapped so that cwd and exported env
**persist per agent**, and a runaway command is **killed inside the
container**:
- Write the user command base64-encoded to `/tmp/.sb/<agent_id>/cmd_<n>.sh`.
- Run a wrapper that restores `cwd` and `env` from `/tmp/.sb/<agent_id>/` and
  traps EXIT to save `pwd` and `export -p`.
- Run it under `timeout -k 5 <T> bash <file>` if `command -v timeout`
  succeeds. Check this once at setup and record it as `has_timeout_binary`.
- The host-side exec timeout is T+15 s.
- Exit code 124 or 137, or a host-side timeout, sets `timed_out=True`.

**LocalDockerEnvironment and LocalShellEnvironment** use the same wrapper
logic, so the tests cover it.

### Tools (`tools.py`)

Every tool is `fn(ctx, args) -> ToolOutput(text, is_error, exit_code, timed_out, timeout_sec, output_chars_raw, truncated, cwd_before, cwd_after, n_exec, exec_sec)`.
`ctx` carries env, agent_id, budget and truncation limit.

- **`bash`**
  - Takes `command` and an optional `timeout_sec`.
  - Default timeout is `budget.command_timeout_sec`; the cap is
    `max_command_timeout_sec`.
  - On a timeout, the text says `[command timed out after Ts; it was killed]`.
  - The description must be honest: cwd and exported variables persist
    between your calls; commands run non-interactively without a TTY; a
    long-running server must be started with `nohup ... > log 2>&1 &`.
- **`read_file(path, start_line=1, end_line=None)`**: numbered lines, capped.
- **`list_dir(path=".", depth=2)`**: via `find -maxdepth`.
- **`search(pattern, path=".", glob=None)`**: rg, falling back to `grep -rn`,
  limited to 200 lines.
- **`edit_file(path, old, new)`**: exact unique replacement, or creates the
  file when `old` is empty. Relative paths resolve against the agent's
  current cwd.

Truncation keeps the head and tail, with a marker giving the elided character
count.

Tool sets:

| Agent | Tools |
|---|---|
| Orchestrator | `bash, edit_file, read_file, list_dir, search, finish` (+ `spawn_subagents` when topology is subagents) |
| Subagent, scope easy | `read_file, list_dir, search, submit` |
| Subagent, scope medium | easy + `bash` |
| Subagent, scope hard | medium + `edit_file` |

Tool registry API (`tools.py`):

- `TOOL_SPECS: dict[str, ToolSpec]`
- `TOOL_FNS: dict[str, Callable[[ToolContext, dict], ToolOutput]]`
- `ORCHESTRATOR_TOOLS: list[str]`, without `finish` or `spawn_subagents`,
  which the orchestrator adds itself.
- `SCOPE_TOOLS: dict[str, list[str]]` for easy, medium and hard, without
  `submit`.
- `run_tool(ctx, name, args) -> ToolOutput` never raises. Exceptions become
  `ToolOutput(is_error=True, error=...)`, except `AgentStopped`, which
  propagates.
- `PathRemap(src, dst)`, `CopyToolContext` (a `ToolContext` with a
  `path_remap`) and `copy_context(ctx, remap)`: path redirection for arbiter
  candidates in private copies. When `ctx.path_remap` is set, `run_tool`
  rewrites the arguments before running the tool and appends
  `[note: <src> was redirected to your private copy <dst>]` to the output:
  - `read_file`, `list_dir`, `search`, `edit_file`: an absolute `path` equal
    to or under `src` becomes the same path under `dst`;
  - `bash`: every literal occurrence of `src` in the command that is a whole
    path prefix is replaced by `dst`. "Whole" means it is not preceded by a
    word or path character or a closing `}` or `)` (so not `/srv/app`,
    `~/app`, `$HOME/app`, `${HOME}/app` or `$(pwd)/app`), and
    it is followed by `/`, whitespace, a quote, `\` or one of
    `` ; & | ( ) < > : , } ] `` or the end (so not `/apple`, `/app_data` or
    `/app.bak`). Quoted text and heredoc bodies are rewritten too, so a file
    written by a command can contain `dst`; the arbiter maps it back (see
    Arbiter).
  - Paths already under `dst` are left alone. `src = /` is never remapped.
    `path_remap.n` counts the calls whose arguments changed. The ToolRecord
    keeps the model's original arguments.

`ToolContext`/`ToolOutput` are defined in `types.py`. Every shell-backed tool
runs through the per-agent wrapper, so relative paths resolve against the
agent's persisted cwd (initially `ctx.initial_cwd`).

### Workspace helpers (`workspace.py`, owner: env-tools)

All of these are shell-only, using coreutils and findutils. Each returns None
on failure rather than raising, except for `AgentStopped`.

```python
resolve_workdir(env) -> str | None                # `pwd` with cwd=None
container_arch(env) -> str | None                 # uname -m
has_timeout_binary(env) -> bool
make_marker(env, name) -> str | None              # touch /tmp/.sb/markers/<name>; returns path
files_modified_since(env, marker, root, cap=200) -> tuple[list[str], int] | None   # excludes /tmp/.sb
dir_size_mb(env, path) -> float | None
copy_tree(env, src, dst) -> bool                  # cp -a src/. dst/
manifest(env, root) -> dict[str, str] | None      # relpath -> md5
changed_paths(orig_manifest, new_manifest) -> list[str]          # relpaths modified, added or deleted
diff_summary(env, orig_root, copy_root, cap_chars=4000) -> str   # changed files + diff -ru excerpt (falls back if no diff binary)
apply_copy(env, copy_root, dst_root, cap=200, *, base=None, only=None, relative=False) -> tuple[list[str], str | None]  # (applied files, error)
replace_in_tree(env, root, old, new, cap=200) -> list[str] | None  # literal old -> new in text files (grep -rlI --null -F + sed) and symlink targets
git_diff_stat(env, workdir) -> str | None
sb_root(env) -> str                               # harness state root: /tmp/.sb in containers, a private dir for LocalShell
```

Path conventions: `files_modified_since` returns absolute paths,
`apply_copy` reports absolute destination paths and `edit_file`'s
`ToolOutput.files_written` holds absolute normalized paths, so all three
merge directly into `files_modified` fields. `manifest` keys are paths
relative to its root. `apply_copy(..., base=<manifest>)` restricts the apply
to what changed relative to `base`; `only=<relpaths>` restricts it to those
files (with a pristine copy as `copy_root`, that restores just them);
`relative=True` returns relpaths, and `cap=None` returns all of them.

`Environment` exposes the attribute `stop_event: threading.Event | None`. When
it is set, `exec` raises `AgentStopped`. The constructor is
`HarborEnvironment(harbor_env, loop, workdir: str | None = None)`.

### Orchestrator (`orchestrator.py`)

- Runs `budget.orchestrator_steps` steps. Each step is one LLM call, then every
  requested tool executed in order.
- After the last tool result of each step, the runtime appends a note:
  `[step k/N complete; N-k remaining; dispatch budget left: D]`. It is
  appended to the content of the step's last tool result (not sent as a
  separate user message), so providers that require alternating roles
  accept the sequence. Subagents get `[note: you have 1 step left; call
  submit ...]` in the result of their second-to-last step.
- **Cut-off replies.** A reply that hit the output-token limit
  (`finish_reason` `length`) before any tool call does not end the
  orchestrator's episode or a subagent's dispatch as `no_tool_call`: the step
  is used, the reply stays in the history (as `[reply cut off at the output
  token limit]` when it had no visible text, since some APIs reject an empty
  assistant message), and a user message tells the agent to reason more
  briefly and call a tool (with the step note; a subagent before its last
  step also gets the last-step note). Checked live on Sail (2026-10-12): the
  cut-off arrives as `length` with no tool call (DeepSeek's had no visible
  text), and after the note both models called a tool. `calls.csv`
  `finish_reason` and `conditions.csv` `length_finish_frac_*` show how often
  it happens.
- Exit reasons: `finished`, `max_steps`, `no_tool_call`, `token_budget`,
  `timeout` (safety cap), `cancelled` (stop event), `billing_error` (re-raised
  after logging), `provider_error`, `error`. A provider failure that survived
  every retry (a plain `LLMError`, section 3) in a call of ANY role
  (orchestrator, subagent, judge, debate) ends the episode at once with
  `provider_error`: such a trial cannot measure the condition (the analysis
  marks it invalid), so the recorder's `on_provider_error` hook sets the
  stop event, every agent unwinds (in-flight siblings end `cancelled`), and
  nothing more is spent on it. `episode.error` keeps the exception class and
  HTTP status of the first failure, and `EpisodeResult.cause` the exception.
  It is not re-raised by the
  orchestrator: the harbor agent raises `ProviderUnavailableError` after
  finalizing, so the trial is invalid and is rerun rather than graded. A
  `ContextLengthError` of the orchestrator (its context outgrew the model) is
  an outcome of the condition and ends it with `error`. Only `BillingError`
  and unexpected exceptions (recorded with their traceback) are re-raised.
  Tools listed after `finish` in the same response are not run.
- Writes a `StepRecord` for each step.

`spawn_subagents(instructions: list[str])`:

- **parallel**: granted = min(requested, max_concurrent, remaining dispatches).
  All granted subagents run concurrently in threads.
- **sequential**: the same grant rule, but subagents run one at a time, in
  order.
- **arbiter**: granted instructions = min(requested,
  max(1, max_concurrent // n_candidates), remaining // n_candidates). Each
  instruction gets `n_candidates` subagents, all running concurrently.
  Arbitrations for different instructions run concurrently.
- The result returned to the orchestrator lists each instruction's report.
  - Arbiter: the winning candidate's report, plus `(arbiter: candidate k of N,
    confidence c, candidates agreed: yes/no)`, plus the applied files. With
    copy isolation the report is the winner's original, pre-debate report,
    followed by its post-debate revision (if any) under a note (Arbiter,
    step 6).
  - A cap note appears when not every request was granted.
- `files_modified` for the group is measured with a marker file created at
  group start and `find <workdir> -newer marker -type f`. It excludes
  `/tmp/.sb`, caps the list and counts the total. A failure gives None.

### Subagent (`subagent.py`)

- The system prompt depends on scope. A medium-scope subagent is told not to
  modify project files, even if the instruction asks it to. It reports the
  exact change instead (a diff or the commands to run). See "Prompt policy"
  in section 1.
- It runs up to `max_steps` LLM calls, using the scope's tool set plus
  `submit(report)`.
- Status values: `submitted`, `no_tool_call` (the text is used as the report),
  `max_steps`, `context_overflow`, `error`, `cancelled`, and `token_budget` /
  `timeout` (before a step, the episode's `total_token_budget` or
  `timeout_sec` safety cap, or the subagent's own `token_budget`, was
  reached). A provider failure gives `error` (its `error` starts with
  `LLMError:`) and the failed call counts in the episode's
  `n_provider_errors`; the episode carries on.
- `files_modified` = its `edit_file` targets plus files under its start
  directory newer than a marker made when it started (skipped for easy
  scope, whose tools are read-only). With the parallel strategy the window
  also catches siblings' changes in the shared workdir (an
  over-approximation); it is exact for sequential runs and private copies.
- A `DispatchRecord` is written when it ends.

**Context guard** (applies when the model has `context_window`): before each
call, estimate prompt tokens as the last reported prompt_tokens plus
chars_added_since/3. If estimate + max_tokens > 0.92 × context_window, elide the
oldest tool-result contents (keeping the latest two) to
`[output elided to fit context: N chars]` until the estimate fits. Count each
elision pass as one `context_trims` and record `context_trimmed_chars` on the
next call. On `ContextLengthError`, do one aggressive trim (keep only the
latest tool result) and retry once; if it fails again, the status is
`context_overflow`.

### Arbiter (`arbiter.py`)

For one instruction:

1. **Isolation (hard scope).** If `du -sm workdir <= max_copy_mb`, take a
   manifest of the pristine workdir, then copy the workdir (`cp -a`) to
   `workspace.sb_root(env)/arb/<arbitration_id>/base` and to
   `.../cand<k>` for each candidate (`/tmp/.sb/arb/...` in containers).
   Each candidate's cwd starts in its copy, and its prompt
   (`PRIVATE_COPY_NOTE`) says: your private working copy of Y is X; make all
   edits there; paths under Y are redirected to X; changes made in Y by
   other means are undone; only Y is copied, and anything outside it is
   shared and changes immediately. If the workdir is too big, use
   `shared_fallback`. Easy and medium scopes use `shared`.
2. **Candidates.** Run the candidates concurrently. Each is a normal
   subagent with `candidate_index` set. **Prevent:** with copy isolation
   its `ctx` is a `CopyToolContext` whose `PathRemap(workdir, copy)`
   redirects file-tool paths and whole-path occurrences in bash commands
   from the real workdir to its copy (section "Tools"). The arbitration
   sums the redirected calls into `n_path_remaps`.
3. **Contain** (copy isolation, as soon as all candidates are done).
   Re-hash the real workdir. Every file that changed since the pristine
   manifest is a leak: a candidate reached the real workdir by a route the
   remap could not see, such as a path assembled at run time or a
   background process. Leaks are restored from the base copy
   (`apply_copy(base, workdir, only=leaked)`: modified and deleted files
   come back, added ones are removed). They are recorded as
   `isolation_violated` and `leaked_files`, and a failed restore goes to
   `apply_error`. The real workdir is thus pristine again before anything
   is applied. Files that a concurrent arbitration of the same episode
   applied meanwhile are not leaks. A per-episode journal lists every
   arbitration's applied files, and one lock serializes isolation setup,
   containment and apply. Then, inside each candidate's copy, mentions of
   the copy's own path (in text files and symlink targets, e.g. written by
   a remapped command) are mapped back to the workdir path
   (`replace_in_tree`), so the copy describes the files as they will be in
   the real workdir.
   **Known limitation:** only the workdir is isolated. Writes outside it
   (`/etc`, `/var`, installed packages, running services, e.g.
   nginx-request-logging's `/etc/nginx`) go live into the shared container
   from every candidate, concurrently. They are neither detected nor
   undone, and a remapped bash command that writes outside the workdir
   leaves the (later deleted) copy path in that file. Empty directories a
   leak created are not detected. Conversely, a process started before the
   arbitration that keeps writing into the workdir (e.g. a server logging to
   a file there) counts as a leak, and its writes during the candidate phase
   are undone.
4. **Judge.** One arbiter-model call. It receives the instruction, each
   candidate's report, and (with copy isolation) a compact diff summary per
   candidate. The summary compares the candidate's copy against the
   pristine **base copy**, not the live workdir, so a leak never shows up
   as a spurious change. It lists the changed files plus a `diff -ru`
   excerpt, capped at 4000 chars per candidate. Reports shown to the judge
   and to debaters are capped at `subagent.max_output_chars` (head and tail
   kept). It must call the tool
   `submit_judgement(winner: int, confidence: number 0..1, agreement: boolean, rationale: string)`.
   If the call has no tool call, parse JSON from the text. If the reply was
   cut off at the output limit (`finish_reason` `length`) without a complete
   judgement, the judge gets exactly one retry: the same input plus a note to
   reason briefly and call `submit_judgement` right away (the round's
   `judge_retry_call_id`; the better-ranked of the two judgements counts, and
   both calls are on the critical path). Before this, with an 8192-token
   limit, 2 of 5 smoke-run judge calls ended this way. If parsing still
   fails, the round's result is winner=0, confidence=None and
   `parse_ok=False`.
5. **Confidence-aware debate** (arXiv 2602.12276). If not
   (agreement and confidence ≥ threshold) and rounds remain, run a debate
   round. Each candidate's model gets one tool-less call with its own report
   plus the others' reports and returns a revised report (role `debate`,
   agent_id = candidate's agent_id). The candidates' calls run concurrently.
   `DEBATE_SYSTEM` tells debaters that nothing can change any more and not
   to claim commands, edits or results they did not make. With copy
   isolation it adds that their copy's files are frozen and are what gets
   compared and applied. Then judge again. The round-2+ judge prompt says
   the revisions were written without tools. With diffs shown, it also says
   the file changes are unchanged since the candidates finished and are
   what would be applied, so they outweigh claims in the revised reports.
   **Which judgement counts:** the latest complete judgement (`parse_ok`),
   else the latest one with a parseable winner. A later round whose judge
   call fails (`LLMError`; judging then stops) or whose reply does not parse
   never replaces it. `final_from_round` records the round. With no usable
   round, the winner is the fallback 0 and `final_from_round` is None.
6. **Apply.** With copy isolation, apply the winner's copy to the real
   workdir, restricted to what the winner changed relative to the pristine
   manifest: `workspace.apply_copy(winner_copy, workdir, base=<pristine
   manifest>)` compares `md5sum` manifests (via `find -type f -print0 |
   xargs -0 md5sum`), copies changed and added files with `cp -a` and
   deletes files the winner removed. (Without `base`, two concurrent
   arbitrations would delete each other's applied files.) Record
   `applied_files` and add them to the episode's journal. Then the copies
   (and the base) are removed.
   **What the orchestrator is told:** with copy isolation, the winner's
   ORIGINAL report, because that is what its applied files reflect. If a
   debate revised it, the revision follows under `POST_DEBATE_NOTE`
   ("[post-debate revision, written without tools ... No files changed after
   the candidates finished ...]"), and `POST_DEBATE_HEADER` ("[original
   report first; its post-debate revision, written without tools, follows at
   the end]") leads the text. The header is there because the orchestrator's
   per-report cap keeps only head and tail: with real-size reports (smoke
   fix-git: a 7312-char report and a 7285-char revision against a
   12000-char cap) the note in the middle is elided. With shared isolation
   the orchestrator gets the winner's final (possibly revised) report.
7. **Record.** Write an `ArbitrationRecord`. The judge calls have role
   `arbiter`, agent_id `arbiter-<arbitration_id>`, and step = round. A
   debate call is recorded under the candidate's agent_id with role
   `debate`; its transcript index restarts at 0 each round. Timing:
   - `isolation_sec`: step 1, before `t_start` (it includes the pristine
     manifest and the base copy);
   - `t_start`..`t_candidates_end`: step 2;
   - `t_candidates_end`..first round `t_start`: containment, copy-path
     mapping and diff summaries;
   - the rounds;
   - the end of the last judge or debate call..`t_end`: apply and cleanup
     (a round aborted mid-debate ends with debate calls and no judge).

   All of it is the strategy's own cost and lies on the critical path
   (section 6). A candidate's `DispatchRecord.files_modified` lists real
   paths inside its copy. Leaks into the real workdir appear only in the
   arbitration's `leaked_files`, not in any candidate's dispatch record:
   the candidates run at the same time in one container, so a change to
   the shared workdir cannot be attributed to one of them.

### Harbor agent (`harness/harbor_agent.py`)

- `OrchestratorAgent(BaseAgent)` accepts harbor's constructor kwargs
  (`logs_dir`, `model_name`, `logger`, ..., `**kwargs`) and calls
  `super().__init__`.
- The config comes from `SUBAGENTS_BENCH_CONFIG` (path) or
  `SUBAGENTS_BENCH_CONFIG_JSON`.
- The episode dir is `self.logs_dir` (`<trial>/agent/`).
- `task_name`/`trial_name` come from the logs_dir parent name
  (`<task>__<suffix>`). `harbor_session_id`/`harbor_trial_id` come from
  `self.session_id`/`self.context_id`.
- `run()` captures the loop, builds the `HarborEnvironment`, and runs the
  orchestrator with `await asyncio.to_thread(...)`.
- On `CancelledError` (harbor's agent timeout):
  1. set the stop event;
  2. wait, shielded, up to 30 s for the thread to unwind;
  3. kill the agent's commands still running in the container. Cancelling
     a `docker exec` client never stops the process inside, and harbor's
     shared-mode verifier grades the same container next. One exec through
     harbor's raw environment (the `HarborEnvironment` wrapper refuses execs
     once stopped), in its own task so a second cancel cannot interrupt it,
     timeout 20 s: for each `<state_root>/*/pid_<nonce>` file the bash
     wrapper wrote (`tools._WRAPPER`; removed when the command ends) whose
     process, or its parent (`timeout`, or the outer script, when the
     command `exec`'d away from the wrapper), is still running
     `wrap_<nonce>.sh`, SIGKILL its process group (unless
     it is the kill's own or init's) and its process tree (from /proc, or
     `ps` outside Linux). The outcome goes to `episode.cancel_kill`
     `{attempted, ok, n_killed, error}`;
  4. finalize the record with `exit_reason="cancelled"` (if the thread
     finalized first, its record is kept and only `cancel_kill` is added);
  5. re-raise.
- It fills `AgentContext` (`n_input_tokens`, `n_cache_tokens`,
  `n_output_tokens`, `cost_usd`, `model_usage` per model, and `metadata` with
  the episode summary keys) **progressively** after every LLM call and at the
  end.
- **BillingError** is logged with `exit_reason="billing_error"` and re-raised,
  so harbor records the exception.
- **Provider failure**: after an episode that ended with
  `exit_reason="provider_error"` is finalized, `run()` raises
  `ProviderUnavailableError` (an `LLMError` carrying the cause's status and
  attempts). Harbor records it as the trial's exception and skips the
  verifier, instead of grading a container the task was never attempted in.
- It records `container_arch` (`uname -m`), `has_timeout_binary`, the harbor
  version, the source fingerprint, git commit and dirty state, and the host
  platform.

### Local runner (`runner.py`)

`run_task(config, task_id, instruction, env, resolved=None)` writes to
`<output_dir>/<name>/<task_id>/<episode_id>/` (or `episode_dir=`, e.g. a
harbor-style `<trial>/agent/`; `trial_name=` goes into `episode.json`). Every
trial gets its own directory; trials never overwrite each other. `resolved`
goes into `episode.json` reward/resolved.

## 5. Sweep launcher (`sweep.py`, owner: sweep)

A sweep spec (YAML):

```yaml
name: pilot
extends: configs/sweeps/pilot.yaml   # optional: deep-merge this spec over a parent spec (smoke reuses pilot's conditions)
dataset: terminal-bench@2.0
tasks: [fix-git, ...]            # explicit list (or task_filter: {difficulty: [...], category: [...], max_expert_min: N})
rounds: 3                        # trials per (task, condition); each round runs every condition once
base: configs/sail/base.yaml     # base ExperimentConfig
conditions:                      # name -> deep-merged overrides
  single_5step: {topology: single}
  ...
harbor:
  env: docker                    # or daytona/modal/prime...
  n_concurrent: 4
  agent_timeout_multiplier: 4.0
  extra_args: []
jobs_dir: jobs/pilot
order_seed: 0                    # condition order shuffled per round with this seed (interleaving)
```

Relative paths in a spec (`extends`, `base`, `jobs_dir`) resolve against the
repository root (the nearest `pyproject.toml`). Sweep and condition names
use letters, digits, `_`, `.` and `-` and never contain `__`, which separates
the parts of a job name `<sweep>__r<round>__<cond>`.

Commands:

- `subagents-bench preflight SPEC [--oracle] [--redo-oracle]` checks:
  - Docker is up;
  - harbor is importable and the dataset is downloadable;
  - every task exists, with its metadata written to `<jobs_dir>/tasks_meta.json`;
  - API keys resolve;
  - one tiny request per distinct (model, provider, tier) succeeds, recorded
    in `<jobs_dir>/preflight.json`;
  - optionally, `harbor run -a oracle` on all tasks passes, so tasks are
    known feasible on this machine. An earlier `_oracle` job is reused only
    if it finished and passed on every task; an unfinished or failing one
    (possibly a transient image pull or memory problem) is moved to
    `_partial/` and rerun, and `--redo-oracle` reruns a passing one too.
    `--oracle` holds the jobs dir's sweep lock (below).

  It exits nonzero on any failure.
- `subagents-bench sweep SPEC [--dry-run] [--only COND ...] [--rounds 1,2]
  [--allow-config-change] [--allow-task-change] [--allow-code-change]`:
  - Takes an exclusive, non-blocking `flock` on `<jobs_dir>/.sweep.lock`
    (holder pid, host, invocation id, argv, start time) for the whole run,
    and passes the descriptor to each harbor job, so the lock also outlives
    a sweep process killed while its job runs. A second invocation on the
    same jobs dir exits 2 naming the holder: it would otherwise see the
    running job as partial, move it aside and relaunch it while the first
    harbor keeps writing trials into the new dir. The kernel drops the lock
    when the holder exits, so a stale file never blocks.
  - Materializes resolved configs to `<jobs_dir>/configs/<cond>.yaml`.
  - Then, for each round, for each condition in the shuffled order, runs:

    ```bash
    harbor run -d <dataset> -i <task>... -a subagents_bench.harness.harbor_agent:OrchestratorAgent \
      -e <env> -n <n_concurrent> -k 1 -o <jobs_dir> --job-name <name>__r<round>__<cond> \
      --agent-timeout-multiplier <m> -y -q
    ```

    The subprocess environment includes `SUBAGENTS_BENCH_CONFIG=<resolved yaml>`
    plus `.env` and the Prime credentials.
  - Job states (`sweep.job_status`). *Finished* means: the job's
    `result.json` parses, `n_completed_trials >= n_total_trials` (harbor
    0.24.0 counts errored trials as completed), nothing is running or
    pending, `finished_at` is set, and every trial dir has its own
    `result.json`. A finished job is `complete` (skipped, which makes the
    sweep resumable) unless a trial needs a rerun (`needs_retry`). An
    unfinished one is `partial`. One with more trial dirs than
    `n_total_trials`, or a task more often than its attempts (harbor
    `config.json` `n_attempts`, default 1), is `duplicate`: two harbor runs
    wrote into it. It is reported, logged as `duplicate_trials`, counted as
    failed and not run; the operator moves it out of the jobs dir.
  - A trial needs a rerun (`sweep.trial_retry_reasons`) when its outcome
    does not measure the condition: `billing` (episode `exit_reason`
    `billing_error` or harbor exception `BillingError`), `provider`
    (`exit_reason` `provider_error`, `n_provider_errors > 0` or harbor
    exception `ProviderUnavailableError`; for records before schema 3, an
    `LLMError:` error in episode.json or calls.jsonl), `infrastructure`
    (harbor's exception happened outside the agent: the analysis's
    `OUTSIDE_AGENT_EXCEPTIONS`, or before the agent started) or
    `agent_exception` (the agent raised and the verifier never ran: a
    harness bug). An agent timeout is an outcome: harbor still verifies.
  - A `partial` job is moved to `<jobs_dir>/_partial/<job>.partial.<k>`, a
    `needs_retry` job to `<job>.retry.<k>` (a `retried` manifest line lists
    each trial's reasons), and the whole job is rerun. Nothing is deleted.
    Rerunning the whole job keeps every trial of a job from one harbor run
    and one code version, but it discards the job's valid trials too, and a
    job is complete only when every trial is clean in a single run: with
    per-trial failure rate p, a 10-trial job needs 1/(1-p)^10 runs on
    average (about 2.9 at p = 0.1). Rerunning only the failed trials (move
    their dirs aside, then `harbor run` again into the job dir, which harbor
    0.24.0 resumes by trial config) would converge per trial; it is not
    implemented because harbor's `lock.json` check on such a resume is
    untested here.
  - A resume refuses to run (exit 2) if it would mix different runs under
    one condition name: a condition's resolved config changed
    (`--allow-config-change`), the task set or the other `harbor run`
    arguments (all but the executable and the job dir's location) differ
    from those of any complete job of the sweep, all rounds and conditions
    (`--allow-task-change`; taken from the job's last manifest `run` line
    `cmd`, else the tasks from harbor's job `config.json`), or the
    harness source fingerprint differs from the one any complete job ran
    in a file episodes use (`--allow-code-change`; from the manifest `run`
    lines and every episode's `source_sha256`; the message names the
    changed files from `provenance/source_<sha16>.json`). Complete jobs
    with no recorded fingerprint (older launcher and runtime) only produce a
    warning.
  - Just before each job it fingerprints the source again (harbor imports
    whatever is on disk when the job starts). If code that episodes use
    changed during the invocation, the sweep stops (exit 2, status
    `aborted_code_change`) unless `--allow-code-change`.
  - Both code checks ignore edits confined to files no episode imports or
    reads (`provenance.NON_RUNTIME_PATHS`: `subagents_bench/analysis/`,
    `plots/`, `sweep.py`, `cli.py`, `configs/` and `scripts/`; a test checks
    that the harbor agent's import closure stays outside them, and condition
    configs are checked separately through their materialized copies). Such
    an edit, e.g. to the analysis code while a 6-hour pilot runs from the
    same checkout, only prints a NOTE mid-sweep or a WARNING on resume and
    adds a fingerprint; a fingerprint without a provenance record (so its
    files are unknown) counts as a runtime change.
  - After each job it scans `*/agent/episode.json` for
    `exit_reason == billing_error` and aborts the sweep if found (exit 3).
    It aborts with exit 4 (`aborted_provider`) when provider failures hit
    more than half of the job's trials (an outage or a rejected key; the
    job is rerun on resume). A job that finished with trials to rerun counts
    as failed. It also aborts after 3 consecutive failed jobs; Ctrl-C gives
    harbor time to remove its containers.
  - It appends to `<jobs_dir>/sweep_manifest.jsonl`, one line per job
    event: `sweep, job, round, condition, event` (`run`, `skipped_complete`,
    `moved_partial`, `retried`, `duplicate_trials`, `interrupted`), `cmd,
    config_path, t_start, t_end, duration_sec, return_code, n_trials,
    n_completed, n_errors, complete`, and for `run` lines also `state,
    status_reason, n_episode_json, billing_error_trials, retry_trials,
    n_provider_error_trials, invocation_id, tasks, harbor, source_sha256,
    git_commit, git_dirty`
    (the code state when the job started); `moved_to` for `moved_partial`
    and `retried`, plus `retry_trials` and `status_reason` for `retried`.
    round/condition agree across a job's lines; consumers join on `job` and
    take outcomes from the last `run` line.
  - It writes `<jobs_dir>/sweep.json`: the resolved spec and the hashes of
    its files, the base config hash, `condition_order` per round, each
    condition's materialized config path and sha256, the provider tiers,
    provenance (`source_sha256`, git commit, `git_dirty` for tracked source
    files only, untracked source files, Python and harbor versions, lockfile
    and its sha256, Docker info, date, an invocation id, and the paths of
    the source records below), `source_sha256s` (the distinct fingerprints
    the sweep's jobs ran, from manifest `run` lines and episodes) with
    `mixed_source` when they hold more than one version of the code under
    test (`sweep.runtime_versions`: the fingerprints' records differ in a
    file episodes use), `mixed_git_commits`, and the history of
    invocations (each with its tasks and harbor settings).
  - Source records under `<jobs_dir>/provenance/`: `source_<sha16>.json`
    (per-file sha256 of the fingerprinted source, once per fingerprint) and,
    when tracked source differs from HEAD or untracked source files exist,
    `<invocation_id>.patch` (`git diff HEAD` of the source paths plus each
    untracked source file as a new-file diff; `git apply` on a checkout of
    `git_commit` restores the code that ran, which a test checks).
  - `<jobs_dir>/tasks_meta.json` is `{dataset, source_dir, generated_at,
    tasks: {name: {difficulty, category, expert_time_min, junior_time_min,
    agent_timeout_sec, verifier_timeout_sec, docker_image, tags}}}` for every
    task the sweep has run: a rewrite merges, so a task dropped from the
    spec keeps its entry for its trials.
  - `--dry-run` prints the plan (with each retry's reasons), the config
    validation, the resume checks, whether another sweep holds the lock,
    and a ROUGH cost and runtime estimate (`sweep.RoughModel`, calibrated
    from the smoke sweep of 2026-10-11/12: 24 all-tools episodes and the
    first 2 `manager_arbiter_hard` episodes; an orchestrator with all its
    tools is assumed not to delegate, as measured, and a manager to use 40%
    of its dispatches). It writes nothing, and exits 2 when a real run would
    refuse because of a config problem or a resume check. A lock held by
    another sweep is only a NOTE (exit 0): the dry run is how one inspects a
    sweep while it runs. `python -m subagents_bench.sweep` is an alternative
    entry point.
  - Exit codes: 0 every selected job complete, 1 a job failed or still needs
    a retry, 2 spec/config problem, refused resume or held lock, 3 billing,
    4 provider outage, 130 interrupted.
  - The oracle feasibility run of `preflight --oracle` writes to
    `<jobs_dir>/_oracle`. Analysis ignores every `_`-prefixed dir.
- `subagents-bench tasks DATASET` prints task metadata: difficulty, category,
  expert and junior minutes, agent timeout, docker image.

## 6. Aggregation (`analysis/` package, owner: analysis)

`subagents-bench aggregate JOBS_DIR [JOBS_DIR...] --out analysis/<name>` finds
every trial dir (one with `agent/episode.json` or a trial-shaped
`result.json`) and every bare episode dir (the local runner's layout). It
skips hidden and `_`-prefixed dirs (`_partial/`, `_oracle/`), which would
otherwise double-count trials. It joins:

- harbor's `<trial>/result.json` (reward, exception_info, phase timings);
- `tasks_meta.json`;
- `sweep_manifest.jsonl` (round, condition), joined on `job`, last `run`
  line; without a manifest the job name `<sweep>__r<round>__<cond>` is
  parsed, and as a last resort the condition is the episode's `experiment`.

It writes CSVs (plus `summary.json`), defined below. Every column is
described in `docs/DATA.md` (generated by `scripts/gen_data_docs.py`).

### `episodes.csv` (one row per trial)

- **keys:** `sweep, job_name, round, condition, trial_name, task_name, episode_id`
- **design:** `topology, strategy, scope, orchestrator_model, subagent_model, arbiter_model, max_concurrent, n_candidates, orchestrator_steps_budget` (also in `conditions.csv`)
- **task:** `difficulty, category, expert_time_min, junior_time_min`
- **outcome:**
  - `reward, resolved (0/1), harbor_exception, exit_reason, episode_status`
  - `valid` is False, with `invalid_reason` (the first that applies), when
    the trial does not measure its condition:
    - `no_episode` (no episode.json);
    - `billing_error`;
    - `harbor_exception_outside_agent` (an environment/setup error type, a
      `CancelledError`, or any exception before the agent started);
    - `provider_error`: the provider could not serve the episode. The
      orchestrator's call failed after all retries (`exit_reason ==
      provider_error`, harbor exception `ProviderUnavailableError`), or any
      call of any role did (`n_provider_errors > 0`, recounted from
      `calls.jsonl`, so a Gemma-only outage invalidates the subagent
      episodes it hit instead of scoring them as failures). Records before
      `schema_version` 3 ended an orchestrator provider failure with
      `exit_reason == error` and an `LLMError:` error, which counts too;
    - `agent_exception`: the agent itself raised (a harness bug, a crashed
      recorder) and harbor therefore skipped the verifier. Not
      `AgentTimeoutError`, after which harbor still grades the container;
    - `no_result` (a harbor trial without `result.json`).

    A verifier exception (the verifier phase started) or an agent timeout
    leaves the trial valid and unsolved (`resolved = 0`) unless graded.
  - `duplicate`: several trials of one (condition, sweep, task, round)
    (a re-run, or the same sweep passed twice) are all kept in
    `episodes.csv`, but only one, the valid one (else the first by trial
    name), enters `conditions*.csv` and `paired.csv`; `summary.json` lists
    the others and warns. Bare local episodes (no round) are never
    duplicates.
- **harbor phases:** `env_setup_sec, agent_setup_sec, agent_exec_sec, verifier_sec, trial_total_sec`
- **episode:** `wall_clock_sec, orchestrator_steps, spawn_calls, delegated (spawn_calls > 0), dispatches, redispatches, arbitrations, instruction_following, dispatch_submitted_frac, n_files_modified`
- **copy isolation** (from `ArbitrationRecord.isolation_violated` /
  `leaked_files`): `n_isolation_checked, n_isolation_violations,
  n_leaked_files`; 0 without arbitrations, NaN when the arbitration records
  predate the leak check.
- **tokens,** for role in `orch, sub, arb, deb`: `{role}_calls, {role}_prompt_tokens, {role}_cached_tokens, {role}_uncached_prompt_tokens, {role}_completion_tokens, {role}_reasoning_tokens`, plus `total_tokens`, and `{role}_length_finishes` (calls cut off at max_tokens)
- **cost**, for each pricing scheme S: `cost_{S}_{role}, cost_{S}_total`, plus
  `cost_runtime_total` (the logged cost_usd)
- **compute:** `flops_{role}`, `flops_total`, where flops = 2 × active_params × tokens
- **time:** `llm_sec_{role}, tool_sec_{role}, backoff_sec, n_retries, n_command_timeouts`
- **critical path** (measured): `cp_total_sec, cp_orch_llm_sec, cp_sub_llm_sec, cp_arb_llm_sec, cp_tool_sec, cp_group_sec, overhead_sec (= wall_clock_sec - cp_total_sec), harness_exec_sec (the harness's measurement execs, which explain most of overhead_sec), sub_busy_sec (sum of dispatch durations), parallel_speedup (sub_busy_sec / cp_group_sec; NaN without dispatches or group time; for arbiter groups the denominator also holds isolation, judging and apply)`
- **extra:** `n_failed_calls, n_provider_errors, n_usage_estimated, container_arch, host_machine`
- **simulated:** `sim_{P}_wall_sec` for each serving profile P. The profile
  `measured` replays logged durations and must come out close to
  `cp_total_sec`.

### Other tables

- `calls.csv`, `tools.csv`, `dispatches.csv`, `spawns.csv`,
  `arbitrations.csv`, `steps.csv`: the records flattened, plus the episode
  keys `condition, task_name, round, episode_id`. Bulky lists are dropped;
  `chunk_offsets_ms` is summarized.
- `conditions.csv` (per condition) and `conditions_by_difficulty.csv`, over
  valid, non-duplicate episodes:
  - `n, n_valid, n_resolved, resolve_rate, resolve_ci_lo, resolve_ci_hi` (Wilson 95%);
  - `n_tasks_valid` and `resolve_rate_task_balanced` (the mean over tasks of
    each task's rate, so a condition that lost some of a task's trials to
    invalid runs is not shifted by the task mix; a task with no valid trial
    left drops out, which `n_tasks_valid` shows);
  - treatment uptake: `n_delegated`, `delegated_frac` (valid episodes with
    `delegated == 1`), and the per-protocol `resolve_rate_delegated` with
    its Wilson CI (`resolve_delegated_ci_lo/_hi`) next to the
    intention-to-treat `resolve_rate`. The per-protocol rate is not
    randomized (episodes that delegate may be easier or harder);
  - for each of `wall_clock_sec`, `sim_*_wall_sec`, `cost_*_total`,
    `total_tokens`, `orch_*`/`sub_*` tokens and `dispatches`: the mean plus a
    bootstrap 95% CI (`_mean, _ci_lo, _ci_hi`; 2000 resamples, seed 0; a
    single value has a mean and NaN bounds);
  - `cost_per_solve_{S}` and `time_per_solve_sec`, plus
    `cost_per_solve_runtime` (`time_per_solve_sec` sums the wall clock of
    the valid trials where it is known and divides by the solves among
    those same trials);
  - `n_isolation_checked, n_isolation_violations, isolation_violated_frac,
    n_leaked_files` (copy-isolation leaks of the arbiter);
  - `length_finish_frac_{role}`: share of the role's calls cut off at
    max_tokens (relevant to the same-model control, whose subagents run
    with the orchestrator's output limit);
  - the design columns of `episodes.csv`.

  Estimand: every interval treats a condition's valid trials as independent
  draws, so it describes these tasks with their trial-to-trial randomness,
  not Terminal-Bench as a whole (the tasks are hand-picked; no interval
  supports that inference). Trials of one task are correlated, so for a
  condition-vs-baseline claim use `paired.csv` (matched trials, McNemar)
  rather than whether two intervals overlap.

  NaN policy for condition aggregates: cost, FLOPs, simulated-time and
  reasoning-token means are NaN if any valid episode of the condition is
  NaN (e.g. an unpriced model), rather than averaged over the priced
  subset; so are the isolation sums when any episode predates the leak
  check; other metrics drop NaN values.
- `paired.csv`: each condition vs each baseline (`single_5step`,
  `single_matched`) over tasks × rounds matched on (sweep, task, round),
  with `n_pairs, both, only_cond, only_base, delta_resolve, mcnemar_p`
  (exact binomial). Rounds restart at 1 in every sweep, so pairs are formed
  within a sweep and aggregating several sweeps adds their pairs. A
  baseline is not paired with itself.
- `calibration.csv`, per (model, provider, completion_window):
  - `n_calls`
  - `ttft_median`, `ttft_p90`
  - `decode_tok_s_median` (completion_tokens / decode_sec where decode_sec > 0)
  - `ttft_fit_a`, `ttft_fit_b_per_1k_prompt_tokens` (least squares)
  - `lat_fit_a, lat_fit_b_per_1k_prompt, lat_fit_c_per_completion_token,
    lat_fit_n`: the joint call-time fit the `fitted` profile uses (below);
    the TTFT fit and decode rate are descriptive
  - `latency_median`
- `summary.json`: counts (validity, exit reasons, harbor exceptions), and
  what the tables were computed from, so they can be traced after a config
  changes: every price used per scheme and model, the sha256 of each input
  file (pricing.yaml and its includes, serving.yaml and any fitted
  params_file, models.yaml), the analysis code's `source_sha256` and git
  state, and per condition the runtime's `source_sha256` values. Also the
  treatment coverage, the duplicate trials, the latency checks (below) and
  the warnings, which name every pricing scheme that lacks a price for a
  model that made calls (its costs are NaN).

### Pricing and serving profiles

- **`configs/pricing.yaml`**: schemes → model → `{in, cached, out}` USD per 1M
  tokens. The schemes are `sail_balanced`, `sail_asap`, `sail_flex`, `prime`
  and `reference` (the reference prices come from
  `configs/pricing_reference.yaml`, produced by the research step with
  sources and date). A scheme is either the flat form below or the nested
  form `{provider, completion_window, retrieved, source, notes, models:
  {...}}`, which records where the prices came from; `cached: null` charges
  cached tokens at the `in` price. `configs/models.yaml` holds per model:
  `active_params_b`, `total_params_b`, `context_window` and sources, plus
  optional keys the loaders use when present (`active_params_prefill_b`,
  `active_params_decode_b`, `bytes_per_param`) and descriptive ones they
  ignore. If a model has no price in a scheme, cost is NaN (never 0).
- **FLOPs** are `2 × active_params × tokens`, with prompt tokens at
  `active_params_prefill_b` and completion tokens at
  `active_params_decode_b` when models.yaml gives them (DeepSeek-V4.1-Flash:
  8B / 16B), else `active_params_b`.
- **`configs/serving.yaml`** holds the serving profiles:
  - `measured` (replay);
  - `fitted` (from calibration.csv: `lat_fit_a + lat_fit_b × prompt/1000 +
    lat_fit_c × completion`, one least-squares fit of the whole served call,
    `ttft_sec + decode_sec` (else `latency_sec − backoff_sec`), with every
    coefficient ≥ 0, per (model, provider, completion_window)). It removes
    run-to-run noise but not the provider's queueing. A TTFT fit plus one
    median decode rate, used before, has no per-call decode intercept, so it
    simulated short replies too fast and long ones too slow (on the
    2026-10-11 smoke calls ×0.80 under 100 completion tokens, ×1.11 at 400+),
    a bias that differed between conditions with their mix of reply lengths;
    single-chunk replies (whole time in TTFT) were also counted twice. The
    joint fit has an intercept, so in-sample its total equals the measured
    total (×0.98–1.03 per length bin on the same calls); `summary.json`
    `latency_checks.fitted_vs_measured` reports this per model and length
    bin. Params files written before the joint fit still load, as
    `c = 1 / decode_tok_s`;
  - `roofline_h100`: theoretical dedicated serving at batch 1. Defined as
    - `prefill_tok_s = mfu × 989e12 / (2 × active_params)`;
    - `decode_tok_s = mbu × 3.35e12 / (active_params × bytes_per_param)`;
    - `ttft = overhead_sec + prompt/prefill_tok_s`.

    The parameters (mfu, mbu, bytes_per_param, overhead_sec) are explicit in
    YAML and marked theoretical. `bytes_per_param` is taken from a per-model
    entry in the profile, else models.yaml's effective value (e.g. 1.026 for
    the NVFP4 Gemma, whose attention and embeddings stay BF16), else the
    profile default (2.0, BF16). Prefill and decode use the prefill/decode
    active-parameter split when given.
  - optional `table` profiles: per model `{ttft_base_sec, prefill_tok_s, decode_tok_s}`
    from a real local-GPU calibration.

File formats (the research step writes the first two; analysis only reads
them):

```yaml
# configs/pricing_reference.yaml
scheme: reference
retrieved: "2026-10-11"
method: "median USD/1M across OpenRouter provider endpoints"
models:
  deepseek-ai/DeepSeek-V4.1-Flash: {in: 0.0, cached: 0.0 | null, out: 0.0, n_providers: 3, providers: [...], source: <url>, openrouter_id: <id>}
# configs/models.yaml
models:
  deepseek-ai/DeepSeek-V4.1-Flash: {active_params_b: .., total_params_b: .., context_window: .., architecture: moe|dense, sources: [<url>...]}
# configs/pricing.yaml (analysis owns)
include: [pricing_reference.yaml]
schemes:
  sail_balanced: {deepseek-ai/DeepSeek-V4.1-Flash: {in: 0.07, cached: 0.005, out: 0.22}, ...}
```

Model ids are the provider model ids used in configs. Aliases (for example,
Prime's `deepseek/deepseek-v4.1-flash`) map through an `aliases:` block in
models.yaml or pricing.yaml (both directions).

### Latency simulation and critical path

Each agent is a sequential chain: its calls and tools ordered by `agent_seq`.
The duration of a chain is the sum of its event durations, where a
`spawn_subagents` tool's duration is replaced by the duration of its group:

| Group strategy | Group duration |
|---|---|
| parallel | max over member dispatch chains (all run concurrently: granted ≤ max_concurrent) |
| sequential | sum over member dispatch chains |
| arbiter | max over the group's arbitrations of: isolation setup (`isolation_sec`, measured) + max(candidate chains) + diff summaries (first round `t_start` − `t_candidates_end`, measured) + Σ over rounds of (judge call + max over that round's debate calls) + apply and cleanup (= t_end − end of the last judge or debate call counted, measured; a round aborted mid-debate has debate calls and no judge, and measuring from the previous judge would count that debate twice) |

The episode duration is the orchestrator chain. With `measured`, the
simulation must reproduce `cp_total_sec`, and that should sit close to
`wall_clock_sec`; the gap is reported as `overhead_sec`. Most of it is the
harness's own measurement commands (`harness_exec_sec`: markers,
modified-file scans, `git diff --stat`), which belong to no chain; the rest
is thread start-up and record writing. `harness_exec_sec` is the union of
those execs' intervals, so concurrent ones (parallel dispatches, arbiter
candidates) count once; for records before `schema_version` 3 it is the plain
sum, which over-explains the overhead of concurrent strategies by up to
(concurrent dispatches - 1) x 2 execs per group. `summary.json` reports the share of
episodes whose overhead, and whose overhead not explained by
`harness_exec_sec`, exceeds 10% of wall-clock. The unexplained part
(`overhead_sec − harness_exec_sec`) is not clipped at 0: its minimum and the
episodes where it is below −0.05 s (`episodes_overhead_over_explained`:
summed execs of old records, or execs that overlapped an agent's chain) are
reported. Even the union can slightly exceed `overhead_sec` for parallel and
arbiter groups: an exec of a subagent that finished early (its final
modified-file scan) runs while a sibling is still on the critical path, so
its time is already inside `cp_total_sec` (offline-pipeline episodes of those strategies
gave −0.003 to −0.020 s). The critical-path
decomposition attributes each chain element to `orch_llm`, `sub_llm`,
`arb_llm` (judge and debate) or `tool`.

`subagents-bench check-run DIR...` validates every record file against
`schema.py` and checks internal consistency:
- episode totals equal the sums over calls and tools;
- every dispatch_id in calls exists in dispatches.jsonl;
- `agent_seq` is 0-based and contiguous per agent;
- timestamps are monotonic per agent;
- spawn `dispatch_ids` match dispatches;
- the arbitration call ids exist.

It prints a report and exits nonzero on problems. Dangling references are
warnings, not errors, in an episode that never finalized or that harbor
cancelled (`exit_reason == cancelled`, e.g. at its agent timeout while a
spawn group was still running), since both legitimately leave them. It also cross-checks
`n_provider_errors` against the calls whose error starts with `LLMError:`
(schema_version ≥ 3), and prints each subagents condition's treatment
coverage (below) with a warning for any condition never treated.

`subagents-bench completeness DIR...` reports, for every schema field, the
fraction of records where it is None, empty or zero, by role. It flags fields
that are never populated, separating fields that are legitimately empty in
a healthy run (failure fields, retry counters, provider extras, role- or
strategy-specific fields); `--strict` exits 1 on any other never-populated
field.

Both also report **treatment coverage**: for each `topology: subagents`
condition, the episodes that delegated (`spawn_calls > 0`, counted from the
records), its dispatches and, for the arbiter strategy, its arbitrations.
`completeness --strict` exits 1 when a subagents condition has no delegated
episode or an arbiter condition ran no arbitration, and names it. Field
coverage alone cannot catch this: a record file nobody wrote has no fields
to flag. (The 2026-10-11 smoke sweep delegated in none of its
subagent-condition episodes and still passed every other check.) `summary.json` carries the
same rows as `treatment_coverage`.

## 7. Plots (`plots/` package, owner: plots)

`subagents-bench plot analysis/<name> --out figures/<name>` reads the CSVs
above. It writes PNG (dpi 200) and PDF for each figure, plus `index.md`, which
lists each figure with a one-paragraph caption of what it shows and how to
read it. Use matplotlib only, a consistent color per strategy and a hatch or
lightness per scope, show baselines as horizontal reference bands, and always
draw error bars from the CI columns. Each figure is skipped, with a log line,
when its inputs are empty. Cost per solve and time per solve have no CI
columns in `conditions.csv`; their whiskers come from a paired bootstrap of
the ratio over `episodes.csv` that the plotter computes (2000 resamples, seed
0), and the caption says so. The bootstrap keeps a resample that draws no
solved episode, with an infinite ratio, rather than dropping it (dropping
gives an interval conditional on a solve, tightest exactly when solves are
rare). The bounds are the outward order statistics, so when more than 2.5% of
the resamples have no solve the upper bound is unbounded. Such a whisker runs
to the right edge of the panel with an arrow head, and the notes list the
condition. Time per solve is bootstrapped over the same episodes as its
`conditions.csv` value (valid trials with a known wall-clock).

Rules that hold for every figure:

* An interval that is unknown (a NaN bound, e.g. fewer than 2 valid values)
  or degenerate (zero width) gets no whisker. A zero-length capped whisker
  would look exact. The figure notes list such conditions as "CI
  unavailable".
* A condition with no valid trial (every trial invalid) is never dropped
  silently. `main_grid` keeps its slot, labelled "no valid trials" (for a
  baseline, the legend says so). `cost_effectiveness` keeps its row. The
  notes of those figures and of the Pareto plots list it, and so does
  `index.md` under the condition list.
* `episodes.csv` rows flagged `duplicate` are dropped, as the aggregator drops
  them from `conditions.csv` and `paired.csv`.
* Invalid trials are left out of every figure, and so are their per-event
  rows: `per_subagent` (dispatches), the cap-reason panel of `spawn_counts`
  (spawn calls) and `arbiter` (arbitrations) keep only rows whose episode is
  valid and not a duplicate (`AnalysisData.event_rows`), and the notes count
  the rows left out. An invalid trial's events did happen, but an outage or
  harness failure shaped them: in the 2026-10-12 manager smoke run, the
  provider-failed `fix-git` trial's dispatches spent up to 593 s in LLM
  calls against 23-48 s in the valid trial. Only `calibration` (provider
  behaviour) uses every call.
* A manager control (`orchestrator_tools` other than `all`) is a control:
  hatched or hollow like the others. Its tick labels carry a `[manager]` tag
  and its legend entry names the orchestrator's tools.
* Log axes put labelled ticks at 1, 2 and 5 per decade and leave minor
  ticks unlabelled.

The figures:

1. `main_grid`: resolve rate for strategy × scope (grouped bars, Wilson CI),
   with the baseline bands. This is the headline result. Under each subagent
   bar, the row "delegated" gives k/n: the valid episodes whose orchestrator
   spawned subagents at least once, out of the valid episodes
   (`n_delegated`, `n_valid`). It is red when k = 0, meaning the condition
   never received its treatment, and the notes list it. A hollow circle
   marks `resolve_rate_delegated`, the per-protocol rate among those
   episodes, next to the intention-to-treat bar. For a `conditions.csv`
   written before these columns existed, the row is counted from
   `episodes.csv` (`delegated`, else `spawn_calls > 0`); with neither there
   is no row, the notes say so, and the caption leaves the row out. The
   caption states the
   estimand: the pooled Wilson intervals describe these tasks and their
   trial-to-trial randomness, not Terminal-Bench as a whole. It points to
   `paired.csv` (McNemar on matched trials) for condition-vs-baseline
   claims, instead of checking whether intervals overlap. When conditions
   differ in `n_tasks_valid`, a note names the ones with fewer valid tasks
   (see `resolve_rate_task_balanced`).
2. `pareto_cost`: resolve rate vs mean cost per episode, one panel per
   pricing scheme plus FLOPs. Points are conditions with x and y error bars.
   The Pareto frontier is drawn.
3. `pareto_time`: resolve rate vs mean wall-clock time. Panels: measured, then
   each simulated serving profile.
4. `time_breakdown`: stacked horizontal bars per condition of the mean
   critical-path composition (orchestrator LLM, subagent LLM, arbiter LLM,
   tools, overhead). Above each bar a bracket starts where the spawn groups'
   segments start, which is the end of the orchestrator LLM segment (the
   arbiter's judge and debate calls run inside the groups, and its segment
   comes before the subagent one). The bracket spans the mean
   `cp_group_sec`. A diamond measured from the same start marks the mean
   `sub_busy_sec`, the total subagent work. The dotted gap between them is
   the work that parallelism took off the critical path
   (`sub_busy_sec - cp_group_sec`). A diamond at or inside the bracket's
   end means nothing was hidden.
5. `token_composition`: stacked bars per condition (orchestrator vs
   subagent/arbiter/debate) × (uncached input, cached input, reasoning output,
   visible output). A second panel shows the cost share by role under the
   reference scheme.
6. `orchestrator_tokens`: box and strip plots of orchestrator (big-model)
   tokens per episode by condition, with the `single_5step` median as a
   reference line. This answers whether delegation offloads the big model.
7. `per_subagent`: violins by scope (and colored by strategy) of tokens per
   dispatch, tool calls per dispatch, steps used, and LLM seconds per
   dispatch. Includes the submitted fraction.
8. `spawn_counts`: first, the share of valid episodes that delegated (Wilson
   CI, k/n, red when none did), so non-uptake shows. Then subagents spawned
   per episode (histogram or ECDF by condition) and instruction following.
   That is `instruction_following_mean`, the mean over episodes of each
   episode's mean granted/requested per spawn call; an episode counts once
   however many spawn calls it made. Last, why spawn requests were cut.
9. `parallelism`: `parallel_speedup` distribution by strategy, plus scatter of
   `sub_busy_sec` vs subagent critical-path time.
10. `task_heatmap`: tasks (rows, sorted by difficulty then expert time) ×
    conditions (columns), where each cell is the solve fraction over rounds.
11. `timelines/`: Gantt charts for representative episodes, at most 2 per
    condition (the median wall-clock episode and the best solved one, i.e.
    the solved episode with the lowest wall-clock). Lanes
    are the orchestrator, then each subagent, then the arbiter. LLM calls
    are split into a TTFT part and a decode part; tool calls are shown
    separately; spawn groups are bracketed.
12. `calibration`: TTFT vs prompt tokens scatter with the fit line, per
    model; decode tok/s distribution per model; latency over sweep time
    colored by round (drift check). The TTFT line is the descriptive
    `ttft_fit_*`; the `fitted` serving profile uses the whole-call `lat_fit_*`
    fit. The TTFT axis shows every point unless a group has 200 calls or
    more, where the top 0.5% may be cut, and the notes count them.
13. `cost_effectiveness`: cost per solve (per scheme) and time per solve, by
    condition. Whiskers come from the ratio bootstrap above, and an
    unbounded one ends in an arrow at the panel edge.
14. `arbiter`: agreement rate, confidence histogram, debate rounds used,
    confidence vs resolved (when available), and winner index distribution.

## 8. CLI (`cli.py`)

Each module exposes `register_cli(subparsers)`, and `cli.main` wires them in:

| Module | Subcommands |
|---|---|
| runner | `validate`, `run` |
| sweep | `preflight`, `sweep`, `tasks` |
| analysis | `aggregate`, `check-run`, `completeness`, `calibrate` |
| plots | `plot` |

A module that fails to import or register is skipped with a warning on
stderr, so the other subcommands keep working.

## 8b. Reproducibility (hard requirement)

A co-author with a fresh clone must be able to reproduce the study by
following `docs/REPRODUCE.md` and running **at most three scripts**:

1. **`scripts/setup.sh`** sets up everything idempotently, and is safe to
   re-run:
   - checks for uv (and prints the install command if it is missing) and
     Docker (installed and running);
   - creates `.venv` with Python 3.13 via `uv venv`;
   - installs the package and pinned dependencies from `uv.lock` or
     `requirements.lock`, harbor pinned to `harbor==0.24.0`;
   - copies `.env.example` to `.env` if it is missing and tells the user which
     keys to fill;
   - verifies credentials (`SAIL_API_KEY`; Prime through `PRIME_API_KEY` +
     `PRIME_TEAM_ID` or a logged-in `prime` CLI);
   - runs the offline unit tests;
   - pre-pulls the dataset (`harbor dataset download terminal-bench@2.0`).

   It ends by printing a clear PASS/FAIL checklist.
2. **`scripts/run_experiment.sh [SPEC] [sweep options]`** runs preflight
   (with the `--oracle` feasibility check, reused while it passes), then the
   sweep. The default SPEC is `configs/sweeps/pilot.yaml`, also when only
   options are given; `--env-file` goes to both steps. It is resumable:
   re-running continues where it stopped and reruns jobs whose trials did
   not measure their condition.
3. **`scripts/analyze.sh [JOBS_DIR...]`** (output name from `OUT`, default
   the last dir's basename) runs check-run, completeness, aggregate and
   plot, in that order, on all the dirs together. It writes
   `analysis/<OUT>/` (tables) and `figures/<OUT>/` (figures + `index.md`).

The scripts take relative paths from the caller's directory, else the repo
root, print their usage with `-h`/`--help`, reject unknown arguments, and
run under macOS's `/bin/bash` 3.2 (tests run each under every bash found).

Pinned for reproducibility, and recorded in `sweep.json`:
- the code: `source_sha256` (the fingerprint of the harness source, which
  tells uncommitted versions apart and exists without git), per invocation,
  per job (manifest `run` lines) and per episode (`episode.json`); the git
  commit and dirty flag; and, when the source differs from the commit,
  `<jobs_dir>/provenance/<invocation_id>.patch`, which restores it on a
  checkout of the commit. A resume refuses to mix versions of the code
  that episodes run unless told to (`mixed_source` then records it);
  analysis-only edits are allowed and leave `mixed_source` unset;
- Python and harbor versions, and the lockfile hash;
- the dataset name@version;
- the resolved per-condition configs;
- the condition order seed;
- the provider tier;
- the date.

`.env.example` lists every variable, with comments.
`docs/REPRODUCE.md` covers:
- prerequisites;
- the three commands;
- the expected runtime and cost of the smoke and pilot sweeps;
- where outputs land;
- how to resume;
- how to run on a cloud sandbox instead of local Docker;
- known caveats: amd64 task images run emulated on Apple Silicon, and tool
  times are measured separately for that reason.

## 9. Testing

**Unit tests** need no network and no Docker:
- an httpx `MockTransport` SSE fake for the clients;
- `LocalShellEnvironment` for tools and the runtime;
- a scripted `FakeClient` for the orchestrator, subagents and arbiter;
- a synthetic episode generator, built from the `schema.py` dataclasses, for
  the analysis and the plots.

The required properties are listed per owner in the build tasks.

**Offline pipeline** (`tests/test_pipeline_offline.py`, no network, no
Docker): real episodes through `runner.run_task` (real orchestrator,
subagents, arbiter, tools and recorder on LocalShell sandboxes, scripted
models) for the pilot's own `single_5step`, `gemma31b_parallel_hard`,
`gemma31b_sequential_medium`, `gemma31b_arbiter_hard` and
`gemma31b_arbiter_easy` conditions, laid out as harbor jobs by the sweep
launcher's own code (`tests/offline_pipeline.py`). It then runs
`check-run`, `completeness --strict`, `aggregate` and `plot` through the CLI
and checks that they agree with the records: the measured replay equals
`cp_total_sec`, `cp_total_sec` is within 10% (+0.5 s) of `wall_clock_sec`,
the offline `sail_balanced` cost equals the logged run-time cost, and every
figure is drawn.

**Live tests** (`pytest -m live`, opt-in): real Sail and Prime calls for each
configured model, checking streaming, usage, headers and tool calls.

**End-to-end** (`configs/sweeps/smoke.yaml` through harbor and Docker), then:
1. `check-run` must pass;
2. `completeness --strict` must pass: no never-populated field that should be
   populated, and every subagents condition delegated in at least one
   episode (every arbiter condition arbitrated);
3. aggregate;
4. plot;
5. the `measured` latency replay must be within 10% of `wall_clock_sec` for
   every episode, or the overhead must be explained.
