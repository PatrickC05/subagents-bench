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
2. `single_matched`: topology single, 40 steps (compute-matched baseline).
3. Nine conditions `gemma31b_{parallel,sequential,arbiter}_{easy,medium,hard}`:
   subagents, `max_concurrent=3`, `max_dispatches=15`, `max_steps=8`.
   Arbiter: `n_candidates=3`, `debate_rounds=1`, `confidence_threshold=0.8`,
   arbiter model = orchestrator model.
4. `samemodel_parallel_hard`: subagent model = orchestrator (same-model control).

Tasks:
- 4 TB-2.0 easy tasks: `fix-git`, `prove-plus-comm`, `cobol-modernization`,
  `overfull-hbox`.
- 6 short medium tasks: `crack-7z-hash`, `raman-fitting`,
  `constraints-scheduling`, `nginx-request-logging`,
  `openssl-selfsigned-cert`, `polyglot-c-py`.

That is 3 rounds. The smoke sweep (`configs/sweeps/smoke.yaml`) is `fix-git`
and `prove-plus-comm` × 1 round × all 12 conditions.

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
sandbox: {kind: harbor|local_docker|local_shell|exploitbench, image, workdir: auto|<path>, network}
task_ids, difficulties, n_trials, seed, output_dir
```

`ModelConfig` adds the following fields:

- `completion_window: str | None` (Sail tier, sent as the `X-Sail-Completion-Window` header)
- `extra_headers: dict`
- `reasoning_passback: bool = True`
- `context_window: int | None` (enables the context guard)
- `max_retries: int = 8`
- `request_timeout_sec: float = 600`
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
  exponential backoff from 2 s, doubled per attempt up to 120 s, with full
  jitter. Honor `Retry-After`. Stop after `max_retries`. Record every
  `Attempt`, with the backoff stored on the failed attempt.
- **Billing errors.** HTTP 402, or any 4xx whose body mentions
  balance/billing/credit/quota/payment, raises `BillingError` immediately.
- **Context-length errors.** An HTTP 400 or 413 whose body mentions
  context/maximum/too long/tokens raises `ContextLengthError`.
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
  `overhead_sec` (section 6).

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
diff_summary(env, orig_root, copy_root, cap_chars=4000) -> str   # changed files + diff -ru excerpt (falls back if no diff binary)
apply_copy(env, copy_root, dst_root, cap=200) -> tuple[list[str], str | None]      # (applied files, error)
git_diff_stat(env, workdir) -> str | None
sb_root(env) -> str                               # harness state root: /tmp/.sb in containers, a private dir for LocalShell
```

Path conventions: `files_modified_since` returns absolute paths,
`apply_copy` reports absolute destination paths and `edit_file`'s
`ToolOutput.files_written` holds absolute normalized paths, so all three
merge directly into `files_modified` fields. `manifest` keys are paths
relative to its root. `apply_copy(..., base=<manifest>)` restricts the apply
to what changed relative to `base`.

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
- Exit reasons: `finished`, `max_steps`, `no_tool_call`, `token_budget`,
  `timeout` (safety cap), `cancelled` (stop event), `billing_error` (re-raised
  after logging), `error`. A provider failure that survived every retry
  (`LLMError`, not `BillingError`) ends the episode with `error` and is not
  re-raised, so the verifier still grades the container; only `BillingError`
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
    confidence c, candidates agreed: yes/no)`, plus the applied files.
  - A cap note appears when not every request was granted.
- `files_modified` for the group is measured with a marker file created at
  group start and `find <workdir> -newer marker -type f`. It excludes
  `/tmp/.sb`, caps the list and counts the total. A failure gives None.

### Subagent (`subagent.py`)

- The system prompt depends on scope.
- It runs up to `max_steps` LLM calls, using the scope's tool set plus
  `submit(report)`.
- Status values: `submitted`, `no_tool_call` (the text is used as the report),
  `max_steps`, `context_overflow`, `error`, `cancelled`, and `token_budget` /
  `timeout` (before a step, the episode's `total_token_budget` or
  `timeout_sec` safety cap, or the subagent's own `token_budget`, was
  reached).
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

1. **Isolation (hard scope).** If `du -sm workdir <= max_copy_mb`, copy the
   workdir to `workspace.sb_root(env)/arb/<arbitration_id>/cand<k>`
   (`/tmp/.sb/arb/...` in containers; `cp -a`). Each
   candidate's cwd starts in its copy, and its prompt says: "your private
   working copy is X; make all edits there; do not modify Y". If the workdir
   is too big, use `shared_fallback`. Easy and medium scopes use `shared`.
2. **Candidates.** Run the candidates concurrently. Each is a normal
   subagent with `candidate_index` set.
3. **Judge.** One arbiter-model call. It receives the instruction, each
   candidate's report, and (with copy isolation) a compact diff summary per
   candidate (changed files + `diff -ru` excerpt capped at 4000 chars per
   candidate). Reports shown to the judge and to debaters are capped at
   `subagent.max_output_chars` (head and tail kept). It must call the tool
   `submit_judgement(winner: int, confidence: number 0..1, agreement: boolean, rationale: string)`.
   If the call has no tool call, parse JSON from the text. If parsing fails,
   use winner=0, confidence=None and `parse_ok=False`.
4. **Confidence-aware debate** (arXiv 2602.12276). If not
   (agreement and confidence ≥ threshold) and rounds remain, run a debate
   round. Each candidate's model gets one tool-less call with its own report
   plus the others' reports and returns a revised report (role `debate`,
   agent_id = candidate's agent_id). The candidates' calls run concurrently.
   Then judge again.
5. **Apply.** With copy isolation, apply the winner's copy to the real
   workdir, restricted to what the winner changed since its copy was made:
   `workspace.apply_copy(winner_copy, workdir, base=<manifest of the fresh
   copy>)` compares `md5sum` manifests (via `find -type f -print0 | xargs -0
   md5sum`), copies changed and added files with `cp -a` and deletes files
   the winner removed. (Without `base`, two concurrent arbitrations would
   delete each other's applied files.) Record `applied_files`. Then the
   copies are removed.
6. **Record.** Write an `ArbitrationRecord`. The judge calls have role
   `arbiter`, agent_id `arbiter-<arbitration_id>`, and step = round. A
   debate call is recorded under the candidate's agent_id with role
   `debate`; its transcript index restarts at 0 each round. Timing:
   `isolation_sec` (step 1, before `t_start`), `t_start`..`t_candidates_end`
   (step 2), `t_candidates_end`..first round `t_start` (diff summaries),
   rounds, and the last judge's end..`t_end` (apply and cleanup). All of it
   is the strategy's own cost and lies on the critical path (section 6).
   A candidate's `DispatchRecord.files_modified` lists real paths inside its
   copy.

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
  3. finalize the record with `exit_reason="cancelled"`;
  4. re-raise.
- It fills `AgentContext` (`n_input_tokens`, `n_cache_tokens`,
  `n_output_tokens`, `cost_usd`, `model_usage` per model, and `metadata` with
  the episode summary keys) **progressively** after every LLM call and at the
  end.
- **BillingError** is logged with `exit_reason="billing_error"` and re-raised,
  so harbor records the exception.
- It records `container_arch` (`uname -m`), `has_timeout_binary`, the harbor
  version, git commit and dirty state, and the host platform.

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

- `subagents-bench preflight SPEC [--oracle]` checks:
  - Docker is up;
  - harbor is importable and the dataset is downloadable;
  - every task exists, with its metadata written to `<jobs_dir>/tasks_meta.json`;
  - API keys resolve;
  - one tiny request per distinct (model, provider, tier) succeeds, recorded
    in `<jobs_dir>/preflight.json`;
  - optionally, `harbor run -a oracle` on all tasks passes, so tasks are
    known feasible on this machine.

  It exits nonzero on any failure.
- `subagents-bench sweep SPEC [--dry-run] [--only COND ...] [--rounds 1,2]`:
  - Materializes resolved configs to `<jobs_dir>/configs/<cond>.yaml`.
  - Then, for each round, for each condition in the shuffled order, runs:

    ```bash
    harbor run -d <dataset> -i <task>... -a subagents_bench.harness.harbor_agent:OrchestratorAgent \
      -e <env> -n <n_concurrent> -k 1 -o <jobs_dir> --job-name <name>__r<round>__<cond> \
      --agent-timeout-multiplier <m> -y -q
    ```

    The subprocess environment includes `SUBAGENTS_BENCH_CONFIG=<resolved yaml>`
    plus `.env` and the Prime credentials.
  - A complete job is skipped, which makes the sweep resumable. Complete
    means: the job's `result.json` parses, `n_completed_trials >=
    n_total_trials` (harbor 0.24.0 counts errored trials as completed),
    nothing is running or pending, `finished_at` is set, and every trial dir
    has its own `result.json`. A partial job is moved to
    `<jobs_dir>/_partial/<job>.partial.<k>` (nothing is deleted) and rerun.
    A resumed sweep refuses to run if a condition's resolved config changed.
  - After each job it scans `*/agent/episode.json` for
    `exit_reason == billing_error` and aborts the sweep if found. It also
    aborts after 3 consecutive failed jobs; Ctrl-C gives harbor time to
    remove its containers.
  - It appends to `<jobs_dir>/sweep_manifest.jsonl`, one line per job
    event: `sweep, job, round, condition, event` (`run`, `skipped_complete`,
    `moved_partial`, `interrupted`), `cmd, config_path, t_start, t_end,
    duration_sec, return_code, n_trials, n_completed, n_errors, complete`,
    and for `run` lines also `status_reason, n_episode_json,
    billing_error_trials, git_commit` (`moved_to` for `moved_partial`).
    round/condition agree across a job's lines; consumers join on `job` and
    take outcomes from the last `run` line.
  - It writes `<jobs_dir>/sweep.json`: the resolved spec and the hashes of
    its files, the base config hash, `condition_order` per round, each
    condition's materialized config path and sha256, the provider tiers,
    provenance (git commit and dirty flag, Python and harbor versions,
    lockfile and its sha256, Docker info, date, an invocation id), whether
    the jobs span several git commits, and the history of invocations.
  - `<jobs_dir>/tasks_meta.json` is `{dataset, source_dir, generated_at,
    tasks: {name: {difficulty, category, expert_time_min, junior_time_min,
    agent_timeout_sec, verifier_timeout_sec, docker_image, tags}}}` for the
    sweep's tasks only.
  - `--dry-run` prints the plan, the config validation and a ROUGH cost and
    runtime estimate from an assumed token/latency model
    (`sweep.RoughModel`, not a measurement). `python -m
    subagents_bench.sweep` is an alternative entry point.
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
  - `valid` is False, with `invalid_reason`, for `no_episode` (no
    episode.json), `billing_error`, `harbor_exception_outside_agent` (an
    environment/setup error type, a `CancelledError`, or any exception
    before the agent started) and `no_result` (a harbor trial without
    `result.json`). A verifier exception leaves the trial valid and
    unsolved (`resolved = 0`).
- **harbor phases:** `env_setup_sec, agent_setup_sec, agent_exec_sec, verifier_sec, trial_total_sec`
- **episode:** `wall_clock_sec, orchestrator_steps, spawn_calls, dispatches, redispatches, arbitrations, instruction_following, dispatch_submitted_frac, n_files_modified`
- **tokens,** for role in `orch, sub, arb, deb`: `{role}_calls, {role}_prompt_tokens, {role}_cached_tokens, {role}_uncached_prompt_tokens, {role}_completion_tokens, {role}_reasoning_tokens`, plus `total_tokens`
- **cost**, for each pricing scheme S: `cost_{S}_{role}, cost_{S}_total`, plus
  `cost_runtime_total` (the logged cost_usd)
- **compute:** `flops_{role}`, `flops_total`, where flops = 2 × active_params × tokens
- **time:** `llm_sec_{role}, tool_sec_{role}, backoff_sec, n_retries, n_command_timeouts`
- **critical path** (measured): `cp_total_sec, cp_orch_llm_sec, cp_sub_llm_sec, cp_arb_llm_sec, cp_tool_sec, cp_group_sec, overhead_sec (= wall_clock_sec - cp_total_sec), harness_exec_sec (the harness's measurement execs, which explain most of overhead_sec), sub_busy_sec (sum of dispatch durations), parallel_speedup (sub_busy_sec / cp_group_sec; NaN without dispatches or group time; for arbiter groups the denominator also holds isolation, judging and apply)`
- **extra:** `n_failed_calls, n_usage_estimated, container_arch, host_machine`
- **simulated:** `sim_{P}_wall_sec` for each serving profile P. The profile
  `measured` replays logged durations and must come out close to
  `cp_total_sec`.

### Other tables

- `calls.csv`, `tools.csv`, `dispatches.csv`, `spawns.csv`,
  `arbitrations.csv`, `steps.csv`: the records flattened, plus the episode
  keys `condition, task_name, round, episode_id`. Bulky lists are dropped;
  `chunk_offsets_ms` is summarized.
- `conditions.csv` (per condition) and `conditions_by_difficulty.csv`:
  - `n, n_valid, n_resolved, resolve_rate, resolve_ci_lo, resolve_ci_hi` (Wilson 95%);
  - for each of `wall_clock_sec`, `sim_*_wall_sec`, `cost_*_total`,
    `total_tokens`, `orch_*`/`sub_*` tokens and `dispatches`: the mean plus a
    bootstrap 95% CI (`_mean, _ci_lo, _ci_hi`; 2000 resamples, seed 0);
  - `cost_per_solve_{S}` and `time_per_solve_sec`, plus
    `cost_per_solve_runtime`;
  - the design columns of `episodes.csv`.

  NaN policy for condition aggregates: cost, FLOPs, simulated-time and
  reasoning-token means are NaN if any valid episode of the condition is
  NaN (e.g. an unpriced model), rather than averaged over the priced
  subset; other metrics drop NaN values.
- `paired.csv`: each condition vs each baseline (`single_5step`,
  `single_matched`) over tasks × rounds matched on (task, round), with
  `n_pairs, both, only_cond, only_base, delta_resolve, mcnemar_p` (exact
  binomial). A baseline is not paired with itself.
- `calibration.csv`, per (model, provider, completion_window):
  - `n_calls`
  - `ttft_median`, `ttft_p90`
  - `decode_tok_s_median` (completion_tokens / decode_sec where decode_sec > 0)
  - `ttft_fit_a`, `ttft_fit_b_per_1k_prompt_tokens` (least squares)
  - `latency_median`

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
  - `fitted` (from calibration.csv: `ttft_fit + completion/decode_rate`, which
    removes run-to-run noise but not the provider's queueing);
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
| arbiter | max over the group's arbitrations of: isolation setup (`isolation_sec`, measured) + max(candidate chains) + diff summaries (first round `t_start` − `t_candidates_end`, measured) + Σ over rounds of (judge call + max over that round's debate calls) + apply and cleanup (= t_end − last judge end, measured) |

The episode duration is the orchestrator chain. With `measured`, the
simulation must reproduce `cp_total_sec`, and that should sit close to
`wall_clock_sec`; the gap is reported as `overhead_sec`. Most of it is the
harness's own measurement commands (`harness_exec_sec`: markers,
modified-file scans, `git diff --stat`), which belong to no chain; the rest
is thread start-up and record writing. `summary.json` reports the share of
episodes whose overhead, and whose overhead not explained by
`harness_exec_sec`, exceeds 10% of wall-clock. The critical-path
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

It prints a report and exits nonzero on problems.

`subagents-bench completeness DIR...` reports, for every schema field, the
fraction of records where it is None, empty or zero, by role. It flags fields
that are never populated, separating fields that are legitimately empty in
a healthy run (failure fields, retry counters, provider extras, role- or
strategy-specific fields); `--strict` exits 1 on any other never-populated
field.

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
0), and the caption says so.

The figures:

1. `main_grid`: resolve rate for strategy × scope (grouped bars, Wilson CI),
   with the baseline bands. This is the headline result.
2. `pareto_cost`: resolve rate vs mean cost per episode, one panel per
   pricing scheme plus FLOPs. Points are conditions with x and y error bars.
   The Pareto frontier is drawn.
3. `pareto_time`: resolve rate vs mean wall-clock time. Panels: measured, then
   each simulated serving profile.
4. `time_breakdown`: stacked horizontal bars per condition of the mean
   critical-path composition (orchestrator LLM, subagent LLM, arbiter LLM,
   tools, overhead), with a marker for mean `sub_busy_sec` (total subagent
   work).
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
8. `spawn_counts`: subagents spawned per episode (histogram or ECDF by
   condition) and instruction-following (granted/requested).
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
    colored by round (drift check).
13. `cost_effectiveness`: cost per solve (per scheme) and time per solve, by
    condition.
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
2. **`scripts/run_experiment.sh [SPEC]`** runs preflight (with the
   `--oracle` feasibility check on first run), then the sweep. The default
   SPEC is `configs/sweeps/pilot.yaml`. It is resumable: re-running continues
   where it stopped.
3. **`scripts/analyze.sh [JOBS_DIR] [OUT]`** runs check-run, completeness,
   aggregate and plot, in that order. It writes `analysis/<sweep>/` (tables)
   and `figures/<sweep>/` (figures + `index.md`).

Pinned for reproducibility, and recorded in `sweep.json`:
- the git commit and dirty flag;
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
2. `completeness` must show no never-populated field that should be populated;
3. aggregate;
4. plot;
5. the `measured` latency replay must be within 10% of `wall_clock_sec` for
   every episode, or the overhead must be explained.
