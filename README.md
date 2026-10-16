# Interactive session traces (2026-10-15/16)

Raw records of every interactive SWE-sweep session behind issues #14–#19, round 1 only. DeepSeek-V4.1-Flash everywhere, 30-minute sessions on Modal. The code that produced them is on branch `exp-all` (the baseline on `swe-interactive-baseline`, PR #13).

Local paths are replaced by `<local path>`. The workspace snapshot patches (`agent/snapshots/`) and harbor logs are left out. Everything the analysis reads is here: `scripts/analyze.sh` on each sweep directory of this branch reproduces our tables exactly (checked for all nine sweeps).

## Sweeps

| Directory | Spec | Repositories | Sessions |
|---|---|---|---|
| `swe_interactive_baseline_modal/` | `swe_interactive_baseline_modal.yaml` | click, echo | 12 |
| `swe_interactive_levels/` | `swe_interactive_levels.yaml` (levels L0–L3, dispatcher included) | click, echo | 48 |
| `swe_interactive_levels_more/` | `swe_interactive_levels_more.yaml` | dateutil, gin, dayjs, luxon | 96 |
| `swe_interactive_change/`, `swe_interactive_change_more/` | change of plan (cancel, undo) | 2 + 4 repos | 10 + 20 |
| `swe_interactive_interrupt/`, `swe_interactive_interrupt_more/` | interrupt on message | 2 + 4 repos | 12 + 24 |
| `swe_interactive_hard/`, `swe_interactive_hard_more/` | symptom reports, lists of 3 | 2 + 4 repos | 20 + 40 |

All except the baseline ran with your `require_eta` fix and the status checks that name the bugs.

## Layout

`<sweep>/<sweep>__r1__<condition>/<trial>/`:

- `agent/transcript.jsonl`: every message of every agent (orchestrator and subagents), with reasoning
- `agent/tools.jsonl`, `calls.jsonl`, `steps.jsonl`: tool calls, model calls (tokens, timings), steps
- `agent/dispatches.jsonl`, `spawns.jsonl`: subagents started, their instructions and outcomes
- `agent/inbox.jsonl`, `replies.jsonl`: what the user sent and what the user saw
- `agent/snapshots.jsonl`: when each workspace snapshot was taken (the patches themselves are left out)
- `agent/episode.json`: config and summary; `fix_states.json`: replayed per-bug test status over time; `answers.json`: judged answers to code questions
- `config.json`, `result.json`, `verifier/eval.json`, `verifier/reward.json`: harbor's trial config and grade
- `<sweep>/hints/`: the bug reports and questions sent; `<sweep>/tasks.json`, `sweep.json`: the sweep's tasks and settings

Field definitions are in `docs/DATA.md` on `exp-all`.

## Settings (condition prefixes)

| Condition prefix | Name in the deck |
|---|---|
| `single` | Orchestrator alone |
| `background_hinted` | Orchestrator + subagents |
| `manager_background` | Subagents only |
| `manager_available` | Orchestrator + subagents (available) |
| `manager_owner` | Orchestrator + subagents (owner) |
| `dispatcher_hinted` | Dispatcher |
| `manager_available_stop` | Orchestrator + subagents (available, stop tool) |

The suffix gives the level (`baseline`/`eta`/`questions`/`urgent` = L0–L3) or the variant (`change`, `interrupt_*`, `symptom`, `web`).

## Rerunning the analysis

From an `exp-all` checkout: `scripts/analyze.sh <path to this branch>/swe_interactive_levels_more` (or any sweep directory). To pool the click+echo and four-repo runs as one six-repo sweep, put the trials of both under one sweep name.
