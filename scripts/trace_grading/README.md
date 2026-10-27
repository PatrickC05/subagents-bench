# Per-trace grading

Some interactive metrics can't be read reliably from the records (issue #15): whether and when
the user was told about a bug, ETAs and done claims written in free text, whether a done claim
was true when made, how the agent handled the urgent bug, whether a change of plan was carried
out. These scripts have an LLM agent read each session's trace and grade it against a rubric,
then aggregate the grades per setting and level.

1. `timeline.py JOBS_DIR OUT_DIR`: one compact text timeline per finished trial (the user's
   messages, the replayed fix states from `fix-times`, the orchestrator's turns, tool calls and
   results, replies, subagent dispatches and reports, interrupts and stop requests), plus
   `index.json`. Run `subagents-bench fix-times` first so the ground truth is there.
2. Grade every timeline with `rubric.md`, one agent per trace, returning the JSON schema in
   `grade_workflow.js`. That file is a Claude Code workflow script (pass it with `args`:
   `files` = the timeline names, `rubric`, `dir`, `cwd`); any agent runner that returns the same
   JSON works. Save the list of results as one JSON file.
3. `aggregate.py JOBS_DIR TIMELINES[,..] GRADES[,..] OUT_JSON`: per setting x level, the median
   time until the user heard back and until the agent noticed each bug, the share within 60 s,
   ETAs met against the replayed fix time, done claims true when made, the user told the right
   status at the end, questions correct within 60 s, status checks answered, how the urgent bug
   was handled, and for changes of plan whether work stopped and the bug was reverted.

Checks on the levels run: the graders' first reply the user saw matched the records on 223 of 240
bugs (the rest were batch replies naming the bug) and their question verdicts matched
`grade-answers` on all 212 judged questions.
