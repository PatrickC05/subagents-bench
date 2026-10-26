"""Write the grades a grading workflow returned, one grades.json per sweep, from its journal:
python grades_from_journal.py JOURNAL_JSONL TIMELINES_DIR

The workflow (grade_workflow.js, or any run that labels each agent ``grade:<timeline name>``)
logs each grader's full return value in its journal.jsonl; returning only a summary keeps a run
of hundreds of graders out of the caller's context. Timeline names start with their sweep's
jobs dir name (``<sweep>__<condition>__<task>``), so each sweep's grades go to
TIMELINES_DIR/<sweep>/grades.json, next to its timelines, for aggregate.py.
"""
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

journal, timelines = Path(sys.argv[1]), Path(sys.argv[2])
label, result = {}, {}
for line in journal.read_text().splitlines():
    r = json.loads(line)
    if r.get("type") == "started" and str(r.get("label", "")).startswith("grade:"):
        label[r["key"]] = r["label"].removeprefix("grade:")
    elif r.get("type") == "result" and r.get("result") is not None:
        result[r["key"]] = r["result"]
by_sweep = defaultdict(list)
for key, value in result.items():
    if key not in label:
        continue
    grade = value if isinstance(value, dict) else ast.literal_eval(value)  # the journal may store a repr
    name = label[key]
    by_sweep[name.split("__")[0]].append({"name": name, **grade})
for sweep, grades in sorted(by_sweep.items()):
    out = timelines / sweep / "grades.json"
    out.write_text(json.dumps(grades))
    print(f"[OK] {out}: {len(grades)} graded sessions")
