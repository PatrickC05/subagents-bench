"""Agent commands that may break SWE-sweep's fairplay rules, flagged for a person to review.

SWE-sweep forbids learning a bug's fix from anywhere but the checked-out source: the code's
"future" (other git objects, other installed versions) and the network. A flag is a pattern
match, not a verdict: reading ``site-packages`` can be innocent, and in the task images the
repository's git holds only its ``baseline`` commit, so a history probe finds nothing. The
first interactive smoke had one (``git cat-file --batch-all-objects`` from a stuck single agent).

Categories:
- ``git_history``: objects or refs beyond the working tree (``git cat-file --batch-all-objects``,
  ``fsck``, ``reflog``, ``log``/``rev-list``/``show-branch`` with ``--all``, ``fetch``, ``pull``,
  ``clone``, ``ls-remote``);
- ``network``: ``curl``, ``wget`` or ``nc`` run as a command (not a word in a grep), ``pip download``,
  ``go get``, ``go mod download``;
- ``installed_copy``: another copy of the code (``site-packages``/``dist-packages``, Go's module
  cache, ``pip show -f``, ``pip download``).
"""

from __future__ import annotations

import re
from typing import Any

PATTERNS = {
    "git_history": re.compile(r"\bgit\s+(?:-C\s+\S+\s+)?(?:cat-file\s[^|;&\n]*--batch-all-objects|fsck\b|reflog\b|"
                              r"(?:log|rev-list|show-branch)\s[^|;&\n]*--all\b|fetch\b|pull\b|clone\b|ls-remote\b)"),
    "network": re.compile(r"(?:^|[;&|(]|\bsudo|\$\()\s*(?:curl|wget|nc)\s|\bpip3?\s+download\b|"
                          r"\bgo\s+(?:get|mod\s+download)\b", re.M),
    "installed_copy": re.compile(r"(?:site|dist)-packages\b|/go/pkg/mod\b|\bpip3?\s+show\s[^|;&\n]*-f\b|"
                                 r"\bpip3?\s+download\b"),
}
PATH_ARGS = {"read_file": "path", "list_dir": "path", "search": "path"}


def tool_flags(tool: dict[str, Any]) -> list[str]:
    """The categories a tool record's arguments match (a bash command, or a read tool's path)."""
    args = tool.get("arguments")
    if not isinstance(args, dict):
        return []
    if tool.get("name") == "bash":
        text = args.get("command")
    elif tool.get("name") in PATH_ARGS:
        text = args.get(PATH_ARGS[tool["name"]])
    else:
        return []
    if not isinstance(text, str):
        return []
    return [k for k, rx in PATTERNS.items() if rx.search(text)]
