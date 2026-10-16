"""When each bug was fixed, from the replayed workspace snapshots (``agent/fix_states.json``).

# stub: replaced by the analysis-eta branch. Only the names and return shapes the levels analysis
# codes against (docs: the levels build contract, section 6); every trial reads as not replayed.
"""

from __future__ import annotations

from typing import Any

from .load import Trial


def bug_fix_rows(trial: Trial) -> list[dict[str, Any]]:  # stub: replaced by the analysis-eta branch
    """One row per sent bug (``fixes.csv``)."""
    return []


def fix_times(trial: Trial) -> dict[str, dict[str, Any]]:  # stub: replaced by the analysis-eta branch
    """subtask id -> {fixed_sec, stable_sec, first_pass_sec, n_breaks, visible_ok_at_fix,
    pass_before_sent, timing_ok, fixed_end}; empty without ``fix_states.json``."""
    return {}


def state_at(trial: Trial, t_session: float) -> dict[str, str]:  # stub: replaced by the analysis-eta branch
    """subtask id -> ``pass``/``fail``/``unknown`` of the workspace at ``t_session`` seconds after
    the session start; empty without ``fix_states.json``."""
    return {}
