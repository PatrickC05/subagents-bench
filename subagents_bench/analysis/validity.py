"""Trial validity and outcome. A trial is invalid when the sweep would rerun it
(:func:`subagents_bench.sweep.retry_reason`: billing error, provider failure, a harbor exception
before the verifier, a grader failure, a separate verifier's failure) or when it has no episode
or no result. An agent timeout is an outcome: harbor still runs the verifier, so the trial is
scored.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..sweep import rewards, retry_reason
from .load import Trial


def _ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def reward(result: dict | None) -> float | None:
    r = rewards(result or {}).get("reward")
    return None if r is None else float(r)


def exception_type(result: dict | None) -> str | None:
    return ((result or {}).get("exception_info") or {}).get("exception_type")


def _outside_agent(result: dict) -> bool:
    """A job cancel, or an exception before the agent started (environment or setup)."""
    exc = result["exception_info"]
    started = _ts((result.get("agent_execution") or {}).get("started_at"))
    occurred = _ts(exc.get("occurred_at"))
    return (exc["exception_type"] == "CancelledError" or started is None
            or (occurred is not None and occurred < started))


def invalid_reason(t: Trial) -> str | None:
    """None for a valid trial, else no_episode, no_result, billing_error, provider_error,
    grader_failed, verifier_failed, harbor_exception or agent_crash."""
    if t.episode is None:
        return "no_episode"
    if t.result is None:
        return "no_result"
    why = retry_reason(t.dir)
    if why in ("billing", "provider"):
        return f"{why}_error"
    if why in ("grader", "verifier"):
        return f"{why}_failed"
    if why:
        return "harbor_exception" if _outside_agent(t.result) else "agent_crash"
    return None


def outcome(t: Trial) -> dict[str, Any]:
    """valid, invalid_reason, harbor_exception, reward, resolved (an unscored valid trial is unsolved)
    and, for SWE-sweep, n_subtasks (the task's bugs, from the task: the grader reports 0 when the
    patch breaks the build), bugs_fixed (reward x n_subtasks) and regressed (the visible tests
    failed, which sets the reward to 0); null for other benchmarks."""
    why, r, g, n = invalid_reason(t), reward(t.result), rewards(t.result or {}), t.n_subtasks
    resolved = (1 if r >= 1.0 else 0) if r is not None else (0 if why is None else None)
    return {"valid": why is None, "invalid_reason": why, "harbor_exception": exception_type(t.result),
            "reward": r, "resolved": resolved, "n_subtasks": n,
            "bugs_fixed": None if r is None or n is None else r * n,
            "regressed": None if g.get("visible_ok") is None else int(not g["visible_ok"])}
