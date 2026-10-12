"""Trial validity and outcome from harbor's ``result.json`` and the episode.

A trial is invalid when it does not measure its condition: no episode, no result, a billing
error, a harbor exception outside the agent (environment/setup, job cancel), a provider
failure (the trial is rerun), or an agent crash before the verifier. An agent timeout is an
outcome: harbor still runs the verifier, so the trial is scored.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

GRADED_EXCEPTIONS = {"AgentTimeoutError"}  # harbor runs the verifier after these


def _ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def reward(result: dict | None) -> float | None:
    rewards = ((result or {}).get("verifier_result") or {}).get("rewards") or {}
    return None if rewards.get("reward") is None else float(rewards["reward"])


def exception_type(result: dict | None) -> str | None:
    return ((result or {}).get("exception_info") or {}).get("exception_type")


def _outside_agent(result: dict) -> bool:
    exc = result["exception_info"]
    started = _ts((result.get("agent_execution") or {}).get("started_at"))
    occurred = _ts(exc.get("occurred_at"))
    return (exc["exception_type"] == "CancelledError" or started is None
            or (occurred is not None and occurred < started))


def invalid_reason(episode: dict | None, result: dict | None) -> str | None:
    """None for a valid trial, else why it is invalid."""
    etype = exception_type(result)
    if episode is None:
        return "no_episode"
    if result is None:
        return "no_result"
    if episode.get("exit_reason") == "billing_error" or etype == "BillingError":
        return "billing_error"
    if etype and _outside_agent(result):
        return "harbor_exception"
    if (episode.get("exit_reason") == "provider_error" or etype == "ProviderUnavailableError"
            or (episode.get("n_provider_errors") or 0) > 0):
        return "provider_error"
    if etype and etype not in GRADED_EXCEPTIONS and not (result.get("verifier") or {}).get("started_at"):
        return "agent_crash"
    return None


def outcome(episode: dict | None, result: dict | None) -> dict[str, Any]:
    """valid, invalid_reason, harbor_exception, reward, resolved (an unscored valid trial is unsolved)."""
    why, r = invalid_reason(episode, result), reward(result)
    resolved = (1 if r >= 1.0 else 0) if r is not None else (0 if why is None else None)
    return {"valid": why is None, "invalid_reason": why, "harbor_exception": exception_type(result),
            "reward": r, "resolved": resolved}
