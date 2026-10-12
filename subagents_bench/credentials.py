"""API credentials: ``.env`` loading and per-provider key resolution.

Where each provider's credentials come from:

* **Sail** -- ``SAIL_API_KEY`` (usually from ``.env``).
* **Prime Intellect** -- ``PRIME_API_KEY`` and ``PRIME_TEAM_ID`` from the
  environment, else the logged-in ``prime`` CLI's ``~/.prime/config.json``
  (keys ``api_key`` / ``team_id``). Each value falls back independently.
* **Anthropic / OpenAI** -- ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``.
* A config's explicit ``api_key_env`` always takes precedence.

Nothing in this module prints or logs a credential value: helpers meant for
preflight checks (:func:`describe_credentials`, :func:`load_dotenv`) report
only names and set/missing states.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Provider key -> environment variable holding its API key.
DEFAULT_API_KEY_ENVS: dict[str, str] = {
    "sail": "SAIL_API_KEY",
    "prime": "PRIME_API_KEY",
    "prime_intellect": "PRIME_API_KEY",
    "pinference": "PRIME_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# Providers served by Prime Intellect's inference gateway (need X-Prime-Team-ID).
PRIME_PROVIDERS = frozenset({"prime", "prime_intellect", "pinference"})

# The credentials the study needs; what describe_credentials() reports by default.
STUDY_CREDENTIALS = ("SAIL_API_KEY", "PRIME_API_KEY", "PRIME_TEAM_ID")

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DQ_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\", "$": "$"}


# ---------------------------------------------------------------------------
# .env
# ---------------------------------------------------------------------------


def parse_dotenv(path: str | os.PathLike = ".env") -> dict[str, str]:
    """Parse a ``KEY=VALUE`` file. Missing file -> ``{}``.

    Supports ``#`` comment lines, an optional ``export`` prefix, single quotes
    (literal), double quotes (``\\n``, ``\\t``, ``\\"``, ``\\\\`` escapes) and
    trailing `` # comment`` after unquoted or quoted values. Lines that do not
    parse are skipped silently (their content is never echoed).
    """
    p = Path(path)
    if not p.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export ") or line.startswith("export\t"):
            line = line[len("export") :].strip()
        key, sep, rest = line.partition("=")
        key = key.strip()
        if not sep or not _KEY_RE.match(key):
            continue
        value = _parse_value(rest.strip())
        if value is None:
            continue
        out[key] = value
    return out


def _parse_value(rest: str) -> str | None:
    """Value part of one line; None if malformed (e.g. unterminated quote)."""
    if not rest:
        return ""
    q = rest[0]
    if q == "'":
        end = rest.find("'", 1)
        if end < 0:
            return None
        return rest[1:end]
    if q == '"':
        buf: list[str] = []
        i = 1
        while i < len(rest):
            c = rest[i]
            if c == "\\" and i + 1 < len(rest):
                nxt = rest[i + 1]
                buf.append(_DQ_ESCAPES.get(nxt, "\\" + nxt))
                i += 2
                continue
            if c == '"':
                return "".join(buf)
            buf.append(c)
            i += 1
        return None
    # Unquoted: an inline comment starts at whitespace followed by '#'.
    m = re.search(r"\s#", rest)
    if m:
        rest = rest[: m.start()]
    return rest.strip()


def load_dotenv(path: str | os.PathLike = ".env", override: bool = False) -> dict[str, str]:
    """Load ``path`` into ``os.environ``.

    Returns ``{key: status}`` for every key in the file, where status is
    ``"set"`` (newly set), ``"kept"`` (already set in the environment and not
    overridden) or ``"overridden"``. Values are never returned or printed.
    A missing file returns ``{}``.
    """
    status: dict[str, str] = {}
    for key, value in parse_dotenv(path).items():
        present = key in os.environ
        if present and not override:
            status[key] = "kept"
            continue
        os.environ[key] = value
        status[key] = "overridden" if present else "set"
    return status


# ---------------------------------------------------------------------------
# Prime Intellect
# ---------------------------------------------------------------------------


def prime_config_path() -> Path:
    """The ``prime`` CLI's config file (follows ``$HOME``)."""
    return Path.home() / ".prime" / "config.json"


def _read_prime_config() -> dict[str, Any]:
    try:
        data = json.loads(prime_config_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _nonempty(v: Any) -> str | None:
    return v if isinstance(v, str) and v.strip() else None


def prime_credentials() -> tuple[str | None, str | None]:
    """``(api_key, team_id)`` for Prime Intellect; each None if not found.

    The environment (``PRIME_API_KEY`` / ``PRIME_TEAM_ID``) wins; otherwise
    the value comes from ``~/.prime/config.json`` (``api_key`` / ``team_id``).
    """
    key = _nonempty(os.environ.get("PRIME_API_KEY"))
    team = _nonempty(os.environ.get("PRIME_TEAM_ID"))
    if key is None or team is None:
        cfg = _read_prime_config()
        key = key or _nonempty(cfg.get("api_key"))
        team = team or _nonempty(cfg.get("team_id"))
    return key, team


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_api_key(cfg: Any) -> str | None:
    """API key for a :class:`~subagents_bench.config.ModelConfig`, or None.

    Order: ``cfg.api_key_env`` (if set and non-empty), the provider's default
    variable, then (Prime providers only) the ``prime`` CLI config file.
    """
    provider = getattr(cfg, "provider", None)
    names = [getattr(cfg, "api_key_env", None), DEFAULT_API_KEY_ENVS.get(provider or "")]
    for name in names:
        if name:
            val = _nonempty(os.environ.get(name))
            if val is not None:
                return val
    if provider in PRIME_PROVIDERS:
        return prime_credentials()[0]
    return None


def describe_credentials(names: Iterable[str] | None = None) -> dict[str, str]:
    """``{name: "set" | "missing"}`` for preflight output. Never includes values.

    ``PRIME_API_KEY`` / ``PRIME_TEAM_ID`` count as set when the ``prime`` CLI
    config provides them.
    """
    prime_key, prime_team = prime_credentials()
    out: dict[str, str] = {}
    for name in names if names is not None else STUDY_CREDENTIALS:
        if name == "PRIME_API_KEY":
            ok = prime_key is not None
        elif name == "PRIME_TEAM_ID":
            ok = prime_team is not None
        else:
            ok = _nonempty(os.environ.get(name)) is not None
        out[name] = "set" if ok else "missing"
    return out
