"""A self-contained HTML monitor of an interactive sweep: per-condition tables, the bugs-only controls
against their twins, an optional comparison with an earlier sweep, and one timeline card per
session with its transcript on request. ``python -m subagents_bench.monitor JOBS_DIR --out DIR``."""

from .build import Monitor, Sweep, build, load_sweep

__all__ = ["Monitor", "Sweep", "build", "load_sweep"]
