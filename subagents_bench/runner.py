"""Local runner: one episode outside harbor, plus the ``validate``/``run`` CLI.

The study's runs go through harbor (:mod:`subagents_bench.harness.harbor_agent`);
this module is the same "probe the sandbox, record, run the orchestrator"
routine for a container we start ourselves (``sandbox.kind: local_docker``) or
the host shell (``local_shell``, tests and offline demos only).

:func:`run_task` writes the episode to
``<output_dir>/<config.name>/<task_id>/<episode_id>/`` -- a fresh directory per
trial, so repeated trials never overwrite each other -- and returns an
:class:`~subagents_bench.orchestrator.EpisodeResult`.

CLI::

    subagents-bench validate YAML [YAML ...] [--print]   # experiment configs, sweep specs, pricing/models/serving
    subagents-bench run CONFIG --task-id ID (--instruction TEXT | --instruction-file PATH)
                        [--workdir DIR] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from . import workspace
from .config import ConfigError, ExperimentConfig
from .environment import LocalDockerEnvironment, LocalShellEnvironment
from .orchestrator import EpisodeResult, Orchestrator
from .telemetry import EpisodeRecorder, new_episode_id

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class EnvFacts:
    """What :func:`probe_environment` learned about the sandbox."""

    workdir: str
    container_arch: str | None
    has_timeout_binary: bool | None


def probe_environment(env: Any, sandbox_workdir: str | None) -> EnvFacts:
    """Resolve the workdir (``auto`` -> ``pwd`` with the backend's default
    cwd) and assign it to ``env.workdir``; probe ``uname -m`` and coreutils
    ``timeout`` (cached for the tools layer). Raises RuntimeError if
    ``auto`` cannot be resolved (two ``pwd`` attempts failed): every later
    command would fail too."""
    if not sandbox_workdir or sandbox_workdir == "auto":
        wd = workspace.resolve_workdir(env) or workspace.resolve_workdir(env)
        if wd is None:
            raise RuntimeError("could not resolve the sandbox working directory (`pwd` failed twice)")
    else:
        wd = sandbox_workdir
    env.workdir = wd
    arch = workspace.container_arch(env)
    try:
        tb: bool | None = workspace.has_timeout_binary(env)
    except Exception:  # noqa: BLE001 - recorded as unknown
        tb = None
    return EnvFacts(workdir=wd, container_arch=arch, has_timeout_binary=tb)


def run_task(
    config: ExperimentConfig,
    task_id: str,
    instruction: str,
    env: Any,
    resolved: bool | None = None,
    *,
    raise_on_error: bool = True,
    client_factory: Callable[[Any], Any] | None = None,
    episode_id: str | None = None,
    sandbox_kind: str | None = None,
    episode_dir: str | os.PathLike | None = None,
    trial_name: str | None = None,
) -> EpisodeResult:
    """Run one episode against ``env`` and record it.

    ``resolved`` (if known, e.g. a local check) goes into ``episode.json``
    as ``resolved`` and ``reward`` (1.0/0.0). See
    :meth:`Orchestrator.run` for ``raise_on_error``. ``episode_dir``
    overrides the default ``<output_dir>/<name>/<task_id>/<episode_id>/``
    (e.g. a harbor-style ``<trial>/agent/``); ``trial_name`` goes into
    ``episode.json``.
    """
    facts = probe_environment(env, config.sandbox.workdir)
    episode_id = episode_id or new_episode_id()
    if episode_dir is None:
        episode_dir = Path(config.output_dir) / config.name / task_id / episode_id
    recorder = EpisodeRecorder(
        episode_dir,
        config=config,
        task_name=task_id,
        instruction=instruction,
        episode_id=episode_id,
        trial_name=trial_name,
        sandbox_kind=sandbox_kind or config.sandbox.kind,
        workdir=facts.workdir,
        container_arch=facts.container_arch,
        has_timeout_binary=facts.has_timeout_binary,
    )
    orch = Orchestrator(config, env, recorder, workdir=facts.workdir, client_factory=client_factory)
    result = orch.run(instruction, raise_on_error=raise_on_error)
    if resolved is not None:
        recorder.set_outcome(1.0 if resolved else 0.0, bool(resolved))
        result.record = recorder.record
    return result


def build_local_environment(config: ExperimentConfig, workdir: str | None = None) -> Any:
    sb = config.sandbox
    wd = workdir or (None if sb.workdir == "auto" else sb.workdir)
    if sb.kind == "local_docker":
        if not sb.image:
            raise ConfigError("sandbox.image is required for sandbox.kind local_docker")
        return LocalDockerEnvironment(sb.image, workdir=wd, network=sb.network)
    if sb.kind == "local_shell":
        return LocalShellEnvironment(workdir=wd or os.getcwd())
    raise ConfigError(
        f"the local runner supports sandbox.kind local_docker or local_shell, not {sb.kind!r}; "
        "harbor configs run through `subagents-bench sweep` (or `harbor run` with the harbor agent)"
    )


def run_local(config: ExperimentConfig, task_id: str, instruction: str, **kwargs: Any) -> EpisodeResult:
    env = build_local_environment(config)
    try:
        return run_task(config, task_id, instruction, env, **kwargs)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _describe(cfg: ExperimentConfig) -> str:
    s, b = cfg.subagent, cfg.budget
    lines = [
        f"name: {cfg.name} (config_hash {cfg.config_hash()})",
        f"  topology: {cfg.topology.value}; orchestrator: {cfg.orchestrator.model} ({cfg.orchestrator.provider}"
        + (f", {cfg.orchestrator.completion_window}" if cfg.orchestrator.completion_window else "") + ")",
        f"  budget: {b.orchestrator_steps} steps, {b.total_token_budget} tokens, {b.timeout_sec}s safety cap, "
        f"commands {b.command_timeout_sec}s (max {b.max_command_timeout_sec}s)",
        f"  sandbox: {cfg.sandbox.kind}, workdir {cfg.sandbox.workdir}",
        f"  orchestrator tools: {', '.join(cfg.resolved_orchestrator_tools())}"
        + (" (all)" if cfg.tools.orchestrator_tools is None else " (restricted)"),
    ]
    if cfg.topology.value == "subagents":
        sub = cfg.resolved_subagent_model()
        lines.append(
            f"  subagents: {s.strategy.value}/{s.scope.value}, model {sub.model} ({sub.provider}), "
            f"max_concurrent {s.max_concurrent}, max_dispatches {cfg.resolved_max_dispatches()}, max_steps {s.max_steps}"
        )
        if s.strategy.value == "arbiter":
            a = s.arbiter
            lines.append(
                f"  arbiter: {a.n_candidates} candidates, {a.debate_rounds} debate round(s), threshold "
                f"{a.confidence_threshold}, judge {cfg.resolved_arbiter_model().model}, max_copy_mb {a.max_copy_mb}"
            )
    return "\n".join(lines)


def yaml_kind(raw: Any) -> str:
    """What a YAML file under ``configs/`` is: ``sweep`` (a sweep spec),
    ``pricing`` (price schemes), ``pricing_reference``, ``models`` (model
    catalog), ``serving`` (serving profiles) or ``experiment``."""
    if isinstance(raw, dict):
        if "conditions" in raw or "extends" in raw:
            return "sweep"
        if "schemes" in raw:
            return "pricing"
        if "scheme" in raw and "models" in raw:
            return "pricing_reference"
        if "profiles" in raw:
            return "serving"
        if "models" in raw and "name" not in raw:
            return "models"
    return "experiment"


def _validate_sweep(path: str, show: bool) -> int:
    from . import sweep as sw  # noqa: PLC0415

    try:
        spec = sw.load_spec(path, root=sw.default_root())
    except sw.SpecError as e:
        print(f"[FAIL] {path}: {e}")
        return 1
    rc = 0
    print(f"[OK] {path}: sweep {spec.name!r}, {len(spec.conditions)} condition(s), {spec.rounds} round(s)")
    for cname, cfg in sw.resolve_conditions(spec).items():
        problems, full = sw.validate_condition(cfg)
        if problems or full is None:
            rc = 1
            for prob in problems:
                print(f"[FAIL]   condition {cname}: {prob}")
            continue
        exp = ExperimentConfig.from_dict(full)
        print(f"[OK]   condition {cname}\n" + "\n".join("  " + line for line in _describe(exp).splitlines()))
        if show:
            print(yaml.safe_dump(full, sort_keys=False))
    return rc


def _validate_analysis_file(path: str, kind: str) -> int:
    """Load a pricing/models/serving file with the analysis package's own loaders."""
    from .analysis.aggregate import default_config_path  # noqa: PLC0415
    from .analysis.latency import load_serving  # noqa: PLC0415
    from .analysis.pricing import load_models, load_pricing  # noqa: PLC0415

    warnings: list[str] = []
    if kind == "models":
        catalog, warnings = load_models(path)
        what = f"{len(catalog.models)} model(s), {len(catalog.aliases)} alias(es)"
    else:
        catalog, w = load_models(default_config_path("models.yaml"))
        if kind in ("pricing", "pricing_reference"):
            if kind == "pricing_reference":  # an include file: load it the way pricing.yaml includes it
                import tempfile  # noqa: PLC0415

                with tempfile.TemporaryDirectory() as tmp:
                    shim = Path(tmp) / "pricing.yaml"
                    shim.write_text(yaml.safe_dump({"include": [str(Path(path).resolve())]}))
                    book = load_pricing(shim, catalog)
            else:
                book = load_pricing(path, catalog)
            warnings = list(book.warnings)
            what = "schemes " + ", ".join(f"{n} ({len(book.schemes[n].prices)} models)" for n in book.scheme_names)
        else:
            profiles, warnings = load_serving(path, catalog, [])
            what = "profiles " + ", ".join(pr.name for pr in profiles)
    for w in warnings:
        print(f"  warning: {w}")
    print(f"[OK] {path}: {kind} file; {what}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    rc = 0
    for path in args.configs:
        try:
            raw = yaml.safe_load(Path(path).read_text())
        except (OSError, yaml.YAMLError) as e:
            print(f"[FAIL] {path}: {e}")
            rc = 1
            continue
        kind = yaml_kind(raw)
        if kind == "sweep":
            rc |= _validate_sweep(path, args.print)
            continue
        if kind != "experiment":
            try:
                rc |= _validate_analysis_file(path, kind)
            except Exception as e:  # noqa: BLE001 - any loader failure is a FAIL for this file
                print(f"[FAIL] {path}: {kind} file does not load: {type(e).__name__}: {e}")
                rc = 1
            continue
        try:
            cfg = ExperimentConfig.from_yaml(path)
        except ConfigError as e:  # the message already names the file
            print(f"[FAIL] {e}")
            rc = 1
            continue
        except (OSError, yaml.YAMLError) as e:
            print(f"[FAIL] {path}: {e}")
            rc = 1
            continue
        print(f"[OK] {path}\n{_describe(cfg)}")
        if args.print:
            print(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    return rc


def cmd_run(args: argparse.Namespace) -> int:
    from .credentials import load_dotenv  # noqa: PLC0415

    load_dotenv(REPO_ROOT / ".env")
    try:
        cfg = ExperimentConfig.from_yaml(args.config)
    except ConfigError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return 2
    except (OSError, yaml.YAMLError) as e:
        print(f"[FAIL] {args.config}: {e}", file=sys.stderr)
        return 2
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.workdir:
        cfg.sandbox.workdir = args.workdir
    if args.instruction_file:
        instruction = Path(args.instruction_file).read_text()
    else:
        instruction = args.instruction
    if not instruction or not instruction.strip():
        print("[FAIL] give the task with --instruction or --instruction-file", file=sys.stderr)
        return 2
    try:
        res = run_local(cfg, args.task_id, instruction)
    except ConfigError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return 2
    print(f"episode {res.episode_id}: exit_reason={res.exit_reason}")
    print(f"records: {res.episode_dir}")
    if res.error:
        print(f"error: {res.error.strip().splitlines()[-1]}")
    return 0 if res.exit_reason in ("finished", "max_steps", "no_tool_call") else 1


def register_cli(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "validate",
        help="load config YAML(s) and print what they resolve to (experiment configs, sweep specs with every "
             "resolved condition, and the pricing/models/serving files)",
    )
    p.add_argument("configs", nargs="+", help="YAML file(s), e.g. configs/**/*.yaml")
    p.add_argument("--print", action="store_true", help="also print the fully resolved config as YAML")
    p.set_defaults(func=cmd_validate)

    p = subparsers.add_parser(
        "run", help="run one episode locally (sandbox.kind local_docker or local_shell)"
    )
    p.add_argument("config", help="ExperimentConfig YAML file")
    p.add_argument("--task-id", required=True, help="name for the task (output subdirectory)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--instruction", "--issue", dest="instruction", help="the task instruction text")
    g.add_argument("--instruction-file", help="read the task instruction from this file")
    p.add_argument("--workdir", help="override sandbox.workdir")
    p.add_argument("--output-dir", help="override output_dir")
    p.set_defaults(func=cmd_run)
