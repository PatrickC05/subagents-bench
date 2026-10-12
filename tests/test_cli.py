"""The ``subagents-bench`` entry point: every module's subcommands are wired
in, a broken module does not take the others down, and ``validate`` accepts
every YAML file the repository ships (experiment configs, sweep specs with
all resolved conditions, and the pricing/models/serving files)."""

from __future__ import annotations

from pathlib import Path

import pytest

from subagents_bench import cli
from subagents_bench.runner import yaml_kind

REPO = Path(__file__).resolve().parents[1]
SUBCOMMANDS = {"validate", "run", "preflight", "sweep", "tasks", "aggregate", "check-run", "completeness",
               "calibrate", "plot"}


def _subcommands(parser) -> set[str]:
    action = next(a for a in parser._actions if a.dest == "cmd")  # noqa: SLF001
    return set(action.choices)


def test_every_subcommand_is_registered():
    assert _subcommands(cli.build_parser()) == SUBCOMMANDS


def test_a_broken_module_is_skipped_with_a_warning(monkeypatch, capsys):
    monkeypatch.setattr(cli, "MODULES", ("runner", "no_such_module", "plots"))
    names = _subcommands(cli.build_parser())
    assert names == {"validate", "run", "plot"}
    err = capsys.readouterr().err
    assert "'no_such_module' subcommands are unavailable" in err and "ModuleNotFoundError" in err


@pytest.mark.parametrize("path", sorted(str(p.relative_to(REPO)) for p in (REPO / "configs").rglob("*.yaml")))
def test_validate_accepts_every_repo_yaml(path, monkeypatch, capsys):
    monkeypatch.chdir(REPO)
    assert cli.main(["validate", path]) == 0
    out = capsys.readouterr().out
    assert "[FAIL]" not in out and out.startswith("[OK]")


def test_validate_reports_each_sweep_condition(monkeypatch, capsys):
    monkeypatch.chdir(REPO)
    assert cli.main(["validate", "configs/sweeps/pilot.yaml"]) == 0
    out = capsys.readouterr().out
    assert out.count("[OK]   condition ") == 12
    assert "condition gemma31b_arbiter_hard" in out and "arbiter: 3 candidates" in out


def test_validate_fails_on_a_bad_config(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\ntopology: bogus\n")
    typo = tmp_path / "typo.yaml"
    typo.write_text("name: x\nsubagent: {strategy: parallel, scop: easy}\n")
    assert cli.main(["validate", str(bad), str(typo)]) == 1
    out = capsys.readouterr().out
    assert "topology: 'bogus' is not one of single, subagents" in out
    assert "scop" in out and out.count("[FAIL]") == 2


@pytest.mark.parametrize(
    "raw, kind",
    [
        ({"name": "x", "conditions": {}}, "sweep"),
        ({"extends": "configs/sweeps/pilot.yaml"}, "sweep"),
        ({"schemes": {}}, "pricing"),
        ({"scheme": "reference", "models": {}}, "pricing_reference"),
        ({"profiles": {}}, "serving"),
        ({"models": {}, "aliases": {}}, "models"),
        ({"name": "x", "topology": "single"}, "experiment"),
        (None, "experiment"),
    ],
)
def test_yaml_kind(raw, kind):
    assert yaml_kind(raw) == kind
