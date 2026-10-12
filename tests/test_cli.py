"""The ``subagents-bench`` entry point: every module's subcommands are wired
in, a broken module does not take the others down, and ``validate`` accepts
every YAML file the repository ships (experiment configs, sweep specs with
all resolved conditions, and the pricing/models/serving files). The three
reproduction scripts parse their arguments as documented under every bash
available (macOS's /bin/bash 3.2 included)."""

from __future__ import annotations

import os
import shutil
import subprocess
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
    assert out.count("[OK]   condition ") == 15
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


# ---------------------------------------------------------------------------
# The three reproduction scripts (scripts/*.sh), under every bash available:
# macOS ships /bin/bash 3.2, so no bash 4 features. They run in a stub repo
# whose .venv/bin/subagents-bench only records its arguments.
# ---------------------------------------------------------------------------

SCRIPTS = ("setup.sh", "run_experiment.sh", "analyze.sh")
BASHES = sorted({b for b in ("/bin/bash", shutil.which("bash")) if b and Path(b).is_file()})

STUB_CLI = r"""#!/bin/bash
# Records one tab-separated line per call; exits with $STUB_RC_<SUBCOMMAND>.
printf '%s\t' "$@" >> "$STUB_LOG"
printf '\n' >> "$STUB_LOG"
rc_var="STUB_RC_$(printf '%s' "$1" | tr 'a-z-' 'A-Z_')"
exit "${!rc_var:-0}"
"""


def _stub_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in SCRIPTS:
        shutil.copy(REPO / "scripts" / name, repo / "scripts" / name)
    (repo / ".venv" / "bin").mkdir(parents=True)
    cli_stub = repo / ".venv" / "bin" / "subagents-bench"
    cli_stub.write_text(STUB_CLI)
    cli_stub.chmod(0o755)
    (repo / "configs" / "sweeps").mkdir(parents=True)
    for spec in ("pilot.yaml", "smoke.yaml", "prime_size.yaml"):
        (repo / "configs" / "sweeps" / spec).write_text("name: x\n")
    (repo / "jobs" / "pilot").mkdir(parents=True)
    (repo / "jobs" / "prime_size").mkdir(parents=True)
    return repo


def _run(bash: str, repo: Path, script: str, *args: str, cwd: Path | None = None, env: dict | None = None):
    full_env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"), "STUB_LOG": str(repo / "stub.log")}
    full_env.update(env or {})
    return subprocess.run([bash, str(repo / "scripts" / script), *args], cwd=cwd or repo, env=full_env,
                          capture_output=True, text=True, timeout=60)


def _calls(repo: Path) -> list[list[str]]:
    log = repo / "stub.log"
    return [line.rstrip("\t").split("\t") for line in log.read_text().splitlines()] if log.exists() else []


@pytest.mark.parametrize("bash", BASHES)
@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_parse(bash, script):
    assert subprocess.run([bash, "-n", str(REPO / "scripts" / script)]).returncode == 0


@pytest.mark.parametrize("bash", BASHES)
@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_prints_usage_and_runs_nothing(tmp_path, bash, script, flag):
    repo = _stub_repo(tmp_path)
    res = _run(bash, repo, script, flag)
    assert res.returncode == 0, res.stderr
    assert f"scripts/{script}" in res.stdout and "set -euo" not in res.stdout and not res.stdout.startswith("#")
    assert _calls(repo) == []


@pytest.mark.parametrize("bash", BASHES)
def test_setup_rejects_arguments(tmp_path, bash):
    repo = _stub_repo(tmp_path)
    res = _run(bash, repo, "setup.sh", "--frozen")
    assert res.returncode == 2 and "unknown argument: --frozen" in res.stderr


@pytest.mark.parametrize("bash", BASHES)
def test_run_experiment_options_without_spec_use_the_pilot(tmp_path, bash):
    repo = _stub_repo(tmp_path)
    res = _run(bash, repo, "run_experiment.sh", "--dry-run", "--only", "single_5step")
    assert res.returncode == 0, res.stderr
    assert "[PASS] dry run" in res.stdout
    assert _calls(repo) == [["sweep", str(repo / "configs/sweeps/pilot.yaml"), "--dry-run", "--only", "single_5step"]]


@pytest.mark.parametrize("bash", BASHES)
def test_run_experiment_resolves_paths_against_the_caller_then_the_repo(tmp_path, bash):
    repo = _stub_repo(tmp_path)
    # Relative to the caller's directory ...
    res = _run(bash, repo, "run_experiment.sh", "smoke.yaml", "--dry-run", cwd=repo / "configs" / "sweeps")
    assert res.returncode == 0, res.stderr
    # ... or, as the docs write them, to the repo root, from anywhere.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    res = _run(bash, repo, "run_experiment.sh", "configs/sweeps/prime_size.yaml", "--dry-run", cwd=elsewhere)
    assert res.returncode == 0, res.stderr
    assert [c[1] for c in _calls(repo)] == [str(repo / "configs/sweeps/smoke.yaml"),
                                            str(repo / "configs/sweeps/prime_size.yaml")]
    res = _run(bash, repo, "run_experiment.sh", "nope.yaml", cwd=elsewhere)
    assert res.returncode == 2 and "sweep spec not found: nope.yaml" in res.stderr


@pytest.mark.parametrize("bash", BASHES)
@pytest.mark.parametrize("form", ["separate", "equals"])
def test_run_experiment_gives_the_env_file_to_preflight_too(tmp_path, bash, form):
    repo = _stub_repo(tmp_path)
    keys = tmp_path / "keys"
    keys.mkdir()
    (keys / "team.env").write_text("SAIL_API_KEY=x\n")
    args = ["--env-file", "team.env"] if form == "separate" else ["--env-file=team.env"]
    res = _run(bash, repo, "run_experiment.sh", "configs/sweeps/smoke.yaml", *args, "--rounds", "1", cwd=keys)
    assert res.returncode == 0, res.stdout + res.stderr
    spec, env_file = str(repo / "configs/sweeps/smoke.yaml"), str(keys / "team.env")
    assert _calls(repo) == [
        ["preflight", spec, "--oracle", "--env-file", env_file],
        ["sweep", spec, "--env-file", env_file, "--rounds", "1"],
    ]
    assert "[PASS] sweep complete" in res.stdout


@pytest.mark.parametrize("bash", BASHES)
@pytest.mark.parametrize("rc, message", [
    (2, "[FAIL] sweep refused to run"), (3, "[FAIL] sweep aborted on a billing error"),
    (4, "[FAIL] sweep aborted: a provider failed in over half of a job's trials"), (130, "[WARN] sweep interrupted"),
])
def test_run_experiment_explains_exit_codes(tmp_path, bash, rc, message):
    repo = _stub_repo(tmp_path)
    res = _run(bash, repo, "run_experiment.sh", env={"STUB_RC_SWEEP": str(rc), "SKIP_PREFLIGHT": "1"})
    assert res.returncode == rc and message in res.stdout
    assert _calls(repo) == [["sweep", str(repo / "configs/sweeps/pilot.yaml")]]


@pytest.mark.parametrize("bash", BASHES)
@pytest.mark.parametrize("rc, message", [
    (1, "[FAIL] preflight failed: fix the [FAIL] items above"),
    (2, "[FAIL] preflight refused to run"),  # spec error, or another sweep holds the jobs dir
])
def test_run_experiment_stops_on_preflight_and_keeps_its_refusal_code(tmp_path, bash, rc, message):
    repo = _stub_repo(tmp_path)
    res = _run(bash, repo, "run_experiment.sh", env={"STUB_RC_PREFLIGHT": str(rc)})
    assert res.returncode == rc and message in res.stderr
    assert _calls(repo) == [["preflight", str(repo / "configs/sweeps/pilot.yaml"), "--oracle"]]


@pytest.mark.parametrize("bash", BASHES)
def test_analyze_takes_several_jobs_dirs_and_out_from_the_environment(tmp_path, bash):
    repo = _stub_repo(tmp_path)
    res = _run(bash, repo, "analyze.sh", "jobs/pilot", "jobs/prime_size/", env={"OUT": "size"}, cwd=tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    dirs = [str(repo / "jobs/pilot"), str(repo / "jobs/prime_size")]
    assert _calls(repo) == [
        ["check-run", *dirs],
        ["completeness", *dirs, "--strict", "--out", "analysis/size/completeness.csv"],
        ["aggregate", *dirs, "--out", "analysis/size"],
        ["plot", "analysis/size", "--out", "figures/size"],
    ]
    assert "analysis checklist (jobs/pilot jobs/prime_size)" in res.stdout
    # OUT defaults to the last dir's name; no argument means jobs/pilot.
    (repo / "stub.log").unlink()
    assert _run(bash, repo, "analyze.sh", str(repo / "jobs/prime_size")).returncode == 0
    assert _run(bash, repo, "analyze.sh").returncode == 0
    calls = _calls(repo)
    assert calls[2] == ["aggregate", dirs[1], "--out", "analysis/prime_size"]
    assert calls[6] == ["aggregate", dirs[0], "--out", "analysis/pilot"]


@pytest.mark.parametrize("bash", BASHES)
def test_analyze_rejects_what_is_not_a_jobs_dir(tmp_path, bash):
    repo = _stub_repo(tmp_path)
    res = _run(bash, repo, "analyze.sh", "jobs/pilot", "pilot_v2")  # the old [JOBS_DIR] [OUT] form
    assert res.returncode == 1 and "jobs dir not found: pilot_v2" in res.stderr and "OUT=NAME" in res.stderr
    res = _run(bash, repo, "analyze.sh", "--strict")
    assert res.returncode == 2 and "unknown option: --strict" in res.stderr
    assert _calls(repo) == []


FAKE_PYTHON = r"""#!/bin/bash
# Stands in for .venv/bin/python in setup.sh: the dataset download fails.
case "$1 $2" in
  "-c "*describe_credentials*) printf 'SAIL_API_KEY=set\nPRIME_API_KEY=missing\nPRIME_TEAM_ID=missing\n' ;;
  "-c "*importlib.metadata*) echo 0.24.0 ;;
  "-c "*platform*) echo 3.13.5 ;;
  "-m pytest") echo "3 passed in 0.10s" ;;
  "-m subagents_bench.sweep")
    echo "download failed (\`harbor dataset download terminal-bench@2.0\` failed (exit 1): Fetching index" >&2
    echo "Error: Dataset 'terminal-bench' (version: '2.0') not found); trying the local copy" >&2
    echo "dataset terminal-bench@2.0 not available under /x/.cache/datasets/terminal-bench@2.0" >&2
    exit 1 ;;
esac
"""


@pytest.mark.parametrize("bash", BASHES)
def test_setup_shows_why_the_dataset_download_failed(tmp_path, bash):
    repo = _stub_repo(tmp_path)
    venv_bin = repo / ".venv" / "bin"
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for path, text in {
        venv_bin / "python": FAKE_PYTHON,
        venv_bin / "harbor": "#!/bin/bash\n",
        stubs / "uv": "#!/bin/bash\n[ \"$1\" = --version ] && echo 'uv 0.9.0'\nexit 0\n",
        stubs / "docker": "#!/bin/bash\n[ \"$2\" = --format ] && echo '28.3.2, aarch64, 12 CPUs'\nexit 0\n",
    }.items():
        path.write_text(text)
        path.chmod(0o755)
    (repo / "uv.lock").write_text("version = 1\n")
    (repo / ".env").write_text("")
    res = _run(bash, repo, "setup.sh", env={"PATH": f"{stubs}:/usr/bin:/bin"})
    assert res.returncode == 1, res.stdout + res.stderr
    assert "Error: Dataset 'terminal-bench' (version: '2.0') not found" in res.stderr
    fail = [line for line in res.stdout.splitlines() if line.startswith("  [FAIL] dataset ")]
    assert len(fail) == 1, res.stdout
    assert "download failed (`harbor dataset download terminal-bench@2.0` failed (exit 1)" in fail[0]
    assert "network access to GitHub" in fail[0] and "SUBAGENTS_BENCH_DATASET_CACHE" in fail[0]
    assert "[PASS] unit tests" in res.stdout and "[PASS] SAIL_API_KEY" in res.stdout
