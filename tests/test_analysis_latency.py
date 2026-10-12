"""Critical path and latency simulation on episodes with known structure.

Every episode below is laid out by the synthetic builder with harness gaps
between events and spawn bookkeeping overhead, so wall-clock time is larger
than the critical path by a known amount. Expected values are computed by
hand from the event durations (to 1e-6).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth import GEMMA, ORCH, Arb, C, EpisodeBuilder, Judge, Spawn, T  # noqa: E402

from subagents_bench.analysis.latency import (  # noqa: E402
    CriticalPath,
    EpisodeGraph,
    FittedProfile,
    FitParams,
    MeasuredProfile,
    RooflineProfile,
    TableProfile,
    critical_path,
    fit_params_from_rows,
    fit_target_sec,
    fitted_vs_measured,
    load_serving,
)
from subagents_bench.analysis.load import load_episode  # noqa: E402
from subagents_bench.analysis.pricing import load_models  # noqa: E402
from subagents_bench.analysis.tables import calibration_table  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TOL = 1e-6

# Member chains shared by the parallel / sequential episodes: sums 3.0, 4.0, 2.7.
MEMBERS = [
    [C(1.0), T("edit_file", 0.5), C(1.5)],
    [C(4.0)],
    [C(0.7), T("bash", 2.0)],
]


def _write(b: EpisodeBuilder, tmp_path: Path, name: str):
    b.write(tmp_path / name)
    return load_episode(tmp_path / name)


def parallel_episode(tmp_path, strategy="parallel", concurrency=None):
    b = EpisodeBuilder(strategy=strategy, t0=1000.0, gap=0.1)
    b.step(C(2.0), [Spawn(members=MEMBERS, concurrency=concurrency, start_overhead=0.05, end_overhead=0.1,
                          member_gap=0.02)])
    b.step(C(1.0), [T("bash", 0.25)])
    b.step(C(0.5), finish=True)
    b.finish(end_gap=0.3)
    return b, _write(b, tmp_path, f"{strategy}{concurrency}")


def arbiter_episode(tmp_path):
    b = EpisodeBuilder(strategy="arbiter", t0=1000.0, gap=0.1)
    arb1 = Arb(
        candidates=[[C(1.0), T("edit_file", 0.5)], [C(2.0)], [C(1.2), T("bash", 0.3)]],  # max chain 2.0
        rounds=[Judge(C(0.8), confidence=0.5, agreement=False),
                Judge(C(0.6), debates=[C(0.4), C(0.9), C(0.3)], confidence=0.9)],  # 0.8 + max(debates) 0.9 + 0.6
        apply_sec=0.2,
        isolation_sec=0.3,  # private copies before the candidates
        diff_sec=0.15,  # diff summaries before the first judge round
    )  # 0.3 + 2.0 + 0.15 + 2.3 + 0.2 = 4.95
    arb2 = Arb(candidates=[[C(3.0)], [C(2.5)], [C(1.0)]], rounds=[Judge(C(0.5))])  # 3.0 + 0.5 = 3.5
    b.step(C(2.0), [Spawn(arbitrations=[arb1, arb2], start_overhead=0.05, end_overhead=0.1)])
    b.step(C(0.5), finish=True)
    b.finish(end_gap=0.3)
    return b, _write(b, tmp_path, "arbiter")


def test_parallel_critical_path(tmp_path):
    _, data = parallel_episode(tmp_path)
    cp = critical_path(data)
    assert cp.total == pytest.approx(7.75, abs=TOL)  # 2.0 + max(3.0, 4.0, 2.7) + 1.0 + 0.25 + 0.5 + finish 0
    assert cp.parts["orch_llm"] == pytest.approx(3.5, abs=TOL)
    assert cp.parts["sub_llm"] == pytest.approx(4.0, abs=TOL)
    assert cp.parts["arb_llm"] == pytest.approx(0.0, abs=TOL)
    assert cp.parts["tool"] == pytest.approx(0.25, abs=TOL)
    assert cp.group_sec == pytest.approx(4.0, abs=TOL)
    # Wall clock adds 5 gaps of 0.1 (call->spawn; step2: before call, call->bash; step3: before call,
    # call->finish), spawn overheads 0.05 + 0.1 and the end gap 0.3 -> overhead 0.95.
    wall = data.episode["wall_clock_sec"]
    assert wall == pytest.approx(8.7, abs=TOL)
    assert wall - cp.total == pytest.approx(0.95, abs=TOL)
    # The measured replay is the same chain model, so it reproduces cp exactly.
    assert CriticalPath(EpisodeGraph(data), MeasuredProfile().call_sec).episode().total == pytest.approx(cp.total, abs=1e-12)


def test_sequential_critical_path(tmp_path):
    _, data = parallel_episode(tmp_path, strategy="sequential")
    cp = critical_path(data)
    assert cp.total == pytest.approx(3.5 + (3.0 + 4.0 + 2.7) + 0.25, abs=TOL)  # 13.45
    assert cp.parts["sub_llm"] == pytest.approx(1.0 + 1.5 + 4.0 + 0.7, abs=TOL)
    assert cp.parts["tool"] == pytest.approx(0.5 + 2.0 + 0.25, abs=TOL)
    assert cp.group_sec == pytest.approx(9.7, abs=TOL)
    # Members really ran one after another in the synthetic timeline.
    ds = sorted(data.dispatches, key=lambda d: d["t_start"])
    assert all(b["t_start"] >= a["t_end"] - 1e-9 for a, b in zip(ds, ds[1:]))


def test_parallel_with_fewer_slots_than_members_is_list_scheduled(tmp_path):
    # concurrency 2: member 1 (3.0) and member 2 (4.0) start together, member 3 (2.7) follows member 1.
    _, data = parallel_episode(tmp_path, concurrency=2)
    cp = critical_path(data)
    assert cp.group_sec == pytest.approx(3.0 + 2.7, abs=TOL)
    assert cp.total == pytest.approx(3.5 + 5.7 + 0.25, abs=TOL)


def test_arbiter_critical_path(tmp_path):
    _, data = arbiter_episode(tmp_path)
    cp = critical_path(data)
    assert cp.total == pytest.approx(2.0 + 4.95 + 0.5, abs=TOL)
    assert cp.parts["orch_llm"] == pytest.approx(2.5, abs=TOL)
    assert cp.parts["sub_llm"] == pytest.approx(2.0, abs=TOL)
    assert cp.parts["arb_llm"] == pytest.approx(0.8 + 0.9 + 0.6, abs=TOL)
    assert cp.parts["tool"] == pytest.approx(0.3 + 0.15 + 0.2, abs=TOL)  # isolation, diffs, apply: measured
    assert cp.group_sec == pytest.approx(4.95, abs=TOL)
    # Debate calls belong to the arbitration, not to the candidates' chains.
    g = EpisodeGraph(data)
    for d in data.dispatches:
        assert all(rec["role"] == "subagent" for _, rec in g.dispatch_chains[d["dispatch_id"]])


def test_arbitration_aborted_mid_debate_counts_the_debate_once(tmp_path):
    # Candidates (max 4 s), round-1 judge 2 s (not confident), round-2 debate cancelled after 10 s
    # (no judge call), cleanup 0.1 s. Apply runs from the end of the last judge OR debate call.
    b = EpisodeBuilder(strategy="arbiter", t0=1000.0)
    arb = Arb(candidates=[[C(4.0)], [C(3.0)], [C(2.0)]],
              rounds=[Judge(C(2.0), confidence=0.5), Judge(None, debates=[C(10.0), C(9.0), C(10.0)])],
              apply_sec=0.1)
    b.step(C(1.0), [Spawn(arbitrations=[arb])])
    b.finish(exit_reason="cancelled")
    data = _write(b, tmp_path, "aborted")
    assert data.arbitrations[0]["rounds"][1]["judge_call_id"] is None
    spawn_sec = next(t for t in data.tools if t["name"] == "spawn_subagents")["duration_sec"]
    cp = critical_path(data)
    assert cp.group_sec == pytest.approx(spawn_sec, abs=TOL) == pytest.approx(16.1, abs=TOL)
    assert cp.parts["arb_llm"] == pytest.approx(2.0 + 10.0, abs=TOL)
    assert cp.parts["tool"] == pytest.approx(0.1, abs=TOL)  # not 10.1: the debate is not re-counted as apply
    assert cp.total == pytest.approx(data.episode["wall_clock_sec"], abs=TOL)


def test_measured_replay_equals_cp_for_every_structure(tmp_path):
    eps = [parallel_episode(tmp_path)[1], parallel_episode(tmp_path, "sequential")[1], arbiter_episode(tmp_path)[1]]
    for data in eps:
        g = EpisodeGraph(data)
        assert CriticalPath(g, MeasuredProfile().call_sec).episode().total == pytest.approx(
            CriticalPath(g).episode().total, abs=1e-12)


def _roofline(catalog):
    return RooflineProfile(name="rf", catalog=catalog, mfu=0.4, mbu=0.6, bytes_per_param=2.0, overhead_sec=0.05,
                           peak_flops=9.89e14, peak_bw=3.35e12)


def test_roofline_simulation_by_hand(tmp_path):
    # Fixture parameter counts and byte widths (not real model facts).
    (tmp_path / "models.yaml").write_text(
        "models:\n"
        f"  {ORCH}: {{active_params_b: 16.0, active_params_prefill_b: 8.0, active_params_decode_b: 16.0, bytes_per_param: 0.8}}\n"
        f"  {GEMMA}: {{active_params_b: 30.0}}\n"
    )
    catalog, _ = load_models(tmp_path / "models.yaml")
    _, data = parallel_episode(tmp_path)
    prof = _roofline(catalog)

    def dur(prompt, completion, pre_b, dec_b, bpp):
        prefill = 0.4 * 9.89e14 / (2 * pre_b * 1e9)
        decode = 0.6 * 3.35e12 / (dec_b * 1e9 * bpp)
        return 0.05 + prompt / prefill + completion / decode

    d_orch = dur(1000, 100, 8.0, 16.0, 0.8)  # models.yaml bytes_per_param wins over the 2.0 default
    d_sub = dur(1000, 100, 30.0, 30.0, 2.0)  # no bytes_per_param in models.yaml -> profile default
    members = [2 * d_sub + 0.5, d_sub, d_sub + 2.0]
    expected = 3 * d_orch + max(members) + 0.25
    sim = CriticalPath(EpisodeGraph(data), prof.call_sec).episode()
    assert sim.total == pytest.approx(expected, abs=TOL)
    assert sim.parts["tool"] == pytest.approx(0.25 + 0.5, abs=TOL)  # winning member is the edit_file chain

    # A per-model override in the profile wins over models.yaml.
    prof.model_overrides = {ORCH: {"bytes_per_param": 0.5}}
    assert prof.bytes_per_param_for(ORCH) == (0.5, "profile override")
    d_orch2 = dur(1000, 100, 8.0, 16.0, 0.5)
    sim2 = CriticalPath(EpisodeGraph(data), prof.call_sec).episode().total
    assert sim2 == pytest.approx(3 * d_orch2 + max(members) + 0.25, abs=TOL)


def test_unknown_model_simulates_to_nan(tmp_path):
    (tmp_path / "models.yaml").write_text(f"models:\n  {ORCH}: {{active_params_b: 16.0}}\n")
    catalog, _ = load_models(tmp_path / "models.yaml")
    _, data = parallel_episode(tmp_path)
    prof = _roofline(catalog)
    assert math.isnan(CriticalPath(EpisodeGraph(data), prof.call_sec).episode().total)  # Gemma unknown
    single = EpisodeBuilder(topology="single", t0=1000.0)
    single.step(C(1.0), [T("bash", 1.0)]).step(C(1.0), finish=True)
    d = _write(single, tmp_path, "single")
    assert CriticalPath(EpisodeGraph(d), prof.call_sec).episode().total == pytest.approx(
        2 * (0.05 + 1000 / (0.4 * 9.89e14 / 32e9) + 100 / (0.6 * 3.35e12 / 32e9)) + 1.0, abs=TOL)


def test_fitted_and_table_profiles(tmp_path):
    _, data = arbiter_episode(tmp_path)
    key = lambda m: (m, "sail", "balanced")  # noqa: E731
    fitted = FittedProfile(name="fitted", source="test", params={
        key(ORCH): FitParams(0.3, 0.02, 1 / 100.0), key(GEMMA): FitParams(0.2, 0.01, 1 / 50.0)})
    d_o = 0.3 + 0.02 * 1.0 + 100 / 100.0  # every synthetic call: 1000 prompt, 100 completion
    d_s = 0.2 + 0.01 * 1.0 + 100 / 50.0
    # Orchestrator and judge calls use DeepSeek; candidates and debates use Gemma.
    # arb1: isolation 0.3, max cand (2 cand chains with 1 call + tool; one with 1 call) -> max(d_s+0.5, d_s, d_s+0.3),
    # diffs 0.15, judge, debate + judge, apply 0.2 (setup, diffs and apply are measured, not re-simulated)
    arb1 = 0.3 + (d_s + 0.5) + 0.15 + d_o + (d_s + d_o) + 0.2
    arb2 = d_s + d_o
    expected = 2 * d_o + max(arb1, arb2)
    assert CriticalPath(EpisodeGraph(data), fitted.call_sec).episode().total == pytest.approx(expected, abs=TOL)

    table = TableProfile(name="local", catalog=load_models(None)[0], models={
        ORCH: {"ttft_base_sec": 0.1, "prefill_tok_s": 5000.0, "decode_tok_s": 80.0},
        GEMMA: {"ttft_base_sec": 0.05, "prefill_tok_s": 10000.0, "decode_tok_s": 120.0}})
    t_o, t_s = 0.1 + 1000 / 5000 + 100 / 80, 0.05 + 1000 / 10000 + 100 / 120
    arb1 = 0.3 + (t_s + 0.5) + 0.15 + t_o + (t_s + t_o) + 0.2
    assert CriticalPath(EpisodeGraph(data), table.call_sec).episode().total == pytest.approx(
        2 * t_o + max(arb1, t_s + t_o), abs=TOL)
    # Missing params for a model -> NaN.
    fitted.params.pop(key(GEMMA))
    assert math.isnan(CriticalPath(EpisodeGraph(data), fitted.call_sec).episode().total)


def test_serving_yaml_loads_and_params_file(tmp_path):
    catalog, _ = load_models(REPO / "configs" / "models.yaml")
    profiles, warnings = load_serving(REPO / "configs" / "serving.yaml", catalog, [])
    assert [p.name for p in profiles] == ["measured", "fitted", "roofline_h100"]
    rf = profiles[2]
    assert (rf.mfu, rf.mbu, rf.bytes_per_param, rf.overhead_sec) == (0.4, 0.6, 2.0, 0.05)
    assert (rf.peak_flops, rf.peak_bw, rf.theoretical) == (9.89e14, 3.35e12, True)
    assert profiles[1].source == "calibration.csv (in-sample)"
    # A params_file written by `calibrate` is used instead of the in-sample fit.
    (tmp_path / "fit.yaml").write_text(
        "params:\n"
        f"  - {{model: {ORCH}, provider: sail, completion_window: balanced, ttft_a_sec: 0.3,"
        " ttft_b_sec_per_1k_prompt_tokens: 0.02, decode_tok_s: 100.0, n_calls: 10}\n"
        f"  - {{model: {GEMMA}, provider: sail, completion_window: balanced, ttft_a_sec: null,"
        " ttft_b_sec_per_1k_prompt_tokens: null, decode_tok_s: 50.0, n_calls: 1}\n"
    )
    (tmp_path / "serving.yaml").write_text(
        "profiles:\n  fitted: {kind: fitted, params_file: fit.yaml}\n  weird: {kind: nope}\n"
        "  missing_file: {kind: fitted, params_file: absent.yaml}\n")
    profiles, warnings = load_serving(tmp_path / "serving.yaml", catalog, [])
    assert [p.name for p in profiles] == ["measured", "fitted", "missing_file"]
    assert list(profiles[1].params) == [(ORCH, "sail", "balanced")]  # null fit skipped -> Gemma calls NaN
    # A params file from before the joint fit (TTFT fit + decode rate) still loads: c = 1 / decode_tok_s.
    p = profiles[1].params[(ORCH, "sail", "balanced")]
    assert (p.a_sec, p.b_sec_per_1k_prompt_tokens, p.c_sec_per_completion_token, p.form) == (
        0.3, 0.02, pytest.approx(0.01), "ttft_fit+decode_rate")
    call = {"model": ORCH, "provider": "sail", "completion_window": "balanced", "prompt_tokens": 2000,
            "completion_tokens": 300}
    assert profiles[1].call_sec(call) == pytest.approx(0.3 + 0.02 * 2 + 300 / 100.0)
    assert any("unknown kind" in w for w in warnings) and any("absent.yaml" in w for w in warnings)
    # No serving file at all: only the measured replay.
    profiles, warnings = load_serving(tmp_path / "none.yaml", catalog, [])
    assert [p.name for p in profiles] == ["measured"] and warnings


def _served_calls(n=120, seed=0, decode_intercept=0.6, tok_s=100.0, single_chunk_every=0):
    """Synthetic successful calls whose decode time has a per-call intercept (as Sail's do):
    decode_sec = 0.6 + completion/100; ttft = 0.4 + 0.05 per 1k prompt tokens. A single-chunk
    reply has decode_sec 0 and its whole time in ttft."""
    rng = np.random.default_rng(seed)
    calls = []
    for i in range(n):
        prompt, completion = int(rng.integers(500, 30000)), int(rng.integers(5, 1500))
        ttft = 0.4 + 0.05 * prompt / 1000
        decode = decode_intercept + completion / tok_s
        if single_chunk_every and i % single_chunk_every == 0:
            ttft, decode = ttft + decode, 0.0
        calls.append({"model": ORCH, "provider": "sail", "completion_window": "balanced", "prompt_tokens": prompt,
                      "completion_tokens": completion, "ttft_sec": ttft, "decode_sec": decode,
                      "latency_sec": ttft + decode + 1.5, "backoff_sec": 1.5, "error": None})
    return calls


def test_fitted_profile_is_unbiased_in_sample_and_across_reply_lengths():
    calls = _served_calls(single_chunk_every=7)
    rows = calibration_table(calls).to_dict("records")
    prof = FittedProfile(name="fitted", source="test", params=fit_params_from_rows(rows))
    p = prof.params[(ORCH, "sail", "balanced")]
    # The model is exactly linear: the joint fit recovers it (decode intercept folded into a).
    assert (p.a_sec, p.b_sec_per_1k_prompt_tokens, p.c_sec_per_completion_token) == (
        pytest.approx(1.0), pytest.approx(0.05), pytest.approx(0.01))
    check = fitted_vs_measured(prof, calls)[ORCH]
    assert check["ratio"] == pytest.approx(1.0, abs=1e-9) and check["n_calls"] == len(calls)
    for b in check["by_completion_tokens"].values():
        assert b["ratio"] == pytest.approx(1.0, abs=1e-6)
    # Retry backoff is not part of a served call: the fit target is ttft + decode.
    assert fit_target_sec(calls[1]) == pytest.approx(calls[1]["latency_sec"] - 1.5)
    # The old estimator (TTFT fit + median decode rate, no decode intercept) was biased by length
    # even without single-chunk replies: short replies too fast, long ones too slow.
    streamed = _served_calls()
    old_rows = [{k: v for k, v in r.items() if not k.startswith("lat_fit")}
                for r in calibration_table(streamed).to_dict("records")]
    old = fitted_vs_measured(FittedProfile(name="old", source="test", params=fit_params_from_rows(old_rows)),
                             streamed)[ORCH]["by_completion_tokens"]
    assert old["0-99"]["ratio"] < 0.9 and old["400+"]["ratio"] > 1.0


def test_fitted_target_without_stream_timings_and_failed_calls():
    c = {"latency_sec": 5.0, "backoff_sec": 2.0, "ttft_sec": None, "decode_sec": None}
    assert fit_target_sec(c) == 3.0
    assert math.isnan(fit_target_sec({"latency_sec": None}))
    calls = _served_calls(n=30)
    calls.append({**calls[0], "error": "LLMError: boom", "ttft_sec": None, "decode_sec": None})
    rows = calibration_table(calls).to_dict("records")
    assert rows[0]["lat_fit_n"] == 30  # failed calls are not fitted
    prof = FittedProfile(name="fitted", source="test", params=fit_params_from_rows(rows))
    assert fitted_vs_measured(prof, calls)[ORCH]["n_calls"] == 30
    # Too few calls to fit: NaN parameters, so the model's calls simulate to NaN (unknown, not made up).
    rows = calibration_table(_served_calls(n=1)).to_dict("records")
    assert math.isnan(rows[0]["lat_fit_a"]) and fit_params_from_rows(rows) == {}


def test_roofline_requires_numeric_params(tmp_path):
    (tmp_path / "s.yaml").write_text("profiles:\n  rf: {kind: roofline, mfu: 0.4}\n")
    with pytest.raises(ValueError, match="roofline profile 'rf'"):
        load_serving(tmp_path / "s.yaml", load_models(None)[0], [])


def test_crashed_episode_partial_chain(tmp_path):
    b = EpisodeBuilder(strategy="parallel", t0=1000.0)
    b.step(C(2.0), [Spawn(members=[[C(1.0), T("bash", 0.3)], [C(1.5)]])])
    b.finish(status="running")
    d = _write(b, tmp_path, "crash")
    # Drop the spawn and dispatch records, as a crash mid-group would.
    (d.dir / "spawns.jsonl").write_text("")
    (d.dir / "dispatches.jsonl").write_text("")
    data = load_episode(d.dir)
    cp = critical_path(data)
    # Members are recovered from the calls' group_id; the group is parallel.
    assert cp.total == pytest.approx(2.0 + max(1.3, 1.5), abs=TOL)
