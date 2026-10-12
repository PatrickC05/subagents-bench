"""Latency / throughput micro-benchmark for OpenAI-compatible subagent endpoints.

Streams chat completions and records, per request:
  ttft     time to first generated token (content or reasoning), seconds
  ttfa     time to first *answer* token (content, after any reasoning), seconds
  e2e      end-to-end latency, seconds
  tps      decode tokens/s after the first token = (completion_tokens - 1) / (e2e - ttft)
  in/out   prompt / completion tokens (from the provider's usage block)

Two prompts: `short` (~100 input tokens) and `long` (this repo's source as
repository context, roughly what a subagent sees). Optional concurrency runs
fire N requests at once to see behaviour under load.

    PRIME_API_KEY=... python scripts/bench_latency.py \
        --model deepseek/deepseek-v4.1-flash --model Qwen/Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE_URLS = {
    "prime": ("https://api.pinference.ai/api/v1", "PRIME_API_KEY"),
    "sail": ("https://api.sailresearch.com/v1", "SAIL_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}

TASK = (
    "You are a coding subagent. In under 300 words, explain where the orchestrator's "
    "budget is enforced and list the functions involved, citing file names."
)


def build_prompts() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1] / "subagents_bench"
    context = "\n\n".join(
        f"### {p.relative_to(root.parent)}\n{p.read_text()}" for p in sorted(root.rglob("*.py"))
    )
    return {
        "short": TASK,
        "long": f"Repository context:\n\n{context}\n\n{TASK}",
    }


def one_request(base: str, key: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    # Explicit User-Agent: the default "Python-urllib" one is rejected (403) by some gateways.
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "User-Agent": "subagents-bench/0.1"}
    if os.environ.get("PRIME_TEAM_ID"):
        headers["X-Prime-Team-ID"] = os.environ["PRIME_TEAM_ID"]
    if os.environ.get("SAIL_COMPLETION_WINDOW"):  # asap | balanced | flex
        headers["X-Sail-Completion-Window"] = os.environ["SAIL_COMPLETION_WINDOW"]
    req = urllib.request.Request(
        f"{base}/chat/completions", data=body, method="POST", headers=headers,
    )
    t0 = time.perf_counter()
    ttft = ttfa = None
    usage: dict = {}
    reasoning_chars = content_chars = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                r = delta.get("reasoning_content") or delta.get("reasoning") or ""
                c = delta.get("content") or ""
                if (r or c) and ttft is None:
                    ttft = time.perf_counter() - t0
                if c and ttfa is None:
                    ttfa = time.perf_counter() - t0
                reasoning_chars += len(r)
                content_chars += len(c)
    e2e = time.perf_counter() - t0
    out = usage.get("completion_tokens") or 0
    decode = e2e - (ttft or e2e)
    return {
        "ttft": ttft, "ttfa": ttfa, "e2e": e2e,
        "tps": (out - 1) / decode if out > 1 and decode > 0 else None,
        "in": usage.get("prompt_tokens"), "out": out,
        "reasoning_chars": reasoning_chars, "content_chars": content_chars,
        "cost": usage.get("cost"),
    }


def summarize(rows: list[dict], key: str) -> str:
    vals = sorted(r[key] for r in rows if r.get(key) is not None)
    if not vals:
        return "n/a"
    p90 = vals[min(len(vals) - 1, int(round(0.9 * (len(vals) - 1))))]
    return f"{statistics.median(vals):7.2f} (p90 {p90:7.2f})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--provider", default="prime", choices=BASE_URLS)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=4, help="0 to skip the load test")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", default=None, help="write raw rows as JSON")
    args = ap.parse_args()

    base, key_env = BASE_URLS[args.provider]
    key = os.environ.get(key_env)
    if not key:
        raise SystemExit(f"set {key_env}")
    prompts = build_prompts()
    raw: list[dict] = []

    def run(model: str, pname: str, mode: str) -> dict:
        try:
            r = one_request(base, key, model, prompts[pname], args.max_tokens)
        except Exception as e:  # keep going; report failures
            r = {"error": repr(e)}
        r.update(model=model, prompt=pname, mode=mode)
        raw.append(r)
        return r

    for model in args.model:
        run(model, "short", "warmup")
        for pname in prompts:
            for _ in range(args.trials):
                run(model, pname, "sequential")
            if args.concurrency:
                with ThreadPoolExecutor(args.concurrency) as pool:
                    list(pool.map(lambda _: run(model, pname, f"conc{args.concurrency}"),
                                  range(args.concurrency)))

    print(f"\n{'model':32} {'prompt':6} {'mode':10} {'n':>3} {'in':>6} {'out':>5} "
          f"{'TTFT s':>22} {'TTFA s':>22} {'decode tok/s':>22} {'e2e s':>22}")
    groups: dict[tuple, list[dict]] = {}
    for r in raw:
        if r["mode"] != "warmup":
            groups.setdefault((r["model"], r["prompt"], r["mode"]), []).append(r)
    for (model, pname, mode), rows in groups.items():
        ok = [r for r in rows if "error" not in r]
        errs = len(rows) - len(ok)
        ins = statistics.median([r["in"] for r in ok if r.get("in")] or [0])
        outs = statistics.median([r["out"] for r in ok if r.get("out")] or [0])
        print(f"{model:32} {pname:6} {mode:10} {len(ok):>3} {ins:>6.0f} {outs:>5.0f} "
              f"{summarize(ok, 'ttft'):>22} {summarize(ok, 'ttfa'):>22} {summarize(ok, 'tps'):>22} {summarize(ok, 'e2e'):>22}"
              + (f"  [{errs} errors]" if errs else ""))
    for r in raw:
        if "error" in r:
            print("error:", r["model"], r["prompt"], r["error"][:200])
    if args.out:
        Path(args.out).write_text(json.dumps(raw, indent=1))


if __name__ == "__main__":
    main()
