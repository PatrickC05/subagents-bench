# Pricing and model-fact sources

This page records how `configs/pricing_reference.yaml`, `configs/models.yaml`
and `configs/provider_models.json` were produced. The analysis step reads all
three and never fetches anything itself. The files are a snapshot. Re-fetching
will give different numbers, because provider prices and endpoint lists change.

**Snapshot time:** 2026-10-11, 21:08–21:16 America/New_York (01:08–01:16 UTC on 2026-10-12).

## What was fetched

| Source | URL | Used for |
|---|---|---|
| Sail model list (key) | `GET https://api.sailresearch.com/v1/models` | Sail model ids. The response has only `id`, `owned_by` and a placeholder `created`: no prices and no context. |
| Sail docs | https://docs.sailresearch.com/pricing, https://docs.sailresearch.com/models | Sail's published per-window prices (asap/balanced/flex), served context, and the checkpoint Sail says it serves |
| Prime model list (key + `X-Prime-Team-ID` + a User-Agent) | `GET https://api.pinference.ai/api/v1/models` | Prime model ids and prices. Context is given for gateway models and is null for the Prime-hosted Qwen3.5 models. |
| OpenRouter (public) | `GET https://openrouter.ai/api/v1/models` and `.../models/<author>/<slug>/endpoints` | Reference prices |
| Hugging Face (public) | `https://huggingface.co/api/models/<id>`, `https://huggingface.co/<id>/raw/main/config.json`, `README.md`, and the safetensors headers (HTTP range reads of the first `8 + N` bytes of each shard, with no weights downloaded) | Parameter counts, architecture, context, weight formats |
| Sail context probe | one `chat/completions` call to `google/gemma-4-12B-it` with a ~20K-token prompt and `max_tokens=1` | The exact served context of that model (it was rejected with HTTP 400, so nothing was generated) |

The HF revisions used are pinned per model under `checkpoint.revision` in
`models.yaml`.

### Model ids (all confirmed in the listings)

| Model | Sail | Prime | OpenRouter |
|---|---|---|---|
| DeepSeek-V4.1-Flash | `deepseek-ai/DeepSeek-V4.1-Flash` | `deepseek/deepseek-v4.1-flash` (gateway) | `deepseek/deepseek-v4.1-flash` |
| Gemma 4 31B NVFP4 | `nvidia/Gemma-4-31B-IT-NVFP4` | none | none (base model: `google/gemma-4-31b-it`) |
| Gemma 4 12B | `google/gemma-4-12B-it` | none | none |
| GLM-5.3-Flash | `zai-org/GLM-5.3-Flash` | `z-ai/glm-5.3-flash` (gateway) | `z-ai/glm-5.3-flash` |
| Qwen3.5 0.8B / 2B / 4B / 9B | none | `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B` (hosted; exact case) | only `qwen/qwen3.5-9b` |

Prime also lists lowercase gateway ids for other Qwen3.5 sizes (for example
`qwen/qwen3.5-35b-a3b`), but not for the four small models. Use the
mixed-case `Qwen/...` ids exactly as written above.

## Reference prices (`pricing_reference.yaml`)

For each model on OpenRouter, the prices come from its `/endpoints` listing:

1. Drop exact duplicate listings (same tag and prices). BaseTen fp8 appears
   twice for both DeepSeek and GLM.
2. For each endpoint, convert `pricing.prompt`, `pricing.completion` and
   `pricing.input_cache_read` (USD per token, as strings) to USD per 1M.
3. Take the median across endpoints. `cached` is the median over the
   endpoints that list a cache-read price. It is `null` when fewer than half
   list one: Qwen3.5-9B has 1 of 6, so a single provider does not set the
   value.
4. `:free`, `:batch` and other variant ids are separate OpenRouter models and
   are not included.

| Model | in | cached | out | endpoints / providers |
|---|---|---|---|---|
| DeepSeek-V4.1-Flash | 0.21 | 0.007 | 1.188 | 31 / 29 |
| GLM-5.3-Flash | 0.10875 | 0.03 | 0.50 | 32 / 29 |
| Gemma-4-31B (NVFP4 and BF16) | 0.14 | 0.10 | 0.40 | 13 / 12 |
| Qwen3.5-9B | 0.10 | null | 0.15 | 6 / 6 |

**No reference price:** `google/gemma-4-12B-it` and `Qwen/Qwen3.5-{0.8B,2B,4B}`
are not on OpenRouter. The only prices found for them are the serving
providers' own lists (Sail and Prime), which are already their own schemes.
Google's Gemini API pricing page lists Gemma 4 as free tier only, with no paid
price (updated 2026-10-09). Third-party aggregator pages quoting other prices
could not be traced to a primary source and were not used. Under the
reference scheme these models' cost is NaN.

**Caveats**

- **Discounts.** The listed endpoint prices already include each endpoint's
  `discount` field. For example, StreamLake lists DeepSeek at 0.147 / 0.588,
  which is 0.30 / 1.20 × (1 − 0.51). The discounted endpoints pull the
  DeepSeek and GLM input medians down: undiscounted they are 0.30 and 0.15,
  exactly the first-party DeepSeek and Z.AI prices. The yaml keeps both: the
  headline `in/cached/out` (as listed on the snapshot date) and
  `undiscounted`. Promotions are temporary, so report a sensitivity check with
  `undiscounted`.
- **Mixed quantization.** OpenRouter treats quantization (fp4, fp8, bf16,
  unknown) as an endpoint attribute, so each median mixes precisions. The
  NVFP4 Gemma model has no OpenRouter listing of its own and takes the
  `google/gemma-4-31b-it` median. Over the 5 fp4 endpoints alone the median is
  0.12 / 0.09 / 0.36.
- **The input medians are sensitive to the listing.** The DeepSeek prompt
  prices on either side of the median are 0.21 and 0.27. Keeping the
  duplicate BaseTen listing alone moves the median to 0.24. Treat the
  reference `in` as accurate to roughly ±0.05.
- **Sail is in the medians.** Sail Research is one of the 31–32 endpoints for
  DeepSeek and GLM, so the reference medians are not fully independent of Sail.
- **Unavailable endpoints are kept.** A few endpoints had a non-zero
  `status` (SambaNova for Gemma; AtlasCloud and SiliconFlow for GLM). Their
  listed prices are still counted.

## Provider price lists (`provider_models.json`, for the `sail_*` and `prime` schemes)

These are not used for `reference`; `configs/pricing.yaml` (owned by analysis)
builds the provider schemes from them.

| Model | Sail asap (default) | Sail balanced | Sail flex | Prime |
|---|---|---|---|---|
| deepseek-ai/DeepSeek-V4.1-Flash | 0.09 / 0.006 / 0.30 | 0.07 / 0.005 / 0.22 | 0.05 / 0.004 / 0.15 | 0.30 / – / 1.20 |
| nvidia/Gemma-4-31B-IT-NVFP4 | 0.14 / 0.07 / 0.40 | 0.11 / 0.06 / 0.32 | 0.07 / 0.04 / 0.20 | – |
| google/gemma-4-12B-it | 0.30 / 0.15 / 2.00 | 0.10 / 0.07 / 2.00 | 0.05 / 0.02 / 1.00 | – |
| zai-org/GLM-5.3-Flash | 0.11 / 0.02 / 0.35 | 0.08 / 0.02 / 0.28 | 0.05 / 0.01 / 0.18 | 0.15 / – / 0.50 |
| Qwen/Qwen3.5-0.8B / 2B / 4B / 9B | – | – | – | 0.04/0.036/0.08, 0.06/0.054/0.18, 0.10/0.09/0.30, 0.18/0.162/0.54 |

Values are in / cached / out, in USD per 1M tokens. Prime's gateway models
list no cache-read price, shown as "–". Prime also lists a cache-write price
equal to the input price for the hosted Qwen models.

## Model facts (`models.yaml`)

All counts come from the HF safetensors headers at the pinned revisions:

- A tensor's parameters are the product of its shape.
- Packed 4-bit tensors count two parameters per stored byte. These are the
  DeepSeek `I8` expert weights and the NVFP4 `U8` weights; for example, an
  `I8 [2304, 2560]` tensor holds a 2304 × 5120 weight.
- Quantization scale tensors count toward bytes but not parameters.
- The language model is the embeddings, decoder layers and LM head. It
  excludes vision and audio encoders, MTP/speculative-draft modules, and
  (DeepSeek only) the Engram lookup tables.
- Vendor statements are copied verbatim into `stated`.

### Active parameters (MoE)

```
active = (language_model − routed_experts) + routed_experts × k / E
```

| Model | Formula | Computed | Vendor (used) |
|---|---|---|---|
| DeepSeek-V4.1-Flash | routed = 40 layers × 384 × 3 × 5120 × 2304 = 543.582B; (551.881 − 543.582) + 543.582 × 6/384 | 16.79B | 16B decode, 8B prefill |
| GLM-5.3-Flash | routed = 42 sparse layers × 288 × 3 × 4096 × 2048 = 304.406B; (313.327 − 304.406) + 304.406 × 8/288 | 17.38B | 18B |

The DeepSeek computed figure includes the 0.66B input embedding, which is a
lookup rather than a matmul. DeepSeek-V4.1-Flash is a causal encoder-decoder:
a prompt token passes through the 20 encoder layers only, about 8B params. A
generated token passes through all 40 layers, about 16B. `active_params_b` is
16, the conventional per-token figure that Sail also quotes. The two phases
are given separately as `active_params_prefill_b` and
`active_params_decode_b`. With a single number, `2 × active × tokens`
overstates prompt-token FLOPs for this model by about 2×. The 196.6B Engram
tables ((384006168 + 384016682) rows × 256) are a sparse lookup. They are
excluded from both `total_params_b` (matching the card's "552B backbone") and
active params.

### Dense models

For dense models, `active_params_b = total_params_b` = the exact language-model
count. Vendor size labels are rounded and include the vision encoder. For
example, Qwen3.5-"4B" has a 4.206B language model, and its checkpoint is
4.660B with the vision encoder and MTP head. Gemma-4-31B's language model is
30.697B, matching the card's "30.7B total".

### `bytes_per_param` (roofline input)

```
bytes_per_param = bytes of the weights read per decoded token at batch 1 / their parameter count
```

These are the active weights (routed experts scaled by k/E), excluding the
input-embedding lookup. With tied embeddings the matrix is counted once, as
the LM head. Scales are included.

| Model | Value | Why it is not the nominal width |
|---|---|---|
| nvidia/Gemma-4-31B-IT-NVFP4 | 1.026 | Only the MLP is NVFP4, at 0.5 + 1/16 B/param. The config's `ignore` list keeps all self-attention, the embeddings/LM head and the vision tower in BF16 (2 B/param). The nominal 0.5 is kept as `bytes_per_param_nominal`. |
| GLM-5.3-Flash (served as nvidia/GLM-5.3-Flash-NVFP4) | 1.236 | Routed experts and dense MLPs are NVFP4; attention, shared experts, router and LM head are BF16. The native FP8 release is 1.339. |
| DeepSeek-V4.1-Flash | 0.808 | Routed experts FP4; attention and shared experts FP8; LM head BF16 |
| BF16 models (Gemma-4-31B/12B, Qwen3.5) | 2.0 | none |

The serving precision is known only where Sail's docs name a reference
checkpoint. OpenRouter also tags Sail's DeepSeek and GLM endpoints `fp4`.
Prime does not state a precision for the Qwen models, so their 2.0 assumes the
BF16 release.

### Context windows

`context_window` is the native window: the config's `max_position_embeddings`,
which matches each card. The served window can be smaller. Sail serves
`google/gemma-4-12B-it` at **16384** tokens: the probe returned
`context_length_exceeded`, with the message "exceeds safe limit of 15872 tokens
for model context window of 16384 tokens (reserves 512 tokens for
model-specific formatting)". This is recorded as `serving_context_window`, and
the run config should set `context_window: 16384` for this model. Sail's docs
list "1M" for DeepSeek and GLM and "256K"/"262K" for the Gemma 31B models. The
Qwen3.5 4B/9B cards allow YaRN extension to 1,010,000 tokens; the native
262,144 is what is recorded.

## Re-deriving

The raw HTTP responses were not committed. `configs/provider_models.json`
keeps the reduced Sail and Prime listings and, under `providers.openrouter`,
the de-duplicated per-endpoint table for each study model (tag, quantization,
status, discount, prices). The medians can be recomputed from it offline. To
check against the live listing (prices will have moved since the snapshot):

```bash
curl -s https://openrouter.ai/api/v1/models/deepseek/deepseek-v4.1-flash/endpoints | python3 -c '
import json, sys, statistics as s
e = json.load(sys.stdin)["data"]["endpoints"]
u = {(x.get("tag"), x["pricing"]["prompt"], x["pricing"]["completion"], x["pricing"].get("input_cache_read")): x for x in e}
print(len(e), len(u), s.median(float(x["pricing"]["prompt"]) * 1e6 for x in u.values()))'
```

At the snapshot this printed `32 31 0.21`.
