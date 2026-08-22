# DubbingStory — Vision Architecture & Dual-GPU Plan

Status: **implemented in the accompanying patch/notebooks**
Target runtime: **Google Colab / Kaggle, Qwen3-VL served locally with vLLM 0.27.1**
Primary validation hardware: **Kaggle NVIDIA T4 x2**

## 1. Goals

This change addresses two independent problems at the correct layer:

1. **Vision output runaway / `finish_reason=length`** must be prevented structurally, not by increasing `max_tokens`.
2. **Scene evidence quality and GPU utilization** must improve so scene analysis has better temporal coverage while T4 x2 can actually run useful work in parallel.

The 640-token per-scene limit is retained as a safety breaker. It is **not** increased.

## 2. Root Cause Recap

Observed failures were not context-window exhaustion. The failing calls still had substantial context budget available, but completion reached exactly the configured output cap while Qwen repeated legal JSON content such as repeated object/text array entries.

Examples from the supplied logs included patterns equivalent to:

- `text_visible: ["P", "P", "P", ...]`
- `visible_objects: ["person", "person", "person", ...]`
- `text_visible: ["Crankshaft", "Crankshaft", ...]`

`response_format={"type":"json_object"}` only guaranteed a JSON object. It did **not** bound array cardinality or string length, so repetition remained valid JSON until `max_tokens` stopped generation.

The logs were also interleaved because scene analysis already used a thread pool with concurrency > 1. Therefore a nearby `[Tokens] scene_X` line was not guaranteed to belong to the next `[Vision]` response line.

## 3. Implemented P0 — Strict Vision Output Contract

### 3.1 Decoder-constrained JSON Schema

A new module is added:

```text
dubbingstory/vision/schemas.py
```

Deep per-scene output is constrained at decode time with:

- `visible_objects`: maximum 8 items
- `text_visible`: maximum 4 items
- array item string length bounds
- all descriptive string fields have hard `maxLength`
- confidence/salience scores constrained to `[0, 1]`
- `additionalProperties: false`
- all required fields declared explicitly

Cheap screening and temporal-flow calls also get dedicated schemas.

For temporal chunks, the schema is generated dynamically from the actual scene IDs and bounds `scenes_enriched` to the current chunk size.

vLLM 0.27.x supports `response_format.type=json_schema`. Its xgrammar backend does not list `maxItems`, `minItems`, `maxLength`, or numeric min/max among its rejected JSON-Schema features; unsupported examples include `uniqueItems`, `contains`, and related keywords. For that reason, uniqueness is intentionally enforced after parsing rather than via `uniqueItems`.

### 3.2 No silent schema downgrade on local vLLM

Default configuration:

```yaml
vision:
  openai_structured_outputs: true
  openai_allow_schema_fallback: false
  openai_repetition_penalty: 1.05
```

The local vLLM path must keep the hard schema. Compatibility fallback to `json_object` exists as an opt-in setting for third-party OpenAI-compatible servers, but is disabled by default.

### 3.3 Secondary repetition guard

Requests include:

```text
repetition_penalty = 1.05
```

This is deliberately secondary. The schema is the primary guarantee.

### 3.4 Deterministic scene identity

The model no longer needs to generate `scene_id` or `time_range` for per-scene calls.

After the response is parsed, the pipeline overwrites/injects:

```text
scene_id
 time_range
 start_time
 end_time
 duration
```

from the known scene object. Concurrent requests can therefore never corrupt scene identity simply because the model emitted a wrong ID.

### 3.5 Defensive normalization

Even with structured decoding, parsed scene results are normalized:

- duplicate `visible_objects` removed case-insensitively
- duplicate `text_visible` removed case-insensitively
- arrays capped again in Python
- scores clamped to `[0, 1]`

This also protects deployments where schema fallback is explicitly enabled.

### 3.6 Scene-aware request tracing

Vision logs now include request/scene identity, e.g.:

```text
[Vision scene_094] token_budget ...
[Vision scene_111] token_budget ...
[Vision scene_094] finish_reason=stop ...
```

Interleaved parallel logs are therefore attributable to the correct scene.

### 3.7 Cache invalidation is architecture-aware

The vision cache key now includes:

- schema version
- cheap/deep analysis kind
- structured-output mode
- repetition penalty
- temperature
- max token configuration
- image mode
- model context setting

Decoder/schema-only changes will no longer accidentally reuse stale pre-fix cache entries.

## 4. Implemented P1 — Better Visual Evidence

### 4.1 Long-shot windowing

The important structural change is not “send more images from a 115-second scene.” Instead, long detected shots are split into bounded scene windows before vision analysis and before summary selection.

Default:

```yaml
segment:
  max_scene_duration: 15.0
```

A 115.117-second detected shot becomes 8 near-equal windows, each below 15 seconds. These windows are normal pipeline scenes, so the summary selector can choose only the useful portion instead of being forced to include/exclude one giant 115-second block.

Metadata is retained:

```text
source_scene_id
source_scene_part
source_scene_parts
```

### 4.2 Keyframe source path fixed

Direct extraction from the original source video now respects the requested strategy:

- `distributed`: evenly distributed timestamps
- `interval`: interval candidates capped by `max_per_scene`

The previous direct-source path ignored the strategy and always used `np.linspace`.

### 4.3 Image quality

Default direct-source vision frames are changed from roughly:

```text
max edge 640 / JPEG 82
```

to:

```text
max edge 768 / JPEG 88
```

This is intended to preserve more fine detail/OCR evidence while remaining modest for T4 inference.

Short scenes (<= ~6 s after boundary margin) use at most two frames to avoid near-duplicate visual tokens.

## 5. Implemented P1 — T4 x2 Hardware Architecture

### 5.1 Why the old notebook was wrong

The supplied notebook had a contradictory GPU branch: when at least one GPU existed it could set tensor parallel size to 2 while the printed message claimed tensor parallel size 1. More importantly, tensor-parallel 2 is not the desired topology for the current 2B model if the goal is parallel independent scene analysis.

### 5.2 New notebook deployment modes

Both returned notebooks expose:

```python
VLLM_PARALLEL_MODE = "auto"
VLLM_QUANTIZATION = None
```

Supported modes:

- `auto`
- `data_parallel`
- `tensor_parallel`
- `single`

### 5.3 Auto policy

For 2 GPUs:

| Model shape | Auto topology | Reason |
|---|---:|---|
| Qwen3-VL 2B FP16 | DP=2, TP=1 | one full model replica per T4; two independent scene requests can execute on separate GPUs |
| Qwen3-VL 4B FP16 | DP=2, TP=1 | expected to fit one 16GB T4 in this workload; switch manually to TP if runtime memory proves otherwise |
| 8B-class FP16/non-quantized | DP=1, TP=2 | safer default because one full FP16 model may not fit a single T4 with runtime overhead |
| quantized model/checkpoint | DP=2, TP=1 | prefer two replicas when the quantized model fits one T4; force TP manually if it does not |

The notebook prints the detected GPU names/VRAM and the selected DP/TP topology before starting the server.

### 5.4 Pipeline concurrency follows the serving topology

For current T4 x2 + 2B:

```text
vLLM: DP=2, TP=1
DubbingStory vision_concurrency=2
```

The pipeline now has a CLI option:

```text
--vision-concurrency N
```

This removes the previous hidden mismatch between the thread pool and serving topology.

### 5.5 vLLM generation settings

The notebook starts vLLM with:

```text
--generation-config vllm
```

so request-level temperature/repetition settings are not unexpectedly overridden by a Hugging Face model repository's `generation_config.json`.

The notebook targets/pins:

```text
vllm==0.27.1
```

for reproducibility of the structured-output and parallel-serving behavior used by this patch.

## 6. Model Upgrade Strategy (After Architecture Validation)

Do **not** change model and architecture simultaneously for the first A/B test.

Recommended sequence:

1. Run current `Qwen/Qwen3-VL-2B-Instruct` with this patch.
2. Confirm zero/near-zero runaway `finish_reason=length`, correct scene IDs, and improved long-scene selection.
3. Benchmark 4B FP16.
4. Benchmark a validated 8B quantized checkpoint suitable for Turing/T4 (AWQ/GPTQ are supported on Turing in current vLLM documentation).
5. Compare factual accuracy, action/change understanding, OCR, hallucination rate, latency, VRAM, and final recap quality.

FP8 should not be the default T4 path. Current vLLM hardware tables list Turing support for AWQ/GPTQ while FP8 paths target newer architectures.

## 7. Runtime Output Locations

The **plan file itself** is returned as:

```text
DUBBINGSTORY_VISION_ARCHITECTURE_PLAN.md
```

When the notebook runs, diagnostics remain easy to inspect:

```text
<notebook working directory>/
├── pipeline.log
├── vllm_server.log
└── outputs/<PROJECT_NAME>/
    ├── segment_manifest.json
    ├── keyframes/
    ├── vision_cache/
    ├── failed_vision.log          # only useful when failures occur
    ├── storyboard.json
    ├── scene_cards.json
    ├── story_plan.json
    ├── story_memory.json
    ├── summary_manifest.json      # summary mode
    ├── narration_plan.json
    ├── narration_qa.json
    └── scripts/...
```

Important validation fields in `segment_manifest.json` for windowed long shots are `source_scene_id`, `source_scene_part`, and `source_scene_parts`.

## 8. Acceptance Criteria

### Vision stability

- Normal structured scene calls finish with `finish_reason=stop`.
- No unbounded `visible_objects` or `text_visible` arrays are possible under the schema.
- A repeated string cannot grow indefinitely because schema string lengths are bounded.
- Model output cannot alter pipeline scene identity/timing.
- If an unexpected length failure still occurs, the existing one-middle-frame rescue remains available.

### Temporal / visual quality

- No detected scene presented to vision is longer than the configured `max_scene_duration` unless that option is disabled.
- A previously ~115-second scene should become multiple ~14-15 second windows.
- Summary selection can choose individual windows.
- Direct-source keyframe extraction respects its configured strategy.

### Hardware

On Kaggle T4 x2 with the default 2B model, startup log should report approximately:

```text
DP=2, TP=1, vision_concurrency=2
```

GPU monitoring should show both T4s holding a model replica and receiving work during parallel scene analysis.

## 9. First Test Procedure

Use the same video/model/settings as the failing baseline so only architecture changes.

Before the first comparison run, use a new project name or remove the old project's `vision_cache`. The patch changes the cache fingerprint, so stale entries should not match, but a clean A/B directory makes comparison easier.

Watch for these log patterns:

```text
[Vision scene_XXX] ... schema=on
[Vision scene_XXX] finish_reason=stop
```

Then inspect:

1. `segment_manifest.json` — confirm long-shot windowing.
2. `storyboard.json` / `scene_cards.json` — check action/change evidence and continuity.
3. `summary_manifest.json` — confirm giant scenes are no longer selected as indivisible blocks.
4. `pipeline.log` — count any `finish_reason=length` occurrences.
5. `vllm_server.log` plus GPU monitor — verify both GPUs are used in DP mode.
6. Final narration/TTS/video — compare explanation accuracy and clip pacing against the previous run.

## 10. Rollback / Tuning Knobs

If the long-shot splitting is too aggressive:

```text
--max-scene-duration 18
```

or disable it via configuration with a null/zero value.

If 4B FP16 OOMs in DP mode:

```python
VLLM_PARALLEL_MODE = "tensor_parallel"
```

If a quantized checkpoint fits one T4 and you want maximum scene throughput:

```python
VLLM_PARALLEL_MODE = "data_parallel"
```

Do not solve repetition by increasing `VISION_MAX_TOKENS`; keep the structural constraint and investigate any remaining failure field.

## 11. Files Changed by the Git Patch

```text
configs/default.yaml
dubbingstory/cli.py
dubbingstory/config.py
dubbingstory/segment/keyframes.py
dubbingstory/segment/scene_detect.py
dubbingstory/vision/openai_vision.py
dubbingstory/vision/prompts.py
dubbingstory/vision/scene_understanding.py
dubbingstory/vision/schemas.py                 # new
DUBBINGSTORY_VISION_ARCHITECTURE_PLAN.md        # this document
```

The two notebook deliverables are separate from the repository patch:

```text
dubbingstory-qwen-local-colab-architecture-fix.ipynb
dubbingstory-qwen-local-kaggle-architecture-fix.ipynb
```

## 12. External Compatibility Notes

Implementation was aligned with current vLLM documentation around August 2026:

- OpenAI-compatible server supports `response_format` with `json_schema`.
- vLLM structured outputs support JSON Schema through structured-output backends such as xgrammar.
- vLLM internal data parallelism can run multiple local engine cores/workers; external-DP restrictions for non-MoE models are a different mode and are not used by this notebook.
- `--generation-config vllm` disables model-repository generation defaults.
- Turing supports AWQ/GPTQ in the current vLLM quantization compatibility table.

References:

- https://docs.vllm.ai/en/v0.27.1/api/vllm/config/parallel/
- https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/
- https://docs.vllm.ai/en/latest/features/structured_outputs/
- https://docs.vllm.ai/en/v0.27.0/api/vllm/v1/structured_output/backend_xgrammar.html
- https://docs.vllm.ai/en/latest/cli/serve/
- https://docs.vllm.ai/en/latest/features/quantization/
