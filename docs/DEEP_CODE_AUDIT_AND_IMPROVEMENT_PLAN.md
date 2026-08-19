# DubbingStory: Deep Code Audit and Improvement Plan

Tanggal audit: 2026-08-19

Scope audit:

- Seluruh modul Python di `dubbingstory/`
- Entry point, konfigurasi, dependency metadata, README, changelog, dan notebook
- Alur lokal: ingest -> segment -> analyze -> narrate -> dub -> render
- Perbandingan konsep dengan `movie-narrator`, khususnya pipeline orchestration, status, artifact, recovery, dan QA

Validasi baseline:

- `python3 -m compileall -q dubbingstory main.py`: lulus
- Test file aplikasi yang terdeteksi: belum ada
- End-to-end video run: belum dijalankan karena membutuhkan asset video, API/provider, dan runtime media yang sesuai

## 1. Executive Summary

DubbingStory memiliki fondasi produk yang tepat untuk automated narration/dubbing:

- Pipeline domain sudah jelas dan mudah dipahami.
- Ingest memiliki fallback subtitle lokal -> subtitle YouTube -> ASR.
- Vision memiliki dua provider dan retry/context-budget logic yang cukup matang.
- Summary mode sudah memiliki scene scoring, two-pass screening, temporal chunking, dan timeline remapping.
- TTS sudah menggunakan interface engine.
- Output disimpan sebagai artifact yang dapat dipakai ulang antar command.

Kelemahan utamanya bukan kurangnya fitur AI. Kelemahan utama ada pada reliability boundary dan kontrak eksekusi:

1. CLI masih menjadi orchestrator sekaligus pemilik business flow.
2. Artifact tersedia, tetapi belum memiliki schema, provenance, checksum, atau status yang konsisten.
3. Banyak fallback menangkap `Exception` dan melanjutkan tanpa status degraded yang dapat dibaca mesin.
4. Tidak ada quality gate final yang memeriksa video hasil render.
5. Tidak ada resume/checkpoint dan belum ada test suite.
6. Konfigurasi runtime tersebar antara YAML, `SimpleNamespace`, CLI mutation, dan dependency files.
7. Timing audio, subtitle, clip, dan source video belum memiliki satu kontrak waktu yang eksplisit.

Kesimpulan: DubbingStory sudah berada pada tahap prototype yang kaya fitur, tetapi belum pada tahap pipeline produksi yang dapat diaudit dan dipulihkan secara deterministik. Prioritas pertama sebaiknya reliability, bukan menambah provider atau Web UI.

## 2. Current Architecture

```text
CLI argparse
  |
  +-- build_config() -> flattened SimpleNamespace
  |
  +-- cmd_ingest()
  |     +-- local_video / youtube
  |     +-- subtitle_reader
  |     +-- optional Faster-Whisper ASR
  |     +-- ingest_manifest.json
  |
  +-- cmd_segment()
  |     +-- PySceneDetect
  |     +-- optional physical scene split
  |     +-- keyframe extraction
  |     +-- segment_manifest.json
  |
  +-- cmd_analyze()
  |     +-- Gemini or OpenAI-compatible Vision
  |     +-- per-scene cache
  |     +-- temporal flow analysis
  |     +-- storyboard.json
  |
  +-- cmd_narrate() or cmd_summary()
  |     +-- Gemini text generation
  |     +-- fallback narration
  |     +-- .txt and .srt scripts
  |
  +-- cmd_dub()
  |     +-- Edge TTS or Piper
  |     +-- per-scene WAV generation
  |     +-- concatenated audio
  |
  +-- cmd_render() or summary render
        +-- optional ratio rescale
        +-- audio mix
        +-- optional subtitle burn
        +-- render_manifest.json
```

### Architectural strengths

- `cli.py` exposes useful, independently rerunnable subcommands.
- Manifest boundaries already make manual inspection possible.
- `scene_understanding.py` has a reasonable separation between per-scene analysis and temporal reasoning.
- `openai_vision.py` accounts for image token cost, context budget, JSON parsing, and bounded network/format retries.
- `scene_understanding.py` uses cache keys containing prompt, model, and keyframe metadata.
- `video_cutter.py` records actual clip durations to address frame rounding.
- `voice_manager.py` isolates provider selection from segment concatenation and duration alignment.
- Rights confirmation exists before URL ingestion.

## 3. High-Severity Findings

### H1. Summary mode is not propagated into Vision analysis

Evidence:

- `scene_understanding.run_analysis()` decides whether to run cheap screening using `cfg.mode` in [scene_understanding.py](../dubbingstory/vision/scene_understanding.py#L300).
- `cmd_run()` reads `args.mode`, but only uses it for choosing `cmd_summary()` versus the full path in [cli.py](../dubbingstory/cli.py#L453).
- CLI override handling never assigns `cfg.mode = args.mode` in [cli.py](../dubbingstory/cli.py#L880).

Impact:

- `dubbingstory run --mode summary` can execute normal per-scene deep analysis for every scene.
- The documented two-pass optimization is not reliably active through the main CLI.
- Cost, latency, and GPU memory usage can grow substantially for long videos.

Recommended fix:

- Make mode part of an explicit pipeline context, not an implicit config lookup.
- As an immediate compatibility fix, assign the parsed mode before `cmd_analyze()`.
- Add a regression test asserting summary mode calls cheap screening and does not deep-analyze unselected scenes.

Priority: P0.

### H2. Render failures can produce a successful-looking pipeline

Evidence:

- `render_project()` catches every render exception per language/ratio, prints an error, and continues in [video_render.py](../dubbingstory/render/video_render.py#L206).
- It writes `render_manifest.json` and returns the successfully rendered subset.
- `cmd_render()` ignores the returned list and always prints `Render selesai` in [cli.py](../dubbingstory/cli.py#L427).
- Summary rendering behaves similarly: exceptions are printed and the command still reaches `Summary pipeline selesai` in [cli.py](../dubbingstory/cli.py#L660).

Impact:

- A run with zero final videos may be reported as successful.
- Automation cannot distinguish complete, partial, and failed output.
- Users may discover failure only by inspecting the output directory manually.

Recommended fix:

- Define statuses: `success`, `partial`, `failed`, `skipped`.
- Return a structured render result containing attempted, succeeded, failed, and artifact paths.
- Make `run` exit non-zero when no requested output was produced.
- Add `--strict` to turn partial optional failures into a hard failure.

Priority: P0.

### H3. There is no final deliverable QA gate

Evidence:

- [video_render.py](../dubbingstory/render/video_render.py#L58) assembles the output but does not validate streams, duration, audio level, or frame content.
- [render_manifest.json](../dubbingstory/render/video_render.py#L228) records only paths, strategy, and ratios.
- FFmpeg success is treated as equivalent to content correctness.

Impact:

Possible undetected outputs include:

- Missing audio stream.
- Audio much shorter than video because of `-shortest`.
- Silent or nearly silent narration.
- Black frames or invalid video stream.
- Incorrect vertical output framing.
- Output duration outside an acceptable tolerance.

Recommended fix:

Create `dubbingstory/render/qa.py` with:

- `ffprobe` stream validation.
- Duration ratio checks.
- Audio silence threshold check.
- Sampled-frame black ratio check.
- File size and readability checks.
- JSON `qa_report.json` persisted per output.

Priority: P0.

### H4. Timing has multiple sources of truth

Evidence:

- Scene timing originates in PySceneDetect.
- Storyboard duration is recomputed as `sum(scene["duration"])` in [_build_storyboard()](../dubbingstory/vision/scene_understanding.py#L480), rather than copied from source video metadata.
- Narration validation accepts LLM-provided `estimated_duration` without enforcing scene bounds in [script_writer.py](../dubbingstory/story/script_writer.py#L122).
- Normal-mode SRT uses original scene timestamps, while summary mode remaps timestamps manually in [cli.py](../dubbingstory/cli.py#L577).
- TTS concatenation relies on filename parsing and file order in [voice_manager.py](../dubbingstory/tts/voice_manager.py#L52).
- Final render uses `-shortest` in [audio_mix.py](../dubbingstory/render/audio_mix.py#L76), which can silently shorten the video to audio duration.

Impact:

- Subtitle, narration, and video can drift without an explicit validation failure.
- A failed or omitted scene can change cumulative timing.
- Full and summary modes use different timing logic that is easy to regress.

Recommended fix:

- Introduce one typed `TimelineSegment` contract with source and output coordinates.
- Keep source timeline and output timeline separate:

```text
source_start, source_end
output_start, output_end
scene_id, language, text
```

- Generate subtitles and audio assembly from the same segment list.
- Probe final audio duration and compare it with output timeline before render completion.

Priority: P0.

## 4. Medium-Severity Findings

### M1. Pipeline orchestration is coupled to CLI command handlers

`cmd_run()` calls `cmd_ingest()`, `cmd_segment()`, `cmd_analyze()`, and so on in [cli.py](../dubbingstory/cli.py#L453). These functions both perform work and own CLI concerns such as printing and `sys.exit()`.

Consequences:

- Web/API integration would need to call CLI-shaped functions.
- Unit testing requires constructing argparse-like objects.
- Retry or resume must understand command internals.
- Summary mode duplicates substantial orchestration inside `cmd_summary()`.

Plan:

- Extract pure/service-level functions first.
- Add `pipeline/context.py`, `pipeline/steps.py`, and `pipeline/runner.py`.
- Keep CLI as a thin adapter that parses arguments and renders status.

Priority: P1.

### M2. Artifact manifests are inconsistent and weakly validated

Existing files include `video_metadata.json`, `ingest_manifest.json`, `segment_manifest.json`, `storyboard.json`, `summary_manifest.json`, and `render_manifest.json`. Their shapes and status fields differ.

Missing common fields:

- Schema/version.
- Run ID.
- Created/updated timestamps.
- Input artifact reference.
- Config/model/provider provenance.
- Error and warning details.
- Checksum or file fingerprint.
- Step duration.

Plan:

Create a common manifest envelope:

```json
{
  "schema_version": 1,
  "run_id": "...",
  "step": "analyze",
  "status": "success",
  "created_at": "...",
  "inputs": {},
  "outputs": {},
  "provider": "openai",
  "config_fingerprint": "...",
  "warnings": [],
  "errors": []
}
```

Priority: P1.

### M3. Fallbacks are not observable as degraded output

Examples:

- Per-scene Vision failure returns `confidence: 0.0` and an `error`, but the final storyboard does not preserve a global degraded-step summary.
- Temporal failure creates neutral enrichment with no explicit failure count.
- Script generation failure silently falls back to generated generic narration in [script_writer.py](../dubbingstory/story/script_writer.py#L115).
- TTS generation logs per-segment failure but may still concatenate remaining segments in [voice_manager.py](../dubbingstory/tts/voice_manager.py#L161).
- YouTube metadata failure is printed and ignored in [youtube.py](../dubbingstory/ingest/youtube.py#L145).

Impact:

The product can produce a video that looks complete while silently omitting scenes, using generic narration, or having partial audio.

Plan:

- Add step-level and artifact-level warnings.
- Record counts: requested, completed, failed, fallback, cached.
- Surface a final quality summary in both terminal and manifest.

Priority: P1.

### M4. Configuration is permissive and has drift risk

[config.py](../dubbingstory/config.py#L80) flattens YAML into `SimpleNamespace`. Unknown YAML keys are accepted. CLI overrides mutate attributes dynamically.

Concrete drift examples:

- `default.yaml` documents `vision.openai_max_tokens: 2048`, while runtime and changelog have multiple historical defaults around 1024/2048.
- `default.yaml` includes `vision.batch_size`, but the analyzer orchestration does not use it.
- `default.yaml` mentions `voxcpm2`, while the actual engine factory only supports Edge and Piper.
- `pyproject.toml` does not list all runtime packages present in `requirements.txt`, reducing the reliability of `pip install -e .`.

Plan:

- Add typed config schema for critical fields.
- Reject unknown keys in a strict validation mode.
- Separate provider credentials from pipeline behavior.
- Generate or test documentation against schema defaults.
- Make `pyproject.toml` the authoritative install metadata.

Priority: P1.

### M5. Dependency metadata does not have one source of truth

`requirements.txt` contains `edge-tts`, `faster-whisper`, and `tqdm`, while [pyproject.toml](../pyproject.toml#L8) does not declare them. The project README tells users to run `pip install -e .`, which may not install all paths used by the application.

The same issue applies to optional media/ASR behavior: imports are lazy, but the installation contract is not explicit enough about which command needs which package.

Plan:

- Move dependencies into project extras such as `[media]`, `[asr]`, `[openai-vision]`, and `[all]`.
- Test fresh installation in CI.
- Make missing optional dependencies produce command-specific guidance.

Priority: P1.

### M6. TTS concatenation is based on text-file parsing rather than structured script data

The script is written as lines like `[scene_id] text` and later parsed with `r'\[(\w+)\]\s*(.*)'` in [voice_manager.py](../dubbingstory/tts/voice_manager.py#L112).

Risks:

- Scene IDs containing characters outside `\w` are truncated or misread.
- Multiline narration is not represented cleanly.
- Notes, estimated durations, and original timestamps are discarded.
- `unknown` segments can be generated and mixed into output.
- Audio order depends on sorted filenames/manifest order, not an explicit timeline.

Plan:

- Save canonical `script_{lang}.json` beside TXT/SRT.
- Generate TTS from JSON segments.
- Keep TXT as a human-readable export only.
- Store per-segment audio metadata and concatenation order.

Priority: P1.

### M7. Audio mixing assumes the source video has an audio stream for duck mode

The duck filter references `[0:a]` directly in [audio_mix.py](../dubbingstory/render/audio_mix.py#L65). A silent video without an audio track will fail, while `mute_original` and `replace` can work.

Plan:

- Probe source streams before selecting the filter.
- Automatically use narration-only mode when no original audio exists.
- Record the selected strategy and reason in QA metadata.

Priority: P1.

### M8. Per-scene Vision concurrency has no provider-aware rate limit or cancellation

`ThreadPoolExecutor` is used in [scene_understanding.py](../dubbingstory/vision/scene_understanding.py#L304) and [scene_understanding.py](../dubbingstory/vision/scene_understanding.py#L319). The configured worker count is not validated against provider quotas, and there is no cancellation or retry budget at the pipeline level.

Risks:

- Rate-limit amplification.
- Too many simultaneous local model requests.
- Difficult interruption of long-running runs.
- A single provider outage creates many repeated failures.

Plan:

- Add bounded provider concurrency and a shared retry policy.
- Add exponential backoff with jitter.
- Expose progress and cancellation at scene boundaries.
- Record cache hits/misses and attempts.

Priority: P1.

## 5. Low-Severity and Maintainability Findings

### L1. Dead or misleading code/config remains

Examples:

- `scene_detect.py` imports `split_video_ffmpeg`, `FrameTimecode`, and `tempfile` without meaningful use in the visible paths.
- `keyframes.extract_keyframes_from_source()` imports NumPy locally and has a `max_edge` parameter that is not configurable through the CLI.
- `GeminiVideoAnalyzer._generate_with_retry()` accepts `response_schema` but does not use it.
- `BaseTTSEngine` documentation still mentions engines that are no longer the default project path.
- README describes files or scripts that may not exist in the current tree, such as a unified `dubbingstory_colab.py` reference.

Plan: perform a focused cleanup only after reliability work, preserving behavior.

### L2. Output directory safety is not formalized

Project names are derived from file names or URL paths in [cli.py](../dubbingstory/cli.py#L37), but there is no general sanitization against path separators, reserved names, or collisions.

Plan: add `sanitize_project_name()` and a run-specific directory policy.

### L3. Repeated runs can leave stale artifacts

`ffmpeg -y` overwrites individual outputs, but old languages, ratios, scripts, cache entries, summary audio, and previous manifests can remain in the same project directory.

Plan:

- Write an explicit run ID.
- Use run-scoped temporary files.
- Mark current artifacts in a manifest.
- Provide a safe `clean` command rather than deleting implicitly.

### L4. Logging is human-only

Most diagnostics use `print()`. This is useful for notebooks, but not sufficient for machine-readable progress, CI, or a future web UI.

Plan: introduce a small logger/console interface while keeping the existing terminal output as the default adapter.

### L5. No static/runtime contract tests

There are no detected test files. Core pure logic is testable without API calls:

- config merge and validation,
- scene merging,
- keyframe position calculation,
- summary selection,
- SRT formatting/wrapping,
- timeline remapping,
- TTS duration alignment command construction,
- manifest transitions.

## 6. Data Contract Audit

### Ingest contract

Current output: `ingest_manifest.json`.

Good:

- Carries video path, subtitle data, context source, ASR usage, and status.

Missing:

- Source type and original input.
- Video fingerprint.
- Metadata path reference.
- Exact fallback attempt history.
- Validation result for video stream/audio stream.

### Segment contract

Current output: `segment_manifest.json`.

Good:

- Scene IDs, timestamps, keyframe paths, and total scene count.

Risks:

- Scene IDs are regenerated after merging.
- `scene_index` and persisted IDs can become stale if a manifest is edited manually.
- No invariant validation that scenes are ordered, non-overlapping, and within source duration.
- Keyframe paths are not checked before downstream use.

### Storyboard contract

Current output: `storyboard.json`.

Good:

- Separates visual analysis from narrative enrichment.
- Preserves confidence and transcript source.

Risks:

- Free-form dictionaries have no schema validation.
- Missing temporal enrichment silently becomes default `process`, importance `0.5`.
- `total_duration` is derived from scene durations, not independently validated.
- Error fields from scene analysis are not promoted to a storyboard quality summary.

### Script contract

Current output: TXT and SRT only.

Risks:

- TXT is both human export and machine input.
- LLM output is only lightly validated.
- Unknown scene IDs and duplicate scene IDs are not rejected.
- Segment timestamps are not checked for monotonicity or overlap.
- LLM `estimated_duration` is not reconciled with actual scene duration.

### Audio contract

Current output: per-language concatenated WAV plus per-scene WAVs.

Risks:

- No audio manifest maps each segment to exact sample/duration offsets.
- Failed TTS segments can be omitted without a hard contract decision.
- Concatenation assumes compatible WAV properties.
- No loudness or silence QA.

### Render contract

Current output: final MP4 files and `render_manifest.json`.

Risks:

- Manifest does not distinguish requested from successful outputs.
- No output QA report.
- No source/output duration relationship.
- No codec, resolution, stream, or encoder metadata.

## 7. Comparison With movie-narrator

### Concepts worth adopting

1. A central mutable context passed to steps.
2. A registry or ordered list of steps owned by the pipeline runner.
3. Hard versus soft step policy.
4. `strict` mode for production/CI.
5. Structured step result with duration, provider, attempts, cache hit, and artifact.
6. Separate execution manifest from product metadata.
7. Checkpoint/resume after completed steps.
8. Final QA gate based on `ffprobe` and media sampling.
9. Typed job configuration with rejected unknown keys.
10. Preview/draft mode for fast iteration.
11. Content-addressable caching for TTS and other deterministic artifacts.

### Concepts to defer

These add operational complexity before the local engine is reliable:

- Web UI and separate API package.
- Remote task queue.
- Distributed rendering.
- Scheduler and dead-letter queue.
- Plugin entry points.
- Circuit breaker infrastructure.
- Multi-candidate race generation.
- External movie research/TMDB integration.

The correct order is to stabilize local contracts first, then expose the engine through a web or queue boundary.

## 8. Recommended Target Architecture

```text
CLI / Notebook / Future API
          |
          v
    build_context()
          |
          v
    run_pipeline(ctx)
          |
          +--> ingest      [hard]
          +--> segment    [hard]
          +--> analyze    [soft/strict]
          +--> narrate    [hard]
          +--> select     [hard in summary mode]
          +--> dub        [hard]
          +--> render     [hard]
          +--> qa         [hard for publish, soft for preview]
          |
          v
  execution_manifest.json
  metadata.json
  qa_report.json
```

Suggested context shape:

```python
@dataclass
class PipelineContext:
    project_dir: Path
    config: PipelineConfig
    mode: str
    artifacts: dict[str, str]
    status: dict[str, str]
    warnings: list[dict]
    errors: list[dict]
    metrics: dict[str, object]
```

Suggested step result shape:

```python
@dataclass
class StepResult:
    status: Literal["success", "partial", "failed", "skipped"]
    artifacts: dict[str, str]
    warnings: list[str]
    metrics: dict[str, object]
```

The initial implementation should remain sequential and explicit. A registry, plugins, and asynchronous task queue can be added later without changing the step contract.

## 9. Improvement Plan

### Phase 0: Regression protection

Goal: create a safety net before refactoring.

Tasks:

1. Add `tests/` with pure unit tests.
2. Test `build_config()` precedence and flattening.
3. Test `_merge_short_scenes()` invariants.
4. Test summary selection duration and chronology.
5. Test SRT timestamp formatting and wrapping.
6. Test summary timeline remapping.
7. Test TTS duration alignment command decisions.
8. Add a fake Vision analyzer and fake TTS engine for pipeline tests.

Exit criteria:

- Tests run without network, GPU, or FFmpeg media assets.
- Existing pure behavior is captured before architectural changes.

Priority: P0.

### Phase 1: Fix correctness blockers

Goal: ensure the documented CLI modes and output status are truthful.

Tasks:

1. Propagate `args.mode` into analysis context/config.
2. Add validation that requested summary mode actually selects the summary analysis path.
3. Make render return structured success/partial/failure results.
4. Make top-level `run` return non-zero when required artifacts are missing.
5. Add source audio detection before selecting duck mode.
6. Validate scene ordering, non-overlap, and duration bounds.
7. Validate narration scene IDs, duplicates, timestamps, and empty output.

Exit criteria:

- `run --mode summary` uses cheap screening plus deep analysis only for selected scenes.
- A failed render cannot print unconditional success.
- Invalid intermediate artifacts fail with actionable messages.

Priority: P0.

### Phase 2: Introduce pipeline context and runner

Goal: move orchestration out of CLI while preserving CLI behavior.

Tasks:

1. Add `dubbingstory/pipeline/context.py`.
2. Add `dubbingstory/pipeline/results.py`.
3. Add `dubbingstory/pipeline/steps.py` with thin adapters around existing modules.
4. Add `dubbingstory/pipeline/runner.py` with an ordered step list.
5. Keep `cmd_ingest`, `cmd_segment`, and other commands as thin adapters for backward compatibility.
6. Move printing and `sys.exit()` to CLI boundaries.
7. Add hard/soft policy and `--strict`.

Exit criteria:

- CLI, a future notebook, and a future API can call the same runner.
- Each step has one input/output contract and one result status.
- Summary and full mode share the same runner with different step policy.

Priority: P1.

### Phase 3: Standardize manifests and observability

Goal: make every run inspectable and debuggable.

Tasks:

1. Add common manifest envelope and schema version.
2. Record run ID, timestamps, provider, model, config fingerprint, and input fingerprint.
3. Add `execution_manifest.json` with per-step duration and attempt count.
4. Record cache hits/misses for Vision and TTS.
5. Record fallback counts and degraded steps.
6. Add machine-readable JSON logging while preserving human console output.

Exit criteria:

- A user can identify why a run was slow, degraded, or incomplete by reading one manifest.
- Re-running with the same inputs makes cache behavior explicit.

Priority: P1.

### Phase 4: Establish timing and artifact contracts

Goal: prevent audio/video/subtitle drift.

Tasks:

1. Add typed `Scene`, `NarrationSegment`, `TimedAudioSegment`, and `TimelineSegment` models.
2. Persist canonical script JSON per language.
3. Generate TXT and SRT as exports from canonical JSON.
4. Store source and output timeline coordinates separately.
5. Build concatenated audio from structured segment order.
6. Validate final audio duration against output video timeline.
7. Add tests for full and summary timeline mapping.

Exit criteria:

- No downstream step parses the human-readable TXT as its canonical input.
- A segment can be traced from storyboard -> script -> audio -> final timeline.

Priority: P1.

### Phase 5: Add QA and preview mode

Goal: reduce wasted API/render time and prevent invalid deliverables.

Tasks:

1. Implement `render/qa.py`.
2. Persist `qa_report.json`.
3. Add publish thresholds for duration, silence, streams, and black frames.
4. Add `--preview` with short duration and fast FFmpeg settings.
5. Add a `draft` render profile.
6. Make QA hard-fail for publish mode and warning-only for preview mode.

Exit criteria:

- Every final artifact has a pass/fail QA report.
- A user can cheaply inspect script, audio, and timing before full render.

Priority: P1.

### Phase 6: Resume, caching, and provider hardening

Goal: make long jobs recoverable and cheaper.

Tasks:

1. Add `pipeline_state.json` after each completed step.
2. Add `resume --project` with input/config compatibility checks.
3. Extend cache to narration and TTS using content-addressed keys.
4. Centralize retry policy with exponential backoff and jitter.
5. Add provider-aware concurrency limits.
6. Add cancellation checks between scene and pipeline steps.

Exit criteria:

- A failed long run can resume without repeating completed expensive work.
- Cache invalidation is deterministic when model, prompt, voice, or source changes.

Priority: P2.

### Phase 7: Extensibility and product surface

Goal: prepare for multiple consumers after the core engine is stable.

Tasks:

1. Add provider registry for Vision, TTS, ASR, and text generation.
2. Add validated job YAML with strict unknown-key handling.
3. Expose a stable engine API for a future web UI.
4. Add optional async local queue.
5. Add multi-candidate narration only after script QA metrics exist.

Exit criteria:

- New providers do not require editing the runner.
- Web/API integration uses the same pipeline runner as CLI.

Priority: P2/P3.

## 10. First Implementation Backlog

Recommended order for actual coding:

1. Add regression tests for summary mode, scene selection, SRT, and manifests.
2. Fix mode propagation from CLI to Vision analysis.
3. Add structured render result and correct top-level exit status.
4. Add source audio fallback for duck mode.
5. Add basic `render/qa.py` and `qa_report.json`.
6. Add `PipelineContext` and migrate only `run` first.
7. Add common execution manifest.
8. Replace TXT parsing with canonical script JSON.
9. Add strict config validation.
10. Add resume and extended cache.

## 11. Definition of Done for the Reliability Milestone

The first milestone should not be considered complete until all statements below are true:

- Full and summary mode execute the intended analysis path.
- Every requested language/ratio has an explicit status.
- Pipeline success means at least the required final artifact passed QA.
- Pipeline warnings and fallbacks are visible in terminal and JSON manifest.
- Final video has valid video and audio streams.
- Final duration is within configured tolerance.
- Subtitle timestamps are on the same timeline as the rendered video.
- Failed intermediate artifacts cannot be silently reused.
- A clean installation contains all dependencies required by the documented commands.
- Core logic has automated tests that do not require an API key or GPU.

## 12. Final Assessment

DubbingStory's product direction is sound and its Vision optimization work is the strongest part of the codebase. The changelog shows active response to real issues such as context overflow, summary desynchronization, and expensive scene splitting.

The next maturity step is to convert those tactical fixes into explicit system contracts. The most important architectural lesson from `movie-narrator` is not its cloud functionality; it is the discipline of treating every pipeline step as an observable unit with a typed result, declared failure policy, and durable artifact record.

Recommended strategic focus:

```text
correctness -> observability -> timing contracts -> QA -> resume/cache -> extensibility
```

Do not prioritize Web UI, distributed execution, or a large plugin system before the first five items are stable. Those features will be much easier once the local runner has one honest, testable contract.

## 13. Deep Research: Vision and Narration Synchronization

This section adds findings from a deeper review of `movie-narrator` and `VIDEOFCK`, focused specifically on visual understanding and dubbing synchronization.

### 13.1 What VIDEOFCK actually does

The most useful part of VIDEOFCK is its simple, explicit timeline model:

1. Extract keyframes either by a fixed number or by frame-difference threshold.
2. Enforce a minimum time separation between threshold-selected frames.
3. Group keyframes into logical segments.
4. Ask an LLM for caption/narration data with timestamps.
5. Save the generated data as `narration_script.json`.
6. Synthesize one audio file per narration segment.
7. Place each segment at its assigned timestamp on a full-duration audio array.
8. Trim audio that exceeds the segment boundary.
9. Mix the original audio at a user-controlled volume.

The important design decision is not the use of MoviePy. It is that the audio mixer receives **segment timestamps**, not only a list of audio files.

In simplified form, VIDEOFCK uses this contract:

```text
script_data[i] = {
            timestamp: "start-end",
            narration: "..."
}

audio[i] -> placed at timestamp.start
audio[i] -> limited to timestamp.end - timestamp.start
```

This gives predictable behavior even when TTS returns audio longer than the target window. Its weakness is that hard trimming can cut words. Therefore we should adopt the explicit timeline contract, but improve the duration strategy with `movie-narrator`-style speed adjustment and alignment.

### 13.2 What VIDEOFCK teaches us about Vision

VIDEOFCK exposes keyframe extraction as a first-class control:

- Fixed number of frames gives predictable API cost.
- Threshold extraction reacts to visual change.
- A minimum three-second separation avoids sending near-duplicate frames.
- `keyframes_per_segment` controls how much visual evidence is sent to the LLM.
- The app stores a combined keyframe sequence image for human inspection.

The combined sequence image is especially useful for DubbingStory. It creates a visual audit artifact that lets the user compare:

```text
scene timeline -> selected frames -> Vision caption -> narration text
```

Our current keyframes are stored individually and passed to Vision, but there is no standard contact sheet or scene review artifact. We should add `scene_contact_sheet.jpg` or `scene_contact_sheet.json` with frame timestamps, even if the LLM still receives individual images.

### 13.3 What movie-narrator teaches us about Vision

The strongest `movie-narrator` idea is not simply sending more frames to a VLM. It builds a semantic bridge between narration and footage:

```text
audio transcript / ASR
                        +
scene timestamps
                        +
optional visual caption
                        |
                        v
scene labels / embeddings
                        |
                        v
narration segment -> matched source scene
```

Important details from its implementation:

- WhisperX is preferred for word-level or segment-level timestamps.
- faster-whisper is a fallback when WhisperX is unavailable.
- Scene captions are built from transcript segments overlapping scene boundaries.
- Placeholder captions are explicitly marked as fake; embedding matching is disabled when most captions are placeholders.
- Narration-to-scene matching uses top-K candidates rather than blindly selecting top-1.
- Recent scene reuse receives a penalty to avoid repetitive footage.
- Source windows are speed-clamped to the narration duration.
- Beat/rhythm metadata can softly bias scene selection toward hook, rising, peak, and settle regions.

This is directly applicable to our summary mode. Our current `scene_selector` selects scenes using Vision importance, but it does not yet match the final narration text back to scene semantics. It assumes one narration segment per selected scene. That is good for a first pipeline, but limits timing quality and makes the system fragile when the LLM skips, duplicates, or merges scenes.

### 13.4 What movie-narrator teaches us about dubbing timing

The relevant synchronization pipeline is:

```text
script segment
            |
            +--> TTS audio duration
            |
            +--> WhisperX / faster-whisper transcript
            |
            +--> segment-level remapping
            |
            +--> word-level remapping when available
            |
            +--> monotonic/non-overlap validation
            |
            +--> SRT and render timeline
```

The implementation has several safeguards that DubbingStory should adopt:

- Detect excessive drift before trusting ASR timestamps.
- Skip unreliable remapping rather than writing obviously wrong timestamps.
- Reject backward timestamp jumps.
- Enforce a minimum segment duration.
- Tighten segment boundaries to first/last word timestamps when word alignment exists.
- Track alignment backend and quality in metadata.
- Preserve TTS-estimated timestamps when alignment quality is insufficient.

The principle is important: **alignment is evidence, not an unconditional replacement for the timeline**. If ASR is unreliable, retain the deterministic TTS timeline and mark alignment as degraded.

## 14. Current DubbingStory Synchronization Gap

The current implementation has two separate synchronization models.

### Full mode

In full mode:

1. Vision creates scenes with original video timestamps.
2. Script validation copies scene `start_time` and `end_time`.
3. SRT uses those source timestamps.
4. TTS generates one audio segment per scene, but `generate_all_audio()` concatenates the segment files into one continuous WAV.
5. Final render maps the continuous WAV from time zero using `-shortest`.

This is safe only if:

```text
TTS segment order == scene order
all scenes have audio
no segment fails
audio duration ~= target timeline
```

Those conditions are not enforced.

### Summary mode

Summary mode remaps narration timestamps to a cumulative summary timeline in [cli.py](../dubbingstory/cli.py#L577), then pads or speeds each TTS segment to the selected scene duration in [voice_manager.py](../dubbingstory/tts/voice_manager.py#L239).

This is closer to correct, but still has gaps:

- The cumulative timeline uses detected durations before the final concatenated file is independently treated as the authoritative timeline.
- Failed TTS segments are omitted from concatenation, while the script timeline can still contain them.
- Audio files are recovered by parsing TXT lines, losing structured timestamps.
- `-shortest` can shorten the final video rather than making a timing defect visible.
- There is no word-level alignment after TTS.

### Root synchronization bug

The root bug is not simply “TTS is too long”. The root bug is that the system does not maintain a single ordered sequence of records carrying all of these fields:

```text
scene_id
source_start
source_end
output_start
output_end
text
audio_path
audio_duration
alignment_status
```

Without that record, video cutting, SRT creation, TTS concatenation, and render muxing can silently disagree.

## 15. Proposed Synchronization Architecture

### 15.1 Canonical timeline segment

Add a canonical JSON model, for example `timeline_{lang}.json`:

```json
{
      "schema_version": 1,
      "language": "id",
      "timeline": [
            {
                  "index": 0,
                  "scene_id": "scene_001",
                  "source_start": 0.0,
                  "source_end": 6.2,
                  "output_start": 0.0,
                  "output_end": 6.2,
                  "text": "...",
                  "audio_path": "audio/id/scene_001.wav",
                  "audio_duration": 5.8,
                  "target_duration": 6.2,
                  "fit_method": "pad",
                  "alignment": {
                        "backend": "tts_estimate",
                        "status": "not_run",
                        "confidence": null
                  }
            }
      ]
}
```

TXT and SRT should be exports from this model. TTS concatenation and render should consume this model directly.

### 15.2 Two-stage duration fitting

For each segment, use this order:

```text
1. Synthesize TTS.
2. Probe actual duration.
3. If close to target: pad or trim with a small tolerance.
4. If over target: apply atempo, capped at a comfortable speed.
5. If still over target: regenerate with shorter text or a faster speaking rate.
6. If under target: add silence, never stretch speech excessively.
7. Record fit method and residual error.
```

Recommended initial thresholds:

```text
acceptable residual: <= 0.10s
soft atempo range: 0.85x to 1.15x
hard safety range: 0.75x to 1.35x
minimum segment duration: 0.10s
```

The exact values should be configurable and measured against Indonesian and English voices separately.

### 15.3 Alignment pass

After concatenated or per-segment TTS is available:

1. Run faster-whisper by default when installed.
2. Use WhisperX when word-level alignment is available.
3. Compare ASR total duration with the produced audio duration.
4. Reject remapping when drift exceeds a configured threshold.
5. Remap timestamps monotonically.
6. Tighten subtitle intervals to word boundaries only when confidence is usable.
7. Store alignment diagnostics in `alignment_qa.json`.

This should be optional for fast preview mode and enabled by default for publish mode.

### 15.4 Render by timeline, not by `-shortest`

The renderer should treat the selected output timeline as authoritative:

```text
video duration = output_timeline.end
audio duration = output_timeline.end
subtitle duration <= output_timeline.end
```

The audio mixer should:

- Build narration audio on a known duration canvas.
- Place each segment at `output_start`.
- Pad gaps explicitly.
- Trim only at segment boundaries.
- Extend or pad the final audio to the target video duration.
- Use `-t` or an explicit `apad` policy rather than relying on `-shortest` to decide the result.

If source audio exists, mix it on the same canvas. If it does not, use narration-only mode and record why.

### 15.5 Visual evidence and frame selection

Adopt a three-level Vision strategy:

```text
Level 1: cheap visual screening
      - 1-2 frames per scene
      - salience, motion/change, dark-frame detection

Level 2: deep scene analysis
      - 3-5 representative frames
      - objects, action, changes, confidence

Level 3: temporal verification
      - selected scenes only
      - narrative role, transitions, plot continuity
```

Improve frame selection with:

- A small boundary margin so frames do not come from the neighboring scene.
- First, middle, and last frame as a stable baseline.
- Optional frame-difference candidates for action-heavy scenes.
- Deduplication using perceptual hash or histogram distance.
- A contact sheet with timestamps for manual audit.
- Explicit frame timestamps in the Vision prompt, not only a scene time range.

Do not send more frames by default. Send more evidence only when confidence is low or frame diversity is high.

## 16. Concrete Improvements Inspired by Both Projects

### P0: Correctness and synchronization

1. Fix `cfg.mode` propagation so summary two-pass Vision actually runs.
2. Create canonical `timeline_{lang}.json` after narration.
3. Stop parsing TXT as the machine source for TTS.
4. Track failed TTS segments and refuse to publish an incomplete timeline unless explicitly allowed.
5. Render audio onto an explicit duration canvas; remove implicit reliance on `-shortest`.
6. Add duration residual checks per segment and globally.

### P1: Alignment quality

1. Add faster-whisper segment alignment as a lightweight default.
2. Add optional WhisperX word alignment for publish mode.
3. Add monotonic timestamp remapping and drift detection.
4. Add alignment confidence and fallback status to manifests.
5. Generate subtitles from aligned timeline segments.

### P1: Vision quality

1. Add frame-difference keyframe strategy alongside distributed sampling.
2. Add per-scene contact sheets with frame timestamps.
3. Add low-confidence re-analysis with one additional diverse frame.
4. Preserve whether a scene was screened, deeply analyzed, cached, or skipped.
5. Use transcript overlap as semantic context for scene selection, not only as prompt text.

### P2: Narration-to-footage matching

1. Extract the narration into semantic segments independent of scene IDs.
2. Build scene captions from subtitle/ASR overlap.
3. Use embeddings only when caption coverage is real; otherwise use deterministic timeline mapping.
4. Use top-K matching with recent-scene reuse penalty.
5. Add speed clamp when a matched source scene is longer or shorter than narration.

## 17. Suggested Implementation Sequence for This Topic

```text
Step 1: Fix mode propagation and add regression test.
Step 2: Persist canonical structured narration JSON.
Step 3: Refactor TTS to return ordered per-segment records.
Step 4: Add explicit timeline audio assembly with padding/trimming metrics.
Step 5: Add render duration/stream QA.
Step 6: Add faster-whisper segment alignment and drift detection.
Step 7: Add optional WhisperX word-level tightening.
Step 8: Add contact sheets and frame-difference selection.
Step 9: Add transcript/embedding scene matching for summary mode.
Step 10: Tune Indonesian/English pacing from real output metrics.
```

### Acceptance criteria

The synchronization milestone is complete when:

- Every narration segment has a stable index and scene/timeline identity.
- TTS output order is independent of filename sorting.
- A failed segment cannot silently shift all subsequent audio.
- Final audio and video durations differ by no more than a configured tolerance.
- Subtitle timestamps are generated from the same timeline used by audio assembly.
- Alignment failure preserves deterministic timestamps and records a warning.
- Word-level alignment never creates backward or overlapping intervals.
- Summary mode uses actual concatenated clip durations as its output timeline.
- A contact sheet or equivalent frame audit artifact exists for Vision review.
- QA can explain whether a mismatch came from script length, TTS duration, alignment, clip extraction, or final muxing.

## 18. Research Sources

- [movie-narrator repository](https://github.com/zcbacxc/movie-narrator)
- [movie-narrator architecture](https://github.com/zcbacxc/movie-narrator/blob/main/docs/ARCHITECTURE.md)
- [movie-narrator alignment implementation](https://github.com/zcbacxc/movie-narrator/blob/main/src/movie_narrator/pipeline/align.py)
- [movie-narrator TTS implementation](https://github.com/zcbacxc/movie-narrator/blob/main/src/movie_narrator/pipeline/tts.py)
- [movie-narrator clip matching implementation](https://github.com/zcbacxc/movie-narrator/blob/main/src/movie_narrator/pipeline/match.py)
- [VIDEOFCK repository](https://github.com/Ido108/VIDEOFCK)
- [VIDEOFCK application orchestration](https://github.com/Ido108/VIDEOFCK/blob/main/app.py)
- [VIDEOFCK video utilities](https://github.com/Ido108/VIDEOFCK/blob/main/utils/video_utils.py)
- [VIDEOFCK Gemini integration](https://github.com/Ido108/VIDEOFCK/blob/main/utils/gemini_utils.py)

The implementations are used here as architectural references. DubbingStory should implement the resulting behavior independently and preserve its own codebase conventions and licensing boundaries.

## 19. Reference Code Patterns

The snippets below are intentionally reduced and rewritten patterns. They show the relevant technique from the two reference repositories without reproducing their complete implementation. They are meant to guide a DubbingStory-native implementation.

### 19.1 VIDEOFCK: timestamped narration as the source contract

VIDEOFCK's useful contract is a list of narration records carrying a timestamp and text. A DubbingStory version should make the structure explicit and avoid encoding timing inside a string.

```python
from dataclasses import dataclass


@dataclass
class NarrationSegment:
      index: int
      start: float
      end: float
      text: str
      scene_id: str | None = None

      @property
      def duration(self) -> float:
            return max(0.0, self.end - self.start)
```

An audio compositor can then place each generated file on a fixed-duration timeline instead of concatenating files based on filename order:

```python
def place_segment_on_timeline(
      canvas,
      segment_audio,
      segment: NarrationSegment,
      sample_rate: int,
):
      target_samples = int(segment.duration * sample_rate)
      audio_samples = segment_audio[:target_samples]

      start = int(segment.start * sample_rate)
      end = min(start + len(audio_samples), len(canvas))
      if start >= end:
            return

      canvas[start:end] += audio_samples[: end - start]
```

This pattern fixes the most dangerous failure mode in the current `voice_manager.py`: if one segment fails, later audio should not slide into the wrong scene. The production implementation must additionally handle channel count, clipping, silence padding, and atomic artifact writes.

### 19.2 VIDEOFCK: frame-difference keyframe selection

The repository offers a threshold mode that compares consecutive grayscale frames and imposes a minimum time distance. A simplified, DubbingStory-compatible version looks like this:

```python
def select_change_frames(
      capture,
      fps: float,
      threshold: float,
      min_gap_seconds: float = 3.0,
):
      selected = []
      previous_gray = None
      frame_index = 0

      while True:
            ok, frame = capture.read()
            if not ok:
                  break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timestamp = frame_index / max(fps, 1.0)

            if previous_gray is None:
                  selected.append((timestamp, frame.copy()))
            else:
                  mean_change = cv2.absdiff(previous_gray, gray).mean()
                  enough_gap = not selected or (
                        timestamp - selected[-1][0] >= min_gap_seconds
                  )
                  if mean_change >= threshold and enough_gap:
                        selected.append((timestamp, frame.copy()))

            previous_gray = gray
            frame_index += 1

      return selected
```

Recommended use in DubbingStory:

- Keep distributed sampling as the stable default.
- Use change-based selection for action-heavy or highly dynamic videos.
- Combine both strategies when a scene has low Vision confidence.
- Never use raw frame difference alone for fade-heavy or camera-shake footage.

### 19.3 VIDEOFCK: human-auditable keyframe sequence

VIDEOFCK creates a combined sequence image. The DubbingStory-native equivalent should preserve timestamps and scene identity in a sidecar manifest:

```python
def build_frame_manifest(scene_id: str, frames: list[tuple[float, str]]) -> dict:
      return {
            "scene_id": scene_id,
            "frames": [
                  {
                        "index": index,
                        "timestamp": round(timestamp, 3),
                        "path": path,
                  }
                  for index, (timestamp, path) in enumerate(frames)
            ],
      }
```

The visual contact sheet is an audit artifact, not necessarily the exact payload sent to the model. This distinction lets us keep the LLM request small while still making Vision decisions inspectable.

### 19.4 movie-narrator: safe duration fitting

`movie-narrator` applies duration feedback instead of blindly accepting the first TTS duration. A reduced version for DubbingStory:

```python
def fit_audio_to_duration(
      input_path: str,
      output_path: str,
      target_duration: float,
      actual_duration: float,
      *,
      min_speed: float = 0.85,
      max_speed: float = 1.15,
) -> dict:
      if target_duration <= 0 or actual_duration <= 0:
            raise ValueError("Audio and target durations must be positive")

      ratio = actual_duration / target_duration
      if abs(ratio - 1.0) <= 0.02:
            filters = [f"apad=pad_dur={target_duration:.3f}"]
            method = "pad"
      elif ratio > max_speed:
            filters = [f"atempo={min(max_speed, ratio):.6f}"]
            method = "speed_up"
      elif ratio < min_speed:
            filters = [f"atempo={max(min_speed, ratio):.6f}"]
            method = "slow_down"
      else:
            filters = [f"atempo={ratio:.6f}"]
            method = "tempo"

      run_ffmpeg([
            "ffmpeg", "-y", "-i", input_path,
            "-af", ",".join(filters),
            "-t", f"{target_duration:.3f}",
            "-c:a", "pcm_s16le", output_path,
      ], label="fit_audio_to_duration")

      return {
            "method": method,
            "target_duration": target_duration,
            "input_duration": actual_duration,
            "ratio": ratio,
      }
```

This is a reference shape, not a final implementation. In production, `atempo` filters must be chained when the ratio exceeds FFmpeg's per-filter limits, and the selected speed range should be measured for each language/voice.

### 19.5 movie-narrator: monotonic alignment remapping

The reference implementation protects the timeline from backward jumps and unreliable ASR. The core rule can be expressed as:

```python
def remap_monotonic(timed_segments, asr_segments, minimum_duration=0.1):
      previous_end = 0.0

      for segment in timed_segments:
            midpoint = (segment.start + segment.end) / 2.0
            match = min(
                  asr_segments,
                  key=lambda item: abs(
                        ((item["start"] + item["end"]) / 2.0) - midpoint
                  ),
            )

            start = max(previous_end, match["start"])
            end = max(start + minimum_duration, match["end"])

            segment.start = start
            segment.end = end
            previous_end = end
```

The real implementation should add the important guard that skips a mapping when the backward jump would crush the original segment. Alignment should be recorded as degraded in that case, while the original TTS estimate remains authoritative.

### 19.6 movie-narrator: word-level subtitle tightening

When WhisperX supplies word timestamps, subtitle intervals can be tightened without changing the overall segment order:

```python
def tighten_to_words(segment, words, minimum_duration=0.1):
      overlapping = [
            word for word in words
            if word["end"] >= segment.start
            and word["start"] <= segment.end
      ]
      if not overlapping:
            return segment

      start = max(segment.start, overlapping[0]["start"])
      end = min(segment.end, overlapping[-1]["end"])
      if end - start >= minimum_duration:
            segment.start = start
            segment.end = end
      return segment
```

This should be applied to subtitle timing only after confidence and monotonicity checks. It should not be used to override a complete timeline when the ASR output is empty, badly drifted, or clearly unrelated to the generated narration.

### 19.7 movie-narrator: semantic scene captions with a truth flag

One subtle but valuable pattern is distinguishing real semantic captions from placeholders:

```python
def build_scene_caption(scene, transcript_segments):
      texts = [
            item["text"]
            for item in transcript_segments
            if item["start"] < scene.end and item["end"] > scene.start
      ]

      if texts:
            return {"text": " ".join(texts)[:200], "is_fake": False}

      return {
            "text": f"scene {scene.index} {scene.start:.1f}-{scene.end:.1f}",
            "is_fake": True,
      }
```

Embedding matching should be disabled or downgraded when most scene captions have `is_fake=True`. Otherwise the system creates a false impression of semantic matching while matching narration against timestamps and scene numbers.

### 19.8 Combined DubbingStory reference flow

The two repositories suggest this concrete flow for our implementation:

```python
scenes = detect_scenes(source_video)
frames = extract_representative_frames(scenes, strategy="distributed")
screening = screen_scenes(frames)
deep_scenes = select_low_confidence_or_high_value(scenes, screening)
storyboard = analyze_temporal_flow(deep_scenes, transcript=asr_context)

script = generate_structured_script(storyboard)
timeline = build_output_timeline(script, mode="full_or_summary")

for segment in timeline:
      raw_audio = synthesize(segment.text)
      segment.audio = fit_audio_to_duration(
            raw_audio,
            target_duration=segment.output_end - segment.output_start,
      )

alignment = align_audio_if_enabled(timeline)
subtitles = export_srt(timeline)
audio = render_audio_canvas(timeline, duration=timeline[-1].output_end)
video = render_video(source_video, audio, subtitles, timeline)
qa = validate_deliverable(video, audio, subtitles, timeline)
```

This flow preserves the strongest ideas from both references:

- VIDEOFCK's explicit timestamped narration and visual audit artifacts.
- `movie-narrator`'s alignment safeguards, semantic matching, cache discipline, and QA metadata.
- DubbingStory's existing subtitle fallback, Vision provider abstraction, summary selector, and local/cloud workflow.

### 19.9 License and adaptation note

These snippets are reference pseudocode and reduced examples. They should be reimplemented in DubbingStory's own modules, naming, tests, and data contracts. Do not copy substantial source blocks from either repository into the project. Review the source licenses before adapting any implementation-level detail:

- `movie-narrator`: AGPL-3.0-or-later.
- `VIDEOFCK`: MIT.
