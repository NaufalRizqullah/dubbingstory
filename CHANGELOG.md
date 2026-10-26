# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

All notable changes to the **dubbingstory** project will be documented in this file.

**The Standard Structure (SemVer)**
- **Major (X.y.z)**: Incremented for incompatible API changes (breaking changes).
- **Patch (x.y.Z)**: Incremented for backward-compatible bug fixes or minor patches.

## [0.9.1]

### Changed
- Add `_canvas` to automatically named layered 9:16 outputs in full/summary pipelines and render recovery, for example `final_summary_id_canvas_9x16.mp4`. Original/source and padded output names, explicit `reframe --output` paths, and existing files remain unchanged.

### Fixed
- Let the installed yt-dlp select its maintained YouTube player clients for video
  and subtitle extraction instead of restricting extraction to Android/Web.
- Distinguish unavailable requested formats from an extractor reporting no usable video,
  and expose allowlisted downloader diagnostics without raw media URLs or credentials.
- Install `yt-dlp[default]` before the Kaggle notebook's download cell so its
  matching EJS solver is available alongside the JavaScript runtime.
- Fix dynamic ducking format negotiation on older FFmpeg builds when mixing stereo 44.1 kHz source audio with mono 24 kHz TTS: pin stereo layouts and 48 kHz rates, with explicit double-sample compressor inputs and float mix boundaries.
- Keep the layered VFR regression fixture compatible with the legacy FFmpeg synchronization option already used by the renderer.

### Added
- Safe Kaggle download setup diagnostics for package versions, tool availability,
  and the checked-out commit; stop reuse of a kernel that imported yt-dlp before an upgrade.
- Render saved summary artifacts with `render --mode summary`, including original 16:9 and layered 9:16 output, without repeating vision, narration, or TTS.
- Opt-in Kaggle media regressions for the reported stereo/mono ducking failure, both render ratios, and ducking attenuation/recovery. Tests are authored but have not been run locally.

## [0.9.0]

### Added
- TikTok URL/share-link routing through the shared yt-dlp ingest backend.
- `--narration-source visual|audio`; visual is the default for finished-video
  storytelling, while audio mode uses subtitles or automatic Whisper ASR as
  dialogue context before generating translated/narrated TTS.
- Opt-in `--vertical-layout layered` for 9:16 output: independently zoomed foreground/background video layers, configurable Gaussian blur and grayscale, and one audio path. Available in full and summary rendering without changing the default padded layout.
- `reframe --input ... --output ...` recomposes existing local exports without rerunning analysis, narration, or TTS; includes Kaggle usage and visual acceptance checks in `docs/layered-vertical.md`.
- Native full-video workflow through `--ratio source`, source-derived timeline metadata, and `--narration-fit rewrite|error` with bounded per-beat rewrite/re-synthesis.
- Additive TTS metadata interface and Edge word-boundary capture; optional WhisperX alignment for Piper/other engines (`.[alignment]`, WhisperX 3.8.6+ for Indonesian and disabled interpolation).
- Versioned word-timing artifacts bound to final narration/audio hashes, reusable ASS word-by-word and active-word karaoke captions, configurable animations/fonts/colors/safe margins, and `--subtitle-style`/`--word-alignment`/`--burn-subtitles` options.
- Shared yt-dlp ingestion for YouTube, Douyin, and Xiaohongshu/RedNote with URL normalization, source identity, reusable validated assets, and classified download errors. Platform support remains conditional on public availability; no authentication/DRM bypass was added.
- Offline regression tests and a Kaggle validation guide for native rendering, TTS fitting, word timing, subtitles, URL routing, and legacy behavior.

### Changed
- Native/animated workflows validate complete speech before rendering; legacy CLI defaults remain available.
- Animated subtitles use measured boundaries rather than equal-duration word allocation. Existing SRT output is retained and refreshed from final speech timing.
- Audio mixing honors the original-volume setting during dynamic ducking, handles sources without audio, and pads audio to retain the video timeline.
- Rendering combines required visual filters in one video-encoding stage; compatible video without visual filtering can be stream-copied.
- Automatic URL project names include source identity. Existing projects remain accessible through explicit `--project`.
- Declared Edge TTS in both installation manifests and kept forced alignment in an optional extra.

### Fixed
- Use legacy-compatible `-vsync 0` for encoded renders and subtitle burns so
  layered output also works with older FFmpeg builds that reject `-fps_mode`.
- Full-mode subtitles no longer rely only on pre-TTS planned timestamps when actual speech placements are available.
- Local subtitle discovery uses the original input basename before the project copy is renamed to `source.mp4`.
- Summary audio volume can be set to zero; FFmpeg test discovery no longer requires a Unix shell.
- Partial legacy TTS keeps the complete narration script for retry and records partial coverage instead of claiming complete audio.
- Strict timelines use video stream/packet duration instead of a potentially longer container/audio duration; classic subtitle font defaults remain unchanged.

### Validation status
- Tests and rendering were not executed on the development computer at the user's request; run the documented suite in Kaggle before treating this as a validated release.
- A metadata-only probe with installed yt-dlp `2026.06.21.235142` did not resolve the supplied RedNote example (direct host unsupported; Xiaohongshu alias returned an extractor error). No successful live download is claimed.

## [0.8.0] - 2026-08-30

### Added
- **Narration Quality Overhaul**: Enhanced `V2_NARRATION_PROMPT` with strict rules for opening beats, pacing variety, and process compression to ensure more natural, engaging, and non-repetitive storytelling.
- **Deep Scene Schema Enrichment**: Expanded vision schema to capture `emotional_tone`, `process_stage`, and `audio_cue` for richer narrative context.
- **Monotonous Pacing Detection**: Added an automated QA check to penalize and rewrite generated scripts that exhibit identical sentence lengths repetitively, improving TTS delivery cadence.
- **Adaptive Keyframe Extraction**: Implemented duration-aware keyframe capping in `keyframes.py` (~1 frame per 2.5s) to drastically reduce VRAM/token usage on long shots without sacrificing detail.
- **CLI Control for Adaptive Keyframes**: Added `--disable-adaptive-keyframes` flag to bypass the new adaptive duration limit when dense frame extraction is explicitly required.
- **CLI Speech Coverage Control**: Added `--speech-coverage` flag (defaulting to a tighter 0.85) to give users direct control over narration density vs video time.

### Changed
- **Jozzr Social Pacing & Closure**: Added specific rules to the `jozzr_social` style requiring narrative closure (payoff) and rhetorical questions for better engagement.
- **Tighter Beat Budgets**: Reduced `target_beat_duration` (11.0s → 9.0s) and `max_beat_duration` (18.0s → 14.0s) in `default.yaml` to prevent overly long, rambling sentences across multiple scenes.
- **Continuity Alias Expansion**: Expanded `scene_continuity.py` alias groups to recognize `tool`, `surface`, `material`, and `result` entities, improving local state-change tracking across cuts.

## [0.7.0] - 2026-08-23

### Added
- **Jozzr Social Narration Style**: Added `jozzr_social`, a process-driven social storytelling profile focused on concise problem → action → change → consequence → next-step narration.
- **Azure Speech TTS Engine**: Added Azure AI Speech as a new TTS backend with native support for `id-ID-ArdiNeural`.
- **Azure TTS Configuration**: Added `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` configuration for Kaggle, Colab, and local environments.
- **Azure TTS Reliability Controls**: Added retry handling, request throttling, timeout configuration, and SSML-based speech-rate control.
- **Azure CLI Support**: Added `azure` as a valid TTS engine for `run` and `dub` workflows.
- **Jozzr/Azure Universal Notebook**: Added notebook configuration and preflight validation for the Jozzr Social + Azure Ardi workflow.

### Changed
- **Narration Style Prompting**: Style-specific language guidance, examples, and storytelling rules are now passed more explicitly into the narration writer.
- **Narration QA Rewrite**: QA rewrites now preserve the selected narration style instead of potentially drifting toward generic narration.
- **Jozzr Delivery Profile**: Added restrained pacing and cadence guidance optimized for informative social-video narration.
- **TTS Engine Selection**: Extended the existing pluggable TTS architecture to support Azure without changing the timeline-aware scheduler.

### Compatibility
- Existing `edge`, `piper`, and `chirp` TTS engines remain supported.
- Existing narration styles remain unchanged.
- V2.1 Vision, semantic matching, evidence verification, and typed repair behavior remain backward compatible.

## [0.6.0] - 2026-08-23

### Added
- **DubbingStory V2.1 Evidence-Aware Summary Pipeline**: Added a story-first recap architecture that plans narrative beats before final footage selection, allowing the pipeline to choose scenes based on the facts each beat needs to prove.
- **Scene Window Identity Contract**: Preserved `parent_scene_id`, `window_index`, and `window_count` metadata across downstream processing so split windows from the same long shot can be recognized and reasoned about consistently.
- **Sibling Shared-Boundary Anchors**: Added continuity-aware shared keyframe boundaries for adjacent windows from the same parent scene while avoiding cross-shot contamination at hard cuts.
- **Local Scene Continuity / Delta**: Added structured scene-to-scene continuity analysis to distinguish setup, continuation, progression, transition, result, reaction, parallel action, and new topics while emphasizing newly visible information and state changes.
- **Normalized Scene Signatures**: Added retrieval-oriented scene signatures combining visual actions, objects, state changes, goals, continuity context, confidence, and useful transcript/audio evidence.
- **Story-First Beat Planning**: Added structured narration beats with intent, required facts, forbidden claims, approximate timeline position, importance, and target duration before footage is fixed.
- **Deterministic Beat Deduplication**: Added inexpensive duplicate-beat filtering before narration expansion to reduce repetitive recap content without requiring another LLM call.
- **Top-K Semantic Footage Matching**: Added pre-write beat-to-scene retrieval and configurable composite reranking using semantic relevance, temporal position, continuity, state change, importance, visual confidence, reuse penalties, and parent-scene reuse penalties.
- **Chronological Match Guard**: Added timeline-aware constraints so semantic matching normally progresses forward while still allowing explicitly justified backtracking.
- **Multi-Window Beat Support**: A narration beat can now use multiple short scene windows when several clips are required to show one coherent process or progression.
- **Evidence Quality Gate**: Added safeguards that avoid treating skipped, empty, or low-confidence scene evidence as reliable semantic-match candidates.
- **On-Demand Deep Vision Promotion**: Weak semantic matches can promote promising Pass-A candidates to deep Qwen analysis and then retry matching instead of deep-analyzing the entire source video.
- **Fixed Evidence Writing Contract**: Narration is generated against the scene evidence selected for each beat, preventing the writer from freely expanding beyond the visual facts assigned to that beat.
- **Narration `claims[]` Output**: Added explicit factual claims alongside generated narration to make post-write evidence validation more precise and debuggable.
- **Post-Write Semantic Verifier**: Added factual verification of narration against assigned scene evidence, including claim coverage, unsupported claims, entity compatibility, state-change compatibility, and continuity checks.
- **Typed Per-Beat Repair**: Added surgical recovery paths for unsupported wording, incorrect footage matches, insufficient evidence, and continuity mismatches instead of relying primarily on whole-script rewrites.
- **Semantic Match Rollout Modes**: Added `--semantic-match active|shadow|off`; `shadow` keeps legacy selection as the final output while generating V2.1 beat/match artifacts for safe A/B evaluation.
- **V2.1 Debug Artifacts**: Added/expanded runtime artifacts such as `scene_delta.json`, `scene_signatures.json`, `beats.json`, and verbose `match_report.json` for matcher and evidence-level debugging.
- **V2.1 Notebook Preflight**: Updated the universal Kaggle/Colab notebook with real source preflight checks for the existing Vision/language fixes and the new V2.1 modules instead of only printing a success message.

### Changed
- **Summary Pipeline Ordering**: Summary mode can now progress from story beats to semantic footage matching before cutting the final recap, instead of permanently selecting/cutting footage before the narration intent is known.
- **Narration Planning Responsibilities**: Kept `narration_planner.py` focused on timing and word-budget scheduling while story intent is handled by the new beat-planning layer.
- **QA Responsibility Split**: Preserved `narration_qa.py` for deterministic language/timing/repetition checks while factual scene-to-narration validation is handled by the semantic verifier.
- **Legacy Scene Selector Role**: The existing selector is retained as a compatibility fallback and shadow-mode baseline rather than being removed.
- **Summary Vision Strategy**: Preserved the efficient two-pass Vision architecture and added selective deep promotion for weak beats instead of requiring full-video deep analysis.

### Fixed
- **Window Lineage Loss**: Prevented downstream loss of parent-shot/window metadata after long scenes are split into bounded analysis windows.
- **Unsafe Cross-Language Verifier Repair**: Prevented weak lexical fallback scores from automatically triggering factual narration repair when a reliable semantic backend is unavailable.

### Tests
- Added V2.1 regression coverage for scene/window continuity identity, beat deduplication, Top-K matching, fixed beat-to-scene narration scheduling, and safe `claims[]` / verifier degraded-mode behavior.
- Verified the implementation with **5/5 V2.1 regression tests passing**, source compilation, CLI smoke tests, clean patch application, and tests rerun from the extracted implementation bundle.

## [0.5.1] - 2026-08-22

### Fixed
- **Indonesian Narration Fallback**: Prevented the `id` fallback script from copying English visual-analysis text when narration generation fails, such as after a Gemini `408 Request Timeout`.
- **Fallback Regression Coverage**: Added a test ensuring Indonesian fallback narration does not contain copied English analysis text.

## [0.5.0] - 2026-08-21

### Added
- **DubbingStory V2**: Began implementation of the next-generation project architecture and feature set for version 2.

## [0.4.13] - 2026-08-21

### Fixed
- **FFmpeg Extraction Drift**: Dynamic video padding (where a scene is re-cut to match TTS length) now correctly reads the *actual* extracted duration from `.durations.json` instead of assuming it matches the exact target duration. This fixes a bug where accumulated FFmpeg rounding errors caused the final summary video to be slightly shorter than expected, leading to misaligned subtitles and cut-off audio at the very end.

### Added
- **Detailed Tracking Logs**: Added verbose `[Tracking]` logs in the CLI for the dynamic extension process (showing Video vs TTS durations) and in the Vision module (showing JSON recovery status from truncated tokens) to aid in debugging vLLM budget issues.
- **Persistent Error Logs**: Failed vLLM (Vision) analysis and failed TTS generation are now explicitly logged into `failed_vision.log` and `failed_tts.log` within the project directory to provide a written record of bottlenecks for debugging.

## [0.4.12] - 2026-08-20

### Fixed
- **vLLM Out of Token Loop (Hallucination Bypass)**: Fixed a bug where Qwen-VL or Llama models served via vLLM would occasionally hallucinate trailing garbage after generating a valid JSON, hitting the `max_tokens` limit (`finish_reason=length`). The pipeline now successfully intercepts and parses the valid JSON from the truncated output instead of discarding it, preventing infinite retries and failed scenes.

## [0.4.11] - 2026-08-20

### Added
- **Dynamic Video Padding (Speedup Fix)**: Voiceover segments that are longer than their visual scenes are no longer heavily sped-up (chipmunk effect). Instead, the video scene is dynamically extended (re-cut from the original footage with a +0.5s padding) to fit the natural duration of the voiceover.
- **Strict Narration Word Limits**: The AI is now given a strict `max_words_allowed` per scene based on the scene's duration (2.5 words/sec) to discourage generating excessively long sentences for short clips.

## [0.4.10] - 2026-08-20

### Fixed
- **Summary Duration Manifest Became Zero**: Fixed summary clip durations being measured after temporary clips were deleted, which produced zero durations and corrupted the summary timeline, SRT timestamps, and audio alignment.
- **Summary Script Missing Timestamps**: Summary text scripts now include each segment's summary-video time range, while the TTS parser removes this display-only metadata before synthesis.

## [0.4.9] - 2026-08-19

### Fixed
- **Adaptive Vision Output Budget**: Kept the vLLM context at `12288` and added a bounded retry for vision responses that end with `finish_reason=length`. Retries reduce the keyframes to one image and allow up to `3072` output tokens while still respecting the remaining context budget.
- **Temporal Flow Truncation**: Limited temporal analysis output to a safe maximum of `4096` tokens instead of requesting the entire model context, reducing JSON truncation risk for Qwen3-VL/vLLM.
- **Temporal Chunk Retry**: Failed temporal chunks are now split into smaller batches and retried before falling back to neutral narrative metadata.

## [0.4.8] - 2026-08-19

### Fixed
- **Temporal Analysis `finish_reason=length`**: The temporal flow analysis was capped at `max_tokens=2048` for output, even though the input was only ~2600 tokens (leaving ~9600 tokens available). With 30 scenes per chunk, the required output (~3000 tokens) exceeded the 2048 cap, causing JSON truncation and failure for most chunks. Now `analyze_temporal_flow()` uses the full allowed output budget (`model_max_context - input_tokens`) so temporal chunks complete successfully.
- **Summary SRT Timestamp Desync (Video Longer Than Audio)**: SRT timestamps were using the original video timeline (e.g., `00:44:31` for a scene at 2633s in the original 44-minute video), but the rendered summary video is a concatenated cut of only ~3 minutes. Added cumulative timeline remapping in `cli.py` so SRT timestamps correctly reflect the summary video's own timeline (e.g., `00:02:50` instead of `00:44:31`).
- **Temporal Chunk Size Reduced**: Default `temporal_chunk_size` changed from 30 → 20 scenes per batch to provide additional safety margin for output token budget.

## [0.4.7] - 2026-08-19

### Fixed
- **Temporal Analysis Context Overflow**: Fixed an issue where videos with many scenes (e.g., >200 scenes) would crash the Temporal Flow Analysis because the combined JSON prompt exceeded the model's 12.288 context limit. Implemented **Temporal Chunking** in `scene_understanding.py` that processes scenes in batches (default: 30 scenes/chunk) and merges the enriched narrative roles. This ensures the `summary` mode correctly selects scenes across the entire video duration (including the climax and ending) rather than defaulting to the first few minutes.

## [0.4.6] - 2026-08-19

### Fixed
- **Vision Token Overflow (`finish_reason=length`)**: Completely overhauled token budget estimation in `openai_vision.py` to account for image tokens (~1300 per image for Qwen3-VL). Previously, only text tokens were counted, causing massive context overflows. Implemented an automatic retry mechanism that reduces the number of keyframes if a context limit error is encountered.
- **Summary Cut Audio Desync**: Fixed an issue where the summary narration script could become desynced from the video cut if the temporal flow analysis failed or generated empty `action` fields for certain scenes. Also added strict clip duration validation (`±1.0s` tolerance) and explicit time-range logging in `video_cutter.py` to ensure FFmpeg cuts align perfectly with the selected scenes.
- **Kaggle Context Mismatch**: Synced `OPENAI_VISION_MODEL_MAX_CONTEXT=12288` in `.env` for Kaggle notebooks to match the vLLM `--max-model-len` flag, unlocking the full context window. Raised default `VISION_MAX_TOKENS` to 2048 to prevent premature output truncation.

## [0.4.5] - 2026-08-18
### Fixed
- **Vision Image Loading Crash (100% scene failure)**: Fixed the `file://` image path mode that caused `InternalServerError: Cannot load local files without --allowed-local-media-path` on every scene. The default `image_mode` is now `"data"` (base64, always works). For efficient local vLLM (Kaggle/Colab), set `OPENAI_VISION_IMAGE_MODE=file` in `.env` and add `--allowed-local-media-path /kaggle/working` to the vLLM startup command.
- **Kaggle Script Auto-Setup**: Updated `mereska-dubbingstory-yt.py` to automatically enable `file://` mode with the correct vLLM `--allowed-local-media-path` flag for optimal performance.

## [0.4.4] - 2026-08-17

### Fixed
- **Summary Cut Audio Desync**: Fixed a critical bug in Summary Mode where scenes that failed vision analysis were skipped by the TTS generation but still cut into the video compilation. This caused silent scenes and desynced dubbing. Failed scenes are now completely excluded from the cut sequence.
- **Vision Token Overflow (OOM/Truncation)**: Mitigated extreme token generation during scene analysis by enforcing strict brevity constraints in the Vision prompt (`finish_reason=length`).
- **Temporal Context Exceeded**: Fixed a critical bug where long videos (e.g. 150+ scenes) would trigger HTTP 400 Context Length errors in local models (e.g. vLLM). Implemented a **Temporal Chunking** algorithm to process global scenes in batches of 40 and condense the JSON payload, ensuring it stays well within the 8192 token limit regardless of video length.

## [0.4.3] - 2026-08-17

### Added
- **Piper TTS Restoration**: Re-implemented the `PiperTTSEngine` with robust auto-download capabilities and reliable path resolution for `piper` models.
- **CLI Voice Overrides**: Added `--voice-id` and `--voice-en` CLI flags to `run` and `dub` commands, allowing users to override TTS voices on the fly for both Edge and Piper engines.

### Fixed
- **CLI Engine Default Bug**: Fixed an issue where the `--engine` CLI argument ignored fallback configurations (e.g. from `.yaml`) because it always forced `"edge"` as the parsed default.

## [0.4.2] - 2026-08-17

### Added
- **Two-Pass Summary Vision**: Introduced a two-pass architecture for summary mode (Cheap Screening -> Deep Analysis on Top N) to drastically reduce GPU compute time.
- **Smart Version-Aware Cache**: Vision cache keys are now hashed using prompt text, model name, and keyframe metadata to prevent stale cache hits when configurations change.

### Changed
- **Adaptive Keyframe Count**: Reduced the default `--max-keyframes` from 7 to 3 to optimize analysis speed without significantly degrading context quality.
- **Hierarchical Temporal Flow**: Summary mode temporal flow now only processes the selected Top N scenes instead of all scenes, preventing LLM context window overflow (HTTP 400 errors).

## [0.4.1] - 2026-08-17

### Added
- **Vision Token Control**: Added `--vision-max-tokens` CLI argument (default: 2048) to give users explicit control over the maximum output tokens generated by the vision model.
- **Direct Keyframe Extraction**: Keyframes are now extracted directly from the source video using timestamps, eliminating the need to physically split the video into hundreds of small `.mp4` scenes, saving significant processing time and disk space.

### Changed
- **Colab/Kaggle Context Length**: Increased `MAX_MODEL_LEN` from 8192 to 12288 in Jupyter Notebooks and runner scripts to better accommodate the new vision prompts and prevent `Context Length Exceeded (400)` errors on T4 GPUs.
- **TTS Engine Cleanup**: Completely removed `piper` and `voxcpm2` offline TTS engines from the codebase. The pipeline now exclusively uses `edge` (Microsoft Edge TTS) to guarantee stability in cloud environments (Kaggle/Colab) where downloading large local voice models often fails.

### Fixed
- **FFmpeg Concat Pathing**: Fixed an issue in `video_cutter.py` where FFmpeg failed to resolve relative paths in the concatenation list by switching to file basenames.


## [0.4.0] - 2026-08-16

### Added
- **TTS Backup**: Automatically exports/backs up the generated TTS audio files (`backup_tts_full_xxx.wav` / `backup_tts_summary_xxx.wav`) to the project root directory.

### Changed
- **Default TTS Engine**: Changed the default TTS engine across the pipeline from Piper to Edge TTS for better quality and stability.
- **Audio Mixing Strategy**: Changed the default audio strategy to `mute_original` to prevent the original video audio from interfering with the new voice-over.
- **Script Generation Prompts**: Enhanced narration prompts to include video description context. Explicitly strictly forbade the model from generating disconnected reactive phrases (like "Look at this!"), ensuring it writes a cohesive, natural story based on the video's actual context.

### Fixed
- **Vision Context Length (HTTP 400)**: Reduced `vision_openai_max_tokens` from 2048 to 1024 to prevent vLLM from hitting the 8192 max context limit during temporal flow analysis, fixing the `finish_reason=length` and HTTP 400 errors.

## [0.3.2] - 2026-08-16

### Added

- **YouTube Ingest (yt-dlp)**: Added `--yt-cookies` CLI flag to support `cookies.txt` files and browser cookies (`--cookies-from-browser`), providing a reliable workaround for YouTube download blocks on Kaggle/Colab.

## [0.3.1] - 2026-08-16

### Fixed

- **YouTube Ingest (yt-dlp)**: Added client spoofing (`player_client=android,web`) to bypass widespread `HTTP Error 403: Forbidden` blocks from YouTube's anti-bot measures, particularly for Kaggle/Colab IP environments.

## [0.3.0] - 2026-08-15

### Added

- **Vision Concurrency**: Scene vision analyses are now processed concurrently using `ThreadPoolExecutor`, drastically speeding up the pipeline.
- Unified Kaggle and Colab notebook experience with improved auto-detect logic for T4 GPUs.

### Changed

- **Pipeline Optimization**: Skipped physical splitting of scenes in `keyframes` mode. Keyframes are now extracted directly from the source video with automatic resizing, eliminating hundreds of unnecessary video renders.
- **FFmpeg Concat Optimization**: Replaced relative paths with basenames and used `-c copy` in `video_cutter.py` to prevent redundant re-encoding and path resolution errors.
- **Vision Parsing & Stability**: Enforced structured JSON output in Qwen/vLLM integration and implemented a robust regex-based fallback decoder. Separated retry logic for network vs. format errors.

## [0.2.1] - 2026-08-14

### Added

- **Dual-Language Dubbing**: Support for both Indonesian and English subtitles/scripts in the same pipeline run.
- Added `language` argument to CLI to specify language for subtitle/script.
- Added `verbose` argument to CLI to enable verbose output.
- Added `help` command to CLI to display help message.

### Changed

- Removed local subtitle file requirement and fallback to YouTube automatic subtitle download and ASR fallback.
- Refactored `dubbingstory.ingest.cli_ingest` to `dubbingstory.cli.cmd_ingest` and updated the ingest manifest structure.

## [0.2.0] - 2026-08-13

### Added
- **Summary Mode (Highlight Recap)**: A major new pipeline mode (`--mode summary`) that intelligently selects the most important scenes from a long video, cuts and concatenates them, and generates a condensed highlight recap narration.
- New CLI arguments for summary mode: `--mode`, `--summary-duration`, and `--summary-max-scenes`.
- Extracted and enhanced `video_cutter.py` and `scene_selector.py` to support the new FFmpeg-based video slicing and ranking algorithm.
- Added support for custom script and audio prefixes in `voice_manager.py` to prevent overwriting full pipeline assets when running summary mode.

## [0.1.1] - 2026-06-24

### Added
- Created `main.py` at the project root as an alternative, easier entry point for execution.

### Changed
- Updated `README.md` and `README_EN.md` execution instructions to use `python main.py` instead of `python -m dubbingstory.cli`.

## [0.1.0] - 2026-06-20

### Added
- Initial project release: `dubbingstory`
- **Ingest Module**: Support for local video validation and YouTube URL downloading (via `yt-dlp`), including automatic reading of SRT/ASS subtitles via `pysubs2` for enhanced context.
- **Segment Module**: Automatic scene detection and video splitting using `PySceneDetect`, plus evenly distributed keyframe extraction using OpenCV.
- **Vision Module**: Integration with Google Gemini API for multimodal visual understanding of video scenes. Includes retry logic and anti-hallucination prompts.
- **Story Module**: Bilingual (Indonesian & English) script generation using LLM with 4 distinct narration styles (`viral_fb`, `documentary`, `technical`, `calm_educational`). Also generates synchronized `.srt` subtitle files.
- **TTS Module**: Initial support for `piper` (CPU-based MVP) and placeholder support for `voxcpm2` (GPU-based advanced TTS).
- **Render Module**: Audio mixing with 3 strategies (`duck`, `replace`, `mute_original`), optional subtitle burn-in, and final video assembly supporting multiple aspect ratios (`16:9`, `9:16`).
- **CLI**: Comprehensive command-line interface with 7 subcommands (`run`, `ingest`, `segment`, `analyze`, `narrate`, `dub`, `render`).
- **Config**: YAML-based pipeline configuration merged with environment variables (`.env`).
