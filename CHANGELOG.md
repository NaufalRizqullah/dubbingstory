# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

All notable changes to the **dubbingstory** project will be documented in this file.

**The Standard Structure (SemVer)**
- **Major (X.y.z)**: Incremented for incompatible API changes (breaking changes).
- **Minor (x.Y.z)**: Incremented for new functionality introduced in a backward-compatible manner.
- **Patch (x.y.Z)**: Incremented for backward-compatible bug fixes or minor patches.


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
