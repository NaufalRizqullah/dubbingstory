# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
