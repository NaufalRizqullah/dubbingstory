# DubbingStory — Context & Architecture

> Context handoff dari project `opensource-clipping`.
> File ini membantu AI agent memahami project ini tanpa perlu membaca seluruh codebase clipping.

## Relationship to `opensource-clipping`

`dubbingstory` adalah **sister project** dari [`opensource-clipping`](../opensource-clipping/).

- **`opensource-clipping`**: AI Auto-Clipper — potong video panjang jadi viral short clips
- **`dubbingstory`**: AI Video Narrator — buat dubbing cerita otomatis dari video tanpa narasi

Keduanya share beberapa pattern:
- `yt-dlp` untuk download video
- `google-genai` untuk Gemini AI analysis
- FFmpeg untuk video processing
- Python 3.10+, `python-dotenv` untuk config

## Key Design Decisions

1. **Gemini API for vision** — free tier supports image/video, project sudah punya `GOOGLE_API_KEY`
2. **Piper TTS as MVP** — CPU-only, MIT license, ada Indonesian voice (`id_ID-news_tts-medium`)
3. **VoxCPM2 as advanced** — GPU, Apache-2.0, native Indonesian, 48kHz, voice cloning
4. **pysubs2 for film subtitles** — baca SRT/ASS/VTT untuk context dari film
5. **PySceneDetect** — auto scene splitting
6. **Default narration style = viral FB/TikTok**
7. **Dual output: 16:9 + 9:16**
8. **Runs on Kaggle** (no local GPU)

## Pipeline Overview

```
Video Input (local / YouTube / film with subs)
  → Ingest (download, validate, read subtitles if any)
  → Segment (PySceneDetect → scene split → keyframe extraction)
  → Vision (Gemini API → per-scene visual analysis → storyboard.json)
  → Story (narration script ID/EN → .srt subtitles)
  → TTS (Piper/VoxCPM2 → audio dubbing per scene)
  → Render (FFmpeg → mix audio + burn subtitles → final video)
```

## Config Pattern

Uses YAML configs in `configs/` + `.env` for API keys + CLI argparse.
Same pattern as `opensource-clipping` but with YAML instead of pure argparse.

## Shared Patterns from `opensource-clipping`

These patterns were adapted (copied, not imported) from the clipping project:

- **yt-dlp download wrapper**: `clipping/engine.py` → `dubbingstory/ingest/youtube.py`
- **Gemini retry logic**: `clipping/engine.py` → `dubbingstory/vision/gemini_analyzer.py`
- **FFmpeg helpers**: `clipping/studio/ffmpeg_utils.py` → `dubbingstory/utils/ffmpeg_helpers.py`
- **Scene trimming**: `clipping/story/assembler.py` → `dubbingstory/utils/ffmpeg_helpers.py`
