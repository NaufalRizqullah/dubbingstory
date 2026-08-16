# 🎬 DubbingStory

*[Baca dalam Bahasa Indonesia](README.md)*

**Automated Video Narration & Dubbing Pipeline** — Turn silent/foreign videos into dubbed stories with AI.

Inspired by the accurately narrated and dubbed Pakistani repair videos on Facebook.

---

## ✨ Features

- 🎥 **Video Ingest** — Supports local files, YouTube URLs, and films with subtitles.
- ✂️ **Auto Scene Detection** — Automatically split videos into scenes using PySceneDetect.
- 🖼️ **Keyframe Extraction** — Extract representative frames per scene.
- 🧠 **Dual LLM Architecture** — Separates the *Vision* (seeing scenes) and *Text* (writing narration) tasks for optimal quality-to-cost balance.
- 👁️ **Visual Understanding (Cloud & Local)** — Understand visual context using **Gemini API** (default) or open-source HuggingFace models (like **Qwen3-VL**) via OpenAI-compatible endpoints. Zero API cost for Vision!
- ☁️ **Kaggle / Colab Ready** — Includes a unified script (`dubbingstory_colab.py`) to run the local vision model for free on T4 cloud GPUs.
- 📝 **Bilingual Narration** — Generate narration scripts in Indonesian & English.
- 🎙️ **TTS Dubbing** — Text-to-Speech support powered by **Edge TTS** by default for maximum stability in cloud environments (Kaggle/Colab).
- 🎬 **Video Render** — Final video render with audio mix and optional subtitles for 16:9 & 9:16 aspect ratios.
- ✂️ **Highlight Recap Mode** — Automatically cut and condense long videos into short highlight reels using the `--mode summary` option.

---

## 🚀 Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Setup API Key

```bash
cp .env.sample .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Run

You can use the `dubbingstory` CLI command or run the `main.py` file directly:

```bash
# Using main.py (alternative if CLI is not recognized in PATH)
python main.py run --input video.mp4

# Full pipeline — local video
dubbingstory run --input video.mp4

# Full pipeline — using local Qwen3-VL via vLLM
dubbingstory run --input video.mp4 --vision-provider openai --engine edge

# Full pipeline — YouTube (with rights)
dubbingstory run --url "https://youtube.com/..." --i-have-rights

# Full pipeline — Summary Mode (Highlight Recap ~60 seconds)
dubbingstory run --url "https://youtube.com/..." --mode summary --summary-duration 60 --i-have-rights

# Step by step
dubbingstory ingest --input video.mp4
dubbingstory segment --project video
dubbingstory analyze --project video --vision-provider openai
dubbingstory narrate --project video --style viral_fb --lang id en
dubbingstory dub --project video --engine edge
dubbingstory render --project video --ratio 16:9 9:16
```

---

## ⚙️ CLI Parameters

### `run` (Full Pipeline)
Run the entire process from start to finish.

| Argument | Default | Description |
|----------|---------|-------------|
| `--input`, `-i` | `None` | Path to local video file. |
| `--url`, `-u` | `None` | YouTube or other video URL. |
| `--i-have-rights` | `False` | Confirm video rights (required for URL). |
| `--project`, `-p` | `auto` | Project name (defaults to filename). |
| `--mode` | `full` | Pipeline mode: `full` (entire video) or `summary` (highlight recap). |
| `--summary-duration`| `auto` | Target duration in seconds for summary mode (e.g., 60). |
| `--summary-max-scenes`| `auto` | Max number of scenes for summary mode. |
| `--style` | `viral_fb` | Narration style (`viral_fb`, `documentary`, `technical`, `calm_educational`). |
| `--lang` | `id en` | Target languages for narration. |
| `--engine` | `edge` | TTS Engine to use (`edge`). |
| `--ratio` | `16:9` | Output video aspect ratios (e.g., `16:9 9:16`). |
| `--vision-provider` | `gemini` | Vision API choice (`gemini` or `openai` for local vLLM/Qwen). |
| `--vision-model` | - | Override model name if using `--vision-provider openai`. |
| `--max-keyframes` | `7` | Max keyframes per scene for vision analysis. Lower (3-5) = faster. |
| `--min-scene-duration`| `2.0` | Minimum scene duration in seconds. Higher (4-6) = fewer scenes = faster. |
| `--scene-threshold` | `3.0` | Scene detection sensitivity. Higher (4-6) = fewer scenes = faster. |

**Example:**
```bash
dubbingstory run --url "https://youtube.com/watch?v=..." --i-have-rights --style documentary --lang id --engine edge --ratio 16:9
```

---

### `ingest` (Download & Validate)
Download (if URL), validate format, and read subtitles.

| Argument | Default | Description |
|----------|---------|-------------|
| `--input`, `-i` | `None` | Local video file. |
| `--url`, `-u` | `None` | Video URL. |
| `--i-have-rights` | `False` | Confirm rights. |
| `--project`, `-p` | `auto` | Project name. |

**Example:**
```bash
dubbingstory ingest --input my_video.mp4 --project my_awesome_project
```

---

### `segment` (Scene & Keyframes)
Detect scene changes and extract keyframes.

| Argument | Default | Description |
|----------|---------|-------------|
| `--project`, `-p` | **(Required)** | Project name. |

**Example:**
```bash
dubbingstory segment --project my_awesome_project
```

---

### `analyze` (Visual Understanding)
Use AI to visually understand each scene.

| Argument | Default | Description |
|----------|---------|-------------|
| `--project`, `-p` | **(Required)** | Project name. |
| `--domain` | `""` | Domain hint to help AI (e.g., `workshop`, `cooking`, `repair`). |
| `--vision-provider` | `gemini` | Vision API choice (`gemini` or `openai` for local vLLM/Qwen). |

**Example:**
```bash
dubbingstory analyze --project my_awesome_project --domain repair
```

---

### `narrate` (Script Generation)
Generate multilingual narration scripts and subtitles (SRT).

| Argument | Default | Description |
|----------|---------|-------------|
| `--project`, `-p` | **(Required)** | Project name. |
| `--style` | `viral_fb`| Narration style. |
| `--lang` | `id en` | Target languages. |

**Example:**
```bash
dubbingstory narrate --project my_awesome_project --style calm_educational --lang id en
```

---

### `dub` (Text-to-Speech)
Convert scripts to audio using AI Voice.

| Argument | Default | Description |
|----------|---------|-------------|
| `--project`, `-p` | **(Required)** | Project name. |
| `--engine` | `edge` | TTS engine (`edge`). |

**Example:**
```bash
dubbingstory dub --project my_awesome_project --engine edge
```

---

### `render` (Final Video Assembly)
Combine original video, narration audio (ducking/replace), and set aspect ratios.

| Argument | Default | Description |
|----------|---------|-------------|
| `--project`, `-p` | **(Required)** | Project name. |
| `--ratio` | `16:9` | Aspect ratios. |

**Example:**
```bash
dubbingstory render --project my_awesome_project --ratio 16:9 9:16
```

---

## 📁 Output Structure

```
outputs/{project_name}/
├── source.mp4              # Original video
├── video_metadata.json     # Video info
├── ingest_manifest.json    # Ingest status
├── scenes/                 # Split scene files
│   ├── scene_001.mp4
│   ├── scene_002.mp4
│   └── ...
├── keyframes/              # Extracted keyframes
│   ├── scene_001/
│   │   ├── scene_001_kf00.jpg
│   │   └── ...
│   └── ...
├── storyboard.json         # Visual analysis + narrative
├── scripts/
│   ├── script_id.txt       # Indonesian narration
│   ├── script_en.txt       # English narration
│   ├── script_id.srt       # Indonesian subtitles
│   └── script_en.srt       # English subtitles
├── audio/
│   ├── audio_id.wav        # Indonesian TTS audio
│   └── audio_en.wav        # English TTS audio
├── final_id_16x9.mp4       # Final dubbed video (ID, landscape)
├── final_en_16x9.mp4       # Final dubbed video (EN, landscape)
├── final_id_9x16.mp4       # Final dubbed video (ID, vertical)
└── final_en_9x16.mp4       # Final dubbed video (EN, vertical)
```

## 🎨 Narration Styles

| Style | Tone | Best For |
|-------|------|----------|
| `viral_fb` (default) | Excited, casual, engaging | Facebook/TikTok repost |
| `documentary` | Formal, neutral, informative | Educational content |
| `technical` | Precise, expert, analytical | Technical tutorials |
| `calm_educational` | Gentle, patient, step-by-step | Learning videos |

## 🔧 TTS Engines

| Engine | GPU | License | Indonesian Support | Quality |
|--------|-----|---------|--------------------|---------|
| **Edge TTS** | ❌ Online API | Free | ✅ Native | Excellent |

## 🎬 Film Support

For films with existing subtitles (SRT/ASS/VTT), DubbingStory reads the subtitle text as additional context for visual understanding:

```bash
# Place subtitle file alongside video:
# movie.mp4 + movie.srt → subtitle will be auto-detected
dubbingstory run --input movie.mp4
```

## 📋 Requirements

- Python 3.10+
- FFmpeg (in PATH)
- Google API Key (Gemini, free tier OK)
- GPU with vLLM (Only if you wish to run the vision model locally via `--vision-provider openai`)

## 🤝 Related Project

Sister project of [opensource-clipping](https://github.com/NaufalRizqullah/opensource-clipping) — AI Auto-Clipper for viral short-form content.
