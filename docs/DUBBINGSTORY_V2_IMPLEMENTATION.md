# DubbingStory v2 — Implementation Notes

This patch implements the first end-to-end storytelling/voiceover redesign discussed after reviewing the generated `summary_script_id.txt` and `backup_tts_summary_id.wav`.

## What changed

### 1. Vision is factual, not the narrator

`dubbingstory/vision/prompts.py` now asks the vision model for compact story-relevant facts such as:

- action / visible change
- character goal (only when supported)
- state before / state after
- cause / effect (only when supported)
- unresolved question
- transcript context

The OpenAI-compatible/Qwen path is also hardened against runaway generation:

- per-scene output capped at 640 tokens
- no retry inflation to 3072 tokens
- one-middle-frame rescue at 384 tokens
- token usage + truncated output head/tail logging

### 2. Summary mode propagation is fixed

`cmd_run()` now sets `cfg.mode = mode` before analysis, so `run --mode summary` actually activates the two-pass summary vision flow.

Summary analysis now uses visual + transcript relevance and keeps timeline coverage in the deep-analysis candidate set.

### 3. New persistent story artifacts

After visual analysis, the pipeline creates:

- `scene_cards.json`
- `story_plan.json`
- `story_memory.json`

`story_plan.json` contains the global premise, goal/conflict, story arc, per-scene story/causal/bridge importance, stable characters/objects, and unresolved threads.

`story_memory.json` carries continuity facts that later writing must preserve.

If the global Gemini planning call fails, deterministic storyboard-derived fallbacks are saved so the pipeline can continue.

### 4. Narration is planned before it is written

`narration_plan.json` is now created before narration generation.

The planner:

- groups adjacent visual shots into human-sized narration beats
- allows a narration thought to span multiple video cuts
- calculates word budgets from actual available duration
- targets roughly 160 WPM by default
- aims for ~84% narration coverage, adjusted by story role
- marks continuity intent and transition strategy

Default config:

```yaml
narration:
  target_wpm: 160
  min_wpm: 135
  max_wpm: 175
  speech_coverage: 0.84
  group_scenes: true
  target_beat_duration: 11.0
  max_beat_duration: 18.0
```

### 5. Story-aware writer + QA

The narration writer now receives:

- global story plan
- story memory
- selected scene evidence
- narration beat plan
- per-beat word budgets

The prompt prioritizes:

`PROCESS/CHANGE -> REASON -> CONSEQUENCE -> NEXT STEP`

It explicitly suppresses repeated generic hype such as `Lihat ini!`, `Luar biasa!`, `Amazing!`, and similar filler.

`narration_qa.json` checks:

- sparse/dense narration
- excessive exclamation marks
- generic reaction phrases
- repetitive sentence openings
- missing planned beats
- a basic continuity score

If enabled, one Gemini rewrite pass is attempted when QA finds issues.

### 6. Summary selection is story-aware

`scene_selector.py` now scores more than visual salience. It uses:

- global story importance
- causal importance
- bridge importance
- narrative role
- confidence
- temporal importance
- duration fitness

The duration hard limit defaults to target + 5%, replacing the previous ~20% overshoot allowance.

### 7. TTS no longer pads every short sentence to a whole scene

This is the major audio-timing change.

The old behavior effectively did:

```text
scene 10s
speech 3s
=> 3s speech + 7s padded silence
```

The new `voice_manager` reads structured narration JSON and places natural-length speech clips on one full-video timeline.

Short clips are **not padded per scene**.

If a clip is slightly too long, the scheduler can gently speed it up (default max 1.12x). It does not aggressively time-stretch normal speech.

Generated audio metadata includes:

- actual placement per narration beat
- total speech duration
- speech coverage
- timeline duration
- overflow warning when applicable

### 8. Dynamic original-audio ducking

The default render strategy is now:

```yaml
render:
  audio_strategy: dynamic_duck
```

FFmpeg side-chain compression lowers original audio only while narration is active. During intentional narration gaps, workshop/road/tool ambience naturally returns instead of remaining permanently muted.

Legacy `duck`, `replace`, and `mute_original` modes remain available.

### 9. Google Chirp 3 HD provider

A new optional engine is available:

```bash
--engine chirp
```

Default Indonesian voice:

```text
id-ID-Chirp3-HD-Charon
```

This provider uses Google Cloud Text-to-Speech Application Default Credentials (ADC), **not** the Gemini `GOOGLE_API_KEY`.

Example environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

Edge and Piper remain available and remain useful for free/local fallback testing.

## New/important output files

A run can now produce:

```text
outputs/<project>/
├── storyboard.json
├── scene_cards.json
├── story_plan.json
├── story_memory.json
├── summary_manifest.json          # summary mode
├── narration_plan.json
├── narration_qa.json
├── scripts/
│   ├── summary_script_id.txt      # or script_id.txt in full mode
│   ├── summary_script_id.json     # structured timeline source for TTS
│   └── summary_script_id.srt
└── audio/
    └── summary_audio_id.wav
```

These artifacts are intentionally separate so bad output can be diagnosed by stage instead of treating the pipeline as one black box.

## Recommended first test

Keep the same Qwen model and existing vLLM settings for the first comparison run:

```text
Qwen/Qwen3-VL-2B-Instruct
max-model-len = 12288
vision max tokens = 640
```

Run the same source video again in summary mode, then compare:

1. `scene_cards.json`
2. `story_plan.json`
3. `narration_plan.json`
4. `summary_script_id.txt`
5. `narration_qa.json`
6. `backup_tts_summary_id.wav`

The most important validation is not merely whether the run finishes. Check whether:

- the script explains what/why/consequence rather than reacting to frames
- the same object/character/process remains consistent
- bridge scenes keep the story understandable
- narration density is substantially higher than the previous ~33% active-speech result
- original ambience returns naturally between speech beats

## Applying the patch

From the root of a clean repository checkout:

```bash
git apply dubbingstory-v2-storytelling.patch
pip install -r requirements.txt
```

Then review:

```bash
git diff
```

If you prefer manual replacement, use the replacement ZIP and copy its directory tree over the repository root.

## Validation performed on this implementation

The implementation was syntax-compiled and tested with six automated tests covering:

- transcript-to-scene-card overlap
- duration-aware narration budgeting
- story-aware summary duration limit
- QA detection of sparse/hype-heavy narration
- full-length timeline audio generation without per-scene padding
- FFmpeg dynamic side-chain ducking

A live end-to-end Gemini + Qwen/vLLM + real TTS run cannot be performed in this isolated implementation environment, so the first Kaggle/Colab run should be treated as an integration validation. The new JSON artifacts and logs were added specifically to make that validation straightforward.
