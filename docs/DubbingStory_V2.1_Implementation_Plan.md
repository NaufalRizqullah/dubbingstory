# DubbingStory V2.1 — Scene Continuity, Story Matching & Quality Plan

> **Status:** Future implementation plan after DubbingStory V2 is stable  
> **Target:** Improve scene-context continuity, narration-to-footage accuracy, story flow, and automated QA  
> **Reference concepts:** VIDEOFCK + movie-narrator  
> **Implementation policy:** Re-implement concepts in DubbingStory's own architecture. Do not copy source code from `movie-narrator`.

---

## 1. Why V2.1 Exists

DubbingStory V2 should first prove that the basic end-to-end pipeline is stable:

```text
video
→ scene detection
→ visual analysis
→ transcript/ASR
→ storyboard
→ scene selection
→ narration
→ TTS
→ timeline
→ render
```

V2.1 is not intended to replace that pipeline.

Its purpose is to solve the next quality problems:

1. Narration describes individual scenes correctly, but the story can still feel disconnected.
2. Adjacent scenes may repeat the same visual information.
3. A narration sentence may be semantically correct for the video globally but mismatched with the footage shown at that exact moment.
4. Summary selection can contain redundant scenes.
5. LLM retries may regenerate blindly instead of correcting a known problem.
6. Narration pacing may be technically valid but not aligned with the dramatic role of each beat.

The V2.1 principle is:

> **Understand locally, plan globally, write with memory, verify against footage.**

---

# 2. Main Ideas Adopted

## 2.1 From VIDEOFCK

The useful concepts are:

- previous-caption → current-caption continuity;
- describe only **new or changed information**;
- previous/current/next context during narration writing;
- treat nearby frames as a continuous sequence;
- per-segment word budget;
- timestamp-aware narration placement.

The most important lesson:

```text
Do not analyze each scene as an isolated image.

Instead:

previous scene
+ current scene
+ next scene
→ determine what actually changed
```

---

## 2.2 From movie-narrator

The useful concepts are:

- two-phase script generation;
- structured plot beats;
- beat time anchors (`approx_ratio`);
- narrative/rhythm roles;
- beat deduplication;
- semantic narration ↔ scene matching;
- diversity / reuse penalty;
- quality judge;
- feedback-guided retries;
- alignment QA.

The most important lesson:

```text
Do not trust the first selected footage or first generated script.

Generate
→ match
→ score
→ judge
→ repair if needed
```

---

# 3. V2.1 Target Architecture

```text
VIDEO
  │
  ▼
Scene Detection
  │
  ▼
Transcript / ASR
  │
  ▼
Qwen Visual Scene Facts
  │
  ▼
scene_cards.json
  │
  ▼
┌──────────────────────────────────────┐
│ Local Scene Continuity Pass          │
│ previous + current + next scene      │
│ → identify new/change/continuation   │
└──────────────────────────────────────┘
  │
  ▼
scene_delta.json
  │
  ▼
Global Story Planner
  │
  ▼
story_plan.json
  │
  ▼
Story Memory Builder
  │
  ▼
story_memory.json
  │
  ▼
Plot Beat Extraction
  │
  ▼
Beat Deduplication
  │
  ▼
beats.json
  │
  ├── act
  ├── approx_ratio
  ├── rhythm_zone
  ├── emotion
  └── story intent
  │
  ▼
Story-aware Scene Selector
  │
  ▼
narration_plan.json
  │
  ▼
Rolling Continuity Writer
  │
  ├── previous narration
  ├── previous scene delta
  ├── current scene delta
  ├── next scene delta
  ├── story memory
  └── remaining word budget
  │
  ▼
narration_script.json
  │
  ▼
Narration ↔ Scene Semantic Lock
  │
  ├── semantic similarity
  ├── timeline proximity
  ├── story-role compatibility
  ├── visual confidence
  ├── continuity score
  └── reuse penalty
  │
  ▼
Mismatch?
  ├── YES → rewrite narration OR rematch scene
  └── NO
  │
  ▼
LLM Script Judge
  │
  ▼
Deterministic Script QA
  │
  ▼
TTS
  │
  ▼
Actual Audio Duration
  │
  ▼
Timeline Scheduler
  │
  ▼
Optional faster-whisper Alignment QA
  │
  ▼
Dynamic Original-Audio Ducking
  │
  ▼
FINAL VIDEO
```

---

# 4. Implementation Order

Do **not** implement everything at once.

Recommended rollout:

| Phase | Feature | Priority | Expected Impact |
|---|---|---:|---:|
| V2.1-A | Scene Delta / Local Continuity | P0 | Very High |
| V2.1-B | Rolling Narration Context | P0 | Very High |
| V2.1-C | Narration ↔ Scene Semantic Lock | P0 | Very High |
| V2.1-D | Beat `approx_ratio` + roles | P1 | High |
| V2.1-E | Beat Deduplication | P1 | High |
| V2.1-F | Judge + Feedback Retry | P1 | High |
| V2.1-G | Scene Reuse / Diversity Penalty | P1 | High |
| V2.1-H | Rhythm + Emotion | P2 | Medium |
| V2.1-I | faster-whisper Alignment QA | P2 | Medium |
| V2.1-J | Emotion-aware TTS/BGM | P3 | Optional |

---

# 5. Prerequisites Before Starting V2.1

V2 should be considered stable before adding V2.1.

Minimum requirements:

- [ ] `--mode summary` correctly reaches summary analysis flow.
- [ ] Qwen3-VL scene analysis no longer frequently dies on runaway `finish_reason=length`.
- [ ] Per-scene output is bounded.
- [ ] Storyboard is generated reliably.
- [ ] Summary scene selection runs reliably.
- [ ] Narration JSON is parseable and validated.
- [ ] TTS works per segment.
- [ ] Actual audio durations are available.
- [ ] Timeline scheduler does not pad every TTS segment to scene length.
- [ ] Final render completes.
- [ ] Pipeline logs enough metadata to debug a failed run.

Only after these are stable should V2.1 begin.

---

# 6. V2.1-A — Scene Cards

## Goal

Create a normalized representation of each scene that can be reused by:

- story planning;
- continuity analysis;
- semantic matching;
- narration;
- QA.

Recommended file:

```text
dubbingstory/story/scene_cards.py
```

Output:

```text
<project>/scene_cards.json
```

---

## 6.1 Scene Card Schema

```json
{
  "scene_id": "scene_024",
  "scene_index": 23,

  "start_time": 142.30,
  "end_time": 149.80,
  "duration": 7.50,

  "visual": {
    "subjects": [
      "mechanic"
    ],
    "objects": [
      "metal component",
      "lathe"
    ],
    "action": "turning the outer diameter of a metal component",
    "setting": "workshop",
    "visible_state": "the component is rotating in a lathe",
    "confidence": 0.91
  },

  "audio": {
    "subtitle_text": "ukurannya harus pas",
    "asr_text": "ukurannya harus pas"
  },

  "semantic": {
    "goal": "reduce the outer diameter to the required size",
    "likely_context": "machining a custom replacement component",
    "importance": 0.72
  },

  "source": {
    "vision_provider": "openai",
    "vision_model": "Qwen/Qwen3-VL-2B-Instruct"
  }
}
```

---

## 6.2 Rules

A scene card should contain **facts**, not narration style.

Avoid fields such as:

```json
{
  "dramatic_description": "The heroic mechanic bravely..."
}
```

Prefer:

```json
{
  "action": "the mechanic reduces the component diameter on a lathe"
}
```

The scene card is the factual layer.

Styling happens later.

---

# 7. V2.1-A — Local Scene Continuity / Scene Delta

## Goal

Determine how the current scene relates to nearby scenes.

Recommended file:

```text
dubbingstory/story/scene_continuity.py
```

Output:

```text
<project>/scene_delta.json
```

---

## 7.1 Context Window

Default:

```text
previous scene
current scene
next scene
```

Optional:

```yaml
continuity_window: 1
```

Meaning:

```text
scene[i-1]
scene[i]
scene[i+1]
```

For difficult videos:

```yaml
continuity_window: 2
```

Meaning:

```text
scene[i-2]
scene[i-1]
scene[i]
scene[i+1]
scene[i+2]
```

Do not send the whole video to the vision model.

---

## 7.2 Scene Delta Schema

```json
{
  "scene_id": "scene_024",

  "continuation_from": "scene_023",

  "relationship": "continuation",

  "persistent_subjects": [
    "mechanic"
  ],

  "persistent_objects": [
    "metal component"
  ],

  "new_objects": [],

  "new_action": "the outer diameter is reduced further",

  "state_change": {
    "before": "rough oversized cylindrical section",
    "after": "smaller and more uniform outer diameter"
  },

  "new_information_only": [
    "the diameter is now being reduced"
  ],

  "continuity_summary":
    "Machining continues on the same component, now reducing its outer diameter.",

  "confidence": 0.86
}
```

---

## 7.3 Relationship Types

Use a controlled enum:

```text
setup
continuation
progression
transition
result
reaction
parallel
new_topic
uncertain
```

Example:

```json
{
  "relationship": "result"
}
```

means:

> this scene primarily shows the result of the process shown previously.

---

## 7.4 Prompt Principle

The continuity model should be instructed:

```text
You are not writing narration.

Your job is to identify:
1. what remains the same;
2. what changed;
3. what is newly visible;
4. whether the current scene continues, transitions, or concludes the previous action.

Do not repeat unchanged information.
Do not speculate beyond visual/transcript evidence.
Return structured JSON only.
```

---

## 7.5 Cheap First Implementation

Do not immediately add another expensive visual inference.

First implementation can operate on:

```text
previous scene card
+
current scene card
+
next scene card
```

using a **text-only LLM call**.

Later, if needed:

```text
scene cards
+
representative keyframes
```

can be used selectively for low-confidence continuity cases.

---

# 8. V2.1-D — Global Story Plan

Recommended file:

```text
dubbingstory/story/story_planner.py
```

Output:

```text
story_plan.json
```

Suggested schema:

```json
{
  "video_type": "repair/process",

  "story_goal":
    "show how a damaged or unsuitable component is replaced with a custom-machined part",

  "main_subjects": [
    "mechanic",
    "motorcycle"
  ],

  "central_object": "custom metal component",

  "story_arc": {
    "setup": "a component needs replacement or modification",
    "development": "measurements and machining are performed",
    "peak": "the finished part is tested or installed",
    "resolution": "the repair is completed or evaluated"
  },

  "must_preserve": [
    "why the component is being modified",
    "the main machining steps",
    "the final result"
  ],

  "can_skip": [
    "repetitive machine operation with no visible state change"
  ]
}
```

---

# 9. Story Memory

Recommended file:

```text
dubbingstory/story/story_memory.py
```

Output:

```text
story_memory.json
```

Purpose:

Prevent the writer from reintroducing the same information.

Example:

```json
{
  "known_entities": {
    "mechanic": {
      "introduced": true,
      "preferred_reference": "mekanik"
    },

    "custom_part": {
      "introduced": true,
      "preferred_reference": "komponen"
    }
  },

  "facts_already_explained": [
    "the component must fit a specific size",
    "the outer diameter is being reduced"
  ],

  "open_questions": [
    "will the finished component fit correctly?"
  ],

  "current_goal":
    "finish shaping the custom component",

  "last_narration":
    "Karena ukurannya belum sesuai, komponen ini harus dibentuk ulang sebelum bisa dipasang."
}
```

---

# 10. V2.1-D — Structured Plot Beats

Recommended file:

```text
dubbingstory/story/beat_planner.py
```

Output:

```text
beats.json
```

---

## 10.1 Beat Schema

```json
{
  "beat_id": "beat_006",

  "text":
    "the mechanic reduces the component diameter until it approaches the required measurement",

  "act": 2,

  "approx_ratio": 0.38,

  "rhythm_zone": "rising",

  "emotion": "focused",

  "importance": 0.82,

  "story_intent":
    "explain why precision machining is necessary",

  "required_facts": [
    "the same component continues to be machined",
    "its diameter becomes smaller"
  ]
}
```

---

## 10.2 `act`

Recommended:

```text
1 = setup
2 = development
3 = climax / key result
4 = resolution
```

---

## 10.3 `approx_ratio`

Range:

```text
0.0 → beginning of original video
1.0 → ending of original video
```

Example:

```json
{
  "approx_ratio": 0.72
}
```

means:

> This story beat is expected around 72% through the original video's timeline.

This is only a **soft time anchor**.

It must not override a clearly better semantic match.

---

## 10.4 `rhythm_zone`

Recommended controlled values:

```text
hook
rising
peak
settle
```

---

## 10.5 `emotion`

Keep the first implementation simple:

```text
neutral
curious
focused
suspense
intense
satisfying
calm
```

Do not make emotion mandatory for scene selection until the core system is stable.

---

# 11. V2.1-E — Beat Deduplication

Recommended file:

```text
dubbingstory/story/beat_dedup.py
```

Problem:

```text
beat A:
the component is machined

beat B:
the lathe continues shaping the metal

beat C:
precision machining continues

beat D:
the metal is worked further
```

These may represent one story event.

Desired output:

```text
the component is progressively machined until it reaches the required dimensions
```

---

## 11.1 First Version

Use normalized text + lexical similarity.

Possible algorithm:

```python
normalize()
→ token set
→ Jaccard similarity
```

Pseudo:

```python
if similarity(beat_a, beat_b) >= 0.82:
    merge_or_drop_lower_importance()
```

Later upgrade:

```text
embedding cosine similarity
```

Suggested config:

```yaml
story:
  beat_dedup_enabled: true
  beat_dedup_threshold: 0.84
```

---

# 12. Story-Aware Scene Selector

The current summary selector should evolve from:

```text
importance
+ confidence
+ duration
+ role
```

to:

```text
semantic relevance
+ story importance
+ temporal proximity
+ state change
+ visual confidence
+ continuity value
- redundancy
- reuse penalty
```

---

## 12.1 Suggested Score

Initial formula:

```python
final_score = (
    semantic_score      * 0.35
    + importance_score  * 0.20
    + temporal_score    * 0.15
    + state_change      * 0.10
    + visual_confidence * 0.10
    + continuity_score  * 0.10
    - redundancy_penalty
    - reuse_penalty
)
```

Do not hard-code forever.

Move weights to config.

---

## 12.2 Suggested Config

```yaml
summary:
  selector_v2_enabled: true

  weights:
    semantic: 0.35
    importance: 0.20
    temporal: 0.15
    state_change: 0.10
    visual_confidence: 0.10
    continuity: 0.10

  redundancy_penalty: 0.15
  reuse_penalty: 0.10
```

---

# 13. V2.1-G — Diversity / Reuse Penalty

Problem:

The same type of footage can dominate a summary because it has high semantic relevance.

Example:

```text
lathe scene
lathe scene
lathe scene
lathe scene
```

Desired behavior:

```text
setup
measurement
machining
inspection
installation
result
```

---

## 13.1 Scene Reuse Memory

Maintain:

```python
recent_scene_ids = deque(maxlen=3)
recent_action_types = deque(maxlen=3)
recent_objects = deque(maxlen=3)
```

Penalty:

```python
if candidate.scene_id in recent_scene_ids:
    score -= reuse_penalty
```

Also penalize near-identical actions:

```python
if candidate.action_embedding ~ recent_action_embedding:
    score -= redundancy_penalty
```

---

# 14. Narration Plan

Recommended output:

```text
narration_plan.json
```

Schema:

```json
{
  "beat_id": "beat_006",

  "scene_ids": [
    "scene_021",
    "scene_022",
    "scene_024"
  ],

  "story_role": "development",

  "act": 2,

  "approx_ratio": 0.38,

  "rhythm_zone": "rising",

  "emotion": "focused",

  "intent":
    "explain why the component needs precision machining",

  "causal_from": "beat_005",

  "causal_to": "beat_007",

  "must_explain": [
    "the component cannot be installed at its current size",
    "its diameter is reduced gradually"
  ],

  "must_not_repeat": [
    "the mechanic is using a lathe"
  ],

  "max_words": 24
}
```

---

# 15. V2.1-B — Rolling Continuity Writer

Recommended file:

```text
dubbingstory/story/continuity_writer.py
```

This should eventually replace or wrap the current summary narration writer.

---

## 15.1 Writer Input

For beat `i`, provide:

```json
{
  "story_memory": {},

  "previous_narration":
    "Karena ukurannya belum sesuai, komponen ini harus dibentuk ulang.",

  "previous_scene_delta": {},

  "current_scene_delta": {},

  "next_scene_delta": {},

  "beat": {},

  "remaining_total_words": 210,

  "max_words_current": 24,

  "style": "viral_fb",

  "language": "id"
}
```

---

## 15.2 Writer Rules

The writer must:

1. continue the story instead of restarting it;
2. focus on **new information**;
3. avoid reintroducing known subjects;
4. maintain causal flow;
5. prepare naturally for the next visual;
6. stay within the word budget;
7. never claim something that is not supported by scene facts;
8. avoid repetitive transition phrases;
9. preserve the chosen narrator perspective;
10. output structured JSON only.

---

## 15.3 Example

Bad:

```text
Mekanik kembali menggunakan mesin bubut untuk memproses komponen logam.
```

Better:

```text
Setelah ukurannya diperiksa, diameter luar kembali dikurangi sedikit demi sedikit agar komponen benar-benar pas.
```

The second version explains **why the repeated-looking scene matters**.

---

# 16. Global Word Budget

Do not only limit individual scenes.

Maintain:

```text
target video duration
× target WPM
```

Example:

```text
180 seconds
= 3 minutes

target_wpm = 145

global target ≈ 435 words
```

Then allocate by importance:

```python
beat_word_budget = (
    total_word_budget
    * normalized_importance
)
```

With min/max clamps.

Suggested config:

```yaml
narration:
  target_wpm: 145
  min_words_per_beat: 5
  max_words_per_beat: 35
```

---

# 17. V2.1-C — Narration ↔ Scene Semantic Lock

This is one of the highest-value V2.1 features.

Recommended file:

```text
dubbingstory/story/semantic_matcher.py
```

---

## 17.1 Scene Embedding Text

Do not embed only ASR.

Build a richer text representation:

```text
action
+
state change
+
objects
+
goal
+
continuity summary
+
subtitle / ASR
```

Example:

```text
The mechanic continues machining the same custom metal component.
Its outer diameter is gradually reduced.
The goal is to reach the required fitting dimension.
Subtitle: "ukurannya harus pas"
```

---

## 17.2 Narration Embedding

Embed:

```text
beat narration text
```

Then compare with scene representations.

---

## 17.3 Candidate Restriction

Do not compare every beat against every scene if unnecessary.

First restrict using:

```text
approx_ratio
```

Example:

```python
expected_time = approx_ratio * total_video_duration
```

Candidate window:

```yaml
semantic_match:
  time_window_ratio: 0.18
```

Then only compare nearby candidate scenes.

Fallback:

```text
if best score is too low
→ expand search window
```

---

## 17.4 Suggested Embedding Model

A multilingual sentence-transformer is appropriate because narration and source transcript may use different languages.

Initial candidate:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Keep model configurable.

```yaml
semantic_match:
  embedding_model: paraphrase-multilingual-MiniLM-L12-v2
```

---

# 18. Match Score

Suggested first version:

```python
final_match_score = (
    semantic_similarity * 0.50
    + temporal_score     * 0.15
    + continuity_score   * 0.15
    + importance_score   * 0.10
    + visual_confidence  * 0.10
    - reuse_penalty
)
```

Example result:

```json
{
  "beat_id": "beat_006",

  "scene_id": "scene_024",

  "semantic_score": 0.83,
  "temporal_score": 0.91,
  "continuity_score": 0.86,
  "importance_score": 0.78,
  "visual_confidence": 0.91,

  "reuse_penalty": 0.00,

  "final_match_score": 0.85
}
```

---

# 19. Match Thresholds

Suggested:

```yaml
semantic_match:
  high_confidence: 0.72
  warning_threshold: 0.58
  repair_threshold: 0.48
```

Behavior:

```text
>= 0.72
→ accept

0.58–0.72
→ accept + log warning

0.48–0.58
→ run repair attempt

< 0.48
→ force rematch or rewrite
```

These numbers must be tuned from real output.

---

# 20. Repair Strategy

When mismatch occurs, do **not** blindly regenerate everything.

Determine the likely failure type.

---

## 20.1 Case A — Footage Is Correct, Narration Is Wrong

Example:

```text
scene:
lathe machining

narration:
the finished component is installed on the motorcycle
```

Repair:

```text
rewrite narration using fixed scene facts
```

Prompt:

```text
The visual scene is fixed and cannot change.

Rewrite the narration so it accurately describes the story event
supported by this scene.

Keep the same story role and word budget.
```

---

## 20.2 Case B — Narration Is Correct, Footage Is Wrong

Example:

```text
beat:
installation

selected scene:
lathe machining

alternative scene:
actual installation
```

Repair:

```text
rematch footage
```

---

## 20.3 Case C — Both Are Ambiguous

Use:

```text
story plan
+
neighboring beats
+
scene delta
```

Then choose the lower-cost correction.

---

# 21. Match Audit Output

Output:

```text
match_report.json
```

Example:

```json
{
  "summary": {
    "total_beats": 14,
    "high_confidence": 11,
    "warnings": 2,
    "repairs": 1,
    "failed": 0
  },

  "matches": [
    {
      "beat_id": "beat_006",
      "scene_ids": [
        "scene_024"
      ],
      "score": 0.85,
      "status": "accepted",
      "repair": null
    }
  ]
}
```

---

# 22. V2.1-F — LLM Script Judge

Recommended file:

```text
dubbingstory/story/script_judge.py
```

The judge should evaluate the **whole narration**, not individual sentences only.

---

## 22.1 Dimensions

Recommended:

```text
continuity
scene_fidelity
causal_clarity
information_density
anti_ai_tone
repetition
timing_fit
ending_quality
```

Example:

```json
{
  "continuity": 8,
  "scene_fidelity": 9,
  "causal_clarity": 7,
  "information_density": 8,
  "anti_ai_tone": 7,
  "repetition": 8,
  "timing_fit": 9,
  "ending_quality": 7,

  "verdict": "pass",

  "issues": []
}
```

---

## 22.2 Example Failure

```json
{
  "continuity": 5,
  "scene_fidelity": 8,
  "causal_clarity": 4,
  "information_density": 6,
  "anti_ai_tone": 5,
  "repetition": 3,
  "timing_fit": 9,
  "ending_quality": 6,

  "verdict": "retry",

  "issues": [
    "beats 4-6 repeat the same machining information",
    "transition between beat 8 and beat 9 lacks a causal connection",
    "several sentences begin with the same transition pattern"
  ]
}
```

---

# 23. Feedback-Guided Retry

Do not retry with the same prompt.

Inject previous judge issues:

```text
Previous attempt failed for these reasons:

1. beats 4-6 repeat the same machining information
2. beat 8 → 9 lacks a causal connection
3. sentence openings are repetitive

Correct those issues while preserving:
- scene assignments
- factual constraints
- maximum word budgets
- story order
```

Retry only affected beats where possible.

---

# 24. Deterministic Script QA

The LLM judge should not be the only validator.

Recommended file:

```text
dubbingstory/story/script_qa.py
```

Checks:

```text
segment count
empty text
max word count
total word count
duplicate sentence similarity
repetitive openings
scene IDs exist
chronological order
match scores available
unsupported scene IDs
```

---

## 24.1 Example QA Output

```json
{
  "passed": false,

  "issues": [
    {
      "type": "word_limit",
      "beat_id": "beat_004",
      "actual_words": 29,
      "max_words": 22
    },

    {
      "type": "near_duplicate",
      "beat_ids": [
        "beat_005",
        "beat_006"
      ],
      "similarity": 0.87
    }
  ]
}
```

---

# 25. Summary Duration Control

V2 previously showed that a target such as:

```text
180 seconds
```

can overshoot.

V2.1 should enforce a proper budget.

Use:

```python
remaining = target_duration

for candidate in ranked_candidates:
    if candidate.duration <= remaining:
        select(candidate)
        remaining -= candidate.duration
```

But also support partial trimming of an overly long final scene if safe.

Suggested config:

```yaml
summary:
  hard_duration_limit: true
  duration_tolerance_sec: 3.0
```

Acceptance:

```text
target = 180s
final source duration should normally be:
177–183s
```

unless no valid scene combination can satisfy it.

---

# 26. TTS Timing Strategy

Keep V2's scheduler architecture.

Do **not** return to:

```text
generate TTS
→ pad each audio to full scene duration
```

Preferred:

```text
generate narration audio
→ measure actual WAV/MP3 duration
→ place audio at scheduled timeline position
→ keep natural silence / original audio between narration beats
```

---

# 27. Optional V2.1-I — faster-whisper Alignment QA

Do not make WhisperX mandatory.

DubbingStory already has `faster-whisper`.

Use it as optional QA:

```text
generated TTS
→ faster-whisper transcribe
→ compare transcript to expected narration
→ compare actual speech duration
```

Possible checks:

```text
missing words
unexpected truncation
very large timing drift
silent TTS
```

Suggested config:

```yaml
audio_qa:
  enabled: false
  backend: faster-whisper
  model: small
```

Enable only after core V2.1 quality features are stable.

---

# 28. Dynamic Original Audio Ducking

Recommended behavior:

```text
Narration speaking:
original audio gain ↓

Narration silent:
original audio gain ↑
```

Example:

```yaml
audio:
  original_volume_normal: 0.85
  original_volume_under_narration: 0.20

  duck_attack_ms: 120
  duck_release_ms: 350
```

This preserves useful original sound instead of muting the entire source.

---

# 29. Config Proposal

Add a dedicated section.

```yaml
story_v2:
  enabled: false

  continuity:
    enabled: true
    window: 1
    visual_fallback: false
    confidence_threshold: 0.55

  memory:
    enabled: true

  beats:
    enabled: true
    dedup_enabled: true
    dedup_threshold: 0.84

  semantic_match:
    enabled: true
    embedding_model: paraphrase-multilingual-MiniLM-L12-v2

    time_window_ratio: 0.18

    weights:
      semantic: 0.50
      temporal: 0.15
      continuity: 0.15
      importance: 0.10
      visual_confidence: 0.10

    high_confidence: 0.72
    warning_threshold: 0.58
    repair_threshold: 0.48

    reuse_window: 3
    reuse_penalty: 0.08

  writer:
    rolling_context: true
    include_previous_narration: true
    include_previous_scene: true
    include_next_scene: true

  judge:
    enabled: true
    max_retries: 2

  qa:
    enabled: true

  rhythm:
    enabled: false

  audio_alignment_qa:
    enabled: false
```

---

# 30. Feature Flags

Every new subsystem should be independently switchable.

Important because V2.1 will be experimental.

Example:

```text
story_v2.enabled=false
```

must preserve V2 behavior.

Then allow:

```text
continuity only
```

or:

```text
continuity + semantic lock
```

without enabling everything.

---

# 31. Suggested Repository Changes

Potential layout:

```text
dubbingstory/
├── story/
│   ├── scene_cards.py
│   ├── scene_continuity.py
│   ├── story_planner.py
│   ├── story_memory.py
│   ├── beat_planner.py
│   ├── beat_dedup.py
│   ├── scene_matcher.py
│   ├── semantic_matcher.py
│   ├── continuity_writer.py
│   ├── script_judge.py
│   └── script_qa.py
│
├── vision/
│   ├── scene_understanding.py
│   └── prompts.py
│
├── tts/
│   └── voice_manager.py
│
├── render/
│   ├── video_cutter.py
│   └── video_render.py
│
└── cli.py
```

Do not create all files on day one.

Suggested actual sequence:

```text
1. scene_continuity.py
2. continuity_writer.py
3. semantic_matcher.py
4. script_qa.py
5. beat_planner.py
6. beat_dedup.py
7. script_judge.py
```

---

# 32. Suggested Pipeline Integration

Pseudo:

```python
storyboard = run_analysis(...)

if cfg.story_v2.enabled:
    scene_cards = build_scene_cards(storyboard)

    scene_deltas = build_scene_continuity(
        scene_cards=scene_cards,
        window=cfg.story_v2.continuity.window,
    )

    story_plan = build_story_plan(
        scene_cards=scene_cards,
        scene_deltas=scene_deltas,
        metadata=video_metadata,
    )

    story_memory = initialize_story_memory(story_plan)

    beats = build_story_beats(
        story_plan=story_plan,
        scene_cards=scene_cards,
        target_duration=summary_target_duration,
    )

    beats = deduplicate_beats(beats)

    narration_plan = assign_scenes_to_beats(
        beats=beats,
        scene_cards=scene_cards,
        scene_deltas=scene_deltas,
    )

    narration = write_narration_with_continuity(
        plan=narration_plan,
        story_memory=story_memory,
    )

    match_report = verify_narration_scene_matches(
        narration=narration,
        scene_cards=scene_cards,
        scene_deltas=scene_deltas,
    )

    narration = repair_low_confidence_matches(
        narration=narration,
        match_report=match_report,
    )

    narration = judge_and_repair_script(
        narration=narration,
        plan=narration_plan,
    )

    run_script_qa(narration)
```

---

# 33. Caching Strategy

V2.1 adds extra LLM and embedding work.

Cache aggressively.

Suggested files:

```text
cache/
├── scene_cards.json
├── scene_delta.json
├── story_plan.json
├── story_memory.json
├── beats.json
├── embeddings/
│   ├── scene_embeddings.npy
│   └── metadata.json
├── match_report.json
└── script_judge.json
```

Cache keys should include:

```text
source video identity
model
prompt version
relevant config
scene timestamps
```

Do not invalidate everything because of an unrelated TTS config change.

---

# 34. Prompt Versioning

Every important structured prompt should have a version.

Example:

```python
SCENE_CONTINUITY_PROMPT_VERSION = "1.0"
BEAT_PLANNER_PROMPT_VERSION = "1.0"
CONTINUITY_WRITER_PROMPT_VERSION = "1.0"
SCRIPT_JUDGE_PROMPT_VERSION = "1.0"
```

Save prompt versions into output JSON.

Example:

```json
{
  "_meta": {
    "prompt_version": "1.0",
    "model": "gemini-2.5-flash"
  }
}
```

This makes regression analysis much easier.

---

# 35. Logging Requirements

Each major V2.1 stage should have visible logs.

Example:

```text
🧠 V2.1 Story pipeline enabled

🔗 Local continuity
   124 scenes
   89 continuation
   14 transition
   8 result
   9 new_topic
   4 uncertain

🧩 Plot beats
   requested: 18
   generated: 18
   deduplicated: 3
   final: 15

🎯 Semantic scene lock
   high confidence: 12
   warning: 2
   repaired: 1
   failed: 0

🧪 Script QA
   duplicate warnings: 0
   word-limit violations: 0
   unmatched beats: 0

✅ V2.1 story pipeline complete
```

---

# 36. Debug Artifacts

When something goes wrong, save enough information to reproduce it.

For each low-match beat:

```text
debug/match/beat_006.json
```

Example:

```json
{
  "beat": {},
  "narration": "...",

  "planned_scene": "scene_024",

  "top_candidates": [
    {
      "scene_id": "scene_024",
      "semantic": 0.44,
      "final": 0.49
    },
    {
      "scene_id": "scene_027",
      "semantic": 0.76,
      "final": 0.81
    }
  ],

  "repair_action": "rematch_scene"
}
```

---

# 37. Tests

Add tests before enabling V2.1 by default.

---

## 37.1 Scene Continuity Tests

```text
test_same_subject_continuation
test_new_action_detected
test_repeated_details_removed
test_new_topic_detected
test_uncertain_scene_does_not_invent_fact
```

---

## 37.2 Beat Tests

```text
test_beats_keep_chronological_order
test_beats_have_valid_ratio
test_beats_have_valid_rhythm_zone
test_duplicate_beats_are_removed
test_important_result_is_preserved
```

---

## 37.3 Matching Tests

```text
test_semantic_match_prefers_correct_scene
test_time_anchor_breaks_semantic_tie
test_strong_semantic_match_overrides_time_anchor
test_recent_scene_gets_reuse_penalty
test_low_match_triggers_repair
test_rematch_does_not_break_chronology
```

---

## 37.4 Writer Tests

```text
test_writer_receives_previous_narration
test_writer_receives_next_scene_context
test_writer_does_not_repeat_known_fact
test_writer_respects_word_limit
test_writer_preserves_scene_facts
```

---

## 37.5 QA Tests

```text
test_word_limit_violation
test_duplicate_detection
test_invalid_scene_id
test_missing_narration
test_non_chronological_assignment
test_low_match_score_warning
```

---

# 38. Acceptance Criteria

V2.1 should not be considered successful just because it renders a video.

Evaluate real output.

---

## 38.1 Continuity

Target:

```text
Narration should not restart the explanation on every selected scene.
```

Manual rating:

```text
1 = disconnected scene descriptions
3 = mostly connected
5 = clearly one continuous story
```

Target:

```text
>= 4/5
```

---

## 38.2 Scene Fidelity

For every narration beat:

```text
Does the visible footage support what the narration says?
```

Target:

```text
>= 90% of beats clearly supported by footage
```

---

## 38.3 Redundancy

Target:

```text
No more than 2 adjacent narration beats should explain essentially the same event.
```

---

## 38.4 Word Budget

Target:

```text
0 hard per-beat word-limit violations
```

---

## 38.5 Summary Duration

Target:

```text
within ±3 seconds of requested duration
```

when possible.

---

## 38.6 Semantic Match

Initial experimental goal:

```text
>= 80% of selected beats have final_match_score >= high_confidence threshold
```

Thresholds must later be calibrated on real videos.

---

# 39. Evaluation Dataset

Do not evaluate V2.1 on only one video.

Create a small regression set.

Recommended:

```text
1. mechanical repair / restoration
2. cooking process
3. tutorial
4. vlog
5. documentary / informational
6. video with little speech
7. video with lots of dialogue
8. fast-cut social-media video
```

Store:

```text
tests/fixtures/eval_manifest.json
```

Example:

```json
[
  {
    "name": "motor_repair",
    "video": "...",
    "mode": "summary",
    "target_duration": 180
  }
]
```

---

# 40. A/B Evaluation

For every test video:

```text
V2
vs
V2.1
```

Compare:

```text
render success
runtime
vision failures
selected scene count
duration
repetition count
scene-match confidence
manual continuity score
manual scene-fidelity score
```

Output:

```text
evaluation/v2_vs_v21.json
```

---

# 41. Performance Guardrails for Kaggle T4

V2.1 should not overload the Qwen vision server.

Keep Qwen responsible mainly for:

```text
visual facts
```

Use text models / embeddings for:

```text
continuity from scene cards
story planning
beat generation
matching
judge
```

Avoid:

```text
re-running Qwen over all scenes multiple times
```

Use visual fallback only for:

```text
low confidence
ambiguous continuity
critical selected beats
```

---

# 42. Important Separation of Responsibilities

Keep layers clean.

## Vision layer

Answers:

```text
What is visible?
What action occurs?
What object changed?
```

## Continuity layer

Answers:

```text
What changed from the previous scene?
```

## Story planner

Answers:

```text
Why does this matter in the whole video?
```

## Narration writer

Answers:

```text
How should we explain it naturally?
```

## Semantic lock

Answers:

```text
Does this narration actually match this footage?
```

## Judge / QA

Answers:

```text
Is the final script coherent, accurate, concise, and non-repetitive?
```

Do not combine all responsibilities into one mega-prompt.

---

# 43. Failure Philosophy

V2.1 should degrade gracefully.

Examples:

```text
semantic embeddings unavailable
→ use timeline + scene score fallback

continuity LLM fails
→ derive simple delta from scene card fields

judge fails
→ deterministic QA still runs

audio alignment QA unavailable
→ normal TTS duration scheduler continues
```

Never silently fabricate fallback narration.

Log degraded stages.

---

# 44. Suggested Metadata

Add:

```json
{
  "story_v2": {
    "enabled": true,

    "continuity": {
      "status": "success",
      "scenes": 124,
      "uncertain": 4
    },

    "beats": {
      "generated": 18,
      "deduplicated": 3,
      "final": 15
    },

    "semantic_match": {
      "high_confidence": 12,
      "warnings": 2,
      "repairs": 1,
      "failed": 0
    },

    "judge": {
      "attempts": 1,
      "verdict": "pass"
    },

    "qa": {
      "passed": true,
      "issues": 0
    }
  }
}
```

---

# 45. V2.1 Minimal Viable Version

The first V2.1 experiment should contain only three major changes:

```text
A. scene delta
B. rolling narration context
C. semantic scene lock
```

Do **not** add rhythm, emotion prosody, WhisperX, and complex BGM at the same time.

Minimal pipeline:

```text
V2 storyboard
→ scene delta
→ existing summary selector
→ existing narration plan
→ continuity writer
→ semantic verification
→ TTS
→ existing V2 scheduler
→ render
```

This gives a clean A/B test against V2.

---

# 46. V2.1-MVP File Plan

Create:

```text
dubbingstory/story/scene_continuity.py
dubbingstory/story/continuity_writer.py
dubbingstory/story/semantic_matcher.py
dubbingstory/story/script_qa.py
```

Modify:

```text
dubbingstory/cli.py
dubbingstory/story/script_writer.py
dubbingstory/story/scene_selector.py
configs/default.yaml
requirements.txt
```

Optional dependency:

```text
sentence-transformers
```

Consider keeping it under optional/advanced dependencies until tested on Kaggle.

---

# 47. V2.1-MVP Milestones

## Milestone 1 — Scene Delta

Success means:

```text
scene_delta.json exists
```

and manually inspecting 20 scenes shows:

```text
new/change/continuation information is mostly correct.
```

---

## Milestone 2 — Rolling Writer

Success means:

```text
the writer explicitly receives prior narration
and nearby scene context.
```

Manual output should contain fewer repetitive introductions.

---

## Milestone 3 — Semantic Lock

Success means:

```text
every narration beat has a match score.
```

Low scores should be visible in logs.

---

## Milestone 4 — Repair

Success means:

```text
intentionally mismatched narration
→ detected
→ corrected or rematched.
```

---

# 48. Example End-to-End V2.1 Story Flow

Raw visual facts:

```text
Scene 20:
mechanic measures metal part

Scene 21:
metal part mounted on lathe

Scene 22:
outer diameter is reduced

Scene 23:
measurement is checked

Scene 24:
machining continues

Scene 25:
part removed from lathe

Scene 26:
part installed
```

Bad narration:

```text
The mechanic measures the part.
The mechanic uses a lathe.
The metal is machined.
The mechanic checks it.
The machining continues.
The part is removed.
The part is installed.
```

V2.1 narration:

```text
Karena ukurannya harus tepat, komponen ini lebih dulu diukur sebelum masuk ke mesin bubut.

Diameter luarnya kemudian dikurangi sedikit demi sedikit, lalu diperiksa kembali untuk memastikan ukurannya tidak kelewat kecil.

Setelah koreksi terakhir selesai, komponen dilepas dari mesin dan akhirnya siap dicoba pada tempatnya.
```

The second version:

```text
groups repetitive scenes
explains causal relationships
avoids repeating obvious visuals
matches the visible progression
```

That is the target quality.

---

# 49. What Not to Copy

## From VIDEOFCK

Do not blindly copy:

```text
fixed 2.5 words/sec
all frames in one giant multimodal request
regex-only output parsing
hard-coded model providers
```

Adopt the **continuity concept**, not the entire implementation.

---

## From movie-narrator

Do not copy source code directly.

Use independent DubbingStory implementations of generic concepts such as:

```text
two-phase planning
time anchors
semantic matching
quality judge
reuse penalty
```

The architecture should remain native to DubbingStory.

---

# 50. License Note

Reference repository considerations:

```text
VIDEOFCK
→ MIT license

movie-narrator
→ AGPL-3.0-or-later in the supplied repository source
```

Therefore:

- concepts/ideas can be studied;
- VIDEOFCK MIT code has permissive reuse terms subject to its license;
- for movie-narrator, prefer **clean independent implementation of the concepts** rather than copying source into DubbingStory;
- keep third-party license notices for any code actually reused.

This document proposes architecture and concepts, not copied source.

---

# 51. Definition of Done for V2.1

V2.1 is done when all of the following are true:

- [ ] V2 behavior remains available through feature flags.
- [ ] Scene continuity metadata exists.
- [ ] Narration writer receives rolling context.
- [ ] Repeated scene information is reduced.
- [ ] Structured story beats exist.
- [ ] Duplicate beats can be removed.
- [ ] Beats have optional temporal anchors.
- [ ] Narration ↔ scene matching produces scores.
- [ ] Low-confidence matches trigger repair or explicit warnings.
- [ ] Scene reuse is penalized.
- [ ] Script QA runs before TTS.
- [ ] Judge feedback can drive targeted rewrite.
- [ ] Word budgets are enforced.
- [ ] Summary duration is kept near the requested target.
- [ ] TTS uses measured audio duration.
- [ ] Final output has no major narration/footage mismatch.
- [ ] Logs and JSON artifacts make failures diagnosable.
- [ ] A/B testing shows a measurable improvement over V2.

---

# 52. Recommended First Coding Session After V2

When V2 has been tested successfully, implement only:

```text
1. scene_continuity.py
2. continuity_writer.py
3. semantic_matcher.py
4. script_qa.py
```

Then run the same video through:

```text
V2
vs
V2.1-MVP
```

Evaluate:

```text
Does narration follow the scene?
Does it stop repeating itself?
Does each transition make sense?
Does the narration claim anything not shown?
Does the final recap feel like one story?
```

If V2.1-MVP clearly wins, continue with:

```text
beats
→ approx_ratio
→ deduplication
→ judge
→ diversity
→ rhythm/emotion
```

---

# 53. Final Design Principle

The final DubbingStory architecture should follow this rule:

> **Vision tells us what is there.  
> Continuity tells us what changed.  
> Story planning tells us why it matters.  
> Narration explains it naturally.  
> Semantic matching confirms that the footage supports the words.  
> QA prevents bad output from reaching TTS/render.**

That is the intended direction for DubbingStory V2.1.
