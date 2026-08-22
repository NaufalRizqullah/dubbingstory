# DubbingStory V2.1 — Reviewed Implementation Plan

> Status: reviewed after Vision Architecture Fix
> References: DubbingStory V2, VIDEOFCK, movie-narrator
> Principle: Understand locally -> normalize facts -> plan globally -> match before writing -> write with memory -> verify after writing -> repair surgically.

## 1. Executive revision

The original V2.1 direction is correct, but semantic matching should happen twice:

```text
video
-> bounded scene windows
-> structured visual facts
-> scene cards + local delta
-> global story plan
-> structured beats
-> PRE-WRITE beat -> scene matching
-> narration written against fixed footage evidence
-> POST-WRITE narration <-> scene verification
-> targeted repair
-> deterministic QA
-> optional LLM judge
-> TTS / scheduler / render
```

Pre-write matching answers: **which footage can prove this beat?**

Post-write verification answers: **did the final wording stay supported by that footage?**

This is the biggest change from the original plan.

---

## 2. Prerequisite: Vision Architecture Fix must be stable

Before V2.1:

- Qwen/vLLM strict JSON Schema is active.
- `visible_objects` / `text_visible` are bounded.
- runaway `finish_reason=length` is rare.
- per-scene request tracing is available.
- vision cache includes schema/prompt versions.
- long scenes are split into <=15 second analysis windows.
- Kaggle T4 x2 deployment is validated.

Initial baseline:

```text
Qwen3-VL-2B-Instruct
VISION_MAX_TOKENS=640
MAX_KEYFRAMES=3
MAX_SCENE_DURATION=15
```

Do not mix model-size changes with the first V2.1 A/B test.

---

## 3. Atomic visual unit = scene window

A detected shot may be 60-120 seconds. It must not remain one factual or selection unit.

Example:

```text
scene_129
-> scene_129_w00
-> scene_129_w01
-> scene_129_w02
...
```

Each window keeps:

```json
{
  "scene_id": "scene_129_w03",
  "parent_scene_id": "scene_129",
  "window_index": 3,
  "start_time": 1691.38,
  "end_time": 1705.77,
  "duration": 14.39
}
```

All downstream systems must preserve both `scene_id` and `parent_scene_id`.

---

## 4. Adopt VIDEOFCK's shared-boundary continuity idea

VIDEOFCK uses overlapping keyframe groups. DubbingStory should adopt the idea, not the full implementation.

For 3-frame windows:

```text
window A: A0, A1, BORDER
window B: BORDER, B1, B2
```

This gives adjacent windows a shared visual anchor and should reduce false identity/state changes.

Suggested config:

```yaml
vision:
  max_scene_duration: 15.0
  max_keyframes: 3
  shared_boundary_anchor: true
```

Do not copy VIDEOFCK's giant all-segments multimodal request, regex response parsing, or permanent fixed words-per-second pacing.

---

## 5. Scene cards remain factual

`scene_cards.json` should contain facts, not narration style.

Minimum fields:

```text
scene_id
parent_scene_id
window_index
start/end/duration
subjects
objects
action
setting
state_before
state_after
goal
likely_context
subtitle/asr
confidence
vision model/schema version
```

---

## 6. Add scene signatures

Create:

```text
dubbingstory/story/scene_signature.py
```

A scene signature is a normalized text representation for retrieval:

```text
ACTION:
mechanic reduces the outer diameter

STATE CHANGE:
oversized component -> smaller diameter

OBJECTS:
mechanic, component, lathe, cutting tool

GOAL:
reach fitting dimension

CONTINUITY:
same machining process continues

AUDIO:
ukurannya harus pas
```

Do not embed ASR alone.

---

## 7. Local continuity / scene delta

Create:

```text
dubbingstory/story/scene_continuity.py
```

Context:

```text
previous window + current window + next window
```

Output:

```json
{
  "scene_id": "scene_129_w03",
  "continuation_from": "scene_129_w02",
  "relationship": "progression",
  "persistent_subjects": ["mechanic"],
  "persistent_objects": ["metal component", "lathe"],
  "new_objects": [],
  "new_action": "the outer diameter is reduced further",
  "state_change": {
    "before": "larger outer diameter",
    "after": "smaller outer diameter"
  },
  "new_information_only": [
    "the component has become visibly smaller"
  ],
  "confidence": 0.87
}
```

Relationship enum:

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

### Hybrid implementation

First calculate deterministic evidence:

```text
subject/object overlap
same parent scene
time gap
known state_before/state_after
normalized action similarity
```

Then use a text LLM only for:

```text
relationship classification
new-information compression
ambiguous entity reconciliation
```

If the LLM fails, keep the deterministic delta and lower confidence.

---

## 8. Add lightweight entity canonicalization

The same person may be called:

```text
man / worker / mechanic / technician
```

Without normalization, a delta engine may incorrectly classify the same person as a new subject.

MVP representation:

```json
{
  "entity_key": "person_primary_01",
  "display_label": "mechanic",
  "aliases": ["man", "worker"]
}
```

Keep this local/adjacent first. Full multi-object tracking is not required for V2.1 MVP.

---

## 9. Global story planner

Inputs:

```text
scene cards
+ scene deltas
+ subtitles/ASR
+ video metadata
```

Outputs:

```text
story_plan.json
story_memory.json
```

The planner determines:

```text
central goal
important stages
must-preserve facts
repetitive material to skip
open questions
causal progression
```

Qwen remains responsible for visual facts, not storytelling style.

---

## 10. Use two-phase beat -> narration generation

Adopt the generic two-phase pattern demonstrated by movie-narrator:

```text
Phase 1: structured beats
Phase 2: narration expansion
```

Create:

```text
dubbingstory/story/beat_planner.py
dubbingstory/story/beat_dedup.py
```

Beat example:

```json
{
  "beat_id": "beat_006",
  "text": "the component is progressively machined toward the required size",
  "story_role": "development",
  "act": 2,
  "approx_ratio": 0.38,
  "importance": 0.82,
  "required_facts": [
    "same component continues to be machined",
    "outer diameter decreases"
  ],
  "forbidden_claims": [
    "the component has already been installed"
  ],
  "target_duration_sec": 12.0,
  "max_words": 25
}
```

Beat count should be duration-aware:

```python
target_count = round(summary_target_duration / target_segment_duration)
```

Deduplicate beats before narration expansion using a cheap deterministic similarity first; embeddings can be added later.

---

## 11. PRE-WRITE Beat -> Scene Matcher

This becomes a formal subsystem.

Create:

```text
dubbingstory/story/semantic_matcher.py
```

Flow:

```text
beat required facts
-> temporal/act candidate restriction
-> scene-signature embeddings
-> top-K retrieval
-> composite rerank
-> chosen scene window(s)
```

Suggested:

```yaml
semantic_match:
  top_k: 5
  time_window_ratio: 0.18
```

If no candidate is good enough, expand the temporal window.

`approx_ratio` is a soft anchor, never stronger than factual relevance.

---

## 12. Top-K rerank + reuse penalty

Initial score:

```python
score = (
    semantic_similarity * 0.45
    + temporal_score * 0.15
    + continuity_score * 0.15
    + state_change_score * 0.10
    + importance_score * 0.05
    + visual_confidence * 0.10
    - reuse_penalty
    - redundancy_penalty
)
```

Keep weights configurable.

Recent-match memory should include:

```text
scene IDs
parent scene IDs
actions
objects/signatures
```

Do not perform hard/random diversity swaps. Let an unused candidate win only when its adjusted score is higher.

---

## 13. Chronological guard

Semantic similarity alone can select footage out of order.

Default rule:

```text
current beat should normally use footage at or after the previous beat
```

Allow a limited backtrack only for an explicit flashback/reference or a substantially better evidence match.

Every backtrack should be logged.

---

## 14. Multi-window beat support

One narration beat can map to several short windows:

```json
{
  "beat_id": "beat_006",
  "scene_ids": [
    "scene_021_w00",
    "scene_022_w00",
    "scene_023_w00"
  ]
}
```

This lets one thought cover a meaningful progression rather than forcing one sentence per scene.

---

## 15. Rolling continuity writer

Create:

```text
dubbingstory/story/continuity_writer.py
```

Input:

```text
story plan
story memory
current beat
FIXED assigned scene facts
previous narration
previous/current delta
next beat intent
remaining word budget
per-beat max words
style/language
```

The writer may improve prose, but may not expand beyond the evidence boundary.

Recommended structured output:

```json
{
  "beat_id": "beat_006",
  "scene_ids": ["scene_022_w00"],
  "text": "Diameter luarnya kemudian dikurangi sedikit demi sedikit agar ukurannya mendekati ukuran yang dibutuhkan.",
  "claims": [
    "outer diameter is reduced",
    "goal is dimensional fit"
  ]
}
```

The `claims` list helps the verifier.

---

## 16. Global word budget

Start with a configurable language/voice WPM estimate.

Do not permanently copy a fixed VIDEOFCK words-per-second constant.

Later:

```text
generated words
-> actual TTS duration
-> observed WPM
-> calibrate future run
```

Actual audio duration remains the source of truth for timeline placement.

---

## 17. POST-WRITE Semantic Lock

After narration is written:

```text
final narration
<-> assigned scene signatures
```

Check:

```text
semantic similarity
claim coverage
unsupported claims
timeline/continuity compatibility
```

Suggested output:

```text
match_report.json
```

Example:

```json
{
  "beat_id": "beat_006",
  "scene_ids": ["scene_022_w00"],
  "prewrite": {
    "semantic": 0.84,
    "final": 0.85
  },
  "postwrite": {
    "narration_scene_similarity": 0.82,
    "claim_coverage": 0.90,
    "unsupported_claims": []
  },
  "status": "accepted"
}
```

---

## 18. Typed repair strategy

### Footage correct, narration wrong

Rewrite narration while keeping scene assignments fixed.

### Beat/narration correct, footage wrong

Rematch from the existing top-K candidates, then expand the candidate pool if required.

### Both ambiguous

Use:

```text
story plan
+ neighboring beats
+ scene deltas
```

and choose the lower-cost correction.

### Evidence insufficient

Do not fabricate. Use a safer generic line or drop the beat.

---

## 19. Deterministic QA before LLM judge

Create:

```text
dubbingstory/story/script_qa.py
```

Checks:

```text
beat count
empty text
word budget
total word budget
duplicate narration
repetitive openings
valid scene/window IDs
chronological order
duration budget
reuse
match scores
unsupported claims
post-write verifier status
TTS overflow
```

This must run even if the judge is unavailable.

---

## 20. LLM judge comes after deterministic QA

Create later:

```text
dubbingstory/story/script_judge.py
```

Judge dimensions:

```text
continuity
scene fidelity
causal clarity
information density
repetition
timing fit
ending quality
style compliance
```

### Important rule learned from movie-narrator

Do not trust the LLM's own `verdict` field.

Normalize the numeric scores and derive pass/retry deterministically.

Then feed concrete issues back into only the affected beats when possible.

---

## 21. Hardware plan

For Qwen 2B/4B that fits on one T4:

```text
GPU0 -> Qwen replica A
GPU1 -> Qwen replica B

DP=2
TP=1
vision_concurrency=2
```

For a model requiring both T4s:

```text
GPU0 + GPU1 -> one model

DP=1
TP=2
vision_concurrency=1 initially
```

Do not assume TP=2 is automatically the best use of two GPUs.

### Embedding model

Default the multilingual sentence-transformer to CPU:

```yaml
semantic_match:
  embedding_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  embedding_device: cpu
  embedding_cache: true
```

Reason: avoid stealing VRAM from the two Qwen replicas. Scene/beat counts are small enough for a lightweight CPU embedding model.

---

## 22. Cache correctness

Cache keys must include the data/model version that actually affects results.

Vision cache:

```text
vision model
schema version
prompt version
window timestamps
keyframe fingerprints
image settings
```

Embedding cache:

```text
embedding model
signature text
signature version
```

Story cache:

```text
text model
planner prompt version
scene-card hashes
scene-delta hashes
```

---

## 23. Revised implementation order

### Phase 0 — validate Vision Architecture Fix

```text
strict schema
bounded window
trace logging
cache versioning
T4 x2 deployment
```

### Phase 1 — continuity foundation

```text
parent_scene_id/window_id contract
shared boundary anchor
scene_signature.py
scene_continuity.py
entity canonicalization lite
```

### Phase 2 — beat planning

```text
beat_planner.py
beat_dedup.py
```

### Phase 3 — pre-write matching

```text
semantic_matcher.py
top-K retrieval
temporal restriction
reuse penalty
chronological guard
```

### Phase 4 — writer

```text
continuity_writer.py
global word budget
claims output
```

### Phase 5 — post-write verification/repair

```text
semantic verifier
typed repair
match_report.json
```

### Phase 6 — deterministic QA

```text
script_qa.py
```

### Phase 7 — LLM judge

```text
script_judge.py
deterministic verdict
targeted feedback retry
```

### Phase 8 — optional layers

```text
rhythm/emotion
alignment QA
advanced BGM/TTS behavior
4B/8B model experiments
```

---

## 24. Revised MVP

The V2.1 MVP should contain:

```text
A. bounded scene-window continuity / delta
B. scene signatures
C. pre-write beat -> scene top-K matching
D. rolling narration writer
E. post-write semantic verification + deterministic QA
```

The LLM judge is not required for the first MVP.

---

## 25. Suggested files

Create:

```text
dubbingstory/story/scene_continuity.py
dubbingstory/story/scene_signature.py
dubbingstory/story/beat_planner.py
dubbingstory/story/beat_dedup.py
dubbingstory/story/semantic_matcher.py
dubbingstory/story/continuity_writer.py
dubbingstory/story/script_qa.py
```

Later:

```text
dubbingstory/story/script_judge.py
```

Modify:

```text
dubbingstory/story/scene_cards.py
dubbingstory/story/story_planner.py
dubbingstory/story/narration_planner.py
dubbingstory/story/scene_selector.py
dubbingstory/story/script_writer.py
dubbingstory/segment/keyframes.py
dubbingstory/cli.py
dubbingstory/config.py
configs/default.yaml
requirements.txt
```

---

## 26. Runtime artifacts

```text
outputs/<project>/
├── storyboard.json
├── scene_cards.json
├── scene_delta.json
├── story_plan.json
├── story_memory.json
├── beats.json
├── narration_plan.json
├── narration_script.json
├── match_report.json
├── script_qa.json
├── script_judge.json       # later
├── summary_manifest.json
├── pipeline.log
└── final.mp4
```

---

## 27. A/B evaluation

Compare:

```text
V2
vs
V2 + Vision Architecture Fix
vs
V2.1 MVP
```

Track:

```text
vision length failures
vision rescue count
selected-window duplicates
parent-scene reuse
semantic match mean/p10
unsupported narration claims
narration repetition
summary-duration error
TTS overflow
runtime
manual continuity
manual scene fidelity
manual story coherence
```

Use multiple video types, especially a visually repetitive process/repair video.

---

## 28. Source-code reuse policy

### VIDEOFCK

The supplied repository is MIT licensed.

Small code reuse is possible subject to its notice/license requirements, but its implementation is not a clean fit for DubbingStory's new architecture. Prefer native implementation of the useful ideas.

Useful concepts:

```text
overlapping visual groups / boundary anchor
continuity from previous context
timestamp-aware segment budget
```

Avoid:

```text
giant multimodal request
regex output contract
fixed provider design
fixed global words/sec
```

### movie-narrator

The supplied/current repository is AGPL-3.0-or-later.

A public GitHub repository is not automatically permissive for direct source transplantation.

Recommended:

```text
study its behavior
study generic algorithms
study tests and failure cases
reimplement independently against DubbingStory data structures
```

Do not copy its source directly unless DubbingStory intentionally adopts compatible AGPL obligations and the licensing consequences are understood.

This is an engineering licensing precaution, not legal advice.

---

## 29. Final architecture

```text
VIDEO
  |
  v
Scene Detection
  |
  v
Bounded Scene Windows <=15s
  |
  +-- shared boundary anchor
  v
Qwen Structured Visual Facts
  |
  v
Scene Cards
  |
  v
Scene Signature + Local Delta
  |
  v
Global Story Plan + Memory
  |
  v
Structured Beats
  |
  v
Beat Dedup
  |
  v
PRE-WRITE Top-K Scene Matching
  |
  +-- temporal/act restriction
  +-- embeddings
  +-- continuity/state score
  +-- reuse penalty
  +-- chronological guard
  v
Fixed Narration Plan / Footage Evidence
  |
  v
Rolling Continuity Writer
  |
  v
POST-WRITE Semantic Verification
  |
  +-- accept
  +-- rewrite narration
  +-- rematch scene
  +-- safe drop/fallback
  v
Deterministic Script QA
  |
  v
LLM Judge + Targeted Retry (later)
  |
  v
TTS -> actual duration -> scheduler -> optional alignment QA -> render
```

---

## 30. Final rule

> **Qwen tells us what is visible.**  
> **Scene delta tells us what changed.**  
> **Story planning tells us why it matters.**  
> **Pre-write matching chooses footage that can prove the beat.**  
> **The writer explains only what that footage supports.**  
> **Post-write verification checks that the words still match the images.**  
> **Deterministic QA prevents a confident LLM from passing its own mistake.**
