# DubbingStory v2 — Rencana Peningkatan Storytelling, Continuity, dan Voiceover

## 1. Tujuan Utama

Target DubbingStory berikutnya bukan sekadar:

> video → deteksi scene → deskripsikan gambar → bacakan dengan TTS

Tetapi menjadi pipeline yang benar-benar **memahami alur video**, mengetahui hubungan antar-scene, lalu menceritakannya kembali secara natural.

Target hasil akhirnya:

- narasi terasa seperti **orang yang memahami video**;
- cerita antar-scene **nyambung**;
- penonton memahami **apa yang terjadi, kenapa terjadi, dan akibatnya**;
- scene tidak terasa seperti potongan-potongan independen;
- narator tidak terus-menerus hanya mengatakan “lihat ini”, “luar biasa”, “kemudian”;
- voiceover lebih natural;
- timing narasi mengikuti video tanpa terlalu banyak dead air;
- summary tetap mempertahankan logika cerita walaupun banyak scene dibuang.

## 2. Masalah Utama yang Sudah Terlihat

Berdasarkan script dan audio hasil pipeline terakhir, masalahnya bukan hanya pada kualitas TTS.

Ada beberapa layer masalah.

### 2.1. Vision memahami scene secara terlalu lokal

Pipeline saat ini cenderung bekerja seperti:

```text
Scene 1
↓
Analisis visual
↓
Buat narasi

Scene 2
↓
Analisis visual
↓
Buat narasi

Scene 3
↓
Analisis visual
↓
Buat narasi
```

Masalahnya, model mengetahui **apa yang terlihat sekarang**, tetapi belum cukup mengetahui:

- apa yang terjadi sebelumnya;
- siapa karakter yang sama;
- apa tujuan karakter;
- apa benda yang sedang dibuat;
- kenapa sebuah proses dilakukan;
- apa akibat dari sebuah tindakan;
- apakah suatu scene merupakan setup, proses, payoff, atau bridge.

Akibatnya narasi menjadi deskripsi visual.

## 3. Problem Storytelling

Script sekarang terlalu banyak memakai gaya **reaction narration**.

Contoh pola:

> “Amazing!”

> “Luar biasa!”

> “Proses ini sungguh memukau, kan?”

> “Ini bagian paling dramatis!”

Pola tersebut memang membuat script terlihat energik, tetapi tidak memberikan banyak informasi baru.

Yang lebih penting dijelaskan adalah:

```text
Apa yang sedang dibuat?
↓
Kenapa ukurannya harus tepat?
↓
Komponen ini akan digunakan untuk apa?
↓
Apa proses selanjutnya?
```

## 4. Hook Juga Masih Generik

Pembukaan sekarang terasa seperti template viral generik.

Hook ideal seharusnya berasal dari **konflik atau tujuan sebenarnya dalam video**.

Misalnya jika yang sebenarnya dilakukan adalah membuat komponen custom:

```text
Motor ini punya satu masalah yang tidak bisa diselesaikan
hanya dengan membeli spare part baru.

Jadi mereka memilih cara yang jauh lebih sulit:
membuat komponennya sendiri dari sepotong logam.
```

Ini lebih menarik karena penonton langsung mengetahui:

- masalah;
- tujuan;
- tantangan;
- alasan untuk terus menonton.

## 5. Masalah Continuity

Masalah terbesar berikutnya adalah narasi belum memiliki **story memory**.

Saat sebuah objek atau karakter muncul lagi, model harus mengetahui bahwa itu merupakan hal yang sama dari scene sebelumnya.

Pipeline membutuhkan persistent memory seperti:

```json
{
  "characters": {},
  "objects": {},
  "locations": {},
  "relationships": {},
  "current_goal": "",
  "current_problem": "",
  "completed_actions": [],
  "unresolved_threads": []
}
```

Contoh:

```json
{
  "important_objects": {
    "custom_metal_part": {
      "introduced_scene": 12,
      "purpose": "komponen untuk motor",
      "current_state": "sedang dibubut",
      "last_seen_scene": 24
    }
  }
}
```

Dengan demikian scene berikutnya tidak kembali mengatakan:

> “sebuah logam”

tetapi bisa mengatakan:

> “Komponen yang tadi mulai dibentuk sekarang memasuki tahap pengeboran.”

Itulah continuity.

## 6. Transcript Harus Menjadi Sumber Utama Cerita

Visual saja tidak cukup.

Arsitektur baru sebaiknya menggunakan:

```text
Transcript / Subtitle
        +
Visual Understanding
        +
Scene Timeline
        ↓
Story Understanding
```

Prioritas informasi:

```text
Dialogue / transcript
        ↓
    PRIMARY

Visual scene
        ↓
   SUPPORTING
```

Jika subtitle tidak tersedia:

```text
Audio asli
↓
Whisper / faster-whisper
↓
Transcript
```

## 7. Arsitektur Pipeline Baru

Pipeline yang disarankan:

```text
VIDEO
  ↓
SEGMENTATION
  ↓
TRANSCRIPT / SUBTITLE
  ↓
VISION ANALYSIS
  ↓
SCENE CARDS
  ↓
GLOBAL STORY UNDERSTANDING
  ↓
STORY MEMORY
  ↓
STORY ARC
  ↓
SUMMARY SCENE SELECTION
  ↓
NARRATION PLAN
  ↓
CONTINUITY SCRIPT WRITER
  ↓
SCRIPT QA / REWRITE
  ↓
TTS
  ↓
VOICE TIMING / FITTING
  ↓
AUDIO MIXING
  ↓
FINAL VIDEO
```

## 8. Tahap Baru: `scene_cards.json`

Vision sebaiknya tidak langsung menulis narasi.

Vision hanya menghasilkan fakta.

Contoh:

```json
{
  "scene_id": "scene_024",
  "characters": ["mekanik"],
  "location": "workshop",
  "visible_action": "komponen logam sedang dibubut",
  "important_objects": ["mesin bubut", "komponen logam"],
  "dialogue_context": "",
  "character_goal": "membentuk komponen sesuai ukuran",
  "state_before": "material masih berbentuk kasar",
  "state_after": "diameter komponen mulai sesuai",
  "cause": "komponen harus memiliki ukuran presisi",
  "effect": "komponen bisa diproses ke tahap berikutnya",
  "emotional_tone": "focused",
  "story_importance": 0.74,
  "confidence": 0.89
}
```

Field paling penting:

```text
goal
cause
effect
state_before
state_after
story_importance
```

Bukan hanya:

```text
visible_objects
visual_description
```

## 9. Global Story Understanding

Setelah seluruh scene dianalisis, baru LLM membaca semuanya.

Input:

```text
scene_cards
+
transcript
+
timestamps
```

Output:

```text
story_plan.json
```

Contoh:

```json
{
  "premise": "Modifikasi sebuah motor membutuhkan pembuatan komponen custom.",
  "main_goal": "membuat komponen yang sesuai lalu memasangnya kembali",
  "main_problem": "komponen harus dibuat dengan presisi",
  "story_arc": {
    "setup": [],
    "problem": [],
    "process": [],
    "complication": [],
    "climax": [],
    "result": []
  },
  "important_objects": {},
  "characters": {},
  "unresolved_threads": []
}
```

Tahap inilah yang membuat model memahami:

> “Video ini sebenarnya sedang bercerita tentang apa?”

## 10. Story Memory

Selanjutnya dibuat:

```text
story_memory.json
```

Contoh:

```json
{
  "characters": {
    "mechanic_1": {
      "role": "mekanik utama",
      "current_goal": "menyelesaikan komponen"
    }
  },
  "objects": {
    "metal_part_1": {
      "purpose": "komponen motor",
      "state": "sedang dibentuk",
      "history": [
        "dipotong",
        "dibubut",
        "dibor"
      ]
    }
  },
  "unresolved_threads": [
    "apakah komponen akan pas ketika dipasang?"
  ]
}
```

Memory ini selalu dibawa ke tahap script generation.

## 11. Summary Selection Harus Berbasis Cerita

Current summary logic jangan hanya mempertimbangkan:

```text
scene menarik
+
scene penting
```

Tetapi:

```text
importance
+
plot relevance
+
character relevance
+
causal continuity
+
story arc coverage
+
bridge importance
-
redundancy
-
jump penalty
```

Scene yang secara visual biasa saja tetap bisa penting sebagai `bridge_scene`.

## 12. Scene Selection dan Vision Harus Dipisahkan dari Script

Sebaiknya prosesnya:

```text
Vision
↓
memahami fakta

Story Planner
↓
memahami cerita

Scene Selector
↓
memilih footage

Narrator
↓
baru menulis cerita
```

Jangan satu model dalam satu prompt diminta:

```text
lihat gambar
+
putuskan importance
+
pahami cerita
+
pilih scene
+
buat narration
```

Terlalu banyak tanggung jawab sekaligus.

## 13. `narration_plan.json`

Setelah scene summary terpilih, seluruh selected scenes diberikan sekali lagi ke story planner.

Contoh:

```json
[
  {
    "scene_id": "scene_001",
    "story_role": "hook",
    "intent": "jelaskan masalah utama",
    "must_explain": [
      "motor membutuhkan modifikasi"
    ]
  },
  {
    "scene_id": "scene_021",
    "story_role": "process",
    "intent": "jelaskan pembuatan komponen",
    "connect_from_previous": true
  },
  {
    "scene_id": "scene_045",
    "story_role": "progress",
    "intent": "jelaskan kenapa proses membutuhkan presisi"
  },
  {
    "scene_id": "scene_102",
    "story_role": "payoff",
    "intent": "tampilkan hasil akhir"
  }
]
```

Baru setelah itu script dibuat.

## 14. Continuity-Aware Script Writer

Script writer menerima:

```text
story_plan
+
story_memory
+
narration_plan
+
selected scene cards
+
transcript
```

Instruksi utamanya:

```text
Jangan mendeskripsikan frame secara mekanis.

Utamakan:
- apa yang sedang terjadi,
- mengapa hal itu dilakukan,
- apa yang berubah,
- apa akibatnya,
- hubungannya dengan scene sebelumnya.

Visual description hanya digunakan bila detail visual
penting untuk memahami cerita.
```

## 15. Hindari Kalimat Template

Kata atau pola seperti:

```text
Lihat ini!
Luar biasa!
Amazing!
Gila!
Kalian pasti gak nyangka!
Ini benar-benar...
Prosesnya makin seru!
```

jangan dilarang 100%, tetapi dibatasi.

Lebih baik mengganti pujian kosong dengan informasi yang menambah pemahaman cerita.

## 16. Style Narator

Target gaya narator:

```text
informative
+
casual
+
curious
+
storytelling
```

Bukan:

```text
hyperactive announcer
```

Komposisi awal yang layak diuji:

```text
70% storytelling / explanation
20% reaction / emotion
10% hook / rhetorical question
```

## 17. Contoh Perubahan

### Current

```text
Dan prosesnya makin seru!
Terus memotong!
Setiap milimeter diperhitungkan.

Ini bukan sembarang pengerjaan, lho!

Belum selesai sampai di situ!
```

### Target

```text
Setelah diameter luarnya mulai sesuai,
bagian tengah komponen masih harus dibor.

Posisinya tidak boleh bergeser,
karena sedikit saja meleset bisa membuat
komponen ini tidak pas saat dipasang ke motor.

Karena itu pengerjaan tetap dilakukan
di mesin yang sama sebelum masuk ke tahap berikutnya.
```

Perbedaannya:

```text
PROCESS
↓
REASON
↓
RISK
↓
NEXT STEP
```

## 18. Masalah Voiceover yang Ditemukan

Masalah audio bukan hanya karena Edge/Piper terdengar kurang natural.

Hasil WAV terakhir menunjukkan pola:

```text
TTS bicara
↓
silence
↓
TTS bicara
↓
silence panjang
↓
TTS berikutnya
```

Artinya pipeline sekarang terlalu mengikat setiap narration chunk ke keseluruhan panjang scene.

Jika:

```text
scene = 10 detik
narration = 3 detik
```

hasilnya menjadi:

```text
3 detik bicara
+
7 detik diam
```

Kalau terjadi puluhan kali, voiceover terasa sangat kosong.

## 19. TTS Timing Harus Diubah

Jangan menggunakan:

```text
scene_duration = narration_slot_duration
```

Sebaliknya gunakan konsep:

```text
Narration Segment
```

Satu narration segment boleh:

- menjelaskan 1 scene;
- menjelaskan 2–3 scene;
- mulai pada scene A;
- terus berbicara melewati scene B;
- berhenti sebelum informasi visual penting.

## 20. Narration Density

Sistem perlu mempunyai target WPM.

Untuk Bahasa Indonesia storytelling, mulai uji kisaran:

```text
145–175 WPM
```

Contoh config:

```yaml
narration:
  target_wpm: 160
  min_silence: 0.25
  max_silence: 2.5
```

Kalau ada gap panjang dan masih ada informasi penting yang belum dijelaskan, writer dapat menambah narration.

Sebaliknya kalau narration terlalu panjang:

```text
rewrite
↓
condense
```

bukan langsung mempercepat TTS secara ekstrem.

## 21. Duration-Aware Script

Sebelum TTS dibuat:

```text
text
↓
estimate speech duration
↓
compare dengan available timeline
```

Rumus awal:

```text
expected_duration =
word_count / target_wpm × 60
```

Jika script terlalu panjang, LLM diminta melakukan `condense`.
Jika terlalu pendek dan masih ada informasi penting, LLM dapat melakukan `expand explanation`.

## 22. Script QA Pass

Sebelum TTS, tambahkan tahap:

```text
Narration QA
```

Cek:

```text
continuity
repetition
hallucination
unnecessary hype
character consistency
object consistency
timeline fit
story completeness
```

Contoh output:

```json
{
  "continuity_score": 0.91,
  "repetition_score": 0.12,
  "story_coverage": 0.88,
  "estimated_duration": 164,
  "issues": []
}
```

Kalau score rendah:

```text
automatic rewrite
```

## 23. TTS Engine

Engine architecture tetap modular:

```text
--engine chirp
--engine azure
--engine elevenlabs
--engine edge
--engine piper
```

Rekomendasi prioritas testing:

```text
Google Chirp 3 HD
↓
Azure Neural / higher-quality voices
↓
ElevenLabs free-tier untuk testing
↓
Edge TTS
↓
Piper
```

Edge dan Piper tetap berguna sebagai:

```text
free fallback
offline/local fallback
development/debug engine
```

## 24. TTS Harus Mendukung Prosody

Interface engine sebaiknya tidak hanya:

```python
tts(text)
```

Tetapi:

```python
tts(
    text,
    voice=...,
    speed=...,
    pitch=...,
    style=...,
    emotion=...,
)
```

Walaupun engine tertentu tidak mendukung semua parameter.

Abstraction:

```text
TTSProvider
├── EdgeProvider
├── PiperProvider
├── ChirpProvider
├── AzureProvider
└── ElevenLabsProvider
```

## 25. Audio Post Processing

Pipeline:

```text
Raw TTS
↓
trim excessive silence
↓
normalize loudness
↓
optional compression
↓
optional EQ
↓
crossfade
↓
mix dengan original audio
```

## 26. Original Audio Jangan Dihilangkan Sepenuhnya

Untuk membuat video terasa hidup:

```text
Original audio
↓
duck volume saat narration
```

Dengan begitu:

- suara mesin;
- suara jalan;
- impact;
- ambience;
- suara alat;

masih terasa.

## 27. Struktur File Baru

Proposed output project:

```text
outputs/project/
│
├── scenes.json
├── transcript.json
├── scene_cards.json
├── story_plan.json
├── story_memory.json
├── summary_selection.json
├── narration_plan.json
├── narration_script.json
├── narration_qa.json
│
├── audio/
│   ├── narration/
│   └── mixed/
│
└── final.mp4
```

Keuntungan: setiap tahap bisa di-debug secara terpisah.

## 28. Model Responsibilities

### Qwen3-VL

```text
Visual understanding
Scene facts
Character/object recognition
Action/state changes
```

### Gemini / Text LLM

```text
Global story understanding
Story arc
Continuity
Scene selection reasoning
Narration planning
Script writing
Script QA
```

### Whisper

```text
Speech transcription
```

### TTS engine

```text
Voice synthesis
```

Jangan satu model bertanggung jawab terhadap semuanya.

## 29. Summary Mode Baru

```text
PASS A
cheap visual screening
↓
all scenes
↓
story relevance estimation

PASS B
deep analysis
↓
candidate scenes

GLOBAL STORY PLANNER
↓
understand full story

STORY-AWARE SELECTOR
↓
select important + bridge scenes

NARRATION PLANNER
↓
build coherent recap

SCRIPT
↓
TTS
```

Perbaikan summary mode yang sebelumnya sudah dilakukan tetap menjadi fondasi.

## 30. Full Mode

Full mode tetap memakai:

```text
Scene cards
↓
Story Planner
↓
Story Memory
↓
Narration Planner
```

Perbedaannya:

```text
summary = compress story

full = explain story in more detail
```

## 31. Tahapan Implementasi

### Phase 1 — Story Foundation

Implement:

```text
scene_cards.json
story_plan.json
story_memory.json
```

Target:

> Model memahami keseluruhan video sebelum menulis script.

### Phase 2 — Narration Pipeline

Implement:

```text
narration_plan.json
continuity writer
script QA
```

Target:

> Script mulai terasa seperti satu cerita utuh.

### Phase 3 — Summary Intelligence

Upgrade selector:

```text
story importance
+
causal continuity
+
bridge scene
+
story arc coverage
```

Target:

> Summary pendek tetapi tetap dapat dipahami.

### Phase 4 — Voice Timing

Implement:

```text
duration estimation
narration segments
silence control
duration-aware rewrite
```

Target:

> Menghilangkan masalah dead air.

### Phase 5 — Better TTS

Tambahkan:

```text
Chirp
Azure
ElevenLabs
```

sementara:

```text
Edge
Piper
```

tetap menjadi fallback.

### Phase 6 — Audio Mastering

Implement:

```text
normalization
ducking
crossfade
original sound preservation
```

## 32. Prioritas

Urutan berdasarkan impact:

```text
1. Story Planner
2. Scene Cards
3. Story Memory
4. Narration Planner
5. Continuity Writer
6. Summary Selector
7. Duration-Aware Narration
8. Better TTS
9. Audio Mixing
```

Hal penting:

> **Jangan mulai dari TTS.**

Karena suara premium yang membaca script buruk tetap akan menghasilkan video buruk.

## 33. Target Akhir

DubbingStory sekarang kira-kira:

```text
AI melihat video
↓
AI mengatakan apa yang terlihat
```

Target DubbingStory v2:

```text
AI melihat video
↓
AI memahami kejadian
↓
AI memahami hubungan kejadian
↓
AI memahami cerita
↓
AI memilih bagian yang diperlukan
↓
AI menyusun cara menceritakannya
↓
AI menulis narration yang kontinu
↓
AI menyesuaikan narration dengan durasi
↓
Natural TTS
↓
Audio mixing
↓
Final storytelling video
```

Secara sederhana:

> **Dari “scene describer” menjadi “video storyteller”.**
