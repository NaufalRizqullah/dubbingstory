# DubbingStory — Diagnosis Error & Rencana Optimisasi Kaggle/Colab

## Ringkasan eksekutif

Ada **tiga masalah terpisah** di pipeline saat ini:

1. **Output Qwen ada, tetapi tidak selalu menjadi JSON yang lengkap/valid.**
   Pesan `Could not extract JSON from response: ```json ...` bukan berarti code fence ` ```json ` adalah penyebab utamanya. Parser yang ada sudah mencoba menghapus code fence dan mencari `{...}`. Jika tetap gagal, biasanya body JSON terpotong, malformed, atau memiliki konten tambahan yang membuat `json.loads()` gagal.

2. **Pipeline vision masih serial.**
   `vision.batch_size: 4` ada di konfigurasi tetapi tidak dipakai oleh loop analisis. Satu scene harus selesai sebelum scene berikutnya dikirim. Ini membuang kemampuan continuous batching vLLM.

3. **Bottleneck terbesar di run lama justru preprocessing video.**
   Pipeline membuat dan **re-encode seluruh 157 scene menjadi MP4**, hanya untuk kemudian membaca 3 keyframe per scene. Pada log lama, tahap split scene memakan sekitar 100 menit. Untuk mode `keyframes`, scene MP4 tidak perlu dibuat sama sekali.

Selain itu, error fatal terakhir pada log upload terjadi di **FFmpeg concat**, bukan parser JSON. Ada bug path pada `concat_list.txt`, dan stderr dipotong pada 500 karakter awal sehingga pesan error sebenarnya tersembunyi.

---

# 1. Mengapa `Could not extract JSON` terjadi?

Kode sekarang melakukan:

```python
response = self.client.chat.completions.create(
    model=self.model,
    messages=messages,
    temperature=self.temperature,
    max_tokens=self.max_tokens,
)

text = response.choices[0].message.content
return _extract_json_from_text(text)
```

Prompt juga meminta model:

```text
Return ONLY a valid JSON object (no markdown, no explanation)
```

Namun model masih dapat menjawab:

````text
```json
{
  "scene_id": "scene_074",
  ...
```
````

Code fence sendiri **bukan masalah besar**, karena parser saat ini sudah mencoba:

- `json.loads(text)`
- mencari fenced code block
- mencari blok `{ ... }` / `[ ... ]`

Jadi jika semuanya masih gagal, kemungkinan nyata adalah:

- JSON **terpotong sebelum `}` terakhir**
- generation berhenti karena `finish_reason="length"`
- ada string yang tidak selesai
- invalid escape
- trailing comma
- model menambahkan dua blok JSON / penjelasan tambahan
- respons berhenti saat server/model sedang menghasilkan field yang panjang

Masalah debugging saat ini: error hanya mencetak **120/200 karakter awal** output. Kita tidak melihat bagian akhir respons, padahal bagian akhir itulah yang paling penting untuk mengetahui apakah ada `}` penutup.

## Yang harus ditambahkan segera

```python
choice = response.choices[0]
text = choice.message.content or ""
finish_reason = choice.finish_reason

print(
    f"      [Vision] finish_reason={finish_reason} "
    f"chars={len(text)}"
)

try:
    return _extract_json_from_text(text)
except json.JSONDecodeError:
    print("      [Vision] RAW HEAD:", repr(text[:400]))
    print("      [Vision] RAW TAIL:", repr(text[-800:]))
    raise
```

Kalau `finish_reason == "length"`, jangan retry enam kali dengan request identik. Respons memang terpotong.

---

# 2. Solusi terbaik: gunakan Structured Output vLLM

Jangan mengandalkan prompt `"Return ONLY JSON"` saja.

vLLM mendukung OpenAI-compatible `response_format` dengan JSON Schema. Dengan ini decoder dibatasi untuk menghasilkan bentuk JSON yang sesuai schema.

Contoh:

```python
from pydantic import BaseModel, Field


class SceneAnalysis(BaseModel):
    scene_id: str
    time_range: str
    visible_objects: list[str]
    people: str
    action: str
    changes: str
    environment: str
    likely_context: str
    text_visible: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


SCENE_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "scene_analysis",
        "schema": SceneAnalysis.model_json_schema(),
    },
}
```

Lalu:

```python
response = self.client.chat.completions.create(
    model=self.model,
    messages=messages,
    temperature=0.0,
    max_tokens=512,
    response_format=SCENE_JSON_SCHEMA,
)

choice = response.choices[0]
text = choice.message.content or ""

if choice.finish_reason == "length":
    raise RuntimeError(
        f"Structured output truncated: {len(text)} chars"
    )

analysis = SceneAnalysis.model_validate_json(text)
return analysis.model_dump()
```

### Kenapa ini lebih baik?

- tidak perlu berharap Qwen patuh pada formatting prompt
- field dan tipe data divalidasi
- kemungkinan retry akibat format turun drastis
- output bisa dibuat lebih pendek
- `max_tokens` tidak perlu 2048 untuk satu scene
- `temperature=0.0` cocok untuk ekstraksi deterministik

**Target awal yang disarankan:** `max_tokens=512`.
Jika output kadang benar-benar membutuhkan lebih banyak, naikkan ke 768 — bukan langsung 2048.

---

# 3. Jangan "passing semua raw output" pada arsitektur saat ini

Untuk repo saat ini, jawabannya: **jangan langsung melewati raw string sebagai pengganti parsing**.

Downstream code melakukan operasi seperti:

```python
analysis["start_time"] = ...
analysis["end_time"] = ...
analysis.get("confidence", 0)
analysis.get("visible_objects", [])
analysis.get("action", "")
```

Artinya kontrak internal pipeline adalah `dict`, bukan string.

Kalau raw output langsung dipassing:

```python
analysis = response_text
```

stage storyboard/scene selector akan pecah atau kehilangan informasi.

### Fallback yang aman

Raw output boleh **disimpan untuk debugging**, tetapi tetap kembalikan object:

```python
fallback = {
    "scene_id": scene_id,
    "time_range": time_range,
    "visible_objects": [],
    "people": "",
    "action": "",
    "changes": "",
    "environment": "",
    "likely_context": "",
    "text_visible": [],
    "confidence": 0.0,
    "parse_status": "failed",
    "raw_model_output": raw_text,
}
```

Jadi data mentah tidak hilang, tetapi pipeline tetap memiliki tipe yang konsisten.

---

# 4. Parser fallback yang lebih robust

Structured output harus menjadi jalur utama. Untuk endpoint lama yang belum mendukung schema, gunakan parser fallback dengan `JSONDecoder.raw_decode()` daripada regex greedy `{.*}`.

```python
import json
import re


def extract_first_json(text: str):
    text = (text or "").strip()

    # Hilangkan opening/closing markdown fence jika ada.
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)

    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise json.JSONDecodeError("No JSON start found", text, 0)

    start = min(starts)
    decoder = json.JSONDecoder()
    obj, _end = decoder.raw_decode(text[start:])
    return obj
```

Keunggulan `raw_decode()`:

- dapat mengambil JSON pertama tanpa tersandung trailing explanation
- tidak membutuhkan closing markdown fence
- tidak greedy sampai object terakhir
- tetap gagal dengan benar jika JSON memang terpotong

Jangan otomatis "memperbaiki" JSON yang terpotong di tengah kalimat lalu menganggapnya valid. Itu berisiko mengubah hasil visual menjadi data yang tidak pernah dihasilkan model.

---

# 5. Ubah strategi retry

Saat ini parse error dianggap retryable dan bisa menghabiskan sampai 6 generation penuh.

Untuk local vLLM, bedakan jenis error:

```python
NETWORK_RETRIES = 3
FORMAT_RETRIES = 1
```

Kebijakan:

| Error | Aksi |
|---|---|
| HTTP 408 / 429 / 5xx | retry 2–3x dengan backoff |
| connection reset / timeout | retry 2–3x |
| JSON malformed | repair/parser fallback, lalu maksimal 1 regeneration |
| `finish_reason=length` | jangan ulang request identik; kecilkan prompt/output schema |
| HTTP 400 context length | **jangan retry**; kecilkan input atau context |
| valid structured JSON | langsung cache |

Retry format 6x adalah sangat mahal jika satu generation membutuhkan puluhan sampai ratusan detik.

---

# 6. Perbaiki masalah context length

Pada run lama ada input sekitar 6622 token sementara server diset 4096.

Masalahnya bukan kapasitas native Qwen3-VL, melainkan **serving limit yang Anda berikan ke vLLM** serta jumlah visual token dari keyframe.

Solusi utamanya bukan hanya terus menaikkan `--max-model-len`, karena context besar menggunakan KV-cache dan dapat menurunkan concurrency.

Prioritas:

1. resize keyframe
2. kurangi jumlah keyframe adaptif
3. pendekkan prompt
4. structured output
5. baru sesuaikan `max-model-len`

### Target keyframe

Jangan kirim JPEG 1080p apa adanya untuk semua scene.

Rekomendasi awal:

- normal scene: 3 frame, longest edge 640–768 px
- low-motion scene: 2 frame
- text/OCR-heavy scene: 3 frame, 960 px hanya jika perlu
- JPEG quality: 75–85

Contoh resize:

```python
def resize_for_vision(frame, max_edge=640):
    h, w = frame.shape[:2]
    scale = min(1.0, max_edge / max(h, w))
    if scale >= 1.0:
        return frame

    nw = max(32, int(round((w * scale) / 32) * 32))
    nh = max(32, int(round((h * scale) / 32) * 32))

    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
```

Kelipatan 32 cocok dengan spatial compression Qwen3-VL.

---

# 7. Bottleneck terbesar: hapus physical scene splitting untuk mode keyframes

Arsitektur sekarang:

```text
source.mp4
  ↓
detect scene
  ↓
157 × ffmpeg re-encode scene_xxx.mp4
  ↓
buka masing-masing scene MP4
  ↓
ambil 3 JPEG
  ↓
Qwen
```

Ini sangat boros.

Arsitektur yang disarankan:

```text
source.mp4
  ↓
detect scene boundaries saja
  ↓
scene start/end timestamps
  ↓
ambil keyframe langsung dari source.mp4
  ↓
Qwen
```

Scene MP4 hanya dibutuhkan kalau:

- benar-benar memakai mode `video_upload`
- debugging/manual inspection
- scene tersebut sudah terpilih untuk final summary

## API baru

```python
def detect_only(
    video_path: str,
    detector: str = "adaptive",
    threshold: float = 3.0,
    min_duration: float = 5.0,
) -> list[dict]:
    # Sama dengan detect_scenes(), tanpa split_video().
    ...
```

Lalu keyframe:

```python
def extract_keyframes_from_source(
    video_path: str,
    scenes: list[dict],
    output_dir: str,
    max_per_scene: int = 3,
    max_edge: int = 640,
):
    cap = cv2.VideoCapture(video_path)

    result = {}

    for scene in scenes:
        sid = scene["scene_id"]
        start = scene["start_time"]
        end = scene["end_time"]

        # Hindari tepat di boundary.
        margin = min(0.25, max(0.0, (end - start) * 0.05))
        a = start + margin
        b = max(a, end - margin)

        count = max_per_scene
        timestamps = np.linspace(a, b, count)

        paths = []

        for i, ts in enumerate(timestamps):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue

            frame = resize_for_vision(frame, max_edge=max_edge)

            path = os.path.join(
                output_dir, sid, f"{sid}_kf{i:02d}.jpg"
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            paths.append(path)

        result[sid] = paths

    cap.release()
    return result
```

Ini menghilangkan 157 transcode H.264 yang tidak perlu.

---

# 8. Gunakan parallel/concurrent requests ke vLLM

Saat ini:

```python
for scene in scenes:
    analysis = analyzer.analyze_scene_from_frames(...)
```

Ini serial.

vLLM baru memberikan keuntungan batching jika **beberapa request outstanding secara bersamaan**.

Konfigurasi `vision.batch_size: 4` saat ini tidak melakukan apa-apa sampai loop dianalisis secara concurrent.

Contoh pendekatan paling sederhana:

```python
from concurrent.futures import ThreadPoolExecutor


def analyze_one_scene(scene):
    # existing per-scene logic
    ...
    return analysis


workers = getattr(cfg, "vision_concurrency", 2)

with ThreadPoolExecutor(max_workers=workers) as pool:
    # executor.map menjaga urutan hasil sama dengan urutan scenes.
    scene_analyses = list(pool.map(analyze_one_scene, scenes))
```

Rekomendasi awal:

- P100 16 GB: `workers=1`, tes `2`
- Colab T4 16 GB: `workers=2`, tes `3`
- Kaggle T4×2 dengan dua server independen: total `workers=4`, misalnya 2 per GPU

Pantau:

```bash
nvidia-smi dmon
```

dan vLLM logs.

Naikkan concurrency sampai GPU utilization bagus tanpa preemption/OOM.

---

# 9. Kaggle T4×2: jangan otomatis Tensor Parallel 2 untuk model 2B

Qwen3-VL-2B sudah cukup kecil untuk satu T4 16 GB pada konfigurasi yang wajar.

Untuk workload Anda — banyak scene independen — lebih masuk akal memakai:

```text
GPU 0 -> vLLM server :8000, TP=1
GPU 1 -> vLLM server :8001, TP=1
```

dan dispatch scene round-robin.

Contoh launch:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 6144 \
  --gpu-memory-utilization 0.85 \
  --dtype half

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --port 8001 \
  --tensor-parallel-size 1 \
  --max-model-len 6144 \
  --gpu-memory-utilization 0.85 \
  --dtype half
```

Kemudian:

```python
VISION_ENDPOINTS = [
    "http://127.0.0.1:8000/v1",
    "http://127.0.0.1:8001/v1",
]
```

Keuntungan:

- tidak ada sinkronisasi TP antar-T4 untuk setiap token
- dua scene bisa dikerjakan benar-benar paralel
- cocok dengan workload banyak request independen

Tetap benchmark karena hasil akhir bergantung ukuran frame, context, dan versi vLLM.

---

# 10. `--enforce-eager`: jangan selalu dipaksa

Notebook saat ini memasang:

```bash
--enforce-eager
```

Flag ini berguna untuk compatibility/debugging dan startup tanpa CUDA graph, tetapi menurunkan steady-state inference performance.

Untuk pipeline 100+ scene, **steady-state throughput lebih penting** daripada sedikit startup tambahan.

Profil yang disarankan:

### Mode performance

```bash
# hapus --enforce-eager
```

### Mode compatibility/fallback

```bash
--enforce-eager
```

Tes kedua mode pada GPU aktual. Jika tanpa eager mengalami OOM/CUDA graph error, gunakan eager. Jangan menganggap eager wajib untuk semua T4.

---

# 11. Profile GPU yang disarankan

## Colab T4 16 GB

```python
TENSOR_PARALLEL_SIZE = 1
MAX_MODEL_LEN = 6144        # coba 8192 hanya setelah frame dikecilkan
GPU_MEM_UTIL = 0.82         # naikkan 0.85 bila stabil
VISION_CONCURRENCY = 2
VISION_MAX_TOKENS = 512
VISION_TEMPERATURE = 0.0
MAX_KEYFRAMES = 3
VISION_MAX_EDGE = 640
```

Mulai tanpa `--enforce-eager`; tambahkan kembali jika ada compatibility/OOM issue.

## Kaggle T4 ×2

Rekomendasi:

```text
2 independent server
TP=1 per server
6144–8192 context per server
2 concurrent request per server
```

Jangan gunakan TP=2 hanya karena dua GPU tersedia.

## P100 16 GB

```python
TENSOR_PARALLEL_SIZE = 1
MAX_MODEL_LEN = 6144
GPU_MEM_UTIL = 0.80
VISION_CONCURRENCY = 1   # benchmark 2
VISION_MAX_TOKENS = 512
VISION_TEMPERATURE = 0.0
MAX_KEYFRAMES = 2 or 3
VISION_MAX_EDGE = 640
```

Gunakan FP16/`--dtype half`. Prioritaskan frame kecil dan output pendek.

---

# 12. Summary mode seharusnya menggunakan two-pass vision

Saat ini Anda menganalisis semua 157 scene secara detail, baru memilih sekitar 12 scene.

Untuk **summary/highlight**, ini terbalik.

## Pass A — cheap screening seluruh scene

2 low-resolution frames per scene.

Output kecil:

```json
{
  "scene_id": "scene_074",
  "action": "worker removes truck wheel hub",
  "visual_change": 0.8,
  "salience": 0.9,
  "confidence": 0.9
}
```

Target 80–150 output token.

## Seleksi

Pilih misalnya:

- top 20–30 scene berdasarkan salience/visual change
- selalu pertahankan beberapa scene awal
- pertahankan kandidat result/conclusion
- jaga distribusi temporal agar tidak semua kandidat berasal dari bagian video yang sama

## Pass B — deep analysis kandidat saja

Baru kirim 3 keyframe 640–960 px dan schema lengkap.

Hasil akhir:

- jauh lebih sedikit vision compute
- summary selection lebih relevan
- deep description hanya digunakan untuk scene yang benar-benar masuk cerita

---

# 13. Temporal-flow stage juga perlu diubah

Kode saat ini memasukkan seluruh per-scene JSON ke satu prompt temporal.

Dengan 100–200 scene, prompt ini mudah menjadi sangat besar.

Untuk summary mode, pilih salah satu:

### Opsi A — skip global temporal LLM sebelum selection

Gunakan score cheap pass untuk memilih kandidat, lalu biarkan script writer merangkai 12–20 scene terpilih.

Ini paling efisien.

### Opsi B — hierarchical

```text
157 scene
  ↓
chunk 20 scene
  ↓
8 ringkasan chunk
  ↓
1 global flow
```

Jangan mengirim seluruh object lengkap 157 scene ke satu request.

---

# 14. Perbaikan FFmpeg concat yang wajib

Kode sekarang membuat:

```python
clips_dir = "outputs/my_dubbing_project/_summary_clips"
concat_list_path = (
    "outputs/my_dubbing_project/_summary_clips/concat_list.txt"
)
```

tetapi menulis:

```text
file 'outputs/my_dubbing_project/_summary_clips/scene_001.mp4'
```

Karena entry relatif pada concat list diselesaikan relatif terhadap lokasi `concat_list.txt`, ini dapat menjadi path seperti:

```text
outputs/my_dubbing_project/_summary_clips/
outputs/my_dubbing_project/_summary_clips/scene_001.mp4
```

## Fix

Gunakan absolute path:

```python
with open(concat_list_path, "w", encoding="utf-8") as f:
    for clip_path in clip_paths:
        safe_path = (
            os.path.abspath(clip_path)
            .replace("\\", "/")
            .replace("'", "'\\''")
        )
        f.write(f"file '{safe_path}'\n")
```

atau karena list dan clip berada di folder sama:

```python
f.write(f"file '{os.path.basename(clip_path)}'\n")
```

### Jangan sembunyikan stderr

Ganti:

```python
stderr = e.stderr.decode(
    "utf-8", errors="replace"
)[:500]
```

menjadi:

```python
stderr = (
    e.stderr.decode("utf-8", errors="replace")
    if e.stderr else ""
)

raise RuntimeError(
    "FFmpeg concat failed:\n"
    + stderr[-5000:]
) from e
```

Lebih bagus:

```python
concat_cmd = [
    "ffmpeg", "-y",
    "-hide_banner",
    "-loglevel", "error",
    "-f", "concat",
    "-safe", "0",
    "-i", concat_list_path,
    ...
]
```

---

# 15. Jangan encode ulang pada tahap concat

Setiap `_extract_clip()` saat ini sudah menjadi H.264 + AAC dengan parameter yang sama.

Maka concat seharusnya dicoba dengan stream copy:

```python
concat_cmd = [
    "ffmpeg", "-y",
    "-hide_banner",
    "-loglevel", "error",
    "-f", "concat",
    "-safe", "0",
    "-i", concat_list_path,
    "-c", "copy",
    "-movflags", "+faststart",
    output_path,
]
```

Ini menghindari encode H.264 kedua kalinya saat concat.

Jika ada mismatch stream parameter, fallback ke re-encode dan tampilkan stderr penuh.

Lebih optimal lagi untuk tahap berikutnya adalah satu FFmpeg `filter_complex` yang melakukan trim + concat + resize + audio mixing dalam satu final encode, tetapi implementasinya lebih kompleks.

---

# 16. Optimisasi scene detection

Scene detection sendiri masih bisa dipercepat, tetapi ini prioritas setelah physical splitting dihapus.

PySceneDetect mendukung:

- automatic/manual downscale
- optional `frame_skip`

Untuk summary 30 fps, benchmark:

```python
scene_manager.detect_scenes(
    video,
    frame_skip=1,  # proses ~1 dari 2 frame
)
```

Gunakan hanya bila hasil scene boundary masih cukup baik karena frame skipping mengurangi akurasi.

Jangan mengorbankan accuracy dulu sebelum bottleneck 100-menit physical split dihilangkan.

---

# 17. Jangan biarkan vLLM idle selama preprocessing CPU

Pada run lama, vLLM sudah di-start sebelum ingest/segmentation panjang.

Strategi lebih baik:

```text
download / ASR / scene detection / keyframe extraction
        ↓
start vLLM
        ↓
vision
```

atau start vLLM paralel dengan bagian akhir preprocessing.

Setelah physical scene splitting dihapus, keuntungan overlap ini lebih kecil tetapi tetap berguna untuk notebook cold-start.

---

# 18. Cache harus version-aware

Cache vision sudah ide bagus, tetapi nama file hanya berdasarkan `scene_id`.

Jika model/prompt/keyframe berubah, cache lama bisa dipakai padahal tidak cocok.

Gunakan cache key:

```python
import hashlib
import json


payload = {
    "scene_id": scene_id,
    "model": self.model,
    "prompt_version": "scene-v3",
    "keyframes": [
        {
            "path": p,
            "mtime": os.path.getmtime(p),
            "size": os.path.getsize(p),
        }
        for p in kf_paths
    ],
}

cache_key = hashlib.sha1(
    json.dumps(payload, sort_keys=True).encode()
).hexdigest()[:12]

cache_path = os.path.join(
    cache_dir,
    f"{scene_id}_{cache_key}.json"
)
```

Dengan ini Anda dapat melanjutkan run tanpa takut cache stale.

---

# 19. Observability yang perlu ditambahkan

Untuk setiap scene log:

```text
scene_074
input_images=3
image_dims=640x352
prompt_chars=1200
queue_wait_ms=...
generation_ms=...
finish_reason=stop
output_tokens=282
parse_ms=1
cache=miss
```

Untuk setiap run:

```text
download_s
scene_detect_s
keyframe_s
vision_s
temporal_s
script_s
tts_s
cut_s
render_s
```

Tanpa phase timing, optimisasi mudah dilakukan di bagian yang sebenarnya bukan bottleneck.

Tambahkan juga PID/run-id ke log. Pada log lama banyak baris muncul dua kali; PID akan menunjukkan apakah itu hanya duplicate logging atau ada dua pipeline process yang benar-benar berjalan.

---

# 20. Urutan implementasi yang disarankan

## P0 — correctness

1. Fix path `concat_list.txt`
2. Tampilkan tail/full FFmpeg stderr
3. Structured JSON schema untuk Qwen
4. Log `finish_reason` + RAW tail saat parser gagal
5. HTTP 400/context error jangan retry

## P1 — performance terbesar

6. Hapus physical split 157 scene pada `analysis_mode=keyframes`
7. Extract keyframe langsung dari source
8. Resize keyframe menjadi 640–768 px
9. Ubah vision dari serial menjadi 2–4 request concurrent
10. Kaggle T4×2: dua server TP=1, bukan satu server TP=2

## P2 — quality + scalability

11. Two-pass summary vision
12. Hilangkan global temporal prompt 157 scene atau chunk secara hierarchical
13. Cache version-aware
14. Adaptive keyframe count
15. Benchmark tanpa `--enforce-eager`
16. Satu-pass FFmpeg render bila pipeline sudah stabil

---

# 21. Estimasi dampak relatif

Jangan menganggap angka berikut sebagai benchmark pasti; hasil bergantung codec video, GPU, resolusi dan scene count.

Namun secara arsitektural:

| Perubahan | Dampak yang diharapkan |
|---|---|
| Skip physical scene MP4 | **sangat besar**; menghapus 157 H.264 re-encode |
| Vision concurrency | **sangat besar** setelah preprocessing diperbaiki |
| 2 server pada T4×2 | besar untuk workload scene independen |
| Resize frame 1080p → 640/768 | besar pada multimodal prefill/VRAM |
| Structured JSON | besar pada reliability dan menghilangkan retry mahal |
| Two-pass summary | besar pada video panjang |
| `-c copy` saat concat | sedang–besar pada final summary |
| Hapus `--enforce-eager` jika stabil | sedang pada steady-state inference |
| frame_skip scene detector | opsional; ada tradeoff accuracy |

---

# 22. Target arsitektur akhir

```text
YouTube/local source
      │
      ├── subtitle/ASR
      │
      ▼
Scene detection (timestamps only)
      │
      ▼
Direct low-res keyframes from source
      │
      ├── Pass A: cheap concurrent vision
      │            2 frames / scene
      │            compact structured JSON
      │
      ▼
Candidate scene selection
      │
      ├── Pass B: detailed vision
      │            only selected candidates
      │
      ▼
Narrative/script generation
      │
      ▼
TTS
      │
      ▼
Single/few-pass FFmpeg cut + concat + audio + scale
      │
      ▼
Final dubbed / narrated video
```

Ini lebih cocok untuk Kaggle/Colab gratis karena menghemat:

- GPU minutes
- CPU transcode minutes
- VRAM
- storage I/O
- retry generation
- context tokens

dan sekaligus memperbaiki robustness pipeline.
