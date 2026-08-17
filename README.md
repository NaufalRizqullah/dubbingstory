# 🎬 DubbingStory

*[Read in English](README_EN.md)*

**Pipa Narasi & Dubbing Video Otomatis** — Ubah video bisu/asing menjadi cerita dengan dubbing AI.

Terinspirasi dari video repair Pakistan di Facebook yang di-dubbing dan dinarasikan secara akurat.

---

## ✨ Fitur

- 🎥 **Video Ingest** — Mendukung file lokal, URL YouTube, dan film bersubtitle.
- ✂️ **Auto Scene Detection** — Split video per scene otomatis menggunakan PySceneDetect.
- 🖼️ **Keyframe Extraction** — Ekstrak frame representatif per scene.
- 🧠 **Dual LLM Architecture** — Memisahkan tugas *Vision* (melihat adegan) dan *Text* (menulis narasi) demi keseimbangan kualitas dan biaya.
- 👁️ **Visual Understanding (Cloud & Local)** — Pahami konteks visual menggunakan **Gemini API** (default) atau model HuggingFace open-source (seperti **Qwen3-VL**) via integrasi OpenAI-compatible endpoint. Bebas biaya API Vision!
- ⚡ **Optimized Summary Mode** — Eksekusi lebih cepat dengan arsitektur *Two-Pass Vision* (Screening kilat -> Deep Analysis pada top-20 adegan) dan sistem *Smart Caching* berbasis hash.
- ☁️ **Kaggle / Colab Ready** — Disediakan skrip terintegrasi (`dubbingstory_colab.py`) untuk menjalankan model vision lokal secara gratis di cloud GPU T4.
- 📝 **Bilingual Narration** — Membuat script narasi dalam bahasa Indonesia & Inggris.
- 🎙️ **TTS Dubbing** — Dukungan Text-to-Speech online menggunakan **Edge TTS** secara bawaan untuk stabilitas tinggi di cloud (Kaggle/Colab).
- 🎬 **Video Render** — Render video final dengan mix audio dan subtitle (opsional) untuk format 16:9 & 9:16.
- ✂️ **Highlight Recap Mode** — Fitur untuk merangkum video panjang menjadi video pendek berisi momen-momen terpenting (gunakan `--mode summary`).

---

## 🚀 Mulai Cepat

### 1. Install

```bash
pip install -e .
```

### 2. Setup API Key

```bash
cp .env.sample .env
# Edit file .env dan tambahkan GOOGLE_API_KEY Anda
```

### 3. Jalankan

Anda dapat menggunakan perintah CLI `dubbingstory` atau menjalankan file `main.py` secara langsung:

```bash
# Menggunakan file main.py (jika perintah CLI belum bisa digunakan)
python main.py run --input video.mp4

# Pipa lengkap — video lokal
dubbingstory run --input video.mp4

# Pipa lengkap — menggunakan Qwen3-VL lokal via vLLM
dubbingstory run --input video.mp4 --vision-provider openai --engine edge

# Pipa lengkap — YouTube (dengan pernyataan hak cipta)
dubbingstory run --url "https://youtube.com/..." --i-have-rights

# Pipa lengkap — Mode Summary (Highlight Recap ~60 detik)
dubbingstory run --url "https://youtube.com/..." --mode summary --summary-duration 60 --i-have-rights

# Langkah demi langkah
dubbingstory ingest --input video.mp4
dubbingstory segment --project video
dubbingstory analyze --project video --vision-provider openai
dubbingstory narrate --project video --style viral_fb --lang id en
dubbingstory dub --project video --engine edge
dubbingstory render --project video --ratio 16:9 9:16
```

---

## ⚙️ Parameter CLI

### `run` (Pipa Lengkap)
Jalankan seluruh proses dari awal sampai akhir.

| Argumen | Default | Deskripsi |
|---------|---------|-----------|
| `--input`, `-i` | `None` | Jalur ke file video lokal. |
| `--url`, `-u` | `None` | Tautan YouTube atau video lainnya. |
| `--i-have-rights` | `False` | Konfirmasi hak cipta video (wajib untuk URL). |
| `--project`, `-p` | `auto` | Nama proyek (default: nama file). |
| `--mode` | `full` | Mode pipeline: `full` (video utuh) atau `summary` (potongan ringkasan highlight). |
| `--summary-duration`| `auto` | Target durasi (detik) untuk mode summary (misal: 60). |
| `--summary-max-scenes`| `auto` | Maksimal scene yang diambil untuk mode summary. |
| `--style` | `viral_fb` | Gaya narasi (`viral_fb`, `documentary`, `technical`, `calm_educational`). |
| `--lang` | `id en` | Bahasa target narasi. |
| `--engine` | `edge` | Mesin TTS yang digunakan (`edge`). |
| `--ratio` | `16:9` | Rasio aspek keluaran video (misal: `16:9 9:16`). |
| `--vision-provider` | `gemini` | Pilihan API Vision (`gemini` atau `openai` untuk vLLM/Qwen lokal). |
| `--vision-model` | - | Override nama model jika menggunakan `--vision-provider openai`. |
| `--max-keyframes` | `7` | Maks keyframe per scene untuk analisis vision. Turunkan (3-5) untuk lebih cepat. |
| `--min-scene-duration`| `2.0` | Durasi minimum scene (detik). Naikkan (4-6) agar scene lebih sedikit/cepat. |
| `--scene-threshold` | `3.0` | Sensitivitas deteksi scene. Naikkan (4-6) agar scene lebih sedikit/cepat. |

**Contoh:**
```bash
dubbingstory run --url "https://youtube.com/watch?v=..." --i-have-rights --style documentary --lang id --engine edge --ratio 16:9
```

---

### `ingest` (Unduh & Validasi)
Download (jika URL), validasi format, dan baca subtitle.

| Argumen | Default | Deskripsi |
|---------|---------|-----------|
| `--input`, `-i` | `None` | File video lokal. |
| `--url`, `-u` | `None` | Tautan video. |
| `--i-have-rights` | `False` | Konfirmasi hak cipta. |
| `--project`, `-p` | `auto` | Nama proyek. |

**Contoh:**
```bash
dubbingstory ingest --input video_saya.mp4 --project proyek_keren
```

---

### `segment` (Scene & Keyframes)
Deteksi pergantian scene dan ambil keyframe.

| Argumen | Default | Deskripsi |
|---------|---------|-----------|
| `--project`, `-p` | **(Wajib)** | Nama proyek. |

**Contoh:**
```bash
dubbingstory segment --project proyek_keren
```

---

### `analyze` (Pemahaman Visual)
Gunakan AI untuk memahami tiap scene secara visual.

| Argumen | Default | Deskripsi |
|---------|---------|-----------|
| `--project`, `-p` | **(Wajib)** | Nama proyek. |
| `--domain` | `""` | Petunjuk domain untuk membantu AI (misal: `workshop`, `cooking`, `repair`). |
| `--vision-provider` | `gemini` | Pilihan API Vision (`gemini` atau `openai` untuk vLLM/Qwen lokal). |

**Contoh:**
```bash
dubbingstory analyze --project proyek_keren --domain repair
```

---

### `narrate` (Pembuatan Skrip)
Hasilkan skrip narasi multibahasa beserta subtitle (SRT).

| Argumen | Default | Deskripsi |
|---------|---------|-----------|
| `--project`, `-p` | **(Wajib)** | Nama proyek. |
| `--style` | `viral_fb`| Gaya narasi. |
| `--lang` | `id en` | Bahasa target. |

**Contoh:**
```bash
dubbingstory narrate --project proyek_keren --style calm_educational --lang id en
```

---

### `dub` (Text-to-Speech)
Ubah skrip menjadi audio menggunakan AI Voice.

| Argumen | Default | Deskripsi |
|---------|---------|-----------|
| `--project`, `-p` | **(Wajib)** | Nama proyek. |
| `--engine` | `edge` | Mesin TTS (`edge`). |

**Contoh:**
```bash
dubbingstory dub --project proyek_keren --engine edge
```

---

### `render` (Penggabungan Video Final)
Gabungkan video asli, audio narasi (ducking/replace), dan atur rasio.

| Argumen | Default | Deskripsi |
|---------|---------|-----------|
| `--project`, `-p` | **(Wajib)** | Nama proyek. |
| `--ratio` | `16:9` | Rasio aspek. |

**Contoh:**
```bash
dubbingstory render --project proyek_keren --ratio 16:9 9:16
```

---

## 📁 Struktur Output

```
outputs/{project_name}/
├── source.mp4              # Video asli
├── video_metadata.json     # Info video
├── ingest_manifest.json    # Status ingest
├── scenes/                 # File video per scene
│   ├── scene_001.mp4
│   ├── scene_002.mp4
│   └── ...
├── keyframes/              # Keyframe yang diekstrak
│   ├── scene_001/
│   │   ├── scene_001_kf00.jpg
│   │   └── ...
│   └── ...
├── storyboard.json         # Analisis visual + narasi
├── scripts/
│   ├── script_id.txt       # Narasi bahasa Indonesia
│   ├── script_en.txt       # Narasi bahasa Inggris
│   ├── script_id.srt       # Subtitle bahasa Indonesia
│   └── script_en.srt       # Subtitle bahasa Inggris
├── audio/
│   ├── audio_id.wav        # Audio TTS Indonesia
│   └── audio_en.wav        # Audio TTS Inggris
├── final_id_16x9.mp4       # Video dubbing final (ID, landscape)
├── final_en_16x9.mp4       # Video dubbing final (EN, landscape)
├── final_id_9x16.mp4       # Video dubbing final (ID, vertikal)
└── final_en_9x16.mp4       # Video dubbing final (EN, vertikal)
```

## 🎨 Gaya Narasi

| Gaya | Nada | Cocok Untuk |
|------|------|-------------|
| `viral_fb` (default) | Bersemangat, kasual, menarik | Repost Facebook/TikTok |
| `documentary` | Formal, netral, informatif | Konten edukasi |
| `technical` | Presisi, ahli, analitis | Tutorial teknis |
| `calm_educational` | Lembut, sabar, bertahap | Video pembelajaran |

## 🔧 Mesin TTS

| Mesin | GPU | Lisensi | Dukungan Bahasa Indonesia | Kualitas |
|-------|-----|---------|---------------------------|----------|
| **Edge TTS** | ❌ API Online | Gratis | ✅ Bawaan | Sangat Baik |

## 🎬 Dukungan Film

Untuk film yang sudah memiliki subtitle (SRT/ASS/VTT), DubbingStory akan membaca teks subtitle sebagai konteks tambahan untuk membantu pemahaman visual:

```bash
# Letakkan file subtitle di folder yang sama dengan video:
# movie.mp4 + movie.srt → subtitle akan terdeteksi otomatis
dubbingstory run --input movie.mp4
```

## 📋 Persyaratan

- Python 3.10+
- FFmpeg (terdaftar di PATH)
- Google API Key (Gemini, free tier bisa digunakan)
- GPU dengan vLLM (Hanya jika Anda ingin menjalankan model vision secara lokal via `--vision-provider openai`)

## 🤝 Proyek Terkait

Proyek saudari dari [opensource-clipping](https://github.com/NaufalRizqullah/opensource-clipping) — AI Auto-Clipper untuk konten video pendek viral.
