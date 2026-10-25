# 🎬 DubbingStory

*[Read in English](README_EN.md)*

**Pipa Narasi & Dubbing Video Otomatis** — Ubah video bisu/asing menjadi cerita dengan dubbing AI.

Terinspirasi dari video repair Pakistan di Facebook yang di-dubbing dan dinarasikan secara akurat.

---

## ✨ Fitur

- 🎥 **Video Ingest** — File lokal, YouTube, TikTok, serta adapter URL/share link Douyin dan Xiaohongshu/RedNote melalui yt-dlp. Ketersediaan unduhan tetap mengikuti akses publik dan dukungan extractor platform.
- ✂️ **Auto Scene Detection** — Split video per scene otomatis menggunakan PySceneDetect.
- 🖼️ **Keyframe Extraction** — Ekstrak frame representatif per scene.
- 🧠 **Dual LLM Architecture** — Memisahkan tugas *Vision* (melihat adegan) dan *Text* (menulis narasi) demi keseimbangan kualitas dan biaya.
- 👁️ **Visual Understanding (Cloud & Local)** — Pahami konteks visual menggunakan **Gemini API** (default) atau model HuggingFace open-source (seperti **Qwen3-VL**) via integrasi OpenAI-compatible endpoint. Bebas biaya API Vision!
- ⚡ **Optimized Summary Mode** — Eksekusi lebih cepat dengan arsitektur *Two-Pass Vision* (Screening kilat -> Deep Analysis pada top-20 adegan) dan sistem *Smart Caching* berbasis hash.
- ☁️ **Kaggle / Colab Ready** — Disediakan skrip terintegrasi (`dubbingstory_colab.py`) untuk menjalankan model vision lokal secara gratis di cloud GPU T4.
- 📝 **Bilingual Narration** — Membuat script narasi dalam bahasa Indonesia & Inggris.
- 🎙️ **TTS Dubbing** — Dukungan Text-to-Speech online menggunakan **Edge TTS** secara bawaan untuk stabilitas tinggi di cloud, serta dukungan TTS offline dengan **Piper**. Dilengkapi fitur override suara via argumen `--voice-id` dan `--voice-en`.
- 🎬 **Video Render** — Render video final dengan mix audio dan subtitle (opsional) untuk format 16:9 & 9:16.
- ✂️ **Highlight Recap Mode** — Fitur untuk merangkum video panjang menjadi video pendek berisi momen-momen terpenting (gunakan `--mode summary`).
- 📱 **Native Storytelling** — `--mode full --ratio source` mempertahankan komposisi dan timeline video, dengan validasi durasi narasi.
- 💬 **Animated Captions** — Word-by-word atau karaoke dari timing kata TTS/forced alignment; SRT tetap tersedia, ASS dapat di-burn ke MP4.
- 🖼️ **Layered Vertical Render** — Canvas 9:16 dengan foreground tajam di tengah dan background video blur/grayscale; zoom kedua layer dapat diatur. `reframe` dapat mengolah MP4 hasil export tanpa mengulang TTS.

---

## 🚀 Mulai Cepat

### 1. Install

```bash
pip install -e .
```

### 2. Setup Lingkungan & API Key

```bash
cp .env.sample .env
```
Edit file `.env` dan konfigurasikan variabel berikut:
- `GOOGLE_API_KEY`: (Wajib) API key untuk Gemini (digunakan untuk menulis naskah).
- `OPENAI_VISION_IMAGE_MODE`: (Opsional) Cara pengiriman gambar ke model vision lokal (vLLM).
  - `data` (default): base64 encoding (aman dan selalu berhasil).
  - `file`: Mengirim file path lokal (jauh lebih cepat dan hemat memori, sangat disarankan untuk Kaggle/Colab). **Catatan**: Jika menggunakan `file`, Anda wajib menambahkan flag `--allowed-local-media-path <path>` saat menjalankan server vLLM.

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

# Video TikTok/RedNote yang sudah selesai diedit: pertahankan canvas asli,
# buat voice-over storytelling dari visual, dan burn karaoke subtitle.
dubbingstory run --url "https://www.tiktok.com/@creator/video/1234567890123456789" `
  --i-have-rights --mode full --lang id --engine edge --ratio source `
  --subtitle-style karaoke --burn-subtitles

# Opsional: gunakan dialog/transkrip audio sebagai konteks narasi.
# Subtitle lokal diprioritaskan; jika tidak ada, ASR dijalankan otomatis.
dubbingstory run --url "https://www.rednote.com/explore/..." `
  --i-have-rights --mode full --narration-source audio --lang id `
  --engine edge --ratio source

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

## 🧠 Bagaimana Pipa Ini Bekerja?

Untuk video vertikal yang editing-nya sudah selesai, gunakan workflow native:

```bash
dubbingstory run --input video.mp4 --mode full --lang id --engine edge --ratio source --subtitle-style karaoke --burn-subtitles
```

Pilihan `source` dan caption animasi mengaktifkan fitting narasi yang ketat: narasi yang terlalu panjang diringkas dan disintesis ulang secara terbatas, lalu gagal dengan pesan jelas jika masih tidak muat. Video tidak dipotong atau dipercepat untuk menyesuaikan suara. Gunakan `--narration-fit error` untuk menghentikan proses tanpa rewrite.

Lihat [panduan native storytelling, konfigurasi subtitle, dan pengujian Kaggle](docs/native-storytelling.md) untuk instalasi opsional alignment, ingest platform, output files, serta batas kompatibilitas.

Untuk mengubah video 16:9 menjadi canvas 9:16 dengan dua layer seperti editor video:

```bash
dubbingstory reframe --input final_summary_id_16x9.mp4 --output final_summary_id_vertical.mp4 --vertical-layout layered --foreground-zoom 25 --background-zoom 100 --background-blur 50 --background-grayscale 1
```

Preset ini juga tersedia pada `run` dan `render` dengan `--ratio 9:16 --vertical-layout layered`. Lihat [panduan layered vertical dan contoh Kaggle](docs/layered-vertical.md) untuk arti zoom 0–100, konfigurasi, serta cara menghasilkan output 16:9 dan 9:16 sekaligus.

DubbingStory membedah dan memproses video Anda secara otomatis melalui beberapa tahapan cerdas, khususnya saat Anda menggunakan **Mode Summary (Ringkasan)**:

1. **Pemotongan Visual (Scene Detection):** Algoritma akan menonton video dari awal hingga akhir dan mendeteksi perubahan drastis pada kamera (*camera cuts*). Video 30 menit mungkin akan terpecah menjadi ~150 adegan (*scenes*). *(Anda dapat mengatur parameter kecepatan seperti `--min-scene-duration` dan `--scene-threshold` untuk mengurangi jumlah potongan kecil).*
2. **Ekstraksi Gambar (Keyframes):** Untuk setiap adegan, program mengambil beberapa jepretan layar (*screenshot*) yang mewakili apa yang terjadi. *(Diatur oleh parameter `--max-keyframes`, turunkan angkanya untuk menghemat RAM GPU).*
3. **Penyaringan Kilat (Cheap Screening):** AI Vision (Gemini/Qwen) secara sekilas menilai dan memberikan skor ketertarikan (0-100) untuk masing-masing adegan.
4. **Pemilihan Adegan Terbaik (The Cut):** Sistem mengurutkan semua adegan berdasarkan skor tertinggi dan memilih adegan-adegan terbaik hingga mencapai **Target Durasi** (misal: 3 menit). Sisa adegan yang kurang menarik akan dibuang. **Penting:** Setelah terpilih, adegan-adegan emas ini akan *diurutkan kembali sesuai kronologi waktu aslinya*, sehingga video tidak akan melompat-lompat atau terbalik alurnya.
5. **Analisis Alur Cerita (Temporal Chunking):** Agar cerita tidak melompat-lompat, AI membaca urutan waktu dari adegan-adegan terpilih (dicicil 40 adegan per putaran untuk mencegah GPU *Out of Memory*) demi merumuskan awalan, proses, dan hasil akhir.
6. **Eksekusi (Cut, Tulis, Dubbing):** 
   - **Video Cutter:** Memotong belasan adegan emas tersebut dari video asli dan menyambungkannya.
   - **Script Writer:** AI Text menulis naskah narasi (sesuai gaya yang dipilih) khusus untuk adegan-adegan tersebut.
   - **TTS & Render:** Naskah diubah menjadi suara manusia (Edge/Piper) lalu digabungkan di atas kompilasi video (sambil mengecilkan suara aslinya).

---

## ⚙️ Parameter CLI

### `run` (Pipa Lengkap)
Jalankan seluruh proses dari awal sampai akhir.

| Argumen | Default | Deskripsi |
|---------|---------|-----------|
| `--input`, `-i` | `None` | Jalur ke file video lokal. |
| `--url`, `-u` | `None` | Tautan YouTube, TikTok, Douyin, atau Xiaohongshu/RedNote. |
| `--i-have-rights` | `False` | Konfirmasi hak cipta video (wajib untuk URL). |
| `--project`, `-p` | `auto` | Nama proyek (default: nama file). |
| `--mode` | `full` | Mode pipeline: `full` (video utuh) atau `summary` (potongan ringkasan highlight). |
| `--summary-duration`| `auto` | Target durasi (detik) untuk mode summary (misal: 60). |
| `--summary-max-scenes`| `auto` | Maksimal scene yang diambil untuk mode summary. |
| `--style` | `viral_fb` | Gaya narasi (`viral_fb`, `documentary`, `technical`, `calm_educational`). |
| `--lang` | `id en` | Bahasa target narasi. |
| `--engine` | `edge` | Mesin TTS yang digunakan (`edge`). |
| `--ratio` | `16:9` | Pilih `source` untuk mempertahankan komposisi; rasio tetap `16:9`, `9:16`, `1:1`, `4:3` juga tersedia. |
| `--vertical-layout` | `pad` | `layered` mengaktifkan foreground tajam dan background blur/grayscale khusus output 9:16. |
| `--foreground-zoom`, `--background-zoom` | `25`, `100` | Zoom tiap layer: 0 = muat utuh, 100 = memenuhi canvas. |
| `--background-blur`, `--background-grayscale` | `50`, `1.0` | Blur 1–100 dan grayscale 0–1 untuk background. |
| `--narration-fit` | `legacy` | `rewrite` atau `error` untuk validasi overflow; native/animasi otomatis memakai `rewrite` jika tidak dioverride. |
| `--subtitle-style` | `classic` | `classic`, `word`, atau `karaoke`; SRT tetap dihasilkan. |
| `--word-alignment` | `auto` | Metadata kata Edge, atau optional `whisperx`; tidak membagi durasi kata secara rata. |
| `--burn-subtitles` | dari config | Burn caption; `--no-burn-subtitles` hanya mengekspor file subtitle. |
| `--vision-provider` | `gemini` | Pilihan API Vision (`gemini` atau `openai` untuk vLLM/Qwen lokal). |
| `--vision-model` | - | Override nama model jika menggunakan `--vision-provider openai`. |
| `--max-keyframes` | `7` | Maks keyframe per scene untuk analisis vision. Turunkan (3-5) untuk lebih cepat. |
| `--disable-adaptive-keyframes`| `False` | Nonaktifkan pembatasan keyframe otomatis berbasis durasi. |
| `--min-scene-duration`| `2.0` | Durasi minimum scene (detik). Naikkan (4-6) agar scene lebih sedikit/cepat. |
| `--scene-threshold` | `3.0` | Sensitivitas deteksi scene. Naikkan (4-6) agar scene lebih sedikit/cepat. |
| `--speech-coverage` | `0.85`| Rasio target durasi narasi terhadap durasi video. |
| `--narration-source` | `visual` | Konteks narasi: `visual` (default) atau `audio` untuk subtitle/ASR dan terjemahan berbasis dialog. |

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
| `--disable-adaptive-keyframes`| `False` | Nonaktifkan pembatasan keyframe otomatis berbasis durasi. |

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
| `--speech-coverage`| `0.85` | Rasio target durasi narasi terhadap durasi video. |

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
