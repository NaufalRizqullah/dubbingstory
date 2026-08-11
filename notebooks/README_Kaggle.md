# DubbingStory — Kaggle & Google Colab Guide

Panduan ini menjelaskan cara menjalankan pipeline DubbingStory di cloud GPU (Kaggle atau Google Colab) menggunakan model vision HuggingFace secara lokal, sehingga Anda bisa menganalisa video **tanpa biaya API model vision**!

Kita akan menggunakan **Qwen3-VL-2B-Instruct** (via `vLLM`), model vision open-source yang sangat kompeten dan muat di GPU T4 gratisan yang disediakan Kaggle/Colab.

---

## Persiapan Environment

1. Buat Notebook baru di [Kaggle](https://www.kaggle.com/) atau [Google Colab](https://colab.research.google.com/).
2. Aktifkan **GPU Accelerator** (T4 x2 di Kaggle, atau T4 di Colab).

## Langkah-langkah

### Step 1: Install Dependencies & Clone Repo

Buat Code cell pertama dan jalankan:

```bash
# Clone repository
!git clone https://github.com/NaufalRizqullah/dubbingstory.git
%cd dubbingstory

# Install core dependencies and vLLM for local vision model
!pip install -e .[advanced,openai-vision,colab]
```

### Step 2: Konfigurasi API Key

DubbingStory menggunakan Gemini API **hanya untuk merangkai narasi (script writer)**. Penggunaan teks ini sangat hemat (gratis jika dalam batas limit tier gratis Gemini). Model *vision* (yang biasanya mahal) akan dijalankan secara lokal di step berikutnya.

Buat cell baru:

```python
import os

# Dapatkan key gratis di https://aistudio.google.com/apikey
os.environ["GOOGLE_API_KEY"] = "MASUKKAN_GEMINI_API_KEY_ANDA_DISINI"
```

### Step 3: Jalankan Pipeline dengan Qwen3-VL

Kami telah menyediakan skrip automasi `dubbingstory_colab.py` yang akan:
1. Menjalankan server `vLLM` di background (download model `Qwen3-VL-2B-Instruct`).
2. Menunggu server siap.
3. Menjalankan pipeline DubbingStory penuh (menggunakan Piper TTS agar TTS juga berjalan lokal/gratis!).

Buat cell baru:

```bash
# Anda bisa mengetik path file video atau link YouTube saat diminta
!python notebooks/dubbingstory_colab.py
```

Saat skrip dijalankan, Anda akan diminta memasukkan input (misal link YouTube). Skrip akan mendownload model, melakukan analisa visual via Qwen, menulis narasi via Gemini, lalu men-dubbing video menggunakan Piper.

---

## FAQ & Tips

### Bagaimana mengganti model Vision?
Secara default, skrip menggunakan `Qwen/Qwen3-VL-2B-Instruct` karena pas untuk VRAM 16GB (T4 GPU).
Jika Anda mendapat akses GPU yang lebih besar (misal A100 atau L4 24GB+), Anda bisa mengganti model ke versi 4B atau 8B di dalam file `notebooks/dubbingstory_colab.py`:

```python
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct" 
```

Anda bisa mencari model Vision alternatif di [HuggingFace](https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending). Selama model tersebut di-support oleh `vLLM` dan merupakan OpenAI-compatible vision model, ia akan bekerja.

### Mengapa tetap butuh Gemini?
Kita menggunakan arsitektur *Dual LLM* seperti **NarratoAI**:
1. **Vision Model (Qwen3-VL)**: Hanya bertugas "melihat" keyframe dan mengekstrak objek/kejadian. (Jalan di GPU lokal, hemat biaya!).
2. **Text Model (Gemini)**: Mengambil hasil teks dari vision model, merangkai cerita, dan memberikan cue narasi. Model teks terbukti lebih baik dari vision model kecil dalam merangkai kata. Karena hanya mengirim teks prompt, biayanya sangat minim/gratis.

### Apakah Piper TTS perlu API Key?
Tidak. Piper TTS (`--engine piper`) berjalan 100% offline dan gratis. Voice model bahasa Indonesia/Inggris akan otomatis di-download saat pertama kali dijalankan.
