# Generated from: mereska-dubbingstory-yt.ipynb
# Converted at: 2026-08-18T13:31:20.633Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # DubbingStory — Kaggle Local Vision (Qwen3-VL)
# 
# Panduan ini menjelaskan cara menjalankan pipeline DubbingStory di cloud GPU (Kaggle) menggunakan model vision HuggingFace secara lokal, sehingga Anda bisa menganalisa video **tanpa biaya API model vision**!
# 
# Kita akan menggunakan **Qwen3-VL-2B-Instruct** (via `vLLM`), model vision open-source yang sangat kompeten dan muat di GPU T4 gratisan yang disediakan Kaggle.
# 
# ### Persiapan Environment
# Pastikan Anda telah mengaktifkan **GPU Accelerator** (T4 x2) di kanan menu pengaturan Kaggle Notebook.


# ## 1. Clone Repo
# 
# Download source code project.


# Clone repository langsung ke current directory agar tidak nested
!rm -rf ./* ./.*
!git clone -b main https://github.com/NaufalRizqullah/dubbingstory.git .


# ## 2. Konfigurasi Pipeline
# 
# Pilih mode pipeline:
# - `full` → Dubbing seluruh video (default)
# - `summary` → Buat highlight recap (video ringkasan dari scene terpenting)
# 


# --- SETTINGS ---
VIDEO_INPUT = "https://www.youtube.com/watch?v=ms4wRkLIO5U"  # Bisa URL atau path file lokal
RESOLUTION = "1080"
PROJECT_NAME = "my_dubbing_project"
STYLE = "viral_fb"
LANGUAGE = "id"
RATIO = "16:9"
ENGINE_TTS = "edge"

# --- PIPELINE MODE ---
MODE = "summary"  # "full" atau "summary"
SUMMARY_DURATION = 60 * 3  # Target durasi ringkasan (detik), None = otomatis
SUMMARY_MAX_SCENES = None  # Maks scene, None = otomatis

# --- SPEED OPTIMIZATION (Mencegah vLLM hang & mempercepat proses) ---
MAX_KEYFRAMES = 3           # Default 7. Turunkan ke 3-4 agar memori vLLM tidak penuh.
MIN_SCENE_DURATION = 4.0    # Default 2.0. Naikkan ke 5.0 agar scene lebih sedikit.
SCENE_THRESHOLD = 4.0       # Default 3.0. Naikkan ke 5.0 agar tidak over-sensitive.

# --- VISION MODEL ---
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
PORT = 8000
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
VISION_MAX_TOKENS = "1024"

# --- COLAB/KAGGLE GPU AUTO-DETECT ---
# Secara default diset untuk Colab/Kaggle T4 dengan TP=1
try:
    import torch
    NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
except Exception:
    NUM_GPUS = 0

if NUM_GPUS >= 1:
    TENSOR_PARALLEL_SIZE = "1"
    MAX_MODEL_LEN = "12288"
    GPU_MEM_UTIL = "0.85"
    print(f"🖥️  Detected {NUM_GPUS} GPUs -> tensor-parallel-size=1, max-model-len=12288")
else:
    TENSOR_PARALLEL_SIZE = "1"
    MAX_MODEL_LEN = "4096"
    GPU_MEM_UTIL = "0.80"
    print("⚠️  No GPU detected — pipeline akan gagal. Pastikan Runtime -> Change runtime type -> GPU.")


# ## 3. Download Video & Subtitles
# 
# Kita unduh video lebih dulu (hanya butuh `yt-dlp`). Jika download gagal (misal: 403 Forbidden), kita tidak perlu repot-repot compile PyTorch atau install vLLM yang memakan waktu.
# 


# Copy Cookies for bypass 403 / bot detection

import os
import shutil

# 0. Use cookies?
USE_COOKIES = True

# 1. Definisikan path asli (dari dataset Kaggle yang read-only)
ORIGINAL_COOKIES = "/kaggle/input/datasets/muhammadnaufal/tiktok-secret-cookies/youtube_cookies.txt"

# 2. Definisikan path baru di direktori kerja yang writable
COOKIES_PATH = "/kaggle/working/youtube_cookies.txt"

# 3. Salin file cookies ke direktori working jika belum ada/berubah
if os.path.exists(ORIGINAL_COOKIES):
    shutil.copy(ORIGINAL_COOKIES, COOKIES_PATH)
    print("✅ File cookies disalin ke /kaggle/working/ agar bisa dibaca/ditulis oleh yt-dlp")
else:
    print("⚠️ File cookies asli tidak ditemukan di dataset!")

import subprocess
import time
import urllib.request
import json
import sys
import os

print("   - Install Deno (JS runtime) + ffmpeg — yt-dlp butuh ini untuk download YouTube tanpa 403")
try:
    subprocess.check_call(
        ["apt-get", "-qq", "update"],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    subprocess.check_call(
        ["apt-get", "-qq", "install", "-y", "ffmpeg", "curl", "unzip"],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    subprocess.check_call(
        ["mkdir", "-p", "/root/.deno/bin"],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    subprocess.check_call(
        ["curl", "-L", "--retry", "5", "--retry-all-errors", "--connect-timeout", "20",
         "-o", "/tmp/deno.zip",
         "https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip"],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    subprocess.check_call(
        ["unzip", "-o", "/tmp/deno.zip", "-d", "/root/.deno/bin"],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    subprocess.check_call(
        ["chmod", "+x", "/root/.deno/bin/deno"],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    os.environ["PATH"] += ":/root/.deno/bin"
    subprocess.check_call(
        ["deno", "--version"],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    print(f"   ✅ Deno terinstall dan PATH updated")
except Exception as e:
    print(f"   ⚠️ Gagal install Deno + ffmpeg (tidak wajib, tapi bisa menghindari 403 di YouTube): {e}")

print("   - Upgrade yt-dlp ke versi terbaru (punya banyak workaround anti-403)")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "yt-dlp"],
    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
)

# Install yt-dlp terlebih dahulu
# !pip install -q yt-dlp

import sys
import os
sys.path.append(".")

from dubbingstory.ingest.youtube import download_video, download_subtitles

print("📥 Memulai proses download video...")
PROJECT_DIR = os.path.join("outputs", PROJECT_NAME)
os.makedirs(PROJECT_DIR, exist_ok=True)

# Download video dengan kualitas 1080p (jika tersedia)
if "http" in VIDEO_INPUT:
    try:
        if USE_COOKIES:
            video_path = download_video(
                VIDEO_INPUT,
                PROJECT_DIR,
                download_height=RESOLUTION,
                cookies=COOKIES_PATH,
            )
        else:
            video_path = download_video(
                VIDEO_INPUT,
                PROJECT_DIR,
                download_height=RESOLUTION,
            )
        
        # Download subtitle (opsional) untuk konteks tambahan Qwen
        download_subtitles(VIDEO_INPUT, PROJECT_DIR)
        print("\n✅ Download SUKSES!")
    except Exception as e:
        print(f"\n❌ DOWNLOAD GAGAL: {e}")
        print("\nJANGAN lanjutkan ke cell berikutnya! Perbaiki URL atau coba di environment lain (bisa jadi terkena 403).")
        raise
else:
    print("ℹ️ VIDEO_INPUT bukan URL. Dianggap sebagai file lokal.")


# ## 4. Install Dependencies & Environment Optimization
# 
# Karena download video sudah sukses, sekarang kita jalankan instalasi dependensi berat (PyTorch, vLLM, OpenCV, dll).
# 


# Install dependencies utama project
!pip install -q -r requirements.txt openai

import subprocess, sys
import os

print("🔧 Fixing environment...")
print("   - Uninstall torchaudio (vLLM tidak butuh; import-nya cuma lewat transformers.loss_rnnt)")
subprocess.check_call(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"],
    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
)

# --- DETEKSI KEMAMPUAN GPU ---
# T4 punya CC 7.5, P100 punya CC 6.0
try:
    import torch
    if torch.cuda.is_available():
        cc_major, cc_minor = torch.cuda.get_device_capability()
    else:
        cc_major, cc_minor = 0, 0
except Exception:
    cc_major, cc_minor = 0, 0

if cc_major > 0 and cc_major < 7:
    # Older GPU (misal: P100 -> CC 6.0)
    print(f"   - Detected Older GPU (Compute Capability {cc_major}.{cc_minor}).")
    print("   - Downgrading to PyTorch 2.7.1 (cu126) for older GPU compatibility...")
    subprocess.check_call(
        [
            sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", "--force-reinstall",
            "torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1",
            "--index-url", "https://download.pytorch.org/whl/cu126",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
else:
    # Modern GPU (misal: T4 -> CC 7.5)
    print(f"   - Detected Modern GPU (Compute Capability {cc_major}.{cc_minor}).")
    print("   - Reinstall torch + torchvision sinkron CUDA 13.0 (cocok dengan default Colab/Kaggle)")
    subprocess.check_call(
        [
            sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
            "torch==2.13.0", "torchvision==0.28.0",
            "--index-url", "https://download.pytorch.org/whl/cu130",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )

print("   - Upgrade vLLM ke versi yang sesuai")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "vllm"],
    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
)

print("✅ Done. Lanjut ke cell pipeline di bawah.")


# ## 5. Setup API Key via Kaggle Secrets
# 
# DubbingStory menggunakan Gemini API **hanya untuk merangkai narasi (script writer)**. Model *vision* akan dijalankan secara lokal.
# 
# 1. Buka tab **Secrets** (kunci) di panel kiri Kaggle.
# 2. Tambahkan rahasia baru dengan nama `GOOGLE_API_KEY` dan isikan API Key Gemini Anda.


from kaggle_secrets import UserSecretsClient
from pathlib import Path

API_KEY_GEMINI = UserSecretsClient().get_secret('GOOGLE_API_KEY') or ""

# Create .env File
env_text = f"""# Auto-generated from notebook userdata
GOOGLE_API_KEY={API_KEY_GEMINI}
"""

Path(".env").write_text(env_text, encoding="utf-8")
print("File .env berhasil dibuat")

# ## 6. Jalankan Pipeline
# 
# Server vLLM akan dijalankan di background, dan pipeline DubbingStory akan memproses video yang sudah diunduh tadi.
# 


import subprocess
import time
import urllib.request
import json
import sys
import os
from dotenv import load_dotenv
load_dotenv()

def wait_for_server(url, timeout=600):
    print(f"\n⏳ Waiting for vLLM server to start at {url} (timeout: {timeout}s)...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(f"{url}/models")
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    print(f"\n✅ vLLM Server is ready! Available models: {[m['id'] for m in data['data']]}")
                    return True
        except Exception:
            pass
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(5)
    print("\n❌ Timeout waiting for server.")
    return False

if not os.environ.get("GOOGLE_API_KEY"):
    print("❌ ERROR: GOOGLE_API_KEY belum di-set di cell sebelumnya (atau di Kaggle Secrets)!")
else:
    print(f"🚀 Starting vLLM server with model: {MODEL_NAME}...")
    vllm_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--port", str(PORT),
        "--max-model-len", MAX_MODEL_LEN,
        "--tensor-parallel-size", TENSOR_PARALLEL_SIZE,
        "--gpu-memory-utilization", GPU_MEM_UTIL,
        "--dtype", "half",
        "--enforce-eager",   # Hapus komentar ini jika sering OOM di T4
    ]

    print(f"   Command: {' '.join(vllm_cmd)}")

    vllm_log = open("vllm_server.log", "w")
    vllm_process = subprocess.Popen(vllm_cmd, stdout=vllm_log, stderr=subprocess.STDOUT)

    try:
        if wait_for_server(BASE_URL):
            print(f"\n🚀 Starting DubbingStory Pipeline (mode: {MODE})...")

            local_video_path = os.path.join("outputs", PROJECT_NAME, "source.mp4")
            if "http" in VIDEO_INPUT and os.path.exists(local_video_path):
                cmd_input = local_video_path
                print(f"Menggunakan video yang sudah didownload: {cmd_input}")
            else:
                cmd_input = VIDEO_INPUT

            cmd = [
                sys.executable, "-u", "main.py", "run",
                "--input", cmd_input,
                "--project", PROJECT_NAME,
                "--style", STYLE,
                "--lang", LANGUAGE,
                "--ratio", RATIO,
                "--mode", MODE,
                "--vision-provider", "openai",
                "--vision-model", MODEL_NAME,
                "--vision-base-url", BASE_URL,
                "--vision-max-tokens", VISION_MAX_TOKENS,
                "--engine", ENGINE_TTS,
                "--max-keyframes", str(MAX_KEYFRAMES),
                "--min-scene-duration", str(MIN_SCENE_DURATION),
                "--scene-threshold", str(SCENE_THRESHOLD)
            ]

            if MODE == "summary":
                if SUMMARY_DURATION is not None:
                    cmd.extend(["--summary-duration", str(SUMMARY_DURATION)])
                if SUMMARY_MAX_SCENES is not None:
                    cmd.extend(["--summary-max-scenes", str(SUMMARY_MAX_SCENES)])

            if "http" in cmd_input:
                cmd[cmd.index("--input")] = "--url"
                cmd.append("--i-have-rights")

            print(f"Executing: {' '.join(cmd)}")

            # ── PERBAIKAN KRITIS ───────────────────────────────────────────────
            # subprocess.run(check=True) raise CalledProcessError yang __str__()
            # TIDAK menyertakan output child. Jadi "returned non-zero exit status 1"
            # TIDAK kasih kita info apa yang sebenarnya salah.
            #
            # SOLUSI: pakai Popen + stream stdout/stderr real-time ke:
            #  1. cell output notebook (echo langsung, kelihatan)
            #  2. file pipeline.log (supaya bisa di-tail saat error)
            pipeline_log_path = "pipeline.log"
            with open(pipeline_log_path, "w") as plog:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,   # gabung stderr ke stdout
                    text=True,
                    bufsize=1,
                )
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    plog.write(line)
                returncode = proc.wait()
            if returncode != 0:
                # Raise supaya masuk ke except handler di bawah, yang akan
                # print tail pipeline.log supaya root cause kelihatan.
                raise subprocess.CalledProcessError(returncode, cmd)

            print(f"\n🎉 Pipeline completed successfully! (mode: {MODE})")
            print(f"📂 Check the 'outputs/{PROJECT_NAME}/' directory for your video.")
        else:
            print("Failed to start vision server. Check vllm_server.log for details.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        # Tampilkan jejak pipeline.log + vllm_server.log supaya root cause kelihatan
        for log_name in ("pipeline.log", "vllm_server.log"):
            if os.path.exists(log_name) and os.path.getsize(log_name) > 0:
                print(f"\n──── tail of {log_name} ────")
                with open(log_name, "r", errors="replace") as lf:
                    print("".join(lf.readlines()[-60:]))
        raise
    finally:
        print("\n🛑 Shutting down vLLM server...")
        vllm_process.terminate()
        try:
            vllm_process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            vllm_process.kill()
            vllm_process.wait()
        vllm_log.close()