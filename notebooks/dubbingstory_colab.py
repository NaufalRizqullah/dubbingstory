"""
DubbingStory — Kaggle & Google Colab Runner
===========================================

This script automates the process of running DubbingStory on a cloud GPU (T4/L4) 
without relying on paid Vision API providers.

It will:
1. Start a local vLLM server with Qwen3-VL-2B-Instruct.
2. Wait for the server to be ready.
3. Run the DubbingStory pipeline using the local vision model and Piper for TTS.

Usage in Kaggle/Colab:
----------------------
1. Create a new notebook with GPU enabled (e.g., T4 x2 or P100).
2. Install dependencies in a cell:
   !pip install vllm openai yt-dlp opencv-python scenedetect google-genai piper-tts pysubs2 Pillow pyyaml python-dotenv fastapi uvicorn
3. Clone or copy DubbingStory code to the notebook environment.
4. Set your Gemini API key (for narration script writing):
   import os
   os.environ["GOOGLE_API_KEY"] = "your-gemini-api-key"
5. Run this script!

Note: You do NOT need a HuggingFace API key for Qwen3-VL-2B-Instruct, as it is an open model.
"""

import os
import subprocess
import sys
import time
import urllib.request
import json

# --- Configuration ---
# 2B is recommended for 16GB T4 GPUs. If you have a 24GB+ GPU, you can use 7B/8B.
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct" 
PORT = 8000
BASE_URL = f"http://127.0.0.1:{PORT}/v1"

def wait_for_server(url, timeout=300):
    """Wait for the vLLM server to become healthy."""
    print(f"⏳ Waiting for vLLM server to start at {url} (timeout: {timeout}s)...")
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
            # Server not ready yet
            pass
            
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(5)
        
    print("\n❌ Timeout waiting for server.")
    return False


def run_pipeline(video_input, project_name="my_project"):
    """Run the DubbingStory pipeline."""
    print("\n🚀 Starting DubbingStory Pipeline...")
    
    # Make sure we use the local vision provider and piper TTS
    cmd = [
        sys.executable, "-u", "-m", "dubbingstory.cli", "run",
        "--input", video_input,
        "--project", project_name,
        "--vision-provider", "openai",
        "--vision-model", MODEL_NAME,
        "--vision-base-url", BASE_URL,
        "--engine", "piper", # Use Piper for local TTS (no API cost)
        "--max-keyframes", "3",
        "--min-scene-duration", "5.0",
        "--scene-threshold", "5.0"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ ERROR: GOOGLE_API_KEY environment variable is not set.")
        print("Gemini API key is still required for writing the narration script.")
        print("Please set it: os.environ['GOOGLE_API_KEY'] = 'your-key'")
        sys.exit(1)

    video_file = input("Enter path to video file or YouTube URL: ").strip()
    if not video_file:
        print("No video provided. Exiting.")
        sys.exit(1)

    # 1. Start vLLM server in the background
    print(f"Starting vLLM server with model: {MODEL_NAME}...")
    # --max-model-len is limited to prevent OOM on T4 GPUs
    # --enforce-eager is used to save memory on some setups
    vllm_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--port", str(PORT),
        "--max-model-len", "4096", 
        "--tensor-parallel-size", "2",
        "--enforce-eager"
    ]
    
    # Open a subprocess, redirecting output so it doesn't clutter the main console
    vllm_log = open("vllm_server.log", "w")
    vllm_process = subprocess.Popen(vllm_cmd, stdout=vllm_log, stderr=subprocess.STDOUT)
    
    try:
        # 2. Wait for server
        if wait_for_server(BASE_URL):
            # 3. Run Pipeline
            run_pipeline(video_file)
            print("\n🎉 Pipeline completed successfully!")
            print("Check the 'outputs/' directory for your dubbed video.")
        else:
            print("Failed to start vision server. Check vllm_server.log for details.")
    finally:
        # Ensure we kill the vLLM server when done
        print("\nShutting down vLLM server...")
        vllm_process.terminate()
        vllm_process.wait()
        vllm_log.close()

if __name__ == "__main__":
    main()
