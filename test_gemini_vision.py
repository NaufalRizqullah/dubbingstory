"""
test_gemini_vision.py — Test Gemini Vision API (free tier)

Standalone test script to validate that Gemini Vision can analyze
video frames. Extracts 5 frames (1 per second from second 1-5) and
sends them to the API.

Usage:
    python test_gemini_vision.py --url "https://youtube.com/watch?v=xxx"
    python test_gemini_vision.py --input "path/to/local/video.mp4"

Requires:
    - GOOGLE_API_KEY in .env or environment
    - opencv-python, google-genai, yt-dlp (for URL mode)
"""

import argparse
import os
import sys
import json
import time

# Load .env from project root
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


def extract_frames(video_path: str, start_sec: int = 1, end_sec: int = 5) -> list[str]:
    """
    Extract 1 frame per second from second 1 to 5 (inclusive).
    Returns list of saved frame paths.
    """
    import subprocess

    print(f"   📹 Extracting frames with FFmpeg...")
    
    # Get video duration via ffprobe
    duration_cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    try:
        duration_str = subprocess.check_output(duration_cmd).decode("utf-8").strip()
        duration = float(duration_str)
        print(f"      Duration: {duration:.1f}s")
    except Exception as e:
        print(f"   ⚠️ Could not get duration: {e}")
        duration = 999  # assume long enough

    if duration < end_sec:
        print(f"   ⚠️ Video shorter than {end_sec}s — adjusting range")
        end_sec = min(int(duration), end_sec)
        if end_sec <= start_sec:
            end_sec = start_sec + 1

    # Create output directory
    frames_dir = os.path.join(os.path.dirname(video_path) or ".", "test_frames")
    os.makedirs(frames_dir, exist_ok=True)

    frame_paths = []
    for sec in range(start_sec, end_sec + 1):
        frame_path = os.path.join(frames_dir, f"frame_sec{sec:02d}.jpg")
        
        # Use FFmpeg to seek and extract a single frame accurately
        cmd = [
            "ffmpeg", "-y", "-ss", str(sec), "-i", video_path,
            "-frames:v", "1", "-q:v", "2", "-v", "error", frame_path
        ]
        try:
            subprocess.run(cmd, check=True)
            if os.path.exists(frame_path):
                frame_paths.append(frame_path)
                print(f"   📸 Extracted frame at {sec}s → {os.path.basename(frame_path)}")
            else:
                print(f"   ⚠️ Output file not found for second {sec}")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️ FFmpeg failed to extract frame at second {sec}: {e}")

    return frame_paths


# Quality label → pixel height mapping
QUALITY_MAP = {
    "720": 720,
    "1080": 1080,
    "2k": 1440,
    "4k": 2160,
}


def download_video(url: str, quality: str = "1080") -> str:
    """Download video from URL using yt-dlp.

    Parameters
    ----------
    url : str
        Video URL.
    quality : str
        Quality label: "720", "1080" (default), "2k", "4k".

    Returns path to downloaded file.
    """
    from yt_dlp import YoutubeDL

    height = QUALITY_MAP.get(quality, 1080)

    output_dir = os.path.join(os.path.dirname(__file__) or ".", "test_output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test_video.mp4")

    # Skip AV1 codec (lacks HW accel on many platforms)
    codec_filter = "[vcodec!^=av01]"

    fmt = (
        f"bestvideo[height<=?{height}][ext=mp4]+bestaudio[ext=m4a]/"
        f"bestvideo[height<=?{height}]{codec_filter}+bestaudio/"
        f"best[height<=?{height}][ext=mp4]/"
        f"best[height<=?{height}]{codec_filter}/"
        f"best"
    )

    print(f"\n   📥 Downloading video from URL...")
    print(f"      🎯 Quality: up to {quality}p ({height}px)")
    ydl_opts = {
        "format": fmt,
        "outtmpl": output_path,
        "quiet": True,
        "merge_output_format": "mp4",
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        print(f"      Title: {info.get('title', 'N/A')[:60]}")
        print(f"      Duration: {info.get('duration', 0)}s")
        ydl.download([url])

    if not os.path.exists(output_path):
        print(f"❌ Download failed — file not found: {output_path}")
        sys.exit(1)

    print(f"   ✅ Downloaded: {output_path}")
    return output_path


def test_vision_with_frames(api_key: str, frame_paths: list[str]) -> dict:
    """
    Test Gemini Vision by sending extracted frames.
    Returns the API response as dict.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    model = "gemini-2.0-flash"  # Free tier model

    print(f"\n{'='*60}")
    print(f"🧪 TEST 1: Gemini Vision — Frame Analysis")
    print(f"{'='*60}")
    print(f"   Model: {model}")
    print(f"   Frames: {len(frame_paths)}")

    # Build content parts: images + text prompt
    parts = []
    for fp in frame_paths:
        with open(fp, "rb") as f:
            data = f.read()
        parts.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))
        print(f"   📎 Attached: {os.path.basename(fp)} ({len(data)/1024:.1f} KB)")

    prompt_text = """You are analyzing 5 consecutive frames extracted from a video (1 frame per second, seconds 1-5).

Describe:
1. What is happening in the video
2. What objects/people are visible
3. The setting/environment
4. Any text or signs visible
5. What changes between the frames

Return ONLY valid JSON:
{
  "description": "Overall description of what's happening",
  "objects": ["list", "of", "visible", "objects"],
  "people": "Description of people if any",
  "environment": "Description of setting",
  "text_visible": ["any text seen"],
  "changes_between_frames": "What changed from first to last frame",
  "confidence": 0.0
}"""

    parts.append(types.Part.from_text(text=prompt_text))
    contents = [types.Content(role="user", parts=parts)]

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.2,
    )

    print(f"\n   ⏳ Sending {len(frame_paths)} frames to Gemini Vision...")
    start_time = time.time()

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        elapsed = time.time() - start_time
        text = getattr(response, "text", None)

        if not text:
            print(f"   ❌ FAIL — Empty response from Gemini")
            return {"status": "FAIL", "error": "Empty response"}

        result = json.loads(text)

        print(f"\n   ✅ SUCCESS! ({elapsed:.1f}s)")
        print(f"\n   📋 Response:")
        print(f"      Description: {result.get('description', 'N/A')[:200]}")
        print(f"      Objects: {result.get('objects', [])}")
        print(f"      People: {result.get('people', 'N/A')[:100]}")
        print(f"      Environment: {result.get('environment', 'N/A')[:100]}")
        print(f"      Text visible: {result.get('text_visible', [])}")
        print(f"      Changes: {result.get('changes_between_frames', 'N/A')[:100]}")
        print(f"      Confidence: {result.get('confidence', 'N/A')}")

        return {"status": "PASS", "elapsed": elapsed, "result": result}

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n   ❌ FAIL ({elapsed:.1f}s)")
        print(f"      Error: {e}")
        return {"status": "FAIL", "elapsed": elapsed, "error": str(e)}


def test_vision_with_video_upload(api_key: str, video_path: str) -> dict:
    """
    Test Gemini Vision by uploading a short video clip (first 5 seconds).
    Uses Gemini Files API.
    """
    import subprocess

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    model = "gemini-2.0-flash"

    print(f"\n{'='*60}")
    print(f"🧪 TEST 2: Gemini Vision — Video Upload (5s clip)")
    print(f"{'='*60}")
    print(f"   Model: {model}")

    # Extract first 5 seconds as a clip
    clip_dir = os.path.join(os.path.dirname(video_path) or ".", "test_frames")
    os.makedirs(clip_dir, exist_ok=True)
    clip_path = os.path.join(clip_dir, "test_clip_5s.mp4")

    print(f"   ✂️ Extracting first 5 seconds...")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-t", "5",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac",
        clip_path,
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception as e:
        print(f"   ❌ FAIL — Could not extract clip: {e}")
        return {"status": "FAIL", "error": f"Clip extraction failed: {e}"}

    clip_size = os.path.getsize(clip_path) / (1024 * 1024)
    print(f"   📎 Clip: {clip_path} ({clip_size:.1f} MB)")

    # Upload video to Gemini Files API
    print(f"   📤 Uploading to Gemini Files API...")
    start_time = time.time()

    try:
        video_file = client.files.upload(file=clip_path)

        # Wait for processing
        while video_file.state == "PROCESSING":
            print(f"      ⏳ Processing... (state={video_file.state})")
            time.sleep(3)
            video_file = client.files.get(name=video_file.name)

        if video_file.state == "FAILED":
            print(f"   ❌ FAIL — Video processing failed")
            return {"status": "FAIL", "error": "Video processing failed"}

        upload_elapsed = time.time() - start_time
        print(f"   ✅ Upload complete ({upload_elapsed:.1f}s)")

        # Now analyze
        prompt = """Analyze this 5-second video clip and describe:
1. What is happening
2. Objects and people visible
3. The setting/environment
4. Any spoken words or text on screen

Return ONLY valid JSON:
{
  "description": "What's happening in the video",
  "objects": ["visible", "objects"],
  "people": "People description",
  "environment": "Setting description",
  "audio_content": "Any speech or sounds detected",
  "confidence": 0.0
}"""

        print(f"   ⏳ Analyzing video...")
        analysis_start = time.time()

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        )

        response = client.models.generate_content(
            model=model,
            contents=[video_file, prompt],
            config=config,
        )

        analysis_elapsed = time.time() - analysis_start
        text = getattr(response, "text", None)

        if not text:
            print(f"   ❌ FAIL — Empty response")
            return {"status": "FAIL", "error": "Empty response"}

        result = json.loads(text)

        total_elapsed = time.time() - start_time
        print(f"\n   ✅ SUCCESS! (upload={upload_elapsed:.1f}s, analysis={analysis_elapsed:.1f}s)")
        print(f"\n   📋 Response:")
        print(f"      Description: {result.get('description', 'N/A')[:200]}")
        print(f"      Objects: {result.get('objects', [])}")
        print(f"      People: {result.get('people', 'N/A')[:100]}")
        print(f"      Environment: {result.get('environment', 'N/A')[:100]}")
        print(f"      Audio: {result.get('audio_content', 'N/A')[:100]}")
        print(f"      Confidence: {result.get('confidence', 'N/A')}")

        # Cleanup uploaded file
        try:
            client.files.delete(name=video_file.name)
            print(f"   🗑️ Cleaned up uploaded file")
        except Exception:
            pass

        return {"status": "PASS", "elapsed": total_elapsed, "result": result}

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n   ❌ FAIL ({elapsed:.1f}s)")
        print(f"      Error: {e}")
        return {"status": "FAIL", "elapsed": elapsed, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Test Gemini Vision API (free tier) with video frames"
    )
    parser.add_argument("--url", "-u", type=str, help="YouTube/video URL to test with")
    parser.add_argument("--input", "-i", type=str, help="Path to local video file")
    parser.add_argument(
        "--quality", "-q", type=str, default="1080",
        choices=["720", "1080", "2k", "4k"],
        help="Download quality (default: 1080). Options: 720, 1080, 2k, 4k"
    )
    parser.add_argument(
        "--skip-upload", action="store_true",
        help="Skip the video upload test (Test 2)"
    )
    args = parser.parse_args()

    if not args.url and not args.input:
        parser.print_help()
        print("\n❌ Harus pilih --url atau --input")
        sys.exit(1)

    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key or api_key == "your-gemini-api-key-here":
        print("❌ GOOGLE_API_KEY not found!")
        print("   Set in .env file or environment variable.")
        print("   Get yours at: https://aistudio.google.com/apikey")
        sys.exit(1)

    print("=" * 60)
    print("🧪 DubbingStory — Gemini Vision API Test")
    print("=" * 60)
    print(f"   API Key: {api_key[:8]}...{api_key[-4:]}")

    # Get video
    if args.url:
        video_path = download_video(args.url, quality=args.quality)
    else:
        video_path = args.input
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            sys.exit(1)
        print(f"\n   📁 Using local video: {video_path}")

    # Extract frames (second 1-5, 1 frame per second = 5 frames)
    print(f"\n{'='*60}")
    print(f"📸 Extracting frames (second 1-5, 1 FPS)")
    print(f"{'='*60}")
    frame_paths = extract_frames(video_path, start_sec=1, end_sec=5)

    if not frame_paths:
        print("❌ No frames extracted — aborting")
        sys.exit(1)

    print(f"   ✅ Extracted {len(frame_paths)} frames")

    # ── Test 1: Frame analysis ──
    result_frames = test_vision_with_frames(api_key, frame_paths)

    # ── Test 2: Video upload ──
    result_upload = {"status": "SKIPPED"}
    if not args.skip_upload:
        result_upload = test_vision_with_video_upload(api_key, video_path)
    else:
        print(f"\n⏭️ Skipping Test 2 (video upload)")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"   Test 1 (Frame Analysis):  {result_frames['status']}")
    print(f"   Test 2 (Video Upload):    {result_upload['status']}")

    if result_frames["status"] == "PASS":
        print(f"\n   🎉 Gemini Vision API WORKS on free tier!")
        print(f"   ✅ You can use vision analysis in the DubbingStory pipeline.")
    else:
        print(f"\n   ⚠️ Gemini Vision API test failed.")
        print(f"   Check your API key and internet connection.")

    # Save full results
    results_path = os.path.join(
        os.path.dirname(video_path) or ".",
        "test_frames", "test_results.json"
    )
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_frame_analysis": result_frames,
            "test_video_upload": result_upload,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n   📄 Full results saved: {results_path}")


if __name__ == "__main__":
    main()
