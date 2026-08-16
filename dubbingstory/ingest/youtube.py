"""
dubbingstory.ingest.youtube — YouTube video download via yt-dlp

Adapted from opensource-clipping/clipping/engine.py download logic.
Includes AV1 codec filter and format selector.
"""

import os
from yt_dlp import YoutubeDL


# ── Quality label → pixel height mapping ─────────────────────────────────────
QUALITY_MAP = {
    "720": 720,
    "1080": 1080,
    "2k": 1440,
    "4k": 2160,
    "max": "max",
}


def _resolve_height(download_height: str | int) -> str | int:
    """Resolve a human-friendly quality label to a pixel height value."""
    if isinstance(download_height, int):
        return download_height
    label = str(download_height).lower().strip()
    return QUALITY_MAP.get(label, label)


def _apply_cookies(opts: dict, cookies: str | None) -> None:
    """Apply cookies to yt-dlp options dictionary."""
    if not cookies:
        return
    if cookies.endswith(".txt") or os.path.isfile(cookies):
        opts["cookiefile"] = cookies
    else:
        # Browser format: (browser, profile, keyring, container)
        opts["cookiesfrombrowser"] = (cookies, None, None, None)


def _build_format_selector(download_height: str | int = "1080") -> str:
    """
    Build yt-dlp format selector string.

    Skips AV1 codec (lacks HW acceleration on many platforms).
    Accepts labels: "720", "1080", "2k", "4k", "max".
    """
    # Skip AV1 codec as it lacks HW acceleration on many platforms (e.g., Colab T4)
    # and causes decoding failures in OpenCV/FFmpeg software fallbacks.
    # Note: Using [vcodec!*=av01] to safely ensure it does not contain 'av01' anywhere.
    codec_filter = "[vcodec!*=av01]"

    h_val = _resolve_height(download_height)

    if h_val == "max":
        return f"bestvideo{codec_filter}+bestaudio/best{codec_filter}"

    try:
        h_val = int(h_val)
    except (ValueError, TypeError):
        h_val = 1080  # fallback to 1080p if invalid

    if 0 < h_val <= 1080:
        # For standard resolutions, strictly prefer native MP4 (H.264/AAC), ensuring no AV1 in mp4
        return (
            f"bestvideo[height<=?{h_val}][ext=mp4]{codec_filter}+bestaudio[ext=m4a]/"
            f"bestvideo[height<=?{h_val}]{codec_filter}+bestaudio/"
            f"best[height<=?{h_val}][ext=mp4]{codec_filter}/"
            f"best[height<=?{h_val}]{codec_filter}"
        )

    return (
        f"bestvideo[height<=?{h_val}]{codec_filter}+bestaudio/"
        f"best[height<=?{h_val}]{codec_filter}"
    )


def extract_video_info(url: str, cookies: str | None = None) -> dict:
    """
    Extract video metadata without downloading.

    Returns
    -------
    dict
        Keys: title, description, duration, uploader, upload_date, etc.
    """
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
        "extractor_args": {"youtube": ["player_client=android,web"]},
    }
    _apply_cookies(ydl_opts, cookies)

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    return {
        "title": info.get("title", ""),
        "description": info.get("description", ""),
        "duration": info.get("duration", 0),
        "uploader": info.get("uploader", ""),
        "upload_date": info.get("upload_date", ""),
        "height": info.get("height", 0),
        "width": info.get("width", 0),
        "url": url,
    }


def download_video(
    url: str,
    output_dir: str,
    download_height: str | int = "1080",
    cookies: str | None = None,
) -> str:
    """
    Download a video from YouTube (or other supported sites) using yt-dlp.

    Parameters
    ----------
    url : str
        Video URL.
    output_dir : str
        Directory to save the downloaded video.
    download_height : str | int
        Target resolution: "720", "1080" (default), "2k", "4k", "max".
    cookies : str | None
        Cookies from browser name (e.g., "chrome") or path to cookies.txt.

    Returns
    -------
    str
        Path to the downloaded video file.
    """
    output_path = os.path.join(output_dir, "source.mp4")

    resolved = _resolve_height(download_height)
    print(f"   📥 Downloading from URL...")
    if resolved == "max":
        print("      🎯 Quality: highest available")
    else:
        print(f"      🎯 Quality: up to {download_height} ({resolved}px)")

    # Extract metadata first
    try:
        info = extract_video_info(url, cookies=cookies)
        print(f"      📋 Title: {info['title'][:60]}...")
        print(f"      📋 Duration: {info['duration']}s")
        print(f"      📋 Uploader: {info['uploader']}")

        # Save video metadata (useful for context hints in vision analysis)
        import json
        meta_path = os.path.join(output_dir, "video_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"      ⚠️ Could not extract metadata: {e}")

    # Download
    ydl_opts = {
        "format": _build_format_selector(download_height),
        "outtmpl": output_path,
        "quiet": True,
        "merge_output_format": "mp4",
        "extractor_args": {"youtube": ["player_client=android,web"]},
    }
    _apply_cookies(ydl_opts, cookies)

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not os.path.exists(output_path):
        raise RuntimeError(
            f"❌ Download gagal — file tidak ditemukan di {output_path}\n"
            "   Pastikan URL valid dan bisa diakses."
        )

    print(f"      ✅ Download selesai: {output_path}")
    return output_path


def download_subtitles(
    url: str,
    output_dir: str,
    languages: list[str] | None = None,
    cookies: str | None = None,
) -> str | None:
    """
    Download subtitles (auto-generated or manual) from YouTube via yt-dlp.

    Tries manual subs first, then falls back to auto-generated subs.
    This is a backup context source when no local subtitle files are found.

    Parameters
    ----------
    url : str
        Video URL.
    output_dir : str
        Directory to save downloaded subtitle files.
    languages : list[str] | None
        Preferred subtitle languages (e.g., ["id", "en"]).
        If None, tries Indonesian then English.

    Returns
    -------
    str or None
        Path to the downloaded subtitle file (.srt), or None if unavailable.
    """
    if languages is None:
        languages = ["id", "en"]

    lang_str = ",".join(languages)

    # Try manual subtitles first, then auto-generated
    for sub_type, opts_extra in [
        ("manual", {}),
        ("auto", {"writeautomaticsub": True}),
    ]:
        label = "manual" if sub_type == "manual" else "auto-generated"
        print(f"      🔍 Checking {label} subtitles ({lang_str})...")

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "subtitleslangs": languages,
            "subtitlesformat": "srt",
            "outtmpl": os.path.join(output_dir, "yt_subs"),
            "extractor_args": {"youtube": ["player_client=android,web"]},
            **opts_extra,
        }
        _apply_cookies(ydl_opts, cookies)

        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                # Check which subs are available
                available_subs = info.get("subtitles", {}) if sub_type == "manual" else {}
                available_auto = info.get("automatic_captions", {}) if sub_type == "auto" else {}

                subs_pool = available_subs if sub_type == "manual" else available_auto

                # Find first matching language
                for lang in languages:
                    if lang in subs_pool:
                        print(f"      ✅ Found {label} subs: {lang}")

                        # Actually download the subtitle
                        dl_opts = {
                            "quiet": True,
                            "no_warnings": True,
                            "skip_download": True,
                            "writesubtitles": sub_type == "manual",
                            "writeautomaticsub": sub_type == "auto",
                            "subtitleslangs": [lang],
                            "subtitlesformat": "srt",
                            "outtmpl": os.path.join(output_dir, "yt_subs"),
                            "extractor_args": {"youtube": ["player_client=android,web"]},
                        }
                        _apply_cookies(dl_opts, cookies)
                        with YoutubeDL(dl_opts) as ydl2:
                            ydl2.download([url])

                        # Find the downloaded subtitle file
                        import glob
                        srt_patterns = [
                            os.path.join(output_dir, f"yt_subs.{lang}.srt"),
                            os.path.join(output_dir, f"yt_subs.{lang}.vtt"),
                        ]
                        for pattern in srt_patterns:
                            if os.path.exists(pattern):
                                print(f"      📄 Subtitle saved: {os.path.basename(pattern)}")
                                return pattern

                        # Glob fallback
                        for match in glob.glob(os.path.join(output_dir, "yt_subs.*")):
                            if match.endswith((".srt", ".vtt")):
                                print(f"      📄 Subtitle saved: {os.path.basename(match)}")
                                return match

        except Exception as e:
            print(f"      ⚠️ {label} subtitle check failed: {e}")
            continue

    print(f"      ❌ No subtitles available for this video")
    return None
