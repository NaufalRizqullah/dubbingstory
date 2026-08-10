"""
dubbingstory.ingest.youtube — YouTube video download via yt-dlp

Adapted from opensource-clipping/clipping/engine.py download logic.
Includes AV1 codec filter and format selector.
"""

import os
from yt_dlp import YoutubeDL


def _build_format_selector(download_height: str | int = "max") -> str:
    """
    Build yt-dlp format selector string.

    Skips AV1 codec (lacks HW acceleration on many platforms).
    """
    codec_filter = "[vcodec!^=av01]"

    if download_height == "max":
        return f"bestvideo{codec_filter}+bestaudio/best{codec_filter}/best"

    try:
        h_val = int(download_height)
    except (ValueError, TypeError):
        h_val = 0

    if 0 < h_val <= 1080:
        return (
            f"bestvideo[height<=?{h_val}][ext=mp4]+bestaudio[ext=m4a]/"
            f"bestvideo[height<=?{h_val}]{codec_filter}+bestaudio/"
            f"best[height<=?{h_val}][ext=mp4]/"
            f"best[height<=?{h_val}]{codec_filter}/"
            f"best"
        )

    return (
        f"bestvideo[height<=?{download_height}]{codec_filter}+bestaudio/"
        f"best[height<=?{download_height}]{codec_filter}/"
        f"best"
    )


def extract_video_info(url: str) -> dict:
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
    }

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
    download_height: str | int = "max",
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
        Target resolution height ("max", "1080", "720", etc.)

    Returns
    -------
    str
        Path to the downloaded video file.
    """
    output_path = os.path.join(output_dir, "source.mp4")

    print(f"   📥 Downloading from URL...")
    if download_height == "max":
        print("      🎯 Quality: highest available")
    else:
        print(f"      🎯 Quality: up to {download_height}p")

    # Extract metadata first
    try:
        info = extract_video_info(url)
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
    }

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
            **opts_extra,
        }

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
                        }
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
