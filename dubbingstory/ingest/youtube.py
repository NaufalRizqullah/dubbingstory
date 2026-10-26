"""Shared yt-dlp backend; historical YouTube entry points remain compatible."""

import hashlib
import json
import os
import re
import tempfile
from yt_dlp import YoutubeDL
from yt_dlp.version import __version__ as YTDLP_VERSION

from .sources import SourceIngestError, classify_download_error, normalize_source_url, resolve_source_url


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
        return f"bestvideo{codec_filter}+bestaudio/best{codec_filter}/bestvideo{codec_filter}"

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
            f"best[height<=?{h_val}]{codec_filter}/"
            f"bestvideo[height<=?{h_val}]{codec_filter}"
        )

    return (
        f"bestvideo[height<=?{h_val}]{codec_filter}+bestaudio/"
        f"best[height<=?{h_val}]{codec_filter}/"
        f"bestvideo[height<=?{h_val}]{codec_filter}"
    )


class _QuietLogger:
    """Prevent downloader diagnostics from exposing tokens/media URLs."""
    def debug(self, message):
        pass

    info = debug
    warning = debug
    error = debug


def _base_options(provider: str, cookies: str | None = None) -> dict:
    opts = {
        "quiet": True, "no_warnings": True, "logger": _QuietLogger(),
        "noplaylist": True, "cachedir": False, "socket_timeout": 30,
        "retries": 2, "extractor_retries": 1, "allow_unplayable_formats": False,
    }
    # Preserve existing YouTube behavior; never apply its special options or
    # browser-cookie configuration to the newly supported providers.
    if provider == "youtube":
        opts.update(
            extractor_args={"youtube": {"player_client": ["android", "web"]}},
            remote_components=["ejs:github"],
        )
        _apply_cookies(opts, cookies)
    return opts


def _require_single_video(info: dict | None) -> dict:
    if not isinstance(info, dict) or info.get("_type") in {"playlist", "multi_video"} or "entries" in info:
        raise SourceIngestError("unsupported_media", "Only a single video is supported, not playlists or image carousels.")
    formats = info.get("formats") or [info]
    if info.get("has_drm") or all(fmt.get("has_drm") for fmt in formats):
        raise SourceIngestError("protected", "DRM/protected media is not supported. Supply a permitted local video.")
    if not any(not fmt.get("has_drm") and fmt.get("vcodec") != "none" and fmt.get("url") for fmt in formats):
        raise SourceIngestError("no_video", "No accessible video stream was found; protected media and image posts are unsupported.")
    if info.get("is_live") or info.get("live_status") in {"is_live", "is_upcoming"}:
        raise SourceIngestError("unsupported_media", "Live streams are not supported. Supply a completed video.")
    return info


def _safe_text(value) -> str:
    # Descriptions may themselves contain share links; do not persist queries.
    return re.sub(r"(https?://[^\s?#]+)[?#][^\s]*", r"\1", str(value or ""))


def _metadata(info: dict, source, requested_source=None) -> dict:
    selected = info.get("requested_formats") or info.get("requested_downloads") or [info]
    video = next((item for item in selected if item.get("vcodec") != "none"), info)
    source_id = str(info.get("id") or source.video_id or "")
    if source.video_id and source_id != source.video_id:
        raise SourceIngestError("identity_mismatch", "The platform returned a different video than requested.")
    return {
        "schema_version": 2,
        "title": _safe_text(info.get("title")), "description": _safe_text(info.get("description")),
        "duration": video.get("duration") or info.get("duration") or 0,
        "uploader": _safe_text(info.get("uploader") or info.get("uploader_id")),
        "upload_date": info.get("upload_date") or "",
        "height": video.get("height") or info.get("height") or 0,
        "width": video.get("width") or info.get("width") or 0,
        "url": source.canonical_url, "canonical_url": source.canonical_url,
        "provider": source.provider, "source_id": source_id,
        "request_identity": (requested_source or source).identity,
        "source_identity": source.identity, "extractor": info.get("extractor_key") or source.extractor_key,
        "yt_dlp_version": YTDLP_VERSION,
        "selected_format": {
            key: video.get(key) for key in ("format_id", "ext", "width", "height", "fps", "vcodec", "acodec")
        },
    }


def extract_video_info(url: str, cookies: str | None = None) -> dict:
    """Extract safe metadata for a supported single video, without downloading."""
    requested = normalize_source_url(url)
    source = resolve_source_url(url)
    opts = _base_options(source.provider, cookies)
    opts["skip_download"] = True
    try:
        with YoutubeDL(opts) as ydl:
            info = _require_single_video(ydl.extract_info(source.url, download=False, ie_key=source.extractor_key))
        return _metadata(info, source, requested)
    except SourceIngestError:
        raise
    except Exception as exc:
        raise classify_download_error(exc) from None


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reusable_source(output_dir: str, requested, quality: str) -> str | None:
    output_path = os.path.join(output_dir, "source.mp4")
    metadata_path = os.path.join(output_dir, "video_metadata.json")
    if not os.path.isfile(output_path):
        if os.path.exists(metadata_path):
            raise SourceIngestError("cache_invalid", "This project has source metadata but its video is missing. Re-ingest into a new project to avoid mixing stale artifacts.")
        return None
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, ValueError):
        raise SourceIngestError("cache_conflict", "This project already contains a source without verified download identity. Use another --project name.") from None
    if not isinstance(metadata, dict):
        raise SourceIngestError("cache_invalid", "The source metadata is invalid. Re-ingest into a new project.")
    identities = {metadata.get("request_identity"), metadata.get("source_identity")}
    if requested.identity not in identities:
        raise SourceIngestError("cache_conflict", "This project contains a different source video. Use another --project name.")
    if str(metadata.get("download_quality")) != quality:
        raise SourceIngestError("cache_conflict", "This project was ingested at another quality. Use another --project name to keep existing artifacts consistent.")
    expected_hash = metadata.get("sha256")
    if not expected_hash or os.path.getsize(output_path) == 0 or _sha256(output_path) != expected_hash:
        raise SourceIngestError("cache_invalid", "The cached source failed integrity validation. Re-ingest into a new project.")
    return output_path


def download_video(
    url: str,
    output_dir: str,
    download_height: str | int = "1080",
    cookies: str | None = None,
) -> str:
    """
    Download a YouTube, Douyin, or Xiaohongshu/RedNote video using yt-dlp.

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
    requested = normalize_source_url(url)
    quality = str(_resolve_height(download_height))
    source = None
    try:
        existing = _reusable_source(output_dir, requested, quality)
    except SourceIngestError as exc:
        if exc.category != "cache_conflict" or not requested.is_short_link:
            raise
        # A new share link can point to a source already present in the project.
        # Resolve its public identity before deciding that it is a conflict.
        source = resolve_source_url(url)
        existing = _reusable_source(output_dir, source, quality)
    if existing:
        print("      ✅ Reusing verified source video.")
        return existing
    source = source or resolve_source_url(url)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "source.mp4")
    print(f"   📥 Downloading {source.provider} video ({'highest available' if quality == 'max' else 'height <= ' + quality})...")
    # Stage media first. Never replace an existing source or publish metadata
    # for a failed/partial download; later stages can safely reuse source.mp4.
    with tempfile.TemporaryDirectory(prefix=".ingest-", dir=output_dir) as staging:
        opts = _base_options(source.provider, cookies)
        opts.update({
            "format": _build_format_selector(download_height),
            "outtmpl": os.path.join(staging, "source.%(ext)s"),
            "merge_output_format": "mp4", "overwrites": False,
            "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
        })
        try:
            with YoutubeDL(opts) as ydl:
                info = _require_single_video(ydl.extract_info(source.url, download=False, ie_key=source.extractor_key))
                # Prevalidate metadata/identity before transferring the media.
                metadata = _metadata(info, source, requested)
                processed = ydl.process_ie_result(info, download=True)
                metadata = _metadata(processed, source, requested)
            staged_path = os.path.join(staging, "source.mp4")
            if not os.path.isfile(staged_path) or os.path.getsize(staged_path) == 0:
                raise SourceIngestError("invalid_media", "Download did not produce a reusable MP4 video.")
            from .local_video import get_video_metadata
            media = get_video_metadata(staged_path)
            if media.get("width", 0) <= 0 or media.get("video_duration", 0) <= 0:
                raise SourceIngestError("invalid_media", "Downloaded media did not pass video validation. Ensure FFprobe is available.")
            # Expose the actual downloaded geometry at the historical top level
            # too; extractor dimensions can describe a different rendition.
            metadata.update(media)
            metadata.update({"media": media, "sha256": _sha256(staged_path), "download_quality": quality})
            staged_metadata = os.path.join(staging, "video_metadata.json")
            with open(staged_metadata, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2)
            if os.path.exists(output_path):
                raise SourceIngestError("cache_conflict", "Another source appeared in this project during ingest. Use another project.")
            os.replace(staged_path, output_path)
            os.replace(staged_metadata, os.path.join(output_dir, "video_metadata.json"))
        except SourceIngestError:
            raise
        except Exception as exc:
            raise classify_download_error(exc) from None
    print(f"      ✅ Download complete: {output_path}")
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
    source = normalize_source_url(url)
    if source.provider != "youtube":
        return None
    url = source.url
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
            **_base_options("youtube", cookies),
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "subtitleslangs": languages,
            "subtitlesformat": "srt",
            "outtmpl": os.path.join(output_dir, "yt_subs"),
            "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
            "remote_components": ["ejs:github"],
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
                            **_base_options("youtube", cookies),
                            "quiet": True,
                            "no_warnings": True,
                            "skip_download": True,
                            "writesubtitles": sub_type == "manual",
                            "writeautomaticsub": sub_type == "auto",
                            "subtitleslangs": [lang],
                            "subtitlesformat": "srt",
                            "outtmpl": os.path.join(output_dir, "yt_subs"),
                            "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
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
            print(f"      ⚠️ {label} subtitle check failed: {classify_download_error(e)}")
            continue

    print(f"      ❌ No subtitles available for this video")
    return None
