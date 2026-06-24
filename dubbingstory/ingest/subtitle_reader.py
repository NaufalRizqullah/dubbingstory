"""
dubbingstory.ingest.subtitle_reader — Read existing subtitles for context

Supports SRT, ASS, SSA, VTT, MicroDVD, and other formats via pysubs2.
Used to provide additional context for visual understanding of films/videos
that already have subtitles available.
"""

import os
import glob

import pysubs2


SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".vtt", ".sub", ".mpl2"}


def find_subtitle_files(video_path: str) -> list[str]:
    """
    Find subtitle files alongside a video file.

    Searches for files with the same base name but subtitle extensions,
    and also any .srt/.ass files in the same directory.

    Parameters
    ----------
    video_path : str
        Path to the video file.

    Returns
    -------
    list[str]
        List of found subtitle file paths.
    """
    found = []
    video_dir = os.path.dirname(video_path)
    video_base = os.path.splitext(os.path.basename(video_path))[0]

    # Check for matching subtitle files (video_name.srt, video_name.en.srt, etc.)
    for ext in SUBTITLE_EXTENSIONS:
        # Exact match: video_name.srt
        exact = os.path.join(video_dir, f"{video_base}{ext}")
        if os.path.exists(exact):
            found.append(exact)

        # Language variants: video_name.en.srt, video_name.id.srt
        pattern = os.path.join(video_dir, f"{video_base}.*{ext}")
        for match in glob.glob(pattern):
            if match not in found:
                found.append(match)

    return sorted(found)


def read_subtitles(subtitle_path: str) -> list[dict]:
    """
    Read a subtitle file and extract text with timing.

    Parameters
    ----------
    subtitle_path : str
        Path to the subtitle file.

    Returns
    -------
    list[dict]
        List of subtitle entries:
        {
            "index": 1,
            "start_ms": 0,
            "end_ms": 5000,
            "start_seconds": 0.0,
            "end_seconds": 5.0,
            "text": "Hello world",
            "style": "Default",  # For ASS/SSA files
        }
    """
    try:
        subs = pysubs2.load(subtitle_path, encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback encodings common for non-English subtitles
        for enc in ["utf-8-sig", "latin-1", "cp1252", "shift_jis"]:
            try:
                subs = pysubs2.load(subtitle_path, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"   ⚠️ Could not decode subtitle file: {subtitle_path}")
            return []

    entries = []
    for i, line in enumerate(subs):
        # Skip comments and drawing events
        if line.is_comment or line.is_drawing:
            continue

        # Clean text (remove ASS formatting tags like {\b1}, {\an8}, etc.)
        clean_text = line.plaintext.strip()
        if not clean_text:
            continue

        entries.append({
            "index": i + 1,
            "start_ms": line.start,
            "end_ms": line.end,
            "start_seconds": line.start / 1000.0,
            "end_seconds": line.end / 1000.0,
            "text": clean_text,
            "style": line.style if hasattr(line, "style") else "Default",
        })

    return entries


def subtitles_to_context_string(entries: list[dict]) -> str:
    """
    Convert subtitle entries to a plain text context string.

    This is useful for feeding into the vision AI as additional context
    to help understand what's happening in the video.

    Parameters
    ----------
    entries : list[dict]
        Subtitle entries from read_subtitles().

    Returns
    -------
    str
        Formatted context string with timestamps.
    """
    lines = []
    for entry in entries:
        start = entry["start_seconds"]
        end = entry["end_seconds"]
        text = entry["text"]
        lines.append(f"[{start:.1f}s - {end:.1f}s] {text}")

    return "\n".join(lines)


def find_and_read_subtitles(
    video_path: str,
    project_dir: str,
) -> dict | None:
    """
    Find and read subtitles for a video, save parsed data.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    project_dir : str
        Project output directory.

    Returns
    -------
    dict or None
        {
            "source_files": [...],
            "entries": [...],
            "context_string": "...",
            "total_lines": 42,
        }
        or None if no subtitles found.
    """
    subtitle_files = find_subtitle_files(video_path)

    if not subtitle_files:
        print("   📝 No subtitle files found alongside video.")
        return None

    print(f"   📝 Found {len(subtitle_files)} subtitle file(s):")
    for sf in subtitle_files:
        print(f"      → {os.path.basename(sf)}")

    # Read the first (primary) subtitle file
    # TODO: Support merging multiple subtitle tracks
    primary = subtitle_files[0]
    entries = read_subtitles(primary)

    if not entries:
        print(f"   ⚠️ Could not parse subtitles from {primary}")
        return None

    context_string = subtitles_to_context_string(entries)

    result = {
        "source_files": subtitle_files,
        "primary_file": primary,
        "entries": entries,
        "context_string": context_string,
        "total_lines": len(entries),
    }

    # Save parsed subtitles
    import json
    subs_path = os.path.join(project_dir, "parsed_subtitles.json")
    with open(subs_path, "w", encoding="utf-8") as f:
        # Don't save full context_string in JSON (can be huge), just reference
        save_data = {
            "source_files": subtitle_files,
            "primary_file": primary,
            "total_lines": len(entries),
            "entries": entries,
        }
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"   ✅ Parsed {len(entries)} subtitle lines from {os.path.basename(primary)}")

    return result
