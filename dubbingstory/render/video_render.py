"""Final assembly on the video timeline, with a single optional video encode."""
from __future__ import annotations

import glob
import json
import math
import os

from dubbingstory.ingest.local_video import get_video_metadata
from dubbingstory.render.audio_mix import _run_ffmpeg, build_audio_filter
from dubbingstory.render.subtitle_burn import build_subtitle_filter

RATIO_DIMS = {"16:9": (1920, 1080), "9:16": (1080, 1920),
              "1:1": (1080, 1080), "4:3": (1440, 1080)}
_MP4_COPY_CODECS = {"h264", "hevc", "av1", "mpeg4", "vp9"}


def _strict_output(cfg, target_ratio=None) -> bool:
    return (target_ratio == "source"
            or str(getattr(cfg, "tts_narration_fit", "legacy")) != "legacy"
            or str(getattr(cfg, "subtitle_style", "classic")) in {"word", "karaoke"})


def _narration_text(script_path: str) -> str:
    if not os.path.isfile(script_path):
        raise FileNotFoundError("Structured narration is missing; run narrate and dub before animated rendering")
    with open(script_path, encoding="utf-8") as source:
        script = json.load(source)
    segments = script.get("segments", []) if isinstance(script, dict) else script
    text = "\n".join(str(item.get("text", "")).strip() for item in segments if str(item.get("text", "")).strip())
    if not text:
        raise ValueError("The structured narration contains no spoken text")
    return text


def _validate_tts_manifest(audio_path: str, video_duration: float, script_path: str | None, source_video: str | None = None) -> dict:
    """Strict renders must not reuse audio left over from a failed synthesis."""
    from dubbingstory.tts.word_alignment import file_sha256, text_sha256
    manifest_path = os.path.splitext(audio_path)[0] + ".manifest.json"
    if not os.path.isfile(manifest_path):
        raise ValueError("Native/animated rendering requires a completed TTS manifest; run dub with --narration-fit rewrite")
    with open(manifest_path, encoding="utf-8") as source:
        manifest = json.load(source)
    if manifest.get("status") != "complete" or manifest.get("audio_sha256") != file_sha256(audio_path):
        raise ValueError("Narration audio is incomplete or changed; regenerate dub before rendering")
    if (manifest.get("schema_version") != 1 or not manifest.get("strict_timing_validated")
            or not manifest.get("complete_coverage") or manifest.get("failed_segment_ids")
            or manifest.get("fit_policy") not in {"rewrite", "error"}):
        raise ValueError("Narration has not passed strict per-beat timing validation; run dub with --narration-fit rewrite or error")
    placements = manifest.get("placements") or []
    expected = manifest.get("expected_segment_ids") or []
    if not expected or len(set(expected)) != len(expected) or [item.get("scene_id") for item in placements] != expected:
        raise ValueError("Narration coverage is incomplete; regenerate all required beats")
    previous_end = 0.0
    for item in placements:
        try:
            start, end, planned_start, planned_end = (
                float(item[key]) for key in ("actual_start", "actual_end", "planned_start", "planned_end"))
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("Narration placement is invalid; regenerate dub") from exc
        if (not all(math.isfinite(value) for value in (start, end, planned_start, planned_end))
                or start < previous_end - 0.001 or start < max(0, planned_start) - 0.001
                or end <= start or end > min(planned_end, video_duration) + 0.001):
            raise ValueError("Narration exceeds a visual beat window; regenerate dub with strict fitting")
        previous_end = end
    if abs(float(manifest.get("timeline_duration", 0)) - video_duration) > 0.02:
        raise ValueError("Narration was generated for a different video duration; regenerate dub")
    if script_path and manifest.get("text_sha256") != text_sha256(_narration_text(script_path)):
        raise ValueError("Narration text changed since synthesis; regenerate dub")
    if float(manifest.get("overflow_seconds", 0)) > 0.001:
        raise ValueError("Narration overflows the video; regenerate dub with --narration-fit rewrite")
    if source_video and manifest.get("source_sha256") != file_sha256(source_video):
        raise ValueError("Source video changed since synthesis; regenerate narration and dub")
    return manifest


def _prepare_subtitle(subtitle_path, narration_audio, width, height, cfg):
    style = str(getattr(cfg, "subtitle_style", "classic"))
    if style not in {"word", "karaoke"}:
        return subtitle_path
    if not subtitle_path:
        raise ValueError("Animated captions require a matching script/SRT path and measured word timings")
    from dubbingstory.story.animated_subtitles import generate_ass
    from dubbingstory.tts.word_alignment import validate_word_payload
    stem = os.path.splitext(subtitle_path)[0]
    words_path = stem + ".words.json"
    if not os.path.isfile(words_path):
        raise ValueError("Word timings are missing; run dub with --subtitle-style word or karaoke (no estimated timings are generated)")
    with open(words_path, encoding="utf-8") as source:
        payload = json.load(source)
    validate_word_payload(payload, narration_audio, _narration_text(stem + ".json"))
    # Per-resolution artifacts keep multiple output layouts independent.
    ass_path = stem + f".{width}x{height}.{style}.ass"
    generate_ass(payload, ass_path, width, height, cfg, style=style)
    with open(ass_path, encoding="utf-8") as source:
        text = source.read()
    with open(stem + ".ass", "w", encoding="utf-8") as dest:
        dest.write(text)
    return ass_path


def build_render_command(
    source_video, narration_audio, output_path, metadata, *, subtitle_path=None,
    audio_strategy="mute_original", original_volume=0.15, burn_subs=False,
    target_ratio=None, cfg=None,
) -> list[str]:
    """Pure command builder, suitable for offline command-contract tests."""
    duration = float(metadata.get("video_duration") or metadata.get("duration") or 0)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Cannot render without a measured positive video duration")
    if target_ratio not in {None, "source", *RATIO_DIMS}:
        raise ValueError(f"Unsupported output ratio: {target_ratio}")
    graph = build_audio_filter(
        duration=duration, has_original_audio=bool(metadata.get("has_audio")),
        strategy=audio_strategy, original_volume=float(original_volume),
        original_start_offset=float(metadata.get("audio_start_time", 0)) - float(metadata.get("video_start_time", 0)),
        narration_volume=float(getattr(cfg, "render_narration_volume", 1.0)),
        duck_threshold=float(getattr(cfg, "render_duck_threshold", 0.025)),
        duck_ratio=float(getattr(cfg, "render_duck_ratio", 8.0)),
        duck_attack_ms=float(getattr(cfg, "render_duck_attack_ms", 20.0)),
        duck_release_ms=float(getattr(cfg, "render_duck_release_ms", 350.0)),
    )
    visual_filters = []
    if target_ratio in RATIO_DIMS:
        width, height = RATIO_DIMS[target_ratio]
        visual_filters.extend([
            "scale=iw*sar:ih,setsar=1",
            f"scale={width}:{height}:force_original_aspect_ratio=decrease",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
        ])
    if burn_subs:
        if not subtitle_path:
            raise ValueError("Subtitle burn was requested but the matching subtitle is missing")
        animated = str(getattr(cfg, "subtitle_style", "classic")) in {"word", "karaoke"}
        setting_prefix = "subtitle_" if animated else "subtitle_classic_"
        font_path = getattr(cfg, setting_prefix + "font_path", None)
        fonts_dir = os.path.dirname(os.path.abspath(font_path)) if font_path else None
        if animated:
            from dubbingstory.story.animated_subtitles import caption_font_directory
            fonts_dir = caption_font_directory(cfg)
        original_size = RATIO_DIMS.get(target_ratio, (
            int(metadata.get("display_width", 0)), int(metadata.get("display_height", 0))))
        visual_filters.append(build_subtitle_filter(
            subtitle_path, fonts_dir=fonts_dir,
            original_size=original_size if animated else None,
            font_family=str(getattr(cfg, setting_prefix + "font_name", "Arial")),
            font_size=int(getattr(cfg, setting_prefix + "font_size", 24)),
            font_color=str(getattr(cfg, setting_prefix + "font_color", "white")),
            outline_color=str(getattr(cfg, setting_prefix + "outline_color", "black")),
            outline_width=float(getattr(cfg, setting_prefix + "outline", 2)),
            position=str(getattr(cfg, setting_prefix + "position", "bottom")),
        ))
    encode = bool(visual_filters) or metadata.get("codec") not in _MP4_COPY_CODECS
    if encode and metadata.get("color_transfer") in {"smpte2084", "arib-std-b67"}:
        raise ValueError("HDR encoding needs an explicit color-management policy; use source stream copy without subtitle burn or provide an SDR source")
    cmd = ["ffmpeg", "-y", "-copyts", "-itsoffset", f"{-float(metadata.get('video_start_time', 0)):.9f}",
           "-i", source_video, "-i", narration_audio]
    stream_index = metadata.get("video_stream_index")
    video_selector = f"0:{stream_index}" if isinstance(stream_index, int) else "0:v:0"
    if encode:
        visual_filters.insert(0, "setpts=PTS-STARTPTS")
        graph += f";[{video_selector}]" + ",".join(visual_filters) + "[video]"
    cmd += ["-filter_complex", graph, "-map", "[video]" if encode else video_selector, "-map", "[out]"]
    if encode:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", str(getattr(cfg, "render_video_quality_crf", 20)),
                "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-metadata:s:v:0", "rotate=0"]
    else:
        cmd += ["-c:v", "copy"]
    # No -shortest and no video -t: retain all source frames. Audio alone is
    # padded/trimmed to the measured video timeline in the graph.
    cmd += ["-c:a", "aac", "-b:a", "192k", "-map_metadata", "0", "-avoid_negative_ts", "disabled",
            "-movflags", "+faststart", output_path]
    return cmd


def render_final(
    source_video: str, narration_audio: str, output_path: str,
    subtitle_path: str | None = None, audio_strategy: str = "mute_original",
    original_volume: float = 0.15, burn_subs: bool = False,
    target_ratio: str | None = None, cfg=None,
) -> str:
    metadata = get_video_metadata(source_video)
    if _strict_output(cfg, target_ratio):
        from dubbingstory.ingest.local_video import require_video_timeline
        metadata = require_video_timeline(source_video, metadata)
    duration = float(metadata.get("video_duration") or metadata.get("duration") or 0)
    if _strict_output(cfg, target_ratio):
        script_path = os.path.splitext(subtitle_path)[0] + ".json" if subtitle_path else None
        _validate_tts_manifest(narration_audio, duration, script_path, source_video)
    width, height = RATIO_DIMS.get(target_ratio, (int(metadata.get("display_width", 0)), int(metadata.get("display_height", 0))))
    subtitle_path = _prepare_subtitle(subtitle_path, narration_audio, width, height, cfg)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    partial_path = os.path.splitext(output_path)[0] + ".rendering.mp4"
    cmd = build_render_command(source_video, narration_audio, partial_path, metadata,
                               subtitle_path=subtitle_path, audio_strategy=audio_strategy,
                               original_volume=original_volume, burn_subs=burn_subs,
                               target_ratio=target_ratio, cfg=cfg)
    try:
        _run_ffmpeg(cmd, "final render; subtitle burn requires FFmpeg with libass")
        os.replace(partial_path, output_path)
    finally:
        if os.path.isfile(partial_path):
            os.remove(partial_path)
    return output_path


def render_project(project_dir: str, ratios: list[str] | None = None, cfg=None) -> list[str]:
    ratios = ratios if ratios is not None else getattr(cfg, "render_ratios", ["16:9"])
    source_video = os.path.join(project_dir, "source.mp4")
    if not os.path.isfile(source_video):
        raise FileNotFoundError(f"Source video not found: {source_video}")
    audio_files = sorted(glob.glob(os.path.join(project_dir, "audio", "audio_*.wav")))
    requested_languages = getattr(cfg, "requested_languages", None)
    if requested_languages is not None:
        audio_files = [path for path in audio_files
                       if os.path.basename(path)[len("audio_"):-len(".wav")] in requested_languages]
    strict = any(_strict_output(cfg, ratio) for ratio in ratios)
    if not audio_files:
        if strict:
            raise ValueError("No complete narration audio found; run dub first")
        print("   No audio files found. Run 'dubbingstory dub' first.")
        return []
    rendered, errors = [], []
    for audio_path in audio_files:
        lang = os.path.basename(audio_path)[len("audio_"):-len(".wav")]
        subtitle_path = os.path.join(project_dir, "scripts", f"script_{lang}.srt")
        for ratio in ratios:
            output_path = os.path.join(project_dir, f"final_{lang}_{ratio.replace(':', 'x')}.mp4")
            try:
                render_final(source_video, audio_path, output_path, subtitle_path,
                             audio_strategy=getattr(cfg, "render_audio_strategy", "mute_original"),
                             original_volume=float(getattr(cfg, "render_original_volume", 0.15)),
                             burn_subs=bool(getattr(cfg, "render_burn_subtitles", False)),
                             target_ratio=ratio if ratio != "16:9" else None, cfg=cfg)
                rendered.append(output_path)
                print(f"   Rendered: {output_path}")
            except Exception as exc:
                errors.append({"language": lang, "ratio": ratio, "error": str(exc)})
                print(f"   Render failed: {exc}")
    with open(os.path.join(project_dir, "render_manifest.json"), "w", encoding="utf-8") as dest:
        json.dump({"rendered_videos": rendered, "ratios": ratios,
                   "audio_strategy": getattr(cfg, "render_audio_strategy", "mute_original"),
                   "status": "failed" if errors else "complete", "errors": errors}, dest, ensure_ascii=False, indent=2)
    if errors and strict:
        raise RuntimeError(f"{len(errors)} render output(s) failed; inspect render_manifest.json")
    return rendered
