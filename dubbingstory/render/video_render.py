"""Final assembly on the video timeline, with a single optional video encode."""
from __future__ import annotations

import glob
import json
import math
import os

from dubbingstory.ingest.local_video import get_video_metadata
from dubbingstory.render.audio_mix import _run_ffmpeg, build_audio_filter
from dubbingstory.render.subtitle_burn import build_subtitle_filter
from dubbingstory.render.vertical_layout import build_layered_video_filter, layout_settings

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
    """Build one encode/mux command. None narration preserves an existing mix."""
    duration = float(metadata.get("video_duration") or metadata.get("duration") or 0)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Cannot render without a measured positive video duration")
    if target_ratio not in {None, "source", *RATIO_DIMS}:
        raise ValueError(f"Unsupported output ratio: {target_ratio}")
    layout = layout_settings(cfg)
    layered = target_ratio == "9:16" and layout["layout"] == "layered"
    graph = build_audio_filter(
        duration=duration, has_original_audio=bool(metadata.get("has_audio")),
        strategy=audio_strategy, original_volume=float(original_volume),
        original_start_offset=float(metadata.get("audio_start_time", 0)) - float(metadata.get("video_start_time", 0)),
        narration_volume=float(getattr(cfg, "render_narration_volume", 1.0)),
        duck_threshold=float(getattr(cfg, "render_duck_threshold", 0.025)),
        duck_ratio=float(getattr(cfg, "render_duck_ratio", 8.0)),
        duck_attack_ms=float(getattr(cfg, "render_duck_attack_ms", 20.0)),
        duck_release_ms=float(getattr(cfg, "render_duck_release_ms", 350.0)),
    ) if narration_audio is not None else ""
    visual_filters = []
    if target_ratio in RATIO_DIMS and not layered:
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
    encode = layered or bool(visual_filters) or metadata.get("codec") not in _MP4_COPY_CODECS
    if encode and metadata.get("color_transfer") in {"smpte2084", "arib-std-b67"}:
        raise ValueError("HDR encoding needs an explicit color-management policy; use source stream copy without subtitle burn or provide an SDR source")
    cmd = ["ffmpeg", "-y", "-copyts", "-itsoffset", f"{-float(metadata.get('video_start_time', 0)):.9f}",
           "-i", source_video]
    if narration_audio is not None:
        cmd += ["-i", narration_audio]
    stream_index = metadata.get("video_stream_index")
    video_selector = f"0:{stream_index}" if isinstance(stream_index, int) else "0:v:0"
    if encode:
        if layered:
            width, height = RATIO_DIMS["9:16"]
            visual_graph = build_layered_video_filter(video_selector, metadata, width, height, cfg)
            # Subtitles use the final canvas: never zoomed, duplicated or blurred.
            visual_graph += ";[vertical_composite]" + (",".join(visual_filters) or "null") + "[video]"
        else:
            visual_filters.insert(0, "setpts=PTS-STARTPTS")
            visual_graph = f"[{video_selector}]" + ",".join(visual_filters) + "[video]"
        graph = ";".join(part for part in (graph, visual_graph) if part)
    if graph:
        cmd += ["-filter_complex", graph]
    cmd += ["-map", "[video]" if encode else video_selector]
    if narration_audio is not None:
        cmd += ["-map", "[out]"]
    elif metadata.get("has_audio"):
        # Reframe already-mixed exports once; no new narration or doubled audio.
        cmd += ["-map", "0:a:0"]
    if encode:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", str(getattr(cfg, "render_video_quality_crf", 20)),
                "-pix_fmt", "yuv420p", "-vsync", "0", "-metadata:s:v:0", "rotate=0"]
    else:
        cmd += ["-c:v", "copy"]
    # No -shortest and no video -t: retain all source frames. Audio alone is
    # padded/trimmed to the measured video timeline in the graph.
    if narration_audio is None and metadata.get("audio_codec") in {"aac", "mp3", "alac", "ac3", "eac3"}:
        cmd += ["-c:a", "copy"]
    elif narration_audio is not None or metadata.get("has_audio"):
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd += ["-map_metadata", "0", "-avoid_negative_ts", "disabled",
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
        label = "final render; subtitle burn requires FFmpeg with libass" if burn_subs else "final render"
        _run_ffmpeg(cmd, label)
        os.replace(partial_path, output_path)
    finally:
        if os.path.isfile(partial_path):
            os.remove(partial_path)
    return output_path


def reframe_video(source_video: str, output_path: str, *, target_ratio: str = "9:16", cfg=None) -> str:
    """Recompose an existing MP4/video without invoking analysis, TTS or mixing.

    Use a new output path to preserve the existing export. A completed MP4 is
    published atomically; a small adjacent manifest records the layout recipe.
    Existing burnt-in text is part of the footage and will be transformed too.
    """
    if not os.path.isfile(source_video):
        raise FileNotFoundError(f"Source video not found: {source_video}")
    if target_ratio != "9:16":
        raise ValueError("reframe currently supports the 9:16 canvas")
    if os.path.splitext(output_path)[1].lower() != ".mp4":
        raise ValueError("Reframe output must be an .mp4 file")
    if os.path.normcase(os.path.realpath(source_video)) == os.path.normcase(os.path.realpath(output_path)):
        raise ValueError("Reframe requires a different output path; the source must be preserved")
    manifest_path = output_path + ".render.json"
    if os.path.exists(output_path) or os.path.exists(manifest_path):
        raise FileExistsError("Reframe output already exists; choose a new --output path")
    layout = layout_settings(cfg)
    metadata = get_video_metadata(source_video)
    if int(metadata.get("display_width", 0)) <= 0 or int(metadata.get("display_height", 0)) <= 0:
        raise ValueError("Cannot reframe a source without a valid video stream")
    # tempfile keeps failed/interrupted work separate from both source and export.
    import tempfile
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reframe-", dir=output_dir) as staging:
        partial_path = os.path.join(staging, "output.mp4")
        cmd = build_render_command(source_video, None, partial_path, metadata,
                                   target_ratio=target_ratio, cfg=cfg)
        _run_ffmpeg(cmd, "vertical reframe")
        manifest = {
            "schema_version": 1, "status": "complete", "operation": "reframe",
            "source_video": os.path.abspath(source_video), "output_video": os.path.abspath(output_path),
            "ratio": target_ratio, "layout": layout, "audio": "existing_mix",
            "video_duration": metadata.get("video_duration"),
        }
        staged_manifest = os.path.join(staging, "render.json")
        with open(staged_manifest, "w", encoding="utf-8") as dest:
            json.dump(manifest, dest, ensure_ascii=False, indent=2)
        # Re-check after rendering in case another job used the selected name.
        if os.path.exists(output_path) or os.path.exists(manifest_path):
            raise FileExistsError("Reframe output appeared during rendering; choose a new --output path")
        os.replace(partial_path, output_path)
        os.replace(staged_manifest, manifest_path)
    return output_path


def _summary_source_for_audio(project_dir: str, source_video: str, audio_path: str, lang: str) -> str:
    """Use a legacy per-language recut only when its TTS manifest requests it."""
    manifest_path = os.path.splitext(audio_path)[0] + ".manifest.json"
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as source:
                manifest = json.load(source)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read summary TTS manifest: {manifest_path}") from exc
        if not isinstance(manifest, dict):
            raise ValueError(f"Invalid summary TTS manifest: {manifest_path}")
        language_source = os.path.join(project_dir, f"summary_source_{lang}.mp4")
        if manifest.get("timeline_scheduled") is False and os.path.isfile(language_source):
            return language_source
    return source_video


def final_video_filename(lang: str, ratio: str, *, mode: str = "full", cfg=None) -> str:
    """Use the same canvas label for pipeline and saved-artifact renders."""
    prefix = "final_summary_" if mode == "summary" else "final_"
    canvas = "_canvas" if ratio == "9:16" and getattr(cfg, "render_vertical_layout", "pad") == "layered" else ""
    return f"{prefix}{lang}{canvas}_{ratio.replace(':', 'x')}.mp4"


def render_project(project_dir: str, ratios: list[str] | None = None, cfg=None, mode: str = "full") -> list[str]:
    """Render saved full or summary artifacts without repeating AI/TTS stages."""
    if mode not in {"full", "summary"}:
        raise ValueError("Render mode must be full or summary")
    ratios = ratios if ratios is not None else getattr(cfg, "render_ratios", ["16:9"])
    summary = mode == "summary"
    source_video = os.path.join(project_dir, "summary_source.mp4" if summary else "source.mp4")
    if not os.path.isfile(source_video):
        raise FileNotFoundError(f"Source video not found: {source_video}")
    audio_prefix = "summary_audio_" if summary else "audio_"
    script_prefix = "summary_script_" if summary else "script_"
    manifest_name = "summary_render_manifest.json" if summary else "render_manifest.json"
    audio_files = sorted(glob.glob(os.path.join(project_dir, "audio", f"{audio_prefix}*.wav")))
    requested_languages = getattr(cfg, "requested_languages", None)
    if requested_languages is not None:
        audio_files = [path for path in audio_files
                       if os.path.basename(path)[len(audio_prefix):-len(".wav")] in requested_languages]
    layout = layout_settings(cfg)
    strict = (any(_strict_output(cfg, ratio) for ratio in ratios)
              or (layout["layout"] == "layered" and "9:16" in ratios))
    if not audio_files:
        if summary:
            raise ValueError("No saved summary narration found (audio/summary_audio_*.wav). Restore the summary artifacts or complete summary synthesis before render --mode summary.")
        if strict:
            raise ValueError("No complete narration audio found; run dub first")
        print("   No audio files found. Run 'dubbingstory dub' first.")
        return []
    rendered, errors = [], []
    for audio_path in audio_files:
        lang = os.path.basename(audio_path)[len(audio_prefix):-len(".wav")]
        subtitle_path = os.path.join(project_dir, "scripts", f"{script_prefix}{lang}.srt")
        for ratio in ratios:
            output_path = os.path.join(project_dir, final_video_filename(lang, ratio, mode=mode, cfg=cfg))
            try:
                language_source = _summary_source_for_audio(project_dir, source_video, audio_path, lang) if summary else source_video
                render_final(language_source, audio_path, output_path, subtitle_path,
                             audio_strategy=getattr(cfg, "render_audio_strategy", "mute_original"),
                             original_volume=float(getattr(cfg, "render_original_volume", 0.15)),
                             burn_subs=bool(getattr(cfg, "render_burn_subtitles", False)),
                             target_ratio=ratio if ratio != "16:9" else None, cfg=cfg)
                rendered.append(output_path)
                print(f"   Rendered: {output_path}")
            except Exception as exc:
                errors.append({"language": lang, "ratio": ratio, "error": str(exc)})
                print(f"   Render failed: {exc}")
    with open(os.path.join(project_dir, manifest_name), "w", encoding="utf-8") as dest:
        json.dump({"rendered_videos": rendered, "ratios": ratios, "mode": mode,
                   "vertical_layout": layout,
                   "audio_strategy": getattr(cfg, "render_audio_strategy", "mute_original"),
                   "status": "failed" if errors else "complete", "errors": errors}, dest, ensure_ascii=False, indent=2)
    if errors and (strict or summary):
        raise RuntimeError(f"{len(errors)} render output(s) failed; inspect {manifest_name}")
    return rendered
