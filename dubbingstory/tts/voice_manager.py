"""
dubbingstory.tts.voice_manager — Voice selection & timeline-aware audio generation.

DubbingStory v2 keeps TTS synthesis and timeline placement separate.  A short
sentence is no longer padded to the full duration of its source scene.  Instead
all speech clips are placed on one full-length narration timeline, preserving
intentional breathing room while avoiding the large dead-air pattern caused by
per-scene ``apad``.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import shutil
import subprocess
from typing import Any

from dubbingstory.tts.base_engine import BaseTTSEngine


def _voice_overrides(name: str, cfg) -> dict[str, str]:
    """Resolve CLI/global voice overrides before engine-specific defaults."""
    if not cfg:
        return {}

    if name == "chirp":
        default_id = getattr(cfg, "tts_chirp_voices_id", None)
        default_en = getattr(cfg, "tts_chirp_voices_en", None)
    elif name == "azure":
        default_id = getattr(cfg, "tts_azure_voices_id", None)
        default_en = getattr(cfg, "tts_azure_voices_en", None)
    elif name == "piper":
        default_id = getattr(cfg, "tts_voices_id", None)
        default_en = getattr(cfg, "tts_voices_en", None)
    else:
        default_id = getattr(cfg, "tts_edge_voices_id", None)
        default_en = getattr(cfg, "tts_edge_voices_en", None)

    # Explicit --voice-id/--voice-en settings are flattened into these generic
    # config attributes by cli.py and take precedence over engine defaults.
    voice_id = getattr(cfg, "tts_voice_id", None) or default_id
    voice_en = getattr(cfg, "tts_voice_en", None) or default_en
    result: dict[str, str] = {}
    if voice_id:
        result["id"] = str(voice_id)
    if voice_en:
        result["en"] = str(voice_en)
    return result


def get_engine(name: str = "edge", cfg=None) -> BaseTTSEngine:
    """Return a configured TTS engine instance."""
    name = (name or "edge").strip().lower()
    custom_voices = _voice_overrides(name, cfg)

    if name == "edge":
        from dubbingstory.tts.edge_tts_engine import EdgeTTSEngine
        return EdgeTTSEngine(custom_voices=custom_voices or None)
    if name == "piper":
        from dubbingstory.tts.piper_tts_engine import PiperTTSEngine
        return PiperTTSEngine(custom_voices=custom_voices or None)
    if name == "chirp":
        from dubbingstory.tts.google_chirp_tts_engine import GoogleChirpTTSEngine
        endpoint = getattr(cfg, "tts_chirp_api_endpoint", None) if cfg else None
        return GoogleChirpTTSEngine(
            custom_voices=custom_voices or None,
            api_endpoint=endpoint or None,
        )
    if name == "azure":
        from dubbingstory.tts.azure_tts_engine import AzureTTSEngine
        return AzureTTSEngine(
            subscription_key=getattr(cfg, "azure_speech_key", "") if cfg else "",
            region=getattr(cfg, "azure_speech_region", "") if cfg else "",
            endpoint=getattr(cfg, "tts_azure_endpoint", "") if cfg else "",
            custom_voices=custom_voices or None,
            output_format=getattr(cfg, "tts_azure_output_format", "riff-24khz-16bit-mono-pcm") if cfg else "riff-24khz-16bit-mono-pcm",
            timeout_seconds=float(getattr(cfg, "tts_azure_timeout_seconds", 60) if cfg else 60),
            rate_limit_per_minute=int(getattr(cfg, "tts_azure_rate_limit_per_minute", 20) if cfg else 20),
            max_retries=int(getattr(cfg, "tts_azure_max_retries", 4) if cfg else 4),
        )

    print(f"⚠️ Unknown TTS engine '{name}', falling back to Edge-TTS.")
    from dubbingstory.tts.edge_tts_engine import EdgeTTSEngine
    return EdgeTTSEngine(custom_voices=_voice_overrides("edge", cfg) or None)


def _style_speaking_rate_multiplier(cfg) -> float:
    if not cfg:
        return 1.0
    style = str(getattr(cfg, "narration_style", "") or "").strip()
    styles = getattr(cfg, "narration_styles", {}) or {}
    style_cfg = styles.get(style, {}) if isinstance(styles, dict) else {}
    tts_cfg = style_cfg.get("tts", {}) if isinstance(style_cfg, dict) else {}
    try:
        return max(0.85, min(1.15, float(tts_cfg.get("speaking_rate_multiplier", 1.0) or 1.0)))
    except (TypeError, ValueError):
        return 1.0


def generate_all_audio(
    scripts_dir: str,
    audio_dir: str,
    engine_name: str = "edge",
    cfg=None,
    script_prefix: str = "script_",
    audio_prefix: str = "audio_",
    segment_durations: dict[str, float] | None = None,
    timeline_duration: float | None = None,
    timeline_kind: str = "source",
    timeline_source_path: str | None = None,
) -> dict[str, dict]:
    """Generate TTS for every language script.

    V2 behavior
    -----------
    If ``{script_prefix}{lang}.json`` exists and contains timeline information,
    a timeline scheduler is used. Each TTS clip is measured and placed on a
    full-length silent canvas. Strict fitting rewrites or rejects speech that
    exceeds its visual window; only legacy fitting permits bounded speedup.
    Short speech clips are *not* padded to the duration of their scene.

    Legacy behavior
    ---------------
    Plain ``.txt`` scripts still work.  ``segment_durations`` is retained for
    backward compatibility with older callers.
    """
    engine = get_engine(engine_name, cfg)
    base_speaking_rate = float(getattr(cfg, "tts_speaking_rate", 1.0) if cfg else 1.0)
    speaking_rate = max(0.75, min(1.25, base_speaking_rate * _style_speaking_rate_multiplier(cfg)))
    use_scheduler = bool(getattr(cfg, "tts_use_timeline_scheduler", True) if cfg else True)

    os.makedirs(audio_dir, exist_ok=True)
    fail_log_path = os.path.join(os.path.dirname(audio_dir), "failed_tts.log")
    if os.path.exists(fail_log_path):
        os.remove(fail_log_path)

    script_files = sorted(glob.glob(os.path.join(scripts_dir, f"{script_prefix}*.txt")))
    requested_languages = getattr(cfg, "requested_languages", None)
    if requested_languages is not None:
        requested_languages = set(requested_languages)
        script_files = [path for path in script_files
                        if os.path.basename(path)[len(script_prefix):-len(".txt")] in requested_languages]
    if not script_files:
        print(f"   ⚠️ No script files found matching '{script_prefix}*.txt' in scripts directory.")
        return {}

    results: dict[str, dict] = {}

    for script_path in script_files:
        basename = os.path.basename(script_path)
        lang = basename.replace(script_prefix, "").replace(".txt", "")
        print(f"\n   🎙️ Generating {lang.upper()} audio with {engine.get_engine_name()}...")

        structured_path = os.path.join(scripts_dir, f"{script_prefix}{lang}.json")
        structured = _load_structured_script(structured_path) if os.path.exists(structured_path) else None

        animated = str(getattr(cfg, "subtitle_style", "classic")) in {"word", "karaoke"}
        strict = str(getattr(cfg, "tts_narration_fit", "legacy")) != "legacy" or animated
        output_audio_path = os.path.join(audio_dir, f"{audio_prefix}{lang}.wav")
        manifest_path = os.path.join(audio_dir, f"{audio_prefix}{lang}.manifest.json")
        # Invalidate completion proof for every attempt, including legacy runs.
        # A failed provider call must not leave yesterday's complete manifest.
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump({"schema_version": 1, "status": "failed", "language": lang,
                       "strict_timing_validated": False}, handle, indent=2)
        # Invalidate derived outputs before attempting synthesis. If a provider
        # fails, a later render must not silently reuse yesterday's voice track.
        if strict:
            for stale_path in [output_audio_path,
                               os.path.join(scripts_dir, f"{script_prefix}{lang}.words.json"),
                               *glob.glob(os.path.join(scripts_dir, f"{script_prefix}{lang}*.ass"))]:
                if os.path.isfile(stale_path):
                    os.remove(stale_path)
        if strict and (not structured or not _has_timeline(structured["segments"])):
            raise ValueError("Native fitting/animated captions require structured narration with timeline windows; run narrate first")
        if structured and (use_scheduler or strict) and _has_timeline(structured["segments"]):
            evidence_path = os.path.join(os.path.dirname(scripts_dir), "storyboard.json")
            evidence = {}
            if os.path.exists(evidence_path):
                with open(evidence_path, encoding="utf-8") as handle:
                    evidence = json.load(handle)
            result = _generate_timeline_audio(
                engine=engine,
                language=lang,
                segments=structured["segments"],
                total_duration=timeline_duration if timeline_duration is not None else structured.get("total_duration"),
                output_path=output_audio_path,
                audio_dir=os.path.join(audio_dir, lang),
                speaking_rate=speaking_rate,
                cfg=cfg,
                fail_log_path=fail_log_path,
                evidence=evidence.get("scenes", []),
            )
            if result:
                from dubbingstory.tts.word_alignment import build_word_payload, file_sha256, text_sha256
                final_text = "\n".join(item["text"] for item in result["placements"])
                # Validate the complete, rendered waveform before publishing
                # rewritten scripts or a successful synthesis manifest.
                payload = None
                if animated:
                    payload = build_word_payload(
                        result["word_timings"], result["concat_path"], final_text,
                        lang, placements=result["placements"],
                    )
                    payload.update(timeline_kind=timeline_kind, total_duration=result["timeline_duration"])
                if result["complete_coverage"]:
                    structured["segments"] = result["final_segments"]
                    structured["total_duration"] = result["timeline_duration"]
                    structured["timeline_kind"] = timeline_kind
                    with open(structured_path, "w", encoding="utf-8") as handle:
                        json.dump(structured, handle, ensure_ascii=False, indent=2)
                    with open(script_path, "w", encoding="utf-8") as handle:
                        for segment in result["final_segments"]:
                            handle.write(f"[{segment['scene_id']}] {segment['text']}\n\n")
                result["timeline_kind"] = timeline_kind
                result["language"] = lang
                result["engine"] = engine_name
                result["speaking_rate"] = speaking_rate
                result["status"] = "complete" if result["complete_coverage"] else "partial"
                result["audio_sha256"] = file_sha256(result["concat_path"])
                result["text_sha256"] = text_sha256(final_text)
                if timeline_source_path and os.path.isfile(timeline_source_path):
                    result["source_sha256"] = file_sha256(timeline_source_path)
                elif strict and timeline_source_path:
                    raise ValueError("Source video disappeared during synthesis; regenerate dub with the source available")
                if animated:
                    words_path = os.path.join(scripts_dir, f"{script_prefix}{lang}.words.json")
                    with open(words_path, "w", encoding="utf-8") as handle:
                        json.dump(payload, handle, ensure_ascii=False, indent=2)
                    result["word_timings_path"] = words_path
                else:
                    # A new synthesis invalidates previously generated captions.
                    for suffix in (".words.json", ".ass"):
                        stale_path = os.path.join(scripts_dir, f"{script_prefix}{lang}{suffix}")
                        if os.path.isfile(stale_path):
                            os.remove(stale_path)
                result["manifest_path"] = manifest_path
                with open(manifest_path, "w", encoding="utf-8") as handle:
                    json.dump({"schema_version": 1, **result}, handle, ensure_ascii=False, indent=2)
                results[lang] = result
                marker = "✅" if result["status"] == "complete" else "⚠️ Partial"
                print(
                    f"   {marker} {lang.upper()} timeline audio: {result['concat_path']} "
                    f"(speech={result.get('speech_duration', 0):.1f}s / "
                    f"timeline={result.get('timeline_duration', 0):.1f}s)"
                )
            continue

        # Legacy plain-text parser.
        with open(script_path, "r", encoding="utf-8") as handle:
            segments = _parse_text_script(handle.read())
        if not segments:
            print(f"   ⚠️ No segments found in {basename}")
            continue

        lang_audio_dir = os.path.join(audio_dir, lang)
        os.makedirs(lang_audio_dir, exist_ok=True)
        segment_paths: list[str] = []
        final_durations: dict[str, float] = {}
        failed_segment_ids: list[str] = []
        spoken_text: list[str] = []

        for seg in segments:
            scene_id = seg["scene_id"]
            seg_audio = os.path.join(lang_audio_dir, f"{scene_id}.wav")
            try:
                engine.synthesize(
                    text=seg["text"],
                    output_path=seg_audio,
                    language=lang,
                    speaking_rate=speaking_rate,
                )
                target_duration = (segment_durations or {}).get(scene_id)
                if target_duration and target_duration > 0:
                    aligned_audio = os.path.join(lang_audio_dir, f"{scene_id}_aligned.wav")
                    final_dur = _align_audio_duration(
                        seg_audio,
                        aligned_audio,
                        target_duration,
                        max_speedup=float(getattr(cfg, "tts_max_speedup", 1.12) if cfg else 1.12),
                        pad_short=False,
                    )
                    segment_paths.append(aligned_audio)
                    final_durations[scene_id] = final_dur
                else:
                    segment_paths.append(seg_audio)
                    final_durations[scene_id] = _get_audio_duration(seg_audio)
                spoken_text.append(seg["text"])
                print(f"      ✅ {scene_id}: {len(seg['text'].split())} words")
            except Exception as exc:
                failed_segment_ids.append(scene_id)
                _log_tts_failure(fail_log_path, lang, scene_id, exc)
                print(f"      ❌ {scene_id}: {exc}")

        if segment_paths:
            from dubbingstory.tts.word_alignment import file_sha256, text_sha256
            concat_path = os.path.join(audio_dir, f"{audio_prefix}{lang}.wav")
            _concatenate_audio(segment_paths, concat_path)
            result = {
                "concat_path": concat_path,
                "durations": final_durations,
                "timeline_scheduled": False,
                "strict_timing_validated": False,
                "fit_policy": "legacy",
                "complete_coverage": not failed_segment_ids,
                "failed_segment_ids": failed_segment_ids,
                "status": "partial" if failed_segment_ids else "complete",
                "language": lang,
                "engine": engine_name,
                "audio_sha256": file_sha256(concat_path),
                "text_sha256": text_sha256("\n".join(spoken_text)),
                "manifest_path": manifest_path,
            }
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump({"schema_version": 1, **result}, handle, ensure_ascii=False, indent=2)
            results[lang] = result
            marker = "✅" if result["status"] == "complete" else "⚠️ Partial"
            print(f"   {marker} {lang.upper()} audio: {concat_path}")

    return results


def _load_structured_script(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"   ⚠️ Could not read structured narration {path}: {exc}")
        return None

    if isinstance(payload, list):
        return {"segments": payload, "total_duration": None}
    if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
        return payload
    print(f"   ⚠️ Structured narration has unsupported shape: {path}")
    return None


def _has_timeline(segments: list[dict]) -> bool:
    return bool(segments) and all(
        isinstance(item, dict)
        and item.get("timeline_start", item.get("start_time")) is not None
        and item.get("timeline_end", item.get("end_time")) is not None
        for item in segments
    )


def _parse_text_script(content: str) -> list[dict[str, str]]:
    segments: list[dict[str, str]] = []
    for line in (content or "").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"\[(\w+)\]\s*(.*)", line)
        if match:
            scene_id = match.group(1)
            text = re.sub(
                r"^\[\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}\]\s*",
                "",
                match.group(2).strip(),
            )
            if text:
                segments.append({"scene_id": scene_id, "text": text})
        else:
            segments.append({"scene_id": "unknown", "text": line})
    return segments


def _generate_timeline_audio(
    *,
    engine: BaseTTSEngine,
    language: str,
    segments: list[dict],
    total_duration: float | None,
    output_path: str,
    audio_dir: str,
    speaking_rate: float,
    cfg,
    fail_log_path: str,
    evidence: list[dict] | None = None,
) -> dict | None:
    os.makedirs(audio_dir, exist_ok=True)
    max_speedup = float(getattr(cfg, "tts_max_speedup", 1.12) if cfg else 1.12)
    min_gap = max(0.0, float(getattr(cfg, "tts_min_gap", 0.20) if cfg else 0.20))
    normalize = bool(getattr(cfg, "tts_normalize_loudness", True) if cfg else True)
    fit_policy = str(getattr(cfg, "tts_narration_fit", "legacy") or "legacy")
    if fit_policy not in {"legacy", "rewrite", "error"}:
        raise ValueError("tts.narration_fit must be legacy, rewrite or error")
    animated = str(getattr(cfg, "subtitle_style", "classic")) in {"word", "karaoke"}
    strict = fit_policy != "legacy" or animated
    if animated and fit_policy == "legacy":
        fit_policy = "error"
    retry_limit = max(0, min(5, int(getattr(cfg, "tts_fit_retries", 2))))

    if not segments:
        if strict:
            raise ValueError("No narration segments were supplied")
        return None
    timeline_end = max(
        float(item.get("timeline_end", item.get("end_time", 0)) or 0)
        for item in segments
    )
    requested_total = float(total_duration or 0)
    if not math.isfinite(requested_total) or not math.isfinite(timeline_end):
        raise ValueError("Narration timeline durations must be finite")
    timeline_duration = requested_total if strict and requested_total > 0 else max(timeline_end, requested_total, 0.1)
    if strict and requested_total <= 0:
        raise ValueError("Native fitting requires the measured video timeline duration")

    placements: list[dict] = []
    durations: dict[str, float] = {}
    previous_end = 0.0
    speech_total = 0.0
    final_segments = []
    all_words = []
    expected_segment_ids = [str(segment.get("scene_id", f"segment_{index + 1:03d}"))
                            for index, segment in enumerate(segments)]
    if strict and len(set(expected_segment_ids)) != len(expected_segment_ids):
        raise ValueError("Strict narration requires a unique scene_id for every beat")
    failed_segment_ids = []
    evidence_map = {str(item.get("scene_id")): item for item in evidence or []}

    for index, seg in enumerate(segments):
        scene_id = str(seg.get("scene_id", f"segment_{index + 1:03d}"))
        text = str(seg.get("text", "") or "").strip()
        if not text:
            if strict:
                raise ValueError(f"{scene_id}: narration text is empty")
            failed_segment_ids.append(scene_id)
            continue

        planned_start = max(0.0, float(seg.get("timeline_start", seg.get("start_time", 0)) or 0))
        planned_end = max(
            planned_start + 0.1,
            float(seg.get("timeline_end", seg.get("end_time", planned_start + 0.1)) or planned_start + 0.1),
        )
        if not math.isfinite(planned_start) or not math.isfinite(planned_end):
            raise ValueError(f"{scene_id}: invalid nonfinite narration window")
        if strict and index and planned_start < float(segments[index - 1].get("timeline_start", segments[index - 1].get("start_time", 0)) or 0):
            raise ValueError(f"{scene_id}: narration windows must follow video chronology")
        available = max(0.25, planned_end - planned_start)
        actual_start = max(planned_start, previous_end + (min_gap if placements else 0.0))
        if strict:
            planned_end = min(planned_end, timeline_duration)
            available = planned_end - actual_start
            if available <= 0:
                raise ValueError(f"{scene_id}: no speech window remains inside video timeline")
        raw_path = os.path.join(audio_dir, f"{index:03d}_{scene_id}_raw.wav")

        try:
            final_segment = dict(seg)
            metadata = {}
            for attempt in range(retry_limit + 1):
                if animated:
                    metadata = engine.synthesize_with_metadata(
                        text=text, output_path=raw_path, language=language, speaking_rate=speaking_rate)
                else:
                    engine.synthesize(text=text, output_path=raw_path, language=language, speaking_rate=speaking_rate)
                actual = _get_audio_duration(raw_path)
                if not math.isfinite(actual) or actual <= 0:
                    raise RuntimeError(f"Cannot determine synthesized duration: {raw_path}")
                if not strict or actual <= available + 0.001:
                    break
                if fit_policy != "rewrite" or attempt >= retry_limit:
                    raise ValueError(
                        f"{scene_id}: narration {actual:.3f}s exceeds window {available:.3f}s; "
                        "shorten this beat or use --narration-fit rewrite. Video and speech were not cut."
                    )
                from dubbingstory.story.script_writer import shorten_for_measured_duration
                final_segment = shorten_for_measured_duration(
                    final_segment, language=language, measured_duration=actual,
                    available_duration=available, cfg=cfg,
                    evidence=[evidence_map[sid] for sid in final_segment.get("covers_scene_ids", [scene_id]) if sid in evidence_map],
                )
                text = final_segment["text"]

            fitted_path = raw_path
            fitted_duration = actual
            if not strict and actual > available + 0.05:
                fitted_path = os.path.join(audio_dir, f"{index:03d}_{scene_id}_fit.wav")
                fitted_duration = _align_audio_duration(
                    raw_path,
                    fitted_path,
                    available,
                    max_speedup=max_speedup,
                    pad_short=False,
                )

            # Keep chronology and avoid overlapping speech.  A preceding beat may
            # naturally continue across a visual cut; the following beat is then
            # shifted slightly instead of overlapping voices.
            actual_start = max(planned_start, previous_end + (min_gap if placements else 0.0))
            actual_end = actual_start + fitted_duration
            if strict and actual_end > planned_end + 0.001:
                raise ValueError(f"{scene_id}: fitted narration exceeds its visual window")
            word_timings = []
            timing_source = None
            if animated:
                from dubbingstory.tts.word_alignment import resolve_word_timings, transform_word_timings
                local_words, timing_source = resolve_word_timings(
                    text, fitted_path, language, cfg,
                    provider_words=metadata.get("word_timings"), provider_source=metadata.get("timing_source"),
                )
                word_timings = transform_word_timings(local_words, offset=actual_start, scene_id=scene_id)
                all_words.extend(word_timings)
            final_segment.update(text=text, start_time=actual_start, end_time=actual_end,
                                 measured_duration=fitted_duration, word_count=len(text.split()))
            final_segments.append(final_segment)
            previous_end = actual_end
            speech_total += fitted_duration
            durations[scene_id] = fitted_duration
            placements.append(
                {
                    "scene_id": scene_id,
                    "path": fitted_path,
                    "planned_start": round(planned_start, 3),
                    "planned_end": round(planned_end, 3),
                    "actual_start": round(actual_start, 3),
                    "actual_end": round(actual_end, 3),
                    "speech_duration": round(fitted_duration, 3),
                    "words": len(text.split()),
                    "text": text,
                    "word_timings": word_timings,
                    "timing_source": timing_source,
                }
            )
            overrun = max(0.0, actual_end - planned_end)
            suffix = f", overrun={overrun:.2f}s" if overrun > 0.05 else ""
            print(
                f"      ✅ {scene_id}: {len(text.split())} words, "
                f"speech={fitted_duration:.2f}s @ {actual_start:.2f}s{suffix}"
            )
        except Exception as exc:
            failed_segment_ids.append(scene_id)
            _log_tts_failure(fail_log_path, language, scene_id, exc)
            print(f"      ❌ {scene_id}: {exc}")
            if strict:
                raise RuntimeError(f"TTS failed for {language}/{scene_id}: {exc}") from exc

    if not placements:
        return None

    overflow = max(0.0, max(item["actual_end"] for item in placements) - timeline_duration)
    if strict and overflow > 0.001:
        raise ValueError(f"Narration exceeds source by {overflow:.3f}s; rendering was stopped")
    if overflow > 0.05:
        print(
            f"   ⚠️ Narration exceeds planned timeline by {overflow:.2f}s. "
            "Consider lowering narration.target_wpm/word budgets or allowing a slightly higher tts.max_speedup."
        )

    # Never create arbitrary per-scene padding.  A single timeline canvas is
    # intentionally as long as the video; silence only exists where no narration
    # is scheduled and original ambience can be heard through dynamic ducking.
    partial_path = os.path.splitext(output_path)[0] + ".synthesizing.wav"
    try:
        _render_timeline_canvas(
            placements=placements,
            output_path=partial_path,
            total_duration=timeline_duration,
            normalize_loudness=normalize,
        )
        rendered_duration = _get_audio_duration(partial_path)
        duration_delta = rendered_duration - timeline_duration
        if strict and (not math.isfinite(rendered_duration) or rendered_duration <= 0
                       or abs(duration_delta) > 0.02):
            raise ValueError(
                "Rendered narration duration differs from the video timeline: "
                f"expected={timeline_duration:.6f}s, actual={rendered_duration:.6f}s, "
                f"delta={duration_delta:+.6f}s (tolerance=0.020000s, "
                f"normalize_loudness={normalize}). No completed audio was published."
            )
        print(f"   🎚️ Narration canvas: expected={timeline_duration:.6f}s, "
              f"actual={rendered_duration:.6f}s, delta={duration_delta:+.6f}s")
        os.replace(partial_path, output_path)
    finally:
        if os.path.isfile(partial_path):
            os.remove(partial_path)

    return {
        "concat_path": output_path,
        "durations": durations,
        "timeline_scheduled": True,
        "timeline_duration": timeline_duration,
        "speech_duration": round(speech_total, 3),
        "speech_coverage": round(min(1.0, speech_total / timeline_duration), 4),
        "overflow_seconds": round(overflow, 3),
        "placements": placements,
        "final_segments": final_segments,
        "word_timings": all_words,
        "expected_segment_ids": expected_segment_ids,
        "failed_segment_ids": failed_segment_ids,
        "complete_coverage": not failed_segment_ids and len(final_segments) == len(segments),
        "fit_policy": fit_policy,
        "strict_timing_validated": strict and not failed_segment_ids and len(final_segments) == len(segments),
    }


def _render_timeline_canvas(
    *,
    placements: list[dict],
    output_path: str,
    total_duration: float,
    normalize_loudness: bool,
) -> None:
    sample_rate = 24000
    if not math.isfinite(total_duration) or total_duration <= 0:
        raise ValueError("A finite positive timeline duration is required for narration mixing")
    target_samples = round(total_duration * sample_rate)
    if target_samples < 1:
        raise ValueError("Narration timeline must contain at least one audio sample")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # An input/output -t works on timestamps and frame boundaries. Bound the
    # silent source by samples instead, including fractional video durations.
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i",
        f"anullsrc=r={sample_rate}:cl=mono,atrim=end_sample={target_samples},asetpts=N/SR/TB",
    ]
    for item in placements:
        cmd.extend(["-i", item["path"]])

    filters: list[str] = []
    mix_labels = ["[0:a]"]
    for index, item in enumerate(placements, start=1):
        delay_ms = max(0, int(round(float(item["actual_start"]) * 1000)))
        label = f"p{index}"
        filters.append(
            f"[{index}:a]aresample={sample_rate},asetpts=N/SR/TB,adelay={delay_ms}:all=1[{label}]"
        )
        mix_labels.append(f"[{label}]")

    output_label = "mixed"
    filters.append(
        "".join(mix_labels)
        + f"amix=inputs={len(mix_labels)}:duration=first:normalize=0[{output_label}]"
    )
    final_label = output_label
    if normalize_loudness:
        # Flush loudnorm's lookahead using silence beyond the video endpoint.
        # Padding AFTER loudnorm could conceal a lost speech tail. This guard
        # is removed below; it never extends the video or the published audio.
        filters.append(
            f"[{output_label}]apad=pad_dur=3,loudnorm=I=-16:LRA=11:TP=-1.5[norm]"
        )
        final_label = "norm"

    # loudnorm can negotiate 192 kHz internally. Establish the final sample
    # rate BEFORE counting samples, then regenerate a zero-based audio clock.
    # Strict speech-window checks above ensure the endpoint cannot cut speech.
    filters.append(
        f"[{final_label}]aresample={sample_rate},atrim=end_sample={target_samples},"
        "asetpts=N/SR/TB[timeline]"
    )
    cmd.extend(
        [
            "-filter_complex", ";".join(filters),
            "-map", "[timeline]",
            "-ar", str(sample_rate), "-ac", "1",
            "-c:a", "pcm_s16le",
            output_path,
        ]
    )
    _run_subprocess(cmd, "timeline_mix")


def _log_tts_failure(path: str, language: str, scene_id: str, exc: Exception) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"[{language.upper()}] [{scene_id}] Error: {exc}\n")


def _concatenate_audio(audio_paths: list[str], output_path: str) -> str:
    """Legacy sequential concatenation without artificial per-segment gaps."""
    if not audio_paths:
        return output_path
    if len(audio_paths) == 1:
        shutil.copy2(audio_paths[0], output_path)
        return output_path

    list_path = output_path + ".concat.txt"
    with open(list_path, "w", encoding="utf-8") as handle:
        for path in audio_paths:
            safe_path = os.path.abspath(path).replace("\\", "/").replace("'", "'\\''")
            handle.write(f"file '{safe_path}'\n")

    try:
        _run_subprocess(
            [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                "-c:a", "pcm_s16le", output_path,
            ],
            "audio_concat",
        )
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)
    return output_path


def _get_audio_duration(audio_path: str) -> float:
    # TTS/canvas outputs are PCM WAV: count samples rather than estimating a
    # container duration. Preserve the ffprobe fallback for other audio types.
    from dubbingstory.tts.word_alignment import WordAlignmentError, wav_duration
    try:
        return wav_duration(audio_path)
    except WordAlignmentError:
        pass
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", audio_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode:
        return 0.0
    try:
        duration = float(result.stdout.strip())
        return duration if math.isfinite(duration) and duration > 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def _align_audio_duration(
    input_path: str,
    output_path: str,
    target_duration: float,
    max_speedup: float = 1.12,
    *,
    pad_short: bool = False,
) -> float:
    """Gently fit speech to a maximum window without manufacturing dead air."""
    actual_duration = _get_audio_duration(input_path)
    if actual_duration <= 0:
        raise RuntimeError(f"Cannot determine audio duration: {input_path}")

    if actual_duration <= target_duration + 0.03:
        shutil.copy2(input_path, output_path)
        return actual_duration

    tempo = min(max(1.0, actual_duration / max(target_duration, 0.1)), max(1.0, max_speedup))
    filters: list[str] = []
    remaining = tempo
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.6f}")

    final_duration = actual_duration / tempo
    if pad_short and final_duration < target_duration:
        filters.append(f"apad=pad_dur={target_duration - final_duration:.3f}")
        final_duration = target_duration

    _run_subprocess(
        [
            "ffmpeg", "-y", "-i", input_path,
            "-af", ",".join(filters),
            "-t", f"{final_duration:.3f}",
            "-ar", "24000", "-ac", "1",
            "-c:a", "pcm_s16le", output_path,
        ],
        "tts_duration_fit",
    )
    return final_duration


def _run_subprocess(cmd: list[str], label: str) -> None:
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg/ffprobe is required for TTS timeline processing") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace")[-1000:] if exc.stderr else ""
        raise RuntimeError(f"FFmpeg failed ({label}): {stderr}") from exc
