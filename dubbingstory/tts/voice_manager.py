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

    print(f"⚠️ Unknown TTS engine '{name}', falling back to Edge-TTS.")
    from dubbingstory.tts.edge_tts_engine import EdgeTTSEngine
    return EdgeTTSEngine(custom_voices=_voice_overrides("edge", cfg) or None)


def generate_all_audio(
    scripts_dir: str,
    audio_dir: str,
    engine_name: str = "edge",
    cfg=None,
    script_prefix: str = "script_",
    audio_prefix: str = "audio_",
    segment_durations: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Generate TTS for every language script.

    V2 behavior
    -----------
    If ``{script_prefix}{lang}.json`` exists and contains timeline information,
    a timeline scheduler is used.  Each TTS clip is synthesized at its natural
    length, gently sped up only when it exceeds its planned window, and then
    placed at ``timeline_start`` on a full-length silent canvas.  Short speech
    clips are *not* padded to the duration of their scene.

    Legacy behavior
    ---------------
    Plain ``.txt`` scripts still work.  ``segment_durations`` is retained for
    backward compatibility with older callers.
    """
    engine = get_engine(engine_name, cfg)
    speaking_rate = float(getattr(cfg, "tts_speaking_rate", 1.0) if cfg else 1.0)
    use_scheduler = bool(getattr(cfg, "tts_use_timeline_scheduler", True) if cfg else True)

    os.makedirs(audio_dir, exist_ok=True)
    fail_log_path = os.path.join(os.path.dirname(audio_dir), "failed_tts.log")
    if os.path.exists(fail_log_path):
        os.remove(fail_log_path)

    script_files = sorted(glob.glob(os.path.join(scripts_dir, f"{script_prefix}*.txt")))
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

        if structured and use_scheduler and _has_timeline(structured["segments"]):
            result = _generate_timeline_audio(
                engine=engine,
                language=lang,
                segments=structured["segments"],
                total_duration=structured.get("total_duration"),
                output_path=os.path.join(audio_dir, f"{audio_prefix}{lang}.wav"),
                audio_dir=os.path.join(audio_dir, lang),
                speaking_rate=speaking_rate,
                cfg=cfg,
                fail_log_path=fail_log_path,
            )
            if result:
                results[lang] = result
                print(
                    f"   ✅ {lang.upper()} timeline audio: {result['concat_path']} "
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
                print(f"      ✅ {scene_id}: {len(seg['text'].split())} words")
            except Exception as exc:
                _log_tts_failure(fail_log_path, lang, scene_id, exc)
                print(f"      ❌ {scene_id}: {exc}")

        if segment_paths:
            concat_path = os.path.join(audio_dir, f"{audio_prefix}{lang}.wav")
            _concatenate_audio(segment_paths, concat_path)
            results[lang] = {
                "concat_path": concat_path,
                "durations": final_durations,
                "timeline_scheduled": False,
            }
            print(f"   ✅ {lang.upper()} audio: {concat_path}")

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
) -> dict | None:
    os.makedirs(audio_dir, exist_ok=True)
    max_speedup = float(getattr(cfg, "tts_max_speedup", 1.12) if cfg else 1.12)
    min_gap = max(0.0, float(getattr(cfg, "tts_min_gap", 0.20) if cfg else 0.20))
    normalize = bool(getattr(cfg, "tts_normalize_loudness", True) if cfg else True)

    timeline_end = max(
        float(item.get("timeline_end", item.get("end_time", 0)) or 0)
        for item in segments
    )
    requested_total = float(total_duration or 0)
    timeline_duration = max(timeline_end, requested_total, 0.1)

    placements: list[dict] = []
    durations: dict[str, float] = {}
    previous_end = 0.0
    speech_total = 0.0

    for index, seg in enumerate(segments):
        scene_id = str(seg.get("scene_id", f"segment_{index + 1:03d}"))
        text = str(seg.get("text", "") or "").strip()
        if not text:
            continue

        planned_start = max(0.0, float(seg.get("timeline_start", seg.get("start_time", 0)) or 0))
        planned_end = max(
            planned_start + 0.1,
            float(seg.get("timeline_end", seg.get("end_time", planned_start + 0.1)) or planned_start + 0.1),
        )
        available = max(0.25, planned_end - planned_start)
        raw_path = os.path.join(audio_dir, f"{index:03d}_{scene_id}_raw.wav")

        try:
            engine.synthesize(
                text=text,
                output_path=raw_path,
                language=language,
                speaking_rate=speaking_rate,
            )
            actual = _get_audio_duration(raw_path)
            if actual <= 0:
                raise RuntimeError(f"Cannot determine synthesized duration: {raw_path}")

            fitted_path = raw_path
            fitted_duration = actual
            if actual > available + 0.05:
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
                }
            )
            overrun = max(0.0, actual_end - planned_end)
            suffix = f", overrun={overrun:.2f}s" if overrun > 0.05 else ""
            print(
                f"      ✅ {scene_id}: {len(text.split())} words, "
                f"speech={fitted_duration:.2f}s @ {actual_start:.2f}s{suffix}"
            )
        except Exception as exc:
            _log_tts_failure(fail_log_path, language, scene_id, exc)
            print(f"      ❌ {scene_id}: {exc}")

    if not placements:
        return None

    overflow = max(0.0, max(item["actual_end"] for item in placements) - timeline_duration)
    if overflow > 0.05:
        print(
            f"   ⚠️ Narration exceeds planned timeline by {overflow:.2f}s. "
            "Consider lowering narration.target_wpm/word budgets or allowing a slightly higher tts.max_speedup."
        )

    # Never create arbitrary per-scene padding.  A single timeline canvas is
    # intentionally as long as the video; silence only exists where no narration
    # is scheduled and original ambience can be heard through dynamic ducking.
    _render_timeline_canvas(
        placements=placements,
        output_path=output_path,
        total_duration=timeline_duration,
        normalize_loudness=normalize,
    )

    return {
        "concat_path": output_path,
        "durations": durations,
        "timeline_scheduled": True,
        "timeline_duration": timeline_duration,
        "speech_duration": round(speech_total, 3),
        "speech_coverage": round(min(1.0, speech_total / timeline_duration), 4),
        "overflow_seconds": round(overflow, 3),
        "placements": placements,
    }


def _render_timeline_canvas(
    *,
    placements: list[dict],
    output_path: str,
    total_duration: float,
    normalize_loudness: bool,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-t", f"{total_duration:.3f}",
        "-i", "anullsrc=r=24000:cl=mono",
    ]
    for item in placements:
        cmd.extend(["-i", item["path"]])

    filters: list[str] = []
    mix_labels = ["[0:a]"]
    for index, item in enumerate(placements, start=1):
        delay_ms = max(0, int(round(float(item["actual_start"]) * 1000)))
        label = f"p{index}"
        filters.append(
            f"[{index}:a]aresample=24000,adelay={delay_ms}:all=1[{label}]"
        )
        mix_labels.append(f"[{label}]")

    output_label = "mixed"
    filters.append(
        "".join(mix_labels)
        + f"amix=inputs={len(mix_labels)}:duration=first:normalize=0[{output_label}]"
    )
    final_label = output_label
    if normalize_loudness:
        filters.append(
            f"[{output_label}]loudnorm=I=-16:LRA=11:TP=-1.5[norm]"
        )
        final_label = "norm"

    cmd.extend(
        [
            "-filter_complex", ";".join(filters),
            "-map", f"[{final_label}]",
            "-t", f"{total_duration:.3f}",
            "-ar", "24000", "-ac", "1",
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
    try:
        return float(result.stdout.strip())
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
