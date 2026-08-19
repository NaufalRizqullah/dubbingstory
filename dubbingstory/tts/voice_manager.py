"""
dubbingstory.tts.voice_manager — Voice selection & audio generation

Manages TTS engine selection, voice consistency across scenes,
and batch audio generation from narration scripts.
"""

import json
import os
import subprocess

from dubbingstory.tts.base_engine import BaseTTSEngine


def get_engine(name: str = "edge", cfg=None) -> BaseTTSEngine:
    """
    Get a TTS engine instance by name.

    Parameters
    ----------
    name : str
        Engine name: "piper" or "voxcpm2".
    cfg : SimpleNamespace | None
        Config object.

    Returns
    -------
    BaseTTSEngine
        TTS engine instance.
    """
    custom_voices = {}
    if cfg:
        vid = getattr(cfg, "tts_voice_id", getattr(cfg, "tts_edge_voices_id", None))
        ven = getattr(cfg, "tts_voice_en", getattr(cfg, "tts_edge_voices_en", None))
        if vid:
            custom_voices["id"] = vid
        if ven:
            custom_voices["en"] = ven

    if name == "edge":
        from dubbingstory.tts.edge_tts_engine import EdgeTTSEngine
        return EdgeTTSEngine(custom_voices=custom_voices or None)
    elif name == "piper":
        from dubbingstory.tts.piper_tts_engine import PiperTTSEngine
        return PiperTTSEngine(custom_voices=custom_voices or None)
    else:
        print(f"⚠️ Unknown TTS engine '{name}', falling back to Edge-TTS.")
        from dubbingstory.tts.edge_tts_engine import EdgeTTSEngine
        return EdgeTTSEngine(custom_voices=custom_voices or None)


def generate_all_audio(
    scripts_dir: str,
    audio_dir: str,
    engine_name: str = "edge",
    cfg=None,
    script_prefix: str = "script_",
    audio_prefix: str = "audio_",
    segment_durations: dict[str, float] | None = None,
) -> dict[str, str]:
    """
    Generate TTS audio for all narration scripts in the scripts directory.

    Looks for {script_prefix}*.txt files and generates corresponding audio files.

    Parameters
    ----------
    scripts_dir : str
        Directory containing narration script files.
    audio_dir : str
        Directory to save generated audio files.
    engine_name : str
        TTS engine to use.
    cfg : SimpleNamespace | None
        Config object.
    script_prefix : str
        Prefix for script filenames (default: "script_").
    audio_prefix : str
        Prefix for output audio filenames (default: "audio_").
    segment_durations : dict[str, float] | None
        Optional target duration per scene for summary alignment.

    Returns
    -------
    dict[str, str]
        Mapping of language → path to concatenated audio file.
    """
    engine = get_engine(engine_name, cfg)
    speaking_rate = getattr(cfg, "tts_speaking_rate", 1.0) if cfg else 1.0

    os.makedirs(audio_dir, exist_ok=True)

    results: dict[str, str] = {}

    # Find all script files
    import glob
    script_files = sorted(glob.glob(os.path.join(scripts_dir, f"{script_prefix}*.txt")))

    if not script_files:
        print(f"   ⚠️ No script files found matching '{script_prefix}*.txt' in scripts directory.")
        return results

    for script_path in script_files:
        # Extract language from filename: script_id.txt → id
        basename = os.path.basename(script_path)
        lang = basename.replace(script_prefix, "").replace(".txt", "")

        print(f"\n   🎙️ Generating {lang.upper()} audio with {engine.get_engine_name()}...")

        # Read script
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse script lines: [scene_id] text
        import re
        segments = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            match = re.match(r'\[(\w+)\]\s*(.*)', line)
            if match:
                scene_id = match.group(1)
                text = match.group(2).strip()
                if text:
                    segments.append({"scene_id": scene_id, "text": text})
            elif line:
                segments.append({"scene_id": "unknown", "text": line})

        if not segments:
            print(f"   ⚠️ No segments found in {basename}")
            continue

        # Generate audio per segment
        lang_audio_dir = os.path.join(audio_dir, lang)
        os.makedirs(lang_audio_dir, exist_ok=True)

        segment_paths = []

        for i, seg in enumerate(segments):
            seg_audio = os.path.join(lang_audio_dir, f"{seg['scene_id']}.wav")

            try:
                engine.synthesize(
                    text=seg["text"],
                    output_path=seg_audio,
                    language=lang,
                    speaking_rate=speaking_rate,
                )
                target_duration = (segment_durations or {}).get(seg["scene_id"])
                if target_duration and target_duration > 0:
                    aligned_audio = os.path.join(
                        lang_audio_dir, f"{seg['scene_id']}_aligned.wav"
                    )
                    _align_audio_duration(seg_audio, aligned_audio, target_duration)
                    segment_paths.append(aligned_audio)
                else:
                    segment_paths.append(seg_audio)
                print(f"      ✅ {seg['scene_id']}: {len(seg['text'].split())} words")
            except Exception as e:
                print(f"      ❌ {seg['scene_id']}: {e}")

        # Concatenate all segments into one audio file
        if segment_paths:
            concat_path = os.path.join(audio_dir, f"{audio_prefix}{lang}.wav")
            _concatenate_audio(segment_paths, concat_path)
            results[lang] = concat_path
            print(f"   ✅ {lang.upper()} audio: {concat_path}")

    return results


def _concatenate_audio(audio_paths: list[str], output_path: str) -> str:
    """
    Concatenate multiple WAV files into one using FFmpeg.

    Adds a small silence gap between segments for natural pacing.
    """
    if not audio_paths:
        return output_path

    if len(audio_paths) == 1:
        import shutil
        shutil.copy2(audio_paths[0], output_path)
        return output_path

    # Build FFmpeg concat file
    list_path = output_path + ".concat.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for i, path in enumerate(audio_paths):
            safe_path = os.path.abspath(path).replace("\\", "/")
            f.write(f"file '{safe_path}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        output_path,
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace")[:200] if e.stderr else ""
        print(f"   ⚠️ Audio concat failed: {stderr}")
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)

    return output_path


def _get_audio_duration(audio_path: str) -> float:
    """Return an audio file duration in seconds, or zero when unavailable."""
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


def _align_audio_duration(input_path: str, output_path: str, target_duration: float) -> str:
    """Fit one scene's narration exactly to its visual scene duration."""
    actual_duration = _get_audio_duration(input_path)
    if actual_duration <= 0:
        raise RuntimeError(f"Cannot determine audio duration: {input_path}")

    filters = []
    if actual_duration > target_duration + 0.03:
        tempo = actual_duration / target_duration
        while tempo > 2.0:
            filters.append("atempo=2.0")
            tempo /= 2.0
        while tempo < 0.5:
            filters.append("atempo=0.5")
            tempo /= 0.5
        filters.append(f"atempo={tempo:.6f}")
    else:
        filters.append(f"apad=pad_dur={target_duration:.3f}")

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", input_path,
            "-af", ",".join(filters),
            "-t", f"{target_duration:.3f}",
            "-c:a", "pcm_s16le", output_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return output_path
