"""
dubbingstory.render.video_render — Final video assembly

Combines original video + narration audio + optional subtitles
into the final dubbed video output.
Supports multiple aspect ratios (16:9, 9:16).
"""

import json
import os
import subprocess

from dubbingstory.render.audio_mix import mix_narration_with_original
from dubbingstory.render.subtitle_burn import burn_subtitles


RATIO_DIMS = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:3": (1440, 1080),
}


def _rescale_video(
    input_path: str,
    output_path: str,
    target_w: int,
    target_h: int,
) -> str:
    """
    Rescale video to target dimensions with padding (letterbox/pillarbox).
    """
    vf = (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"format=yuv420p"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        output_path,
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace")[:500] if e.stderr else ""
        raise RuntimeError(f"Video rescale failed: {stderr}") from e

    return output_path


def render_final(
    source_video: str,
    narration_audio: str,
    output_path: str,
    subtitle_path: str | None = None,
    audio_strategy: str = "mute_original",
    original_volume: float = 0.15,
    burn_subs: bool = False,
    target_ratio: str | None = None,
) -> str:
    """
    Render the final dubbed video.

    Pipeline:
    1. Optionally rescale to target ratio
    2. Mix narration audio with original
    3. Optionally burn subtitles

    Parameters
    ----------
    source_video : str
        Path to the source video.
    narration_audio : str
        Path to the narration audio.
    output_path : str
        Path for the final output.
    subtitle_path : str | None
        Path to .srt file for optional burn-in.
    audio_strategy : str
        Audio mixing strategy.
    original_volume : float
        Volume for original audio in duck mode.
    burn_subs : bool
        Whether to burn subtitles into the video.
    target_ratio : str | None
        Target aspect ratio ("16:9", "9:16"). None = keep original.

    Returns
    -------
    str
        Path to the final video.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    working_video = source_video

    # Step 1: Rescale if needed
    if target_ratio and target_ratio in RATIO_DIMS:
        target_w, target_h = RATIO_DIMS[target_ratio]
        rescaled_path = output_path + ".rescaled.mp4"
        print(f"      📐 Rescaling to {target_ratio} ({target_w}x{target_h})...")
        working_video = _rescale_video(working_video, rescaled_path, target_w, target_h)

    # Step 2: Mix audio
    mixed_path = output_path + ".mixed.mp4"
    print(f"      🔊 Mixing audio (strategy: {audio_strategy})...")
    mix_narration_with_original(
        original_video=working_video,
        narration_audio=narration_audio,
        output_path=mixed_path,
        strategy=audio_strategy,
        original_volume=original_volume,
    )
    working_video = mixed_path

    # Step 3: Burn subtitles (optional)
    if burn_subs and subtitle_path and os.path.exists(subtitle_path):
        burned_path = output_path + ".burned.mp4"
        print(f"      📝 Burning subtitles...")
        burn_subtitles(
            video_path=working_video,
            subtitle_path=subtitle_path,
            output_path=burned_path,
        )
        working_video = burned_path

    # Step 4: Final copy/rename
    if working_video != output_path:
        import shutil
        shutil.move(working_video, output_path)

    # Cleanup temp files
    for temp_suffix in [".rescaled.mp4", ".mixed.mp4", ".burned.mp4"]:
        temp_path = output_path + temp_suffix
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return output_path


def render_project(
    project_dir: str,
    ratios: list[str] | None = None,
    cfg=None,
) -> list[str]:
    """
    Render all final videos for a project.

    Generates one video per language per ratio.

    Parameters
    ----------
    project_dir : str
        Project output directory.
    ratios : list[str]
        Output aspect ratios.
    cfg : SimpleNamespace
        Config object.

    Returns
    -------
    list[str]
        Paths to rendered video files.
    """
    if ratios is None:
        ratios = getattr(cfg, "render_ratios", ["16:9"])

    audio_strategy = getattr(cfg, "render_audio_strategy", "mute_original")
    original_volume = getattr(cfg, "render_original_volume", 0.15)
    burn_subs = getattr(cfg, "render_burn_subtitles", False)

    source_video = os.path.join(project_dir, "source.mp4")
    if not os.path.exists(source_video):
        raise FileNotFoundError(f"Source video not found: {source_video}")

    audio_dir = os.path.join(project_dir, "audio")
    scripts_dir = os.path.join(project_dir, "scripts")

    # Find available audio files
    import glob
    audio_files = glob.glob(os.path.join(audio_dir, "audio_*.wav"))

    if not audio_files:
        print("   ⚠️ No audio files found. Run 'dubbingstory dub' first.")
        return []

    rendered = []

    for audio_path in audio_files:
        # Extract language from filename: audio_id.wav → id
        basename = os.path.basename(audio_path)
        lang = basename.replace("audio_", "").replace(".wav", "")

        # Find matching subtitle
        srt_path = os.path.join(scripts_dir, f"script_{lang}.srt")

        for ratio in ratios:
            ratio_label = ratio.replace(":", "x")
            output_path = os.path.join(project_dir, f"final_{lang}_{ratio_label}.mp4")

            print(f"\n   🎬 Rendering: {lang.upper()} ({ratio})...")

            try:
                render_final(
                    source_video=source_video,
                    narration_audio=audio_path,
                    output_path=output_path,
                    subtitle_path=srt_path if os.path.exists(srt_path) else None,
                    audio_strategy=audio_strategy,
                    original_volume=original_volume,
                    burn_subs=burn_subs,
                    target_ratio=ratio if ratio != "16:9" else None,  # Assume source is 16:9
                )

                rendered.append(output_path)
                print(f"   ✅ {output_path}")

            except Exception as e:
                print(f"   ❌ Render failed: {e}")

    # Save render manifest
    manifest = {
        "rendered_videos": rendered,
        "audio_strategy": audio_strategy,
        "ratios": ratios,
    }
    manifest_path = os.path.join(project_dir, "render_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return rendered
