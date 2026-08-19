"""
dubbingstory.cli — Command-line interface

Subcommands:
    run       Full pipeline (ingest → segment → analyze → narrate → dub → render)
    ingest    Download/validate video + read subtitles
    segment   Detect scenes + extract keyframes
    analyze   Visual understanding with Gemini
    narrate   Generate narration scripts
    dub       Generate TTS audio
    render    Mix audio + render final video
"""

import argparse
import os
import sys
import json

from dubbingstory import __version__
from dubbingstory.config import build_config


def _setup_project_dir(cfg, project_name: str | None = None) -> str:
    """Create and return the project output directory."""
    output_dir = getattr(cfg, "project_output_dir", "outputs")
    name = project_name or getattr(cfg, "project_name", "") or "untitled"
    project_dir = os.path.join(output_dir, name)
    os.makedirs(project_dir, exist_ok=True)

    # Create subdirectories
    for subdir in ["scenes", "keyframes", "audio", "scripts"]:
        os.makedirs(os.path.join(project_dir, subdir), exist_ok=True)

    return project_dir


def _resolve_project_name(args) -> str:
    """Derive project name from input or explicit --project flag."""
    if hasattr(args, "project") and args.project:
        return args.project
    if hasattr(args, "input") and args.input:
        return os.path.splitext(os.path.basename(args.input))[0]
    if hasattr(args, "url") and args.url:
        # Sanitize URL into a usable project name
        from urllib.parse import urlparse
        parsed = urlparse(args.url)
        slug = parsed.path.strip("/").replace("/", "_")[:50] or "youtube_video"
        return slug
    return "untitled"


# ==============================================================================
# SUBCOMMAND: ingest
# ==============================================================================

def cmd_ingest(args, cfg):
    """Download/validate video and read subtitles."""
    from dubbingstory.ingest import local_video, youtube, subtitle_reader

    project_name = _resolve_project_name(args)
    project_dir = _setup_project_dir(cfg, project_name)

    print("=" * 70)
    print(f"📥 DubbingStory v{__version__} — Ingest")
    print("=" * 70)

    video_path = None

    if args.input:
        # Local video
        print(f"   Source: Local file → {args.input}")
        video_path = local_video.validate_and_copy(args.input, project_dir)
    elif args.url:
        # YouTube / URL download
        if not getattr(args, "i_have_rights", False):
            print("⚠️  Anda harus menambahkan --i-have-rights untuk mengunduh dari URL.")
            print("   Ini menandakan Anda memiliki hak atas video tersebut.")
            sys.exit(1)

        print(f"   Source: URL → {args.url}")
        download_q = getattr(args, "quality", None) or getattr(cfg, "ingest_download_height", "1080")
        video_path = youtube.download_video(
            url=args.url,
            output_dir=project_dir,
            download_height=download_q,
            cookies=getattr(args, "yt_cookies", None) or getattr(cfg, "ingest_yt_cookies", None),
        )
    else:
        print("❌ Harus menyediakan --input atau --url")
        sys.exit(1)

    # ── Context acquisition fallback chain ──
    # Priority: 1) Local subtitle files → 2) YouTube auto-subs → 3) Whisper ASR
    subtitle_data = None
    context_source = "none"
    asr_used = False

    # Step 1: Look for local subtitle files (SRT/ASS/VTT alongside video)
    if getattr(cfg, "ingest_read_subtitles", True):
        subtitle_data = subtitle_reader.find_and_read_subtitles(
            video_path, project_dir
        )
        if subtitle_data:
            context_source = subtitle_data.get("source", "manual")

    # Step 2: If no local subs and source is YouTube, try downloading subs
    if subtitle_data is None and args.url:
        print("   🌐 No local subtitles — trying YouTube subtitle download...")
        yt_sub_path = youtube.download_subtitles(
            url=args.url,
            output_dir=project_dir,
            languages=getattr(cfg, "narration_languages", ["id", "en"]),
            cookies=getattr(args, "yt_cookies", None) or getattr(cfg, "ingest_yt_cookies", None),
        )
        if yt_sub_path:
            # Read the downloaded subtitle file
            from dubbingstory.ingest.subtitle_reader import read_subtitles, subtitles_to_context_string
            entries = read_subtitles(yt_sub_path)
            if entries:
                context_string = subtitles_to_context_string(entries)
                subtitle_data = {
                    "source_files": [yt_sub_path],
                    "primary_file": yt_sub_path,
                    "entries": entries,
                    "context_string": context_string,
                    "total_lines": len(entries),
                    "source": "youtube_auto",
                }
                context_source = "youtube_auto"
                print(f"   ✅ YouTube subs: {len(entries)} lines")

    # Step 3: If still no context, use Whisper ASR as final fallback
    if (
        subtitle_data is None
        and getattr(cfg, "ingest_use_asr_fallback", True)
        and getattr(args, "use_asr", False)
    ):
        from dubbingstory.asr.whisper_asr import transcribe_video

        print("   🎙️  No subtitles found — attempting ASR fallback...")
        asr_result = transcribe_video(
            video_path=video_path,
            project_dir=project_dir,
            model_name=getattr(cfg, "whisper_model", "base"),
            language=getattr(cfg, "whisper_language", None),
            device=getattr(cfg, "whisper_device", "cpu"),
            compute_type=getattr(cfg, "whisper_compute_type", "int8"),
        )
        if asr_result:
            subtitle_data = asr_result
            asr_used = True
            context_source = "asr"
            print(f"   ✅ ASR fallback produced {asr_result['total_lines']} entries")
        else:
            print("   ⚠️  ASR fallback failed or unavailable; continuing without transcript.")

    # Save ingest manifest
    manifest = {
        "project_name": project_name,
        "video_path": video_path,
        "subtitle_data": subtitle_data,
        "context_source": context_source,
        "asr_used": asr_used,
        "status": "ingested",
    }
    manifest_path = os.path.join(project_dir, "ingest_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Ingest selesai!")
    print(f"   Video: {video_path}")
    if subtitle_data:
        print(f"   Context: {context_source} ({subtitle_data.get('total_lines', 0)} lines)")
    else:
        print(f"   Context: None (vision-only mode)")
    print(f"   Manifest: {manifest_path}")

    return manifest


# ==============================================================================
# SUBCOMMAND: segment
# ==============================================================================

def cmd_segment(args, cfg):
    """Detect scenes and extract keyframes."""
    from dubbingstory.segment import scene_detect, keyframes

    project_name = _resolve_project_name(args)
    project_dir = _setup_project_dir(cfg, project_name)

    # Load ingest manifest
    manifest_path = os.path.join(project_dir, "ingest_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"❌ Ingest manifest not found: {manifest_path}")
        print("   Jalankan 'dubbingstory ingest' terlebih dahulu.")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        ingest = json.load(f)

    video_path = ingest["video_path"]

    print("=" * 70)
    print(f"✂️  DubbingStory v{__version__} — Segment")
    print("=" * 70)
    print(f"   Video: {video_path}")

    analysis_mode = getattr(cfg, "vision_analysis_mode", "keyframes")
    
    if analysis_mode == "keyframes":
        # Don't split the video physically, just detect boundaries
        scene_list = scene_detect.detect_scenes(
            video_path=video_path,
            detector=getattr(cfg, "segment_detector", "adaptive"),
            threshold=getattr(cfg, "segment_adaptive_threshold", 3.0),
            min_duration=getattr(cfg, "segment_min_scene_duration", 2.0),
            merge_short=getattr(cfg, "segment_merge_short_scenes", True),
        )
    else:
        # Scene detection + split
        scenes_dir = os.path.join(project_dir, "scenes")
        scene_list = scene_detect.detect_and_split(
            video_path=video_path,
            output_dir=scenes_dir,
            detector=getattr(cfg, "segment_detector", "adaptive"),
            threshold=getattr(cfg, "segment_adaptive_threshold", 3.0),
            min_duration=getattr(cfg, "segment_min_scene_duration", 2.0),
            merge_short=getattr(cfg, "segment_merge_short_scenes", True),
        )

    # Keyframe extraction
    keyframes_dir = os.path.join(project_dir, "keyframes")
    all_keyframes = keyframes.extract_all_scenes(
        scenes=scene_list,
        scenes_dir=os.path.join(project_dir, "scenes"),
        output_dir=keyframes_dir,
        strategy=getattr(cfg, "keyframes_strategy", "distributed"),
        max_per_scene=getattr(cfg, "keyframes_max_per_scene", 7),
        source_video=video_path,
    )

    # Save segment manifest
    segment_manifest = {
        "video_path": video_path,
        "total_scenes": len(scene_list),
        "scenes": scene_list,
        "keyframes": all_keyframes,
        "status": "segmented",
    }
    segment_path = os.path.join(project_dir, "segment_manifest.json")
    with open(segment_path, "w", encoding="utf-8") as f:
        json.dump(segment_manifest, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Segmentation selesai!")
    print(f"   Scenes: {len(scene_list)}")
    print(f"   Keyframes: {sum(len(kf) for kf in all_keyframes.values())}")
    print(f"   Manifest: {segment_path}")

    return segment_manifest


# ==============================================================================
# SUBCOMMAND: analyze
# ==============================================================================

def cmd_analyze(args, cfg):
    """Run visual understanding on scenes."""
    from dubbingstory.vision import scene_understanding

    project_name = _resolve_project_name(args)
    project_dir = _setup_project_dir(cfg, project_name)

    print("=" * 70)
    print(f"👁️  DubbingStory v{__version__} — Visual Analysis")
    print("=" * 70)

    # Load segment manifest
    segment_path = os.path.join(project_dir, "segment_manifest.json")
    if not os.path.exists(segment_path):
        print(f"❌ Segment manifest not found. Jalankan 'dubbingstory segment' dulu.")
        sys.exit(1)

    with open(segment_path, "r", encoding="utf-8") as f:
        segment_data = json.load(f)

    # Load ingest manifest for subtitle context
    ingest_path = os.path.join(project_dir, "ingest_manifest.json")
    subtitle_context = None
    if os.path.exists(ingest_path):
        with open(ingest_path, "r", encoding="utf-8") as f:
            ingest_data = json.load(f)
        subtitle_context = ingest_data.get("subtitle_data")

    storyboard = scene_understanding.run_analysis(
        segment_data=segment_data,
        project_dir=project_dir,
        cfg=cfg,
        subtitle_context=subtitle_context,
        domain_hint=getattr(cfg, "vision_domain_hint", ""),
    )

    # Save storyboard
    storyboard_path = os.path.join(project_dir, "storyboard.json")
    with open(storyboard_path, "w", encoding="utf-8") as f:
        json.dump(storyboard, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Visual analysis selesai!")
    print(f"   Storyboard: {storyboard_path}")

    return storyboard


# ==============================================================================
# SUBCOMMAND: narrate
# ==============================================================================

def cmd_narrate(args, cfg):
    """Generate narration scripts."""
    from dubbingstory.story import script_writer, subtitle_gen

    project_name = _resolve_project_name(args)
    project_dir = _setup_project_dir(cfg, project_name)

    print("=" * 70)
    print(f"📝 DubbingStory v{__version__} — Narration")
    print("=" * 70)

    # Load storyboard
    storyboard_path = os.path.join(project_dir, "storyboard.json")
    if not os.path.exists(storyboard_path):
        print(f"❌ Storyboard not found. Jalankan 'dubbingstory analyze' dulu.")
        sys.exit(1)

    with open(storyboard_path, "r", encoding="utf-8") as f:
        storyboard = json.load(f)

    languages = args.lang if hasattr(args, "lang") and args.lang else \
        getattr(cfg, "narration_languages", ["id", "en"])
    style = args.style if hasattr(args, "style") and args.style else \
        getattr(cfg, "narration_style", "viral_fb")

    print(f"   Style: {style}")
    print(f"   Languages: {languages}")

    # Generate narration
    narration = script_writer.generate_narration(
        storyboard=storyboard,
        languages=languages,
        style=style,
        cfg=cfg,
    )

    # Save scripts
    scripts_dir = os.path.join(project_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    for lang, segments in narration.items():
        # Save text script
        script_path = os.path.join(scripts_dir, f"script_{lang}.txt")
        with open(script_path, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(f"[{seg['scene_id']}] {seg['text']}\n\n")

        # Save SRT
        srt_path = os.path.join(scripts_dir, f"script_{lang}.srt")
        subtitle_gen.generate_srt(segments, srt_path)

        print(f"   📄 {script_path}")
        print(f"   📄 {srt_path}")

    print(f"\n✅ Narration selesai!")

    return narration


# ==============================================================================
# SUBCOMMAND: dub
# ==============================================================================

def cmd_dub(args, cfg):
    """Generate TTS audio dubbing."""
    from dubbingstory.tts import voice_manager

    project_name = _resolve_project_name(args)
    project_dir = _setup_project_dir(cfg, project_name)

    print("=" * 70)
    print(f"🎙️ DubbingStory v{__version__} — TTS Dubbing")
    print("=" * 70)

    engine_name = args.engine if hasattr(args, "engine") and args.engine else \
        getattr(cfg, "tts_engine", "edge")

    print(f"   Engine: {engine_name}")

    scripts_dir = os.path.join(project_dir, "scripts")
    audio_dir = os.path.join(project_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    voice_manager.generate_all_audio(
        scripts_dir=scripts_dir,
        audio_dir=audio_dir,
        engine_name=engine_name,
        cfg=cfg,
    )

    # Backup TTS audio to project dir
    import glob
    import shutil
    audio_files = glob.glob(os.path.join(audio_dir, "audio_*.wav"))
    for audio_path in audio_files:
        basename = os.path.basename(audio_path)
        lang = basename.replace("audio_", "").replace(".wav", "")
        backup_path = os.path.join(project_dir, f"backup_tts_full_{lang}.wav")
        shutil.copy2(audio_path, backup_path)
        print(f"   💾 TTS Backup saved to: {backup_path}")

    print(f"\n✅ TTS dubbing selesai!")
    print(f"   Audio: {audio_dir}")


# ==============================================================================
# SUBCOMMAND: render
# ==============================================================================

def cmd_render(args, cfg):
    """Render final dubbed video."""
    from dubbingstory.render import video_render

    project_name = _resolve_project_name(args)
    project_dir = _setup_project_dir(cfg, project_name)

    print("=" * 70)
    print(f"🎬 DubbingStory v{__version__} — Render")
    print("=" * 70)

    ratios = getattr(cfg, "render_ratios", ["16:9"])

    video_render.render_project(
        project_dir=project_dir,
        ratios=ratios,
        cfg=cfg,
    )

    print(f"\n✅ Render selesai!")


# ==============================================================================
# SUBCOMMAND: run (full pipeline)
# ==============================================================================

def cmd_run(args, cfg):
    """Run the full pipeline end-to-end."""
    mode = getattr(args, "mode", "full")

    print("=" * 70)
    print(f"🚀 DubbingStory v{__version__} — Full Pipeline (mode: {mode})")
    print("=" * 70)

    cmd_ingest(args, cfg)
    cmd_segment(args, cfg)
    cmd_analyze(args, cfg)

    if mode == "summary":
        cmd_summary(args, cfg)
    else:
        cmd_narrate(args, cfg)
        cmd_dub(args, cfg)
        cmd_render(args, cfg)

    print("\n" + "=" * 70)
    print(f"✅ Pipeline selesai! (mode: {mode})")
    print("=" * 70)


# ==============================================================================
# SUBCOMMAND: summary (highlight recap pipeline)
# ==============================================================================

def cmd_summary(args, cfg):
    """Generate a highlight recap video from the most important scenes."""
    from dubbingstory.story import script_writer, subtitle_gen, scene_selector
    from dubbingstory.render import video_cutter, video_render
    from dubbingstory.tts import voice_manager

    project_name = _resolve_project_name(args)
    project_dir = _setup_project_dir(cfg, project_name)

    print("=" * 70)
    print(f"📋 DubbingStory v{__version__} — Summary Mode (Highlight Recap)")
    print("=" * 70)

    # ── Step 1: Load storyboard ──────────────────────────────────────────
    storyboard_path = os.path.join(project_dir, "storyboard.json")
    if not os.path.exists(storyboard_path):
        print(f"❌ Storyboard not found. Jalankan 'dubbingstory analyze' dulu.")
        sys.exit(1)

    with open(storyboard_path, "r", encoding="utf-8") as f:
        storyboard = json.load(f)

    # ── Step 2: Select best scenes ───────────────────────────────────────
    target_duration = getattr(cfg, "summary_target_duration", None)
    max_scenes = getattr(cfg, "summary_max_scenes", None)
    min_score = getattr(cfg, "summary_min_scene_score", 0.3)

    selected = scene_selector.select_scenes(
        storyboard=storyboard,
        target_duration=target_duration,
        max_scenes=max_scenes,
        min_score=min_score,
    )

    if not selected:
        print("❌ Tidak ada scene yang terpilih. Pipeline dihentikan.")
        sys.exit(1)

    manifest_path = scene_selector.save_summary_manifest(
        selected_scenes=selected,
        storyboard=storyboard,
        project_dir=project_dir,
        target_duration=target_duration,
    )
    print(f"   📄 Summary manifest: {manifest_path}")

    # ── Step 3: Cut & concat video ───────────────────────────────────────
    source_video = os.path.join(project_dir, "source.mp4")
    if not os.path.exists(source_video):
        print(f"❌ Source video not found: {source_video}")
        sys.exit(1)

    summary_video = os.path.join(project_dir, "summary_source.mp4")
    video_cutter.cut_and_concat(
        source_video=source_video,
        selected_scenes=selected,
        output_path=summary_video,
    )

    # Use the durations of the clips that actually made it into the concat.
    # This accounts for frame rounding and removes failed extractions before
    # narration, SRT, and TTS alignment are generated.
    durations_path = summary_video + ".durations.json"
    if os.path.exists(durations_path):
        with open(durations_path, "r", encoding="utf-8") as f:
            actual_durations = json.load(f)
        selected = [
            scene for scene in selected
            if scene["scene_id"] in actual_durations
        ]
        for scene in selected:
            scene["duration"] = actual_durations[scene["scene_id"]]
        print(f"   🧭 Using actual durations for {len(selected)} summary clips")
        manifest_path = scene_selector.save_summary_manifest(
            selected_scenes=selected,
            storyboard=storyboard,
            project_dir=project_dir,
            target_duration=target_duration,
        )

    # ── Step 4: Generate summary narration ────────────────────────────────
    languages = args.lang if hasattr(args, "lang") and args.lang else \
        getattr(cfg, "narration_languages", ["id", "en"])
    style = args.style if hasattr(args, "style") and args.style else \
        getattr(cfg, "narration_style", "viral_fb")

    narration = script_writer.generate_summary_narration(
        storyboard=storyboard,
        selected_scenes=selected,
        languages=languages,
        style=style,
        cfg=cfg,
    )

    # ── Remap timestamps to summary video timeline ─────────────────────
    # The narration segments have start_time/end_time from the ORIGINAL video
    # (e.g. scene_070 starts at 2633s). But the summary video is a concat of
    # selected clips only, so scene_070 might start at second ~180.
    # Build a mapping: scene_id → (summary_start, summary_end)
    summary_timeline = {}
    cumulative = 0.0
    for scene in selected:
        sid = scene["scene_id"]
        summary_timeline[sid] = {
            "start": cumulative,
            "end": cumulative + scene["duration"],
        }
        cumulative += scene["duration"]

    for lang, segments in narration.items():
        for seg in segments:
            sid = seg.get("scene_id", "")
            if sid in summary_timeline:
                seg["start_time"] = summary_timeline[sid]["start"]
                seg["end_time"] = summary_timeline[sid]["end"]

    # Save scripts
    scripts_dir = os.path.join(project_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    for lang, segments in narration.items():
        script_path = os.path.join(scripts_dir, f"summary_script_{lang}.txt")
        with open(script_path, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(
                    f"[{seg['scene_id']}] "
                    f"[{subtitle_gen._format_srt_time(seg.get('start_time', 0))} --> "
                    f"{subtitle_gen._format_srt_time(seg.get('end_time', 0))}] "
                    f"{seg['text']}\n\n"
                )

        srt_path = os.path.join(scripts_dir, f"summary_script_{lang}.srt")
        subtitle_gen.generate_srt(segments, srt_path)
        print(f"   📄 {script_path}")
        print(f"   📄 {srt_path}")

    # ── Step 5: TTS dubbing ──────────────────────────────────────────────
    engine_name = args.engine if hasattr(args, "engine") and args.engine else \
        getattr(cfg, "tts_engine", "edge")

    audio_dir = os.path.join(project_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # Generate audio from summary scripts
    voice_manager.generate_all_audio(
        scripts_dir=scripts_dir,
        audio_dir=audio_dir,
        engine_name=engine_name,
        cfg=cfg,
        script_prefix="summary_script_",
        audio_prefix="summary_audio_",
        segment_durations={
            scene["scene_id"]: scene["duration"]
            for scene in selected
        },
    )

    # Backup summary TTS audio
    import glob
    import shutil
    summary_audio_files = glob.glob(os.path.join(audio_dir, "summary_audio_*.wav"))
    for audio_path in summary_audio_files:
        basename = os.path.basename(audio_path)
        lang = basename.replace("summary_audio_", "").replace(".wav", "")
        backup_path = os.path.join(project_dir, f"backup_tts_summary_{lang}.wav")
        shutil.copy2(audio_path, backup_path)
        print(f"   💾 Summary TTS Backup saved to: {backup_path}")

    # ── Step 6: Render final summary video ───────────────────────────────
    ratios = getattr(cfg, "render_ratios", ["16:9"])
    audio_strategy = getattr(cfg, "render_audio_strategy", "mute_original")
    burn_subs = getattr(cfg, "render_burn_subtitles", False)

    import glob
    summary_audio_files = glob.glob(os.path.join(audio_dir, "summary_audio_*.wav"))

    for audio_path in summary_audio_files:
        basename = os.path.basename(audio_path)
        lang = basename.replace("summary_audio_", "").replace(".wav", "")

        srt_path = os.path.join(scripts_dir, f"summary_script_{lang}.srt")

        for ratio in ratios:
            ratio_label = ratio.replace(":", "x")
            output_path = os.path.join(project_dir, f"final_summary_{lang}_{ratio_label}.mp4")

            print(f"\n   🎬 Rendering Summary: {lang.upper()} ({ratio})...")

            try:
                video_render.render_final(
                    source_video=summary_video,
                    narration_audio=audio_path,
                    output_path=output_path,
                    subtitle_path=srt_path if os.path.exists(srt_path) else None,
                    audio_strategy=audio_strategy,
                    original_volume=0.1,
                    burn_subs=burn_subs,
                    target_ratio=ratio if ratio != "16:9" else None,
                )
                print(f"   ✅ {output_path}")
            except Exception as e:
                print(f"   ❌ Summary render failed: {e}")

    print(f"\n✅ Summary pipeline selesai!")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="dubbingstory",
        description="DubbingStory — Automated Video Narration & Dubbing Pipeline",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", type=str, default=None, help="Custom YAML config path")

    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # ── run ───────────────────────────────────────────────────────────────
    p_run = subparsers.add_parser("run", help="Full pipeline")
    p_run.add_argument("--input", "-i", type=str, help="Path to local video file")
    p_run.add_argument("--url", "-u", type=str, help="YouTube/video URL")
    p_run.add_argument("--i-have-rights", action="store_true", help="Confirm video rights")
    p_run.add_argument("--project", "-p", type=str, help="Project name")
    p_run.add_argument("--style", type=str, default=None,
                        choices=["viral_fb", "documentary", "technical", "calm_educational"])
    p_run.add_argument("--lang", nargs="+", default=None, help="Languages (e.g., id en)")
    p_run.add_argument("--engine", type=str, default=None, choices=["edge", "piper"], help="TTS engine (default: edge)")
    p_run.add_argument("--voice-id", type=str, default=None, help="Specific voice name/ID for Indonesian (e.g., id-ID-ArdiNeural or id_ID-news_tts-medium)")
    p_run.add_argument("--voice-en", type=str, default=None, help="Specific voice name/ID for English")
    p_run.add_argument("--ratio", nargs="+", default=None, help="Output ratios (16:9, 9:16)")
    p_run.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "summary"],
        help="Pipeline mode: 'full' (dub entire video) or 'summary' (highlight recap)"
    )
    p_run.add_argument(
        "--summary-duration", type=int, default=None,
        help="Target duration for summary mode in seconds (default: auto ~60-120s)"
    )
    p_run.add_argument(
        "--summary-max-scenes", type=int, default=None,
        help="Maximum number of scenes to include in summary (default: auto)"
    )
    p_run.add_argument(
        "--vision-provider", type=str, default=None,
        choices=["gemini", "openai"],
        help="Vision provider: 'gemini' (default) or 'openai' (Qwen3-VL via vLLM)"
    )
    p_run.add_argument(
        "--vision-model", type=str, default=None,
        help="Vision model name override (e.g. 'Qwen/Qwen3-VL-4B-Instruct')"
    )
    p_run.add_argument(
        "--vision-base-url", type=str, default=None,
        help="Vision API base URL (e.g. 'http://127.0.0.1:8000/v1')"
    )
    p_run.add_argument(
        "--vision-max-tokens", type=int, default=None,
        help="Vision max output tokens (default: 2048)"
    )
    p_run.add_argument(
        "--quality", "-q", type=str, default=None,
        choices=["720", "1080", "2k", "4k", "max"],
        help="Download quality (default: 1080). Options: 720, 1080, 2k, 4k, max"
    )
    p_run.add_argument(
        "--yt-cookies", type=str, default=None,
        help="Cookies from browser (e.g. 'chrome', 'firefox') or path to cookies.txt for yt-dlp"
    )
    p_run.add_argument(
        "--use-asr",
        action="store_true",
        help="If no subtitles are found, use Whisper ASR to generate a transcript.",
    )
    p_run.add_argument(
        "--whisper-model",
        type=str,
        default=None,
        help="Faster-Whisper model size (tiny, base, small, medium, large, large-v3).",
    )
    p_run.add_argument(
        "--whisper-device",
        type=str,
        default=None,
        help="Device for Faster-Whisper inference (cpu/cuda).",
    )
    p_run.add_argument(
        "--whisper-compute-type",
        type=str,
        default=None,
        help="Compute type for Faster-Whisper (float16, int8, etc.).",
    )
    # ── Pipeline speed optimization flags ──
    p_run.add_argument(
        "--max-keyframes", type=int, default=None,
        help="Max keyframes per scene for vision analysis (default: 3). "
             "Lower = faster but less accurate. Recommended: 3 for speed."
    )
    p_run.add_argument(
        "--min-scene-duration", type=float, default=None,
        help="Minimum scene duration in seconds (default: 2.0). "
             "Higher = fewer scenes = faster. Recommended: 4-6 for speed."
    )
    p_run.add_argument(
        "--scene-threshold", type=float, default=None,
        help="Scene detection sensitivity (default: 3.0). "
             "Higher = fewer scenes detected = faster. Recommended: 4-6 for speed."
    )

    # ── ingest ────────────────────────────────────────────────────────────
    p_ingest = subparsers.add_parser("ingest", help="Download/validate video")
    p_ingest.add_argument("--input", "-i", type=str, help="Path to local video file")
    p_ingest.add_argument("--url", "-u", type=str, help="YouTube/video URL")
    p_ingest.add_argument("--i-have-rights", action="store_true")
    p_ingest.add_argument("--project", "-p", type=str)
    p_ingest.add_argument(
        "--quality", "-q", type=str, default=None,
        choices=["720", "1080", "2k", "4k", "max"],
        help="Download quality (default: 1080). Options: 720, 1080, 2k, 4k, max"
    )
    p_ingest.add_argument(
        "--yt-cookies", type=str, default=None,
        help="Cookies from browser (e.g. 'chrome', 'firefox') or path to cookies.txt for yt-dlp"
    )
    p_ingest.add_argument(
        "--use-asr",
        action="store_true",
        help="If no subtitles are found, transcribe audio with Whisper ASR "
             "to generate a transcript fallback.",
    )
    p_ingest.add_argument(
        "--whisper-model",
        type=str,
        default=None,
        help="Faster-Whisper model size (tiny, base, small, medium, large, large-v3). "
             "Overrides config.whisper.model.",
    )
    p_ingest.add_argument(
        "--whisper-device",
        type=str,
        default=None,
        help="Device for Faster-Whisper inference (cpu/cuda). Overrides config.whisper.device.",
    )
    p_ingest.add_argument(
        "--whisper-compute-type",
        type=str,
        default=None,
        help="Compute type for Faster-Whisper (float16, int8, etc.). "
             "Overrides config.whisper.compute_type.",
    )

    # ── segment ───────────────────────────────────────────────────────────
    p_segment = subparsers.add_parser("segment", help="Detect scenes + keyframes")
    p_segment.add_argument("--project", "-p", type=str, required=True)
    p_segment.add_argument(
        "--max-keyframes", type=int, default=None,
        help="Max keyframes per scene (default: 3). Lower = faster."
    )
    p_segment.add_argument(
        "--min-scene-duration", type=float, default=None,
        help="Minimum scene duration in seconds (default: 2.0). Higher = fewer scenes."
    )
    p_segment.add_argument(
        "--scene-threshold", type=float, default=None,
        help="Scene detection sensitivity (default: 3.0). Higher = fewer scenes."
    )

    # ── analyze ───────────────────────────────────────────────────────────
    p_analyze = subparsers.add_parser("analyze", help="Visual understanding")
    p_analyze.add_argument("--project", "-p", type=str, required=True)
    p_analyze.add_argument("--domain", type=str, default="",
                            help="Domain hint (workshop, cooking, etc.)")
    p_analyze.add_argument(
        "--vision-provider", type=str, default=None,
        choices=["gemini", "openai"],
        help="Vision provider: 'gemini' (default) or 'openai' (Qwen3-VL via vLLM)"
    )
    p_analyze.add_argument(
        "--vision-model", type=str, default=None,
        help="Vision model name override (e.g. 'Qwen/Qwen3-VL-4B-Instruct')"
    )
    p_analyze.add_argument(
        "--vision-base-url", type=str, default=None,
        help="Vision API base URL (e.g. 'http://127.0.0.1:8000/v1')"
    )
    p_analyze.add_argument(
        "--vision-max-tokens", type=int, default=None,
        help="Vision max output tokens (default: 2048)"
    )

    # ── narrate ───────────────────────────────────────────────────────────
    p_narrate = subparsers.add_parser("narrate", help="Generate narration scripts")
    p_narrate.add_argument("--project", "-p", type=str, required=True)
    p_narrate.add_argument("--style", type=str, default=None,
                            choices=["viral_fb", "documentary", "technical", "calm_educational"])
    p_narrate.add_argument("--lang", nargs="+", default=None)

    # ── dub ───────────────────────────────────────────────────────────────
    p_dub = subparsers.add_parser("dub", help="Generate TTS audio")
    p_dub.add_argument("--project", "-p", type=str, required=True)
    p_dub.add_argument("--engine", type=str, default=None, choices=["edge", "piper"], help="TTS engine (default: edge)")
    p_dub.add_argument("--voice-id", type=str, default=None, help="Specific voice name/ID for Indonesian")
    p_dub.add_argument("--voice-en", type=str, default=None, help="Specific voice name/ID for English")

    # ── render ────────────────────────────────────────────────────────────
    p_render = subparsers.add_parser("render", help="Render final video")
    p_render.add_argument("--project", "-p", type=str, required=True)
    p_render.add_argument("--ratio", nargs="+", default=None)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Build config
    cfg = build_config(config_path=args.config)

    # Apply CLI overrides
    if hasattr(args, "ratio") and args.ratio:
        cfg.render_ratios = args.ratio
    if hasattr(args, "asr_model") and args.asr_model:
        cfg.asr_model_name = args.asr_model
    if hasattr(args, "voice_id") and args.voice_id:
        cfg.tts_voice_id = args.voice_id
    if hasattr(args, "voice_en") and args.voice_en:
        cfg.tts_voice_en = args.voice_en
    if hasattr(args, "vision_provider") and args.vision_provider:
        cfg.vision_provider = args.vision_provider
    if hasattr(args, "vision_model") and args.vision_model:
        cfg.vision_openai_model = args.vision_model
    if hasattr(args, "vision_base_url") and args.vision_base_url:
        cfg.vision_openai_base_url = args.vision_base_url
    if hasattr(args, "vision_max_tokens") and args.vision_max_tokens is not None:
        cfg.vision_openai_max_tokens = args.vision_max_tokens

    # Summary mode overrides
    if hasattr(args, "summary_duration") and args.summary_duration is not None:
        cfg.summary_target_duration = args.summary_duration
    if hasattr(args, "summary_max_scenes") and args.summary_max_scenes is not None:
        cfg.summary_max_scenes = args.summary_max_scenes

    # Pipeline speed optimization overrides
    if hasattr(args, "max_keyframes") and args.max_keyframes is not None:
        cfg.keyframes_max_per_scene = args.max_keyframes
    if hasattr(args, "min_scene_duration") and args.min_scene_duration is not None:
        cfg.segment_min_scene_duration = args.min_scene_duration
    if hasattr(args, "scene_threshold") and args.scene_threshold is not None:
        cfg.segment_adaptive_threshold = args.scene_threshold

    # Dispatch
    commands = {
        "run": cmd_run,
        "ingest": cmd_ingest,
        "segment": cmd_segment,
        "analyze": cmd_analyze,
        "narrate": cmd_narrate,
        "dub": cmd_dub,
        "render": cmd_render,
        "summary": cmd_summary,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args, cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
