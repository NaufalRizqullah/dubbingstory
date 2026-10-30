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
import math

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
        from dubbingstory.ingest.sources import source_project_name
        return source_project_name(args.url)
    return "untitled"


def _strict_output(cfg) -> bool:
    """Enhanced outputs require a complete, validated narration timeline."""
    return (
        "source" in (getattr(cfg, "render_ratios", []) or [])
        or getattr(cfg, "subtitle_style", "classic") in {"word", "karaoke"}
        or getattr(cfg, "tts_narration_fit", "legacy") != "legacy"
        or bool(getattr(cfg, "narration_action_sync", False))
    )


def _video_timeline_duration(video_path: str, *, required: bool = False) -> float | None:
    from dubbingstory.ingest.local_video import get_video_metadata

    if not os.path.isfile(video_path):
        if required:
            raise FileNotFoundError(f"Source video required for timeline validation: {video_path}")
        return None
    metadata = get_video_metadata(video_path)
    if required:
        from dubbingstory.ingest.local_video import require_video_timeline
        metadata = require_video_timeline(video_path, metadata)
    duration = float(metadata.get("video_duration") or metadata.get("duration") or 0)
    if not math.isfinite(duration) or duration <= 0:
        if required:
            raise ValueError("Cannot determine the source video duration; check the media and FFprobe.")
        return None
    return duration


def _refresh_speech_subtitles(scripts_dir: str, results: dict, prefix: str = "script_") -> None:
    """SRT follows completed synthesis, including rewrites and actual placement."""
    from dubbingstory.story.subtitle_gen import generate_srt, generate_srt_from_words

    for language, result in results.items():
        subtitle_path = os.path.join(scripts_dir, f"{prefix}{language}.srt")
        if result.get("word_timings"):
            generate_srt_from_words(result["word_timings"], subtitle_path)
            continue
        segments = result.get("final_segments")
        if not segments:
            continue  # Legacy plain-text outputs retain their existing subtitle behavior.
        placement_map = {item["scene_id"]: item for item in result.get("placements", [])}
        timed = []
        for segment in segments:
            item = dict(segment)
            placement = placement_map.get(item.get("scene_id"))
            if placement:
                item["start_time"] = placement["actual_start"]
                item["end_time"] = placement["actual_end"]
                timed.append(item)
        if timed:
            generate_srt(timed, subtitle_path)


# ==============================================================================
# SUBCOMMAND: ingest
# ==============================================================================

def cmd_ingest(args, cfg):
    """Download/validate video and read subtitles."""
    from dubbingstory.ingest import local_video, youtube, subtitle_reader
    from dubbingstory.ingest.sources import download_source, normalize_source_url

    if bool(args.input) == bool(args.url):
        raise ValueError("Provide exactly one of --input or --url.")
    if args.url and not getattr(args, "i_have_rights", False):
        raise ValueError("Tambahkan --i-have-rights untuk mengonfirmasi hak penggunaan video URL.")
    source = normalize_source_url(args.url) if args.url else None

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
        # All remote sources have already passed the rights gate above.
        print(f"   Source: {source.provider} → {source.canonical_url}")
        download_q = getattr(args, "quality", None) or getattr(cfg, "ingest_download_height", "1080")
        video_path = download_source(
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

    narration_source = getattr(cfg, "narration_source", "visual")
    if narration_source not in {"visual", "audio"}:
        raise ValueError("narration.source must be visual or audio")

    # Keep ingesting local subtitle files for backwards-compatible artifacts.
    # Visual mode suppresses this context at analysis time; audio mode opts in.
    if getattr(cfg, "ingest_read_subtitles", True):
        subtitle_data = subtitle_reader.find_and_read_subtitles(
            args.input or video_path, project_dir
        )
        if subtitle_data:
            context_source = subtitle_data.get("source", "manual")

    # Step 2: If no local subs and source is YouTube, try downloading subs
    if subtitle_data is None and source and source.provider == "youtube":
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
        narration_source == "audio"
        and subtitle_data is None
        and getattr(cfg, "ingest_use_asr_fallback", True)
        and (getattr(args, "use_asr", False) or narration_source == "audio")
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
        "rights_confirmed": bool(getattr(args, "i_have_rights", False)) if source else None,
    }
    metadata_path = os.path.join(project_dir, "video_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        manifest["source"] = {
            key: metadata[key]
            for key in ("provider", "source_id", "canonical_url", "request_identity", "sha256")
            if key in metadata
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
            max_duration=getattr(cfg, "segment_max_scene_duration", 15.0),
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
            max_duration=getattr(cfg, "segment_max_scene_duration", 15.0),
            merge_short=getattr(cfg, "segment_merge_short_scenes", True),
        )

    # Keyframe extraction
    keyframes_dir = os.path.join(project_dir, "keyframes")
    all_keyframes = keyframes.extract_all_scenes(
        scenes=scene_list,
        scenes_dir=os.path.join(project_dir, "scenes"),
        output_dir=keyframes_dir,
        strategy=getattr(cfg, "keyframes_strategy", "distributed"),
        interval_seconds=getattr(cfg, "keyframes_interval_seconds", 2.0),
        max_per_scene=getattr(cfg, "keyframes_max_per_scene", 7),
        source_video=video_path,
        max_edge=getattr(cfg, "keyframes_max_edge", 768),
        jpeg_quality=getattr(cfg, "keyframes_jpeg_quality", 88),
        shared_boundary_anchor=bool(getattr(cfg, "keyframes_shared_boundary_anchor", True)),
        adaptive=getattr(cfg, "keyframes_adaptive", True),
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
        if getattr(cfg, "narration_source", "visual") == "audio":
            subtitle_context = ingest_data.get("subtitle_data")

    storyboard = scene_understanding.run_analysis(
        segment_data=segment_data,
        project_dir=project_dir,
        cfg=cfg,
        subtitle_context=subtitle_context,
        domain_hint=getattr(cfg, "vision_domain_hint", ""),
    )

    if getattr(cfg, "vision_temporal_enabled", False) and getattr(args, "mode", "full") == "full":
        from dubbingstory.vision.temporal_analysis import run_temporal_analysis
        storyboard = run_temporal_analysis(segment_data, storyboard, project_dir, cfg)

    # Save storyboard
    storyboard_path = os.path.join(project_dir, "storyboard.json")
    with open(storyboard_path, "w", encoding="utf-8") as f:
        json.dump(storyboard, f, ensure_ascii=False, indent=2)

    # V2: transform local scene facts into persistent global story artifacts.
    # This runs after vision so narration/selection can reason over the whole
    # story instead of independently describing one scene at a time.
    try:
        from dubbingstory.story.story_planner import generate_story_artifacts

        generate_story_artifacts(
            storyboard,
            subtitle_context=subtitle_context,
            cfg=cfg,
            project_dir=project_dir,
        )
        print("   🧠 Story artifacts: scene_cards.json, story_plan.json, story_memory.json")
    except Exception as exc:
        from dubbingstory.story.gemini_json import raise_if_strict_text_error

        raise_if_strict_text_error(cfg, exc, "story-artifact-generation")
        # Analysis remains usable even if the optional global planner fails.
        print(f"   ⚠️ Story artifact generation failed; narration will use fallback context: {exc}")

    print(f"\n✅ Visual analysis selesai!")
    print(f"   Storyboard: {storyboard_path}")

    return storyboard


# ==============================================================================
# SUBCOMMAND: narrate
# ==============================================================================

def cmd_narrate(args, cfg):
    """Generate continuity-aware narration scripts."""
    from dubbingstory.story import script_writer, subtitle_gen
    from dubbingstory.story.narration_planner import build_narration_plan
    from dubbingstory.story.narration_qa import evaluate_narration
    from dubbingstory.story.story_planner import load_story_artifacts

    project_name = _resolve_project_name(args)
    project_dir = _setup_project_dir(cfg, project_name)

    print("=" * 70)
    print(f"📝 DubbingStory v{__version__} — Narration")
    print("=" * 70)

    storyboard_path = os.path.join(project_dir, "storyboard.json")
    if not os.path.exists(storyboard_path):
        print("❌ Storyboard not found. Jalankan 'dubbingstory analyze' dulu.")
        sys.exit(1)

    with open(storyboard_path, "r", encoding="utf-8") as f:
        storyboard = json.load(f)

    visual_proof = {}
    if getattr(cfg, "narration_action_sync", False):
        from dubbingstory.tts.word_alignment import file_sha256
        timeline_path = os.path.join(project_dir, "visual_timeline.json")
        if not os.path.isfile(timeline_path):
            raise ValueError("Action synchronization requires complete temporal analysis; rerun analyze with --temporal-analysis")
        with open(timeline_path, encoding="utf-8") as handle:
            visual_timeline = json.load(handle)
        source_hash = file_sha256(os.path.join(project_dir, "source.mp4"))
        if (visual_timeline.get("status") != "complete"
                or visual_timeline.get("provenance", {}).get("source_sha256") != source_hash
                or storyboard.get("visual_timeline", {}).get("source_sha256") != source_hash
                or storyboard.get("visual_timeline", {}).get("artifact_sha256") != file_sha256(timeline_path)):
            raise ValueError("Visual timeline is incomplete or belongs to another source; rerun analyze before narrate")
        visual_proof = {"visual_source_sha256": source_hash,
                        "visual_timeline_sha256": file_sha256(timeline_path)}

    languages = args.lang if hasattr(args, "lang") and args.lang else \
        getattr(cfg, "narration_languages", ["id", "en"])
    style = args.style if hasattr(args, "style") and args.style else \
        getattr(cfg, "narration_style", "viral_fb")

    print(f"   Style: {style}")
    print(f"   Languages: {languages}")

    story_plan, story_memory = load_story_artifacts(project_dir, storyboard)
    scenes = storyboard.get("scenes", []) or []
    if _strict_output(cfg):
        cfg.narration_source_duration = _video_timeline_duration(
            os.path.join(project_dir, "source.mp4"), required=True,
        )
    narration_plan = build_narration_plan(
        storyboard,
        scenes,
        story_plan=story_plan,
        story_memory=story_memory,
        cfg=cfg,
        mode="full",
    )
    plan_path = os.path.join(project_dir, "narration_plan.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(narration_plan, f, ensure_ascii=False, indent=2)
    print(f"   🧭 Narration plan: {plan_path}")

    narration = script_writer.generate_narration(
        storyboard=storyboard,
        languages=languages,
        style=style,
        cfg=cfg,
        story_plan=story_plan,
        story_memory=story_memory,
        narration_plan=narration_plan,
    )

    scripts_dir = os.path.join(project_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    qa_by_language = {}

    for lang, segments in narration.items():
        script_path = os.path.join(scripts_dir, f"script_{lang}.txt")
        with open(script_path, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(f"[{seg['scene_id']}] {seg['text']}\n\n")

        # Structured copy is consumed by the V2 TTS timeline scheduler.
        structured_path = os.path.join(scripts_dir, f"script_{lang}.json")
        with open(structured_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mode": "full",
                    "language": lang,
                    "total_duration": narration_plan.get("total_duration", 0),
                    "target_wpm": narration_plan.get("target_wpm", 160),
                    "action_sync_enabled": narration_plan.get("action_sync_enabled", False),
                    **visual_proof,
                    "segments": segments,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        srt_path = os.path.join(scripts_dir, f"script_{lang}.srt")
        subtitle_gen.generate_srt(segments, srt_path)
        qa_by_language[lang] = evaluate_narration(segments, narration_plan, language=lang)

        print(f"   📄 {script_path}")
        print(f"   📄 {structured_path}")
        print(f"   📄 {srt_path}")

    qa_path = os.path.join(project_dir, "narration_qa.json")
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_by_language, f, ensure_ascii=False, indent=2)
    print(f"   🧪 Narration QA: {qa_path}")
    print("\n✅ Narration selesai!")
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

    audio_results = voice_manager.generate_all_audio(
        scripts_dir=scripts_dir,
        audio_dir=audio_dir,
        engine_name=engine_name,
        cfg=cfg,
        timeline_duration=_video_timeline_duration(
            os.path.join(project_dir, "source.mp4"), required=_strict_output(cfg),
        ),
        timeline_kind="source",
        timeline_source_path=os.path.join(project_dir, "source.mp4"),
    )
    if not audio_results and _strict_output(cfg):
        raise RuntimeError("No narration audio was produced. Generate structured scripts before dubbing.")
    _refresh_speech_subtitles(scripts_dir, audio_results)

    # Backup TTS audio to project dir
    import shutil
    audio_files = [result["concat_path"] for result in audio_results.values()]
    for audio_path in audio_files:
        basename = os.path.basename(audio_path)
        lang = basename.replace("audio_", "").replace(".wav", "")
        backup_path = os.path.join(project_dir, f"backup_tts_full_{lang}.wav")
        shutil.copy2(audio_path, backup_path)
        print(f"   💾 TTS Backup saved to: {backup_path}")

    print(f"\n✅ TTS dubbing selesai!")
    print(f"   Audio: {audio_dir}")
    return audio_results


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

    rendered = video_render.render_project(
        project_dir=project_dir,
        ratios=ratios,
        cfg=cfg,
        mode=getattr(args, "mode", "full"),
    )
    if not rendered and _strict_output(cfg):
        raise RuntimeError("No final videos were rendered. Complete dubbing before rendering.")

    print(f"\n✅ Render selesai!")
    return rendered


def cmd_reframe(args, cfg):
    """Recompose an exported video without analyzing or synthesizing it again."""
    from dubbingstory.render import video_render

    output_path = video_render.reframe_video(
        args.input, args.output, target_ratio="9:16", cfg=cfg,
    )
    print(f"\n✅ Reframe selesai: {output_path}")
    return output_path


# ==============================================================================
# SUBCOMMAND: run (full pipeline)
# ==============================================================================

def cmd_run(args, cfg):
    """Run the full pipeline end-to-end."""
    mode = getattr(args, "mode", "full")
    # scene_understanding reads cfg.mode to activate summary two-pass analysis.
    cfg.mode = mode

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
    """Generate a V2.1 evidence-aware recap with legacy-safe fallback."""
    from dubbingstory.story import script_writer, subtitle_gen, scene_selector
    from dubbingstory.story.beat_planner import plan_story_beats
    from dubbingstory.story.narration_planner import build_narration_plan
    from dubbingstory.story.narration_qa import evaluate_narration
    from dubbingstory.story.semantic_matcher import (
        make_similarity_engine,
        match_beats,
        selected_scenes_from_report,
        suggest_promotion_candidates,
    )
    from dubbingstory.story.semantic_verifier import verify_narration, repair_unsupported_beats
    from dubbingstory.story.story_planner import (
        load_evidence_artifacts,
        load_story_artifacts,
        rebuild_evidence_artifacts,
    )
    from dubbingstory.render import video_cutter, video_render
    from dubbingstory.tts import voice_manager
    import copy
    import shutil

    project_name = _resolve_project_name(args)
    project_dir = _setup_project_dir(cfg, project_name)

    print("=" * 70)
    print(f"📋 DubbingStory v{__version__} — Summary Mode (V2.1 Evidence-Aware Recap)")
    print("=" * 70)

    storyboard_path = os.path.join(project_dir, "storyboard.json")
    if not os.path.exists(storyboard_path):
        print("❌ Storyboard not found. Jalankan 'dubbingstory analyze' dulu.")
        sys.exit(1)
    with open(storyboard_path, "r", encoding="utf-8") as f:
        storyboard = json.load(f)

    story_plan, story_memory = load_story_artifacts(project_dir, storyboard)
    _, _, signatures = load_evidence_artifacts(project_dir, storyboard)

    # Resolve the same automatic budgets as the legacy selector so shadow/active
    # comparisons are apples-to-apples.
    total_duration = float(storyboard.get("total_duration", 0) or 0)
    requested_target = getattr(cfg, "summary_target_duration", None)
    target_duration = int(requested_target) if requested_target is not None else int(max(60, min(total_duration * 0.12, 120)))
    requested_max = getattr(cfg, "summary_max_scenes", None)
    max_scenes = int(requested_max) if requested_max is not None else max(3, min(int(target_duration // 10), 18))
    min_score = getattr(cfg, "summary_min_scene_score", 0.3)
    duration_tolerance = float(getattr(cfg, "summary_duration_tolerance", 1.05) or 1.05)

    # Legacy selection is intentionally preserved. It is both the A/B reference
    # and the automatic safety path if semantic evidence is not strong enough.
    legacy_selected = scene_selector.select_scenes(
        storyboard=storyboard,
        target_duration=target_duration,
        max_scenes=max_scenes,
        min_score=min_score,
        story_plan=story_plan,
        duration_tolerance=duration_tolerance,
    )
    if not legacy_selected:
        print("❌ Tidak ada scene yang terpilih. Pipeline dihentikan.")
        sys.exit(1)

    semantic_enabled = bool(getattr(cfg, "semantic_match_enabled", True))
    semantic_shadow = bool(getattr(cfg, "semantic_match_shadow_mode", False))
    mode_override = getattr(args, "semantic_match", None) if hasattr(args, "semantic_match") else None
    if mode_override == "off":
        semantic_enabled = False
    elif mode_override == "shadow":
        semantic_enabled = True
        semantic_shadow = True
    elif mode_override == "active":
        semantic_enabled = True
        semantic_shadow = False

    beats = []
    match_report = {
        "version": "v2.1.0",
        "mode": "disabled",
        "legacy_selected_scene_ids": [s.get("scene_id") for s in legacy_selected],
        "selected_scene_ids": [s.get("scene_id") for s in legacy_selected],
        "metrics": {"legacy_fallback_rate": 1.0},
    }
    selected = legacy_selected
    semantic_active = False
    engine = None

    if semantic_enabled:
        print(f"   🧩 V2.1 semantic matcher: {'SHADOW' if semantic_shadow else 'ACTIVE'}")
        engine = make_similarity_engine(cfg, project_dir=project_dir)
        beats = plan_story_beats(
            story_plan=story_plan,
            story_memory=story_memory,
            signatures=signatures,
            target_duration=target_duration,
            cfg=cfg,
        )
        beats_path = os.path.join(project_dir, "beats.json")
        with open(beats_path, "w", encoding="utf-8") as f:
            json.dump(beats, f, ensure_ascii=False, indent=2)
        print(f"   🧭 Story-first beats: {len(beats)} → {beats_path}")

        if beats:
            match_report = match_beats(
                beats=beats,
                signatures=signatures,
                storyboard=storyboard,
                story_plan=story_plan,
                cfg=cfg,
                target_duration=target_duration,
                max_scenes=max_scenes,
                project_dir=project_dir,
                engine=engine,
            )

            # Weak beats may promote a small number of screening-only windows to
            # deep Qwen analysis. The promotion reuses the existing vision stack,
            # including structured schema and length/runaway recovery.
            promoted_ids = []
            promotion_proposals = {}
            if bool(getattr(cfg, "semantic_match_on_demand_promotion", True)) and match_report.get("weak_beat_ids"):
                promotion_proposals = suggest_promotion_candidates(
                    beats=beats,
                    match_report=match_report,
                    storyboard=storyboard,
                    cfg=cfg,
                    project_dir=project_dir,
                    engine=engine,
                )
                promotion_limit = max(0, int(getattr(cfg, "semantic_match_max_promotions", 6) or 6))
                requested = []
                for beat in beats:
                    for sid in promotion_proposals.get(str(beat.get("beat_id")), []) or []:
                        if sid not in requested:
                            requested.append(sid)
                        if len(requested) >= promotion_limit:
                            break
                    if len(requested) >= promotion_limit:
                        break
                if requested:
                    from dubbingstory.vision.scene_understanding import promote_scenes_on_demand

                    print(f"   🔎 Weak match recovery: promoting {len(requested)} scene window(s) to deep vision")
                    storyboard, promoted_ids = promote_scenes_on_demand(
                        project_dir=project_dir,
                        storyboard=storyboard,
                        scene_ids=requested,
                        cfg=cfg,
                        domain_hint=getattr(cfg, "vision_domain_hint", ""),
                    )
                    if promoted_ids:
                        with open(storyboard_path, "w", encoding="utf-8") as f:
                            json.dump(storyboard, f, ensure_ascii=False, indent=2)
                        ingest_path = os.path.join(project_dir, "ingest_manifest.json")
                        subtitle_context = None
                        if os.path.exists(ingest_path):
                            try:
                                with open(ingest_path, "r", encoding="utf-8") as f:
                                    subtitle_context = (json.load(f) or {}).get("subtitle_data")
                            except Exception:
                                subtitle_context = None
                        _, _, signatures = rebuild_evidence_artifacts(
                            storyboard,
                            subtitle_context=subtitle_context,
                            project_dir=project_dir,
                        )
                        match_report = match_beats(
                            beats=beats,
                            signatures=signatures,
                            storyboard=storyboard,
                            story_plan=story_plan,
                            cfg=cfg,
                            target_duration=target_duration,
                            max_scenes=max_scenes,
                            project_dir=project_dir,
                            engine=engine,
                        )
            match_report["promotion_proposals"] = promotion_proposals
            match_report["promoted_scene_ids"] = promoted_ids
            match_report.setdefault("metrics", {})["on_demand_deep_promotions"] = len(promoted_ids)
            match_report["legacy_selected_scene_ids"] = [s.get("scene_id") for s in legacy_selected]

            proposed = selected_scenes_from_report(match_report, storyboard)
            # Attach beat/match metadata to selected scene copies for the manifest.
            scene_meta = {}
            for beat_report in match_report.get("beats", []) or []:
                for sid in beat_report.get("selected", []) or []:
                    meta = scene_meta.setdefault(str(sid), {"beats": [], "score": 0.0})
                    meta["beats"].append(str(beat_report.get("beat_id")))
                    meta["score"] = max(meta["score"], float(beat_report.get("prewrite_score", 0) or 0))
            enriched = []
            for scene in proposed:
                item = dict(scene)
                meta = scene_meta.get(str(item.get("scene_id")), {})
                item["matched_beat_ids"] = meta.get("beats", [])
                item["semantic_match_score"] = round(float(meta.get("score", 0) or 0), 4)
                enriched.append(item)
            proposed = enriched

            coverage = float(match_report.get("beat_coverage", 0) or 0)
            proposed_duration = sum(float(s.get("duration", 0) or 0) for s in proposed)
            min_coverage = float(getattr(cfg, "semantic_match_min_beat_coverage", 0.75) or 0.75)
            min_duration_ratio = float(getattr(cfg, "semantic_match_min_duration_ratio", 0.60) or 0.60)
            fallback_reason = None
            if not proposed:
                fallback_reason = "no_semantic_selection"
            elif coverage < min_coverage:
                fallback_reason = f"beat_coverage_{coverage:.2f}_below_{min_coverage:.2f}"
            elif proposed_duration < target_duration * min_duration_ratio:
                fallback_reason = f"duration_{proposed_duration:.1f}_below_min_ratio_{min_duration_ratio:.2f}"

            if semantic_shadow:
                match_report["activation"] = "shadow"
                selected = legacy_selected
            elif fallback_reason:
                print(f"   ⚠️ Semantic selection fallback → legacy selector ({fallback_reason})")
                match_report["activation"] = "legacy_fallback"
                match_report["fallback_reason"] = fallback_reason
                selected = legacy_selected
            else:
                match_report["activation"] = "active"
                selected = proposed
                semantic_active = True
                print(
                    f"   ✅ Semantic selection active: {len(selected)} scenes / "
                    f"{proposed_duration:.1f}s, beat coverage={coverage:.0%}"
                )

            match_report.setdefault("metrics", {})["legacy_fallback_rate"] = 0.0 if semantic_active else 1.0

    match_path = os.path.join(project_dir, "match_report.json")
    with open(match_path, "w", encoding="utf-8") as f:
        json.dump(match_report, f, ensure_ascii=False, indent=2)
    print(f"   🧪 Match report: {match_path}")

    manifest_path = scene_selector.save_summary_manifest(
        selected_scenes=selected,
        storyboard=storyboard,
        project_dir=project_dir,
        target_duration=target_duration,
    )
    print(f"   📄 Summary manifest: {manifest_path}")

    # Cut only after beat->scene matching has fixed the footage assignment.
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

    durations_path = summary_video + ".durations.json"
    if os.path.exists(durations_path):
        with open(durations_path, "r", encoding="utf-8") as f:
            actual_durations = json.load(f)
        selected = [scene for scene in selected if scene["scene_id"] in actual_durations]
        for scene in selected:
            scene["duration"] = float(actual_durations[scene["scene_id"]])
        print(f"   🧭 Using actual durations for {len(selected)} summary clips")
        scene_selector.save_summary_manifest(
            selected_scenes=selected,
            storyboard=storyboard,
            project_dir=project_dir,
            target_duration=target_duration,
        )

    # Narration planner is now a scheduler. With active semantic matching it
    # consumes the fixed beat->scene assignments instead of inventing new beats.
    narration_plan = build_narration_plan(
        storyboard,
        selected,
        story_plan=story_plan,
        story_memory=story_memory,
        cfg=cfg,
        mode="summary",
        story_beats=beats if semantic_active else None,
        match_report=match_report if semantic_active else None,
    )
    plan_path = os.path.join(project_dir, "narration_plan.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(narration_plan, f, ensure_ascii=False, indent=2)
    print(
        f"   🧭 Narration plan: {narration_plan.get('total_target_words', 0)} target words / "
        f"{narration_plan.get('total_duration', 0):.1f}s"
    )

    languages = args.lang if hasattr(args, "lang") and args.lang else getattr(cfg, "narration_languages", ["id", "en"])
    style = args.style if hasattr(args, "style") and args.style else getattr(cfg, "narration_style", "viral_fb")
    print(f"   Languages: {languages}")

    narration = script_writer.generate_summary_narration(
        storyboard=storyboard,
        selected_scenes=selected,
        languages=languages,
        style=style,
        cfg=cfg,
        story_plan=story_plan,
        story_memory=story_memory,
        narration_plan=narration_plan,
    )

    scripts_dir = os.path.join(project_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    qa_by_language = {}
    postwrite_by_language = {}
    repair_by_language = {}
    for lang, segments in list(narration.items()):
        verification = None
        repair_log = []
        if semantic_active:
            verification = verify_narration(
                narration=segments,
                narration_plan=narration_plan,
                signatures=signatures,
                match_report=match_report,
                cfg=cfg,
                language=lang,
                project_dir=project_dir,
                engine=engine,
            )
            repaired, repair_log = repair_unsupported_beats(
                narration=segments,
                verification=verification,
                narration_plan=narration_plan,
                signatures=signatures,
                cfg=cfg,
                language=lang,
            )
            if repair_log:
                segments = repaired
                narration[lang] = segments
                verification = verify_narration(
                    narration=segments,
                    narration_plan=narration_plan,
                    signatures=signatures,
                    match_report=match_report,
                    cfg=cfg,
                    language=lang,
                    project_dir=project_dir,
                    engine=engine,
                )
            postwrite_by_language[lang] = verification
            repair_by_language[lang] = repair_log

        txt_path = os.path.join(scripts_dir, f"summary_script_{lang}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(f"[{seg['scene_id']}] {seg['text']}\n\n")

        json_path = os.path.join(scripts_dir, f"summary_script_{lang}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mode": "summary",
                    "language": lang,
                    "total_duration": narration_plan.get("total_duration", 0),
                    "target_wpm": narration_plan.get("target_wpm", 160),
                    "segments": segments,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        qa_by_language[lang] = evaluate_narration(
            segments,
            narration_plan,
            language=lang,
            semantic_report=verification,
        )
        print(f"   📄 {txt_path}")
        print(f"   📄 {json_path}")

    if semantic_enabled:
        match_report["postwrite"] = postwrite_by_language
        match_report["repairs"] = repair_by_language
        match_report.setdefault("metrics", {})["repair_rate"] = round(
            sum(len(v) for v in repair_by_language.values()) / max(1, len(beats)), 3
        )
        with open(match_path, "w", encoding="utf-8") as f:
            json.dump(match_report, f, ensure_ascii=False, indent=2)

    qa_path = os.path.join(project_dir, "narration_qa.json")
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_by_language, f, ensure_ascii=False, indent=2)
    print(f"   🧪 Narration QA: {qa_path}")

    # 4) TTS.  The JSON script makes voice_manager use its timeline scheduler:
    # speech is placed on one video-length canvas; short sentences are NOT padded
    # independently to full scene length.
    engine_name = args.engine if hasattr(args, "engine") and args.engine else \
        getattr(cfg, "tts_engine", "edge")
    audio_dir = os.path.join(project_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    audio_results = voice_manager.generate_all_audio(
        scripts_dir=scripts_dir,
        audio_dir=audio_dir,
        engine_name=engine_name,
        cfg=cfg,
        script_prefix="summary_script_",
        audio_prefix="summary_audio_",
        timeline_duration=_video_timeline_duration(summary_video, required=_strict_output(cfg)),
        timeline_kind="summary",
        timeline_source_path=summary_video,
    )
    if not audio_results and _strict_output(cfg):
        raise RuntimeError("Summary narration did not produce any complete audio.")

    for lang, audio_info in audio_results.items():
        backup_path = os.path.join(project_dir, f"backup_tts_summary_{lang}.wav")
        shutil.copy2(audio_info["concat_path"], backup_path)
        print(
            f"   💾 Summary TTS Backup: {backup_path} "
            f"(coverage={audio_info.get('speech_coverage', 0) * 100:.1f}%)"
        )

    # 5) Render. Timeline TTS may flow across scene cuts, so per-scene TTS length
    # must no longer trigger video recutting. Legacy audio keeps the old safety.
    ratios = getattr(cfg, "render_ratios", ["16:9"])
    audio_strategy = getattr(cfg, "render_audio_strategy", "dynamic_duck")
    burn_subs = getattr(cfg, "render_burn_subtitles", False)
    original_volume = float(getattr(cfg, "render_original_volume", 0.15))

    for lang, audio_info in audio_results.items():
        audio_path = audio_info["concat_path"]
        lang_selected = copy.deepcopy(selected)
        lang_summary_video = summary_video

        if not audio_info.get("timeline_scheduled"):
            # Backward-compatible emergency path for plain-text/legacy scripts.
            needs_rebuild = False
            for scene in lang_selected:
                sid = scene["scene_id"]
                tts_dur = float(audio_info.get("durations", {}).get(sid, 0) or 0)
                if tts_dur > float(scene.get("duration", 0) or 0) + 0.05:
                    scene["duration"] = tts_dur + 0.35
                    needs_rebuild = True
            if needs_rebuild:
                lang_summary_video = os.path.join(project_dir, f"summary_source_{lang}.mp4")
                video_cutter.cut_and_concat(
                    source_video=source_video,
                    selected_scenes=lang_selected,
                    output_path=lang_summary_video,
                )
        elif float(audio_info.get("overflow_seconds", 0) or 0) > 0.05:
            print(
                f"   ⚠️ {lang.upper()} voiceover overflows video by "
                f"{audio_info['overflow_seconds']:.2f}s; inspect narration_qa.json / TTS pacing."
            )

        # Use actual speech placements for subtitle timing in V2.
        segments = audio_info.get("final_segments") or narration.get(lang, [])
        placement_map = {
            item["scene_id"]: item
            for item in audio_info.get("placements", [])
            if item.get("scene_id")
        }
        if placement_map:
            for seg in segments:
                placement = placement_map.get(seg.get("scene_id"))
                if placement:
                    seg["start_time"] = placement["actual_start"]
                    seg["end_time"] = placement["actual_end"]
        else:
            cumulative = 0.0
            timeline = {}
            for scene in lang_selected:
                timeline[scene["scene_id"]] = (cumulative, cumulative + scene["duration"])
                cumulative += scene["duration"]
            for seg in segments:
                if seg.get("scene_id") in timeline:
                    seg["start_time"], seg["end_time"] = timeline[seg["scene_id"]]

        srt_path = os.path.join(scripts_dir, f"summary_script_{lang}.srt")
        if audio_info.get("word_timings"):
            subtitle_gen.generate_srt_from_words(audio_info["word_timings"], srt_path)
        elif segments:
            subtitle_gen.generate_srt(segments, srt_path)
        else:
            srt_path = None

        # Preserve the complete script for retry when legacy TTS only produced
        # some beats; the SRT above still describes the audio actually present.
        txt_path = os.path.join(scripts_dir, f"summary_script_{lang}.txt")
        if audio_info.get("complete_coverage", True):
            with open(txt_path, "w", encoding="utf-8") as f:
                for seg in segments:
                    f.write(
                        f"[{seg['scene_id']}] "
                        f"[{subtitle_gen._format_srt_time(seg.get('start_time', 0))} --> "
                        f"{subtitle_gen._format_srt_time(seg.get('end_time', 0))}] "
                        f"{seg['text']}\n\n"
                    )

        for ratio in ratios:
            output_path = os.path.join(
                project_dir, video_render.final_video_filename(lang, ratio, mode="summary", cfg=cfg),
            )
            print(f"   🎬 Rendering Summary: {lang.upper()} ({ratio})...")
            try:
                video_render.render_final(
                    source_video=lang_summary_video,
                    narration_audio=audio_path,
                    output_path=output_path,
                    subtitle_path=srt_path if (srt_path and os.path.exists(srt_path)) else None,
                    audio_strategy=audio_strategy,
                    original_volume=original_volume,
                    burn_subs=burn_subs,
                    target_ratio=ratio if ratio != "16:9" else None,
                    cfg=cfg,
                )
                print(f"   ✅ {output_path}")
            except Exception as exc:
                print(f"   ❌ Summary render failed: {exc}")
                if _strict_output(cfg) or (
                    ratio == "9:16" and getattr(cfg, "render_vertical_layout", "pad") == "layered"
                ):
                    raise

    print("\n✅ Summary pipeline selesai!")


# ==============================================================================
# MAIN
# ==============================================================================

def _add_layout_arguments(parser) -> None:
    parser.add_argument(
        "--vertical-layout", choices=["pad", "layered"], default=None,
        help="9:16 layout: pad (existing black bars) or layered (sharp foreground over blurred background).",
    )
    parser.add_argument(
        "--foreground-zoom", type=float, default=None,
        help="Foreground zoom from 0 (fit entire video) to 100 (fill canvas); default 25.",
    )
    parser.add_argument(
        "--background-zoom", type=float, default=None,
        help="Background zoom from 0 (fit entire video) to 100 (fill canvas); default 100.",
    )
    parser.add_argument(
        "--background-blur", type=float, default=None,
        help="Background Gaussian blur strength, 1 to 100; default 50.",
    )
    parser.add_argument(
        "--background-grayscale", type=float, default=None,
        help="Background grayscale amount, 0 (original color) to 1 (fully gray); default 1.",
    )


def _configure_layout(args, cfg) -> None:
    from dubbingstory.render.vertical_layout import layout_settings

    for name in ("vertical_layout", "foreground_zoom", "background_zoom",
                 "background_blur", "background_grayscale"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, "render_" + name, value)
    layout_settings(cfg)
    if (getattr(cfg, "render_vertical_layout", "pad") == "layered"
            and "9:16" not in getattr(cfg, "render_ratios", ["16:9"])):
        raise ValueError("The layered layout requires --ratio 9:16 (other output ratios may be included).")


def _add_caption_arguments(parser, *, synthesis: bool = False) -> None:
    parser.add_argument(
        "--subtitle-style", choices=["classic", "word", "karaoke"], default=None,
        help="Caption style; animated styles require validated word timing (SRT is retained).",
    )
    parser.add_argument(
        "--word-alignment", choices=["auto", "provider", "edge", "whisperx"], default=None,
        help="Word timing backend; auto prefers provider metadata, otherwise optional WhisperX.",
    )
    if synthesis:
        parser.add_argument(
            "--narration-fit", choices=["legacy", "rewrite", "error"], default=None,
            help="Handle speech overflow; native/animated workflows default to bounded rewrite.",
        )
    else:
        parser.add_argument(
            "--burn-subtitles", action=argparse.BooleanOptionalAction, default=None,
            help="Burn subtitles into MP4, or export subtitle files only.",
        )


def _add_temporal_arguments(parser) -> None:
    parser.add_argument("--temporal-analysis", action=argparse.BooleanOptionalAction, default=None,
                        help="Analyze dense timestamped frames for full-video storytelling.")
    parser.add_argument("--frame-interval", type=float, default=None,
                        help="Seconds between analyzed frames (0.5 = two per second); enables temporal analysis.")


def _add_action_sync_arguments(parser) -> None:
    parser.add_argument("--action-sync", action=argparse.BooleanOptionalAction, default=None,
                        help="Align selected spoken phrases with observed actions (full mode).")
    parser.add_argument("--action-sync-policy", choices=["best_effort", "error"], default=None,
                        help="Report remaining action timing misses or stop before publishing audio.")


def _configure_enhancements(args, cfg) -> None:
    """Apply additive options without replacing configured values with parser defaults."""
    for arg_name, config_name in (
        ("subtitle_style", "subtitle_style"),
        ("word_alignment", "subtitle_alignment"),
        ("narration_fit", "tts_narration_fit"),
        ("burn_subtitles", "render_burn_subtitles"),
        ("narration_source", "narration_source"),
        ("temporal_analysis", "vision_temporal_enabled"),
        ("frame_interval", "vision_temporal_interval_seconds"),
        ("action_sync", "narration_action_sync"),
        ("action_sync_policy", "narration_action_sync_policy"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(cfg, config_name, value)
    if getattr(args, "frame_interval", None) is not None:
        if getattr(args, "temporal_analysis", None) is False:
            raise ValueError("--frame-interval cannot be combined with --no-temporal-analysis")
        cfg.vision_temporal_enabled = True
    if getattr(args, "mode", "full") == "summary":
        if getattr(args, "action_sync", None) or getattr(args, "temporal_analysis", None) or getattr(args, "frame_interval", None) is not None:
            raise ValueError("Dense temporal analysis and action synchronization currently require --mode full")
        # A shared YAML can still render/run the existing summary workflow.
        cfg.narration_action_sync = False
        cfg.vision_temporal_enabled = False
    interval = float(getattr(cfg, "vision_temporal_interval_seconds", 0.5))
    if not math.isfinite(interval) or not 0.1 <= interval <= 10:
        raise ValueError("vision.temporal.interval_seconds must be finite and between 0.1 and 10")
    tolerance = float(getattr(cfg, "narration_action_sync_tolerance", 0.5))
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 2:
        raise ValueError("narration.action_sync_tolerance must be between 0 and 2 seconds")
    if getattr(cfg, "narration_action_sync_policy", "best_effort") not in {"best_effort", "error"}:
        raise ValueError("narration.action_sync_policy must be best_effort or error")
    style = getattr(cfg, "subtitle_style", "classic")
    if style not in {"classic", "word", "karaoke"}:
        raise ValueError("subtitle.style must be classic, word, or karaoke")
    if getattr(cfg, "subtitle_alignment", "auto") not in {"auto", "provider", "edge", "whisperx"}:
        raise ValueError("subtitle.alignment must be auto, provider, edge, or whisperx")
    if getattr(cfg, "narration_source", "visual") not in {"visual", "audio"}:
        raise ValueError("narration.source must be visual or audio")
    ratios = getattr(cfg, "render_ratios", ["16:9"])
    if not isinstance(ratios, (list, tuple)) or not ratios or any(
        ratio not in {"source", "16:9", "9:16", "1:1", "4:3"} for ratio in ratios
    ):
        raise ValueError("render.ratios must contain source, 16:9, 9:16, 1:1, or 4:3")
    fit = getattr(cfg, "tts_narration_fit", "legacy")
    if fit not in {"legacy", "rewrite", "error"}:
        raise ValueError("tts.narration_fit must be legacy, rewrite, or error")
    if ("source" in ratios or style != "classic" or getattr(cfg, "narration_action_sync", False)) and fit == "legacy":
        if getattr(args, "narration_fit", None) == "legacy":
            raise ValueError("Native/animated output requires --narration-fit rewrite or error.")
        cfg.tts_narration_fit = "rewrite"
    if _strict_output(cfg) and not getattr(cfg, "tts_use_timeline_scheduler", True):
        raise ValueError("Native/animated output requires tts.use_timeline_scheduler: true")
    if not 0 <= float(getattr(cfg, "render_original_volume", 0.15)) <= 2:
        raise ValueError("render.original_volume must be between 0 and 2")


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
    p_run.add_argument("--url", "-u", type=str, help="YouTube, TikTok, Douyin, or Xiaohongshu/RedNote video/share URL")
    p_run.add_argument("--i-have-rights", action="store_true", help="Confirm video rights")
    p_run.add_argument("--project", "-p", type=str, help="Project name")
    p_run.add_argument("--style", type=str, default=None,
                        choices=["viral_fb", "jozzr_social", "documentary", "technical", "calm_educational"])
    p_run.add_argument("--lang", nargs="+", default=None, help="Languages (e.g., id en)")
    p_run.add_argument("--engine", type=str, default=None, choices=["edge", "azure", "piper", "chirp", "elevenlabs"], help="TTS engine (default: edge)")
    p_run.add_argument("--voice-id", type=str, default=None, help="Specific voice name/ID for Indonesian (e.g., id-ID-ArdiNeural or id_ID-news_tts-medium)")
    p_run.add_argument("--voice-en", type=str, default=None, help="Specific voice name/ID for English")
    p_run.add_argument("--speech-coverage", type=float, default=None, help="Target ratio of narration time vs video time (e.g. 0.85)")
    p_run.add_argument("--ratio", nargs="+", default=None,
                       choices=["source", "16:9", "9:16", "1:1", "4:3"],
                       help="Output ratios; source preserves original composition and duration")
    _add_layout_arguments(p_run)
    _add_caption_arguments(p_run, synthesis=True)
    _add_temporal_arguments(p_run)
    _add_action_sync_arguments(p_run)
    p_run.add_argument("--burn-subtitles", action=argparse.BooleanOptionalAction, default=None)
    p_run.add_argument("--disable-adaptive-keyframes", action="store_true", help="Disable adaptive keyframe count capping")
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
        "--semantic-match", type=str, default=None, choices=["active", "shadow", "off"],
        help="V2.1 summary matcher mode: active, shadow, or off (default from config)"
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
        "--vision-concurrency", type=int, default=None,
        help="Parallel per-scene vision requests. Match local data-parallel replicas when possible."
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
        "--narration-source", type=str, default=None, choices=["visual", "audio"],
        help="Narration context: visual-only (default) or original audio/transcript.",
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
        "--max-scene-duration", type=float, default=None,
        help="Split long continuous shots into bounded analysis windows (recommended: 12-18s)."
    )
    p_run.add_argument(
        "--scene-threshold", type=float, default=None,
        help="Scene detection sensitivity (default: 3.0). "
             "Higher = fewer scenes detected = faster. Recommended: 4-6 for speed."
    )

    # ── ingest ────────────────────────────────────────────────────────────
    p_ingest = subparsers.add_parser("ingest", help="Download/validate video")
    p_ingest.add_argument("--input", "-i", type=str, help="Path to local video file")
    p_ingest.add_argument("--url", "-u", type=str, help="YouTube, TikTok, Douyin, or Xiaohongshu/RedNote video/share URL")
    p_ingest.add_argument("--i-have-rights", action="store_true")
    p_ingest.add_argument("--project", "-p", type=str)
    p_ingest.add_argument(
        "--narration-source", type=str, default=None, choices=["visual", "audio"],
        help="Narration context: visual-only (default) or original audio/transcript.",
    )
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
    p_ingest.add_argument("--disable-adaptive-keyframes", action="store_true", help="Disable adaptive keyframe count capping")

    # ── segment ───────────────────────────────────────────────────────────
    p_segment = subparsers.add_parser("segment", help="Detect scenes + keyframes")
    p_segment.add_argument("--project", "-p", type=str, required=True)
    p_segment.add_argument(
        "--max-keyframes", type=int, default=None,
        help="Max keyframes per scene (default: 3). Lower = faster."
    )
    p_segment.add_argument("--disable-adaptive-keyframes", action="store_true", help="Disable adaptive keyframe count capping")
    p_segment.add_argument(
        "--min-scene-duration", type=float, default=None,
        help="Minimum scene duration in seconds (default: 2.0). Higher = fewer scenes."
    )
    p_segment.add_argument(
        "--max-scene-duration", type=float, default=None,
        help="Split long detected shots into bounded analysis/selectable windows."
    )
    p_segment.add_argument(
        "--scene-threshold", type=float, default=None,
        help="Scene detection sensitivity (default: 3.0). Higher = fewer scenes."
    )

    # ── analyze ───────────────────────────────────────────────────────────
    p_analyze = subparsers.add_parser("analyze", help="Visual understanding")
    p_analyze.add_argument("--project", "-p", type=str, required=True)
    _add_temporal_arguments(p_analyze)
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
    p_analyze.add_argument(
        "--vision-concurrency", type=int, default=None,
        help="Parallel per-scene vision requests."
    )

    # ── narrate ───────────────────────────────────────────────────────────
    p_narrate = subparsers.add_parser("narrate", help="Generate narration scripts")
    p_narrate.add_argument("--project", "-p", type=str, required=True)
    _add_action_sync_arguments(p_narrate)
    p_narrate.add_argument("--style", type=str, default=None,
                            choices=["viral_fb", "jozzr_social", "documentary", "technical", "calm_educational"])
    p_narrate.add_argument("--lang", nargs="+", default=None)
    p_narrate.add_argument("--speech-coverage", type=float, default=None, help="Target ratio of narration time vs video time (e.g. 0.85)")

    # ── dub ───────────────────────────────────────────────────────────────
    p_dub = subparsers.add_parser("dub", help="Generate TTS audio")
    p_dub.add_argument("--project", "-p", type=str, required=True)
    p_dub.add_argument("--engine", type=str, default=None, choices=["edge", "azure", "piper", "chirp", "elevenlabs"], help="TTS engine (default: edge)")
    _add_action_sync_arguments(p_dub)
    p_dub.add_argument("--voice-id", type=str, default=None, help="Specific voice name/ID for Indonesian")
    p_dub.add_argument("--voice-en", type=str, default=None, help="Specific voice name/ID for English")
    _add_caption_arguments(p_dub, synthesis=True)

    # ── render ────────────────────────────────────────────────────────────
    p_render = subparsers.add_parser("render", help="Render final video")
    p_render.add_argument("--project", "-p", type=str, required=True)
    p_render.add_argument("--mode", choices=["full", "summary"], default="full",
                          help="Render saved full or summary artifacts without rerunning analysis or TTS")
    p_render.add_argument("--ratio", nargs="+", default=None,
                          choices=["source", "16:9", "9:16", "1:1", "4:3"])
    _add_layout_arguments(p_render)
    _add_caption_arguments(p_render)

    # ── reframe ───────────────────────────────────────────────────────────
    p_reframe = subparsers.add_parser("reframe", help="Recompose an existing video onto a 9:16 canvas")
    p_reframe.add_argument("--input", "-i", type=str, required=True, help="Existing local video, including a final summary export")
    p_reframe.add_argument("--output", "-o", type=str, required=True, help="New MP4 output path (must not already exist)")
    p_reframe.add_argument("--ratio", choices=["9:16"], default="9:16")
    _add_layout_arguments(p_reframe)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Build config
    cfg = build_config(config_path=args.config)

    # Apply CLI overrides
    if hasattr(args, "ratio") and args.ratio:
        cfg.render_ratios = [args.ratio] if isinstance(args.ratio, str) else args.ratio
    if hasattr(args, "style") and args.style:
        cfg.narration_style = args.style
    if getattr(args, "lang", None):
        cfg.narration_languages = args.lang
        cfg.requested_languages = args.lang
    if hasattr(args, "speech_coverage") and args.speech_coverage is not None:
        cfg.narration_speech_coverage = args.speech_coverage
    if hasattr(args, "narration_source") and args.narration_source:
        cfg.narration_source = args.narration_source
    if hasattr(args, "disable_adaptive_keyframes") and args.disable_adaptive_keyframes:
        cfg.keyframes_adaptive = False
    if hasattr(args, "engine") and args.engine:
        cfg.tts_engine = args.engine
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
    if hasattr(args, "vision_concurrency") and args.vision_concurrency is not None:
        cfg.vision_concurrency = max(1, int(args.vision_concurrency))

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
    if hasattr(args, "max_scene_duration") and args.max_scene_duration is not None:
        cfg.segment_max_scene_duration = args.max_scene_duration
    if hasattr(args, "scene_threshold") and args.scene_threshold is not None:
        cfg.segment_adaptive_threshold = args.scene_threshold

    if args.command in {"run", "render", "reframe"}:
        _configure_layout(args, cfg)
    if args.command != "reframe":
        _configure_enhancements(args, cfg)

    # Dispatch
    commands = {
        "run": cmd_run,
        "ingest": cmd_ingest,
        "segment": cmd_segment,
        "analyze": cmd_analyze,
        "narrate": cmd_narrate,
        "dub": cmd_dub,
        "render": cmd_render,
        "reframe": cmd_reframe,
        "summary": cmd_summary,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args, cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
