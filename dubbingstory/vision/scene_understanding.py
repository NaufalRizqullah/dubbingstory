"""
dubbingstory.vision.scene_understanding — Scene understanding orchestrator

Coordinates per-scene visual analysis and builds the final storyboard.
"""

import json
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor

from dubbingstory.vision.gemini_analyzer import GeminiVideoAnalyzer
from dubbingstory.vision.openai_vision import OpenAIVisionAnalyzer
from dubbingstory.vision.prompts import build_scene_prompt, build_temporal_prompt, build_cheap_scene_prompt


def _get_subtitle_for_scene(
    subtitle_entries: list[dict] | None,
    start_time: float,
    end_time: float,
) -> tuple[str, list[dict]]:
    """Extract subtitle text and word-level data that falls within a scene's time range.

    Returns
    -------
    tuple[str, list[dict]]
        (subtitle_text, words_in_scene) — the joined text lines and any
        word-level timing data from ASR.
    """
    if not subtitle_entries:
        return "", []

    lines = []
    words_in_scene = []
    for entry in subtitle_entries:
        # Check if subtitle overlaps with scene time range
        sub_start = entry.get("start_seconds", 0)
        sub_end = entry.get("end_seconds", 0)

        if sub_start < end_time and sub_end > start_time:
            lines.append(entry["text"])

            # Collect word-level data if available (from ASR)
            for w in entry.get("words", []):
                if w["start"] < end_time and w["end"] > start_time:
                    words_in_scene.append(w)

    text = "\n".join(lines) if lines else ""
    return text, words_in_scene


def _create_analyzer(cfg):
    """Create the appropriate vision analyzer based on config.

    Supports:
    - "gemini": Google Gemini API (default)
    - "openai": OpenAI-compatible API (for Qwen3-VL via vLLM, HuggingFace, etc.)

    For model discovery, browse:
        https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending
    """
    provider = getattr(cfg, "vision_provider", "gemini")

    if provider == "openai":
        api_key = (
            getattr(cfg, "api_key_openai_vision", "")
            or getattr(cfg, "api_key_hf", "")
            or "local"
        )
        base_url = getattr(cfg, "vision_openai_base_url", "http://127.0.0.1:8000/v1")
        model = getattr(cfg, "vision_openai_model", "Qwen/Qwen3-VL-2B-Instruct")
        temperature = getattr(cfg, "vision_openai_temperature", 0.2)
        max_tokens = getattr(cfg, "vision_openai_max_tokens", 2048)

        return OpenAIVisionAnalyzer(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif provider == "gemini":
        api_key = getattr(cfg, "api_key_gemini", "")
        if not api_key:
            raise ValueError(
                "❌ GOOGLE_API_KEY not found.\n"
                "   Set via .env file or environment variable."
            )
        model = getattr(cfg, "vision_gemini_model", "gemini-2.5-flash")
        fallback = getattr(cfg, "vision_gemini_fallback_model", "gemini-2.0-flash")

        return GeminiVideoAnalyzer(
            api_key=api_key,
            model=model,
            fallback_model=fallback,
        )

    else:
        raise ValueError(
            f"❌ Unknown vision provider: '{provider}'\n"
            f"   Supported: 'gemini', 'openai'"
        )


def run_analysis(
    segment_data: dict,
    project_dir: str,
    cfg,
    subtitle_context: dict | None = None,
    domain_hint: str = "",
) -> dict:
    """
    Run visual understanding on all scenes.

    Pipeline:
    1. Analyze each scene independently (keyframes or video upload)
    2. Build temporal flow understanding across all scenes
    3. Generate storyboard with narration cues

    Parameters
    ----------
    segment_data : dict
        Segment manifest from scene detection step.
    project_dir : str
        Project output directory.
    cfg : SimpleNamespace
        Config object.
    subtitle_context : dict | None
        Parsed subtitle data from ingest step.
    domain_hint : str
        Domain hint (e.g., "workshop", "repair").

    Returns
    -------
    dict
        Complete storyboard with scene analyses and narrative context.
    """
    provider = getattr(cfg, "vision_provider", "gemini")
    analysis_mode = getattr(cfg, "vision_analysis_mode", "keyframes")

    analyzer = _create_analyzer(cfg)

    scenes = segment_data.get("scenes", [])
    keyframes_data = segment_data.get("keyframes", {})
    video_path = segment_data.get("video_path", "")

    # Get subtitle entries for context
    subtitle_entries = None
    full_subtitle_context = ""
    if subtitle_context:
        subtitle_entries = subtitle_context.get("entries")
        full_subtitle_context = subtitle_context.get("context_string", "")

    # Get video metadata for context hints
    video_meta = {}
    meta_path = os.path.join(project_dir, "video_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            video_meta = json.load(f)

    # Build context hint from video title/description
    context_hint = domain_hint
    if video_meta.get("title"):
        context_hint += f"\nVideo title: {video_meta['title']}"
    if video_meta.get("description"):
        desc = video_meta["description"][:500]
        context_hint += f"\nVideo description: {desc}"

    print(f"\n   👁️ Analyzing {len(scenes)} scenes (provider: {provider}, mode: {analysis_mode})...")
    if domain_hint:
        print(f"   🏷️ Domain hint: {domain_hint}")
    if subtitle_entries:
        print(f"   📝 Using subtitle context ({len(subtitle_entries)} lines)")

    # ── Step 1: Per-scene analysis ──────────────────────────────────────
    scene_analyses = []
    cache_dir = os.path.join(project_dir, "vision_cache")
    os.makedirs(cache_dir, exist_ok=True)

    def analyze_scene(scene, is_cheap=False):
        scene_id = scene["scene_id"]
        time_range = f"{scene['start_time']:.1f}s - {scene['end_time']:.1f}s"
        
        kf_paths = keyframes_data.get(scene_id, [])
        if is_cheap:
            if len(kf_paths) > 2:
                kf_paths = [kf_paths[0], kf_paths[-1]]
            prompt = build_cheap_scene_prompt(
                scene_id=scene_id,
                time_range=time_range,
                n_frames=len(kf_paths),
                domain=domain_hint or "general",
            )
            scene_words = []
        else:
            scene_subtitle, scene_words = _get_subtitle_for_scene(
                subtitle_entries,
                scene["start_time"],
                scene["end_time"],
            )
            prompt = build_scene_prompt(
                scene_id=scene_id,
                time_range=time_range,
                n_frames=len(kf_paths),
                domain=domain_hint or "general",
                context_hint=context_hint,
                subtitle_text=scene_subtitle,
            )

        # Version-aware cache key
        payload = {
            "scene_id": scene_id,
            "model": getattr(analyzer, "model", "gemini"),
            "prompt": prompt,
            "keyframes": [
                {
                    "path": os.path.basename(p),
                    "size": os.path.getsize(p) if os.path.exists(p) else 0,
                    "mtime": os.path.getmtime(p) if os.path.exists(p) else 0,
                }
                for p in kf_paths
            ]
        }
        cache_key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        cache_path = os.path.join(cache_dir, f"{scene_id}_{cache_key}.json")

        # Check cache
        if os.path.exists(cache_path):
            print(f"   ⏩ {scene_id}: cached, skip.")
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)



        try:
            if analysis_mode == "video_upload" and scene.get("file_path"):
                analysis = analyzer.analyze_scene_from_video(
                    video_path=scene["file_path"],
                    prompt=prompt,
                )
            else:
                analysis = analyzer.analyze_scene_from_frames(
                    keyframe_paths=kf_paths,
                    prompt=prompt,
                )

            analysis["start_time"] = scene["start_time"]
            analysis["end_time"] = scene["end_time"]
            analysis["duration"] = scene["duration"]
            if not is_cheap and scene_words:
                analysis["words"] = scene_words

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)

            conf = analysis.get("confidence", 0)
            print(f"   ✅ {scene_id}: analyzed (confidence: {conf:.2f})")
            return analysis

        except Exception as e:
            print(f"   ❌ {scene_id}: analysis failed — {e}")
            if is_cheap:
                return {"scene_id": scene_id, "salience": 0.0}
            return {
                "scene_id": scene_id,
                "time_range": time_range,
                "start_time": scene["start_time"],
                "end_time": scene["end_time"],
                "duration": scene["duration"],
                "visible_objects": [],
                "people": "Unable to analyze",
                "action": "Unable to analyze",
                "changes": "Unable to analyze",
                "environment": "Unable to analyze",
                "likely_context": "Unable to analyze",
                "text_visible": [],
                "confidence": 0.0,
                "error": str(e),
            }

    workers = getattr(cfg, "vision_concurrency", 2)
    mode = getattr(cfg, "mode", "full")

    if mode == "summary":
        print(f"   🔍 Pass A: Screening {len(scenes)} scenes...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            cheap_analyses = list(pool.map(lambda s: analyze_scene(s, is_cheap=True), scenes))
            
        summary_max_scenes = getattr(cfg, "summary_max_scenes", 20)
        if not summary_max_scenes:
            summary_max_scenes = 20

        for scene, analysis in zip(scenes, cheap_analyses):
            scene["_salience"] = analysis.get("salience", 0.0)
            
        selected_scenes = sorted(scenes, key=lambda x: x.get("_salience", 0.0), reverse=True)[:summary_max_scenes]
        selected_scenes = sorted(selected_scenes, key=lambda x: x.get("scene_id", ""))
        selected_ids = {s["scene_id"] for s in selected_scenes}

        print(f"   🎯 Pass B: Deep analysis on {len(selected_scenes)} selected scenes...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            deep_analyses = list(pool.map(lambda s: analyze_scene(s, is_cheap=False), selected_scenes))
            
        deep_map = {a["scene_id"]: a for a in deep_analyses}
        for scene in scenes:
            sid = scene["scene_id"]
            if sid in selected_ids:
                analysis = deep_map.get(sid, {})
                analysis["selected"] = True
                scene_analyses.append(analysis)
            else:
                scene_analyses.append({
                    "scene_id": sid,
                    "start_time": scene["start_time"],
                    "end_time": scene["end_time"],
                    "duration": scene["duration"],
                    "action": "Skipped in summary mode",
                    "confidence": 0.0,
                    "selected": False,
                })
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            scene_analyses = list(pool.map(lambda s: analyze_scene(s, is_cheap=False), scenes))

    # ── Step 2: Temporal flow analysis ──────────────────────────────────
    print(f"\n   🔄 Building temporal flow understanding...")

    scenes_for_temporal = [a for a in scene_analyses if a.get("selected", True)]

    temporal_prompt = build_temporal_prompt(
        scene_analyses=scenes_for_temporal,
        domain=domain_hint or "general",
        subtitle_context=full_subtitle_context,
    )

    try:
        temporal_data = analyzer.analyze_temporal_flow(
            scene_analyses=scene_analyses,
            prompt=temporal_prompt,
        )
    except Exception as e:
        print(f"   ⚠️ Temporal analysis failed: {e}")
        temporal_data = {
            "video_summary": "Unable to generate summary.",
            "narrative_arc": "unknown",
            "domain": domain_hint or "unknown",
            "scenes_enriched": [],
        }

    # ── Step 3: Build storyboard ────────────────────────────────────────
    transcript_source = "none"
    if subtitle_context:
        transcript_source = subtitle_context.get("source", "manual") or "manual"

    storyboard = _build_storyboard(
        scene_analyses=scene_analyses,
        temporal_data=temporal_data,
        video_meta=video_meta,
        transcript_source=transcript_source,
    )

    print(f"\n   📋 Storyboard built:")
    print(f"      Summary: {storyboard.get('video_summary', '')[:80]}...")
    print(f"      Arc: {storyboard.get('narrative_arc', 'unknown')}")
    print(f"      Scenes: {len(storyboard.get('scenes', []))}")

    return storyboard


def _build_storyboard(
    scene_analyses: list[dict],
    temporal_data: dict,
    video_meta: dict,
    transcript_source: str = "none",
) -> dict:
    """
    Build the final storyboard JSON from analyses.

    This is the single source of truth that can be edited manually
    before generating narration.
    """
    enriched_scenes = temporal_data.get("scenes_enriched", [])

    # Merge per-scene analysis with temporal enrichment
    scenes = []
    for analysis in scene_analyses:
        scene_id = analysis.get("scene_id", "")

        # Find matching temporal enrichment
        enrichment = {}
        for e in enriched_scenes:
            if e.get("scene_id") == scene_id:
                enrichment = e
                break

        scene_entry = {
            "scene_id": scene_id,
            "start_time": analysis.get("start_time", 0),
            "end_time": analysis.get("end_time", 0),
            "duration": analysis.get("duration", 0),
            "analysis": {
                "visible_objects": analysis.get("visible_objects", []),
                "people": analysis.get("people", ""),
                "action": analysis.get("action", ""),
                "changes": analysis.get("changes", ""),
                "environment": analysis.get("environment", ""),
                "likely_context": analysis.get("likely_context", ""),
                "text_visible": analysis.get("text_visible", []),
                "confidence": analysis.get("confidence", 0),
                # Add word-level data if available (from ASR)
                "words": analysis.get("words", []),
            },
            "narrative": {
                "role": enrichment.get("narrative_role", "process"),
                "importance": enrichment.get("narration_importance", 0.5),
                "cue": enrichment.get("narration_cue", ""),
                "connects_to_next": enrichment.get("connects_to_next", ""),
            },
        }

        scenes.append(scene_entry)

    total_duration = sum(s["duration"] for s in scenes)

    return {
        "video_title": video_meta.get("title", ""),
        "video_description": video_meta.get("description", "")[:500],
        "video_summary": temporal_data.get("video_summary", ""),
        "narrative_arc": temporal_data.get("narrative_arc", "unknown"),
        "domain": temporal_data.get("domain", "general"),
        "video_transcript_source": transcript_source,
        "total_scenes": len(scenes),
        "total_duration": round(total_duration, 2),
        "scenes": scenes,
    }
