"""
dubbingstory.vision.scene_understanding — Scene understanding orchestrator

Coordinates per-scene visual analysis and builds the final storyboard.
"""

import json
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor

from dubbingstory.vision.gemini_analyzer import GeminiVideoAnalyzer
from dubbingstory.vision.openai_vision import OpenAIVisionAnalyzer, _estimate_tokens
from dubbingstory.vision.prompts import build_scene_prompt, build_temporal_prompt, build_cheap_scene_prompt
from dubbingstory.vision.schemas import VISION_SCHEMA_VERSION


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
        temperature = getattr(cfg, "vision_openai_temperature", 0.7)
        max_tokens = getattr(cfg, "vision_openai_max_tokens", 640)
        model_max_context = getattr(cfg, "vision_openai_model_max_context", None)
        image_mode = getattr(cfg, "vision_openai_image_mode", "data")
        structured_outputs = getattr(cfg, "vision_openai_structured_outputs", True)
        allow_schema_fallback = getattr(cfg, "vision_openai_allow_schema_fallback", False)
        repetition_penalty = getattr(cfg, "vision_openai_repetition_penalty", 1.0)
        top_p = getattr(cfg, "vision_openai_top_p", 0.8)
        top_k = getattr(cfg, "vision_openai_top_k", 20)
        presence_penalty = getattr(cfg, "vision_openai_presence_penalty", 1.5)
        image_token_cost = getattr(cfg, "vision_openai_image_token_cost", 512)
        truncation_retry_tokens = getattr(cfg, "vision_openai_truncation_retry_tokens", 896)
        runaway_rescue_tokens = getattr(cfg, "vision_openai_runaway_rescue_tokens", 448)
        temporal_truncation_retry_tokens = getattr(
            cfg, "vision_openai_temporal_truncation_retry_tokens", 6144
        )
        runaway_threshold = getattr(cfg, "vision_openai_runaway_threshold", 0.35)

        return OpenAIVisionAnalyzer(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            model_max_context=model_max_context,
            image_mode=image_mode,
            structured_outputs=structured_outputs,
            allow_schema_fallback=allow_schema_fallback,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            image_token_cost=image_token_cost,
            truncation_retry_tokens=truncation_retry_tokens,
            runaway_rescue_tokens=runaway_rescue_tokens,
            temporal_truncation_retry_tokens=temporal_truncation_retry_tokens,
            runaway_threshold=runaway_threshold,
        )

    elif provider == "gemini":
        from dubbingstory.llm.text_client import gemini_retry_options

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
            **gemini_retry_options(cfg),
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
    
    fail_log_path = os.path.join(project_dir, "failed_vision.log")
    if os.path.exists(fail_log_path):
        os.remove(fail_log_path)

    def analyze_scene(scene, is_cheap=False):
        scene_id = scene["scene_id"]
        time_range = f"{scene['start_time']:.1f}s - {scene['end_time']:.1f}s"
        
        kf_paths = keyframes_data.get(scene_id, [])
        scene_subtitle, scene_words = _get_subtitle_for_scene(
            subtitle_entries,
            scene["start_time"],
            scene["end_time"],
        )
        if is_cheap:
            if len(kf_paths) > 2:
                kf_paths = [kf_paths[0], kf_paths[-1]]
            prompt = build_cheap_scene_prompt(
                scene_id=scene_id,
                time_range=time_range,
                n_frames=len(kf_paths),
                domain=domain_hint or "general",
                subtitle_text=scene_subtitle,
            )
            # Cheap pass keeps text context for story-aware candidate selection
            # but does not carry word-level timing into temporal analysis.
            scene_words = []
        else:
            prompt = build_scene_prompt(
                scene_id=scene_id,
                time_range=time_range,
                n_frames=len(kf_paths),
                domain=domain_hint or "general",
                context_hint=context_hint,
                subtitle_text=scene_subtitle,
            )

        # Version-aware cache key
        analyzer_fingerprint = {}
        if hasattr(analyzer, "cache_fingerprint"):
            try:
                analyzer_fingerprint = analyzer.cache_fingerprint()
            except Exception:
                analyzer_fingerprint = {}
        payload = {
            "scene_id": scene_id,
            "parent_scene_id": scene.get("parent_scene_id", scene.get("source_scene_id", scene_id)),
            "window_index": scene.get("window_index", max(0, int(scene.get("source_scene_part", 1) or 1) - 1)),
            "window_count": scene.get("window_count", max(1, int(scene.get("source_scene_parts", 1) or 1))),
            "analysis_kind": "cheap" if is_cheap else "deep",
            "vision_schema_version": VISION_SCHEMA_VERSION,
            "model": getattr(analyzer, "model", "gemini"),
            "analyzer_config": analyzer_fingerprint,
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
                cached = json.load(f)
            # Identity is deterministic pipeline metadata, so old cache entries
            # are upgraded in memory instead of forcing an expensive reanalysis.
            cached["scene_id"] = scene_id
            cached["parent_scene_id"] = scene.get("parent_scene_id", scene.get("source_scene_id", scene_id))
            cached["window_index"] = scene.get("window_index", max(0, int(scene.get("source_scene_part", 1) or 1) - 1))
            cached["window_count"] = scene.get("window_count", max(1, int(scene.get("source_scene_parts", 1) or 1)))
            cached["start_time"] = scene["start_time"]
            cached["end_time"] = scene["end_time"]
            cached["duration"] = scene["duration"]
            cached.setdefault("analysis_kind", "cheap" if is_cheap else "deep")
            return cached



        try:
            # Log per-scene prompt token estimate and keyframe sizes to help
            # identify offenders that blow up the context window.
            try:
                prompt_tokens = _estimate_tokens(prompt, model=getattr(analyzer, "model", None))
            except Exception:
                prompt_tokens = None

            total_kf_bytes = sum(os.path.getsize(p) for p in kf_paths if os.path.exists(p))
            print(
                f"   [Tokens] {scene_id}: prompt_tokens~{prompt_tokens} keyframes={len(kf_paths)} "
                f"total_kf_bytes={total_kf_bytes}"
            )

            if analysis_mode == "video_upload" and scene.get("file_path"):
                analysis = analyzer.analyze_scene_from_video(
                    video_path=scene["file_path"],
                    prompt=prompt,
                )
            else:
                if isinstance(analyzer, OpenAIVisionAnalyzer):
                    analysis = analyzer.analyze_scene_from_frames(
                        keyframe_paths=kf_paths,
                        prompt=prompt,
                        analysis_kind="cheap" if is_cheap else "deep",
                        trace_id=scene_id,
                    )
                else:
                    analysis = analyzer.analyze_scene_from_frames(
                        keyframe_paths=kf_paths,
                        prompt=prompt,
                    )

            # Scene identity/timing are deterministic pipeline facts. Never trust
            # the model to reproduce them, especially under concurrent requests.
            analysis["scene_id"] = scene_id
            analysis["parent_scene_id"] = scene.get("parent_scene_id", scene.get("source_scene_id", scene_id))
            analysis["window_index"] = scene.get("window_index", max(0, int(scene.get("source_scene_part", 1) or 1) - 1))
            analysis["window_count"] = scene.get("window_count", max(1, int(scene.get("source_scene_parts", 1) or 1)))
            analysis["time_range"] = time_range
            analysis["start_time"] = scene["start_time"]
            analysis["end_time"] = scene["end_time"]
            analysis["duration"] = scene["duration"]
            analysis["analysis_kind"] = "cheap" if is_cheap else "deep"
            if not is_cheap:
                analysis["transcript_text"] = scene_subtitle
                if scene_words:
                    analysis["words"] = scene_words

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)

            conf = analysis.get("confidence", 0)
            print(f"   ✅ {scene_id}: analyzed (confidence: {conf:.2f})")
            return analysis

        except Exception as e:
            err_msg = str(e)
            print(f"   ❌ {scene_id}: analysis failed — {err_msg}")
            
            # Save failure to log file
            fail_log_path = os.path.join(project_dir, "failed_vision.log")
            with open(fail_log_path, "a", encoding="utf-8") as f:
                f.write(f"[{scene_id}] Error: {err_msg}\n")
                
            if is_cheap:
                return {"scene_id": scene_id, "salience": 0.0}
            return {
                "scene_id": scene_id,
                "parent_scene_id": scene.get("parent_scene_id", scene.get("source_scene_id", scene_id)),
                "window_index": scene.get("window_index", max(0, int(scene.get("source_scene_part", 1) or 1) - 1)),
                "window_count": scene.get("window_count", max(1, int(scene.get("source_scene_parts", 1) or 1))),
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

    workers = max(1, int(getattr(cfg, "vision_concurrency", 2) or 1))
    mode = getattr(cfg, "mode", "full")
    print(f"   ⚙️ Vision concurrency: {workers} worker(s)")

    if mode == "summary":
        print(f"   🔍 Pass A: Screening {len(scenes)} scenes with visual + transcript relevance...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            cheap_analyses = list(pool.map(lambda s: analyze_scene(s, is_cheap=True), scenes))

        final_max_scenes = getattr(cfg, "summary_max_scenes", None)
        base_candidates = int(getattr(cfg, "summary_analysis_candidates", 32) or 32)
        if final_max_scenes:
            base_candidates = max(base_candidates, int(final_max_scenes) * 2)
        candidate_count = min(len(scenes), max(8, base_candidates))

        for scene, analysis in zip(scenes, cheap_analyses):
            salience = float(analysis.get("salience", 0.0) or 0.0)
            story_rel = float(analysis.get("story_relevance", salience) or salience)
            confidence = float(analysis.get("confidence", 0.5) or 0.5)
            scene["_screen_score"] = max(0.0, min(1.0, 0.35 * salience + 0.55 * story_rel + 0.10 * confidence))

        # Preserve timeline coverage in the expensive pass.  Selecting only the
        # global top visual scores can erase quiet bridge/problem scenes before
        # the story-aware selector ever gets a chance to evaluate them.
        by_time = sorted(scenes, key=lambda item: item.get("start_time", 0))
        selected_scenes = []
        selected_ids = set()
        if by_time and candidate_count:
            timeline_end = max(float(item.get("end_time", 0) or 0) for item in by_time)
            window_count = min(candidate_count, len(by_time))
            window_size = timeline_end / max(1, window_count)
            for index in range(window_count):
                start = index * window_size
                end = timeline_end if index == window_count - 1 else (index + 1) * window_size
                options = [
                    item for item in by_time
                    if item["scene_id"] not in selected_ids
                    and start <= float(item.get("start_time", 0) or 0) < end
                ]
                if not options:
                    continue
                best = max(options, key=lambda item: item.get("_screen_score", 0.0))
                selected_scenes.append(best)
                selected_ids.add(best["scene_id"])

        for scene in sorted(scenes, key=lambda item: item.get("_screen_score", 0.0), reverse=True):
            if len(selected_scenes) >= candidate_count:
                break
            if scene["scene_id"] not in selected_ids:
                selected_scenes.append(scene)
                selected_ids.add(scene["scene_id"])

        selected_scenes.sort(key=lambda item: item.get("start_time", 0))
        selected_ids = {s["scene_id"] for s in selected_scenes}

        print(f"   🎯 Pass B: Deep analysis on {len(selected_scenes)} story candidates...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            deep_analyses = list(pool.map(lambda s: analyze_scene(s, is_cheap=False), selected_scenes))
            
        deep_map = {a["scene_id"]: a for a in deep_analyses}
        cheap_map = {a.get("scene_id"): a for a in cheap_analyses if a.get("scene_id")}
        for scene in scenes:
            sid = scene["scene_id"]
            screening = cheap_map.get(sid, {}) or {}
            if sid in selected_ids:
                analysis = deep_map.get(sid, {})
                analysis["selected"] = True
                analysis["screen_score"] = round(float(scene.get("_screen_score", 0) or 0), 4)
                analysis["screening"] = screening
                scene_analyses.append(analysis)
            else:
                scene_analyses.append({
                    "scene_id": sid,
                    "parent_scene_id": scene.get("parent_scene_id", scene.get("source_scene_id", sid)),
                    "window_index": scene.get("window_index", max(0, int(scene.get("source_scene_part", 1) or 1) - 1)),
                    "window_count": scene.get("window_count", max(1, int(scene.get("source_scene_parts", 1) or 1))),
                    "start_time": scene["start_time"],
                    "end_time": scene["end_time"],
                    "duration": scene["duration"],
                    "action": "Skipped in summary mode",
                    "transcript_text": screening.get("transcript_text", ""),
                    "confidence": 0.0,
                    "selected": False,
                    "analysis_kind": "screening_only",
                    "screen_score": round(float(scene.get("_screen_score", 0) or 0), 4),
                    "screening": screening,
                })
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            scene_analyses = list(pool.map(lambda s: analyze_scene(s, is_cheap=False), scenes))

    # ── Step 2: Temporal flow analysis ──────────────────────────────────
    print(f"\n   🔄 Building temporal flow understanding...")

    scenes_for_temporal = [a for a in scene_analyses if a.get("selected", True)]
    
    temporal_chunk_size = getattr(cfg, "temporal_chunk_size", 30)
    if not temporal_chunk_size or temporal_chunk_size < 1:
        temporal_chunk_size = 30
        
    enriched_scenes_all = []
    video_summaries = []
    narrative_arcs = []
    domain_res = domain_hint or "unknown"
    
    print(f"   [Temporal] Chunking {len(scenes_for_temporal)} scenes with size {temporal_chunk_size}")

    for i in range(0, len(scenes_for_temporal), temporal_chunk_size):
        chunk = scenes_for_temporal[i:i+temporal_chunk_size]
        
        temporal_prompt = build_temporal_prompt(
            scene_analyses=chunk,
            domain=domain_hint or "general",
            subtitle_context=full_subtitle_context,
        )

        # Log temporal prompt token estimate to detect large combined prompts
        try:
            temporal_tokens = _estimate_tokens(temporal_prompt, model=getattr(analyzer, "model", None))
        except Exception:
            temporal_tokens = None
        print(f"   [Tokens] temporal_prompt_tokens~{temporal_tokens} for chunk {i//temporal_chunk_size + 1}")

        try:
            if isinstance(analyzer, OpenAIVisionAnalyzer):
                chunk_data = analyzer.analyze_temporal_flow(
                    scene_analyses=chunk,
                    prompt=temporal_prompt,
                    trace_id=f"temporal-{i//temporal_chunk_size + 1}",
                )
            else:
                chunk_data = analyzer.analyze_temporal_flow(
                    scene_analyses=chunk,
                    prompt=temporal_prompt,
                )
            enriched_scenes_all.extend(chunk_data.get("scenes_enriched", []))
            if chunk_data.get("video_summary"):
                video_summaries.append(chunk_data["video_summary"])
            if chunk_data.get("narrative_arc") and chunk_data["narrative_arc"] != "unknown":
                narrative_arcs.append(chunk_data["narrative_arc"])
        except Exception as e:
            print(f"   ⚠️ Temporal analysis failed for chunk {i//temporal_chunk_size + 1}: {e}")
            # Retry smaller text-only batches before falling back. A temporal
            # JSON response can still exceed the model output budget even when
            # the input prompt itself fits the context window.
            retry_chunks = [chunk]
            if len(chunk) > 1:
                midpoint = max(1, len(chunk) // 2)
                retry_chunks = [chunk[:midpoint], chunk[midpoint:]]

            for retry_index, retry_chunk in enumerate(retry_chunks, start=1):
                try:
                    retry_prompt = build_temporal_prompt(
                        scene_analyses=retry_chunk,
                        domain=domain_hint or "general",
                        subtitle_context=full_subtitle_context,
                    )
                    if isinstance(analyzer, OpenAIVisionAnalyzer):
                        retry_data = analyzer.analyze_temporal_flow(
                            scene_analyses=retry_chunk,
                            prompt=retry_prompt,
                            trace_id=f"temporal-{i//temporal_chunk_size + 1}-retry-{retry_index}",
                        )
                    else:
                        retry_data = analyzer.analyze_temporal_flow(
                            scene_analyses=retry_chunk,
                            prompt=retry_prompt,
                        )
                    enriched_scenes_all.extend(retry_data.get("scenes_enriched", []))
                    if retry_data.get("video_summary"):
                        video_summaries.append(retry_data["video_summary"])
                    if retry_data.get("narrative_arc") and retry_data["narrative_arc"] != "unknown":
                        narrative_arcs.append(retry_data["narrative_arc"])
                except Exception as retry_error:
                    print(
                        f"   ⚠️ Temporal retry {retry_index} failed: {retry_error}"
                    )
                    for scene in retry_chunk:
                        enriched_scenes_all.append({
                            "scene_id": scene.get("scene_id"),
                            "narrative_role": "process",
                            "narration_importance": 0.5,
                            "narration_cue": "",
                            "connects_to_next": "",
                        })

    temporal_data = {
        "video_summary": " ".join(video_summaries) if video_summaries else "Unable to generate summary.",
        "narrative_arc": narrative_arcs[-1] if narrative_arcs else "unknown",
        "domain": domain_res,
        "scenes_enriched": enriched_scenes_all,
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
            "parent_scene_id": analysis.get("parent_scene_id", scene_id),
            "window_index": int(analysis.get("window_index", 0) or 0),
            "window_count": max(1, int(analysis.get("window_count", 1) or 1)),
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
                "character_goal": analysis.get("character_goal", ""),
                "state_before": analysis.get("state_before", ""),
                "state_after": analysis.get("state_after", ""),
                "cause": analysis.get("cause", ""),
                "effect": analysis.get("effect", ""),
                "unresolved_question": analysis.get("unresolved_question", ""),
                "transcript_text": analysis.get("transcript_text", ""),
                "text_visible": analysis.get("text_visible", []),
                "confidence": analysis.get("confidence", 0),
                "selected": bool(analysis.get("selected", True)),
                "analysis_kind": analysis.get("analysis_kind", "deep" if analysis.get("selected", True) else "screening_only"),
                "screen_score": analysis.get("screen_score", 0),
                "screening": analysis.get("screening", {}) if isinstance(analysis.get("screening"), dict) else {},
                "promoted_on_demand": bool(analysis.get("promoted_on_demand", False)),
                # Add word-level data if available (from ASR)
                "words": analysis.get("words", []),
            },
            "narrative": {
                "role": enrichment.get("narrative_role", "process"),
                "importance": enrichment.get("narration_importance", 0.5),
                "causal_importance": enrichment.get("causal_importance", 0.0),
                "bridge_importance": enrichment.get("bridge_importance", 0.0),
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

def promote_scenes_on_demand(
    *,
    project_dir: str,
    storyboard: dict,
    scene_ids: list[str],
    cfg,
    domain_hint: str = "",
) -> tuple[dict, list[str]]:
    """Deep-analyze screening-only scenes and merge them back into storyboard.

    This intentionally reuses ``run_analysis`` on a tiny subset instead of
    introducing a second vision request implementation.  Therefore all current
    Qwen schema/truncation/runaway safeguards remain active during promotion.
    """
    import copy

    requested = [str(sid) for sid in scene_ids if str(sid)]
    if not requested:
        return storyboard, []

    segment_path = os.path.join(project_dir, "segment_manifest.json")
    if not os.path.exists(segment_path):
        return storyboard, []
    with open(segment_path, "r", encoding="utf-8") as handle:
        segment_data = json.load(handle)

    source_scenes = {str(s.get("scene_id")): s for s in segment_data.get("scenes", []) or []}
    chosen = [source_scenes[sid] for sid in requested if sid in source_scenes]
    if not chosen:
        return storyboard, []

    ingest_path = os.path.join(project_dir, "ingest_manifest.json")
    subtitle_context = None
    if os.path.exists(ingest_path):
        try:
            with open(ingest_path, "r", encoding="utf-8") as handle:
                subtitle_context = (json.load(handle) or {}).get("subtitle_data")
        except Exception:
            subtitle_context = None

    subset = {
        "video_path": segment_data.get("video_path", ""),
        "total_scenes": len(chosen),
        "scenes": chosen,
        "keyframes": {sid: (segment_data.get("keyframes", {}) or {}).get(sid, []) for sid in requested},
        "status": "promotion",
    }
    promote_cfg = copy.copy(cfg)
    promote_cfg.mode = "full"
    promoted_board = run_analysis(
        segment_data=subset,
        project_dir=project_dir,
        cfg=promote_cfg,
        subtitle_context=subtitle_context,
        domain_hint=domain_hint or getattr(cfg, "vision_domain_hint", ""),
    )
    promoted_map = {str(s.get("scene_id")): s for s in promoted_board.get("scenes", []) or []}

    applied: list[str] = []
    for scene in storyboard.get("scenes", []) or []:
        sid = str(scene.get("scene_id", ""))
        promoted = promoted_map.get(sid)
        if not promoted:
            continue
        new_analysis = dict(promoted.get("analysis", {}) or {})
        old_analysis = scene.get("analysis", {}) or {}
        new_analysis["selected"] = True
        new_analysis["analysis_kind"] = "deep_promoted"
        new_analysis["promoted_on_demand"] = True
        new_analysis["screening"] = old_analysis.get("screening", {})
        new_analysis["screen_score"] = old_analysis.get("screen_score", 0)
        scene["analysis"] = new_analysis
        scene["parent_scene_id"] = promoted.get("parent_scene_id", scene.get("parent_scene_id", sid))
        scene["window_index"] = promoted.get("window_index", scene.get("window_index", 0))
        scene["window_count"] = promoted.get("window_count", scene.get("window_count", 1))
        # Use promotion temporal enrichment when available; otherwise preserve old narrative.
        if promoted.get("narrative"):
            scene["narrative"] = promoted["narrative"]
        applied.append(sid)

    return storyboard, applied
