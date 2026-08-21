import json
import os
import subprocess
import tempfile
import unittest
from types import SimpleNamespace

from dubbingstory.story.scene_cards import build_scene_cards
from dubbingstory.story.narration_planner import build_narration_plan
from dubbingstory.story.narration_qa import evaluate_narration
from dubbingstory.story.scene_selector import select_scenes
from dubbingstory.tts.voice_manager import _get_audio_duration, _render_timeline_canvas
from dubbingstory.render.audio_mix import mix_narration_with_original


class StorytellingV2Tests(unittest.TestCase):
    def setUp(self):
        self.storyboard = {
            "video_title": "Repair",
            "video_summary": "A part is machined and installed.",
            "total_duration": 30.0,
            "total_scenes": 3,
            "scenes": [
                {
                    "scene_id": "scene_001", "start_time": 0.0, "end_time": 10.0, "duration": 10.0,
                    "analysis": {"action": "A damaged part is inspected", "confidence": 0.9},
                    "narrative": {"role": "setup", "importance": 0.8},
                },
                {
                    "scene_id": "scene_002", "start_time": 10.0, "end_time": 20.0, "duration": 10.0,
                    "analysis": {"action": "Metal is machined on a lathe", "confidence": 0.9},
                    "narrative": {"role": "process", "importance": 0.6},
                },
                {
                    "scene_id": "scene_003", "start_time": 20.0, "end_time": 30.0, "duration": 10.0,
                    "analysis": {"action": "The finished part is installed", "confidence": 0.9},
                    "narrative": {"role": "resolution", "importance": 0.9},
                },
            ],
        }
        self.story_plan = {
            "scene_roles": [
                {"scene_id": "scene_001", "role": "problem", "story_importance": .9, "causal_importance": .9, "bridge_importance": .2, "must_keep": True},
                {"scene_id": "scene_002", "role": "bridge", "story_importance": .7, "causal_importance": .8, "bridge_importance": .9, "must_keep": False},
                {"scene_id": "scene_003", "role": "resolution", "story_importance": 1.0, "causal_importance": .9, "bridge_importance": .2, "must_keep": True},
            ]
        }

    def test_scene_cards_overlap_transcript(self):
        subtitles = {"entries": [
            {"start_seconds": 9.5, "end_seconds": 11.0, "text": "Now we machine the replacement."}
        ]}
        cards = build_scene_cards(self.storyboard, subtitles)
        self.assertEqual(cards[1]["transcript"], "Now we machine the replacement.")
        self.assertEqual(cards[0]["transcript"], "Now we machine the replacement.")

    def test_narration_plan_has_dense_but_bounded_word_budgets(self):
        cfg = SimpleNamespace(
            narration_target_wpm=160,
            narration_min_wpm=135,
            narration_max_wpm=175,
            narration_speech_coverage=.84,
            narration_plan_llm_refine=False,
        )
        plan = build_narration_plan(
            self.storyboard, self.storyboard["scenes"],
            story_plan=self.story_plan, story_memory={}, cfg=cfg, mode="summary",
        )
        self.assertAlmostEqual(plan["total_duration"], 30.0)
        self.assertGreater(plan["total_target_words"], 40)
        self.assertLess(plan["total_target_words"], 90)
        self.assertEqual(plan["segments"][1]["timeline_start"], 10.0)
        self.assertLess(plan["segments"][1]["target_words"], plan["segments"][0]["target_words"])

    def test_story_selector_respects_five_percent_duration_tolerance(self):
        selected = select_scenes(
            self.storyboard,
            target_duration=20,
            max_scenes=3,
            min_score=0.0,
            story_plan=self.story_plan,
            duration_tolerance=1.05,
        )
        total = sum(s["duration"] for s in selected)
        self.assertLessEqual(total, 21.0 + 1e-6)
        self.assertIn("scene_001", {s["scene_id"] for s in selected})
        self.assertIn("scene_003", {s["scene_id"] for s in selected})

    def test_qa_flags_old_hype_and_sparse_script(self):
        plan = {
            "total_duration": 30,
            "target_wpm": 160,
            "segments": [
                {"scene_id": "scene_001", "target_words": 25},
                {"scene_id": "scene_002", "target_words": 25},
            ],
        }
        qa = evaluate_narration(
            [
                {"scene_id": "scene_001", "text": "Lihat ini! Luar biasa! Amazing!"},
                {"scene_id": "scene_002", "text": "Gila! Keren banget!"},
            ],
            plan,
            language="id",
        )
        self.assertTrue(qa["needs_rewrite"])
        self.assertTrue(any(i.startswith("too_many_exclamations") for i in qa["issues"]))
        self.assertTrue(any(i.startswith("narration_too_sparse") for i in qa["issues"]))

    @unittest.skipUnless(subprocess.call(["sh", "-c", "command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null"]) == 0, "ffmpeg unavailable")
    def test_timeline_canvas_keeps_video_length_without_per_scene_padding(self):
        with tempfile.TemporaryDirectory() as td:
            tone1 = os.path.join(td, "a.wav")
            tone2 = os.path.join(td, "b.wav")
            out = os.path.join(td, "timeline.wav")
            for path in (tone1, tone2):
                subprocess.run([
                    "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
                    "-ar", "24000", "-ac", "1", path,
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            _render_timeline_canvas(
                placements=[
                    {"path": tone1, "actual_start": 0.0},
                    {"path": tone2, "actual_start": 2.0},
                ],
                output_path=out,
                total_duration=3.0,
                normalize_loudness=False,
            )
            self.assertAlmostEqual(_get_audio_duration(out), 3.0, delta=0.08)

    @unittest.skipUnless(subprocess.call(["sh", "-c", "command -v ffmpeg >/dev/null"]) == 0, "ffmpeg unavailable")
    def test_dynamic_duck_filter_executes(self):
        with tempfile.TemporaryDirectory() as td:
            video = os.path.join(td, "video.mp4")
            narr = os.path.join(td, "narr.wav")
            out = os.path.join(td, "mixed.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "color=size=320x180:rate=24:duration=2",
                "-f", "lavfi", "-i", "sine=frequency=220:duration=2",
                "-shortest", "-c:v", "libx264", "-c:a", "aac", video,
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=880:duration=1",
                "-af", "adelay=500:all=1,apad=pad_dur=0.5", "-t", "2", narr,
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            mix_narration_with_original(video, narr, out, strategy="dynamic_duck")
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 1000)


if __name__ == "__main__":
    unittest.main()
