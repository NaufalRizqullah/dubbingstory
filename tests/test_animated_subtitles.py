"""Offline caption contract tests, intended to be run on Kaggle."""

import os
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dubbingstory.story import animated_subtitles as captions
from dubbingstory.story.subtitle_gen import generate_srt, generate_srt_from_words


class MeasuredFont:
    """Deterministic font metrics for geometry tests without system fonts."""
    def __init__(self, size):
        self.size = size

    def getbbox(self, text):
        return (0, 0, len(text) * self.size / 2, self.size)

    def getmetrics(self):
        return (self.size * .75, self.size * .25)

    def getname(self):
        return ("Measured Fixture", "Regular")


class AnimatedSubtitleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = os.path.join(self.directory.name, "subtitle.ass")
        self.config = SimpleNamespace(
            subtitle_font_name="Configured alias", subtitle_font_size=64,
            subtitle_font_color="#FFFFFF", subtitle_active_color="#FFCC00",
            subtitle_animation="none", subtitle_safe_margin_x=.1,
            subtitle_safe_margin_y=.18, subtitle_outline=3,
            subtitle_max_words=5, subtitle_pop_scale=1.2,
        )
        self.font_patch = patch.object(captions, "_load_font", side_effect=lambda name, path, size: MeasuredFont(size))
        self.font_patch.start()
        self.addCleanup(self.font_patch.stop)
        self.words = [
            {"text": "Lihat", "start": .1, "end": .3, "scene_id": "one"},
            {"text": "hasilnya!", "start": .5, "end": 1.0, "scene_id": "one"},
        ]

    def generate(self, style="karaoke", words=None, width=1080, height=1920):
        words = self.words if words is None else words
        payload = {"words": words, "text": " ".join(word["text"] for word in words)}
        captions.generate_ass(payload, self.output, width, height, self.config, style)
        with open(self.output, encoding="utf-8") as source:
            return source.read()

    @staticmethod
    def events(document):
        return [line.split(",", 9) for line in document.splitlines() if line.startswith("Dialogue:")]

    def test_word_style_keeps_measured_start_end_and_silence(self):
        events = self.events(self.generate(style="word"))
        self.assertEqual([(event[1], event[2]) for event in events], [
            ("0:00:00.10", "0:00:00.30"), ("0:00:00.50", "0:00:01.00"),
        ])
        self.assertIn("Lihat", events[0][9])
        self.assertNotIn("hasilnya", events[0][9])

    def test_karaoke_has_one_active_word_and_neutral_text_during_pause(self):
        events = self.events(self.generate())
        self.assertEqual(len(events), 3)
        highlight = r"\c&H00CCFF&"
        self.assertEqual([event[9].count(highlight) for event in events], [1, 0, 1])
        plain = [re.sub(r"\{[^}]*\}", "", event[9]) for event in events]
        self.assertEqual(plain, ["Lihat hasilnya!"] * 3)

    def test_phrase_breaks_at_long_pause_and_scene_transition(self):
        words = [
            {"text": "Satu", "start": 0, "end": .2, "scene_id": "one"},
            {"text": "dua", "start": 1.0, "end": 1.2, "scene_id": "one"},
            {"text": "tiga", "start": 1.3, "end": 1.5, "scene_id": "two"},
        ]
        events = self.events(self.generate(words=words))
        self.assertEqual(len(events), 3)
        self.assertNotIn("dua", events[0][9])
        self.assertNotIn("tiga", events[1][9])

    def test_karaoke_pop_continues_across_short_word_boundary(self):
        self.config.subtitle_animation = "pop"
        self.config.subtitle_pop_scale = 1.12
        words = [{"text": "A", "start": 0, "end": .03}, {"text": "B", "start": .03, "end": .5}]
        events = self.events(self.generate(words=words))
        self.assertIn(r"\fscx112.000", events[0][9])
        self.assertIn(r"\fscx108.000", events[1][9])
        self.assertNotIn(r"\fscx112.000", events[1][9])

    def test_karaoke_fade_continues_instead_of_restarting_each_word(self):
        self.config.subtitle_animation = "fade"
        words = [{"text": "A", "start": 0, "end": .03}, {"text": "B", "start": .03, "end": .5}]
        events = self.events(self.generate(words=words))
        self.assertIn(r"\alpha&HFF&", events[0][9])
        self.assertIn(r"\alpha&H40&", events[1][9])
        self.assertIn(r"\t(440,470,\alpha&HFF&)", events[1][9])

    def test_maximum_pop_and_outline_fit_within_safe_area(self):
        self.config.subtitle_animation = "pop"
        words = [{"text": "a" * 30, "start": 0, "end": 1}]
        document = self.generate(style="word", words=words)
        event = self.events(document)[0][9]
        fitted_size = int(re.search(r"\\fs(\d+)", event).group(1))
        # Measured glyph width + both outlines, at the largest animation scale.
        self.assertLessEqual((30 * fitted_size / 2 + 6) * 1.2, 1080 * .8)
        self.assertIn("Style: Caption,Measured Fixture,64,", document)
        y = float(re.search(r"\\pos\([^,]+,([\d.]+)\)", event).group(1))
        self.assertLessEqual(y + 3 * 1.2, 1920 * .82 + .05)

    def test_font_size_scales_with_canvas_width(self):
        document = self.generate(style="word", width=720, height=1280)
        self.assertIn("PlayResX: 720", document)
        self.assertIn("PlayResY: 1280", document)
        self.assertIn("Style: Caption,Measured Fixture,43,", document)

    def test_text_cannot_inject_ass_tags_or_line_breaks(self):
        escaped = captions._escape_ass(r"{\pos(1,1)}Hello\Nworld" + "\n")
        self.assertNotIn("{", escaped)
        self.assertNotIn("\\", escaped)
        self.assertNotIn("\n", escaped)
        self.assertIn("Hello", escaped)

    def test_timing_too_short_for_ass_fails_instead_of_overlapping_words(self):
        words = [{"text": "A", "start": .001, "end": .002},
                 {"text": "B", "start": .003, "end": .004}]
        with self.assertRaisesRegex(ValueError, "10-ms"):
            self.generate(style="word", words=words)

    def test_layout_measures_the_actual_escaped_display_glyphs(self):
        class WideReservedFont(MeasuredFont):
            def getbbox(self, text):
                width = sum(100 if char in "｛｝＼" else 10 for char in text)
                return (0, 0, width, self.size)

        font = WideReservedFont(64)
        words = [{"text": "a{"}, {"text": "b}"}]
        self.assertEqual(captions._layout(words, font, 150), [[0], [1]])

    def test_invalid_style_colors_and_geometry_fail_explicitly(self):
        with self.assertRaises(ValueError):
            self.generate(style="unknown")
        with self.assertRaises(ValueError):
            self.generate(width=0)
        self.config.subtitle_active_color = "not-a-color"
        with self.assertRaises(ValueError):
            self.generate()

    def test_missing_font_is_actionable_not_a_silent_layout_substitution(self):
        self.font_patch.stop()
        self.config.subtitle_font_path = os.path.join(self.directory.name, "missing-font.ttf")
        with self.assertRaisesRegex(ValueError, "font_path"):
            self.generate()


class MeasuredSRTTests(unittest.TestCase):
    def test_srt_uses_actual_boundaries_across_long_pause(self):
        words = [
            {"text": "Lihat", "start": .125, "end": .5},
            {"text": "hasilnya", "start": 2, "end": 2.75},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "subtitle.srt")
            generate_srt_from_words(words, path)
            with open(path, encoding="utf-8") as source:
                content = source.read()
        self.assertIn("00:00:00,125 --> 00:00:00,500", content)
        self.assertIn("00:00:02,000 --> 00:00:02,750", content)

    def test_legacy_segment_srt_remains_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "subtitle.srt")
            generate_srt([{"text": "Narasi lama", "start_time": 1, "end_time": 3}], path)
            with open(path, encoding="utf-8") as source:
                content = source.read()
        self.assertIn("00:00:01,000 --> 00:00:03,000", content)
        self.assertIn("Narasi lama", content)


if __name__ == "__main__":
    unittest.main()
