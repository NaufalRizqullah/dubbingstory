from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dubbingstory.config import build_config
from dubbingstory.story.prompt_templates import build_rewrite_prompt, build_v2_narration_prompt
from dubbingstory.tts.azure_tts_engine import AzureTTSEngine
from dubbingstory.tts.voice_manager import _style_speaking_rate_multiplier, get_engine


class JozzrProfileTests(unittest.TestCase):
    def test_profile_is_loaded_and_prompted(self):
        cfg = build_config()
        profile = cfg.narration_styles["jozzr_social"]
        self.assertIn("process-first", profile["tone"])

        prompt = build_v2_narration_prompt(
            scenes=[],
            narration_plan={"segments": []},
            story_plan={},
            story_memory={},
            language="id",
            style="jozzr_social",
            style_config=profile,
            hedging_threshold=0.5,
            mode="summary",
        )
        self.assertIn("Jozzr-style Process Storytelling", prompt)
        self.assertIn("Bagian ini belum bisa langsung dipasang", prompt)
        self.assertIn("LANGUAGE STYLE", prompt)

        rewrite = build_rewrite_prompt(
            current_segments=[],
            qa={"issues": ["repetitive_sentence_openers"]},
            narration_plan={},
            story_plan={},
            language="id",
            style="jozzr_social",
            style_config=profile,
        )
        self.assertIn("Jozzr-style Process Storytelling", rewrite)
        self.assertIn("STYLE RULES", rewrite)

    def test_profile_rate_multiplier(self):
        cfg = build_config()
        cfg.narration_style = "jozzr_social"
        self.assertAlmostEqual(_style_speaking_rate_multiplier(cfg), 1.02, places=3)


class AzureTTSTests(unittest.TestCase):
    def test_engine_defaults_to_ardi(self):
        cfg = SimpleNamespace(
            azure_speech_key="dummy",
            azure_speech_region="southeastasia",
            tts_azure_voices_id="id-ID-ArdiNeural",
            tts_azure_voices_en="en-US-GuyNeural",
            tts_azure_endpoint="",
            tts_azure_output_format="riff-24khz-16bit-mono-pcm",
            tts_azure_timeout_seconds=60,
            tts_azure_rate_limit_per_minute=0,
            tts_azure_max_retries=1,
        )
        engine = get_engine("azure", cfg)
        self.assertEqual(engine.get_engine_name(), "azure")
        self.assertEqual(engine.voices["id"], "id-ID-ArdiNeural")

    def test_ssml_escapes_and_applies_rate(self):
        engine = AzureTTSEngine(
            subscription_key="dummy",
            region="southeastasia",
            rate_limit_per_minute=0,
        )
        ssml = engine._ssml("A & B < aman", "id", "id-ID-ArdiNeural", 1.02)
        self.assertIn('name="id-ID-ArdiNeural"', ssml)
        self.assertIn('rate="+2%"', ssml)
        self.assertIn("A &amp; B &lt; aman", ssml)

    @patch("dubbingstory.tts.azure_tts_engine.requests.post")
    def test_synthesize_writes_wav_payload(self, mock_post):
        response = Mock()
        response.status_code = 200
        response.content = b"RIFFmockwav"
        response.text = ""
        response.headers = {}
        mock_post.return_value = response

        engine = AzureTTSEngine(
            subscription_key="dummy",
            region="southeastasia",
            rate_limit_per_minute=0,
            max_retries=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "voice.wav")
            result = engine.synthesize("Tes narasi.", out, language="id", speaking_rate=1.0)
            self.assertEqual(result, out)
            with open(out, "rb") as handle:
                self.assertEqual(handle.read(), b"RIFFmockwav")

        args, kwargs = mock_post.call_args
        self.assertIn("southeastasia.tts.speech.microsoft.com", args[0])
        self.assertEqual(kwargs["headers"]["X-Microsoft-OutputFormat"], "riff-24khz-16bit-mono-pcm")
        self.assertIn(b"id-ID-ArdiNeural", kwargs["data"])


if __name__ == "__main__":
    unittest.main()
