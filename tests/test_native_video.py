"""Native render contracts and Kaggle FFmpeg regressions; no network required."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace

from dubbingstory.render.audio_mix import build_audio_filter
from dubbingstory.render.video_render import build_render_command, render_final, _validate_tts_manifest
from dubbingstory.tts.word_alignment import file_sha256, text_sha256


class NativeRenderContractTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {"video_duration": 5.0, "codec": "h264", "has_audio": True,
                         "video_start_time": 2.0, "audio_start_time": 2.25,
                         "display_width": 1080, "display_height": 1920}

    def test_native_copy_does_not_scale_or_shorten_video(self):
        cmd = build_render_command("source.mp4", "voice.wav", "out.mp4", self.metadata, target_ratio="source")
        self.assertEqual(cmd[cmd.index("-c:v") + 1], "copy")
        self.assertNotIn("-shortest", cmd)
        self.assertNotIn("-t", cmd)
        self.assertNotIn("scale=", cmd[cmd.index("-filter_complex") + 1])
        self.assertEqual(cmd[cmd.index("-itsoffset") + 1], "-2.000000000")

    def test_burn_encodes_once_without_rescaling_source(self):
        with tempfile.TemporaryDirectory() as td:
            srt = os.path.join(td, "captions.srt")
            with open(srt, "w", encoding="utf-8") as output:
                output.write("1\n00:00:00,000 --> 00:00:01,000\nHello\n")
            cmd = build_render_command("source.mp4", "voice.wav", "out.mp4", self.metadata,
                                       target_ratio="source", subtitle_path=srt, burn_subs=True)
            self.assertEqual(cmd.count("-c:v"), 1)
            self.assertEqual(cmd[cmd.index("-c:v") + 1], "libx264")
            graph = cmd[cmd.index("-filter_complex") + 1]
            self.assertIn("subtitles=", graph)
            self.assertNotIn("scale=", graph)
            self.assertNotIn("-shortest", cmd)

    def test_ducking_honors_volume_and_audio_start_offset(self):
        graph = build_audio_filter(duration=5, has_original_audio=True, strategy="dynamic_duck",
                                   original_volume=0.23, original_start_offset=0.25)
        self.assertIn("volume=0.230000", graph)
        self.assertIn("adelay=250.000:all=1", graph)
        self.assertIn("sidechaincompress=", graph)
        self.assertIn("apad,atrim=duration=5.000000", graph)

    def test_original_audio_before_video_is_trimmed_not_shifted(self):
        graph = build_audio_filter(duration=5, has_original_audio=True, original_start_offset=-0.3)
        self.assertIn("atrim=start=0.300000,asetpts=PTS-STARTPTS", graph)

    def test_silent_source_and_zero_original_volume_do_not_require_original_audio(self):
        for has_audio, volume in [(False, 0.15), (True, 0.0)]:
            graph = build_audio_filter(duration=5, has_original_audio=has_audio, original_volume=volume)
            self.assertNotIn("[0:a", graph)
            self.assertNotIn("sidechaincompress", graph)

    def test_invalid_duration_and_hdr_encode_fail_explicitly(self):
        with self.assertRaises(ValueError):
            build_audio_filter(duration=float("nan"), has_original_audio=False)
        metadata = {**self.metadata, "color_transfer": "smpte2084"}
        with self.assertRaisesRegex(ValueError, "HDR"):
            build_render_command("a", "b", "c", metadata, target_ratio="9:16")

    def test_manifest_rejects_stale_audio_text_and_source(self):
        with tempfile.TemporaryDirectory() as td:
            audio = os.path.join(td, "audio_id.wav")
            source = os.path.join(td, "source.mp4")
            script = os.path.join(td, "script_id.json")
            for path in [audio, source]:
                with open(path, "wb") as output:
                    output.write(b"fixture")
            with open(script, "w", encoding="utf-8") as output:
                json.dump({"segments": [{"text": "Halo dunia"}]}, output)
            manifest = {"status": "complete", "audio_sha256": file_sha256(audio),
                        "text_sha256": text_sha256("Halo dunia"), "timeline_duration": 5,
                        "source_sha256": file_sha256(source), "overflow_seconds": 0}
            manifest.update(schema_version=1, strict_timing_validated=True, complete_coverage=True,
                            fit_policy="error", expected_segment_ids=["one"], failed_segment_ids=[],
                            placements=[{"scene_id": "one", "planned_start": 0, "planned_end": 5,
                                         "actual_start": 0, "actual_end": 1}])
            with open(os.path.join(td, "audio_id.manifest.json"), "w", encoding="utf-8") as output:
                json.dump(manifest, output)
            _validate_tts_manifest(audio, 5, script, source)
            with open(source, "ab") as output:
                output.write(b"changed")
            with self.assertRaisesRegex(ValueError, "Source video changed"):
                _validate_tts_manifest(audio, 5, script, source)
            with open(audio, "ab") as output:
                output.write(b"changed")
            with self.assertRaisesRegex(ValueError, "audio is incomplete or changed"):
                _validate_tts_manifest(audio, 5, script)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe required on Kaggle")
class NativeFFmpegTests(unittest.TestCase):
    def _run(self, command):
        return subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_short_narration_without_original_audio_preserves_all_video_frames(self):
        with tempfile.TemporaryDirectory() as td:
            source, audio, output = [os.path.join(td, name) for name in ("source.mp4", "audio_id.wav", "out.mp4")]
            self._run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=size=180x320:rate=24:duration=2",
                       "-an", "-c:v", "libx264", source])
            self._run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5", audio])
            with open(os.path.join(td, "audio_id.manifest.json"), "w", encoding="utf-8") as target:
                json.dump({"schema_version": 1, "status": "complete", "audio_sha256": file_sha256(audio),
                           "source_sha256": file_sha256(source), "timeline_duration": 2,
                           "overflow_seconds": 0, "strict_timing_validated": True,
                           "complete_coverage": True, "fit_policy": "error",
                           "expected_segment_ids": ["one"], "failed_segment_ids": [],
                           "placements": [{"scene_id": "one", "planned_start": 0, "planned_end": 2,
                                           "actual_start": 0, "actual_end": .5}]}, target)
            render_final(source, audio, output, target_ratio="source", audio_strategy="dynamic_duck")
            hashes = []
            for path in [source, output]:
                result = self._run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0", "-f", "framemd5", "-"])
                hashes.append([line for line in result.stdout.decode().splitlines() if not line.startswith("#")])
            self.assertEqual(hashes[0], hashes[1])
            probe = self._run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", output])
            streams = json.loads(probe.stdout)["streams"]
            video = next(stream for stream in streams if stream["codec_type"] == "video")
            audio_stream = next(stream for stream in streams if stream["codec_type"] == "audio")
            self.assertEqual((video["width"], video["height"]), (180, 320))
            self.assertAlmostEqual(float(video["duration"]), 2, delta=1 / 24)
            self.assertAlmostEqual(float(audio_stream["duration"]), 2, delta=0.06)


if __name__ == "__main__":
    unittest.main()
