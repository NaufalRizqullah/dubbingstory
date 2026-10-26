"""Video timeline and render option regressions; mocked probes are offline."""
import json
from types import SimpleNamespace

import pytest

from dubbingstory.ingest.local_video import require_video_timeline
from dubbingstory.ingest.sources import SourceIngestError
from dubbingstory.ingest.youtube import _base_options
from dubbingstory.render.video_render import build_render_command


def test_video_packets_override_longer_container_audio_tail(monkeypatch):
    packets = [{"pts_time": "2.0", "duration_time": "0.04"},
               {"pts_time": "2.1", "duration_time": "0.03"},
               {"pts_time": "2.04", "duration_time": "0.06"}]
    seen = []
    def probe(command, **kwargs):
        seen.append(command)
        return SimpleNamespace(stdout=json.dumps({"packets": packets}))
    monkeypatch.setattr("dubbingstory.ingest.local_video.subprocess.run", probe)
    metadata = {"video_duration": 10, "duration": 10, "video_duration_source": "container",
                "video_start_time": 2, "video_stream_index": 1}
    resolved = require_video_timeline("input.mkv", metadata)
    assert resolved["video_duration"] == pytest.approx(.13)
    assert resolved["duration"] == 10  # Historical container metadata remains available.
    assert resolved["video_duration_source"] == "packets"
    assert seen[0][seen[0].index("-select_streams") + 1] == "1"
    assert metadata["video_duration_source"] == "container"


def test_unknown_final_vfr_duration_is_not_estimated(monkeypatch):
    monkeypatch.setattr("dubbingstory.ingest.local_video.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=json.dumps({"packets": [{"pts_time": "0", "duration_time": ".04"},
                                                                        {"pts_time": ".04"}]})))
    with pytest.raises(SourceIngestError, match="final video packet"):
        require_video_timeline("input.mkv", {"video_duration_source": "container", "duration": 12})


def test_known_stream_duration_does_not_scan_packets(monkeypatch):
    monkeypatch.setattr("dubbingstory.ingest.local_video.subprocess.run", lambda *args, **kwargs: pytest.fail("Unnecessary packet scan"))
    assert require_video_timeline("input.mp4", {"video_duration_source": "stream", "video_duration": 3})["video_duration"] == 3


def test_youtube_options_allow_upstream_to_select_supported_clients():
    assert "extractor_args" not in _base_options("youtube")
    assert _base_options("youtube")["remote_components"] == ["ejs:github"]
    assert "extractor_args" not in _base_options("xiaohongshu")
    assert "cookiesfrombrowser" not in _base_options("douyin", "chrome")


def test_animated_defaults_do_not_change_classic_srt_style(tmp_path):
    srt = tmp_path / "caption.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHalo\n", encoding="utf-8")
    cfg = SimpleNamespace(subtitle_style="classic", subtitle_font_size=64, subtitle_outline=3,
                          subtitle_font_name="DejaVu Sans")
    cmd = build_render_command("source.mp4", "voice.wav", "out.mp4",
                               {"video_duration": 2, "codec": "h264", "has_audio": False},
                               subtitle_path=str(srt), burn_subs=True, cfg=cfg)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "FontSize=24" in graph
    assert "Outline=2.0" in graph
    assert "FontSize=64" not in graph


def test_native_ass_passes_display_geometry_for_non_square_pixels(tmp_path, monkeypatch):
    ass = tmp_path / "caption.ass"
    ass.write_text("[Script Info]\nPlayResX: 768\nPlayResY: 576\n", encoding="utf-8")
    monkeypatch.setattr("dubbingstory.story.animated_subtitles.caption_font_directory", lambda cfg: str(tmp_path))
    cfg = SimpleNamespace(subtitle_style="karaoke")
    cmd = build_render_command("source.mp4", "voice.wav", "out.mp4",
                               {"video_duration": 2, "codec": "h264", "has_audio": False,
                                "display_width": 768, "display_height": 576, "video_stream_index": 1},
                               subtitle_path=str(ass), burn_subs=True, target_ratio="source", cfg=cfg)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "original_size=768x576" in graph
    assert "force_style" not in graph
    assert "[0:1]" in graph
    assert "scale=" not in graph
